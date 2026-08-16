from __future__ import annotations

from collections import deque

import numpy as np


class DrlVOObservationBuilder:
    def __init__(self, config):
        self.cfg = config
        self.scan_history = {}
        self._pool_edges = np.linspace(0, self.cfg.lidar_num, self.cfg.lidar_pool_bins + 1, dtype=int)

    @staticmethod
    def normalize(data, data_min, data_max):
        data = np.asarray(data, dtype=np.float32)
        scale = max(float(data_max - data_min), 1e-6)
        return (2.0 * (data - data_min) / scale - 1.0).astype(np.float32)

    @staticmethod
    def world_to_base(x, y, xr, yr, yaw):
        dx = x - xr
        dy = y - yr
        c = np.cos(yaw)
        s = np.sin(yaw)
        xb = c * dx + s * dy
        yb = -s * dx + c * dy
        return xb, yb

    @staticmethod
    def world_vel_to_base(vx, vy, yaw):
        c = np.cos(yaw)
        s = np.sin(yaw)
        vx_b = c * vx + s * vy
        vy_b = -s * vx + c * vy
        return vx_b, vy_b

    def reset(self, env):
        self.scan_history = {}
        for ego_id in range(len(env.robots)):
            scan = self.get_lidar_scan(env, ego_id)
            history = deque(maxlen=self.cfg.lidar_history_len)
            for _ in range(self.cfg.lidar_history_len):
                history.append(scan.copy())
            self.scan_history[ego_id] = history

    def get_robot_states(self, env):
        states = []
        for robot in env.robots:
            info = robot.get_vel_and_pos()
            states.append(
                {
                    "x": float(info["pos"][0]),
                    "y": float(info["pos"][1]),
                    "yaw": float(robot.theta),
                    "vx": float(info["vel"][0]),
                    "vy": float(info["vel"][1]),
                    "wz": float(info["angular_vel"]),
                }
            )
        return states

    def get_subgoal(self, env, ego_id, states):
        robot = env.robots[ego_id]
        ego = states[ego_id]
        goal = np.asarray(robot.target_pos, dtype=np.float32)
        pos = np.asarray([ego["x"], ego["y"]], dtype=np.float32)
        vec = goal - pos
        dist = float(np.linalg.norm(vec))

        if dist > self.cfg.lookahead_distance:
            vec = vec / max(dist, 1e-6) * self.cfg.lookahead_distance
            subgoal_world = pos + vec
        else:
            subgoal_world = goal

        x_rel, y_rel = self.world_to_base(subgoal_world[0], subgoal_world[1], ego["x"], ego["y"], ego["yaw"])
        return np.array([x_rel, y_rel], dtype=np.float32), dist

    def get_lidar_scan(self, env, ego_id):
        scan = np.asarray(env.robots[ego_id].laser, dtype=np.float32)
        scan = np.nan_to_num(
            scan,
            nan=self.cfg.lidar_range_max,
            posinf=self.cfg.lidar_range_max,
            neginf=self.cfg.lidar_range_min,
        )
        scan = np.clip(scan, self.cfg.lidar_range_min, self.cfg.lidar_range_max)
        if self.cfg.lidar_left_to_right:
            scan = scan[::-1]
        return scan.astype(np.float32)

    def build_pedestrian_map(self, states, ego_id):
        size = self.cfg.ped_map_size
        ped_map = np.zeros((2, size, size), dtype=np.float32)

        ego = states[ego_id]
        xr, yr, yaw = ego["x"], ego["y"], ego["yaw"]
        row_res = self.cfg.ped_forward_range / size
        col_res = (2.0 * self.cfg.ped_lateral_range) / size

        for rid, st in enumerate(states):
            if rid == ego_id:
                continue

            x_rel, y_rel = self.world_to_base(st["x"], st["y"], xr, yr, yaw)
            if not (0.0 <= x_rel <= self.cfg.ped_forward_range and abs(y_rel) <= self.cfg.ped_lateral_range):
                continue

            vx_rel, vy_rel = self.world_vel_to_base(st["vx"], st["vy"], yaw)
            r = int(np.floor(x_rel / row_res))
            c = int(np.floor((self.cfg.ped_lateral_range - y_rel) / col_res))
            r = min(max(r, 0), size - 1)
            c = min(max(c, 0), size - 1)

            ped_map[0, r, c] = np.clip(vx_rel, self.cfg.ped_vel_min, self.cfg.ped_vel_max)
            ped_map[1, r, c] = np.clip(vy_rel, self.cfg.ped_vel_min, self.cfg.ped_vel_max)

        return ped_map

    def pool_scan(self, scan):
        pooled = np.zeros((2, self.cfg.lidar_pool_bins), dtype=np.float32)
        for i in range(self.cfg.lidar_pool_bins):
            segment = scan[self._pool_edges[i]: self._pool_edges[i + 1]]
            if segment.size == 0:
                value = scan[min(i, len(scan) - 1)]
                pooled[0, i] = value
                pooled[1, i] = value
            else:
                pooled[0, i] = np.min(segment)
                pooled[1, i] = np.mean(segment)
        return pooled

    def build_lidar_map(self, ego_id, scan):
        history = self.scan_history.setdefault(ego_id, deque(maxlen=self.cfg.lidar_history_len))
        history.append(scan.copy())

        rows = np.zeros((self.cfg.lidar_history_len * 2, self.cfg.lidar_pool_bins), dtype=np.float32)
        for idx, hist_scan in enumerate(history):
            rows[2 * idx: 2 * idx + 2] = self.pool_scan(hist_scan)

        lidar_map = np.tile(rows.reshape(-1), 4)
        return self.normalize(lidar_map, self.cfg.lidar_range_min, self.cfg.lidar_range_max)

    def build_for_robot(self, env, ego_id, states):
        subgoal, goal_dist = self.get_subgoal(env, ego_id, states)
        scan = self.get_lidar_scan(env, ego_id)
        ped_map = self.build_pedestrian_map(states, ego_id)

        ped_feat = self.normalize(ped_map.reshape(-1), self.cfg.ped_vel_min, self.cfg.ped_vel_max)
        lidar_feat = self.build_lidar_map(ego_id, scan)
        goal_feat = self.normalize(subgoal, self.cfg.goal_min, self.cfg.goal_max)
        obs = np.concatenate([ped_feat, lidar_feat, goal_feat], axis=0).astype(np.float32)
        meta = {"subgoal": subgoal, "states": states, "scan": scan, "goal_dist_world": goal_dist}
        return obs, meta

    def build_all(self, env):
        states = self.get_robot_states(env)
        obs_list = []
        meta_list = []
        for ego_id in range(len(env.robots)):
            obs, meta = self.build_for_robot(env, ego_id, states)
            obs_list.append(obs)
            meta_list.append(meta)
        return np.stack(obs_list).astype(np.float32), meta_list
