from __future__ import annotations

import numpy as np


class COAObservationBuilder:
    def __init__(self, config):
        self.cfg = config

    @staticmethod
    def wrap_to_pi(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    @staticmethod
    def rotate_to_frame(x: float, y: float, heading: float):
        c = np.cos(heading)
        s = np.sin(heading)
        return c * x + s * y, -s * x + c * y

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

    def _nearest_human_summary(self, states, ego_id):
        ego = states[ego_id]
        nearest = None
        for rid, st in enumerate(states):
            if rid == ego_id:
                continue
            dx = st["x"] - ego["x"]
            dy = st["y"] - ego["y"]
            dist = float(np.hypot(dx, dy))
            if nearest is None or dist < nearest["dist"]:
                px, py = self.rotate_to_frame(dx, dy, ego["yaw"])
                rvx, rvy = self.rotate_to_frame(st["vx"] - ego["vx"], st["vy"] - ego["vy"], ego["yaw"])
                nearest = {
                    "px": px,
                    "py": py,
                    "rvx": rvx,
                    "rvy": rvy,
                    "dist": dist,
                    "visible": 1.0,
                }
        if nearest is None:
            return np.zeros(6, dtype=np.float32)
        return np.asarray(
            [nearest["px"], nearest["py"], nearest["rvx"], nearest["rvy"], nearest["dist"], nearest["visible"]],
            dtype=np.float32,
        )

    def build_for_robot(self, env, base_observations, states, ego_id):
        robot = env.robots[ego_id]
        raw = np.asarray(base_observations[ego_id], dtype=np.float32)
        min_scan = float(np.min(np.asarray(robot.laser, dtype=np.float32)))
        nearest_human = self._nearest_human_summary(states, ego_id)

        state = np.concatenate(
            [
                np.asarray(
                    [
                        raw[0],
                        raw[1],
                        raw[2],
                        raw[3],
                        min_scan,
                        self.cfg.preferred_velocity,
                    ],
                    dtype=np.float32,
                ),
                nearest_human,
            ],
            axis=0,
        ).astype(np.float32)

        meta = {
            "ego_id": ego_id,
            "states": states,
            "goal_dist": float(raw[0]),
            "goal_angle": float(raw[1]),
            "linear_speed": float(raw[2]),
            "angular_speed": float(raw[3]),
            "min_scan": min_scan,
            "nearest_human": nearest_human,
        }
        return state, meta

    def build_all(self, env, base_observations):
        states = self.get_robot_states(env)
        obs_list = []
        meta_list = []
        for ego_id in range(len(env.robots)):
            obs, meta = self.build_for_robot(env, base_observations, states, ego_id)
            obs_list.append(obs)
            meta_list.append(meta)
        return np.stack(obs_list).astype(np.float32), meta_list
