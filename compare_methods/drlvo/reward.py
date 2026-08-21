from __future__ import annotations

import math

import numpy as np


class DrlVOReward:
    def __init__(self, config):
        self.cfg = config

    @staticmethod
    def wrap_to_pi(angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

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

    def search_desired_heading(self, subgoal, states, ego_id, linear_speed):
        theta_goal = float(np.arctan2(subgoal[1], subgoal[0]))
        ego = states[ego_id]
        xr, yr, yaw = ego["x"], ego["y"], ego["yaw"]
        samples = np.linspace(-np.pi, np.pi, self.cfg.vo_samples)
        best_theta = theta_goal
        best_dist = np.inf

        for theta_u in samples:
            free = True
            for rid, st in enumerate(states):
                if rid == ego_id:
                    continue

                px, py = self.world_to_base(st["x"], st["y"], xr, yr, yaw)
                dist = np.linalg.norm([px, py])
                if dist < 1e-6:
                    free = False
                    break

                radius_sum = self.cfg.robot_radius + self.cfg.pedestrian_radius
                if dist <= radius_sum:
                    free = False
                    break

                theta = math.atan2(py, px)
                beta = math.asin(min(radius_sum / dist, 1.0))
                vx_b, vy_b = self.world_vel_to_base(st["vx"], st["vy"], yaw)
                rel_heading = math.atan2(
                    linear_speed * math.sin(theta_u) - vy_b,
                    linear_speed * math.cos(theta_u) - vx_b,
                )
                heading_error = self.wrap_to_pi(rel_heading - theta)
                if abs(heading_error) <= beta:
                    free = False
                    break

            if free:
                dist_to_goal = abs(self.wrap_to_pi(theta_u - theta_goal))
                if dist_to_goal < best_dist:
                    best_dist = dist_to_goal
                    best_theta = theta_u

        return best_theta

    def compute(self, env, ego_id, meta, prev_goal_dist, sim_time, action):
        robot = env.robots[ego_id]
        subgoal = meta["subgoal"]
        states = meta["states"]
        scan = meta["scan"]
        goal_dist = float(np.linalg.norm(subgoal))
        min_scan_dist = float(np.min(scan)) if scan.size > 0 else self.cfg.lidar_range_max

        terminated = False
        truncated = False

        if goal_dist < self.cfg.goal_margin:
            r_goal = self.cfg.r_goal
            terminated = True
        elif sim_time >= self.cfg.max_episode_time:
            r_goal = -self.cfg.r_goal
            truncated = True
        else:
            r_goal = self.cfg.r_path * (prev_goal_dist - goal_dist)

        if env.checkCollision(robot.robot) or min_scan_dist <= self.cfg.collision_dist:
            r_collision = self.cfg.r_collision
            terminated = True
        elif min_scan_dist <= self.cfg.obstacle_margin:
            r_collision = self.cfg.r_obstacle * (self.cfg.obstacle_margin - min_scan_dist)
        else:
            r_collision = 0.0

        angular_speed = abs(float(states[ego_id]["wz"]))
        if angular_speed > self.cfg.omega_smooth_threshold:
            r_smooth = self.cfg.r_rotation * angular_speed
        else:
            r_smooth = 0.0

        desired_heading = self.search_desired_heading(
            subgoal=subgoal,
            states=states,
            ego_id=ego_id,
            linear_speed=max(float(action[0]), self.cfg.min_linear_speed),
        )
        r_heading = self.cfg.r_angle * (self.cfg.theta_margin - abs(desired_heading))

        reward = float(r_goal + r_collision + r_smooth + r_heading)
        info = {
            "reward_goal": float(r_goal),
            "reward_collision": float(r_collision),
            "reward_smooth": float(r_smooth),
            "reward_heading": float(r_heading),
            "min_scan_dist": float(min_scan_dist),
            "goal_dist": float(goal_dist),
            "desired_heading": float(desired_heading),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        return reward, terminated, truncated, goal_dist, info

    def compute_all(self, env, meta_list, prev_goal_dists, sim_time, actions):
        rewards = []
        te = []
        tr = []
        goal_dists = []
        infos = []

        for ego_id, meta in enumerate(meta_list):
            reward, terminated, truncated, goal_dist, info = self.compute(
                env=env,
                ego_id=ego_id,
                meta=meta,
                prev_goal_dist=prev_goal_dists[ego_id],
                sim_time=sim_time,
                action=actions[ego_id],
            )
            rewards.append(reward)
            te.append(terminated)
            tr.append(truncated)
            goal_dists.append(goal_dist)
            infos.append(info)

        return (
            np.asarray(rewards, dtype=np.float32),
            np.asarray(te, dtype=bool),
            np.asarray(tr, dtype=bool),
            np.asarray(goal_dists, dtype=np.float32),
            infos,
        )
