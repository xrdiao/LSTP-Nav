from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env_sim.my_env import MyEnv

try:
    from .config import DrlVOConfig, URDF_PATH
    from .observation import DrlVOObservationBuilder
    from .reward import DrlVOReward
except ImportError:  # pragma: no cover
    from config import DrlVOConfig, URDF_PATH
    from observation import DrlVOObservationBuilder
    from reward import DrlVOReward


class DrlVOEnv:
    def __init__(self, config: DrlVOConfig):
        self.cfg = config
        self.base_env = self._create_base_env()
        self.obs_builder = DrlVOObservationBuilder(self.cfg)
        self.reward_model = DrlVOReward(self.cfg)
        self.name = self.base_env.name
        self.robots = self.base_env.robots
        self.robots_num = self.base_env.robots_num
        self.num_obs = self.cfg.observation_dim
        self.num_actions = 2
        self.collision_num = 0
        self.reach_num = 0
        self.prev_goal_dists = np.zeros(self.robots_num, dtype=np.float32)

    def _create_base_env(self):
        env_args = SimpleNamespace(
            render=self.cfg.render,
            random_robot=self.cfg.random_robot,
            robots_num=self.cfg.robots_num,
            boundary=self.cfg.boundary,
            random_obstacles=self.cfg.obstacle_num,
            x_lim=self.cfg.x_lim,
            y_lim=self.cfg.y_lim,
            x_range=self.cfg.x_range,
            y_range=self.cfg.y_range,
            radius=self.cfg.radius,
            safe=False,
            control_rate=self.cfg.control_rate,
            name=self.cfg.name,
            ori_reward=self.cfg.ori_reward,
            test_mode=self.cfg.test_mode,
            random_angle_obs=self.cfg.random_angle_obs,
            robot_camera=self.cfg.robot_camera,
        )
        env = MyEnv(env_args, urdf_path=str(URDF_PATH))
        env.set_max_step(self.cfg.max_episode_steps)
        return env

    def reset(self, tr=None, te=None, seed=0):
        _base_obs, info = self.base_env.reset(tr=tr, te=te, seed=seed)
        self.robots = self.base_env.robots
        self.robots_num = self.base_env.robots_num
        self.obs_builder.reset(self.base_env)
        state_obs, meta = self.obs_builder.build_all(self.base_env)
        self.prev_goal_dists = np.asarray(
            [float(np.linalg.norm(item["subgoal"])) for item in meta],
            dtype=np.float32,
        )
        self.collision_num = self.base_env.collision_num
        self.reach_num = self.base_env.reach_num
        return state_obs, {"meta": meta, "base_info": info}

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        base_obs, base_rewards, _te, _tr, base_info = self.base_env.step(actions)
        state_obs, meta = self.obs_builder.build_all(self.base_env)
        sim_time = self.base_env.simulate_steps / max(self.cfg.control_rate, 1)
        paper_rewards, te, tr, goal_dists, reward_infos = self.reward_model.compute_all(
            env=self.base_env,
            meta_list=meta,
            prev_goal_dists=self.prev_goal_dists,
            sim_time=sim_time,
            actions=actions,
        )
        rewards = np.asarray(base_rewards, dtype=np.float32) + paper_rewards
        self.prev_goal_dists = goal_dists
        self.collision_num = self.base_env.collision_num
        self.reach_num = self.base_env.reach_num

        infos = []
        for idx in range(len(meta)):
            info = {
                "goal_dist": float(goal_dists[idx]),
                "subgoal": meta[idx]["subgoal"],
                "terminated": bool(te[idx]),
                "truncated": bool(tr[idx]),
            }
            info.update(reward_infos[idx])
            infos.append(info)

        return state_obs, rewards, te, tr, {"base_obs": base_obs, "base_info": base_info, "robot_infos": infos}

    def close(self):
        self.base_env.close()
