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
    from .config import URDF_PATH, VUCANavConfig
    from .observation import COAObservationBuilder
    from .reward import PaperReward
except ImportError:  # pragma: no cover
    from config import URDF_PATH, VUCANavConfig
    from observation import COAObservationBuilder
    from reward import PaperReward


class VUCANavEnv:
    def __init__(self, config: VUCANavConfig):
        self.cfg = config
        self.base_env = self._create_base_env()
        self.obs_builder = COAObservationBuilder(self.cfg)
        self.reward_model = PaperReward(self.cfg)
        self.name = self.base_env.name
        self.robots_num = self.base_env.robots_num
        self.num_obs = self.cfg.state_dim
        self.num_actions = 2
        self.observation_space = SimpleNamespace(shape=(self.cfg.state_dim,))
        self.action_space = self.base_env.action_space
        self.robots = self.base_env.robots
        self.collision_num = 0
        self.reach_num = 0

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

    def _build_state(self, base_observations):
        return self.obs_builder.build_all(self.base_env, base_observations)

    def reset(self, tr=None, te=None, seed=0):
        base_obs, info = self.base_env.reset(tr=tr, te=te, seed=seed)
        self.robots = self.base_env.robots
        self.robots_num = self.base_env.robots_num
        state_obs, meta = self._build_state(base_obs)
        self.collision_num = self.base_env.collision_num
        self.reach_num = self.base_env.reach_num
        return state_obs, {"meta": meta, "base_info": info}

    def step(self, actions):
        base_obs, base_rewards, te, tr, base_info = self.base_env.step(actions)
        state_obs, meta = self._build_state(base_obs)
        paper_rewards, paper_infos = self.reward_model.compute_all(meta)
        rewards = np.asarray(base_rewards, dtype=np.float32) + self.cfg.paper_reward_weight * paper_rewards
        self.collision_num = self.base_env.collision_num
        self.reach_num = self.base_env.reach_num
        infos = []
        for idx in range(len(meta)):
            info = {"base_reward": float(base_rewards[idx]), "paper_reward": float(paper_rewards[idx])}
            info.update(paper_infos[idx])
            infos.append(info)
        return state_obs, rewards, te, tr, {"base_info": base_info, "robot_infos": infos}

    def close(self):
        self.base_env.close()
