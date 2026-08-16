from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import os
import sys

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from env_sim.argument import LASER_NUM, MAX_ROTATION_SPEED, ROBOT_LASER_BUFFER

URDF_PATH = ROOT_DIR / "env_sim" / "utils" / "data" / "turtlebot.urdf"


@dataclass
class VUCANavConfig:
    name: str = "circle"
    render: bool = True
    seed: int = 1

    robots_num: int = 10
    obstacle_num: int = 35
    boundary: int = 0
    x_range: float = 10.0
    y_range: float = 10.0
    x_lim: float = 10.0
    y_lim: float = 10.0
    radius: float = 16.0
    control_rate: int = 60
    random_angle_obs: bool = True
    robot_camera: bool = False
    random_robot: int = 0
    test_mode: int = 1
    ori_reward: bool = False

    robot_radius: float = 0.185
    human_radius: float = 0.185
    preferred_velocity: float = 1.0
    angular_limit: float = MAX_ROTATION_SPEED
    min_linear_speed: float = 0.05
    discomfort_dist: float = 0.2
    velocity_weight: float = 0.2

    lidar_num: int = LASER_NUM
    laser_buffer_len: int = ROBOT_LASER_BUFFER

    state_dim: int = 12
    policy_name: str = "paper"
    laser_encoder_dims: Tuple[int, int] = (128, 128)
    state_encoder_dims: Tuple[int, int] = (64, 64)
    policy_net_arch: Tuple[int, int] = (128, 64)
    value_net_arch: Tuple[int, int] = (128, 64)
    init_logstd: float = -0.5
    robot_state_dim: int = 6
    social_state_dim: int = 6
    paper_rnn_hidden_dim: int = 64
    paper_pairwise_dims: Tuple[int, int] = (64, 64)
    paper_attention_dims: Tuple[int, int] = (64, 64)
    paper_lidar_dims: Tuple[int, int] = (128, 128)

    paper_reward_weight: float = 1.0

    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    n_steps: int = 256
    minibatch_size: int = 256
    n_epochs: int = 10
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 1.0
    max_grad_norm: float = 1.0
    total_timesteps: int = 1_000_000
    cuda: bool = True
    norm_adv: bool = True
    clip_vloss: bool = True
    target_kl: float = 0.01
    torch_deterministic: bool = True
    save_interval_updates: int = 50
    early_stop_reach_times: int = 15
    max_episode_steps: int = 3000

    model_dir: Path = ROOT_DIR / "vuca_nav" / "checkpoints"
    log_dir: Path = ROOT_DIR / "vuca_nav" / "runs"

    @property
    def batch_size(self) -> int:
        return self.n_steps * self.robots_num
