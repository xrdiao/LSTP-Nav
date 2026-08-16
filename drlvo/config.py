from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import sys

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from env_sim.argument import LASER_LENGTH, LASER_NUM, MAX_ROTATION_SPEED

URDF_PATH = ROOT_DIR / "env_sim" / "utils" / "data" / "turtlebot.urdf"


@dataclass
class DrlVOConfig:
    name: str = "circle"
    render: bool = True
    seed: int = 1

    robots_num: int = 1
    obstacle_num: int = 0
    boundary: int = 0
    x_range: float = 10.0
    y_range: float = 10.0
    x_lim: float = 10.0
    y_lim: float = 10.0
    radius: float = 16.0
    control_rate: int = 20
    random_angle_obs: bool = True
    robot_camera: bool = False
    random_robot: int = 0
    test_mode: int = 1
    ori_reward: bool = False

    robot_radius: float = 0.185
    pedestrian_radius: float = 0.185
    preferred_velocity: float = 0.5
    angular_limit: float = 2.0
    min_linear_speed: float = 0.0

    raw_lidar_num: int = LASER_NUM
    lidar_num: int = LASER_NUM
    lidar_history_len: int = 10
    lidar_pool_bins: int = 80
    lidar_range_min: float = 0.0
    lidar_range_max: float = LASER_LENGTH
    lidar_left_to_right: bool = True

    ped_map_size: int = 80
    ped_forward_range: float = 20.0
    ped_lateral_range: float = 10.0
    ped_vel_min: float = -2.0
    ped_vel_max: float = 2.0

    lookahead_distance: float = 2.0
    goal_min: float = -2.0
    goal_max: float = 2.0
    goal_margin: float = 0.3
    max_episode_time: float = 25.0
    max_episode_steps: int = 3000

    linear_vel_min: float = 0.0
    linear_vel_max: float = 0.5
    angular_vel_min: float = -2.0
    angular_vel_max: float = 2.0

    r_goal: float = 20.0
    r_path: float = 3.2
    r_collision: float = -20.0
    r_obstacle: float = -0.2
    collision_dist: float = 0.3
    obstacle_margin: float = 1.2
    r_rotation: float = -0.1
    omega_smooth_threshold: float = 1.0
    r_angle: float = 0.6
    theta_margin: float = np.pi / 6.0
    vo_samples: int = 61

    feature_dim: int = 256
    actor_hidden_dims: Tuple[int, int] = (256, 128)
    critic_hidden_dims: Tuple[int, int] = (256, 128)
    init_logstd: float = -0.5

    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    n_steps: int = 128
    minibatch_size: int = 32
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

    model_dir: Path = ROOT_DIR / "drlvo" / "checkpoints"
    log_dir: Path = ROOT_DIR / "drlvo" / "runs"

    @property
    def observation_dim(self) -> int:
        return 2 * self.ped_map_size * self.ped_map_size + self.ped_map_size * self.ped_map_size + 2

    @property
    def batch_size(self) -> int:
        return self.n_steps * self.robots_num
