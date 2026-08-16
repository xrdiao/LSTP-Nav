from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from tqdm.auto import tqdm

try:
    from .agent import DrlVOAgent
    from .config import DrlVOConfig
    from .env import DrlVOEnv
except ImportError:  # pragma: no cover
    from agent import DrlVOAgent
    from config import DrlVOConfig
    from env import DrlVOEnv


DEFAULT_ROBOT_NUMS = [1, 5, 10]
DEFAULT_OBSTACLE_NUMS = [5, 10, 15, 20, 25, 30, 35]
DEFAULT_EVAL_TIMES = 100
DEFAULT_ENV_RADIUS = 15.0
DEFAULT_MAX_EPISODE_STEPS = 3000


class Evaluator:
    def __init__(self, env: DrlVOEnv, cfg: DrlVOConfig):
        self.cfg = cfg
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")
        self.agent = DrlVOAgent(cfg).to(self.device)
        self.agent.load_state_dict(
            torch.load(Path(cfg.model_dir) / f"{self.agent.name}_{env.name}.pth", map_location=self.device)
        )
        self.agent.eval()
        self.robots_num = env.robots_num

    def evaluate(self, times: int = 20):
        rewards = []
        next_obs, _ = self.env.reset(seed=self.cfg.seed)

        collision_times = 0
        reach_times = 0
        trap_times = 0
        tot_time = 0.0
        tot_step = 0

        ep_avg_reward_list = []
        ep_reach_rate_list = []
        ep_trap_rate_list = []
        ep_collision_rate_list = []
        ep_avg_time_list = []
        ep_avg_step_list = []

        tq_bar = tqdm(range(1, times + 1), desc="Evaluating")

        with torch.no_grad():
            for episode_idx in tq_bar:
                start_time = time.time()
                episode_rewards = []

                while True:
                    obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
                    action = self.agent.convert_action_for_env(self.agent.get_action(obs_tensor)).cpu().numpy()
                    action[:, 0] = np.clip(action[:, 0], self.cfg.min_linear_speed, self.cfg.preferred_velocity)

                    for idx, robot in enumerate(self.env.robots):
                        if robot.reach_goal:
                            action[idx] = np.array([0.0, 0.0], dtype=np.float32)

                    next_obs, reward, te, tr, _ = self.env.step(action)
                    episode_rewards.append(reward)

                    for robot in self.env.robots:
                        if robot.collision_num == 0 and robot.reach_goal and not robot.end_test:
                            reach_times += 1
                            robot.end_test = True
                            tot_time += time.time() - start_time
                            tot_step += self.env.base_env.simulate_steps

                    done = np.logical_or(te, tr)
                    if np.all(done):
                        ep_return_vec = np.sum(np.asarray(episode_rewards), axis=0)
                        rewards.append(ep_return_vec)

                        for robot in self.env.robots:
                            if not robot.end_test:
                                if robot.collision_num == 0:
                                    trap_times += 1
                                else:
                                    collision_times += 1

                        cur_reach_rate = reach_times / episode_idx / self.robots_num
                        cur_trap_rate = trap_times / episode_idx / self.robots_num
                        cur_collision_rate = collision_times / episode_idx / self.robots_num
                        cur_avg_time = tot_time / (reach_times + 1e-8)
                        cur_avg_step = tot_step / (reach_times + 1e-8)

                        tq_bar.set_postfix(
                            {
                                "reach_rate": f"{cur_reach_rate:.2f}",
                                "trap_rate": f"{cur_trap_rate:.2f}",
                                "collision_rate": f"{cur_collision_rate:.2f}",
                                "avg_time": f"{cur_avg_time:.2f}",
                                "avg_step": f"{cur_avg_step:.2f}",
                            }
                        )

                        ep_avg_reward_list.append(float(np.mean(ep_return_vec)))
                        ep_reach_rate_list.append(float(cur_reach_rate))
                        ep_trap_rate_list.append(float(cur_trap_rate))
                        ep_collision_rate_list.append(float(cur_collision_rate))
                        ep_avg_time_list.append(float(cur_avg_time))
                        ep_avg_step_list.append(float(cur_avg_step))

                        next_obs = self.env.reset(tr=done, te=done)[0]
                        break

                    if np.all(te):
                        next_obs = self.env.reset(tr=tr, te=te)[0]

        tot_test_times = times * self.robots_num
        data_dict = {
            "avg_rewards": float(np.mean(ep_avg_reward_list)) if ep_avg_reward_list else 0.0,
            "collision_rate": collision_times / max(tot_test_times, 1),
            "reach_rate": reach_times / max(tot_test_times, 1),
            "trap_rate": trap_times / max(tot_test_times, 1),
            "avg_time": tot_time / (reach_times + 1e-8),
            "avg_step": tot_step / (reach_times + 1e-8),
            "var_avg_rewards": float(np.var(ep_avg_reward_list, ddof=0)) if ep_avg_reward_list else 0.0,
            "var_reach_rate": float(np.var(ep_reach_rate_list, ddof=0)) if ep_reach_rate_list else 0.0,
            "var_trap_rate": float(np.var(ep_trap_rate_list, ddof=0)) if ep_trap_rate_list else 0.0,
            "var_collision_rate": float(np.var(ep_collision_rate_list, ddof=0)) if ep_collision_rate_list else 0.0,
            "var_avg_time": float(np.var(ep_avg_time_list, ddof=0)) if ep_avg_time_list else 0.0,
            "var_avg_step": float(np.var(ep_avg_step_list, ddof=0)) if ep_avg_step_list else 0.0,
            "agent_name": self.agent.name,
            "env_name": self.env.name,
            "test_times": int(times),
            "obstacles": self.env.base_env.random_obstacles,
            "laser_num": self.cfg.lidar_num,
            "robots_num": self.env.robots_num,
            "x_lim": self.env.base_env.x_lim,
            "y_lim": self.env.base_env.y_lim,
        }

        output_dir = Path("data")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{self.agent.name}_{self.robots_num}_{self.env.base_env.random_obstacles}.json"
        output_path.write_text(json.dumps(data_dict, sort_keys=False, indent=4, separators=(",", ": ")))
        return data_dict


def build_eval_config(
    robot_num: int,
    obs_num: int,
    *,
    render: bool = False,
    radius: float = DEFAULT_ENV_RADIUS,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
) -> DrlVOConfig:
    cfg = DrlVOConfig()
    cfg.render = render
    cfg.name = "circle"
    cfg.robots_num = robot_num
    cfg.obstacle_num = obs_num
    cfg.radius = radius
    cfg.max_episode_steps = max_episode_steps
    return cfg


def evaluate_single_setting(
    robot_num: int = 1,
    obs_num: int = 15,
    *,
    times: int = DEFAULT_EVAL_TIMES,
    render: bool = False,
    radius: float = DEFAULT_ENV_RADIUS,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
):
    cfg = build_eval_config(
        robot_num=robot_num,
        obs_num=obs_num,
        render=render,
        radius=radius,
        max_episode_steps=max_episode_steps,
    )
    env = DrlVOEnv(cfg)
    try:
        evaluator = Evaluator(env, cfg)
        stats = evaluator.evaluate(times=times)
        print(evaluator.agent.name)
        print(stats)
        return stats
    finally:
        env.close()


def main(
    robot_nums: Iterable[int] = DEFAULT_ROBOT_NUMS,
    obstacle_nums: Iterable[int] = DEFAULT_OBSTACLE_NUMS,
    *,
    times: int = DEFAULT_EVAL_TIMES,
    render: bool = False,
    radius: float = DEFAULT_ENV_RADIUS,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
):
    all_stats = {}
    for robot_num in robot_nums:
        for obs_num in obstacle_nums:
            all_stats[(robot_num, obs_num)] = evaluate_single_setting(
                robot_num=robot_num,
                obs_num=obs_num,
                times=times,
                render=render,
                radius=radius,
                max_episode_steps=max_episode_steps,
            )
    return all_stats


if __name__ == "__main__":
    main()
