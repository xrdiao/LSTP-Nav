from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

try:
    from project_paths import DATA_DIR
except ImportError:  # pragma: no cover
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from project_paths import DATA_DIR

try:
    from .agent import build_agent
    from .config import VUCANavConfig
    from .env import VUCANavEnv
except ImportError:  # pragma: no cover
    from agent import build_agent
    from config import VUCANavConfig
    from env import VUCANavEnv


class Evaluator:
    def __init__(self, env: VUCANavEnv, cfg: VUCANavConfig):
        self.cfg = cfg
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")
        self.agent = build_agent(cfg).to(self.device)
        self.agent.load_state_dict(
            torch.load(Path(cfg.model_dir) / f"PaperCOAAgent_circle.pth", map_location=self.device)
        )
        self.agent.eval()
        self.robots_num = env.robots_num

    def _get_scan_data(self):
        laser = []
        for robot in self.env.robots:
            laser.append(torch.tensor(robot.laser_buffer[-1], dtype=torch.float32, device=self.device))
        return torch.stack(laser)

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
                done = [False] * self.robots_num
                while True:
                    obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
                    laser_tensor = self._get_scan_data()
                    action = self.agent.convert_action_for_env(
                        self.agent.get_action(laser_tensor, obs_tensor)
                    ).cpu().numpy()
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

                    done = [i or j or d for i, j, d in zip(te, tr, done)]

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

        output_dir = DATA_DIR
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{self.agent.name}_{self.robots_num}_{self.env.base_env.random_obstacles}.json"
        output_path.write_text(json.dumps(data_dict, sort_keys=False, indent=4, separators=(",", ": ")))
        return data_dict


def main():
    obstacles_num = [10, 15, 20, 25, 30, 35]
    robot_nums = [5]

    for robot_num in robot_nums:
        for obs_num in obstacles_num:
            cfg = VUCANavConfig()
            cfg.render = False
            cfg.name = "circle"
            cfg.robots_num = robot_num
            cfg.obstacle_num = obs_num
            cfg.radius = 15.0
            cfg.max_episode_steps = 10000

            env = VUCANavEnv(cfg)
            evaluator = Evaluator(env, cfg)
            stats = evaluator.evaluate(times=100)
            print(evaluator.agent.name)
            print(stats)
            env.close()


if __name__ == "__main__":
    main()
