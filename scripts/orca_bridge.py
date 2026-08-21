import json
import math
import random
import time
from pathlib import Path
import sys
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import rvo.math as rvo_math
import numpy as np
from rvo.vector import Vector2
from rvo.simulator import Simulator

from env_sim.env_util import *
from env_sim.argument import *
from train import create_env

from project_paths import DATA_DIR

# Constants
ROBOT_WIDTH = 0.5  # Should be defined in argument.py or here
MAX_SPEED = 1.0    # Should be defined in argument.py or here

class SimulatorBridge:
    """Handles communication between MyEnv and RVO2 Simulator"""

    def __init__(self, env):
        self.env = env
        self.simulator = Simulator()
        self.simulator.set_time_step(env.time_step)
        self.goals = []
        self.obstacles = []

        self._initialize_simulator()

    def _initialize_simulator(self):
        """Initialize simulator with default agent parameters"""
        self.simulator.set_agent_defaults(
            4.0, 10, 10.0, 10.0,
            ROBOT_WIDTH * 1.7, MAX_SPEED,
            Vector2(0.0, 0.0)
        )

        # Add agents
        for rob in self.env.robots:
            self.simulator.add_agent(Vector2(rob.init_pos[0], rob.init_pos[1]))
            self.goals.append(Vector2(rob.target_pos[0], rob.target_pos[1]))

        # Add obstacles
        for obstacle in self.env.obstacles:
            sim_obstacle = self._convert_obstacle_to_sim(obstacle)
            self.simulator.add_obstacle(sim_obstacle)
            self.obstacles.append(sim_obstacle)

        self.simulator.process_obstacles()

    def _convert_obstacle_to_sim(self, position):
        """Convert environment obstacle to simulator format"""
        x, y = position[0], position[1]
        size = 0.5
        left = x - size
        right = x + size
        top = y + size
        bottom = y - size

        return [
            Vector2(right, bottom),
            Vector2(right, top),
            Vector2(left, top),
            Vector2(left, bottom)
        ]

    def reset(self):
        """Reset simulator for new episode"""
        self.simulator.reset()
        self.simulator.agents_ = []
        self.goals = []
        self.obstacles = []
        self._initialize_simulator()

    def set_preferred_velocities(self):
        """Set preferred velocities for all agents"""
        for i in range(self.simulator.num_agents):
            goal_vector = self.goals[i] - self.simulator.agents_[i].position_

            if rvo_math.abs_sq(goal_vector) > 1.0:
                goal_vector = rvo_math.normalize(goal_vector)

            self.simulator.set_agent_pref_velocity(i, goal_vector)

            # Add some randomness
            angle = random.random() * 2.0 * math.pi
            dist = random.random() * 0.5
            self.simulator.set_agent_pref_velocity(
                i,
                self.simulator.agents_[i].pref_velocity_ +
                dist * Vector2(math.cos(angle), math.sin(angle))
            )

    def get_actions(self):
        """Compute actions for all agents"""
        actions = []
        self.simulator.kd_tree_.build_agent_tree()

        for agent_no in range(self.simulator.num_agents):
            self.simulator.agents_[agent_no].compute_neighbors()
            self.simulator.agents_[agent_no].compute_new_velocity()

            action = self.simulator.agents_[agent_no].new_velocity_
            robot = self.env.robots[agent_no]
            actions.append(robot.cal_effective_cmd([action.x, action.y]))

        return actions

    def update_agents(self):
        """Update simulator with current agent states from environment"""
        for agent_no, agent in enumerate(self.env.robots):
            obs_dict = agent.get_vel_and_pos()
            pos = obs_dict['pos']
            self.simulator.agents_[agent_no].position_ = Vector2(pos[0], pos[1])

class ExperimentRunner:
    """Handles running and tracking the experiment"""

    def __init__(self, env, num_trials=1000):
        self.env = env
        self.num_trials = num_trials
        self.sim_bridge = SimulatorBridge(env)

        # Metrics (累计)
        self.reach_times = 0
        self.collision_times = 0
        self.trap_times = 0
        self.tot_time = 0
        self.tot_step = 0

        # reward累计（按episode记录向量/标量）
        self.rewards = []

        # ========= 逐episode记录（用于“按episode波动”的方差）=========
        self.ep_avg_reward_list = []     # 每个episode：跨机器人平均的episode return
        self.ep_reach_rate_list = []     # 每个episode结束时：累计 reach_rate（与你postfix口径一致）
        self.ep_trap_rate_list = []
        self.ep_collision_rate_list = []
        self.ep_avg_time_list = []
        self.ep_avg_step_list = []
        # ===========================================================

        # 用于逐episode累计reward
        self._episode_rewards = []

    def run(self):
        """Run the experiment"""
        self.env.reset()
        self.sim_bridge.reset()

        tq_bar = tqdm(range(1, 1 + self.num_trials))

        for trial in tq_bar:
            start_time = time.time()

            # 每个episode开始，清空累计reward
            self._episode_rewards = []

            self._run_episode(start_time, trial)
            self._update_progress(tq_bar, trial)

    def _run_episode(self, start_time, trial_idx: int):
        """Run a single episode"""
        while True:
            self.sim_bridge.set_preferred_velocities()
            actions = self.sim_bridge.get_actions()

            # Step environment
            next_obs, reward, te, tr, info = self.env.step(actions)

            # 记录逐step reward（通常是每个机器人一个reward向量）
            self._episode_rewards.append(reward)

            done = [i or j for i, j in zip(te, tr)]

            # Update metrics
            self._update_metrics(start_time)

            if all(item for item in done):
                self._handle_episode_end(done, trial_idx)
                break

            # Step simulator and update agent positions
            self.sim_bridge.simulator.step()
            self.sim_bridge.update_agents()

    def _update_metrics(self, start_time):
        """Update performance metrics"""
        for rob in self.env.robots:
            if rob.collision_num == 0 and rob.reach_goal and not rob.end_test:
                self.reach_times += 1
                rob.end_test = True
                self.tot_time += time.time() - start_time
                self.tot_step += self.env.simulate_steps

    def _handle_episode_end(self, done, trial_idx: int):
        """Handle end of episode"""
        # 1) episode return（每个机器人一个return）
        ep_return_vec = np.sum(np.array(self._episode_rewards), axis=0)
        self.rewards.append(ep_return_vec)

        # 2) 更新trap/collision计数
        for rob in self.env.robots:
            if not rob.end_test:
                if rob.collision_num == 0:
                    self.trap_times += 1
                else:
                    self.collision_times += 1

        # 3) 记录“每个episode结束时”的累计指标（用于按episode求方差）
        cur_reach_rate = self.reach_times / trial_idx / self.env.robots_num
        cur_trap_rate = self.trap_times / trial_idx / self.env.robots_num
        cur_collision_rate = self.collision_times / trial_idx / self.env.robots_num
        cur_avg_time = self.tot_time / (self.reach_times + 1e-8)
        cur_avg_step = self.tot_step / (self.reach_times + 1e-8)

        self.ep_avg_reward_list.append(float(np.mean(ep_return_vec)))
        self.ep_reach_rate_list.append(float(cur_reach_rate))
        self.ep_trap_rate_list.append(float(cur_trap_rate))
        self.ep_collision_rate_list.append(float(cur_collision_rate))
        self.ep_avg_time_list.append(float(cur_avg_time))
        self.ep_avg_step_list.append(float(cur_avg_step))

        # 4) Reset environment and simulator
        self.env.reset(tr=done, te=done)[0] if all(item for item in done) else None
        self.sim_bridge.reset()

    def _update_progress(self, tq_bar, trial):
        """Update progress bar with current metrics"""
        tq_bar.set_postfix({
            'reach_rate': f'{self.reach_times / trial / self.env.robots_num:.2f}',
            'trap_rate': f'{self.trap_times / trial / self.env.robots_num:.2f}',
            'collision_rate': f'{self.collision_times / trial / self.env.robots_num:.2f}',
            'avg_time': f'{self.tot_time / (self.reach_times + 1e-8):.2f}',
            'avg_step': f'{self.tot_step / (self.reach_times + 1e-8):.2f}'
        })

    def save_results(self):
        """Save experiment results to JSON file (含按episode波动的方差)"""
        tot_test_times = self.num_trials * self.env.robots_num

        avg_rewards = float(np.mean(self.ep_avg_reward_list)) if len(self.ep_avg_reward_list) > 0 else 0.0

        # 总体方差 ddof=0；如需样本方差可改 ddof=1
        var_dict = {
            'var_avg_rewards': float(np.var(self.ep_avg_reward_list, ddof=0)) if len(self.ep_avg_reward_list) > 0 else 0.0,
            'var_reach_rate': float(np.var(self.ep_reach_rate_list, ddof=0)) if len(self.ep_reach_rate_list) > 0 else 0.0,
            'var_trap_rate': float(np.var(self.ep_trap_rate_list, ddof=0)) if len(self.ep_trap_rate_list) > 0 else 0.0,
            'var_collision_rate': float(np.var(self.ep_collision_rate_list, ddof=0)) if len(self.ep_collision_rate_list) > 0 else 0.0,
            'var_avg_time': float(np.var(self.ep_avg_time_list, ddof=0)) if len(self.ep_avg_time_list) > 0 else 0.0,
            'var_avg_step': float(np.var(self.ep_avg_step_list, ddof=0)) if len(self.ep_avg_step_list) > 0 else 0.0,
        }

        data_dict = {
            'avg_rewards': avg_rewards,
            'collision_rate': self.collision_times / tot_test_times,
            'reach_rate': self.reach_times / tot_test_times,
            'trap_rate': self.trap_times / tot_test_times,
            'avg_time': self.tot_time / (self.reach_times + 1e-8),
            'avg_step': self.tot_step / (self.reach_times + 1e-8),

            # 方差写入
            **var_dict,

            'agent_name': 'orca',
            'env_name': self.env.name,
            'test_times': self.num_trials,
            'obstacles': self.env.random_obstacles,
            'laser_num': LASER_NUM,
            'robots_num': self.env.robots_num,
            'x_lim': self.env.x_lim,
            'y_lim': self.env.y_lim
        }

        filename = DATA_DIR / f"ORCA_{self.env.robots_num}_{self.env.random_obstacles}.json"
        filename.parent.mkdir(parents=True, exist_ok=True)
        with filename.open('w') as f:
            json.dump(data_dict, f, sort_keys=False, indent=4, separators=(',', ': '))

def main(robot_num=1, obs_num=15):
    """Main function to run the experiment"""
    env, _ = create_env(render=False, obstacle_num=obs_num, robot_num=robot_num)
    env.set_max_step(8000)
    print('policy: orca', 'robot_nums:', env.robots_num, 'obstacle_nums:', env.random_obstacles)
    experiment = ExperimentRunner(env, num_trials=100)
    experiment.run()
    experiment.save_results()
    env.close()

if __name__ == '__main__':
    obstacles_num = [5, 10, 15, 20, 25, 30, 35]
    robot_nums = [15]
    for robot_num_ in robot_nums:
        for obs_num_ in obstacles_num:
            main(robot_num=robot_num_, obs_num=obs_num_)
