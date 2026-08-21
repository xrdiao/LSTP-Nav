import copy
import os
from pathlib import Path
from rl.util_raw import *
import time
import pybullet as p
from evaluation.recorder import Recorder
import json
from tqdm.auto import tqdm

try:
    from project_paths import DATA_DIR, MODEL_DIR, PATH_DIR, RECORD_TRAJECTORY_DIR
except ImportError:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from project_paths import DATA_DIR, MODEL_DIR, PATH_DIR, RECORD_TRAJECTORY_DIR

class Evaluator(object):
    def __init__(self, env, agent=None, is_PPO=True, is_pre=False, stack_laser=False,
                 model_path=None, result_path=None, cli_args=None, metadata=None,
                 save_artifacts=True):
        self.args = parse_args(cli_args)

        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.args.cuda else "cpu")
        self.robots_num = 0
        self.last_position = None
        self.recorder = Recorder(self.env.max_simulate_steps)
        self.stack_laser = stack_laser

        self.result_path = result_path
        self.metadata = metadata or {}
        self.save_artifacts = save_artifacts

        if agent is None:
            self.agent = RandomAgent()
        else:
            self.agent = agent().to(self.device)
            if is_pre:
                load_path = model_path or (MODEL_DIR / f"pre_{self.agent.name}.pth")
            else:
                load_path = model_path or (MODEL_DIR / f"{self.agent.name}_{self.env.name}.pth")
            self.agent.load_state_dict(torch.load(load_path, map_location=self.device))

        self.tra_color = [
            [1, 0, 0],        # 红色
            [0, 1, 0],        # 绿色
            [0, 0, 1],        # 蓝色
            [1, 0, 1],        # 紫色
            [0, 1, 1],        # 青色
            [1, 0.5, 0],      # 橙色
            [0.6, 0, 1],      # 紫罗兰色
            [0.7, 1, 0],      # 黄绿色
            [1, 0.4, 0.6],    # 粉红色
            [0.3, 0.7, 0.9],  # 浅蓝色
            [0.8, 0.2, 0.4],  # 深粉红/玫红色
            [0.5, 0.5, 0],    # 橄榄绿（新增）
            [0, 0.5, 0.5],    # 蓝绿色（新增）
            [0.9, 0.6, 0],    # 金色（新增）
            [0.4, 0, 0.8],    # 深蓝色（新增）
            [0.2, 0.8, 0.2]   # 荧光绿（新增）
        ]

        print('policy:', self.agent.name, 'robot_nums:', self.env.robots_num, 'obstacle_nums:', self.env.random_obstacles)

    def plot_trajectories(self, positions):
        [p.addUserDebugLine(self.last_position[i], positions[i], self.tra_color[i], 5,
                            physicsClientId=self.env.physics_client_id) for i in range(len(positions))]

    def record_all_robot_trajectories(self):
        positions = [rob.cur_pos + [0.01] for rob in self.env.robots]
        self.recorder.add_path(copy.deepcopy(positions))
        return positions

    def save_current_test_trajectory(self, test_index):
        trajectory_dir = RECORD_TRAJECTORY_DIR / f"{self.agent.name}_{self.robots_num}_{self.env.random_obstacles}"
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        self.recorder.save(trajectory_dir / f"{test_index}.pkl")

    def get_obstacle_records(self):
        if hasattr(self.env, 'get_obstacle_records'):
            return self.env.get_obstacle_records()
        return copy.deepcopy(self.env.obstacles)

    def evaluate(self, times: int = 5, debug: bool = False, plt_render: bool = False):
        rewards = []
        next_obs, _ = self.env.reset()

        self.last_position = [state[:3] for state in self.env.init_state]
        convert = self.agent.convert_action_for_env

        collision_times = 0
        reach_times = 0
        trap_times = 0
        tot_time = 0
        tot_step = 0
        tq_bar = tqdm(range(1, 1 + times)) if self.save_artifacts else range(1, 1 + times)

        self.robots_num = len(self.env.robots)

        # ========= 逐episode记录（用于“按episode波动”的方差）=========
        ep_avg_reward_list = []     # 每个episode：跨机器人平均的episode return
        ep_reach_rate_list = []     # 每个episode结束时：累计 reach_rate（与你postfix口径一致）
        ep_trap_rate_list = []
        ep_collision_rate_list = []
        ep_avg_time_list = []
        ep_avg_step_list = []
        # ===========================================================

        with torch.no_grad():
            for z in tq_bar:
                done = [False] * self.robots_num

                start_time = time.time()
                r = []

                while True:
                    if plt_render:
                        self.env.plot_in_plt()

                    positions = [rob.cur_pos + [0.01] for rob in self.env.robots]
                    # self.plot_trajectories(positions)

                    self.recorder.add_path(self.last_position)
                    self.last_position = positions

                    next_obs = torch.Tensor(next_obs).to(self.device)

                    if self.stack_laser:
                        laser_datas = []
                        for rob in self.env.robots:
                            laser_data = [i for i in rob.laser_buffer]
                            laser_datas.append(torch.Tensor(laser_data).to(self.device))
                        laser_datas = torch.stack(laser_datas).to(self.device)

                        info = self.agent.get_action_and_value(laser_datas, next_obs[:, :4])
                    else:
                        info = self.agent.get_action_and_value(next_obs)

                    action = convert(info[0]).cpu().numpy()
                    action[:, 0] = np.clip(action[:, 0], 0.1, 1)

                    for i, rob in enumerate(self.env.robots):
                        if rob.reach_goal:
                            action[i] = torch.Tensor([-0, 0])

                    next_obs, reward, te, tr, info_ = self.env.step(action)
                    r.append(reward)

                    if debug:
                        print('action:', action, 'reward:', reward)

                    for rob in self.env.robots:
                        if rob.collision_num == 0 and rob.reach_goal and not rob.end_test:
                            reach_times += 1
                            rob.end_test = True
                            tot_time += time.time() - start_time
                            tot_step += self.env.simulate_steps

                    done = [i or j or d for i, j, d in zip(te, tr, done)]

                    if all(item for item in done):
                        # 本episode的回报向量（每个机器人一个return）
                        ep_return_vec = np.sum(np.array(r), axis=0)  # shape: (robots_num,)
                        rewards.append(ep_return_vec)

                        # 统计本episode结束时，未end_test的机器人属于trap/collision
                        for rob in self.env.robots:
                            if not rob.end_test:
                                if rob.collision_num == 0:
                                    trap_times += 1
                                else:
                                    collision_times += 1

                        # 当前累计指标（与你原postfix一致）
                        cur_reach_rate = reach_times / z / self.robots_num
                        cur_trap_rate = trap_times / z / self.robots_num
                        cur_collision_rate = collision_times / z / self.robots_num
                        cur_avg_time = tot_time / (reach_times + 1e-8)
                        cur_avg_step = tot_step / (reach_times + 1e-8)

                        if self.save_artifacts:
                            tq_bar.set_postfix({
                                'reach_rate': f'{cur_reach_rate:.2f}',
                                'trap_rate': f'{cur_trap_rate:.2f}',
                                'collision_rate': f'{cur_collision_rate:.2f}',
                                'avg_time': f'{cur_avg_time:.2f}',
                                'avg_step': f'{cur_avg_step:.2f}'
                            })

                        # ====== 记录每个episode的标量，用于方差（按episode波动）======
                        ep_avg_reward_list.append(float(np.mean(ep_return_vec)))
                        ep_reach_rate_list.append(float(cur_reach_rate))
                        ep_trap_rate_list.append(float(cur_trap_rate))
                        ep_collision_rate_list.append(float(cur_collision_rate))
                        ep_avg_time_list.append(float(cur_avg_time))
                        ep_avg_step_list.append(float(cur_avg_step))
                        # ============================================================

                        if self.save_artifacts:
                            self.recorder.add_env_info(self.get_obstacle_records(),
                                                       copy.deepcopy(self.env.init_state),
                                                       copy.deepcopy(self.env.init_goal))
                        next_obs = self.env.reset(tr=done, te=done)[0]
                        self.last_position = [state[:3] for state in self.env.init_state]
                        if self.save_artifacts:
                            self.save_current_test_trajectory(z)
                            path_dir = PATH_DIR / self.agent.name
                            path_dir.mkdir(parents=True, exist_ok=True)
                            self.recorder.save(path_dir / f"{z}.pkl")
                            self.recorder.clear()
                        break

                    next_obs = self.env.reset(tr=tr, te=te)[0] if all(item for item in te) else next_obs

        # ============ 最终统计 + 方差（按episode波动）============
        tot_test_times = times * self.robots_num
        reach_rate = reach_times / tot_test_times
        trap_rate = trap_times / tot_test_times
        collision_rate = collision_times / tot_test_times
        avg_time = tot_time / (reach_times + 1e-8)
        avg_step = tot_step / (reach_times + 1e-8)

        avg_rewards = float(np.mean(ep_avg_reward_list)) if len(ep_avg_reward_list) > 0 else 0.0

        # 总体方差 ddof=0；如需样本方差可改 ddof=1
        var_dict = {
            'var_avg_rewards': float(np.var(ep_avg_reward_list, ddof=0)) if len(ep_avg_reward_list) > 0 else 0.0,
            'var_reach_rate': float(np.var(ep_reach_rate_list, ddof=0)) if len(ep_reach_rate_list) > 0 else 0.0,
            'var_trap_rate': float(np.var(ep_trap_rate_list, ddof=0)) if len(ep_trap_rate_list) > 0 else 0.0,
            'var_collision_rate': float(np.var(ep_collision_rate_list, ddof=0)) if len(ep_collision_rate_list) > 0 else 0.0,
            'var_avg_time': float(np.var(ep_avg_time_list, ddof=0)) if len(ep_avg_time_list) > 0 else 0.0,
            'var_avg_step': float(np.var(ep_avg_step_list, ddof=0)) if len(ep_avg_step_list) > 0 else 0.0,
        }

        data_dict = dict({
            'avg_rewards': avg_rewards,
            'collision_rate': float(collision_rate),
            'reach_rate': float(reach_rate),
            'trap_rate': float(trap_rate),
            'avg_time': float(avg_time),
            'avg_step': float(avg_step),
            'SR': float(reach_rate),
            'CR': float(collision_rate),
            'TR': float(trap_rate),
            'AT': float(avg_time),
            'AS': float(avg_step),

            # 写入方差
            **var_dict,

            'agent_name': self.agent.name,
            'env_name': self.env.name,
            'test_times': int(times),
            'obstacles': self.env.random_obstacles,
            'laser_num': LASER_NUM,
            'robots_num': self.env.robots_num,
            'x_lim': self.env.x_lim,
            'y_lim': self.env.y_lim
        })
        data_dict.update(self.metadata)

        data_json = json.dumps(data_dict, sort_keys=False, indent=4, separators=(',', ': '))
        result_path = Path(self.result_path) if self.result_path is not None else (
            DATA_DIR / f"{self.agent.name}_{self.robots_num}_{self.env.random_obstacles}.json"
        )
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with result_path.open('w') as f:
            f.write(data_json)

        return data_dict
