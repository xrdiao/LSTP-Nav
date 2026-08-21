from neupan import neupan
import numpy as np

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env_sim.env_util import *
from train import create_env
from tqdm.auto import tqdm
import json
import time

from project_paths import DATA_DIR, NEUPAN_CONVEX_DIFF_PLANNER_PATH

def get_robot_state(env, robot_id):
    scans = {
        'ranges': [], 'angle_min': -np.pi/2, 'angle_max': np.pi/2, 'angle_increment': np.pi/100,
        'range_min': 0.02, 'range_max': 4.0, 'scan_time': 0.1, 'time_increment': 0.0004999999914709196,
    }
    rob = env.robots[robot_id]

    pos = rob.cur_pos
    yaw = rob.theta
    scans['ranges'] = np.array(rob.laser)

    return np.array([pos[0], pos[1], yaw]).reshape(3, 1), scans

def nerpan_step(
    env,
    point_vel=False,
    robot_id=0,
    neupan_planner=None,
):
    robot_state, lidar_scan = get_robot_state(env, robot_id)

    if point_vel:
        points, point_velocities = neupan_planner.scan_to_point_velocity(robot_state, lidar_scan)
    else:
        points = neupan_planner.scan_to_point(robot_state, lidar_scan)
        point_velocities = None

    action, info = neupan_planner(robot_state, points, point_velocities)
    return action

def run(env, times=1000):
    planner_path_file = str(NEUPAN_CONVEX_DIFF_PLANNER_PATH)
    controllers = []
    rewards = []
    next_obs_, _ = env.reset()
    tq_bar = tqdm(range(1, 1 + times))

    collision_times = 0
    reach_times = 0
    trap_times = 0
    tot_time = 0
    tot_step = 0

    # ========= 逐episode记录（用于“按episode波动”的方差）=========
    ep_avg_reward_list = []     # 每个episode：跨机器人平均的episode return
    ep_reach_rate_list = []     # 每个episode结束时：累计 reach_rate（与你postfix口径一致）
    ep_trap_rate_list = []
    ep_collision_rate_list = []
    ep_avg_time_list = []
    ep_avg_step_list = []
    # ===========================================================

    for rob in env.robots:
        neupan_planner = neupan.init_from_yaml(planner_path_file)
        neupan_planner.update_initial_path_from_waypoints([
            np.array([rob.init_pos[0], rob.init_pos[1], rob.theta]).reshape(3, 1),
            np.array([rob.target_pos[0], rob.target_pos[1], rob.theta]).reshape(3, 1)
        ])
        controllers.append(neupan_planner)

    for z in tq_bar:
        r = []
        start_time = time.time()

        while True:
            actions = []
            for rob_id in range(env.robots_num):
                action = nerpan_step(env, False, rob_id, controllers[rob_id])
                actions.append(action)

            next_obs, reward, te, tr, info_ = env.step(actions)
            r.append(reward)

            for rob in env.robots:
                if rob.collision_num == 0 and rob.reach_goal and not rob.end_test:
                    reach_times += 1
                    rob.end_test = True
                    tot_time += time.time() - start_time
                    tot_step += env.simulate_steps

            done = [i or j for i, j in zip(te, tr)]

            if all(item for item in done):
                # 本episode的回报向量（每个机器人一个return）
                ep_return_vec = np.sum(np.array(r), axis=0)  # shape: (robots_num,)
                rewards.append(ep_return_vec)

                controllers = []

                for rob in env.robots:
                    if not rob.end_test:
                        if rob.collision_num == 0:
                            trap_times += 1
                        else:
                            collision_times += 1

                    neupan_planner = neupan.init_from_yaml(planner_path_file)
                    neupan_planner.update_initial_path_from_waypoints([
                        np.array([rob.init_pos[0], rob.init_pos[1], rob.theta]).reshape(3, 1),
                        np.array([rob.target_pos[0], rob.target_pos[1], np.pi]).reshape(3, 1)
                    ])
                    controllers.append(neupan_planner)

                next_obs = env.reset(tr=done, te=done)[0]

                # 当前累计指标（与你原postfix一致）
                cur_reach_rate = reach_times / z / env.robots_num
                cur_trap_rate = trap_times / z / env.robots_num
                cur_collision_rate = collision_times / z / env.robots_num
                cur_avg_time = tot_time / (reach_times + 1e-8)
                cur_avg_step = tot_step / (reach_times + 1e-8)

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

                break

            next_obs = env.reset(tr=tr, te=te)[0] if all(item for item in te) else next_obs

        # 你原来是每10次写一次，这里保留：每10次更新一次json（包含方差）
        if z % 10 == 0:
            tot_test_times = z * env.robots_num
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

                # 方差写入
                **var_dict,

                'agent_name': 'neupan',
                'env_name': env.name,
                'test_times': int(z),
                'obstacles': env.random_obstacles,
                'laser_num': 100,
                'robots_num': env.robots_num,
            })

            data_json = json.dumps(data_dict, sort_keys=False, indent=4, separators=(',', ': '))
            output_path = DATA_DIR / f"nerpan_{env.robots_num}_{env.random_obstacles}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open('w') as f:
                f.write(data_json)

def main(robot_num=1, obs_num=15):
    env, _ = create_env(render=False, name='circle', robot_num=robot_num, obstacle_num=obs_num)
    env.max_simulate_steps = 6000
    print('policy: neupan', 'robot_nums:', env.robots_num, 'obstacle_nums:', env.random_obstacles)
    run(env, times=100)
    env.close()

if __name__ == '__main__':
    obstacles_num = [5, 10,15,20,25, 30, 35]
    robot_nums = [10]
    for robot_num_ in robot_nums:
        for obs_num_ in obstacles_num:
            main(robot_num=robot_num_, obs_num=obs_num_)
