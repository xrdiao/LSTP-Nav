import time
import torch
from env_sim.env_util import *
from pathlib import Path

import pybullet as p
import numpy as np

from env_sim.my_env import MyEnv
from rl.util import *
# from rl.util_raw import *
from orca_bridge import SimulatorBridge

try:
    from project_paths import LASER_BUFFER_PATH, TURTLEBOT_URDF_PATH
except ImportError:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from project_paths import LASER_BUFFER_PATH, TURTLEBOT_URDF_PATH

def keyboard_callback(env=None):
    key_dict = p.getKeyboardEvents()
    new_vel = np.array([0., 0.])  # 初始化速度
    
    # 检测所有被按下的方向键（支持组合按键）
    if p.B3G_UP_ARROW in key_dict and (key_dict[p.B3G_UP_ARROW] & p.KEY_IS_DOWN):
        new_vel += np.array([1, 0])  # 上键增加正向速度
    
    if p.B3G_DOWN_ARROW in key_dict and (key_dict[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN):
        new_vel += np.array([-5, 0])  # 下键增加负向速度
    
    if p.B3G_LEFT_ARROW in key_dict and (key_dict[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN):
        new_vel += np.array([0, 1.5])   # 左键增加左转速度
    
    if p.B3G_RIGHT_ARROW in key_dict and (key_dict[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN):
        new_vel += np.array([0, -1.5])  # 右键增加右转速度
    
    if p.B3G_DELETE in key_dict and key_dict[p.B3G_DELETE] == p.KEY_WAS_RELEASED:
        env.reset(lim=0)
    
    # 如果有方向键被按下，返回组合速度；否则返回原速度
    return new_vel

def _update_metrics(env):
    """Update performance metrics"""
    for rob in env.robots:
        if rob.collision_num == 0 and rob.reach_goal and not rob.end_test:
            rob.end_test = True


def keyboard_control():
    env_name = ['base', 'opposite', 'transverse', 'circle']
    env_arg = env_args()
    env_arg.name = env_name[-1]
    env_arg.render = True
    env_arg.random_obstacles = 5
    env_arg.x_lim = 5
    env_arg.y_lim = 5
    env_arg.x_range=10
    env_arg.y_range=10

    env_arg.robots_num = 1
    env_arg.boundary = 0  # 50
    env_arg.random_robot = 1
    env_arg.ori_reward = True
    env_arg.radius = 7
    # env_arg.robot_camera = True

    env = MyEnv(env_arg, urdf_path=str(TURTLEBOT_URDF_PATH))
    sim_bridge = SimulatorBridge(env)

    next_obs, _ = env.reset()
    sim_bridge.reset()

    device = 'cuda'
    agent = LinearAgent().to(device)
    # Optional manual checkpoint loading can be added here if needed.
    convert = agent.convert_action_for_env

    random_agent = RandomAgent().to(device)
    env.robots[0].set_color([0.95, 0.85, 0.5, 1])
    # env.robots[1].robot_camera=True
    # env.robots[1].set_color([0.2, 0.8, 0.2,1])

    random_num = 10
    while True:
        next_obs = torch.Tensor(next_obs).to(device)

        # 跟随机器人
        # env.robots[1].set_target_pos(env.robots[0].cur_pos)
        # info = agent.get_action_and_value(next_obs[1].unsqueeze(0))
        # action = convert(info[0]).cpu().numpy().tolist()

        # orca
        sim_bridge.set_preferred_velocities()
        actions = sim_bridge.get_actions()

        # random move
        random_vel = random_agent.get_deterministic_action(next_obs[1:1+random_num])[0].cpu().numpy().tolist()

        # 人控机器人
        # vel = [keyboard_callback(env)] + action + random_vel + actions[(2+random_num-env_arg.robots_num):]
        vel = [keyboard_callback(env)]

        # 环境更新
        next_obs, reward, te, tr, info_ = env.step(vel)

        # env.plot_in_plt()
        np.save(LASER_BUFFER_PATH, env.robots[0].laser_buffer)

        need_reset = any(tr) or any(te)
        next_obs = env.reset(tr=tr, te=te)[0] if need_reset else next_obs

        done = [i or j for i, j in zip(te, tr)]
        
        # Update metrics
        _update_metrics(env)
        
        if all(item for item in done):
            sim_bridge.reset()
            break
        
        # Step simulator and update agent positions
        sim_bridge.simulator.step()
        sim_bridge.update_agents()

        env.simulate_steps = 0


if __name__ == '__main__':
    keyboard_control()
