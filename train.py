from pathlib import Path

from env_sim.argument import LASER_NUM
# from rl.model import PPO
from rl.model_raw import PPO
from env_sim.env_util import *
from env_sim.my_env import MyEnv

try:
    from project_paths import TURTLEBOT_URDF_PATH
except ImportError:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from project_paths import TURTLEBOT_URDF_PATH

env_name = ['circle', 'u', 'dumbbell', 'room']

def create_env(name=env_name[0], render=True, robot_camera=False, robot_num=6, obstacle_num=35, radius=15,
               x_lim=10, y_lim=10, x_range=10, y_range=10, env_overrides=None, cli_args=None):
    env_arg = env_args(cli_args)
    env_arg.name = name
    env_arg.render = render
    env_arg.random_obstacles = obstacle_num
    env_arg.x_lim = x_lim
    env_arg.y_lim = y_lim
    env_arg.x_range=x_range
    env_arg.y_range=y_range

    env_arg.boundary = 0
    env_arg.safe = False
    env_arg.ori_reward = True
    env_arg.robot_camera = robot_camera

    env_arg.random_robot=0
    env_arg.test_mode = 1

    env_arg.robots_num = robot_num
    env_arg.radius = radius  # 15, 16

    if env_overrides:
        for key, value in env_overrides.items():
            setattr(env_arg, key, value)

    env = MyEnv(env_arg, urdf_path=str(TURTLEBOT_URDF_PATH))
    env.set_max_step(2500)
    return env, env_arg

def train():
    env, env_arg = create_env(render=False)
    # agent = PPO(env, policy="LinearAgent")
    agent = PPO(env, policy="AttentionAgent")

    # agent.load_model()
    print('robot_nums:', env.robots_num, 'agent: ' + agent.agent.name + ' env: ' + env.name, 'laser nums: ', LASER_NUM, 'ori_reward:', env_arg.ori_reward)
    print('Start training PPO')
    # env.set_max_step(1500)
    agent.train(random_robot=env_arg.random_robot)
    print('robot_nums:', env.robots_num, ' env: ' + env.name)


if __name__ == '__main__':
    train()
    # test()
