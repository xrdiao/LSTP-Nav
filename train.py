import os

import numpy as np

from rl.model import PPO
import gymnasium as gym
import pybullet as p
from env_sim.my_env import MyEnv

base_path = os.path.dirname(os.path.abspath(__file__))
urdf_path = base_path + '/env_sim/utils/data/turtlebot.urdf'


def main():
    render = False
    # env = gym.make('MyEnv-v0', render=render, urdf_path=urdf_path)
    env = MyEnv(render=render, urdf_path=urdf_path)
    test_env = MyEnv(render=render, urdf_path=urdf_path)
    agent = PPO(env, test_env)

    robot_nums = 1
    lim = 5

    print('robot_nums:', robot_nums)
    for i in range(robot_nums):
        env.add_random_robot(lim=lim)
    for _ in range(robot_nums):
        test_env.add_random_robot(lim=lim)

    if render:
        env.show_goal_point()
        p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=0, cameraPitch=-89.9,
                                     cameraTargetPosition=[0, 0, 0])
    print('start train 1')
    agent.update(lim=lim)


if __name__ == '__main__':
    main()
