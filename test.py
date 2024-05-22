from env_sim.my_env import MyEnv
from rl.model import *
import numpy as np
import gymnasium as gym
import os
import pybullet as p

base_path = os.path.dirname(os.path.abspath(__file__))
urdf_path = base_path + '/env_sim/utils/data/turtlebot.urdf'

device = torch.device('cuda') if torch.cuda.is_available() \
    else torch.device('cpu')


def test():
    render = True
    # env = gym.make('MyEnv-v0', render=render, urdf_path=urdf_path)
    env = MyEnv(render=render, urdf_path=urdf_path)
    p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=0, cameraPitch=-89.9,
                                 cameraTargetPosition=[0, 0, 0])

    robot_nums = 1
    lim = 5

    print('robot_nums:', robot_nums)
    for i in range(robot_nums):
        goal = [i, 0.3 + i]

        yaw = np.pi * robot_nums
        ori = p.getQuaternionFromEuler([0, 0, np.pi/2], env.physics_client_id)
        state = [i, 0, 0.01] + list(ori)
        env.add_robot(state, goal)
        # env.add_random_robot(lim=lim)

    agent = PPO(env, test_env=env)
    rewards = []
    action = np.zeros([env.robots_num, 2])
    # next_obs, _ = env.reset()

    # 只收集了一个机器人的rewards
    with torch.no_grad():
        for times in range(3):
            print('times:', times)
            r = []

            while True:
                for i, rob in enumerate(env.robots):
                    vel = rob.goto(rob.target_pos)
                    action[i] = vel

                next_obs, reward, te, tr, info_ = env.step(action)
                r.append(reward)
                done = env.check_done(te=te, tr=tr, lim=0)
                # print('reward:', reward)

                if np.array(done).all():
                    rewards.append(np.sum(np.array(r), axis=0))
                    break

        print('rewards:', rewards)


def main():
    render = True
    # env = gym.make('MyEnv-v0', render=render, urdf_path=urdf_path)
    env = MyEnv(render=render, urdf_path=urdf_path)

    p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=0, cameraPitch=-89.9,
                                 cameraTargetPosition=[0, 0, 0])

    robot_nums = 1
    lim = 5

    print('robot_nums:', robot_nums)
    for i in range(robot_nums):
        env.add_random_robot(lim=lim)

    agent = PPO(env, test_env=env)
    agent.evaluate(debug=True)


if __name__ == '__main__':
    main()
    # test()
