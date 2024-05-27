import os
import math
from env_sim.my_env import MyEnv
import rvo.math as rvo_math

from env_sim.argument import *
from rvo.vector import Vector2
from rvo.simulator import Simulator

base_path = os.path.dirname(os.path.abspath(__file__))
urdf_path = base_path + '/env_sim/utils/data/turtlebot.urdf'
import pybullet as p

import numpy as np


def reached_goal(simulator, goals):
    """
    Check if all agents have reached their goals.
    """
    for i in range(simulator.num_agents):
        if rvo_math.abs_sq(simulator.agents_[i].position_ - goals[i]) > simulator.agents_[i].radius_ * \
                simulator.agents_[i].radius_:
            return False

    return True


def set_preferred_velocities(simulator, goals):
    for i in range(simulator.num_agents):
        goal_vector = goals[i] - simulator.agents_[i].position_

        if rvo_math.abs_sq(goal_vector) > 1.0:
            goal_vector = rvo_math.normalize(goal_vector)

        simulator.set_agent_pref_velocity(i, goal_vector)


def obs2simulator(position: list):
    x, y = position[0], position[1]
    size = 1
    left = x - size
    right = x + size
    top = y + size
    bottom = y - size

    obstacle = [Vector2(right, bottom), Vector2(right, top), Vector2(left, top), Vector2(left, bottom)]
    return obstacle


def test():
    # 目的是搭建一个沟通Myenv和simulator的桥梁
    render = True

    env = MyEnv(render=render, urdf_path=urdf_path)
    simulator = Simulator()
    p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=0, cameraPitch=-89.9,
                                 cameraTargetPosition=[0, 0, 0])

    simulator.set_time_step(env.time_step)

    # obstacles = [[0,0],[0, 3], [0, -3], [3, 0], [-3, 0]]
    # for obstacle in obstacles:
    #     env.place_cube(obstacle)
    #     obs = obs2simulator(obstacle)
    #     simulator.add_obstacle(obs)
    # simulator.process_obstacles()

    robot_nums = 1
    lim = 5

    # for i in range(robot_nums):
    #     env.add_random_robot(lim=lim)

    for i in range(robot_nums):
        goal = [i, i]
        ori = p.getQuaternionFromEuler([0, 0, np.pi / 2], env.physics_client_id)
        state = [i, 0, 0.01] + list(ori)
        env.add_robot(state, goal)

    env.show_goal_point()

    goals = []
    simulator.set_agent_defaults(15.0, 10, 10.0, 10.0, ROBOT_WIDTH + 0.1, MAX_SPEED, Vector2(0.0, 0.0))

    for rob in env.robots:
        simulator.add_agent(Vector2(rob.init_pos[0], rob.init_pos[1]))
        goals.append(Vector2(rob.target_pos[0], rob.target_pos[1]))

    while True:
        # while True:

        set_preferred_velocities(simulator, goals)

        # get action from 2D simulator
        actions = []
        simulator.kd_tree_.build_agent_tree()

        # print('-------------------------------------------------')
        for agentNo in range(simulator.num_agents):
            simulator.agents_[agentNo].compute_neighbors()
            simulator.agents_[agentNo].compute_new_velocity()

            action = simulator.agents_[agentNo].new_velocity_
            angle = env.robots[agentNo].follow_vector_angle([action.x, action.y])

            a = [abs(action), angle]

            actions.append(a)
            # print(action, cur_vel, a, (action @ cur_vel) / (abs(action) * abs(cur_vel)))

        # update pybullet
        next_obs, reward, te, tr, info = env.step(actions)
        done = env.check_done(te=te, tr=tr, lim=lim)
        print(done)
        for i, d in enumerate(done):
            if d:
                rob = env.robots[i]
                goals[i] = Vector2(rob.target_pos[0], rob.target_pos[1])


        # update simulator
        for agentNo, agent in enumerate(env.robots):
            obs_dict = agent.get_vel_and_pos()
            vel, pos = obs_dict['vel'], obs_dict['pos']
            angle = agent.get_forward_vector()[:2]

            simulator.agents_[agentNo].velocity_ = Vector2(abs(vel[0]) * angle[0], abs(vel[0]) * angle[1])
            simulator.agents_[agentNo].position_ = Vector2(pos[0], pos[1])


if __name__ == '__main__':
    test()
