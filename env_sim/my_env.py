import sys
import os
from typing import Optional, Dict, Any

import gymnasium as gym
import pybullet as p
import pybullet_data
import numpy as np
from gymnasium import spaces
import time

# path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# path = sys.path[0] + '\\env_sim'
# sys.path.append(path)

from env_sim.robot import Robot
from env_sim.argument import *

import warnings

warnings.filterwarnings("ignore")


class MyEnv(gym.Env):
    def __init__(self, scene_name: str = "plane_static_obstacle-A", render: bool = False, evaluate: bool = False,
                 urdf_path: Optional[str] = 'utils/data/turtlebot.urdf'):
        self.time_step = 1. / 240.
        self.max_simulate_steps = 10000
        self.simulate_steps = 0

        self.random_mode = render
        self._physics_client_id = p.connect(p.GUI if self.random_mode else p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self._physics_client_id)
        p.setGravity(0., 0., -9.8, physicsClientId=self._physics_client_id)
        p.setRealTimeSimulation(0, physicsClientId=self._physics_client_id)  # 1表示随着真实时间仿真，0表示要用p.step()进行步进
        p.setTimeStep(self.time_step, physicsClientId=self._physics_client_id)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", physicsClientId=self._physics_client_id)

        self.time_limit = None
        self.robots = []  # a Robot instance representing the robot
        self.obstacles = []

        self.global_time = 0
        self.step_counter = 0
        self.collision_num = 0

        self.robots_num = 0
        self.urdf_path = urdf_path
        self.TARGET_VELOCITY = TARGET_VELOCITY
        self.LASER_NUM = LASER_NUM
        self.LASER_LENGTH = LASER_LENGTH
        self.MAX_DISTANCE = MAX_DISTANCE

        # 动作空间: 左轮速度， 右轮速度
        self.action_space = spaces.Box(
            low=np.array([-self.TARGET_VELOCITY, -self.TARGET_VELOCITY], dtype=np.float32),
            high=np.array([self.TARGET_VELOCITY, self.TARGET_VELOCITY], dtype=np.float32),
        )
        # 状态空间: laser1, ..., 5,   distance, alpha
        self.observation_space = spaces.Box(
            low=np.array([0.] * self.LASER_NUM + [0., 0.], dtype=np.float32),
            high=np.array([self.LASER_LENGTH + 1] * self.LASER_NUM + [self.MAX_DISTANCE, np.pi], dtype=np.float32),
        )

        self.init_state = []
        self.init_goal = []

    def add_robot(self, state, goal=None):
        if goal is None:
            goal = [0, 0]
        assert len(state) == 7, 'state must have 7 elements, [pos, ori]'
        for _state in self.init_state:
            assert self.__distance(state[:2],
                                   _state[:2]) > 2 * ROBOT_WIDTH, 'the robot with id={} is incorrect'.format(
                self.robots_num)

        robot = Robot(base_pos=state[:3], base_ori=state[3:], client_id=self._physics_client_id,
                      urdf_path=self.urdf_path)
        robot.set_target_pos(goal)
        robot.cur_dis = self.__distance(np.array(state[:2]), np.array(goal[:2]))
        robot.cur_pos = state[:2]

        self.robots.append(robot)
        self.robots_num += 1
        self.init_state.append(state)
        self.init_goal.append(goal)

    def check_done(self, te, tr, lim=5):
        assert len(te) == len(tr), "the length of te or tr are incorrect"

        # 到达最大仿真步数时，重置环境
        if self.simulate_steps >= self.max_simulate_steps:
            te = [True] * len(te)
            self.reset(lim=lim, tr=tr, te=te)
            done = [True] * len(te)
        else:
            # 机器人都到达目标点时te为true
            done_te = True
            for i in range(len(te)):
                if not te[i]:
                    done_te = False
                    break

            # 有机器人发生碰撞时tr为true
            done_tr = False
            for i in range(len(tr)):
                if tr[i]:
                    done_tr = True
                    break

            if done_te or done_tr:
                self.reset(lim=lim, tr=tr, te=te)
            done = [i or j for i, j in zip(te, tr)]
        return done

    def random_point(self, lim=5, seed=0):
        """
        :param lim: range of the point
        :param seed: 0 - init pos, 1- init goal
        :return:
        """
        point = None
        done = False
        while not done:
            point = np.random.uniform(low=-lim, high=lim, size=2).tolist() + [0.01]
            done = True

            for obs in self.obstacles:
                done = True and done if self.__distance(obs[:2], point[:2]) > (
                        np.sqrt(2) + ROBOT_WIDTH + 0.1) else False and done

            for rob in self.robots:
                done = True and done if self.__distance(rob.cur_pos[:2], point[:2]) > (
                        2 * ROBOT_WIDTH + 0.1) else False and done

            if seed == 1:
                for state in self.init_goal:
                    done = True and done if self.__distance(state[:2],
                                                            point[:2]) > 2.5 * ROBOT_WIDTH else False and done
            elif seed == 0:
                for state in self.init_state:
                    done = True and done if self.__distance(state[:2],
                                                            point[:2]) > 2.5 * ROBOT_WIDTH else False and done
            if self.robots_num == 0:
                break
        return point

    def add_random_robot(self, lim=5):
        """
        生成一个随机机器人
        :param lim: The range limit of the robot positions
        """
        pos = self.random_point(lim=lim, seed=0)
        goal = self.random_point(lim=lim, seed=1)[:2]

        angle = np.random.uniform(low=-np.pi, high=np.pi)
        ori = list(p.getQuaternionFromEuler([0, 0, angle], self._physics_client_id))

        robot = Robot(base_pos=pos, base_ori=ori, client_id=self._physics_client_id, urdf_path=self.urdf_path)
        robot.set_target_pos(goal)
        robot.cur_dis = self.__distance(np.array(pos[:2]), np.array(goal[:2]))
        robot.cur_pos = pos

        state = pos + ori
        self.robots.append(robot)
        self.robots_num += 1
        self.init_state.append(state)
        self.init_goal.append(goal)

    def checkCollision(self, robot_id, debug=True):
        # 也可以用距离判断
        # for i, robot in enumerate(self.robots):
        # if i != robot_id - 1 - len(self.obstacles):
        # if np.linalg.norm(np.array(self.robots[i].cur_pos) - np.array(robot.cur_pos)) <= 2 * ROBOT_WIDTH:
        #     self.collision_num += 1
        #     if debug:
        #         print(
        #             "robot with id={} collides!, state:{}, physical id:{}, collision number:{}".format(robot_id,
        #                                                                                                self.robots[
        #                                                                                                    robot_id - 1 - len(
        #                                                                                                        self.obstacles)].cur_pos,
        #                                                                                                self._physics_client_id,
        #                                                                                                self.collision_num))
        #     return True
        #     for obs in self.obstacles:
        #         if np.linalg.norm(np.array(robot.cur_pos) - np.array(obs)) < ROBOT_WIDTH + np.sqrt(2):
        #             self.collision_num += 1
        #             if debug:
        #                 print(
        #                     "robot with id={} collides!, state:{}, physical id:{}, collision number:{}".format(
        #                         robot_id,
        #                         self.robots[
        #                             robot_id - 1 - len(
        #                                 self.obstacles)].cur_pos,
        #                         self._physics_client_id,
        #                         self.collision_num))
        #             return True
        # return False
        if p.getContactPoints(bodyA=robot_id, linkIndexA=-1, physicsClientId=self._physics_client_id):
            self.collision_num += 1
            if debug:
                print("robot with id={} collides!, state:{}, physical id:{}, collision number:{}".format(robot_id,
                                                                                                         self.robots[
                                                                                                             robot_id - 1 - len(
                                                                                                                 self.obstacles)].cur_pos,
                                                                                                         self._physics_client_id,
                                                                                                         self.collision_num))
            return True
        return False

    def __reward_func(self, distances: list):
        rewards = []
        # terminated表示智能体是否到达终点，truncated表示智能体因时间或物理碰撞等因素停止运行
        te, tr = np.zeros_like(self.robots, dtype=bool), np.zeros_like(self.robots, dtype=bool)

        for i, rob in enumerate(self.robots):
            # 到达目标点奖励
            if rob.is_reachable():
                te[i] = True
                if not rob.reach_goal:
                    rg = ARRIVAL_REWARD
                    rob.reach_goal = True
                else:  # 到达目标点后不重复获得奖励
                    rg = 0
            # 距离目标点距离奖励
            else:
                rg = DISTANCE_REWARD_WEIGHT * (rob.cur_dis.item() - distances[i])

                # 到达目标点后离开目标点需要扣除已获得的奖励
                if rob.reach_goal:
                    rob.reach_goal = False
                    rg = -ARRIVAL_REWARD + rg

            # 碰撞时惩罚
            if self.checkCollision(rob.robot):
                rc = COLLISION_REWARD
                tr[i] = True
            else:
                rc = 0

            # 过大角速度时惩罚
            rw = ANGULAR_VELOCITY_PENALTY * abs(rob.del_angle) if abs(rob.del_angle) > 0.5 else 0
            rv = VELOCITY_PENALTY * rob.del_vel if abs(rob.del_vel) > 1000 else 0
            rewards.append(rg + rc + rw + rv)

        return rewards, te, tr

    def __distance(self, v1, v2):
        v1, v2 = np.array(v1), np.array(v2)
        return np.linalg.norm(v1 - v2)

    def step(self, actions, FPS=1):
        """
        :param actions: [velocity, angular_vel]
        :param FPS: 一个动作的持续帧率
        :return:
        """
        assert self.robots is not None, 'no robots loaded'
        assert self.robots_num == len(actions), 'incorrect number of the actions'

        self.global_time += self.time_step
        self.simulate_steps += 1

        # 更新 t 时刻机器人距终点的距离，计算 t+1 和 t 时刻间的变化量
        for i, rob in enumerate(self.robots):
            rob.cur_dis = self.__distance(rob.cur_pos[:2], rob.target_pos[:2])
            rob.del_vel = abs(rob.cur_action[0] - actions[i][0])
            rob.del_angle = abs(rob.cur_action[1] - actions[i][1])

        for i, action in enumerate(actions):
            self.robots[i].apply_action(action)

        # 为了让一个动作的作用更加明显，让同一个动作执行多次
        for _ in range(FPS):
            p.stepSimulation(physicsClientId=self._physics_client_id)

        # 收集 t+1 时刻机器人的观测量，计算奖励，
        distances = []
        current_state = []
        for i in range(self.robots_num):
            obs = self.robots[i].get_observation()  # 获取 t+1 时刻的观测值，并且更新机器人的本地记录
            laser = obs['laser']
            distance = obs['distance']
            angle = obs['angle']
            current_state.append(laser + distance + angle)
            distances.append(distance[0])
        reward, te, tr = self.__reward_func(distances)

        info = {"distance": distances, "collision_num": self.collision_num}
        return np.array(current_state), reward, te, tr, info

    def reset_robot(self, idx: int, lim: int = 5):
        """
        Reset the position of the robot
        """
        if lim:
            pos = self.random_point(lim=lim, seed=0)
            angle = np.random.uniform(low=-np.pi, high=np.pi)
            ori = list(p.getQuaternionFromEuler([0, 0, angle], self._physics_client_id))
        else:
            pos, ori = self.init_state[idx][:3], self.init_state[idx][3:]

        p.resetBasePositionAndOrientation(self.robots[idx].robot, pos, ori, self._physics_client_id)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None, phase='train', test_case=None,
              tr: list = None, te: list = None, lim=5):
        """
            - 所有机器人到达目标点后重设场景
            - 部分机器人发生碰撞后重设机器人
            :param lim: 机器人随机生成的范围限制，0意味着按照初始值生成
        """
        assert self.robots is not None, 'no robots loaded'

        te_done = True
        if te is not None:
            for d in te:
                te_done = d and te_done

        local_reset = False
        if tr is not None:
            reset_id = np.where(tr == 1)[0]
            if len(reset_id) > 0:
                local_reset = True

        current_state = []

        if local_reset:
            assert len(tr) == self.robots_num
            for idx in reset_id:
                self.reset_robot(idx, lim=lim)

            for i in range(self.robots_num):
                obs = self.robots[i].get_observation()
                laser = obs['laser']
                distance = obs['distance']
                angle = obs['angle']
                current_state.append(laser + distance + angle)

        elif te_done:
            # reset scene
            p.resetSimulation(physicsClientId=self._physics_client_id)
            p.setGravity(0., 0., -9.8, physicsClientId=self._physics_client_id)
            p.setRealTimeSimulation(0, physicsClientId=self._physics_client_id)
            p.setTimeStep(self.time_step, physicsClientId=self._physics_client_id)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.loadURDF("plane.urdf", physicsClientId=self._physics_client_id)

            robot_num = self.robots_num

            self.collision_num = 0
            self.robots_num = 0
            self.simulate_steps = 0
            self.global_time = 0

            init_state = self.init_state.copy()
            init_goal = self.init_goal.copy()

            self.init_state.clear()
            self.init_goal.clear()
            self.robots.clear()

            for obs in self.obstacles:
                self.place_cube(obs)

            # reload robot
            if lim:
                for i in range(robot_num):
                    self.add_random_robot(lim)
            else:
                for i in range(robot_num):
                    self.add_robot(init_state[i], init_goal[i])

            for i in range(robot_num):
                obs = self.robots[i].get_observation()
                laser = obs['laser']
                distance = obs['distance']
                angle = obs['angle']
                current_state.append(laser + distance + angle)

            self.show_goal_point()
        return np.array(current_state), dict()

    def show_goal_point(self):
        p.addUserDebugPoints([[_g[0], _g[1], 0.5] for _g in self.init_goal], [[1, 0, 0]] * self.robots_num, 10,
                             physicsClientId=self._physics_client_id)
        for idx in range(self.robots_num):
            pos = [self.init_goal[idx][0], self.init_goal[idx][1], 0.5]
            p.addUserDebugText('{}'.format(idx), pos, [1, 0, 0], 1, physicsClientId=self._physics_client_id)

    def render(self, mode='human'):
        pass

    # def seed(self, seed=None):
    #     self.np_random, seed = gym.utils.seeding.np_random(seed)
    #     return [seed]

    def close(self):
        if self._physics_client_id >= 0:
            p.disconnect(physicsClientId=self._physics_client_id)
        self._physics_client_id = -1

    def place_cube(self, position: list):
        x, y = position[0], position[1]
        assert self.robots_num == 0, 'must place obstacles first'
        self.obstacles.append([x, y])
        p.loadURDF("cube.urdf", [x, y, 0.5], physicsClientId=self._physics_client_id, useFixedBase=1,
                   useMaximalCoordinates=1)

    @property
    def physics_client_id(self):
        return self._physics_client_id


def createBoundaries(length, width):
    """
        create rectangular boundaries with length and width

        Args:

        length: integer

        width: integer
    """
    for i in range(length):
        p.loadURDF("cube.urdf", [i, -1, 0.5])
        p.loadURDF("cube.urdf", [i, width, 0.5])
    for i in range(width):
        p.loadURDF("cube.urdf", [-1, i, 0.5])
        p.loadURDF("cube.urdf", [length, i, 0.5])
        if abs(i - length / 2) < 2:
            continue
        p.loadURDF("cube.urdf", [length / 2, i, 0.5])


def main():
    env = MyEnv(render=True)

    robot_nums = 2
    lim = 5

    for i in range(robot_nums):
        goal = [i, 1 + i]

        yaw = np.pi * robot_nums
        ori = p.getQuaternionFromEuler([0, 0, yaw], env.physics_client_id)
        state = [i, 0, 0.01] + list(ori)
        env.add_robot(state, goal)
        # env.add_random_robot(lim=lim)
    env.show_goal_point()

    p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=0, cameraPitch=-89.9,
                                 cameraTargetPosition=[0, 0, 0])
    # createBoundaries(10, 10)

    env.reset()

    velocity = np.zeros([env.robots_num, 2])
    while True:
        for i, rob in enumerate(env.robots):
            vel = rob.goto(rob.target_pos)
            velocity[i] = vel
        states, reward, te, tr, info = env.step(velocity)
        print(states)
        # time.sleep(1 / 240)

        done_te = True
        for i in range(env.robots_num):
            done_te = done_te and te[i]

        done_tr = False
        for i in range(env.robots_num):
            if tr[i]:
                done_tr = True
                break

        if done_te or done_tr:
            a, _ = env.reset(lim=lim, tr=tr, te=te)
            # print('reset state', a)


if __name__ == '__main__':
    # env_ = gym.make('MyEnv-v0')
    main()
    # print(1)
