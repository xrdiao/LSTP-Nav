import sys
import os
import gymnasium as gym
import pybullet as p
import pybullet_data
import numpy as np
from gymnasium import spaces
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from env_sim.robot import Robot
from env_sim.argument import *


class MyEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, scene_name: str = "plane_static_obstacle-A", render: bool = False, evaluate: bool = False):
        self.time_step = 1. / 240.

        self._physics_client_id = p.connect(p.GUI if render else p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setGravity(0., 0., -9.8, physicsClientId=self._physics_client_id)
        p.setRealTimeSimulation(0)  # 1表示随着真实时间仿真，0表示要用p.step()进行步进
        p.setTimeStep(self.time_step)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", physicsClientId=self._physics_client_id)

        self.time_limit = None
        self.robots = []  # a Robot instance representing the robot
        self.global_time = 0
        self.step_counter = 0
        self.step_num = 0
        self.collision_num = 0

        self.robots_num = 0
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

    def add_robot(self, state, goal=None, urdf_path='utils/data/turtlebot.urdf'):
        if goal is None:
            goal = [0, 0]
        assert len(state) == 7, 'state must have 7 elements, [pos, ori]'
        for _state in self.init_state:
            assert self.__distance(state[:2],
                                   _state[:2]) > 2 * ROBOT_WIDTH, 'the robot with id={} is incorrect'.format(
                self.robots_num)

        robot = Robot(base_pos=state[:3], base_ori=state[3:], client_id=self._physics_client_id, urdf_path=urdf_path)
        robot.set_target_pos(goal)
        robot.cur_dis = self.__distance(np.array(state[:2]), np.array(goal[:2]))
        self.robots.append(robot)
        self.robots_num += 1
        self.init_state.append(state)
        self.init_goal.append(goal)

    def add_random_robot(self, lim=1, updf_path='utils/data/turtlebot.urdf'):
        '''
        Add a random robot to the environment
        :param lim: The range limit of the robot positions
        '''
        done = False
        while not done:
            pos = np.random.uniform(low=-lim, high=lim, size=2)
            pos = pos.tolist() + [0.01]
            angle = np.random.uniform(low=-np.pi, high=np.pi)
            ori = list(p.getQuaternionFromEuler([0, 0, angle], self._physics_client_id))

            # 判断随机放置的机器人是否会与已有机器人发生碰撞
            for state in self.init_state:
                if self.__distance(state[:2], pos[:2]) > 3 * ROBOT_WIDTH:
                    done = True
            if self.robots_num == 0:
                done = True

        done = False
        while not done:
            goal = np.random.uniform(low=-lim, high=lim, size=2)
            for state in self.init_goal:
                if self.__distance(state[:2], goal[:2]) > 2 * ROBOT_WIDTH:
                    done = True
            if self.robots_num == 0:
                done = True

        robot = Robot(base_pos=pos, base_ori=ori, client_id=self._physics_client_id, urdf_path=updf_path)
        robot.set_target_pos(goal)
        robot.cur_dis = self.__distance(np.array(pos[:2]), np.array(goal[:2]))

        state = pos + ori
        self.robots.append(robot)
        self.robots_num += 1
        self.init_state.append(state)
        self.init_goal.append(goal)

    def checkCollision(self, robot_id, debug=False):
        # 也可以用距离判断
        if p.getContactPoints(bodyA=robot_id, linkIndexA=-1, physicsClientId=self._physics_client_id):
            if debug:
                print("robot with id={} collides!".format(robot_id))
            return True
        return False

    def __reward_func(self, distances: list):
        rewards = []
        # terminated表示智能体是否到达终点，truncated表示智能体因时间或物理碰撞等因素停止运行
        te, tr = np.zeros_like(self.robots, dtype=bool), np.zeros_like(self.robots, dtype=bool)
        for i, rob in enumerate(self.robots):
            if rob.is_reachable():
                rg = ARRIVAL_REWARD
                te[i] = True
            else:
                rg = DISTANCE_REWARD_WEIGHT * (rob.cur_dis.item() - distances[i])

            if self.checkCollision(rob.robot):
                rc = COLLISION_REWARD
                self.collision_num += 1
                tr[i] = True
            else:
                rc = 0

            rewards.append(rg + rc)

        return rewards, te, tr

    def __distance(self, v1, v2):
        v1, v2 = np.array(v1), np.array(v2)
        return np.linalg.norm(v1 - v2)

    def step(self, actions):
        assert self.robots is not None, 'no robots loaded'
        assert self.robots_num == len(actions), 'incorrect number of the actions'

        self.global_time += self.time_step

        # 判断是否到达终点
        for i, action in enumerate(actions):
            if self.robots[i].is_reachable():
                self.robots[i].apply_action([0, 0])
            else:
                self.robots[i].apply_action(action)

        p.stepSimulation(physicsClientId=self._physics_client_id)
        self.step_num += 1

        # 收集机器人观测量，计算奖励
        distances = []
        current_state = []
        for i in range(self.robots_num):
            obs = self.robots[i].get_observation()
            laser = obs['laser']
            distance = obs['distance']
            angle = obs['angle']
            current_state.append(laser + distance + angle)
            distances.append(distance[0])
        reward, te, tr = self.__reward_func(distances)

        #  更新上一个时刻的距目标点距离
        for i, rob in enumerate(self.robots):
            rob.cur_dis = distances[i]
        _info = {"distance": distances, "collision_num": self.collision_num}
        return np.array(current_state), reward, te, tr, _info

    def reset(self, phase='train', test_case=None, urdf_path='utils/data/turtlebot.urdf', tr: list = None,
              lim=0):
        """
            - reset scene items
            - reload robot
        """

        assert self.robots is not None, 'no robots loaded'
        local_reset = False
        if tr is not None:
            reset_id = np.where(tr == 1)[0]
            if len(reset_id) > 0:
                local_reset = True

        if local_reset:
            assert len(tr) == self.robots_num
            for idx in reset_id:
                self.robots[idx].reset()

        else:
            # reset scene
            p.resetSimulation(physicsClientId=self._physics_client_id)
            p.setGravity(0., 0., -9.8, physicsClientId=self._physics_client_id)
            p.setRealTimeSimulation(0)
            p.setTimeStep(self.time_step)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.loadURDF("plane.urdf", physicsClientId=self._physics_client_id)

            robot_num = self.robots_num

            self.step_num = 0
            self.collision_num = 0
            self.robots_num = 0

            init_state = self.init_state.copy()
            init_goal = self.init_goal.copy()

            self.init_state.clear()
            self.init_goal.clear()
            self.robots.clear()

            # reload robot
            if lim:
                for i in range(robot_num):
                    self.add_random_robot(lim)
            else:
                for i in range(robot_num):
                    pos, ori = init_state[i][:3], init_state[i][3:]
                    self.add_robot(pos + ori, init_goal[i])

            current_state = []
            for i in range(robot_num):
                obs = self.robots[i].get_observation()
                laser = obs['laser']
                distance = obs['distance']
                angle = obs['angle']
                current_state.append(laser + distance + angle)

            self.show_goal_point()
            return np.array(current_state), ''

    def show_goal_point(self):
        p.addUserDebugPoints([[_g[0], _g[1], 0.5] for _g in self.init_goal], [[1, 0, 0]] * self.robots_num, 10)
        for idx in range(self.robots_num):
            pos = [self.init_goal[idx][0], self.init_goal[idx][1], 0.5]
            p.addUserDebugText('{}'.format(idx), pos, [1, 0, 0], 1)

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def close(self):
        if self._physics_client_id >= 0:
            p.disconnect()
        self._physics_client_id = -1

    def place_cube(self, x, y, size=0.5):
        p.loadURDF("cube.urdf", [x, y, size])

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
    robot_nums = 3
    lim = 1

    for i in range(robot_nums):
        goal = [0, 1 + i]

        yaw = np.pi * i / robot_nums
        ori = p.getQuaternionFromEuler([0, 0, yaw], env.physics_client_id)
        state = [i, 0, 0.01] + list(ori)
        # env.add_robot(state, goal, 'utils/data/turtlebot.urdf')
        env.add_random_robot(lim=lim)
    env.show_goal_point()

    p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=0, cameraPitch=-89.9,
                                 cameraTargetPosition=[0, 0, 0])
    # createBoundaries(10, 10)

    velocity = np.zeros([env.robots_num, 2])
    while True:
        for i, rob in enumerate(env.robots):
            vel = rob.goto(rob.target_pos)
            velocity[i] = vel
        states, reward, te, tr, info = env.step(velocity)
        # print(reward, te, tr)
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
            env.reset(urdf_path='utils/data/turtlebot.urdf', lim=lim, tr=tr)


if __name__ == '__main__':
    main()
