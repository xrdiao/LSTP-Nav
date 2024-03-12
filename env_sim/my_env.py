import sys
import os
import gym
import pybullet as p
import pybullet_data
import numpy as np
from gym import spaces
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from env_sim.robot import Robot
from env_sim.argument import *


class MyEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, scene_name: str = "plane_static_obstacle-A", render: bool = False, evaluate: bool = False):

        self._physics_client_id = p.connect(p.GUI if render else p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setGravity(0., 0., -9.8, physicsClientId=self._physics_client_id)
        p.setRealTimeSimulation(0)  # 1表示随着真实时间仿真，0表示要用p.step()进行步进
        p.setTimeStep(1. / 240.)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", physicsClientId=self._physics_client_id)

        self.time_limit = None
        self.time_step = None
        self.robots = []  # a Robot instance representing the robot
        self.humans = []  # a list of Human instances, representing all humans in the environment
        self.global_time = 0
        self.step_counter = 0
        self.step_num = 0
        self.collision_num = 0

        self.robots_num = 0
        self.TARGET_VELOCITY = TARGET_VELOCITY
        self.LASER_NUM = LASER_NUM
        self.LASER_LENGTH = LASER_LENGTH
        self.MAX_DISTANCE = MAX_DISTANCE

        # # 读入各项参数
        # for file in os.listdir("./data/config"):
        #     param_path = os.path.join("./robot/config/", file)
        #     param_dict = load(open(param_path, "r", encoding="utf-8"), Loader=Loader)
        #     for key, value in param_dict.items():
        #         setattr(self, key, value)

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

    def add_robot(self, state, _goal=None, urdf_path="env_sim/utils/data/turtlebot.urdf"):
        if _goal is None:
            _goal = [0, 0]
        assert len(state) == 7, 'state must have 7 elements, [pos, ori]'
        _robot = Robot(base_pos=state[:3], base_ori=state[3:], client_id=self._physics_client_id, urdf_path=urdf_path)
        _robot.set_target_pos(_goal)
        _robot.cur_dis = self.__distance(np.array(state[:2]), np.array(_goal[:2]))
        self.robots.append(_robot)
        self.robots_num += 1
        self.init_state.append(state)
        self.init_goal.append(_goal)

    def checkCollision(self, robot_id, debug=False):
        if p.getContactPoints(bodyA=robot_id, linkIndexA=-1, physicsClientId=self._physics_client_id):
            if debug:
                print("collsion happen!")
            return True
        return False

    def __reward_func(self, distances: list):
        rewards = []
        _done = False
        for i, _rob in enumerate(self.robots):
            if distances[i] < 1e-2:
                rg = ARRIVAL_REWARD
                _done = True
            else:
                rg = DISTANCE_REWARD_WEIGHT * (_rob.cur_dis.item() - distances[i])

            if self.checkCollision(_rob.robot):
                rc = COLLISION_REWARD
                _done = True
            else:
                rc = 0
            rewards.append(rg + rc)

        return rewards, _done

    def __distance(self, v1, v2):
        v1, v2 = np.array(v1), np.array(v2)
        return np.linalg.norm(v1 - v2)

    def step(self, actions):
        assert self.robots is not None, 'no robots loaded'
        assert self.robots_num == len(actions), 'incorrect action'

        for i, action in enumerate(actions):
            self.robots[i].apply_action(action)
        p.stepSimulation(physicsClientId=self._physics_client_id)
        self.step_num += 1
        _states = []
        for i in range(self.robots_num):
            _states.append(self.robots[i].get_observation())

        laser_states = [_state[0] for _state in _states]
        distances = [_state[-1] for _state in _states]
        _reward, _done = self.__reward_func(distances)

        #  更新上一个时刻的距目标点距离
        for i, _rob in enumerate(self.robots):
            _rob.cur_dis = distances[i]
        _info = {"distance": distances, "collision_num": self.collision_num}
        return np.array(laser_states), _reward, _done, _info

    def reset(self, phase='train', test_case=None, init_state=None, init_goal=None, urdf_path="env_sim/utils/data/turtlebot.urdf"):
        """
            what you need do here:
            - reset scene items
            - reload robot
        """
        assert self.robots is not None, 'no robots loaded'

        if init_state is not None or init_goal is not None:
            self.init_state = init_state
            self.init_goal = init_goal

        p.resetSimulation(physicsClientId=self._physics_client_id)
        p.setGravity(0., 0., -9.8, physicsClientId=self._physics_client_id)
        p.setRealTimeSimulation(0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", physicsClientId=self._physics_client_id)

        self.step_num = 0
        self.collision_num = 0
        self.robots = []

        for i in range(self.robots_num):
            _pos, _ori = self.init_state[i][:3], self.init_state[i][3:]
            _robot = Robot(base_pos=_pos, base_ori=_ori, client_id=self._physics_client_id, urdf_path=urdf_path)
            _robot.set_target_pos(self.init_goal[i])
            _robot.cur_dis = self.__distance(_pos[:2], self.init_goal[i])
            self.robots.append(_robot)

        _states = []
        for i in range(self.robots_num):
            _states.append(self.robots[i].get_observation())
        laser_states = [_state[0] for _state in _states]
        return laser_states

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def close(self):
        if self._physics_client_id >= 0:
            p.disconnect()
        self._physics_client_id = -1


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
    p.loadURDF("cube.urdf", [length, -1, 0.5])
    p.loadURDF("cube.urdf", [length, width, 0.5])
    p.loadURDF("cube.urdf", [-1, width, 0.5])
    p.loadURDF("cube.urdf", [-1, -1, 0.5])


if __name__ == "__main__":
    env = MyEnv(render=True)
    goal = [1, 0]
    env.add_robot([0, 0, 0, 0, 0, 0, 1.5], goal, 'utils/data/turtlebot.urdf')
    env.add_robot([0, 1.5, 0, 0, 0, 0, 1], goal, 'utils/data/turtlebot.urdf')
    p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=0, cameraPitch=-89.9,
                                 cameraTargetPosition=[0, 0, 0])
    # createBoundaries(10, 10)
    while True:
        velocity = []
        for rob in env.robots:
            velocity.append(rob.goto(goal))
        states, reward, done, info = env.step(velocity)
        print(reward)
        time.sleep(1 / 240)

        if done:
            env.reset(urdf_path='utils/data/turtlebot.urdf')
