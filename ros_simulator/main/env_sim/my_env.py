import subprocess
import os
import numpy as np

import time
import rospy
from gazebo_msgs.msg import ModelState
from std_srvs.srv import Empty

from env_sim.robot import Robot
from env_sim.argument import *
from scipy.spatial.transform import Rotation as R

world_path = os.path.join(os.path.dirname(__file__), "launch", 'empty_env.launch')


class MyEnv:
    def __init__(self):
        self.max_simulate_steps = 100000
        self.simulate_steps = 0
        self.TIME_DELTA = 0.1

        subprocess.Popen(["roscore"])
        rospy.init_node("gazebo", anonymous=True)
        subprocess.Popen(['roslaunch', world_path])
        rospy.wait_for_service("/gazebo/spawn_urdf_model")

        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)

        self.set_state = rospy.Publisher(
            "gazebo/set_model_state", ModelState, queue_size=10
        )

        self.time_limit = None
        self.robots = []  # a Robot instance representing the robot
        self.obstacles = []

        self.global_time = 0
        self.step_counter = 0
        self.collision_num = 0

        self.robots_num = 0
        self.TARGET_VELOCITY = TARGET_VELOCITY
        self.LASER_NUM = LASER_NUM
        self.LASER_LENGTH = LASER_LENGTH

        # 动作空间: 左轮速度， 右轮速度
        self.action_space_size = 2
        # 状态空间: laser1, ..., 5,   distance, alpha
        self.observation_space_size = LASER_NUM + 4

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

        robot = Robot(base_pos=state[:3], base_ori=state[3:], robot_idx=self.robots_num)
        robot.set_target_pos(goal)
        robot.cur_dis = self.__distance(np.array(state[:2]), np.array(goal[:2]))
        robot.cur_pos = state[:2]

        self.robots.append(robot)
        self.robots_num += 1
        self.init_state.append(state)
        self.init_goal.append(goal)
        time.sleep(1)

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
        ori = list(R.from_euler('z', angle).as_quat())

        robot = Robot(base_pos=pos, base_ori=ori, robot_idx=self.robots_num)
        robot.set_target_pos(goal)
        robot.cur_dis = self.__distance(np.array(pos[:2]), np.array(goal[:2]))
        robot.cur_pos = pos

        state = pos + ori
        self.robots.append(robot)
        self.robots_num += 1
        self.init_state.append(state)
        self.init_goal.append(goal)
        time.sleep(1)

    def checkCollision(self, robot_id):
        # 也可以用距离判断
        for i, robot in enumerate(self.robots):
            if i != robot_id:
                if np.linalg.norm(np.array(self.robots[robot_id].cur_pos) - np.array(robot.cur_pos)) <= ROBOT_WIDTH:
                    self.collision_num += 1
                    print('collision')
                    return True

        for obs in self.obstacles:
            if np.linalg.norm(np.array(self.robots[robot_id].cur_pos) - np.array(obs)) < ROBOT_WIDTH + np.sqrt(2):
                self.collision_num += 1
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
            if self.checkCollision(i):
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

    def step(self, actions):
        """
        :param actions: [velocity, angular_vel]
        :return:
        """
        assert self.robots is not None, 'no robots loaded'
        assert self.robots_num == len(actions), 'incorrect number of the actions'

        self.simulate_steps += 1

        # 更新 t 时刻机器人距终点的距离，计算 t+1 和 t 时刻间的变化量
        for i, rob in enumerate(self.robots):
            rob.cur_dis = self.__distance(rob.cur_pos[:2], rob.target_pos[:2])
            rob.del_vel = abs(rob.cur_action[0] - actions[i][0])
            rob.del_angle = abs(rob.cur_action[1] - actions[i][1])

        for i, action in enumerate(actions):
            self.robots[i].apply_action(action)

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")

        # propagate state for TIME_DELTA seconds
        time.sleep(self.TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            pass
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed")

        # 收集 t+1 时刻机器人的观测量，计算奖励，
        distances = []
        current_state = []
        for i in range(self.robots_num):
            obs = self.robots[i].get_observation()  # 获取 t+1 时刻的观测值，并且更新机器人的本地记录
            state = []
            for key in obs:
                state += obs[key]
            current_state.append(state)
            distances.append(obs['distance'][0])
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
            ori = list(R.from_euler('z', angle).as_quat())
        else:
            pos, ori = self.init_state[idx][:3], self.init_state[idx][3:]
        self.robots[idx].reset(pos, ori)

    def reset(self, tr: list = None, te: list = None, lim=5):
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
                state = []
                for key in obs:
                    state += obs[key]
                current_state.append(state)

        elif te_done:
            # reset scene
            rospy.wait_for_service("/gazebo/reset_world")
            try:
                self.reset_proxy()

            except rospy.ServiceException as e:
                print("/gazebo/reset_simulation service call failed")

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
                self.place_box(obs)

            # reload robot
            if lim:
                for i in range(robot_num):
                    self.add_random_robot(lim)
            else:
                for i in range(robot_num):
                    self.add_robot(init_state[i], init_goal[i])

            for i in range(robot_num):
                obs = self.robots[i].get_observation()
                state = []
                for key in obs:
                    state += obs[key]
                current_state.append(state)

            self.show_goal_point()
        return np.array(current_state), dict()

    def show_goal_point(self):
        pass

    def close(self):
        pass

    def place_box(self, pos):
        assert pos is not None, 'you should give the pos of the box'

        name = "cardboard_box_" + str(len(self.obstacles))
        box_state = ModelState()
        box_state.model_name = name
        box_state.pose.position.x = pos[0]
        box_state.pose.position.y = pos[1]
        box_state.pose.position.z = 0.0
        box_state.pose.orientation.x = 0.0
        box_state.pose.orientation.y = 0.0
        box_state.pose.orientation.z = 0.0
        box_state.pose.orientation.w = 1.0
        self.set_state.publish(box_state)
        self.obstacles.append(pos)


def main():
    env = MyEnv()

    robot_nums = 3
    lim = 5

    for i in range(robot_nums):
        env.add_robot([i, 0, 0, 0, 0, 0, np.pi * i], goal=[0, -1 ** i])
        # env.add_random_robot(lim=lim)

    velocity = np.zeros([env.robots_num, 2])
    while True:
        for i, rob in enumerate(env.robots):
            action = rob.goto(rob.target_pos)
            action[1] = action[1] / 10
            action[0] = action[0] / 5
            velocity[i] = action
        states, reward, te, tr, info = env.step(velocity)
        env.check_done(te=te, tr=tr, lim=0)
        # print(states)


if __name__ == '__main__':
    main()
