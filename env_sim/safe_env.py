from typing import Optional

from env_sim.argument import *
from env_sim.my_env import MyEnv
import numpy as np
import pybullet as p


class SafeEnv(MyEnv):
    def __init__(self, env_args, urdf_path: Optional[str] = 'utils/data/turtlebot.urdf'):
        super(SafeEnv, self).__init__(env_args, urdf_path)

    def __reward_func(self, distances: list):
        rewards = []
        cost = []
        # terminated表示智能体是否到达终点，truncated表示智能体因时间或物理碰撞等因素停止运行
        te, tr = np.zeros_like(self.robots, dtype=bool), np.zeros_like(self.robots, dtype=bool)

        for i, rob in enumerate(self.robots):
            reach_reward, te = self.reach_reward(idx=i, distances=distances, te=te)
            collision_cost, tr = self.collision_cost(idx=i, tr=tr, min_dist_boundary=1.)

            rg = reach_reward
            rc = collision_cost
            rw = ANGULAR_VELOCITY_PENALTY * abs(rob.cur_angle_vel) if abs(rob.cur_angle_vel) > 2 else 0

            rewards.append(round(rg + rw, 3))
            cost.append(-round(rc, 3))

        te = self.check_done(te)
        return rewards, cost, te, tr

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
            rob.cur_dis = round(self.distance(rob.cur_pos[:2], rob.target_pos[:2]), 4)
            rob.delta_angle = abs(rob.cur_action[1] - actions[i][1])

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
            state = []
            for key in obs:
                state += obs[key]
            current_state.append(state)
            distances.append(obs['distance'][0])
        reward, cost, te, tr = self.__reward_func(distances)

        info = {"distance": distances, "collision_num": self.collision_num}

        current_state = np.array(current_state)
        return current_state, reward, cost, te, tr, info
