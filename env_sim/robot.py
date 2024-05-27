import numpy as np
import pybullet as p
import math
from functools import partial
from env_sim.argument import *


class Robot(object):
    def __init__(self, base_pos: list = None, base_ori: list = None, client_id: int = 0,
                 urdf_path="env_sim/utils/data/turtlebot.urdf"):
        if base_ori is None:
            base_ori = [0., 0., 0., 1.]
        if base_pos is None:
            base_pos = [0., 0., 0.]
        self.client_id = client_id
        self.urdf_path = urdf_path
        self.robot = p.loadURDF(
            fileName=self.urdf_path,
            basePosition=base_pos,
            baseOrientation=base_ori,
            physicsClientId=client_id
        )
        self.target_pos = None
        self.init_pos = base_pos
        self.init_ori = base_ori
        self.last_state = base_pos + base_ori
        self.clipv = partial(np.clip, a_min=-MAX_SPEED / ROBOT_WHEEL_RADIUS, a_max=MAX_SPEED / ROBOT_WHEEL_RADIUS)
        self.reach_goal = False

        # 用于记录上一时刻的数据，其中acceleration是标量
        self.cur_dis = 0
        self.cur_pos = None
        self.cur_acc = None
        self.cur_vel = None
        self.cur_action = np.zeros(2)

        # 记录动作的变化
        self.del_vel = None
        self.del_angle = None

    def reset(self):
        # reset可能会导致有些机器人直接被卡住，所以随机reset吧
        p.resetBasePositionAndOrientation(self.robot, self.init_pos, self.init_ori, self.client_id)

    def set_target_pos(self, target_pos):
        self.target_pos = target_pos

    def get_id(self):
        return self

    def get_vel_and_pos(self):
        vel, angular_vel = p.getBaseVelocity(self.robot, self.client_id)
        pos, ori = p.getBasePositionAndOrientation(self.robot, self.client_id)
        return dict(vel=vel[:2], angular_vel=angular_vel[-1], pos=pos, ori=ori, acc=self.cur_acc)

    def get_forward_vector(self):  # 获取机器人朝向的向量
        _, baseOri = p.getBasePositionAndOrientation(self.robot, self.client_id)
        matrix = p.getMatrixFromQuaternion(baseOri)
        return [matrix[0], matrix[3], matrix[6]]

    def get_observation(self):  # 根据目的地的坐标得到机器人目前的状态
        assert self.target_pos is not None, "the goal of robot %d is not initialized" % self.client_id

        # obversation: laser1, ..., lasern, distance, alpha
        obs_dict = self.get_vel_and_pos()
        # 观测时同时更新当前时刻 （t） 机器人的位置，t 时刻的位置要用于计算 t 时刻距终点的距离
        vel, angular_vel, pos, ori = obs_dict['vel'], obs_dict['angular_vel'], obs_dict['pos'], obs_dict['ori']
        self.cur_pos, self.cur_vel = list(pos), vel

        # vel = np.array(vel)
        # if self.cur_vel is None:
        #     self.cur_acc = np.linalg.norm(vel)
        # else:
        #     self.cur_acc = np.linalg.norm(vel - self.cur_vel)

        # laser = self.ray_sensor()
        laser = []  # 暂时不考虑雷达

        x_, y_ = self.target_pos[0] - pos[0], self.target_pos[1] - pos[1]
        angle = self.follow_vector_angle([x_, y_])
        distance = np.linalg.norm(np.array(pos)[:2] - self.target_pos)

        return dict(laser=laser, distance=[distance], angle=[angle])

    def apply_action(self, action):  # 施加动作
        if not (isinstance(action, list) or isinstance(action, np.ndarray)):
            assert f"apply_action() only receive list or ndarray, but receive {type(action)}"
        self.cur_action = action.copy()
        left_v, right_v = self.action2commend(action)

        left_v = self.clipv(left_v)
        right_v = self.clipv(right_v)

        p.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=[0, 1],
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[left_v, right_v],
            forces=[10, 10],
            physicsClientId=self.client_id
        )

    def check_pos(self, Pos, goal, bias):
        '''
        check if the robot reaches the goal
        :param Pos:
        :param goal:
        :param bias:
        :return:
        '''
        if goal[0] + bias > Pos[0] > goal[0] - bias and goal[1] + bias > Pos[1] > goal[1] - bias:
            return True
        else:
            return False

    def ray_sensor(self, showRay=False):
        """
        函数功能: 添加单线激光射线传感器，用于检测障碍物
        """
        rayLength = LASER_LENGTH  # 射线长度
        rayNum = LASER_NUM  # 射线数量

        start, ori = p.getBasePositionAndOrientation(self.robot, self.client_id)  # 射线从躯干射出
        _, _, yaw = p.getEulerFromQuaternion(ori)
        # 调整激光雷达安装位置
        start = list(start)
        bia = 0.15
        start[2] += 0.1
        start[0] += bia * np.cos(yaw)
        start[1] += bia * np.sin(yaw)

        angle = np.pi  # 激光雷达扫射范围
        yaw = yaw - angle / 2
        # 调整激光雷达角度

        rayFromPos = [start for _ in range(rayNum)]
        rayToPos = np.array([[start[0] + rayLength * np.cos(angle * float(i) / (rayNum - 1) + yaw),
                              start[1] + rayLength * np.sin(angle * float(i) / (rayNum - 1) + yaw),
                              start[2]] for i in range(rayNum)])
        results = p.rayTestBatch(rayFromPos, rayToPos, self.client_id)  # 激光射线函数 返回被命中对象id、命中对象连杆索引（base连杆为-1）

        hit_position = [result[2] * rayLength for result in results]

        # 将激光射线可视化，没有命中障碍物的射线为红色，命中障碍物的射线为绿色
        if showRay:
            p.removeAllUserDebugItems()

            for index, result in enumerate(results):
                if result[0] == -1:
                    p.addUserDebugLine(rayFromPos[index], rayToPos[index], [1, 0, 0], physicsClientId=self.client_id)
                else:
                    p.addUserDebugLine(rayFromPos[index], rayToPos[index], [0, 1, 0], physicsClientId=self.client_id)
        return hit_position

    def follow_vector_angle(self, vector):
        x, y = self.get_forward_vector()[:2]
        x_, y_ = vector[0], vector[1]
        theta = np.arccos((x * x_ + y * y_) / (np.linalg.norm([x, y]) * np.linalg.norm([x_, y_]) + 1e-7))  # 向量点乘
        signal = -1 if x_ * y - y_ * x > 0 else 1  # 叉乘
        return signal * abs(theta)

    def goto(self, goal):
        basePos = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.client_id)
        current_pos = basePos[0]

        x_, y_ = goal[0] - current_pos[0], goal[1] - current_pos[1]
        angle = self.follow_vector_angle([x_, y_])

        distance = np.linalg.norm(np.array(goal) - np.array(current_pos[:2]))
        linear = distance if distance < MAX_SPEED else MAX_SPEED
        angular = angle
        return [linear, angular]

    def action2commend(self, action):
        '''
        :param action: [velocity, angular vel]
        :return: commend:[u_left, u_right]，两轮差速小车左右轮控制指令
        '''
        v, w = action[0], action[1]  # v是小车前进速度，w是小车角速度

        # R = v / w
        # v_left = (R + ROBOT_WIDTH / 2) * w
        # v_right = (R - ROBOT_WIDTH / 2) * w
        v_left = v - w * ROBOT_WIDTH / 2.0
        v_right = v + w * ROBOT_WIDTH / 2.0

        return v_left / ROBOT_WHEEL_RADIUS, v_right / ROBOT_WHEEL_RADIUS

    def is_reachable(self):
        pos, _ = p.getBasePositionAndOrientation(self.robot, self.client_id)
        distance = [pos[i] - self.target_pos[i] for i in range(2)]
        distance = np.linalg.norm(distance)
        if distance < 0.1:
            return True
        return False
