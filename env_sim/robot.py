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
        self.clipv = partial(np.clip, a_min=-MAX_SPEED, a_max=MAX_SPEED)
        self.cur_dis = 0

    def reset(self):
        self.robot = p.loadURDF(
            fileName=self.urdf_path,
            basePosition=self.init_pos,
            baseOrientation=self.init_ori,
            physicsClientId=self.client_id
        )

    def set_target_pos(self, target_pos):
        self.target_pos = target_pos

    def get_state(self):
        return self.get_observation()

    def get_id(self):
        return self

    def __get_forward_vector(self):  # 获取机器人朝向的向量
        _, baseOri = p.getBasePositionAndOrientation(self.robot)
        matrix = p.getMatrixFromQuaternion(baseOri)
        return [matrix[0], matrix[3], matrix[6]]

    def __angle(self, v1, v2):
        v1 = np.array(v1)
        v2 = np.array(v2)
        cosangle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.arccos(cosangle)

    def get_observation(self):  # 根据目的地的坐标得到机器人目前的状态
        assert self.target_pos is not None, "the goal of robot %d is not initialized" % self.client_id
        # obversation: laser1, ..., n, distance, alpha
        pos, ori = p.getBasePositionAndOrientation(self.robot)
        ori = p.getMatrixFromQuaternion(ori)
        laser = self.ray_sensor()

        angle = self.__angle(
            v1=[ori[0], ori[3], ori[6]],
            v2=[y - x for x, y in zip(pos, self.target_pos)] + [0.]
        )

        return laser + [np.linalg.norm(np.array(pos)[:2] - self.target_pos), angle]

    def apply_action(self, action):  # 施加动作
        if not (isinstance(action, list) or isinstance(action, np.ndarray)):
            assert f"apply_action() only receive list or ndarray, but receive {type(action)}"
        _action = [MAX_SPEED * action[0], MAX_ROTATION_SPEED * action[1]]
        left_v, right_v = self.action2commend(action)

        left_v = self.clipv(left_v)
        right_v = self.clipv(right_v)

        p.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=[0, 1],
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[left_v, right_v],
            forces=[10, 10]
        )

    def check_pos(self, Pos, goal, bias):
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

        start, ori = p.getBasePositionAndOrientation(self.robot)  # 射线从躯干射出
        _, _, yaw = p.getEulerFromQuaternion(ori)
        # 调整激光雷达安装位置
        start = list(start)
        bia = 0.1
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
        results = p.rayTestBatch(rayFromPos, rayToPos)  # 激光射线函数 返回被命中对象id、命中对象连杆索引（base连杆为-1）

        hit_position = [result[2] * rayLength for result in results]

        # 将激光射线可视化，没有命中障碍物的射线为红色，命中障碍物的射线为绿色
        if showRay:
            p.removeAllUserDebugItems()

            for index, result in enumerate(results):
                if result[0] == -1:
                    p.addUserDebugLine(rayFromPos[index], rayToPos[index], [1, 0, 0])
                else:
                    p.addUserDebugLine(rayFromPos[index], rayToPos[index], [0, 1, 0])
        return hit_position

    def goto(self, goal):
        goal_x, goal_y = goal[0], goal[1]
        basePos = p.getBasePositionAndOrientation(self.robot)
        current_x = basePos[0][0]
        current_y = basePos[0][1]

        current_orientation = list(p.getEulerFromQuaternion(basePos[1]))[2]
        goal_direction = math.atan2((goal_y - current_y), (goal_x - current_x))
        if current_orientation < 0:
            current_orientation = current_orientation + 2 * math.pi
        if goal_direction < 0:
            goal_direction = goal_direction + 2 * math.pi

        theta = goal_direction - current_orientation
        if theta < 0 and abs(theta) > abs(theta + 2 * math.pi):
            theta = theta + 2 * math.pi
        elif theta > 0 and abs(theta - 2 * math.pi) < theta:
            theta = theta - 2 * math.pi

        k_linear = 10
        k_angular = 30
        linear = k_linear * math.cos(theta)
        angular = k_angular * theta

        rightWheelVelocity = linear + angular
        leftWheelVelocity = linear - angular
        # p.setJointMotorControl2(self.robot, 0, p.VELOCITY_CONTROL, targetVelocity=leftWheelVelocity, force=10)
        # p.setJointMotorControl2(self.robot, 1, p.VELOCITY_CONTROL, targetVelocity=rightWheelVelocity, force=10)
        return [leftWheelVelocity, rightWheelVelocity]

    def action2commend(self, action):
        '''
        :param action: [velocity, rotation]
        :return: commend:[u_left, u_right]，两轮差速小车左右轮控制指令
        '''
        v, w = action[0], action[1]  # v是小车前进速度，w是小车角速度
        if abs(w) < 1e-3:
            u_left, u_right = v, v
            return u_left, u_right

        R = v / w
        v_left = (R + ROBOT_WIDTH / 2) * w
        v_right = (R - ROBOT_WIDTH / 2) * w

        return v_left, v_right
