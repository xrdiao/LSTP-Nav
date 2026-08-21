import copy
from collections import deque
from pathlib import Path

import cv2
import pybullet as p
import math
from functools import partial
from matplotlib import pyplot as plt
from env_sim.argument import *
import torch

try:
    from project_paths import TURTLEBOT_URDF_PATH
except ImportError:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from project_paths import TURTLEBOT_URDF_PATH


class Node:
    def __init__(self, pos, ori, scan):
        self.pos = np.array(pos)
        self.scan = scan
        self.ori = ori
        self.reach_num = 1

    def get_data(self):
        pos = self.pos.tolist()
        ori = list(self.ori)
        return torch.Tensor(self.scan + pos + ori + [self.reach_num])


class Graph:
    def __init__(self, init_pos, init_ori, init_scan):
        self.V = [Node(init_pos, init_ori, init_scan)]
        self.E = [[0, 0]]
        self.d = 0.5
        self.last_idx = 0

    def add_edge(self, i, j):
        e = [i, j]
        if e not in self.E:
            self.E.append(e)
            self.E.append(e[::-1])

    def cal_similarity(self, pos):
        return np.array([np.linalg.norm(v.pos - pos) for v in self.V])

    def add_node(self, pos, ori, scan):
        similarity = self.cal_similarity(pos)
        min_sim, i = np.min(similarity), np.argmin(similarity)

        if min_sim >= self.d:
            idx = copy.deepcopy(len(self.V) - 1)
            self.V.append(Node(pos, ori, scan))
            self.add_edge(idx + 1, idx)
            self.last_idx = len(self.V) - 1

        elif min_sim < self.d and i != self.last_idx:
            idx = self.last_idx
            self.add_edge(i, idx)
            self.V[i].reach_num += 1
            self.last_idx = i

    def reset_graph(self, pos, ori, scan):
        self.V = [Node(pos, ori, scan)]
        self.E = [[0, 0]]
        self.last_idx = 0

    def get_data(self):
        data = []
        for v in self.V:
            data.append(v.get_data())
        return torch.stack(data), torch.Tensor(self.E)

    def plot_graph(self):
        for v in self.V:
            pos = v.pos
            circle = plt.Circle((pos[0], pos[1]), 0.1, color="black")
            plt.gcf().gca().add_artist(circle)

        for e in self.E:
            a = self.V[e[0]].pos
            b = self.V[e[1]].pos

            plt.plot([a[0], b[0]], [a[1], b[1]], color=[1, 0, 0])


class Robot(object):
    def __init__(self, base_pos: list = None, base_ori: list = None, client_id: int = 0,
                 urdf_path=None,
                 robot_camera=False, lidar_noise_std=0.01, lidar_noise_clip=0.01,
                 lidar_dropout_ratio=0.0, lidar_max_range=LASER_LENGTH):
        if base_ori is None:
            base_ori = [0., 0., 0., 1.]
        if base_pos is None:
            base_pos = [0., 0., 0.]
        self.client_id = client_id
        self.urdf_path = urdf_path or str(TURTLEBOT_URDF_PATH)
        self.robot = p.loadURDF(
            fileName=self.urdf_path,
            basePosition=base_pos,
            baseOrientation=base_ori,
            physicsClientId=client_id,
        )

        self.target_pos = None
        self.sub_goal = None
        self.init_pos = base_pos
        self.init_ori = base_ori
        self.last_state = base_pos + base_ori

        max_speed = MAX_SPEED * (np.random.rand() * 0.25 + 0.75)
        self.clip_v = partial(np.clip, a_min=-max_speed, a_max=max_speed)
        self.clip_w = partial(np.clip, a_min=-MAX_ROTATION_SPEED, a_max=MAX_ROTATION_SPEED)

        self.reach_goal = False
        self.collision_num = 0

        self.lidar_noise_std = lidar_noise_std
        self.lidar_noise_clip = lidar_noise_clip
        self.lidar_dropout_ratio = lidar_dropout_ratio
        self.lidar_max_range = lidar_max_range

        self.laser_buffer = deque(maxlen=ROBOT_LASER_BUFFER)
        for _ in range(self.laser_buffer.maxlen):
            self.laser_buffer.append(self.ray_sensor())
        self.laser = self.ray_sensor()

        # 用于记录上一时刻的数据
        self.last_obs = None
        self.cur_pos = base_pos[:2]
        self.end_test = False

        self.cur_angle_vel = 0
        self.cur_min_laser = LASER_LENGTH

        self.D = 0.175
        self.cur_vel = np.zeros(2)
        self.theta = p.getEulerFromQuaternion(self.init_ori)[-1]

        self.memory = deque(maxlen=300)
        # self.memory_graph = Graph(self.init_pos[:2], self.init_ori, self.laser_buffer[-1])
        self.set_color()
        self.robot_camera = robot_camera

    def set_color(self, color=[0.7, 0.2, 0.3, 1]):
        num_joints = p.getNumJoints(self.robot, physicsClientId=self.client_id)

        for link_idx in range(-1, num_joints):  # -1 表示基座，0~num_joints-1 是其他链接   
            p.changeVisualShape(
                self.robot,
                link_idx,
                rgbaColor=color,
                physicsClientId=self.client_id
            )

    def reset(self):
        p.resetBasePositionAndOrientation(self.robot, self.init_pos, self.init_ori, self.client_id)

    def set_target_pos(self, target_pos):
        self.target_pos = target_pos
        self.sub_goal = target_pos

    def get_id(self):
        return self.robot

    def get_vel_and_pos(self):
        vel, angular_vel = p.getBaseVelocity(self.robot, self.client_id)
        pos, ori = p.getBasePositionAndOrientation(self.robot, self.client_id)
        _, _, self.theta = p.getEulerFromQuaternion(ori)

        self.cur_vel = list(vel[:2])
        self.cur_angle_vel = angular_vel[-1]
        return dict(vel=vel[:2], angular_vel=angular_vel[-1], pos=pos, ori=ori)

    def get_forward_vector(self):  # 获取机器人朝向的向量
        _, baseOri = p.getBasePositionAndOrientation(self.robot, self.client_id)
        matrix = p.getMatrixFromQuaternion(baseOri)
        return [matrix[0], matrix[3], matrix[6]]

    def get_observation(self):  # 根据目的地的坐标得到机器人目前的状态
        assert self.sub_goal is not None, "the goal of robot %d is not initialized" % self.client_id

        # obversation: laser1, ..., lasern, distance, alpha, velocity
        obs_dict = self.get_vel_and_pos()
        # 观测时同时更新当前时刻t机器人的位置
        vel, angular_vel, pos, ori = obs_dict['vel'], obs_dict['angular_vel'], obs_dict['pos'], obs_dict['ori']
        self.cur_pos, vel, angle_vel = list(pos[:2]), [np.linalg.norm(vel)], [angular_vel]

        # 雷达信号缓存
        # self.laser = self.ray_sensor()
        self.laser_buffer.append(self.ray_sensor())
        self.laser = self.laser_buffer[-1]

        x_, y_ = self.sub_goal[0] - pos[0], self.sub_goal[1] - pos[1]
        angle = self.follow_vector_angle([x_, y_])
        distance = np.linalg.norm(self.sub_goal-np.array(pos)[:2])

        self.memory.append(dict(laser=self.laser, vel=vel, angular_vel=angular_vel, pos=pos, ori=ori))
        # self.memory_graph.add_node(pos[:2], ori, self.laser_buffer[-1])

        obs_dict = dict(distance=[distance], angle=[angle], vel=vel, angle_vel=angle_vel, laser=list(self.laser))

        if self.robot_camera:
            rgb_image = self.get_camera_image()
            
            cv2.imshow("Robot Camera", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        obs = []
        for key in obs_dict:
            obs += obs_dict[key]

        if self.last_obs == None:
            self.last_obs = obs
        return obs
    
    def apply_action(self, action):  # 施加动作
        """
        给机器人施加指令
        :param action: [velocity, angular_vel]
        """
        if not (isinstance(action, list) or isinstance(action, np.ndarray)):
            assert f"apply_action() only receive list or ndarray, but receive {type(action)}"
        cur_action = action.copy()

        cur_action[0] = self.clip_v(cur_action[0])
        cur_action[1] = self.clip_w(cur_action[1])
        left_v, right_v = self.action2commend(cur_action)

        p.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=[0, 1],
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[left_v, right_v],
            forces=[10, 10],
            physicsClientId=self.client_id
        )

    def ray_sensor(self, showRay=False, bias=False):
        """
        函数功能: 添加单线激光射线传感器，用于检测障碍物
        """
        rayLength = LASER_LENGTH  # 射线长度
        rayNum = LASER_NUM  # 射线数量

        start, ori = p.getBasePositionAndOrientation(self.robot, self.client_id)
        _, _, yaw = p.getEulerFromQuaternion(ori)

        # angle = SCAN_ANGLE  # 激光雷达扫射范围
        start = list(start)
        start[2] += 0.1

        # 调整激光雷达安装位置
        if bias:
            # 激光雷达安装位置前移
            bia = 0.18
            start[0] += bia * math.cos(yaw)
            start[1] += bia * math.sin(yaw)
            yaw = yaw - SCAN_ANGLE / 2  # 调整激光雷达角度

            # 设定激光雷达射线
            rayFromPos = [start for _ in range(rayNum)]
        else:
            yaw = yaw - SCAN_ANGLE / 2  # 调整激光雷达角度
            # 激光雷达安装位置在中心
            rayFromPos = np.array([[start[0] + ROBOT_WIDTH * math.cos(SCAN_ANGLE * float(i) / (rayNum - 1) + yaw),
                                    start[1] + ROBOT_WIDTH * math.sin(SCAN_ANGLE * float(i) / (rayNum - 1) + yaw),
                                    start[2]] for i in range(rayNum)])

        rayToPos = np.array([[start[0] + rayLength * math.cos(SCAN_ANGLE * float(i) / (rayNum - 1) + yaw),
                              start[1] + rayLength * math.sin(SCAN_ANGLE * float(i) / (rayNum - 1) + yaw),
                              start[2]] for i in range(rayNum)])
        results = p.rayTestBatch(rayFromPos, rayToPos, self.client_id)  # 激光射线函数 返回被命中对象id、命中对象连杆索引（base连杆为-1）

        hit_position = []
        for result in results:
            distance = result[2] * rayLength
            if self.lidar_noise_std > 0:
                distance += np.clip(
                    np.random.normal(loc=0.0, scale=self.lidar_noise_std),
                    -self.lidar_noise_clip,
                    self.lidar_noise_clip
                )
            distance = float(np.clip(distance, 0.0, self.lidar_max_range))
            if self.lidar_dropout_ratio > 0 and np.random.rand() < self.lidar_dropout_ratio:
                distance = self.lidar_max_range
            hit_position.append(distance)

        # 将激光射线可视化，没有命中障碍物的射线为红色，命中障碍物的射线为绿色
        if showRay:
            p.removeAllUserDebugItems()
            for index, result in enumerate(results):
                if result[0] == -1:
                    p.addUserDebugLine(rayFromPos[index], rayToPos[index], [1, 0, 0], physicsClientId=self.client_id)
                else:

                    plot_line = np.array(
                        [start[0] + hit_position[index] * math.cos(SCAN_ANGLE * float(index) / (rayNum - 1) + yaw),
                         start[1] + hit_position[index] * math.sin(SCAN_ANGLE * float(index) / (rayNum - 1) + yaw),
                         start[2]])
                    p.addUserDebugLine(rayFromPos[index], plot_line, [0, 1, 0], physicsClientId=self.client_id)
        return hit_position

    def follow_vector_angle(self, vector):
        x, y = self.get_forward_vector()[:2]
        x_, y_ = vector[0], vector[1]
        theta = np.arccos((x * x_ + y * y_) / (np.linalg.norm([x, y]) * np.linalg.norm([x_, y_]) + 1e-7))  # 向量点乘
        signal = -1 if x_ * y - y_ * x > 0 else 1  # 叉乘
        return signal * abs(theta)

    def getEffectiveVel(self):
        w = self.cur_angle_vel
        v = self.cur_vel
        vr = v + 0.5 * w * ROBOT_WIDTH
        vl = 2 * v - vr
        x_vel = (0.5 * math.cos(self.theta) + self.D * math.sin(self.theta) / ROBOT_WIDTH) * vl + (
                0.5 * math.cos(self.theta) - self.D * math.sin(self.theta) / ROBOT_WIDTH) * vr
        y_vel = (0.5 * math.sin(self.theta) - self.D * math.cos(self.theta) / ROBOT_WIDTH) * vl + (
                0.5 * math.sin(self.theta) + self.D * math.cos(self.theta) / ROBOT_WIDTH) * vr
        return np.array([x_vel, y_vel])

    def cal_effective_cmd(self, pref_vel):
        A = 0.5 * math.cos(self.theta) + self.D * math.sin(self.theta) / ROBOT_WIDTH
        B = 0.5 * math.cos(self.theta) - self.D * math.sin(self.theta) / ROBOT_WIDTH
        C = 0.5 * math.sin(self.theta) - self.D * math.cos(self.theta) / ROBOT_WIDTH
        D = 0.5 * math.sin(self.theta) + self.D * math.cos(self.theta) / ROBOT_WIDTH

        vx = pref_vel[0]
        vy = pref_vel[1]
        vr = (vy - C / A * vx) / (D - B * C / A)
        vl = (vx - B * vr) / A

        w = (vr - vl) / ROBOT_WIDTH
        w = np.clip(-MAX_ROTATION_SPEED, w, MAX_ROTATION_SPEED)
        v = 0.5 * (vl + vr)
        v = np.clip(0, v, MAX_SPEED)
        return [v, w]

    def goto(self):
        goal = self.sub_goal
        basePos = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.client_id)
        current_pos = basePos[0]

        x_, y_ = goal[0] - current_pos[0], goal[1] - current_pos[1]
        angle = self.follow_vector_angle([x_, y_])

        distance = np.linalg.norm(np.array(goal) - np.array(current_pos[:2]))
        linear = distance if distance < MAX_SPEED else MAX_SPEED
        angular = angle
        return [linear, angular]

    def action2commend(self, action):
        """
        :param action: [velocity, angular vel]
        :return: commend:[u_left, u_right]，两轮差速小车左右轮控制指令
        """
        v, w = action[0], action[1]  # v是小车前进速度，w是小车角速度
        v_left = v - w * ROBOT_WIDTH / 2.0
        v_right = v + w * ROBOT_WIDTH / 2.0
        return v_left / ROBOT_WHEEL_RADIUS, v_right / ROBOT_WHEEL_RADIUS

    def is_reachable(self):
        pos, _ = p.getBasePositionAndOrientation(self.robot, self.client_id)
        distance = [pos[i] - self.sub_goal[i] for i in range(2)]
        distance = np.linalg.norm(distance)
        if distance < 2.3:   # 2.3
            # if not self.reach_goal:
            #     print("Robot with id:{} reach goal".format(self.robot))
            return True
        return False
    
    def get_camera_image(
        self,
        camera_width=160,
        camera_height=120,
        fov=120,
        near_plane=0.01,
        far_plane=20.0,
        camera_forward_offset=0.08,
        camera_height_offset=0.32,
        look_ahead=1.0,
        pitch_deg=-10.0,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    ):
        """
        ViNT-oriented front camera approximation.

        Notes:
        - 4:3 output, consistent with ViNT preprocessing
        - wide-angle perspective approximation
        - lower camera mount and slight downward pitch
        - still not a true fisheye model
        """
        robot_pos, robot_orn = p.getBasePositionAndOrientation(self.robot, self.client_id)
        _, _, yaw = p.getEulerFromQuaternion(robot_orn)

        cam_x = robot_pos[0] + camera_forward_offset * math.cos(yaw)
        cam_y = robot_pos[1] + camera_forward_offset * math.sin(yaw)
        cam_z = robot_pos[2] + camera_height_offset

        pitch = math.radians(pitch_deg)
        target_x = cam_x + look_ahead * math.cos(yaw)
        target_y = cam_y + look_ahead * math.sin(yaw)
        target_z = cam_z + look_ahead * math.tan(pitch)

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[cam_x, cam_y, cam_z],
            cameraTargetPosition=[target_x, target_y, target_z],
            cameraUpVector=[0, 0, 1],
        )

        projection_matrix = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=camera_width / camera_height,
            nearVal=near_plane,
            farVal=far_plane,
        )

        _, _, rgb_img, _, _ = p.getCameraImage(
            width=camera_width,
            height=camera_height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=renderer,
            physicsClientId=self.client_id,
        )

        rgb_array = np.asarray(rgb_img, dtype=np.uint8).reshape(camera_height, camera_width, 4)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array
