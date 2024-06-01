import os
import subprocess
from collections import deque

import numpy as np
from functools import partial

from gazebo_msgs.srv import GetModelState, GetModelStateRequest

from env_sim.argument import *
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import rospy
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R
from gazebo_msgs.msg import ModelState

from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

robot_path = os.path.join(os.path.dirname(__file__), "launch", 'robot.launch')


class Robot(object):
    def __init__(self, base_pos: list = None, base_ori: list = None, robot_idx: int = 0, ):
        if base_ori is None:
            base_ori = [0., 0., 0., 1.57]
        if base_pos is None:
            base_pos = [0., 0., 0.]

        self.robot = '/{}'.format(robot_idx)
        self.id = '{}'.format(robot_idx)

        rospy.wait_for_service("/gazebo/spawn_urdf_model")
        subprocess.Popen(
            ['roslaunch', robot_path, 'robot:={}'.format(robot_idx), 'x:={}'.format(base_pos[0]),
             'y:={}'.format(base_pos[1]), 'yaw:={}'.format(base_ori[-1])])

        self.target_pos = None
        self.init_pos = base_pos
        self.init_ori = base_ori
        self.last_state = base_pos + base_ori
        self.clipv = partial(np.clip, a_min=-MAX_SPEED / ROBOT_WHEEL_RADIUS, a_max=MAX_SPEED / ROBOT_WHEEL_RADIUS)
        self.reach_goal = False

        self.last_odom = None
        self.laser = [0.] * LASER_NUM
        self.cur_vel = [0, 0]

        self.vel_pub = rospy.Publisher(self.robot + "/cmd_vel", Twist, queue_size=1)
        self.laser_sub = rospy.Subscriber(
            self.robot + "/scan", LaserScan, self.ray_sensor, queue_size=1
        )
        self.odom = rospy.Subscriber(
            self.robot + "/odom", Odometry, self.odom_callback, queue_size=1
        )
        self.get_state_service = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

        self.laser_buffer = deque(maxlen=3)
        for _ in range(self.laser_buffer.maxlen):
            self.laser_buffer.append([0.] * LASER_NUM)

        # 用于记录上一时刻的数据
        self.cur_dis = 0
        self.cur_pos = None
        self.cur_action = np.zeros(2)

        # 记录动作(action)的变化
        self.del_vel = None
        self.del_angle = None

        self.set_self_state = ModelState()
        self.set_self_state.model_name = '{}'.format(robot_idx)
        self.set_self_state.pose.position.x = base_pos[0]
        self.set_self_state.pose.position.y = base_pos[1]
        self.set_self_state.pose.position.z = 0.0
        self.set_self_state.pose.orientation.x = base_ori[0]
        self.set_self_state.pose.orientation.y = base_ori[1]
        self.set_self_state.pose.orientation.z = base_ori[2]
        self.set_self_state.pose.orientation.w = base_ori[3]
        self.set_state = rospy.Publisher(
            "gazebo/set_model_state", ModelState, queue_size=10
        )

    def odom_callback(self, od_data):
        self.last_odom = od_data

    def reset(self, pos=None, ori=None):
        if pos is None:
            pos = self.init_pos
        if ori is None:
            ori = self.init_ori

        object_state = self.set_self_state
        object_state.pose.position.x = pos[0]
        object_state.pose.position.y = pos[1]
        object_state.pose.position.z = 0.
        object_state.pose.orientation.w = ori[3]
        object_state.pose.orientation.x = ori[0]
        object_state.pose.orientation.y = ori[1]
        object_state.pose.orientation.z = ori[2]
        object_state.twist.linear.x = 0.0
        object_state.twist.linear.y = 0.0
        object_state.twist.linear.z = 0.0
        object_state.twist.angular.x = 0.0
        object_state.twist.angular.y = 0.0
        object_state.twist.angular.z = 0.0
        object_state.reference_frame = "world"
        self.set_state.publish(object_state)

    def set_target_pos(self, target_pos):
        self.target_pos = target_pos

    def get_id(self):
        return self.robot

    def get_vel_and_pos(self):
        model = GetModelStateRequest()
        model.model_name = self.id
        obj_state = self.get_state_service(model)

        pos = [obj_state.pose.position.x, obj_state.pose.position.y, 0]
        ori = [
            obj_state.pose.orientation.x,
            obj_state.pose.orientation.y,
            obj_state.pose.orientation.z,
            obj_state.pose.orientation.w
        ]
        vel = obj_state.twist.linear.x
        angular_vel = obj_state.twist.angular.z
        return dict(vel=[vel, angular_vel], pos=pos, ori=ori)

    def quart_to_rpy(self, ori):
        rot = R.from_quat(ori).as_matrix()
        return rot

    def get_forward_vector(self):  # 获取机器人朝向的向量
        ori = self.get_vel_and_pos()['ori']
        matrix = self.quart_to_rpy(ori)
        return [matrix[0, 0], matrix[1, 0], 0]

    def get_observation(self):  # 根据目的地的坐标得到机器人目前的状态
        assert self.target_pos is not None, "the goal of robot %d is not initialized" % self.robot

        # obversation: laser1, ..., lasern, distance, alpha, velocity
        obs_dict = self.get_vel_and_pos()
        # 观测时同时更新当前时刻 （t） 机器人的位置，t 时刻的位置要用于计算 t 时刻距终点的距离
        vel, pos, ori = obs_dict['vel'], obs_dict['pos'], obs_dict['ori']
        self.cur_pos, vel = pos, vel

        x_, y_ = self.target_pos[0] - pos[0], self.target_pos[1] - pos[1]
        angle = self.follow_vector_angle([x_, y_])
        distance = np.linalg.norm(np.array(pos)[:2] - self.target_pos)
        return dict(laser=self.laser, distance=[distance], angle=[angle], vel=vel)

    def apply_action(self, action):  # 施加动作
        """
        给机器人施加指令
        :param action: [velocity, angular_vel]
        """
        if not (isinstance(action, list) or isinstance(action, np.ndarray)):
            assert f"apply_action() only receive list or ndarray, but receive {type(action)}"
        self.cur_action = action.copy()
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        self.vel_pub.publish(vel_cmd)

    def ray_sensor(self, v):
        """
        函数功能: 添加单线激光射线传感器，用于检测障碍物
        """
        laser = [i if i < LASER_LENGTH else LASER_LENGTH for i in list(v.ranges)]

        self.laser = laser
        self.laser_buffer.append(self.laser)

    def follow_vector_angle(self, vector):
        x, y = self.get_forward_vector()[:2]
        x_, y_ = vector[0], vector[1]
        theta = np.arccos((x * x_ + y * y_) / (np.linalg.norm([x, y]) * np.linalg.norm([x_, y_]) + 1e-7))  # 向量点乘
        signal = -1 if x_ * y - y_ * x > 0 else 1  # 叉乘
        return signal * abs(theta)

    def goto(self, goal):
        current_pos = self.get_vel_and_pos()['pos']

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
        pos = self.get_vel_and_pos()['pos']
        distance = [pos[i] - self.target_pos[i] for i in range(2)]
        distance = np.linalg.norm(distance)
        if distance < 0.1:
            return True
        return False
