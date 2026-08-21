import rospy
from pathlib import Path
from sensor_msgs.msg import LaserScan
from collections import deque
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from rl.util import *
import time
import argparse
from distutils.util import strtobool
from scipy.spatial.transform import Rotation as R

try:
    from project_paths import MODEL_DIR
except ImportError:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from project_paths import MODEL_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default='', nargs="?", const=True, help="the name of the environment")
args = parser.parse_args()

def follow_vector_angle(goal, pos):
    x, y = pos[0],pos[1]
    x_, y_ = goal[0], goal[1]
    theta = np.arccos((x * x_ + y * y_) / (np.linalg.norm([x, y]) * np.linalg.norm([x_, y_]) + 1e-7))  # 向量点乘
    signal = -1 if x_ * y - y_ * x > 0 else 1  # 叉乘
    return signal * abs(theta)

class RealWorld:
    def __init__(self):
        # print('start init')
        self.LASER_NUM = 72
        self.LASER_LENGTH = 4

        self.last_odom = None
        self.last_vel = [0,0]
        self.name = args.name

        # print(self.name[1:])

        rospy.init_node(self.name[1:], anonymous=True)
        self.laser_sub = rospy.Subscriber(self.name+"/scan", LaserScan, self.ray_sensor, queue_size=1)
        self.vel_pub = rospy.Publisher(self.name+"/cmd_vel", Twist, queue_size=1)
        self.odom_sub = rospy.Subscriber(self.name+"/odom", Odometry, self.odom_callback, queue_size=1)
        self.rate = rospy.Rate(10)
        # print('end sub')

        self.laser_buffer = deque(maxlen=3)
        for _ in range(self.laser_buffer.maxlen):
            self.laser_buffer.append([self.LASER_LENGTH] * self.LASER_NUM)
        # print('end init')

    def odom_callback(self, od_data):
        self.last_odom = od_data

    def ray_sensor(self, v):
        """
        函数功能: 添加单线激光射线传感器，用于检测障碍物
        """
        resolution = 0.017501922324299812
        idx = 72
        v = np.array(v.ranges)

        internal = 2
        scan = np.hstack([v[360-idx::internal], v[:idx:internal]])

        laser = [i-0.15 if i < self.LASER_LENGTH and i>0 else self.LASER_LENGTH for i in scan]
        self.laser_buffer.append(laser)

    def reset(self):
        while True:
            vel_cmd = Twist()
            vel_cmd.linear.x = 0
            vel_cmd.angular.z = 0
            self.vel_pub.publish(vel_cmd)

    def main(self):
        print('start')
        goal = np.array([3.5, 0])

        agent = LstmAgent()
        agent.load_state_dict(torch.load(MODEL_DIR / "Agent_Lstm_base.pth"))

        # agent=AttentionAgent()
        # agent.load_state_dict(torch.load(MODEL_DIR / "AttentionAgent_circle.pth"))

        convert = agent.convert_action_for_env

        vel_cmd = Twist()
        vel_cmd.linear.x = 0
        vel_cmd.angular.z = 0
        self.vel_pub.publish(vel_cmd)
        action = [[0,0]]

        vel_scale = 6

        while True:
            if self.last_odom == None:
                print('not laser')
                continue
            else:
                pos = [self.last_odom.pose.pose.position.x, self.last_odom.pose.pose.position.y]
                # cur_vel = [self.last_odom.twist.twist.linear.x*5, self.last_odom.twist.twist.angular.z*np.pi]
                cur_vel = [action[0][0]*vel_scale, action[0][1]*np.pi]

                dis = np.linalg.norm(goal - pos)
                dis = dis if dis<2 else 4
                ori = [
                        self.last_odom.pose.pose.orientation.x,
                        self.last_odom.pose.pose.orientation.y,
                        self.last_odom.pose.pose.orientation.z,
                        self.last_odom.pose.pose.orientation.w]
                rot = R.from_quat(ori).as_dcm()
                cur_angle = [rot[0, 0], rot[1,0]]
                goal_angle = goal-np.array(pos)
                delta_angle = follow_vector_angle(goal_angle, cur_angle)
                # print(cur_angle)

                relative_pos = [dis, delta_angle]
                state = torch.Tensor([relative_pos+cur_vel])
                # print(self.name, state)

                laser_datas = []
                laser_data = [i for i in self.laser_buffer]
                laser_datas.append(torch.clip(torch.Tensor(laser_data),0.02,4))
                laser_datas = torch.stack(laser_datas)
                # print(laser_data)

                info = agent.get_action_and_value(laser_datas, state)
                action = info[0]

                action = convert(action)
                action[:, 0] = action[:, 0] / vel_scale
                action[:, 1] = action[:, 1]
                # print(action)

                vel_cmd = Twist()
                vel_cmd.linear.x = action[0][0]
                vel_cmd.angular.z = action[0][1]
                self.vel_pub.publish(vel_cmd)
                self.rate.sleep()

                if dis<0.7:
                    self.reset()

if __name__ == '__main__':
    print(args)
    rw = RealWorld()
    # rw.reset()
    rw.main()
