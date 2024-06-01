import subprocess
import os
import time
import rospy
from sensor_msgs.msg import PointCloud2, LaserScan
from robot import Robot
from env_sim.my_env import MyEnv

import sensor_msgs.point_cloud2 as pc2
import numpy as np
import math

# world_path = os.path.join(os.path.dirname(__file__), "launch", 'empty_env.launch')
# robot_path = os.path.join(os.path.dirname(__file__), "launch", 'robot.launch')

# subprocess.Popen(["roscore"])
# rospy.init_node("gazebo", anonymous=True)
# subprocess.Popen(['roslaunch', world_path])
# rospy.wait_for_service("/gazebo/spawn_urdf_model")

env = MyEnv()

robots_num = 3
robots = []
for i in range(robots_num):
    robots.append(Robot(base_pos=[i, 0, 0], robot_idx=i))
    robots[i].target_pos = [i, i+1]

time.sleep(robots_num)

while not rospy.is_shutdown():
    for i, rob in enumerate(robots):
        action = rob.goto(rob.target_pos)
        action[1] = action[1]/10
        action[0] = action[0]/5
        rob.apply_action(action)

        if rob.is_reachable():
            rob.reset()
