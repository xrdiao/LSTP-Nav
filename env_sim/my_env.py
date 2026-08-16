import json
import math
import random
import time
from typing import Optional, Dict, Any

import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
from matplotlib import pyplot as plt

# import gym
# from gym import spaces
from env_sim.robot import Robot
from env_sim.argument import *

import warnings

warnings.filterwarnings("ignore")


class MyEnv(gym.Env):
    def __init__(self, env_args, urdf_path: Optional[str] = '/home/oem/direction_based_obstacle_avoidance/env_sim/utils/data/turtlebot.urdf'):
        """
        :param env_args: Arguments to environment
        :param urdf_path: the urdf path of robot
        """
        self.time_step = 1. / 240.
        self.max_simulate_steps = 4000
        self.simulate_steps = 0

        self.env_args = env_args
        self.random_obstacles = self.env_args.random_obstacles
        self.random_mode = self.env_args.render
        self.boundary = self.env_args.boundary
        self.x_lim = self.env_args.x_lim
        self.y_lim = self.env_args.y_lim

        self._physics_client_id = p.connect(p.GUI if self.random_mode else p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self._physics_client_id)
        p.setGravity(0., 0., -9.8, physicsClientId=self._physics_client_id)
        p.setRealTimeSimulation(0, physicsClientId=self._physics_client_id)  # 1表示随着真实时间仿真，0表示要用p.step()进行步进
        p.setTimeStep(self.time_step, physicsClientId=self._physics_client_id)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        plane_id = p.loadURDF("plane.urdf", physicsClientId=self._physics_client_id)
        p.changeVisualShape(plane_id, -1, 
                rgbaColor=[255, 255, 255, 1],
                specularColor=[0, 0, 0],  # 关闭镜面反射
                textureUniqueId=-1)

        # 禁用不必要的视觉效果
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)  # 确保渲染开启
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)    # 禁用阴影
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)        # 禁用GUI控件
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)  # 禁用备用渲染器
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)  # 禁用RGB预览
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)  # 禁用深度预览
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)  # 禁用分段标记预览

        self.time_limit = None
        self.robots = []  # a Robot instance representing the robot
        self.obstacles = []

        self.global_time = 0
        self.step_counter = 0
        self.collision_num = 0
        self.reach_num = 0

        self.num_envs = self.robots_num = self.env_args.robots_num
        self.urdf_path = urdf_path
        self.TARGET_VELOCITY = TARGET_VELOCITY
        self.LASER_NUM = LASER_NUM
        self.LASER_LENGTH = LASER_LENGTH

        # 动作空间: 左轮速度， 右轮速度
        self.action_space = spaces.Box(
            low=np.array([-self.TARGET_VELOCITY, -self.TARGET_VELOCITY], dtype=np.float32),
            high=np.array([self.TARGET_VELOCITY, self.TARGET_VELOCITY], dtype=np.float32),
        )
        # 状态空间: laser1, ..., 5,   distance, alpha
        self.observation_space = spaces.Box(
            low=np.array([0.] * self.LASER_NUM + [-MAX_DISTANCE, -np.pi * 2] + [-MAX_SPEED, -MAX_ROTATION_SPEED],
                         dtype=np.float32),
            high=np.array([self.LASER_LENGTH + 1] * self.LASER_NUM + [MAX_DISTANCE, np.pi * 2] +
                          [MAX_SPEED, MAX_ROTATION_SPEED], dtype=np.float32),
        )

        self.num_obs = self.observation_space.low.shape[0]
        self.num_actions = 2
        self.init_state = []
        self.init_goal = []
        self.layout_obstacle_specs = []
        self.layout_robot_colors = []

        self.control_rate = self.env_args.control_rate
        self.FPS = int(1 / self.time_step / self.control_rate)
        self.delta_time = self.time_step * self.FPS
        self.hs_reward_sigma = max(float(getattr(self.env_args, "hs_reward_sigma", 0.1)), 1e-6)
        self.lidar_noise_std = float(getattr(self.env_args, "lidar_noise_std", 0.01))
        self.lidar_noise_clip = float(getattr(self.env_args, "lidar_noise_clip", self.lidar_noise_std))
        self.lidar_dropout_ratio = float(getattr(self.env_args, "lidar_dropout_ratio", 0.0))
        self.dyn_scan_weight = np.exp(-(np.linspace(0, 1, LASER_NUM)) ** 2 / (self.hs_reward_sigma ** 2) / 2)
        self.dyn_scan_weight = self.dyn_scan_weight/self.dyn_scan_weight.sum()/2

        self.name = self.env_args.name
        if self.render:
            p.resetDebugVisualizerCamera(cameraDistance=15, cameraYaw=0, cameraPitch=-89.9,
                                         cameraTargetPosition=[0, 0, 0])
            
        if self.env_args.test_mode:
            self.create_env()
        self.robot_camera = self.env_args.robot_camera

                # 定义形状参数（字典映射）
        self.shape_params = {
            "BOX": {
                "visual": {"shapeType": p.GEOM_BOX, "halfExtents": [0.5, 0.5, 0.5], "rgbaColor": [1, 0.8, 0, 1]},
                "collision": {"shapeType": p.GEOM_BOX, "halfExtents": [0.5, 0.5, 0.5]},
                "height" :0.5

            },
            "CYLINDER": {
                "visual": {"shapeType": p.GEOM_CYLINDER, "radius": 0.5, "length": 1.0, "rgbaColor": [0.8, 0.5, 1, 1]},
                "collision": {"shapeType": p.GEOM_CYLINDER, "radius": 0.5, "height": 1.0},
                "height" :0.5
            },
            "SPHERE": {
                "visual": {"shapeType": p.GEOM_SPHERE, "radius": 0.5, "rgbaColor": [1, 0.5, 0.3, 1]},
                "collision": {"shapeType": p.GEOM_SPHERE, "radius": 0.5},
                "height" :0.

            },
            "CAPSULE": {
                "visual": {"shapeType": p.GEOM_CAPSULE, "radius": 0.5, "length": 2.0, "rgbaColor": [0.3, 1, 0.6, 1]},
                "collision": {"shapeType": p.GEOM_CAPSULE, "radius": 0.5, "height": 2.0},
                "height" :0.
            },
        }
        self.obstacle_path = './obstacle'

    def set_max_step(self, max_step):
        self.max_simulate_steps = max_step

    def add_robot(self, state, goal=None, random_angle=False):
        if goal is None:
            goal = [0, 0]
        assert len(state) == 7, 'state must have 7 elements, [pos, ori]'
        for _state in self.init_state:
            assert self.distance(state[:2],
                                 _state[:2]) > 1.5 * ROBOT_WIDTH, 'the robot with id={} is incorrect'.format(
                self.robots_num)

        if random_angle:
            angle = np.random.uniform(low=-np.pi, high=np.pi)
            ori = list(p.getQuaternionFromEuler([0, 0, angle], self._physics_client_id))
        else:
            ori = state[3:]

        robot = Robot(
            base_pos=state[:3],
            base_ori=ori,
            client_id=self._physics_client_id,
            urdf_path=self.urdf_path,
            robot_camera=self.env_args.robot_camera,
            lidar_noise_std=self.lidar_noise_std,
            lidar_noise_clip=max(self.lidar_noise_clip, self.lidar_noise_std),
            lidar_dropout_ratio=self.lidar_dropout_ratio,
            lidar_max_range=self.LASER_LENGTH
        )
        self._apply_robot_sensor_config(robot)
        robot.set_target_pos(goal[:2])

        self.robots.append(robot)
        self.robots_num += 1
        self.init_state.append(state)
        self.init_goal.append(goal)

    def _get_env_setting(self, name, default):
        return getattr(self.env_args, name, default)

    def _add_layout_box_spec(self, center_x, center_y, size_x, size_y, angle=0.):
        if size_x <= 0 or size_y <= 0:
            return

        self.layout_obstacle_specs.append({
            "shape_type": "BOX",
            "x": float(center_x),
            "y": float(center_y),
            "angle": float(angle),
            "half_extents": [float(size_x) / 2, float(size_y) / 2, 0.5],
            "color": [0.12, 0.12, 0.12, 1.0],
        })

    def _spawn_layout_obstacles(self):
        visual_cache = {}
        collision_cache = {}

        for spec in self.layout_obstacle_specs:
            half_extents = tuple(spec.get("half_extents", [0.5, 0.5, 0.5]))
            color = tuple(spec.get("color", [0.12, 0.12, 0.12, 1.0]))

            if (half_extents, color) not in visual_cache:
                visual_cache[(half_extents, color)] = p.createVisualShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=list(half_extents),
                    rgbaColor=list(color),
                    physicsClientId=self._physics_client_id
                )

            if half_extents not in collision_cache:
                collision_cache[half_extents] = p.createCollisionShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=list(half_extents),
                    physicsClientId=self._physics_client_id
                )

            angle = float(spec.get("angle", 0.))
            body_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_cache[half_extents],
                baseVisualShapeIndex=visual_cache[(half_extents, color)],
                basePosition=[float(spec["x"]), float(spec["y"]), half_extents[2]],
                baseOrientation=p.getQuaternionFromEuler([0, 0, angle], self.physics_client_id),
                physicsClientId=self._physics_client_id
            )
            self.obstacles.append([
                float(spec["x"]),
                float(spec["y"]),
                angle,
                body_id,
                spec.get("shape_type", "BOX"),
                {
                    "size_x": float(half_extents[0] * 2),
                    "size_y": float(half_extents[1] * 2),
                }
            ])

    def _generate_points_in_rect(self, center_x, center_y, width, height, count, margin=0.9):
        if count <= 0:
            return []

        usable_width = max(float(width) - 2 * margin, 0.1)
        usable_height = max(float(height) - 2 * margin, 0.1)
        if count == 1:
            return [[float(center_x), float(center_y)]]

        aspect = usable_width / max(usable_height, 0.1)
        cols = max(1, min(count, int(np.ceil(np.sqrt(count * aspect)))))
        rows = max(1, int(np.ceil(count / cols)))

        xs = [float(center_x)] if cols == 1 else np.linspace(
            center_x - usable_width / 2, center_x + usable_width / 2, cols
        ).tolist()
        ys = [float(center_y)] if rows == 1 else np.linspace(
            center_y + usable_height / 2, center_y - usable_height / 2, rows
        ).tolist()

        points = []
        for row_idx, y in enumerate(ys):
            row_x = xs if row_idx % 2 == 0 else xs[::-1]
            for x in row_x:
                points.append([float(x), float(y)])
                if len(points) == count:
                    return points
        return points

    @staticmethod
    def _angles_from_pairs(pos, goal):
        return [float(math.atan2(g[1] - p[1], g[0] - p[0])) for p, g in zip(pos, goal)]

    def check_done(self, te):
        # 到达最大仿真步数时，重置环境
        if self.simulate_steps >= self.max_simulate_steps:
            te = [True] * len(te)
            # print('max steps reached: {}'.format(self.simulate_steps))
        return te

    def random_point(self, init_point=[]):
        """
        :param seed: 0 - init pos, 1- init goal
        :return:
        """
        point = None
        done = False
        while not done:
            point = np.random.uniform(low=-self.env_args.x_range, high=self.env_args.x_range, size=1).tolist() + np.random.uniform(
                low=-self.env_args.y_range, high=self.env_args.y_range, size=1).tolist()
            done = True

            for obs in self.obstacles:
                done = True and done if self.distance(obs[:2], point) > (3 + ROBOT_WIDTH * 3) else False

            for rob in self.robots:
                done = True and done if self.distance(rob.cur_pos[:2], point) > (4 * ROBOT_WIDTH) else False
                done = True and done if self.distance(rob.target_pos[:2], point) > (4 * ROBOT_WIDTH) else False

            if init_point:
                done = True and done if self.distance(init_point[:2], point) > (4 * ROBOT_WIDTH) else False

        return point + [0.01]

    def random_points(self, batch_size):
        points = [self.random_point() for _ in range(batch_size)]
        return points

    def add_random_robot(self):
        """
        生成一个随机机器人
        """
        pos = self.random_point()
        angle = np.random.uniform(low=-np.pi, high=np.pi)
        ori = list(p.getQuaternionFromEuler([0, 0, angle], self._physics_client_id))
        robot = Robot(
            base_pos=pos,
            base_ori=ori,
            client_id=self._physics_client_id,
            urdf_path=self.urdf_path,
            lidar_noise_std=self.lidar_noise_std,
            lidar_noise_clip=max(self.lidar_noise_clip, self.lidar_noise_std),
            lidar_dropout_ratio=self.lidar_dropout_ratio,
            lidar_max_range=self.LASER_LENGTH
        )
        self._apply_robot_sensor_config(robot)

        goal = self.random_point(init_point=pos)[:2]
        robot.set_target_pos(goal)

        state = pos + ori
        self.robots.append(robot)
        self.robots_num += 1
        self.init_state.append(state)
        self.init_goal.append(goal)

    def _apply_robot_sensor_config(self, robot):
        robot.lidar_noise_std = self.lidar_noise_std
        robot.lidar_noise_clip = max(self.lidar_noise_clip, self.lidar_noise_std)
        robot.lidar_dropout_ratio = self.lidar_dropout_ratio
        robot.lidar_max_range = self.LASER_LENGTH

    def checkCollision(self, robot_id):
        # 也可以用距离判断
        if p.getContactPoints(bodyA=robot_id, linkIndexA=-1, physicsClientId=self._physics_client_id):
            return True
        return False

    def collision_cost(self, idx, tr, min_dist_boundary=1):
        rob = self.robots[idx]
        # 碰撞惩罚
        # scan = rob.laser_buffer[-1]
        scan = rob.laser
        min_scan_dist = np.amin(np.array(scan))

        if self.env_args.test_mode:
            is_collision = (min_scan_dist <= 0.01) or self.checkCollision(rob.robot)
        else:
            is_collision = (min_scan_dist <= 0.04) or self.checkCollision(rob.robot)

        if is_collision:
            tr[idx] = True
            rc = COLLISION_REWARD
            self.collision_num += 1
            rob.collision_num += 1
            # print("robot with id={} collides! physical id:{},".format(rob.robot, self.physics_client_id))
        else:
            if self.env_args.ori_reward:
                a = rob.cur_angle_vel / 8  # dt 之后的速度与当前朝向的夹角
                alpha = np.clip(a + SCAN_ANGLE / 2, 0, SCAN_ANGLE)
                angle_resolution = SCAN_ANGLE / LASER_NUM
                angle_idx = int(np.floor(alpha / angle_resolution))

                dyn_scan_W = np.hstack(
                    [self.dyn_scan_weight[:angle_idx][::-1], self.dyn_scan_weight[:(LASER_NUM - angle_idx)]])
                rc = dyn_scan_W @ (LASER_LENGTH - np.array(scan)) * SCAN_ORI_DISTANCE_PENALTY

                if min_scan_dist < min_dist_boundary:
                    rc += SCAN_DISTANCE_PENALTY * (min_dist_boundary - min_scan_dist)

            else:
                if min_scan_dist < min_dist_boundary:
                    rc = SCAN_DISTANCE_PENALTY * (min_dist_boundary - min_scan_dist) + np.clip(
                        (min_scan_dist - rob.cur_min_laser) * (min_dist_boundary - min_scan_dist) * SCAN_VEL_PENALTY,
                        -0.2, 0)
                else:
                    rc = 0
        rob.cur_min_laser = min_scan_dist
        return rc, tr

    def reach_reward(self, idx, te, state):
        rob = self.robots[idx]
        
        # 到达目标点奖励
        if rob.is_reachable():
            te[idx] = True
            rg=0
            if not rob.reach_goal:
                rob.reach_goal = True
                self.reach_num += 1
                rg = ARRIVAL_REWARD

        # 距离目标点距离奖励
        else:
            # 到达目标点后离开目标点需要扣除已获得的奖励
            last_dis = np.linalg.norm(rob.last_obs[:2])
            dis = np.linalg.norm(state[:2])
            rg = DISTANCE_REWARD_WEIGHT * (last_dis - dis)

            if rob.reach_goal:
                rob.reach_goal = False
                self.reach_num -= 1
                rg = -ARRIVAL_REWARD
                # print("robot with id:{} left the goal".format(rob.robot))

        rob.last_obs = state
        return rg, te

    def _reward_func(self, states: list):
        rewards = []
        # terminated表示智能体是否到达终点，truncated表示智能体因时间或物理碰撞等因素停止运行
        te, tr = np.zeros_like(self.robots, dtype=bool), np.zeros_like(self.robots, dtype=bool)

        for i, _ in enumerate(self.robots):
            reach_reward, te = self.reach_reward(idx=i, state=states[i], te=te)
            collision_cost, tr = self.collision_cost(idx=i, tr=tr)

            rg = reach_reward
            rc = collision_cost
            # 角速度惩罚
            # rw = ANGULAR_VELOCITY_PENALTY * abs(rob.cur_angle_vel) if abs(rob.cur_angle_vel) > 2 else 0
            # rv = -0.001 if np.linalg.norm(rob.cur_vel) < 0.05 else 0
            reward = rg + rc if rg < 2 else rg
            rewards.append(reward)

        te = self.check_done(te)
        return rewards, te, tr

    @staticmethod
    def distance(v1, v2):
        v1, v2 = np.array(v1), np.array(v2)
        return np.linalg.norm(v1 - v2)

    def step(self, actions):
        """
        :param actions: [velocity, angular_vel]
        :return:
        """
        assert self.robots is not None, 'no robots loaded'
        assert self.robots_num == len(actions), 'incorrect number of the actions'

        self.global_time += self.time_step
        self.simulate_steps += 1

        # 更新 t 时刻机器人距终点的距离，计算 t+1 和 t 时刻间的变化量
        for i, rob in enumerate(self.robots):
            rob.apply_action(actions[i])

        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 0)
        for _ in range(self.FPS):
            p.stepSimulation(physicsClientId=self._physics_client_id)

        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)

        # 收集 t+1 时刻机器人的观测量，计算奖励，
        states = []
        for i in range(self.robots_num):
            states.append(self.robots[i].get_observation())
        reward, te, tr = self._reward_func(states)
        states = np.array(states)

        states[:, 0] = np.clip(states[:, 0], 0, 4)

        info = {"collision_num": self.collision_num}
        return np.array(states), reward, te, tr, info

    def reset_robot(self, idx: int):
        """
        Reset the position of the robot
        """
        rob = self.robots[idx]

        if self.env_args.random_robot:
            pos = self.random_point()
            angle = np.random.uniform(low=-np.pi, high=np.pi)
            ori = list(p.getQuaternionFromEuler([0, 0, angle], self._physics_client_id))
            # goal = self.random_point()[:2]
            # rob.set_target_pos(goal)
            # self.init_goal[idx-len(self.obstacles)] = goal
        else:
            # pos, ori = self.init_state[idx][:3], self.init_state[idx][3:]
            memory = self.robots[idx].memory[0]
            pos, ori = memory['pos'], memory['ori']
        p.resetBasePositionAndOrientation(rob.robot, pos, ori, self._physics_client_id)

        rob.last_obs = None
        rob.cur_pos = pos
        rob.memory.clear()
        # rob.memory_graph.reset_graph(pos[:2], ori, rob.laser_buffer[-1])

    def plot_arrow(self, length=0.5, width=0.1):  # pragma: no cover
        for rob in self.robots:
            x, y, yaw = rob.cur_pos[0], rob.cur_pos[1], rob.theta
            plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                      head_length=width, head_width=width)
            plt.plot(x, y)

    def plot_obstacle(self):
        for obstacle in self.obstacles:
            if len(obstacle) < 3:
                continue

            x, y, yaw = obstacle[0], obstacle[1], obstacle[2]
            width = 1.0
            height = 1.0
            if len(obstacle) >= 6 and isinstance(obstacle[5], dict):
                width = float(obstacle[5].get("size_x", width))
                height = float(obstacle[5].get("size_y", height))

            dx = -width / 2 * np.cos(yaw) + height / 2 * np.sin(yaw)
            dy = -width / 2 * np.sin(yaw) - height / 2 * np.cos(yaw)
            square = plt.Rectangle(
                xy=(x + dx, y + dy),
                width=width,
                height=height,
                angle=obstacle[2] / np.pi * 180
            )
            plt.gcf().gca().add_artist(square)

    def get_obstacle_records(self):
        obstacle_records = []
        for obstacle in self.obstacles:
            if len(obstacle) < 3:
                continue

            shape_type = obstacle[4] if len(obstacle) >= 5 else "BOX"
            shape_param = self.shape_params.get(shape_type, self.shape_params["BOX"])
            visual_param = shape_param.get("visual", {})
            obstacle_meta = obstacle[5] if len(obstacle) >= 6 and isinstance(obstacle[5], dict) else {}
            record = {
                "x": float(obstacle[0]),
                "y": float(obstacle[1]),
                "yaw": float(obstacle[2]),
                "shape_type": shape_type,
            }

            if shape_type == "BOX":
                half_extents = visual_param.get("halfExtents", [0.5, 0.5, 0.5])
                record["size_x"] = float(obstacle_meta.get("size_x", half_extents[0] * 2))
                record["size_y"] = float(obstacle_meta.get("size_y", half_extents[1] * 2))
            elif shape_type in ("CYLINDER", "SPHERE"):
                record["radius"] = float(visual_param.get("radius", 0.5))
            elif shape_type == "CAPSULE":
                record["radius"] = float(visual_param.get("radius", 0.5))
                record["length"] = float(visual_param.get("length", 2.0))

            obstacle_records.append(record)

        return obstacle_records

    def plot_robot(self):  # pragma: no cover
        for rob in self.robots:
            x, y, yaw = rob.cur_pos[0], rob.cur_pos[1], rob.theta
            circle = plt.Circle((x, y), ROBOT_WIDTH, color="b")
            plt.gcf().gca().add_artist(circle)
            rob.memory_graph.plot_graph()
            plt.plot(rob.target_pos[0], rob.target_pos[1], "xb", color=[1, 0, 0])

    def plot_in_plt(self):
        plt.cla()
        self.plot_obstacle()
        self.plot_robot()
        self.plot_arrow()

        plt.axis("equal")
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        # plt.grid(True)
        plt.pause(1e-8)

    def reset(self, tr: list = None, te: list = None, seed=0):
        """
            - 所有机器人到达目标点后重设场景
            - 部分机器人发生碰撞后重设机器人
            :param te:
            :param tr:
            :param test_case:
            :param phase:
            :param options:
            :param seed:
        """
        assert self.robots is not None, 'no robots loaded'

        te_done = True
        if te is not None:
            for d in te:
                te_done = d and te_done

        local_reset = False
        if tr is not None:
            reset_id = [i for i, (t, e) in enumerate(zip(tr, te)) if t or e]
            # reset_id = np.where(tr == 1 or te == 1)[0]
            if len(reset_id) > 0:
                local_reset = True
        else:
            reset_id = []

        if local_reset and not te_done:
            assert len(tr) == self.robots_num
            for idx in reset_id:
                self.reset_robot(idx)

        elif te_done:
            # reset scene
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # 构建场景时禁用渲染

            p.resetSimulation(physicsClientId=self._physics_client_id)
            p.setGravity(0., 0., -9.8, physicsClientId=self._physics_client_id)
            p.setRealTimeSimulation(0, physicsClientId=self._physics_client_id)
            p.setTimeStep(self.time_step, physicsClientId=self._physics_client_id)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            plane_id = p.loadURDF("plane.urdf", physicsClientId=self._physics_client_id)
            p.changeVisualShape(plane_id, -1, 
                rgbaColor=[255, 255, 255, 1],
                specularColor=[0, 0, 0],  # 关闭镜面反射
                textureUniqueId=-1)

            robot_num = self.env_args.robots_num

            self.collision_num = 0
            self.robots_num = 0
            self.simulate_steps = 0
            self.reach_num = 0
            self.global_time = 0

            init_state = self.init_state.copy()
            init_goal = self.init_goal.copy()

            self.init_state.clear()
            self.init_goal.clear()
            self.robots.clear()
            self.obstacles.clear()

            if self.boundary:
                self.add_boundary(self.boundary, self.boundary + 4)

            if self.layout_obstacle_specs:
                self._spawn_layout_obstacles()
            else:
                self.random_place_obstacle(self.random_obstacles)

            # self.load_and_place_obstacles()
            # self.add_loose_obstacle()

            # reload robot
            if not self.env_args.test_mode:
                for i in range(robot_num):
                    self.add_random_robot()
            else:
                for i in range(robot_num):
                    self.add_robot(init_state[i], init_goal[i])
                    if i < len(self.layout_robot_colors):
                        self.robots[-1].set_color(self.layout_robot_colors[i])

            if self.env_args.render:
                self.show_goal_point()

        current_state = []
        for i in range(self.robots_num):
            current_state.append(self.robots[i].get_observation())

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)  # 构建完成后启用渲染
        # self.save_obstacles_to_json()
        return np.array(current_state), dict({'reset': True})

    def show_goal_point(self):
        p.addUserDebugPoints([[_g[0], _g[1], 0.5] for _g in self.init_goal], [[1, 0, 0]] * self.robots_num, 13,
                             physicsClientId=self._physics_client_id)
        for idx in range(self.robots_num):
            pos = [self.init_goal[idx][0], self.init_goal[idx][1], 0.5]
            p.addUserDebugText('{}'.format(idx), pos, [255, 0, 0], 1.5, physicsClientId=self._physics_client_id)

    def render(self, mode='human'):
        pass
        
    def random_place_obstacle(self, batch_size=10):
        """批量随机放置障碍物，每个障碍物独立随机选择形状"""
        positions = self.random_points(batch_size)
        angles = np.random.uniform(-math.pi, math.pi, batch_size) if self.env_args.random_angle_obs else np.zeros(batch_size)
        assert self.robots_num == 0, '必须优先放置障碍物'

        # 批量创建物体
        for (x, y, _), angle in zip(positions, angles):
            shape_type = random.choice(list(self.shape_params.keys()))

            # 创建视觉和碰撞形状（统一处理）
            visual_params = {
                **self.shape_params[shape_type]["visual"],  # 继承基础参数
                "physicsClientId": self._physics_client_id  # 必须明确指定
            }
            visual_shape_id = p.createVisualShape(**visual_params)

            collision_params = {
                **self.shape_params[shape_type]["collision"],  # 继承基础参数
                "physicsClientId": self._physics_client_id  # 必须明确指定
            }
            collision_shape_id = p.createCollisionShape(**collision_params)

            if shape_type == "CAPSULE":
                # 胶囊体默认是直立的，我们需要将其旋转90度使其倒下
                # 绕x轴旋转90度（pi/2弧度）
                ori = p.getQuaternionFromEuler([math.pi/2, 0, angle], self.physics_client_id)
                # 调整高度，因为倒下后高度会变小
            else:
                ori = p.getQuaternionFromEuler([0, 0, angle], self.physics_client_id)

            # 创建物体
            body_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_shape_id,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=[x, y, self.shape_params[shape_type]["height"]],
                baseOrientation=ori,
                physicsClientId=self._physics_client_id
            )
            self.obstacles.append([x, y, angle, body_id, shape_type])

    def add_loose_obstacle(self):
        ori = p.getQuaternionFromEuler([0, 0, 0], self.physics_client_id)
        cube_id = p.loadURDF("cube.urdf", [10, 10, 0.5], baseOrientation=ori, physicsClientId=self._physics_client_id,
                   useFixedBase=0, useMaximalCoordinates=1)
        p.changeVisualShape(cube_id, -1, rgbaColor=[0.5, 0, 0, 1], 
                   physicsClientId=self._physics_client_id)

    def close(self):
        if self._physics_client_id >= 0:
            p.disconnect(physicsClientId=self._physics_client_id)
        self._physics_client_id = -1

    def add_boundary(self, length=20, width=20):
        # 设置方块大小（缩小为0.5x0.5x0.5）
        half_size = 0.25  # 边长的一半为0.25，所以边长为0.5
        spacing = 0.5     # 方块中心之间的间距（等于边长）
        
        # 创建黑色立方体视觉形状
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[half_size, half_size, half_size],  # 缩小方块尺寸
            rgbaColor=[0, 0, 0, 1],       # 黑色 (R,G,B,Alpha)
            physicsClientId=self._physics_client_id
        )
        
        # 创建碰撞形状
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[half_size, half_size, half_size],  # 缩小方块尺寸
            physicsClientId=self._physics_client_id
        )
        
        # 计算边界起始位置，确保边界居中
        start_x = -length/2 * spacing
        start_y = -width/2 * spacing
        
        # 上边界和下边界
        for i in range(length):
            # 计算x坐标（考虑间距）
            x = start_x + i * spacing
            
            # 上边界位置
            y_top = width/2 * spacing
            self.obstacles.append([x, y_top, 0])
            
            # 下边界位置
            y_bottom = -width/2 * spacing
            self.obstacles.append([x, y_bottom, 0])
        
        # 左边界和右边界
        for i in range(width):
            # 计算y坐标（考虑间距）
            y = start_y + i * spacing
            
            # 右边界位置
            x_right = length/2 * spacing
            self.obstacles.append([x_right, y, 0])
            
            # 左边界位置
            x_left = -length/2 * spacing
            self.obstacles.append([x_left, y, 0])
        
        # 一次性创建所有边界方块
        for pos in self.obstacles:
            p.createMultiBody(
                baseMass=0,  # 质量为0表示静态物体
                baseCollisionShapeIndex=collision_shape_id,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=pos,
                physicsClientId=self._physics_client_id
            )

    @property
    def physics_client_id(self):
        return self._physics_client_id

    def create_env(self):
        """
        :return: env class
        """
        self.layout_obstacle_specs = []
        self.layout_robot_colors = []

        if self.name == 'base':
            if self.env_args.random_robot:
                for i in range(self.env_args.robot_nums):
                    self.add_random_robot()
            else:
                pos = [[2.5, -2.5], [-9, 9], [-9, -2.5], [5.5, -5], [6.5, 7], [0, -7], [9, 0], [-9, -0.5]]
                angle = [np.pi * 3 / 4, -np.pi / 4, np.pi /4, np.pi * 3 / 4, -np.pi * 3 / 4, np.pi/2, np.pi, 0]
                goal = [[-5, 9], [-1, -2.5], [5, 9], [-4, 10], [-9, -5], [4, 10], [-9, -2], [9, 0]]

        elif self.name == 'opposite':
            pos_1 = np.stack([np.linspace(-self.env_args.x_range, self.env_args.x_range, self.env_args.robot_nums),
                              self.env_args.y_range * np.ones(self.env_args.robot_nums)]).T
            pos_2 = np.stack([np.linspace(-self.env_args.x_range, self.env_args.x_range, self.env_args.robot_nums),
                              -self.env_args.y_range * np.ones(self.env_args.robot_nums)]).T
            pos = np.vstack([pos_1, pos_2])
            angle = [np.pi * 3 / 2] * self.env_args.robot_nums + [np.pi / 2] * self.env_args.robot_nums
            goal = np.vstack([pos_2, pos_1])
            self.env_args.robots_num *= 2

        elif self.name == 'transverse':
            pos_1 = np.stack([np.linspace(-self.env_args.x_range, self.env_args.x_range, self.env_args.robot_nums),
                              (self.env_args.y_range + 1) * np.ones(self.env_args.robot_nums)]).T
            pos_2 = np.stack([(self.env_args.x_range + 1) * np.ones(self.env_args.robot_nums),
                              np.linspace(-self.env_args.y_range, self.env_args.y_range, self.env_args.robot_nums)]).T
            pos = np.vstack([pos_1, pos_2])
            angle = [np.pi * 3 / 2] * self.env_args.robot_nums + [np.pi] * self.env_args.robot_nums

            goal_1 = np.stack([np.linspace(-self.env_args.x_range, self.env_args.x_range, self.env_args.robot_nums),
                               -(self.env_args.y_range + 1) * np.ones(self.env_args.robot_nums)]).T
            goal_2 = np.stack([-(self.env_args.x_range + 1) * np.ones(self.env_args.robot_nums),
                               np.linspace(-self.env_args.y_range, self.env_args.y_range, self.env_args.robot_nums)]).T
            goal = np.vstack([goal_1, goal_2])
            self.env_args.robots_num *= 2

        elif self.name == 'circle':
            self.name = 'circle'

            theta = np.linspace(0, 2 * np.pi, self.env_args.robots_num + 1)

            pos = np.stack([self.env_args.radius * np.cos(theta), self.env_args.radius * np.sin(theta)]).T
            goal = np.stack(
                [self.env_args.radius * np.cos(theta + np.pi), self.env_args.radius * np.sin(theta + np.pi)]).T

            angle = [a + np.pi for a in theta]

        elif self.name in ('u_shape', 'u'):
            inner_width = float(self._get_env_setting('u_inner_width', 5.0))
            inner_depth = float(self._get_env_setting('u_inner_depth', 6.0))
            wall_thickness = float(self._get_env_setting('u_wall_thickness', 0.6))
            goal_offset = float(self._get_env_setting('u_goal_offset', 2.5))
            safe_margin = max(0.9, wall_thickness + ROBOT_WIDTH * 2)

            left_wall_x = -(inner_width / 2 + wall_thickness / 2)
            right_wall_x = inner_width / 2 + wall_thickness / 2
            bottom_wall_y = -(inner_depth / 2 + wall_thickness / 2)

            self._add_layout_box_spec(left_wall_x, -wall_thickness / 2, wall_thickness, inner_depth + wall_thickness)
            self._add_layout_box_spec(right_wall_x, -wall_thickness / 2, wall_thickness, inner_depth + wall_thickness)
            self._add_layout_box_spec(0.0, bottom_wall_y, inner_width + 2 * wall_thickness, wall_thickness)

            pos = self._generate_points_in_rect(
                center_x=0.0,
                center_y=-0.8,
                width=max(inner_width - 0.6, 1.5),
                height=max(inner_depth - 1.8, 1.5),
                count=self.env_args.robots_num,
                margin=safe_margin,
            )
            goal = self._generate_points_in_rect(
                center_x=0.0,
                center_y=inner_depth / 2 + goal_offset,
                width=inner_width + 2.0,
                height=2.0,
                count=self.env_args.robots_num,
                margin=0.8,
            )
            angle = self._angles_from_pairs(pos, goal)

        elif self.name in ('dumbbell', 'barbell'):
            room_size = float(self._get_env_setting('dumbbell_room_size', 10.0))
            corridor_length = float(self._get_env_setting('dumbbell_corridor_length', 10.0))
            corridor_width = float(self._get_env_setting('dumbbell_corridor_width', 3))
            wall_thickness = float(self._get_env_setting('dumbbell_wall_thickness', 0.2))
            safe_margin = max(1.8, wall_thickness + ROBOT_WIDTH * 2 + 0.5)

            room_offset_x = room_size / 2 + corridor_length / 2
            left_center_x = -room_offset_x
            right_center_x = room_offset_x
            outer_wall_span = room_size + 2 * wall_thickness
            inner_wall_x_offset = room_size / 2 + wall_thickness / 2
            door_segment_height = max((room_size - corridor_width) / 2, 0.0)
            door_segment_center_y = corridor_width / 2 + door_segment_height / 2

            self._add_layout_box_spec(left_center_x, room_size / 2 + wall_thickness / 2, outer_wall_span, wall_thickness)
            self._add_layout_box_spec(left_center_x, -(room_size / 2 + wall_thickness / 2), outer_wall_span, wall_thickness)
            self._add_layout_box_spec(left_center_x - inner_wall_x_offset, 0.0, wall_thickness, outer_wall_span)

            self._add_layout_box_spec(right_center_x, room_size / 2 + wall_thickness / 2, outer_wall_span, wall_thickness)
            self._add_layout_box_spec(right_center_x, -(room_size / 2 + wall_thickness / 2), outer_wall_span, wall_thickness)
            self._add_layout_box_spec(right_center_x + inner_wall_x_offset, 0.0, wall_thickness, outer_wall_span)

            if door_segment_height > 0:
                self._add_layout_box_spec(left_center_x + inner_wall_x_offset, door_segment_center_y, wall_thickness, door_segment_height)
                self._add_layout_box_spec(left_center_x + inner_wall_x_offset, -door_segment_center_y, wall_thickness, door_segment_height)
                self._add_layout_box_spec(right_center_x - inner_wall_x_offset, door_segment_center_y, wall_thickness, door_segment_height)
                self._add_layout_box_spec(right_center_x - inner_wall_x_offset, -door_segment_center_y, wall_thickness, door_segment_height)

            self._add_layout_box_spec(0.0, corridor_width / 2 + wall_thickness / 2, corridor_length, wall_thickness)
            self._add_layout_box_spec(0.0, -(corridor_width / 2 + wall_thickness / 2), corridor_length, wall_thickness)

            left_count = (self.env_args.robots_num + 1) // 2
            right_count = self.env_args.robots_num - left_count

            left_pos = self._generate_points_in_rect(left_center_x, 0.0, room_size, room_size, left_count, margin=safe_margin)
            right_pos = self._generate_points_in_rect(right_center_x, 0.0, room_size, room_size, right_count, margin=safe_margin)
            room_shift = right_center_x - left_center_x

            left_goal = [[x + room_shift, y] for x, y in left_pos]
            right_goal = [[x - room_shift, y] for x, y in right_pos]

            pos = left_pos + right_pos
            goal = left_goal + right_goal
            angle = [0.0] * len(left_pos) + [np.pi] * len(right_pos)
            self.layout_robot_colors = (
                [[0.22, 0.55, 0.92, 1.0]] * len(left_pos) +
                [[0.94, 0.48, 0.20, 1.0]] * len(right_pos)
            )

        elif self.name in ('room', 'room_like'):
            outer_width = float(self._get_env_setting('room_outer_width', 14.0))
            outer_height = float(self._get_env_setting('room_outer_height', 10.0))
            wall_thickness = float(self._get_env_setting('room_wall_thickness', 0.6))
            main_partition_x = float(self._get_env_setting('room_main_partition_x', -1.0))
            main_door_width = float(self._get_env_setting('room_main_door_width', 1.8))
            main_door_y = float(self._get_env_setting('room_main_door_y', -1.8))
            side_partition_y = float(self._get_env_setting('room_side_partition_y', 1.2))
            side_door_width = float(self._get_env_setting('room_side_door_width', 1.6))
            side_door_x = float(self._get_env_setting('room_side_door_x', 3.2))
            safe_margin = max(0.9, wall_thickness + ROBOT_WIDTH * 2)

            half_w = outer_width / 2
            half_h = outer_height / 2

            self._add_layout_box_spec(0.0, half_h + wall_thickness / 2, outer_width + 2 * wall_thickness, wall_thickness)
            self._add_layout_box_spec(0.0, -(half_h + wall_thickness / 2), outer_width + 2 * wall_thickness, wall_thickness)
            self._add_layout_box_spec(-(half_w + wall_thickness / 2), 0.0, wall_thickness, outer_height + 2 * wall_thickness)
            self._add_layout_box_spec(half_w + wall_thickness / 2, 0.0, wall_thickness, outer_height + 2 * wall_thickness)

            lower_main_height = max((main_door_y - main_door_width / 2) - (-half_h), 0.0)
            upper_main_height = max(half_h - (main_door_y + main_door_width / 2), 0.0)
            if lower_main_height > 0:
                self._add_layout_box_spec(
                    main_partition_x,
                    (-half_h + (main_door_y - main_door_width / 2)) / 2,
                    wall_thickness,
                    lower_main_height,
                )
            if upper_main_height > 0:
                self._add_layout_box_spec(
                    main_partition_x,
                    ((main_door_y + main_door_width / 2) + half_h) / 2,
                    wall_thickness,
                    upper_main_height,
                )

            left_side_width = max((side_door_x - side_door_width / 2) - main_partition_x, 0.0)
            right_side_width = max(half_w - (side_door_x + side_door_width / 2), 0.0)
            if left_side_width > 0:
                self._add_layout_box_spec(
                    (main_partition_x + (side_door_x - side_door_width / 2)) / 2,
                    side_partition_y,
                    left_side_width,
                    wall_thickness,
                )
            if right_side_width > 0:
                self._add_layout_box_spec(
                    ((side_door_x + side_door_width / 2) + half_w) / 2,
                    side_partition_y,
                    right_side_width,
                    wall_thickness,
                )

            pos = self._generate_points_in_rect(
                center_x=(main_partition_x - half_w) / 2,
                center_y=0.0,
                width=max(main_partition_x + half_w, 1.5),
                height=outer_height,
                count=self.env_args.robots_num,
                margin=safe_margin,
            )
            goal = self._generate_points_in_rect(
                center_x=(main_partition_x + half_w) / 2,
                center_y=(side_partition_y + half_h) / 2,
                width=max(half_w - main_partition_x, 1.5),
                height=max(half_h - side_partition_y, 1.5),
                count=self.env_args.robots_num,
                margin=safe_margin,
            )
            angle = self._angles_from_pairs(pos, goal)

        else:
            self.name = 'defult'
            pos = [[0, -3]]
            goal = [[0, 2.5]]
            angle = [np.pi / 2]

        if self.layout_obstacle_specs:
            self.random_obstacles = len(self.layout_obstacle_specs)

        # if self.env_args.random_angle:
        #     angle = np.random.uniform(low=-np.pi, high=np.pi, size=(len(angle), 2))

        for i in range(self.env_args.robots_num):
            ori = p.getQuaternionFromEuler([0, 0, angle[i]], self.physics_client_id)
            state = list(pos[i]) + [BIAS_ROBOT_LOCATION] + list(ori)
            self.init_state.append(state)
            self.init_goal.append(goal[i])
            # self.add_robot(state, goal[i])

    def save_obstacles_to_json(self, file_path='./obstacle/1.json'):
        """将当前环境障碍物保存为JSON文件"""
        obstacles_data = []
        for obs in self.obstacles:
            # 确保数据类型可序列化
            obstacle_dict = {
                "x": float(obs[0]),
                "y": float(obs[1]),
                "angle": float(obs[2]),
                "body_id": int(obs[3]),
                "shape_type": obs[4]
            }
            obstacles_data.append(obstacle_dict)
        
        try:
            with open(file_path, 'w') as f:
                json.dump(obstacles_data, f, indent=4)
            print(f"障碍物信息已保存至: {file_path}")
        except Exception as e:
            print(f"保存失败: {str(e)}")

    def load_and_place_obstacles(self, file_path='./obstacle/1.json'):
        """
        从JSON文件读取障碍物信息并放置到环境中
        
        参数:
            file_path: JSON文件路径
        """
        try:
            with open(file_path, 'r') as f:
                obstacles_data = json.load(f)
            
            print(f"从 {file_path} 加载了 {len(obstacles_data)} 个障碍物")
            
            # 清空当前障碍物
            self.clear_obstacles()
            
            # 放置每个障碍物
            for obs in obstacles_data:
                x = obs["x"]
                y = obs["y"]
                angle = obs["angle"]
                shape_type = obs["shape_type"]
                
                # 验证形状类型是否有效
                if shape_type not in self.shape_params:
                    print(f"警告: 无效的形状类型 '{shape_type}'，使用默认BOX代替")
                    shape_type = "BOX"
                
                # 创建视觉和碰撞形状
                visual_params = {
                    **self.shape_params[shape_type]["visual"],
                    "physicsClientId": self._physics_client_id
                }
                visual_shape_id = p.createVisualShape(**visual_params)
                
                collision_params = {
                    **self.shape_params[shape_type]["collision"],
                    "physicsClientId": self._physics_client_id
                }
                collision_shape_id = p.createCollisionShape(**collision_params)
                
                # 处理胶囊体的特殊旋转
                if shape_type == "CAPSULE":
                    # 胶囊体需要绕x轴旋转90度使其倒下
                    ori = p.getQuaternionFromEuler([math.pi/2, 0, angle], self._physics_client_id)
                else:
                    ori = p.getQuaternionFromEuler([0, 0, angle], self._physics_client_id)
                
                # 创建物体
                body_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=collision_shape_id,
                    baseVisualShapeIndex=visual_shape_id,
                    basePosition=[x, y, self.shape_params[shape_type]["height"]],
                    baseOrientation=ori,
                    physicsClientId=self._physics_client_id
                )
                
                # 添加到障碍物列表
                self.obstacles.append([x, y, angle, body_id, shape_type])
           
            return True
        
        except FileNotFoundError:
            print(f"错误: 文件 {file_path} 不存在")
            return False
        except json.JSONDecodeError:
            print(f"错误: 文件 {file_path} 不是有效的JSON格式")
            return False
        except KeyError as e:
            print(f"错误: JSON数据缺少必要字段 {str(e)}")
            return False
        except Exception as e:
            print(f"加载障碍物时发生错误: {str(e)}")
            return False

    def clear_obstacles(self):
        """清除当前环境中的所有障碍物"""
        # 先移除物理引擎中的物体
        for obs in self.obstacles:
            body_id = obs[3]
            p.removeBody(body_id, physicsClientId=self._physics_client_id)
        
        # 清空障碍物列表
        self.obstacles = []
        print("已清除所有障碍物")

def main():
    env = MyEnv(render=True)

    robot_nums = 2
    random_robot = 5

    for i in range(robot_nums):
        goal = [i, 1 + i]

        yaw = np.pi * robot_nums
        ori = p.getQuaternionFromEuler([0, 0, yaw], env.physics_client_id)
        state = [i, 0, 0.01] + list(ori)
        env.add_robot(state, goal)
        # env.add_random_robot(random_robot=random_robot)
    env.show_goal_point()

    p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=0, cameraPitch=-89.9,
                                 cameraTargetPosition=[0, 0, 0])
    # createBoundaries(10, 10)

    env.reset()

    velocity = np.zeros([env.robots_num, 2])
    while True:
        for i, rob in enumerate(env.robots):
            vel = rob.goto(rob.target_pos)
            velocity[i] = vel
        states, reward, te, tr, info = env.step(velocity)
        print(states)


if __name__ == '__main__':
    # env_ = gym.make('MyEnv-v0')
    main()
    # print(1)

