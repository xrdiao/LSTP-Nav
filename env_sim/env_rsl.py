import pybullet_data
import torch
from pathlib import Path
from env_sim.my_env import MyEnv
import numpy as np
import pybullet as p
from env_sim.argument import *

try:
    from project_paths import TURTLEBOT_URDF_PATH
except ImportError:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from project_paths import TURTLEBOT_URDF_PATH

class RENV(MyEnv):

    def __init__(self, env_args, urdf_path=None):
        super().__init__(env_args, urdf_path or str(TURTLEBOT_URDF_PATH))
        self.device = 'cuda'
        self.num_privileged_obs = None

    @staticmethod
    def convert_action_for_env(action):
        # action = copy.deepcopy(a)
        action = torch.tanh(action) * 1.001 
        # 线性映射到实际环境范围
        action[:, 0] = action[:, 0] * 0.5 * MAX_SPEED + 0.5 * MAX_SPEED  # 速度映射到[0, MAX_SPEED]
        action[:, 1] = action[:, 1] * MAX_ROTATION_SPEED  # 角速度映射到[-MAX_ROTATION_SPEED, MAX_ROTATION_SPEED]

        return action
    
    def step(self, actions):
        """
        :param actions: [velocity, angular_vel]
        :return:
        """
        assert self.robots is not None, 'no robots loaded'
        assert self.robots_num == len(actions), 'incorrect number of the actions'
        env_action = self.convert_action_for_env(actions).cpu().numpy()

        self.global_time += self.time_step
        self.simulate_steps += 1

        # 更新 t 时刻机器人距终点的距离，计算 t+1 和 t 时刻间的变化量
        for i, rob in enumerate(self.robots):
            rob.apply_action(env_action[i])

        for _ in range(self.FPS):
            p.stepSimulation(physicsClientId=self._physics_client_id)

        # 收集 t+1 时刻机器人的观测量，计算奖励，
        states = []
        for i in range(self.robots_num):
            states.append(self.robots[i].get_observation())
        reward, te, tr = self._reward_func(states)

        done = torch.logical_or(
            torch.tensor(te, device=self.device), 
            torch.tensor(tr, device=self.device)
        ).to(self.device)
        info = {"collision_num": self.collision_num}

        need_reset = any(tr) or any(te)
        states = self.reset(tr=tr, te=te)[0] if need_reset else states
        return torch.tensor(states, device=self.device, dtype=torch.float32), torch.tensor(reward, device=self.device, dtype=torch.float32), done, info
    
    def get_observations(self):
        states = [self.robots[i].get_observation() for i in range(self.robots_num)]
        return torch.tensor(states, device=self.device, dtype=torch.float32)
    
    def reset(self, tr: list = None, te: list = None):
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
            p.resetSimulation(physicsClientId=self._physics_client_id)
            p.setGravity(0., 0., -9.8, physicsClientId=self._physics_client_id)
            p.setRealTimeSimulation(0, physicsClientId=self._physics_client_id)
            p.setTimeStep(self.time_step, physicsClientId=self._physics_client_id)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.loadURDF("plane.urdf", physicsClientId=self._physics_client_id)

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

            obst_num = np.random.randint(low=int(self.random_obstacles/2), high=self.random_obstacles) if self.random_obstacles else 0
            self.random_place_cubes(obst_num)

            # reload robot
            if not self.env_args.test_mode:
                for i in range(robot_num):
                    self.add_random_robot()
            else:
                for i in range(robot_num):
                    self.add_robot(init_state[i], init_goal[i])

            if self.env_args.render:
                self.show_goal_point()

        return self.get_observations(), dict({'reset': True})
