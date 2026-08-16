import argparse
import copy
from distutils.util import strtobool

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
import torch.nn.init as init

from env_sim.argument import LASER_NUM, ROBOT_LASER_BUFFER, MAX_ROTATION_SPEED, MAX_SPEED
from rl.memory import Memory


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='elu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                init.kaiming_normal_(param.data, mode='fan_in', nonlinearity='elu')
            elif 'weight_hh' in name:
                init.orthogonal_(param.data, gain=1.0)  # RNN 隐藏状态用正交初始化
            elif 'bias' in name:
                init.constant_(param.data, 0)


class BaseAgent(nn.Module):
    def __init__(self):
        super(BaseAgent, self).__init__()
        self.actor_logstd = nn.Parameter(torch.zeros(1, 2))  # action_dim需根据实际情况定义

    def init_layer(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                nn.init.orthogonal_(m.weight, np.sqrt(2))

    def get_action(self, observations):
        return torch.zeros_like(observations)

    def get_value(self, observations):
        return torch.zeros_like(observations)

    def get_deterministic_action(self, observations):
        action = self.get_action(observations)
        value = self.get_value(observations)
        return action, value

    def get_action_and_value(self, observations, action=None, epsilon=0.0, **kwargs):
        action_mean = self.get_action(observations)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        probs = Normal(action_mean, action_std)
        if action is None:
            if np.random.rand() < epsilon:
                action = torch.rand_like(action_mean)
            else:
                action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.get_value(observations)


    def _build_mlp(self, input_dim, hidden_dims, output_dim, activation):
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(activation)
        for i in range(len(hidden_dims)):
            if i == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[i], output_dim))
            else:
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                layers.append(activation)
        return nn.Sequential(*layers)
    
    def reset(self, dones=None):
        pass

    @staticmethod
    def convert_action_for_env(action):
        # action = copy.deepcopy(a)
        action = torch.tanh(action) * 1.001 
        # 线性映射到实际环境范围
        action[:, 0] = action[:, 0] * 0.5 * MAX_SPEED + 0.5 * MAX_SPEED  # 速度映射到[0, MAX_SPEED]
        action[:, 1] = action[:, 1] * MAX_ROTATION_SPEED  # 角速度映射到[-MAX_ROTATION_SPEED, MAX_ROTATION_SPEED]
        return action


class RandomAgent(BaseAgent):
    def __init__(self):
        super(RandomAgent, self).__init__()

    def get_deterministic_action(self, observation):
        action = torch.normal(0.5, 1, size=[observation.shape[0], 2])
        return action, None


class IJRRAgent(BaseAgent):
    def __init__(self):
        super(IJRRAgent, self).__init__()
        # assert LASER_NUM == 512, "LASER_NUM should be 512"
        self.laser_encode = nn.Sequential(nn.Conv1d(1, 32, 5, 2), nn.ReLU(), nn.Conv1d(32, 32, 3, 2),
                                          nn.ReLU(), nn.Flatten(), nn.Linear(992, 256))

        self.actor = nn.Sequential(nn.Linear(260, 128), nn.ReLU(), nn.Linear(128, 2))
        self.critic = nn.Sequential(nn.Linear(260, 128), nn.ReLU(), nn.Linear(128, 1))

        self.actor_logstd = nn.Parameter(torch.zeros(1, 2))
        self.name = 'Agent_IJRR'

    def forward(self, observations):
        laser = observations[:, 2:-2].unsqueeze(1)  # (batch_size, ROBOT_LASER_BUFFER, LASER_NUM)
        state = torch.cat([observations[:, :2], observations[:,-2:]], dim=1)  # goal and state

        laser = self.laser_encode(laser)
        data = torch.cat((laser, state), 1)
        return data

    def get_value(self, observations):
        data = self.forward(observations)
        v = self.critic(data)
        return v

    def get_action(self, observations):
        data = self.forward(observations)
        action = self.actor(data)
        return action


class Conv1dAgent(BaseAgent):
    def __init__(self):
        super(Conv1dAgent, self).__init__()
        assert LASER_NUM == 512, "LASER_NUM should be 512"
        self.hidden_dim = 256

        self.laser = nn.Sequential(nn.Conv1d(ROBOT_LASER_BUFFER, 32, 5, 2), nn.ReLU(), nn.Conv1d(32, 32, 3, 2),
                                   nn.ReLU(), nn.Flatten(), nn.Linear(4032, self.hidden_dim))

        self.att = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=1, batch_first=True)
        self.q = nn.Sequential(nn.Linear(4, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.hidden_dim))
        self.k = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
                               nn.Linear(self.hidden_dim, self.hidden_dim))
        self.v = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim))

        self.critic = nn.Sequential(nn.Linear(self.hidden_dim + 4, int(self.hidden_dim / 2)), nn.ReLU(),
                                    nn.Linear(int(self.hidden_dim / 2), 1))
        self.actor = nn.Sequential(nn.Linear(self.hidden_dim + 4, int(self.hidden_dim / 2)), nn.ReLU(),
                                   nn.Linear(int(self.hidden_dim / 2), 2))
        self.actor_logstd = nn.Parameter(torch.zeros(1, 2))

        # self.init_layer()
        self.name = 'Agent_conv'

    def forward(self, laser, goal):
        laser = self.laser(laser)

        q = self.q(goal)
        k = self.k(laser)
        v = self.v(laser)
        laser, emb_weights = self.att(q, k, v)

        data = torch.cat((laser, goal), 1)
        return data

    def get_value(self, laser, goal):
        data = self.forward(laser, goal)
        v = self.critic(data)
        return v

    def get_action(self, laser, goal):
        data = self.forward(laser, goal)
        action = self.actor(data)
        return action
    

class LinearAgent(BaseAgent):
    def __init__(self):
        super(LinearAgent, self).__init__()
        self.hidden_dim = 256
        mlp_input_dim_a = 512
        mlp_input_dim_c = 512
        num_actions = 2
        num_obs = LASER_NUM + 4

        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation=nn.ELU()

        self.goal = nn.Sequential(nn.Linear(2, self.hidden_dim), nn.ELU(),
                                   )
        self.goal_1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.state = nn.Sequential(nn.Linear(LASER_NUM+2, self.hidden_dim), nn.ELU(),
                                   )
        self.state_1 = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # self.init_layer()

        self.name = 'Agent_Linear'

    def forward(self, observations):
        goal = self.goal(observations[:,:2])
        goal1 = self.goal_1(goal) + goal
        state = self.state(observations[:,2:])
        state1 = self.state_1(goal) + state

        data = torch.cat((goal1, state1), 1)
        return data

    def get_value(self, observations):
        data = self.forward(observations)
        v = self.critic(data)
        return v

    def get_action(self, observations):
        data = self.forward(observations)
        action = self.actor(data)
        return action


class LstmAgent(BaseAgent):
    def __init__(self):
        super(LstmAgent, self).__init__()
        self.hidden_dim = 252
        input_size = LASER_NUM
        rnn_type = 'lstm'
        rnn_num_layers = 2

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_dim, num_layers=rnn_num_layers, batch_first=True)
        self.state_q_critic = nn.Sequential(nn.Linear(256, 128), nn.ReLU(),
                                            nn.Linear(128, 1))
        self.state_q_actor = nn.Sequential(nn.Linear(256, 128), nn.ReLU(),
                                           nn.Linear(128, 2))

        self.actor_logstd = nn.Parameter(torch.zeros(1, 2))
        self.memory = Memory(input_size, type=rnn_type, num_layers=rnn_num_layers, hidden_size=self.hidden_dim)
        self.name = 'Agent_Lstm'

    def forward(self, observations, masks=None, hidden_states=None):
        # state = (observations[:, :2])

        laser = observations[:, 2:-2]  
        laser, _ = self.memory(laser, masks, hidden_states)
        laser = laser.sum(1)  # 适合静态
        # laser = laser[:, -1, :]  # 适合动态
        emb = torch.cat([laser, observations[:, :2], observations[:,-2:]], dim=1)
        return emb

    def get_value(self, observations):
        emb = self.forward(observations)
        value = self.state_q_critic(emb)
        return value

    def get_action(self, observations):
        emb = self.forward(observations)
        action = self.state_q_actor(emb)
        return action
    
    def reset(self, dones=None):
        self.memory.reset(dones)

    def get_hidden_states(self):
        return self.memory.hidden_states

class AttentionAgent(BaseAgent):
    def __init__(self):
        super(AttentionAgent, self).__init__()
        self.hidden_dim = 256
        mlp_input_dim_a = 512
        mlp_input_dim_c = 512
        num_actions = 2
        num_obs = LASER_NUM + 4

        # 新增障碍物处理参数
        self.laser_dim = LASER_NUM  # 激光雷达维度
        self.obstacle_dim = 256     # 障碍物特征维度

        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = nn.ELU()

        self.obstacle_encoder = nn.Sequential(
            nn.Linear(self.laser_dim, 128),
            nn.ELU(),
            nn.Linear(128, self.obstacle_dim),
        )

        self.velocity_encoder = nn.Sequential(
            nn.Linear(2, 128),  # 假设速度是2维
            nn.ELU(),
            nn.Linear(128, self.obstacle_dim),
        )

        # 目标处理保持不变
        self.goal = nn.Sequential(
            nn.Linear(2, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, 2*(self.hidden_dim))
        )
        
        # ==== 改进2：双路径注意力机制 ====
        # 路径1：目标-状态交叉注意力
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=2*(self.hidden_dim),
            num_heads=4,
            batch_first=False
        )

        # 策略网络（输入维度保持512）
        self.actor = self._build_mlp(mlp_input_dim_a, actor_hidden_dims, num_actions, activation)
        self.critic = self._build_mlp(mlp_input_dim_c, critic_hidden_dims, 1, activation)

        self.name = 'AttentionAgent'

    def forward(self, observations):
        # 分解输入
        goal_input = observations[:, :2]  # 目标位置
        laser_input = observations[:, 4:]  # 激光雷达
        velocity_input = observations[:, 2:4]  # 速度信息
        
        # 目标处理
        goal_proj = self.goal(goal_input)
        obstacle = self.obstacle_encoder(laser_input)  
        velocity = self.velocity_encoder(velocity_input) 
        
        # ==== 路径1：目标-状态交叉注意力 ====
        # 使用目标查询，状态=障碍物+速度
        state_features = torch.cat([obstacle, velocity], dim=1)

        # 交叉注意力
        goal_query = goal_proj.unsqueeze(0)  
        state_kv = state_features.unsqueeze(0)  
        attn_output, _ = self.cross_attn(goal_query, state_kv, state_kv)
        attn_output = attn_output.squeeze(0)

        # 合并特征
        combined = goal_proj + attn_output
        return combined

    def get_value(self, observations):
        data = self.forward(observations)
        return self.critic(data)

    def get_action(self, observations):
        data = self.forward(observations)
        return self.actor(data)

class LagAgent(AttentionAgent):
    def __init__(self):
        super(LagAgent, self).__init__()
        self.decode_cost = nn.Sequential(nn.Linear(self.hidden_dim + 4, self.hidden_dim), nn.ReLU(),
                                         nn.Linear(self.hidden_dim, 1))

        self.init_layer()
        self.name = 'Agent_Lag'

    def get_cost(self, laser, state):
        laser = self.forward(laser, state)
        v_c = self.decode_cost(laser)
        return v_c

    def get_value(self, laser, state):
        v_c = self.get_cost(copy.deepcopy(laser), state)
        laser = self.forward(laser, state)
        v_r = self.decode_critic(laser)
        return v_r, v_c

    def get_action(self, laser, state):
        laser = self.forward(laser, state)
        action_mean = self.decode_actor(laser)
        return action_mean

    def get_deterministic_action(self, laser, state):
        action = self.get_action(laser, state)
        v_r, v_c = self.get_value(laser, state)
        return action, v_r, v_c

    def get_action_and_value(self, laser, state, action=None, epsilon=0.0):
        action_mean = self.get_action(laser, state)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        v_r, v_c = self.get_value(laser, state)

        probs = Normal(action_mean, action_std)
        if action is None:
            if np.random.rand() < epsilon:
                action = torch.rand_like(action_mean)
            else:
                action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), v_r, v_c


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", type=float, default=5e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=3000000,
                        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=1,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=256,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=8,
                        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=3,
                        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=1,
                        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=1,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--reward-scaling", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="relay buffer")
    parser.add_argument("--lagrangian-multiplier-init", type=float, default=0.001, nargs="?", const=True,
                        help="initial value of lagrangian multiplier")
    parser.add_argument("--lagrangian-multiplier-lr", type=float, default=0.03, nargs="?", const=True,
                        help="learning rate of lagrangian multiplier")
    parser.add_argument("--cost-limit", type=float, default=25.0, nargs="?", const=True,
                        help="cost lim")
    parser.add_argument("--target-kl", type=float, default=0.01, nargs="?", const=True,
                    help="target kl")


    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Agent(BaseAgent):
    def __init__(self):
        super(Agent, self).__init__()
        self.conv = nn.Sequential(nn.Conv1d(ROBOT_LASER_BUFFER, 32, kernel_size=5, stride=2), nn.ReLU(),
                                  nn.Conv1d(32, 32, kernel_size=3, stride=2), nn.ReLU())
        self.linear = nn.Linear(4480, 256)

        self.fc1_critic = nn.Sequential(nn.Linear(256, 128), nn.ReLU(),
                                        nn.Linear(128, 1))
        self.fc1_actor = nn.Sequential(nn.Linear(256, 128), nn.ReLU(),
                                       nn.Linear(128, 2))

        self.actor_logstd = nn.Parameter(torch.zeros(1, 2))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                nn.init.orthogonal_(m.weight, np.sqrt(2))
            if isinstance(m, nn.Conv1d):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

        self.name = 'Agent'

    def get_value(self, laser, state):
        laser = self.conv(laser)
        laser = laser.flatten()
        laser = self.linear(laser)

        data = torch.cat([laser, state], dim=-1)
        v = self.fc1_critic(data)
        return v

    def get_action(self, laser, state):
        laser = self.conv(laser)
        laser = laser.flatten()
        laser = self.linear(laser)

        data = torch.cat([laser, state], dim=-1)
        action_mean = self.fc1_actor(data)
        return action_mean
