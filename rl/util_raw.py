import argparse
from distutils.util import strtobool
import torch.nn.functional as F
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from env_sim.argument import LASER_NUM, ROBOT_LASER_BUFFER, MAX_ROTATION_SPEED, MAX_SPEED


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class BaseAgent(nn.Module):
    def __init__(self):
        super(BaseAgent, self).__init__()

    def init_layer(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                nn.init.orthogonal_(m.weight, np.sqrt(2))
            if isinstance(m, nn.LSTM):
                for name, param in self.lstm_critic.named_parameters():
                    if "bias" in name:
                        nn.init.constant_(param, 0)
                    elif "weight" in name:
                        nn.init.orthogonal_(param, 1.0)
            if isinstance(m, nn.Conv1d):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def get_action(self, laser, goal):
        return torch.zeros_like(goal)

    def get_value(self, laser, goal):
        return torch.zeros_like(goal)

    def get_deterministic_action(self, laser, state):
        action = self.get_action(laser, state)
        value = self.get_value(laser, state)
        return action, value

    def get_action_and_value(self, laser, state, action=None, epsilon=0.0):
        action_mean = self.get_action(laser, state)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        probs = Normal(action_mean, action_std)
        if action is None:
            if np.random.rand() < epsilon:
                action = torch.rand_like(action_mean)
            else:
                action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.get_value(laser, state)

    @staticmethod
    def convert_action_for_env(action):
        # action = copy.deepcopy(a)
        action[:, 0] = torch.tanh(action[:, 0]) / 2 + 0.5 * MAX_SPEED
        action[:, 1] = torch.tanh(action[:, 1]) * MAX_ROTATION_SPEED
        return action
    
class RandomAgent(BaseAgent):
    def __init__(self):
        super(RandomAgent, self).__init__()

    def get_deterministic_action(self, laser, state):
        action = torch.normal(0.5, 1, size=[1, 2])
        return action, None

class AttentionAgent(BaseAgent):
    def __init__(self):
        super(AttentionAgent, self).__init__()
        self.hidden_dim = 256
        input_size = LASER_NUM

        # LSTM层
        self.lstm = nn.GRU(input_size=input_size, hidden_size=self.hidden_dim, 
                                  num_layers=2, batch_first=True)
        
        # 多头注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, 
            num_heads=4,
            batch_first=True
        )
        
        # 状态编码层（带残差连接）
        self.state_encoder = nn.Sequential(
            nn.Linear(4, self.hidden_dim), 
            nn.ELU(),
        )
        self.state_residual = nn.Linear(self.hidden_dim, self.hidden_dim)  # 残差连接

        # 动作输出层
        self.actor_mean = nn.Sequential(
            nn.Linear(self.hidden_dim*2, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim//2),
            nn.ELU(),
            nn.Linear(self.hidden_dim//2, 2)
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, 2))
        
        # 值函数输出层
        self.critic = nn.Sequential(
            nn.Linear(self.hidden_dim*2, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim//2),
            nn.ELU(),
            nn.Linear(self.hidden_dim//2, 1)
        )

        self.name = 'Agent_Lstm_Attn'

    def forward(self, laser, state):
        # LSTM处理激光序列 (batch, seq_len, features)
        lstm_out, _ = self.lstm(laser)  # shape: [B, T, D]
        
        # ===== 使用MultiheadAttention =====
        # 使用LSTM输出的最后一个时间步作为查询
        query = lstm_out[:, -1:, :]  # [B, 1, D]
        
        # 整个序列作为键和值
        key = value = lstm_out  # [B, T, D]
        
        # 计算注意力
        attn_output, _ = self.attention(
            query=query,
            key=key,
            value=value
        )
        context = attn_output.squeeze(1)  # [B, D]
        
        # 状态编码 (带残差连接)
        state_e = self.state_encoder(state)
        state_enc = self.state_residual(state_e) + state_e
        
        # 融合激光上下文和状态信息
        data = torch.cat((context, state_enc), dim=1)
        return data

    def get_value(self, laser, state):
        data = self.forward(laser, state)
        v = self.critic(data)
        return v

    def get_action(self, laser, state):
        data = self.forward(laser, state)
        action = self.actor_mean(data)
        return action

class IJRRAgent(BaseAgent):
    def __init__(self):
        super(IJRRAgent, self).__init__()
        # assert LASER_NUM == 512, "LASER_NUM should be 512"
        self.laser_encode = nn.Sequential(nn.Conv1d(5, 32, 5, 2), nn.ReLU(), nn.Conv1d(32, 32, 3, 2),
                                          nn.ReLU(), nn.Flatten(), nn.Linear(992, 256))

        self.actor = nn.Sequential(nn.Linear(260, 128), nn.ReLU(), nn.Linear(128, 2))
        self.critic = nn.Sequential(nn.Linear(260, 128), nn.ReLU(), nn.Linear(128, 1))

        self.actor_logstd = nn.Parameter(torch.zeros(1, 2))
        self.name = 'Agent_IJRR'

    def forward(self, laser, state):

        laser = self.laser_encode(laser)
        data = torch.cat((laser, state), 1)
        return data

    def get_value(self, laser, state):
        data = self.forward(laser, state)
        v = self.critic(data)
        return v

    def get_action(self, laser, state):
        data = self.forward(laser, state)
        action = self.actor(data)
        return action
    
class LstmAgent(BaseAgent):
    def __init__(self):
        super(LstmAgent, self).__init__()
        self.hidden_dim = 256
        input_size = LASER_NUM

        self.lstm = nn.GRU(input_size=input_size, hidden_size=self.hidden_dim, 
                                  num_layers=2, batch_first=True)
        self.lstm_norm = nn.LayerNorm(self.hidden_dim)

        self.state = nn.Sequential(nn.Linear(4, self.hidden_dim), nn.ELU(),
                                   )
        self.state_1 = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.actor_mean = nn.Sequential(
            nn.Linear(self.hidden_dim*2, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim//2),
            nn.ELU(),
            nn.Linear(self.hidden_dim//2, 2)
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, 2))
        
        self.critic = nn.Sequential(
            nn.Linear(self.hidden_dim*2, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim//2),
            nn.ELU(),
            nn.Linear(self.hidden_dim//2, 1)
        )

        self.name = 'Agent_Lstm'

    def forward(self, laser, state):
        laser_enc, _ = self.lstm(laser)
        # laser_enc = self.lstm_norm(laser_enc[:,-1,:])  # 添加归一化
        laser_enc = laser_enc[:,-1,:]

        sta = self.state(state)
        state1 = self.state_1(sta) + sta

        data = torch.cat((laser_enc, state1), 1)
        return data

    def get_value(self, laser, state):
        data = self.forward(laser, state)
        v = self.critic(data)
        return v

    def get_action(self, laser, state):
        data = self.forward(laser, state)
        action = self.actor_mean(data)
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
        self.actor_logstd = nn.Parameter(torch.zeros(1, 2))

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

        self.laser = nn.Sequential(nn.Linear(LASER_NUM, self.hidden_dim), nn.ELU(),
                                   )
        self.laser_1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.state = nn.Sequential(nn.Linear(4, self.hidden_dim), nn.ELU(),
                                   )
        self.state_1 = nn.Linear(self.hidden_dim, self.hidden_dim)

        # self.init_layer()

        self.name = 'Agent_Linear'

    def forward(self, laser, state):
        laser = self.laser(laser[:,-1,:])
        laser1 = self.laser_1(laser) + laser
        state = self.state(state)
        state1 = self.state_1(laser) + state

        data = torch.cat((laser1, state1), 1)
        return data
    
def parse_args(argv=None):
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", type=float, default=5e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
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
    
    args = parser.parse_args(argv)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args
