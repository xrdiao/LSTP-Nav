import argparse
import copy
from distutils.util import strtobool

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
    def convert_action_for_env(a):
        action = copy.deepcopy(a)
        action[:, 0] = torch.tanh(action[:, 0]) / 2 + 0.5 * MAX_SPEED
        action[:, 1] = torch.tanh(action[:, 1]) * MAX_ROTATION_SPEED
        return action


class RandomAgent(BaseAgent):
    def __init__(self):
        super(RandomAgent, self).__init__()

    def get_deterministic_action(self, a, b):
        action = torch.normal(0, 1, size=[1, 2])
        return action, None


class IJRRAgent(BaseAgent):
    def __init__(self):
        super(IJRRAgent, self).__init__()
        assert LASER_NUM == 512, "LASER_NUM should be 512"
        self.laser_encode = nn.Sequential(nn.Conv1d(ROBOT_LASER_BUFFER, 32, 5, 2), nn.ReLU(), nn.Conv1d(32, 32, 3, 2),
                                          nn.ReLU(), nn.Flatten(), nn.Linear(4032, 256))

        self.actor = nn.Sequential(nn.Linear(260, 128), nn.ReLU(), nn.Linear(128, 2))
        self.critic = nn.Sequential(nn.Linear(260, 128), nn.ReLU(), nn.Linear(128, 1))

        self.actor_logstd = nn.Parameter(torch.zeros(1, 2))
        self.a = torch.zeros(1)

        self.name = 'Agent_IJRR'

    def forward(self, laser, goal):
        laser = self.laser_encode(laser)
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


class Conv1dAgent(BaseAgent):
    def __init__(self):
        super(Conv1dAgent, self).__init__()
        assert LASER_NUM == 512, "LASER_NUM should be 512"
        self.hidden_dim = 256

        self.laser = nn.Sequential(nn.Conv1d(ROBOT_LASER_BUFFER, 32, 5, 2), nn.ReLU(), nn.Conv1d(32, 32, 3, 2),
                                   nn.ReLU(), nn.Flatten(), nn.Linear(4032, 256))

        self.att = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=4, batch_first=True)
        self.q = nn.Sequential(nn.Linear(4, self.hidden_dim))

        self.critic = nn.Sequential(nn.Linear(self.hidden_dim + 4, int(self.hidden_dim / 2)), nn.ReLU(),
                                    nn.Linear(int(self.hidden_dim / 2), 1))
        self.actor = nn.Sequential(nn.Linear(self.hidden_dim + 4, int(self.hidden_dim / 2)), nn.ReLU(),
                                   nn.Linear(int(self.hidden_dim / 2), 2))
        self.a = nn.Parameter(torch.ones(1))
        self.actor_logstd = nn.Parameter(torch.zeros(1, 2))

        self.init_layer()

        self.name = 'Agent_conv'

    def forward(self, laser, goal):
        laser = self.laser(laser)

        q = self.q(goal)
        k = laser
        v = laser
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

        self.laser = nn.Sequential(nn.Linear(LASER_NUM, self.hidden_dim), nn.ReLU(),
                                   nn.Linear(self.hidden_dim, self.hidden_dim))
        self.att = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=4, batch_first=True)
        self.q = nn.Sequential(nn.Linear(4, self.hidden_dim), nn.ReLU(),
                               nn.Linear(self.hidden_dim, self.hidden_dim))
        self.k = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
                               nn.Linear(self.hidden_dim, self.hidden_dim))
        self.v = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
                               nn.Linear(self.hidden_dim, self.hidden_dim))

        self.critic = nn.Sequential(nn.Linear(self.hidden_dim + 4, self.hidden_dim), nn.ReLU(),
                                    nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
                                    nn.Linear(self.hidden_dim, 1))
        self.actor = nn.Sequential(nn.Linear(self.hidden_dim + 4, self.hidden_dim), nn.ReLU(),
                                   nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
                                   nn.Linear(self.hidden_dim, 2))
        self.a = nn.Parameter(torch.ones(1))
        self.actor_logstd = nn.Parameter(torch.zeros(1, 2))

        self.init_layer()

        self.name = 'Agent_Linear'

    def forward(self, laser, goal):
        laser = self.laser(laser[:, -1, :])

        q = self.q(goal)
        k = self.k(laser)
        v = self.v(laser)
        x, emb_weights = self.att(q, k, v)

        laser = self.a * laser + (1 - self.a) * x

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


class LstmAgent(BaseAgent):
    def __init__(self):
        super(LstmAgent, self).__init__()
        self.hidden_dim = 252
        input_size = LASER_NUM

        self.fc1 = nn.Sequential(nn.Linear(input_size, self.hidden_dim), nn.ReLU(),
                                 nn.Linear(self.hidden_dim, self.hidden_dim))
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_dim, num_layers=2, batch_first=True)
        self.state_q_critic = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                                            nn.Linear(256, 1))
        self.state_q_actor = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                                           nn.Linear(256, 2))

        # self.init_layer()

        self.actor_logstd = nn.Parameter(torch.zeros(1, 2))
        self.a = nn.Parameter(torch.ones(1))
        self.name = 'Agent_Lstm'

    def forward(self, laser, state):
        x = self.fc1(laser[:, -1, :])
        laser, _ = self.lstm(laser)
        laser = self.a * laser[:, -1, :] + (1 - self.a) * x

        emb = torch.cat([laser, state], dim=1)
        return emb

    def get_value(self, laser, state):
        emb = self.forward(laser, state)
        value = self.state_q_critic(emb)
        return value

    def get_action(self, laser, state):
        emb = self.forward(laser, state)
        action = self.state_q_actor(emb)
        return action


class AttentionAgent(BaseAgent):
    def __init__(self):
        super(AttentionAgent, self).__init__()
        self.hidden_dim = 256
        input_size = LASER_NUM
        self.activate = nn.Tanh()

        self.lstm_critic = nn.LSTM(input_size=input_size, hidden_size=self.hidden_dim, num_layers=3, batch_first=True)
        self.att_critic = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=4, batch_first=True)
        self.q_critic = nn.Sequential(nn.Linear(4, self.hidden_dim), self.activate,
                                      nn.Linear(self.hidden_dim, self.hidden_dim))
        self.k_critic = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), self.activate,
                                      nn.Linear(self.hidden_dim, self.hidden_dim))
        self.v_critic = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), self.activate,
                                      nn.Linear(self.hidden_dim, self.hidden_dim))

        self.decode_actor = nn.Sequential(nn.Linear(self.hidden_dim + 4, self.hidden_dim), self.activate,
                                          nn.Linear(self.hidden_dim, 2))
        self.decode_critic = nn.Sequential(nn.Linear(self.hidden_dim + 4, self.hidden_dim), self.activate,
                                           nn.Linear(self.hidden_dim, 1))

        self.actor_logstd = nn.Parameter(torch.zeros(1, 2))
        self.a = nn.Parameter(torch.ones(1))
        self.name = 'AttentionAgent'

        self.init_layer()

    def forward(self, laser, state):
        q = self.q_critic(state)

        laser, _ = self.lstm_critic(laser)
        laser = laser.sum(1)

        k = self.k_critic(laser)
        v = self.v_critic(laser)

        x, emb_weights = self.att_critic(q, k, v)

        laser = self.a * x + (1 - self.a) * laser
        laser = torch.cat([laser, state], dim=1)
        return laser

    def get_value(self, laser, state):
        laser = self.forward(laser, state)
        v = self.decode_critic(laser)
        return v

    def get_action(self, laser, state):
        laser = self.forward(laser, state)
        action_mean = self.decode_actor(laser)
        return action_mean


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
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=2000000,
                        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=1,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=512,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.98,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=8,
                        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
                        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--reward_scaling", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="relay buffer")
    parser.add_argument("--lagrangian-multiplier-init", type=float, default=0.001, nargs="?", const=True,
                        help="initial value of lagrangian multiplier")
    parser.add_argument("--lagrangian-multiplier-lr", type=float, default=0.03, nargs="?", const=True,
                        help="learning rate of lagrangian multiplier")
    parser.add_argument("--cost-limit", type=float, default=25.0, nargs="?", const=True,
                        help="cost lim")

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


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 2  # 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CNNAgent(BaseAgent):
    def __init__(self):
        self.name = 'CNNAgent'
        # network parameters:
        block = Bottleneck
        layers = [2, 1, 1]
        zero_init_residual = True
        groups = 1
        width_per_group = 64
        replace_stride_with_dilation = None
        norm_layer = None

        # inherit the superclass properties/methods
        #
        super(CNNAgent, self).__init__()
        # define the model
        #
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1_critic = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1,
                                      bias=False)
        self.bn1_critic = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.layer1_critic = self._make_layer(block, 64, layers[0])
        self.layer2_critic = self._make_layer(block, 128, layers[1], stride=2,
                                              dilate=replace_stride_with_dilation[0])
        self.layer3_critic = self._make_layer(block, 256, layers[2], stride=2,
                                              dilate=replace_stride_with_dilation[1])

        self.conv2_2_critic = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(256)
        )
        self.downsample2_critic = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(2, 2), padding=(0, 0)),
            nn.BatchNorm2d(256)
        )
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3_2_critic = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(512)
        )
        self.downsample3_critic = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=512, kernel_size=(1, 1), stride=(4, 4), padding=(0, 0)),
            nn.BatchNorm2d(512)
        )
        self.relu3 = nn.ReLU(inplace=True)

        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
        #                               dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear_fc_critic = nn.Sequential(
            nn.Linear(256 * block.expansion + 4, 1),
            # nn.BatchNorm1d(features_dim),
            nn.ReLU()
        )

        self.inplanes = 64
        self.conv1_actor = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.bn1_actor = norm_layer(self.inplanes)
        self.layer1_actor = self._make_layer(block, 64, layers[0])
        self.layer2_actor = self._make_layer(block, 128, layers[1], stride=2,
                                             dilate=replace_stride_with_dilation[0])
        self.layer3_actor = self._make_layer(block, 256, layers[2], stride=2,
                                             dilate=replace_stride_with_dilation[1])

        self.conv2_2_actor = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(256)
        )
        self.downsample2_actor = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(2, 2), padding=(0, 0)),
            nn.BatchNorm2d(256)
        )

        self.conv3_2_actor = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(512)
        )
        self.downsample3_actor = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=512, kernel_size=(1, 1), stride=(4, 4), padding=(0, 0)),
            nn.BatchNorm2d(512)
        )

        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
        #                               dilate=replace_stride_with_dilation[2])
        self.linear_fc_actor = nn.Sequential(
            nn.Linear(256 * block.expansion + 4, 2),
            # nn.BatchNorm1d(features_dim),
            nn.ReLU()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):  # add by xzt
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def get_value(self, laser, state):
        fusion_in = laser.reshape(-1, 1, 3, LASER_NUM)

        x = self.conv1_critic(fusion_in)
        x = self.bn1_critic(x)
        x = self.relu(x)
        x = self.maxpool(x)

        identity3 = self.downsample3_critic(x)

        x = self.layer1_critic(x)

        identity2 = self.downsample2_critic(x)

        x = self.layer2_critic(x)

        x = self.conv2_2_critic(x)
        x += identity2
        x = self.relu2(x)

        x = self.layer3_critic(x)

        x = self.conv3_2_critic(x)
        x += identity3
        x = self.relu3(x)

        x = self.avgpool(x)
        fusion_out = torch.flatten(x, 1)

        goal_in = state.reshape(-1, 4)
        goal_out = torch.flatten(goal_in, 1)

        # Combine
        fc_in = torch.cat((fusion_out, goal_out), dim=1)
        x = self.linear_fc_critic(fc_in)
        return x

    def get_action(self, laser, state):
        fusion_in = laser.reshape(-1, 1, 3, LASER_NUM)

        x = self.conv1_actor(fusion_in)
        x = self.bn1_actor(x)
        x = self.relu(x)
        x = self.maxpool(x)

        identity3 = self.downsample3_actor(x)

        x = self.layer1_actor(x)

        identity2 = self.downsample2_actor(x)

        x = self.layer2_actor(x)

        x = self.conv2_2_actor(x)
        x += identity2
        x = self.relu2(x)

        x = self.layer3_actor(x)

        x = self.conv3_2_actor(x)
        x += identity3
        x = self.relu3(x)

        x = self.avgpool(x)
        fusion_out = torch.flatten(x, 1)

        goal_in = state.reshape(-1, 4)
        goal_out = torch.flatten(goal_in, 1)

        # Combine
        fc_in = torch.cat((fusion_out, goal_out), dim=1)
        x = self.linear_fc_actor(fc_in)
        return x


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
