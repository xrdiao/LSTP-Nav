import argparse
import os
from distutils.util import strtobool

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal, Categorical
from torch.nn import functional as F

from env_sim.argument import LASER_NUM


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class SimpleAgent(nn.Module):
    def __init__(self):
        super(SimpleAgent, self).__init__()
        self.hidden_size = 4
        self.actor = nn.Sequential(nn.Linear(2, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, 2))
        self.critic = nn.Sequential(nn.Linear(2, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, 1))
        self.actor_logstd = nn.Parameter(torch.zeros(1, 2))
        self.name = 'SimpleAgent'

    def get_deterministic_action(self, x):
        return self.actor(x), self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    def get_value(self, x):
        return self.critic(x)


class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.conv1_actor = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5, stride=2)
        self.conv2_actor = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=3, stride=2)
        self.fc1_actor = nn.Linear(126, 124)
        self.fc2_actor = nn.Linear(128, 2)

        self.laser_actor = nn.Sequential(self.conv1_actor, nn.ReLU(), self.conv2_actor, nn.ReLU(), self.fc1_actor)

        self.conv1_critic = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5, stride=2)
        self.conv2_critic = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=3, stride=2)
        self.fc1_critic = nn.Linear(126, 124)
        self.fc2_critic = nn.Linear(128, 1)

        self.laser_critic = nn.Sequential(self.conv1_critic, nn.ReLU(), self.conv2_critic, nn.ReLU(), self.fc1_critic)
        self.name = 'Agent'

    def get_value(self, laser, state):
        state = state.unsqueeze(1)
        laser = self.laser_critic(laser)
        data = torch.cat((laser, state), -1)
        v = self.fc2_critic(data)
        return v

    def get_action(self, laser, state):
        state = state.unsqueeze(1)
        laser = self.laser_actor(laser)
        data = torch.cat((laser, state), -1)
        vel = self.fc2_actor(data)
        return vel

    def get_deterministic_action(self, laser, state):
        action = self.get_action(laser, state)
        value = self.get_value(laser, state)
        return action, value


class AttentionAgent(nn.Module):
    def __init__(self, env):
        super(AttentionAgent, self).__init__()
        self.hidden_dim = 256

        self.lstm_actor = nn.LSTM(input_size=LASER_NUM, hidden_size=self.hidden_dim, num_layers=3, batch_first=True)
        self.lstm_actor_k = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lstm_actor_v = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.att_actor = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=2, batch_first=True)
        self.state_q_actor = nn.Linear(4, self.hidden_dim)

        self.lstm_critic = nn.LSTM(input_size=LASER_NUM, hidden_size=self.hidden_dim, num_layers=3, batch_first=True)
        self.lstm_critic_k = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lstm_critic_v = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.att_critic = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=2, batch_first=True)
        self.state_q_critic = layer_init(nn.Linear(4, self.hidden_dim))

        self.decode_actor = nn.Sequential(nn.Linear(self.hidden_dim, int(self.hidden_dim / 2)), nn.ReLU(),
                                          nn.Linear(int(self.hidden_dim / 2), 2))
        self.decode_critic = nn.Sequential(nn.Linear(self.hidden_dim, int(self.hidden_dim / 2)), nn.ReLU(),
                                           nn.Linear(int(self.hidden_dim / 2), 2))

        self.fc_actor = nn.Linear(6, 2)
        self.fc_critic = nn.Linear(6, 1)

        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(env.action_space.shape)))
        self.action_bound = env.action_space.high[0]
        self.name = 'AttentionAgent'

    def get_value(self, laser, state):
        laser, _ = self.lstm_critic(laser)
        laser = F.relu(laser[:, -1, :])
        q = self.state_q_critic(state)
        k = self.lstm_critic_k(laser)
        v = self.lstm_critic_v(laser)
        emb, emb_weights = self.att_critic(q, k, v)
        emb = F.relu(emb)

        decoder = self.decode_critic(emb)
        final_emb = torch.cat([decoder, state], dim=1)
        y = self.fc_critic(final_emb)
        return y

    def get_action(self, laser, state):

        laser, _ = self.lstm_actor(laser)
        laser = F.relu(laser[:, -1, :])
        q = self.state_q_actor(state)
        k = self.lstm_actor_k(laser)
        v = self.lstm_actor_v(laser)
        emb, emb_weights = self.att_actor(q, k, v)
        emb = F.relu(emb)

        decoder = F.relu(self.decode_actor(emb))
        final_emb = torch.cat([decoder, state], dim=1)
        y = self.fc_actor(final_emb)
        return y

    def get_action_and_value(self, laser, state, action=None):
        action_mean = self.get_action(laser, state)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.get_value(laser, state)

    def get_deterministic_action(self, laser, state):
        action = self.get_action(laser, state)
        value = self.get_value(laser, state)
        return action, value


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="MyEnv-v0",
                        help="the id of the gym environment")
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
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation-details",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=1,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=648,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=16,
                        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
                        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
                        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args
