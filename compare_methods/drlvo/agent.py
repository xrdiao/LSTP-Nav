from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

try:
    from .custom_cnn_full import PaperFeatureExtractor
except ImportError:  # pragma: no cover
    from custom_cnn_full import PaperFeatureExtractor


def make_mlp(in_dim, hidden_dims, out_dim):
    layers = []
    last_dim = in_dim
    for dim in hidden_dims:
        layers.append(nn.Linear(last_dim, dim))
        layers.append(nn.ReLU(inplace=True))
        last_dim = dim
    layers.append(nn.Linear(last_dim, out_dim))
    return nn.Sequential(*layers)


class DrlVOAgent(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.name = "DRLVOAgent"
        self.encoder = PaperFeatureExtractor(features_dim=cfg.feature_dim)
        self.actor = make_mlp(cfg.feature_dim, cfg.actor_hidden_dims, 2)
        self.critic = make_mlp(cfg.feature_dim, cfg.critic_hidden_dims, 1)
        self.actor_logstd = nn.Parameter(torch.full((1, 2), cfg.init_logstd))

    def encode(self, obs):
        return self.encoder(obs)

    def get_action(self, obs):
        return self.actor(self.encode(obs))

    def get_value(self, obs):
        return self.critic(self.encode(obs))

    def get_action_and_value(self, obs, action=None):
        features = self.encode(obs)
        action_mean = self.actor(features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        value = self.critic(features)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value

    def convert_action_for_env(self, action):
        env_action = action.clone()
        env_action[:, 0] = (torch.tanh(env_action[:, 0]) / 2.0 + 0.5) * (
            self.cfg.linear_vel_max - self.cfg.linear_vel_min
        ) + self.cfg.linear_vel_min
        env_action[:, 1] = torch.tanh(env_action[:, 1]) * self.cfg.angular_limit
        return env_action
