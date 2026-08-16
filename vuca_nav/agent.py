from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def make_mlp(in_dim, dims, out_dim=None, activation=nn.ELU):
    layers = []
    last_dim = in_dim
    for dim in dims:
        layers.append(nn.Linear(last_dim, dim))
        layers.append(activation())
        last_dim = dim
    if out_dim is not None:
        layers.append(nn.Linear(last_dim, out_dim))
    return nn.Sequential(*layers)


class BaseAgent(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.actor_logstd = nn.Parameter(torch.full((1, 2), cfg.init_logstd))

    def get_action(self, laser, state):
        raise NotImplementedError

    def get_value(self, laser, state):
        raise NotImplementedError

    def get_action_and_value(self, laser, state, action=None):
        action_mean = self.get_action(laser, state)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.get_value(laser, state)

    def convert_action_for_env(self, action):
        env_action = action.clone()
        env_action[:, 0] = (torch.tanh(env_action[:, 0]) / 2.0 + 0.5) * self.cfg.preferred_velocity
        env_action[:, 1] = torch.tanh(env_action[:, 1]) * self.cfg.angular_limit
        return env_action


class SimplePaperAgent(BaseAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.name = "PaperSimpleAgent"

        self.laser_encoder = make_mlp(cfg.lidar_num, cfg.laser_encoder_dims[:-1], out_dim=cfg.laser_encoder_dims[-1])
        self.state_encoder = make_mlp(cfg.state_dim, cfg.state_encoder_dims[:-1], out_dim=cfg.state_encoder_dims[-1])

        fused_dim = cfg.laser_encoder_dims[-1] + cfg.state_encoder_dims[-1]
        self.actor = nn.Sequential(
            make_mlp(fused_dim, cfg.policy_net_arch[:-1], out_dim=cfg.policy_net_arch[-1]),
            nn.Linear(cfg.policy_net_arch[-1], 2),
        )
        self.critic = nn.Sequential(
            make_mlp(fused_dim, cfg.value_net_arch[:-1], out_dim=cfg.value_net_arch[-1]),
            nn.Linear(cfg.value_net_arch[-1], 1),
        )

    def encode(self, laser, state):
        laser_feature = self.laser_encoder(laser)
        state_feature = self.state_encoder(state)
        return torch.cat((laser_feature, state_feature), dim=1)

    def get_action(self, laser, state):
        return self.actor(self.encode(laser, state))

    def get_value(self, laser, state):
        return self.critic(self.encode(laser, state))


class PaperCOAAgent(BaseAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.name = "PaperCOAAgent"

        social_in_dim = cfg.robot_state_dim + 7 + 3
        self.rnn = nn.GRUCell(social_in_dim, cfg.paper_rnn_hidden_dim)
        self.pairwise = make_mlp(cfg.paper_rnn_hidden_dim, cfg.paper_pairwise_dims[:-1], out_dim=cfg.paper_pairwise_dims[-1])
        self.attention = make_mlp(cfg.paper_rnn_hidden_dim, cfg.paper_attention_dims[:-1], out_dim=1)
        self.lidar_encoder = make_mlp(cfg.lidar_num, cfg.paper_lidar_dims[:-1], out_dim=cfg.paper_lidar_dims[-1])

        fused_dim = cfg.paper_pairwise_dims[-1] + cfg.paper_lidar_dims[-1] + cfg.robot_state_dim
        self.actor = nn.Sequential(
            make_mlp(fused_dim, cfg.policy_net_arch[:-1], out_dim=cfg.policy_net_arch[-1]),
            nn.Linear(cfg.policy_net_arch[-1], 2),
        )
        self.critic = nn.Sequential(
            make_mlp(fused_dim, cfg.value_net_arch[:-1], out_dim=cfg.value_net_arch[-1]),
            nn.Linear(cfg.value_net_arch[-1], 1),
        )

    def _split_state(self, state):
        robot_state = state[:, : self.cfg.robot_state_dim]
        social = state[:, self.cfg.robot_state_dim :]
        batch = state.shape[0]

        if social.shape[1] == 0:
            human_state = state.new_zeros((batch, 7))
            spatial = state.new_zeros((batch, 3))
            mask = state.new_zeros((batch, 1))
            return robot_state, human_state, spatial, mask

        human_xyvr = social[:, :4]
        human_dist = social[:, 4:5]
        human_visible = social[:, 5:6]
        human_state = torch.cat(
            [
                human_xyvr[:, 0:1],
                human_xyvr[:, 1:2],
                human_xyvr[:, 2:3],
                human_xyvr[:, 3:4],
                human_dist.new_full((batch, 1), self.cfg.human_radius),
                human_dist,
                human_dist.new_full((batch, 1), self.cfg.robot_radius + self.cfg.human_radius),
            ],
            dim=1,
        )
        spatial = state.new_zeros((batch, 3))
        mask = human_visible
        return robot_state, human_state, spatial, mask

    def encode(self, laser, state):
        robot_state, human_state, spatial, mask = self._split_state(state)
        social_input = torch.cat([robot_state, human_state, spatial], dim=1)
        hidden = self.rnn(social_input, state.new_zeros((state.shape[0], self.cfg.paper_rnn_hidden_dim)))
        pairwise = self.pairwise(hidden)
        logits = self.attention(hidden)
        weights = torch.sigmoid(logits) * mask
        crowd_feature = pairwise * weights
        no_human = (mask <= 0.0).float()
        crowd_feature = crowd_feature * (1.0 - no_human)

        lidar_feature = self.lidar_encoder(laser)
        return torch.cat((crowd_feature, lidar_feature, robot_state), dim=1)

    def get_action(self, laser, state):
        return self.actor(self.encode(laser, state))

    def get_value(self, laser, state):
        return self.critic(self.encode(laser, state))


def build_agent(cfg):
    agent_name = cfg.policy_name.lower()
    if agent_name == "simple":
        return SimplePaperAgent(cfg)
    if agent_name == "paper":
        return PaperCOAAgent(cfg)
    raise ValueError(f"Unsupported policy_name={cfg.policy_name!r}")
