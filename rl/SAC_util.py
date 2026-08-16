import torch.nn as nn
import torch
from torch import Tensor
from torch.distributions import Normal

from env_sim.argument import LASER_NUM


def build_mlp(dims: [int], activation: nn = None, if_raw_out: bool = True) -> nn.Sequential:
    """
    build MLP (MultiLayer Perceptron)

    dims: the middle dimension, `dims[-1]` is the output dimension of this network
    activation: the activation function
    if_remove_out_layer: if remove the activation function of the output layer.
    """
    if activation is None:
        activation = nn.ReLU
    net_list = []
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), activation()])
    if if_raw_out:
        del net_list[-1]  # delete the activation function of the output layer to keep raw output
    return nn.Sequential(*net_list)


def layer_init_with_orthogonal(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)


class ActorBase(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = None  # build_mlp(dims=[state_dim, *dims, action_dim])
        self.explore_noise_std = None  # standard deviation of exploration action noise
        self.ActionDist = torch.distributions.normal.Normal

        self.state_avg = nn.Parameter(torch.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((state_dim,)), requires_grad=False)

        self.name = 'SAC_Actor'

    def state_norm(self, state: Tensor) -> Tensor:
        return (state - self.state_avg) / self.state_std


class ActorSAC(ActorBase):
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)

        self.hidden_dim = 256
        self.lstm = nn.LSTM(input_size=state_dim - 4, hidden_size=self.hidden_dim, num_layers=3, batch_first=True)
        self.att = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=4, batch_first=True)
        self.q = nn.Sequential(nn.Linear(4, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.hidden_dim))
        self.k = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
                               nn.Linear(self.hidden_dim, self.hidden_dim))
        self.v = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
                               nn.Linear(self.hidden_dim, self.hidden_dim))

        self.decode = nn.Sequential(nn.Linear(self.hidden_dim + 4, self.hidden_dim), nn.ReLU(),
                                    nn.Linear(self.hidden_dim, 4))

        self.a = 0.85

    def encode_laser(self, laser, state):
        q = self.q(state)

        laser, _ = self.lstm(laser)
        laser = laser.sum(1)
        # laser = laser[:,-1,:]

        k = self.k(laser)
        v = self.v(laser)

        x, emb_weights = self.att(q, k, v)

        laser = self.a * x + (1 - self.a) * laser
        laser = torch.cat([laser, state], dim=1)
        return laser

    def forward(self, laser, state):
        laser = self.encode_laser(laser, state)
        a_avg = self.decode(laser)[:, :self.action_dim]
        return a_avg.tanh()  # action

    def get_action(self, laser, state):
        s_enc = self.encode_laser(laser, state)
        a_avg, a_std_log = self.decode(s_enc).chunk(2, dim=1)
        a_std = a_std_log.clamp(-16, 2).exp()

        dist = Normal(a_avg, a_std)
        return dist.rsample().tanh()  # action (re-parameterize)

    def get_action_logprob(self, laser, state):
        s_enc = self.encode_laser(laser, state)
        a_avg, a_std_log = self.decode(s_enc).chunk(2, dim=1)
        a_std = a_std_log.clamp(-16, 2).exp()

        dist = Normal(a_avg, a_std)
        action = dist.rsample()

        action_tanh = action.tanh()
        logprob = dist.log_prob(a_avg)
        logprob -= (-action_tanh.pow(2) + 1.000001).log()  # fix logprob using the derivative of action.tanh()
        return action_tanh, logprob.sum(1)


class CriticBase(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = None  # build_mlp(dims=[state_dim + action_dim, *dims, 1])

        self.state_avg = nn.Parameter(torch.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((state_dim,)), requires_grad=False)
        self.value_avg = nn.Parameter(torch.zeros((1,)), requires_grad=False)
        self.value_std = nn.Parameter(torch.ones((1,)), requires_grad=False)

    def state_norm(self, state: Tensor) -> Tensor:
        return (state - self.state_avg) / self.state_std

    def value_re_norm(self, value: Tensor) -> Tensor:
        return value * self.value_std + self.value_avg


class CriticTwin(CriticBase):  # shared parameter
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.hidden_dim = 256

        self.lstm = nn.LSTM(input_size=state_dim - 4, hidden_size=self.hidden_dim, num_layers=3, batch_first=True)
        self.att = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=4, batch_first=True)
        self.q = nn.Sequential(nn.Linear(4, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.hidden_dim))
        self.k = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
                               nn.Linear(self.hidden_dim, self.hidden_dim))
        self.v = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
                               nn.Linear(self.hidden_dim, self.hidden_dim))

        self.decode = nn.Sequential(nn.Linear(self.hidden_dim + 6, self.hidden_dim), nn.ReLU(),
                                    nn.Linear(self.hidden_dim, 4))
        self.a = 0.85

    def encode_laser(self, laser, state, action):
        q = self.q(state)

        laser, _ = self.lstm(laser)
        laser = laser.sum(1)
        # laser = laser[:,-1,:]

        k = self.k(laser)
        v = self.v(laser)

        x, emb_weights = self.att(q, k, v)

        laser = self.a * x + (1 - self.a) * laser
        values = torch.cat((laser, state[:, -4:], action), dim=1)

        return values

    def forward(self, laser, state, action):
        values = self.encode_laser(laser, state[:, -4:], action)
        values = self.decode(values)

        values = self.value_re_norm(values)
        return values.mean(dim=1)  # mean Q value

    def get_q_min(self, laser, state, action):
        values = self.encode_laser(laser, state[:, -4:], action)
        values = self.decode(values)

        values = self.value_re_norm(values)
        return torch.min(values, dim=1)[0]  # min Q value

    def get_q1_q2(self, laser, state, action):
        values = self.encode_laser(laser, state[:, -4:], action)
        values = self.decode(values)

        values = self.value_re_norm(values)
        return values[:, 0], values[:, 1]  # two Q values
