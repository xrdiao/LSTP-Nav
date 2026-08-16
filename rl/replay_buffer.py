import os
import math
import torch
from typing import Tuple
from torch import Tensor

from env_sim.argument import ROBOT_LASER_BUFFER, LASER_NUM


class ReplayBuffer:  # for off-policy
    def __init__(self,
                 max_size: int,
                 state_dim: int,
                 action_dim: int,
                 gpu_id: int = 0,
                 num_seqs: int = 1,
                 if_use_per: bool = False):
        self.p = 0  # pointer
        self.if_full = False
        self.cur_size = 0
        self.add_size = 0
        self.add_item = None
        self.max_size = max_size
        self.num_seqs = num_seqs
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        """The struction of ReplayBuffer (for example, num_seqs = num_workers * num_envs == 2*4 = 8
        ReplayBuffer:
        worker0 for env0:   sequence of sub_env0.0  self.states  = Tensor[s, s, ..., s, ..., s]     
                                                    self.actions = Tensor[a, a, ..., a, ..., a]   
                                                    self.rewards = Tensor[r, r, ..., r, ..., r]   
                                                    self.undones = Tensor[d, d, ..., d, ..., d]
                                                                          <-----max_size----->
                                                                          <-cur_size->
                                                                                     ↑ pointer
                            sequence of sub_env0.1  s, s, ..., s    a, a, ..., a    r, r, ..., r    d, d, ..., d
                            sequence of sub_env0.2  s, s, ..., s    a, a, ..., a    r, r, ..., r    d, d, ..., d
                            sequence of sub_env0.3  s, s, ..., s    a, a, ..., a    r, r, ..., r    d, d, ..., d
        worker1 for env1:   sequence of sub_env1.0  s, s, ..., s    a, a, ..., a    r, r, ..., r    d, d, ..., d
                            sequence of sub_env1.1  s, s, ..., s    a, a, ..., a    r, r, ..., r    d, d, ..., d
                            sequence of sub_env1.2  s, s, ..., s    a, a, ..., a    r, r, ..., r    d, d, ..., d
                            sequence of sub_env1.3  s, s, ..., s    a, a, ..., a    r, r, ..., r    d, d, ..., d
        
        D: done=True
        d: done=False
        sequence of transition: s-a-r-d, s-a-r-d, s-a-r-D  s-a-r-d, s-a-r-d, s-a-r-d, s-a-r-d, s-a-r-D  s-a-r-d, ...
                                <------trajectory------->  <----------trajectory--------------------->  <-----------
        """
        self.states = torch.empty((max_size, num_seqs, state_dim), dtype=torch.float32, device=self.device)
        self.lasers = torch.empty((max_size, num_seqs, ROBOT_LASER_BUFFER, LASER_NUM), dtype=torch.float32,
                                  device=self.device)
        self.actions = torch.empty((max_size, num_seqs, action_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.empty((max_size, num_seqs), dtype=torch.float32, device=self.device)
        self.undones = torch.empty((max_size, num_seqs), dtype=torch.float32, device=self.device)

        self.if_use_per = if_use_per

        self.sum_trees = None
        self.per_alpha = None
        self.per_beta = None

    def update(self, items: Tuple[Tensor, ...]):
        self.add_item = items
        states, actions, rewards, undones, lasers = items
        self.add_size = rewards.shape[0]

        p = self.p + self.add_size  # pointer
        if p > self.max_size:
            self.if_full = True
            p0 = self.p
            p1 = self.max_size
            p2 = self.max_size - self.p
            p = p - self.max_size

            self.states[p0:p1], self.states[0:p] = states[:p2], states[-p:]
            self.lasers[p0:p1], self.lasers[0:p] = lasers[:p2], lasers[-p:]
            self.actions[p0:p1], self.actions[0:p] = actions[:p2], actions[-p:]
            self.rewards[p0:p1], self.rewards[0:p] = rewards[:p2], rewards[-p:]
            self.undones[p0:p1], self.undones[0:p] = undones[:p2], undones[-p:]
        else:
            self.states[self.p:p] = states
            self.lasers[self.p:p] = lasers
            self.actions[self.p:p] = actions
            self.rewards[self.p:p] = rewards
            self.undones[self.p:p] = undones

        self.p = p
        self.cur_size = self.max_size if self.if_full else self.p

    def sample(self, batch_size: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        sample_len = self.cur_size - 1

        ids = torch.randint(sample_len * self.num_seqs, size=(batch_size,), requires_grad=False)
        ids0 = torch.fmod(ids, sample_len)  # ids % sample_len
        ids1 = torch.div(ids, sample_len, rounding_mode='floor')  # ids // sample_len

        return (self.states[ids0, ids1],
                self.actions[ids0, ids1],
                self.rewards[ids0, ids1],
                self.undones[ids0, ids1],
                self.states[ids0 + 1, ids1],
                self.lasers[ids0, ids1],
                self.lasers[ids0 + 1, ids1],)  # next_state
