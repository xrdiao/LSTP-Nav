import copy
from collections import deque
import random
import torch
import numpy as np


class RelayBuffer:
    def __init__(self, buffer_size=100, step_length=10, robots_num=10, args=None):
        self.robots_num = robots_num
        self.step_buffer = [deque(maxlen=step_length) for _ in range(robots_num)]
        self.buffer = deque(maxlen=buffer_size)
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.args.cuda else "cpu")

    def add(self, scan, state, action, reward, done, logprob, value):
        for i in range(self.robots_num):
            self.step_buffer[i].append(
                (scan[i], state[i], action[i], reward[i], done[i], logprob[i], value[i]))

    def add_trajectory(self, idx):
        buffer = copy.deepcopy(self.step_buffer[idx])
        scan, state, action, reward, done, logprob, value = zip(*buffer)
        scan = torch.stack(list(scan))
        state = torch.stack(list(state))
        action = torch.stack(list(action))
        reward = list(reward)
        done = list(done)
        logprob = torch.stack(list(logprob))
        value = torch.stack(list(value))

        advantages = torch.zeros_like(value).to(self.device)
        lastgaelam = 0
        length = len(buffer)
        for t in reversed(range(length)):
            if t == length - 1:
                nextnonterminal = 0
                nextvalues = 0
            else:
                nextnonterminal = 1.0 - done[t + 1]
                nextvalues = value[t + 1]
            delta = reward[t] + self.args.gamma * nextvalues * nextnonterminal - value[t]
            advantages[
                t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + value
        for i in range(len(scan)):
            self.buffer.append((scan[i], state[i], action[i], advantages[i], logprob[i], returns[i], value[i]))

        self.step_buffer[idx].clear()

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        scan, state, action, adv, logprob, q, value = zip(*samples)

        scan = torch.stack(list(scan))
        state = torch.stack(list(state))
        action = torch.stack(list(action))
        logprob = torch.stack(list(logprob))
        value = torch.stack(list(value))
        adv = torch.stack(list(adv))
        q = torch.stack(list(q))

        return scan, state, action, adv.squeeze(), logprob, q.squeeze(), value.squeeze()

    def clear(self):
        self.buffer.clear()
        [i.clear() for i in self.step_buffer]
