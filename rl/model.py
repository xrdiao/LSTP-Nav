# 用于连续动作的PPO
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class PolicyNet(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions, embed_dim, num_heads):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)

        self.fc_mu = nn.Linear(n_hiddens, n_actions)
        self.fc_std = nn.Linear(n_hiddens, n_actions)

        # self.att_goal = nn.MultiheadAttention(embed_dim, num_heads)
        # self.Q_goal = nn.Linear(2, embed_dim)
        # self.K_goal = nn.Linear(2, embed_dim)
        # self.W_goal = nn.Linear(2, embed_dim)
        #
        # self.att_laser = nn.MultiheadAttention(embed_dim, num_heads)
        # self.Q_laser = nn.Linear(n_states-2, embed_dim)
        # self.K_laser = nn.Linear(n_states-2, embed_dim)
        # self.W_laser = nn.Linear(n_states-2, embed_dim)

    # 前向传播
    def forward(self, x):
        x = self.fc1(x)  # [b, n_states] --> [b, n_hiddens]
        x = F.relu(x)
        mu = self.fc_mu(x)
        mu = F.sigmoid(mu)# [b, n_hiddens] --> [b, n_actions] mu=[velocity, rotational velocity] * robots_num
        std = self.fc_std(x)  # [b, n_hiddens] --> [b, n_actions]
        std = F.softplus(std)  # 值域 小于0的部分逼近0，大于0的部分几乎不变
        return mu, std


class ValueNet(nn.Module):
    def __init__(self, n_states, n_hiddens):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, 1)

    # 前向传播
    def forward(self, x):
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = F.relu(x)
        x = self.fc2(x)  # [b,n_hiddens]-->[b,1]
        return x


class PPO:
    def __init__(self, n_states, n_hiddens, n_actions,
                 actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device, embed_dim, num_heads, if_retrain=False):
        # 实例化策略网络
        self.actor = PolicyNet(n_states, n_hiddens, n_actions, embed_dim, num_heads).to(device)
        # 实例化价值网络
        self.critic = ValueNet(n_states, n_hiddens).to(device)
        # 策略网络的优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # 价值网络的优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        if if_retrain:
            print('retraining model')
            self.actor.load_state_dict(torch.load('./model/actor.pth'))
            self.critic.load_state_dict(torch.load('./model/critic.pth'))

        # 属性分配
        self.lmbda = lmbda  # GAE优势函数的缩放因子
        self.epochs = epochs  # 一条序列的数据用来训练多少轮
        self.eps = eps  # 截断范围
        self.gamma = gamma  # 折扣系数
        self.device = device

        # 动作选择

    def take_action(self, states):  # 输入当前时刻的状态
        '''

        :param states: 所有机器人观测到的状态
        :return: actions :所有机器人的下一个动作[robot1, robot2, ...]
        '''
        # [n_states]-->[1,n_states]-->tensor
        robots_num = len(states)
        actions = []
        for i in range(robots_num):
            state = torch.tensor(states[i], dtype=torch.float).to(self.device)
            mu, std = self.actor(state)  # 预测下一个动作

            vel_dict = torch.distributions.Normal(mu[0], std[0])
            _vel = vel_dict.sample().item()

            rot_dict = torch.distributions.Normal(mu[1], std[1])
            _rot = rot_dict.sample().item()

            actions.append([_vel, _rot])
        return actions  # 返回动作值

    def states2probs(self, states):
        mu, std = self.actor(states)  # [b,1]
        vel_dists = torch.distributions.Normal(mu[:, 0], std[:, 0])
        rot_dists = torch.distributions.Normal(mu[:, 1], std[:, 1])
        return vel_dists, rot_dists

    # 训练
    def update(self, transition_dict):
        # 提取数据集
        robot_num = len(transition_dict['states'][0])
        advantages = []
        old_log_probs = []
        next_states_values_set = []
        td_targets_set = []

        with torch.autograd.set_detect_anomaly(True):
            for i in range(robot_num):
                next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float)[:, i].to(
                    self.device)
                rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float)[:, i].view(-1, 1).to(
                    self.device)
                states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float)[:, i].to(self.device)
                actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float)[:, i].to(self.device)

                # 时序差分
                next_states_values_set.append(self.critic(next_states))
                current_state_values = self.critic(states)
                td_targets_set.append(rewards + self.gamma * next_states_values_set[-1])

                delta = td_targets_set[-1] - current_state_values
                delta = delta.cpu().detach().numpy()

                # 计算优势函数
                advantage = delta[0]  # 优势函数初始值
                advantage_list = [advantage]
                for delta in delta[1:]:
                    advantage = advantage + self.gamma * self.lmbda * delta
                    advantage_list.append(advantage)
                advantages.append(torch.tensor(advantage_list, dtype=torch.float).to(self.device))

                # 保存旧策略下选取当前动作的概率
                vel_dists, rot_dists = self.states2probs(states)
                # 这里假定速度和角速度是独立同分布，所以log(a,b)=log(a)+log(b)
                old_log_probs.append(vel_dists.log_prob(actions[:, 0]) + rot_dists.log_prob(actions[:, 1]))

            # 一个序列训练epochs次
            for _ in range(self.epochs):
                for i in range(robot_num):
                    states = torch.tensor(transition_dict['states'], dtype=torch.float)[:, i].to(self.device)
                    actions = torch.tensor(transition_dict['actions'], dtype=torch.float)[:, i].to(self.device)

                    vel_dists, rot_dists = self.states2probs(states)

                    # 当前策略在 t 时刻智能体处于状态 s 所采取的行为概率
                    log_prob = vel_dists.log_prob(actions[:, 0]) + rot_dists.log_prob(actions[:, 1])

                    ratio = torch.exp(log_prob - old_log_probs[i].detach())
                    surr1 = ratio * advantages[i]
                    surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantages[i]

                    # 策略网络的损失PPO-clip
                    actor_loss = torch.mean(-torch.min(surr1, surr2))
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

            for _ in range(self.epochs):
                critic_loss = torch.tensor(0.0).to(self.device)
                for i in range(robot_num):
                    states = torch.tensor(transition_dict['states'], dtype=torch.float)[:, i].to(self.device)
                    critic_loss = critic_loss + torch.mean(
                        F.mse_loss(self.critic(states), td_targets_set[i].detach()))

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
