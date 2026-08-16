import os
import math
import sys

from env_sim.argument import ROBOT_LASER_BUFFER, LASER_NUM
from rl.replay_buffer import ReplayBuffer

base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_path + '\\rl')

from rl.SAC_util import *
from rl.util import parse_args


class SAC:
    def __init__(self, env):
        self.args = parse_args()
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.args.cuda else "cpu")

        self.learning_rate = 1e-4

        # 实例化策略网络
        self.env = env
        self.act = self.act_target = ActorSAC([256, 256], self.env.observation_space.shape[0], 2).to(self.device)
        self.cri = self.cri_target = CriticTwin([256, 256], self.env.observation_space.shape[0], 2).to(self.device)
        self.act_optimizer = torch.optim.AdamW(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = torch.optim.AdamW(self.cri.parameters(), self.learning_rate)
        self.robots_num = self.env.robots_num

        self.min_epsilon = self.args.min_epsilon
        self.decay_rate = self.args.epsilon_decay_rate
        self.max_epsilon = self.args.max_epsilon
        self.epsilon = self.max_epsilon

        self.alpha_log = torch.tensor((-1,), dtype=torch.float32, requires_grad=True, device=self.device)  # trainable
        self.alpha_optimizer = torch.optim.AdamW((self.alpha_log,), lr=self.learning_rate * 4)
        self.criterion = torch.nn.SmoothL1Loss(reduction="mean")

        self.reward_scale = 1
        self.state_value_tau = 0
        self.last_state = None
        self.last_lasers = None
        self.repeat_times = 4
        self.soft_update_tau = 5e-3
        self.batch_size = 128
        self.target_entropy = 2
        self.obj_c = 1.0

        self.reach_times = 0

    def load_model(self):
        print('Loading agent', self.act.name)
        self.act.load_state_dict(torch.load('model/' + 'sac' + '.pth'))
        # self.cri.load_state_dict(torch.load('model/critic/' + 'sac' + '.pth'))

    def save_model(self):
        torch.save(self.act.state_dict(), 'model/' + 'sac' + '.pth')
        # torch.save(self.cri.state_dict(), 'model/critic/' + 'sac' + '.pth')

    def get_scan_data(self):
        laser_datas = []
        for rob in self.env.robots:
            laser_data = [i for i in rob.laser_buffer]
            laser_datas.append(torch.Tensor(laser_data).to(self.device))
        # laser_datas = torch.stack(laser_datas)
        return torch.stack(laser_datas)

    def update_avg_std_for_normalization(self, returns: Tensor):
        tau = self.state_value_tau
        if tau == 0:
            return

        returns_avg = returns.mean(dim=0)
        returns_std = returns.std(dim=0)
        self.cri.value_avg[:] = self.cri.value_avg * (1 - tau) + returns_avg * tau
        self.cri.value_std[:] = self.cri.value_std * (1 - tau) + returns_std * tau + 1e-4

    def explore_env(self, env, horizon_len: int, if_random: bool = False, lim: int = 0, FPS=3):
        states = torch.zeros((horizon_len, self.robots_num, self.env.observation_space.shape[0]),
                             dtype=torch.float32).to(
            self.device)
        actions = torch.zeros((horizon_len, self.robots_num, self.env.action_space.shape[0]), dtype=torch.float32).to(
            self.device)
        rewards = torch.zeros((horizon_len, self.robots_num), dtype=torch.float32).to(self.device)
        dones = torch.zeros((horizon_len, self.robots_num), dtype=torch.bool).to(self.device)
        lasers = torch.zeros((horizon_len, self.robots_num) + (ROBOT_LASER_BUFFER, LASER_NUM)).to(self.device)

        state = self.last_state
        laser_datas = self.last_lasers

        get_action = self.act.get_action
        for t in range(horizon_len):

            action = torch.rand(self.robots_num, 2) if if_random else get_action(laser_datas, state[:, -4:])
            states[t] = state

            ary_action = action.detach().cpu().numpy()
            ary_state, reward, te, tr, info = env.step(ary_action, FPS=FPS)  # next_state

            value = self.cri_target(laser_datas, state[:, -4:], action.to(self.device))

            print(reward, ary_state[:, -4:-2], self.env.robots[0].cur_action, ary_action, value[0].item())
            if self.robots_num == self.env.reach_num:
                self.reach_times += 1
            elif all(item for item in te) or any(item for item in tr):
                self.reach_times = 0

            done = [i or j for i, j in zip(te, tr)]
            ary_state = self.env.reset(tr=tr, te=te, lim=lim)[0] if any(item for item in tr) or all(
                item for item in te) else ary_state
            state = torch.as_tensor(ary_state, dtype=torch.float32, device=self.device)
            actions[t] = action
            rewards[t] = torch.tensor(reward).to(self.device).view(-1)
            lasers[t] = laser_datas

            laser_datas = self.get_scan_data()
            dones[t] = torch.Tensor(done).to(self.device)

        self.last_state = state
        self.last_lasers = laser_datas

        rewards *= self.reward_scale
        undones = 1.0 - dones.type(torch.float32)
        print('reach times: ', self.reach_times)
        return states, actions, rewards, undones, lasers

    def get_cumulative_rewards(self, rewards: Tensor, undones: Tensor) -> Tensor:
        returns = torch.empty_like(rewards)

        masks = undones * self.args.gamma
        horizon_len = rewards.shape[0]

        last_state = self.last_state
        last_laser = self.last_lasers

        next_action = self.act_target(last_laser, last_state[:, -4:])
        next_value = self.cri_target(last_laser, last_state[:, -4:], next_action).detach()
        for t in range(horizon_len - 1, -1, -1):
            returns[t] = next_value = rewards[t] + masks[t] * next_value
        return returns

    def get_obj_critic_raw(self, buffer, batch_size: int):
        with torch.no_grad():
            states, actions, rewards, undones, next_ss, lasers, next_lasers = buffer.sample(
                batch_size)  # next_ss: next states
            next_as, next_logprobs = self.act.get_action_logprob(next_lasers, next_ss[:, -4:])  # next actions
            next_qs = self.cri_target.get_q_min(next_lasers, next_ss[:, -4:], next_as)  # next q values

            alpha = self.alpha_log.exp().detach()
            q_labels = rewards + undones * self.args.gamma * (next_qs - next_logprobs * alpha)

        q1, q2 = self.cri.get_q1_q2(lasers, states[:, -4:], actions)
        obj_critic = self.criterion(q1, q_labels) + self.criterion(q2, q_labels)  # twin critics
        return obj_critic, states, lasers

    def optimizer_update(self, optimizer: torch.optim, objective: Tensor):
        """minimize the optimization objective via update the network parameters

        optimizer: `optimizer = torch.optim.SGD(net.parameters(), learning_rate)`
        objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
        """
        optimizer.zero_grad()
        objective.backward()
        nn.utils.clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.args.max_grad_norm)
        optimizer.step()

    @staticmethod
    def soft_update(target_net: torch.nn.Module, current_net: torch.nn.Module, tau: float):
        """soft update target network via current network

        target_net: update target network via current network to make training more stable.
        current_net: current network update via an optimizer
        tau: tau of soft target update: `target_net = target_net * (1-tau) + current_net * tau`
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def update_net(self, buffer: ReplayBuffer):
        with torch.no_grad():
            states, actions, rewards, undones, _ = buffer.add_item
            self.update_avg_std_for_normalization(
                returns=self.get_cumulative_rewards(rewards=rewards, undones=undones).reshape((-1,))
            )

        '''update network'''
        obj_critics = 0.0
        obj_actors = 0.0
        alphas = 0.0

        update_times = int(buffer.add_size * self.repeat_times)
        assert update_times >= 1
        update_a = 0
        for update_c in range(1, update_times + 1):
            '''objective of critic (loss function of critic)'''
            obj_critic, state, laser = self.get_obj_critic_raw(buffer, self.batch_size)
            obj_critics += obj_critic.item()
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
            self.obj_c = 0.995 * self.obj_c + 0.005 * obj_critic.item()  # for reliable_lambda

            reliable_lambda = math.exp(-self.obj_c ** 2)  # for reliable_lambda
            if update_a / update_c < 1 / (2 - reliable_lambda):  # auto TTUR
                '''objective of alpha (temperature parameter automatic adjustment)'''
                action_pg, log_prob = self.act.get_action_logprob(laser, state[:, -4:])  # policy gradient
                obj_alpha = (self.alpha_log * (self.target_entropy - log_prob).detach()).mean()
                self.optimizer_update(self.alpha_optimizer, obj_alpha)

                '''objective of actor'''
                alpha = self.alpha_log.exp().detach()
                alphas += alpha.item()
                with torch.no_grad():
                    self.alpha_log[:] = self.alpha_log.clamp(-16, 2)

                q_value_pg = self.cri_target(laser, state[:, -4:], action_pg).mean()
                obj_actor = (q_value_pg - log_prob * alpha).mean()
                obj_actors += obj_actor.item()
                self.optimizer_update(self.act_optimizer, -obj_actor)

        return obj_critics / update_times, obj_actors / update_times, alphas / update_times
