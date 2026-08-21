import copy
from collections import deque
from pathlib import Path

import torch
from torch import optim

from env_sim.argument import DISTANCE_REWARD_WEIGHT
from rl.lagrange import Lagrange
from rl.util import *
from tqdm.auto import tqdm
from tensorboardX import SummaryWriter
from datetime import datetime
import numpy as np
import time
import sys

try:
    from project_paths import MODEL_DIR, RUNS_DIR
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from project_paths import MODEL_DIR, RUNS_DIR


class PPOLag:
    def __init__(self, env):
        self.args = parse_args()
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.args.cuda else "cpu")
        self.hooks = []

        # 实例化策略网络
        self.env = env
        self.agent = LagAgent().to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.args.learning_rate, eps=1e-5)
        self.robots_num = self.env.robots_num

        self.min_epsilon = self.args.min_epsilon
        self.decay_rate = self.args.epsilon_decay_rate
        self.max_epsilon = self.args.max_epsilon
        self.epsilon = self.max_epsilon

        self.lagrange = Lagrange(
            cost_limit=self.args.cost_limit,
            lagrangian_multiplier_init=self.args.lagrangian_multiplier_init,
            lagrangian_multiplier_lr=self.args.lagrangian_multiplier_lr,
        )

        self.rew_deque = deque(maxlen=50)
        self.cost_deque = deque(maxlen=50)
        self.ep_cost = 0.0
        self.ep_reward = 0.0

    def load_model(self):
        print('Loading agent', self.agent.name)
        model_path = MODEL_DIR / f"{self.agent.name}_{self.env.name}.pth"
        self.agent.load_state_dict(torch.load(model_path))

    def get_scan_data(self):
        laser_datas = []
        for rob in self.env.robots:
            laser_data = [i for i in rob.laser_buffer]
            laser_datas.append(torch.Tensor(laser_data).to(self.device))
        return torch.stack(laser_datas)

    def get_graph_data(self):
        graph_datas = []
        edges = []
        idx = 0
        for rob in self.env.robots:
            graph_data, edge = rob.memory_graph.get_data()
            graph_datas.append(graph_data.to(self.device))
            edges.append((edge + idx * torch.ones_like(edge)).to(self.device))
            idx += graph_data.shape[0]

        return torch.vstack(graph_datas), torch.vstack(edges)

    def cal_adv(self, next_value, values, rewards, dones, next_done):
        # GAE
        advantages = torch.zeros_like(rewards).to(self.device)
        lastgaelam = 0
        for t in reversed(range(self.args.num_steps)):
            if t == self.args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + self.args.gamma * nextvalues * nextnonterminal - values[t]
            advantages[
                t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values

        return advantages, returns

    def cal_cri_loss(self, newvalue, b_returns, mb_inds):
        newvalue = newvalue.view(-1)
        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
        return v_loss

    def update(self, FPS=3, lim=3):
        # 环境反馈的频率为 FPS / 240 Hz，FPS设置为10意味着24Hz
        time_stamp = "{0:%Y-%m-%d-%H}".format(datetime.now())
        run_name = RUNS_DIR / time_stamp / self.agent.name / self.env.name
        run_name.mkdir(parents=True, exist_ok=True)
        self.robots_num = self.env.robots_num
        torch.backends.cudnn.deterministic = self.args.torch_deterministic

        writer = SummaryWriter(log_dir=str(run_name))
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.args).items()])),
        )

        obs = torch.zeros((self.args.num_steps, self.robots_num) + self.env.observation_space.shape).to(self.device)
        actions = torch.zeros((self.args.num_steps, self.robots_num) + self.env.action_space.shape).to(self.device)
        logprobs = torch.zeros((self.args.num_steps, self.robots_num)).to(self.device)
        rewards = torch.zeros((self.args.num_steps, self.robots_num)).to(self.device)
        costs = torch.zeros((self.args.num_steps, self.robots_num)).to(self.device)
        dones = torch.zeros((self.args.num_steps, self.robots_num)).to(self.device)
        values_r = torch.zeros((self.args.num_steps, self.robots_num)).to(self.device)
        values_c = torch.zeros((self.args.num_steps, self.robots_num)).to(self.device)
        lasers = torch.zeros((self.args.num_steps, self.robots_num) + (ROBOT_LASER_BUFFER, LASER_NUM)).to(self.device)
        # graph_datas = []
        # edges = []

        # start the game
        global_step = 0
        start_time = time.time()
        obs_, _ = self.env.reset(lim=lim)
        next_obs = torch.Tensor(obs_).to(self.device)
        next_done = torch.zeros(self.robots_num).to(self.device)
        num_updates = self.args.total_timesteps // self.args.batch_size

        recent_best_reward = -np.inf
        max_reward = -np.inf
        reach_times = 0

        tq_bar = tqdm(range(1, num_updates + 1))
        convert = self.agent.convert_action_for_env

        ep_ret, ep_cost, ep_len = (
            np.zeros(self.robots_num),
            np.zeros(self.robots_num),
            np.zeros(self.robots_num),
        )

        for update in tq_bar:
            tq_bar.set_description(f'Episode [ {update} ]')

            if self.args.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * self.args.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            if self.args.if_maxstep_decay:
                self.env.max_simulate_steps = self.env.max_simulate_steps - 1 if self.env.max_simulate_steps > 3500 else 3500

            for step in range(0, self.args.num_steps):
                global_step += 1 * self.robots_num
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    laser_datas = self.get_scan_data()
                    # graph_data, edge = self.get_graph_data()

                    action, logprob, _, v_r, v_c = self.agent.get_action_and_value(laser_datas, next_obs[:, -4:],
                                                                                   update=True)
                    values_r[step] = v_r.flatten()
                    values_c[step] = v_c.flatten()

                actions[step] = action
                logprobs[step] = logprob
                lasers[step] = laser_datas
                # graph_datas.append(graph_data)
                # edges.append(edge)

                next_obs, reward, cost, te, tr, info = self.env.step(convert(action).cpu().numpy(), FPS=FPS)

                ep_ret += np.array(reward)
                ep_cost += np.array(cost)

                done = [i or j for i, j in zip(te, tr)]
                for idx, d in enumerate(done):
                    if d and self.agent.h_0 is not None:
                        self.agent.h_0[:, idx, :] = 0
                        self.agent.c_0[:, idx, :] = 0

                print(reward, cost, next_obs[:, -4:-2], self.env.robots[0].cur_action, action, v_r[0].item(),
                      v_c[0].item())

                if self.robots_num == self.env.reach_num:
                    reach_times += 1
                elif all(item for item in te) or any(item for item in tr):
                    reach_times = 0

                for idx, (d, time_out) in enumerate(zip(te, tr)):
                    if d or time_out:
                        self.rew_deque.append(ep_ret[idx])
                        self.cost_deque.append(ep_cost[idx])
                        self.ep_cost = np.mean(list(self.cost_deque))
                        self.ep_reward = np.mean(list(self.rew_deque))
                        ep_cost[idx] = 0.0
                        ep_ret[idx] = 0.0

                next_obs = self.env.reset(lim=lim, tr=tr, te=te)[0] if any(item for item in tr) or all(
                    item for item in te) else next_obs
                # next_obs = self.env.reset(lim=lim, tr=tr, te=te)[0] if all(item for item in te) else next_obs

                rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                costs[step] = torch.tensor(cost).to(self.device).view(-1)

                next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(done).to(self.device)

            # bootstrap value if not done
            with torch.no_grad():
                laser_datas = self.get_scan_data()
                graph_data, edge = self.get_graph_data()

                next_value_r, next_value_c = self.agent.get_value(laser_datas, next_obs[:, -4:], update=True)
                next_value_r, next_value_c = next_value_r.reshape(1, -1), next_value_c.reshape(1, -1)
                adv_r, ret_r = self.cal_adv(next_value_r, values_r, rewards, dones, next_done)
                adv_c, ret_c = self.cal_adv(next_value_c, values_c, costs, dones, next_done)

            self.lagrange.update_lagrange_multiplier(self.ep_cost)
            advantages = adv_r - self.lagrange.lagrangian_multiplier * adv_c
            advantages /= (self.lagrange.lagrangian_multiplier + 1)

            # flatten the batch
            b_obs = obs[:, :, -4:].reshape((-1, 4,))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + self.env.action_space.shape)
            b_advantages = advantages.reshape(-1)

            b_returns_r = ret_r.reshape(-1)
            b_returns_c = ret_c.reshape(-1)
            b_values_r = values_r.reshape(-1)

            b_lasers = lasers.reshape((-1,) + (ROBOT_LASER_BUFFER, LASER_NUM))

            b_obs = (b_obs - b_obs.mean()) / (b_obs.std() + 1e-8)
            b_lasers = (b_lasers - b_lasers.mean()) / (b_lasers.std() + 1e-8)

            # b_graphs = graph_datas
            # b_edge = edges

            # Optimizing the policy and value network
            b_inds = np.arange(self.args.batch_size)
            clipfracs = []

            print('reach times:', reach_times, 'episode:', update)
            for epoch in range(self.args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.args.batch_size, self.args.minibatch_size):

                    for name, param in self.agent.named_parameters():
                        if torch.isnan(param.data).any():
                            print(f"NaN detected after layer: {name}")

                    end = start + self.args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue_r, newvalue_c = self.agent.get_action_and_value(b_lasers[mb_inds],
                                                                                                     b_obs[mb_inds],
                                                                                                     b_actions[mb_inds],
                                                                                                     update=True)
                    logratio = (newlogprob - b_logprobs[mb_inds])
                    ratio = logratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if self.args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    loss_r = self.cal_cri_loss(newvalue_r, b_returns_r, mb_inds)
                    loss_c = self.cal_cri_loss(newvalue_c, b_returns_c, mb_inds)

                    entropy_loss = entropy.mean()

                    loss = pg_loss - self.args.ent_coef * entropy_loss + (loss_r + loss_c) * self.args.vf_coef

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()

            # edges.clear()
            # graph_datas.clear()
            y_pred, y_true = b_values_r.cpu().numpy(), b_returns_r.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/collision_num", self.env.collision_num, global_step)
            writer.add_scalar("charts/reach_num", self.env.reach_num, global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            writer.add_scalar("charts/epsilon", self.epsilon, global_step)
            writer.add_scalar("charts/a", self.agent.a.item(), global_step)

            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)

            train_reward = rewards.sum(0).mean().item()
            recent_best_reward = train_reward if train_reward > recent_best_reward else recent_best_reward
            writer.add_scalar("rewards/train rewards", train_reward, global_step)

            tq_bar.set_postfix({
                'lastMeanRewards': f'{train_reward:.2f}',
                'BEST': f'{recent_best_reward:.2f}',
                'bestTestRewards': f'{max_reward:.2f}'
            })
            model_path = MODEL_DIR / f"{self.agent.name}_{self.env.name}.pth"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.agent.state_dict(), model_path)
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * update)
            if reach_times >= 10:
                print(
                    'distance reward:{}, lim={}, episodes:{}, '.format(DISTANCE_REWARD_WEIGHT, lim, update))
                break
