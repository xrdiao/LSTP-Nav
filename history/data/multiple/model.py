import os
import time
from datetime import datetime
from rl.normalization import RewardScaling
from rl.util import *
import torch.optim as optim
from tqdm.auto import tqdm
from env_sim.argument import ROBOT_LASER_BUFFER, DISTANCE_REWARD_WEIGHT
from tensorboardX import SummaryWriter

import sys

base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_path + '\\rl')


class PPO:
    def __init__(self, env):
        self.args = parse_args()

        self.args.vf_coef = 0.2
        self.args.reward_scaling = False
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.args.cuda else "cpu")
        self.hooks = []

        # 实例化策略网络
        self.env = env
        # self.agent = Conv1dAgent().to(self.device)
        # self.agent = Agent().to(self.device)
        # self.agent = LinearAgent().to(self.device)
        # self.agent = LstmAgent().to(self.device)
        self.agent = AttentionAgent().to(self.device)
        # self.agent = IJRRAgent().to(self.device)

        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.args.learning_rate, eps=1e-5)
        self.robots_num = self.env.robots_num

        self.reward_scaling = RewardScaling(1, self.args.gamma)

    def load_model(self):
        print('Loading agent', self.agent.name)
        self.agent.load_state_dict(torch.load('model/' + self.agent.name + '_' + self.env.name + '.pth'))

    def get_scan_data(self):
        laser_datas = []
        for rob in self.env.robots:
            laser_data = [i for i in rob.laser_buffer]
            laser_datas.append(torch.Tensor(laser_data).to(self.device))
        # laser_datas = torch.stack(laser_datas)
        return torch.stack(laser_datas)

    # 训练
    def update(self, lim: float = 3):
        # 环境反馈的频率为 FPS / 240 Hz，FPS设置为10意味着24Hz
        time_stamp = "{0:%Y-%m-%d-%H}".format(datetime.now())
        run_name = f"runs/{time_stamp}/{self.agent.name}/{self.env.name}"
        self.robots_num = self.env.robots_num
        torch.backends.cudnn.deterministic = self.args.torch_deterministic

        writer = SummaryWriter(log_dir=run_name)
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.args).items()])),
        )

        obs = torch.zeros((self.args.num_steps, self.robots_num) + self.env.observation_space.shape).to(self.device)
        actions = torch.zeros((self.args.num_steps, self.robots_num) + self.env.action_space.shape).to(self.device)
        logprobs = torch.zeros((self.args.num_steps, self.robots_num)).to(self.device)
        rewards = torch.zeros((self.args.num_steps, self.robots_num)).to(self.device)
        dones = torch.zeros((self.args.num_steps, self.robots_num)).to(self.device)
        values = torch.zeros((self.args.num_steps, self.robots_num)).to(self.device)
        lasers = torch.zeros((self.args.num_steps, self.robots_num) + (ROBOT_LASER_BUFFER, LASER_NUM)).to(self.device)

        # start the game
        global_step = 0
        start_time = time.time()
        obs_, _ = self.env.reset(lim=lim)
        next_obs = torch.Tensor(obs_).to(self.device)
        next_done = torch.zeros(self.robots_num).to(self.device)
        num_updates = self.args.total_timesteps // self.args.batch_size

        recent_best_reward = -np.inf
        reach_times = 0

        tq_bar = tqdm(range(1, num_updates + 1))
        convert = self.agent.convert_action_for_env

        v_loss = torch.zeros(1)
        pg_loss = torch.zeros(1)
        entropy_loss = torch.zeros(1)

        for update in tq_bar:
            tq_bar.set_description(f'Episode [ {update} ]')

            if self.args.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * self.args.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.args.num_steps):
                global_step += 1 * self.robots_num
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    laser_datas = self.get_scan_data()
                    action, logprob, _, value = self.agent.get_action_and_value(laser_datas, next_obs[:, -4:])
                    values[step] = value.flatten()

                actions[step] = action
                logprobs[step] = logprob
                lasers[step] = laser_datas

                action = convert(action).cpu().numpy()
                # action[:, 0] = np.clip(action[:, 0], 0.05, 1)
                next_obs, reward, te, tr, info = self.env.step(action)
                done = [i or j for i, j in zip(te, tr)]

                if self.args.reward_scaling:
                    reward = np.clip(self.reward_scaling(reward), -10, 10)
                    for idx, d in enumerate(done):
                        if d:
                            self.reward_scaling.reset(idx)
                            print('reset scaling')

                print(reward, next_obs[:, -4:-2], self.env.robots[0].cur_action, actions[step][0].cpu().numpy(),value[0].item())

                if self.robots_num == self.env.reach_num:
                    reach_times += 1
                elif all(item for item in te) or any(item for item in tr):
                    reach_times = 0

                next_obs = self.env.reset(lim=lim, tr=tr, te=te)[0] if any(item for item in tr) or all(
                    item for item in te) else next_obs
                # next_obs = self.env.reset(lim=lim, tr=tr, te=te)[0] if all(item for item in te) else next_obs

                rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(done).to(self.device)

            # bootstrap value if not done
            with torch.no_grad():
                laser_datas = self.get_scan_data()
                next_value = self.agent.get_value(laser_datas, next_obs[:, -4:]).reshape(1, -1)

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

            # flatten the batch
            b_obs = obs[:, :, -4:].reshape((-1, 4,))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + self.env.action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)
            b_lasers = lasers.reshape((-1,) + (ROBOT_LASER_BUFFER, LASER_NUM))

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

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_lasers[mb_inds],
                                                                                       b_obs[mb_inds],
                                                                                       b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
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

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.args.clip_coef,
                            self.args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()

                    loss = pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("charts/collision_num", self.env.collision_num, global_step)
            writer.add_scalar("charts/reach_num", self.env.reach_num, global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
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
            })
            torch.save(self.agent.state_dict(), 'model/' + self.agent.name + '_' + self.env.name + '.pth')
            if reach_times >= 15:
                print(
                    'distance reward:{}, lim={}, episodes:{}, '.format(DISTANCE_REWARD_WEIGHT, lim, update))
                break
