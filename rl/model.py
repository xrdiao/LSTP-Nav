import os
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter

import sys

base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_path + '\\rl')

from rl.util import *
import torch.optim as optim
from tqdm.auto import tqdm


class PPO:
    def __init__(self, env, test_env=None):
        self.args = parse_args()
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.args.cuda else "cpu")

        # 实例化策略网络
        self.env = env
        self.agent = Agent(env).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.args.learning_rate, eps=1e-5)
        self.robots_num = 0

        if test_env is not None:
            self.test_env = test_env

    def evaluate(self, steps: int = 5000, times: int = 3):
        assert self.test_env is not None, "Please input a test environment"

        rewards = []
        next_obs, _ = self.test_env.reset()

        with torch.no_grad():
            for _ in range(times):
                r = []

                for step in range(steps):
                    action, _, _, value = self.agent.get_action_and_value(torch.Tensor(next_obs).to(self.device))

                    next_obs, reward, te, tr, info_ = self.test_env.step(action.cpu().numpy())
                    r.append(reward)
                    done = self.check_done(te=te, tr=tr)

                    if np.array(done).all() or step == steps - 1:
                        rewards.append(np.sum(np.array(r), axis=0))
                        break

        return np.mean(rewards)

    def check_done(self, te, tr, lim=0):
        assert len(te) == len(tr), "the length of te or tr are incorrect"

        done_te = True
        for i in range(len(te)):
            if not te[i]:
                done_te = False
                break

        done_tr = False
        for i in range(len(tr)):
            if tr[i]:
                done_tr = True
                break

        if done_te or done_tr:
            a, _ = self.env.reset(lim=lim, tr=tr, te=te)
        done = [i or j for i, j in zip(te, tr)]
        return done

    def load_model(self):
        self.agent.load_state_dict(torch.load('agent.pth'))

    # 训练
    def update(self, FPS=5):

        run_name = f"{self.args.gym_id}__{self.args.exp_name}"
        self.robots_num = self.env.robots_num

        writer = SummaryWriter(f"runs/{run_name}")
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

        # start the game
        global_step = 0
        start_time = time.time()
        obs_, _ = self.env.reset()
        next_obs = torch.Tensor(obs_).to(self.device)
        next_done = torch.zeros(self.robots_num).to(self.device)
        num_updates = self.args.total_timesteps // self.args.batch_size

        max_reward = -np.inf
        recent_best_reward = -np.inf

        tq_bar = tqdm(range(1, num_updates + 1))

        for update in tq_bar:
            # Annealing the rate if instructed to do so.
            tq_bar.set_description(f'Episode [ {update} ]')

            if self.args.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * self.args.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            # 在开局时的动作时效延长
            # frac_fps = 1 - (update - 1) / num_updates
            # FPS = int(FPS * frac_fps)
            # FPS = FPS if FPS >= 1 else 1

            for step in range(0, self.args.num_steps):
                global_step += 1 * self.robots_num
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                    # print(action[0].cpu().numpy())
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, te, tr, info = self.env.step(action.cpu().numpy(), FPS=FPS)
                done = self.check_done(te=te, tr=tr)
                rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(done).to(self.device)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)

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
            b_obs = obs.reshape((-1,) + self.env.observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + self.env.action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(self.args.batch_size)
            clipfracs = []
            for epoch in range(self.args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.args.batch_size, self.args.minibatch_size):
                    end = start + self.args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds],
                                                                                       b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
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

                if self.args.target_kl is not None:
                    if approx_kl > self.args.target_kl:
                        break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            # print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            writer.add_scalar("charts/FPS", FPS, global_step)

            train_reward = rewards.sum(0).mean().item()
            recent_best_reward = train_reward if train_reward > recent_best_reward else recent_best_reward

            tq_bar.set_postfix({
                'lastMeanRewards': f'{train_reward:.2f}',
                'BEST': f'{recent_best_reward:.2f}',
                "bestTestReward": f'{max_reward:.2f}'
            })

            test_reward = self.evaluate()
            writer.add_scalar("rewards/test rewards", test_reward, global_step)
            writer.add_scalar("rewards/train rewards", train_reward, global_step)

            torch.save(self.agent.state_dict(), 'agent.pth')
            if max_reward < test_reward:
                max_reward = test_reward
                print('\nupdate agent with train reward:{}, test reward:{}'.format(train_reward, test_reward))
