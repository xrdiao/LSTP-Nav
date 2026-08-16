from __future__ import annotations

import time
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm.auto import tqdm

try:
    from .agent import build_agent
    from .config import VUCANavConfig
    from .env import VUCANavEnv
except ImportError:  # pragma: no cover
    from agent import build_agent
    from config import VUCANavConfig
    from env import VUCANavEnv


class PPOTrainer:
    def __init__(self, env: VUCANavEnv, cfg: VUCANavConfig):
        self.env = env
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")
        self.agent = build_agent(cfg).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=cfg.learning_rate, eps=1e-5)
        self.learning_rate = cfg.learning_rate
        self.robots_num = env.robots_num

    def get_scan_data(self):
        laser_datas = []
        for robot in self.env.robots:
            laser_data = np.asarray(robot.laser_buffer[-1], dtype=np.float32)
            laser_datas.append(torch.tensor(laser_data, dtype=torch.float32, device=self.device))
        return torch.stack(laser_datas)

    def cal_advantage(self, next_value, values, rewards, dones, next_done):
        advantages = torch.zeros_like(rewards, device=self.device)
        lastgaelam = 0
        for t in reversed(range(self.cfg.n_steps)):
            nextnonterminal = 1 - (dones[t + 1] if t < self.cfg.n_steps - 1 else next_done)
            nextvalues = next_value if t == self.cfg.n_steps - 1 else values[t + 1]
            delta = rewards[t] + self.cfg.gamma * nextvalues * nextnonterminal - values[t]
            lastgaelam = delta + self.cfg.gamma * self.cfg.gae_lambda * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + values
        return advantages, returns

    def train(self):
        time_stamp = "{0:%Y-%m-%d-%H}".format(datetime.now())
        run_name = str(self.cfg.log_dir / time_stamp / self.agent.name / self.env.name)
        writer = SummaryWriter(log_dir=run_name)

        obs = torch.zeros((self.cfg.n_steps, self.robots_num, self.cfg.state_dim), device=self.device)
        actions = torch.zeros((self.cfg.n_steps, self.robots_num, self.env.num_actions), device=self.device)
        logprobs = torch.zeros((self.cfg.n_steps, self.robots_num), device=self.device)
        rewards = torch.zeros((self.cfg.n_steps, self.robots_num), device=self.device)
        dones = torch.zeros((self.cfg.n_steps, self.robots_num), device=self.device)
        values = torch.zeros((self.cfg.n_steps, self.robots_num), device=self.device)
        lasers = torch.zeros((self.cfg.n_steps, self.robots_num, self.cfg.lidar_num), device=self.device)

        tra_reward = np.zeros(self.robots_num, dtype=np.float32)
        reward_deque = deque(maxlen=100)
        reward_deque.append(0.0)

        global_step = 0
        start_time = time.time()
        obs_np, _ = self.env.reset(seed=self.cfg.seed)
        next_obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
        next_done = torch.zeros(self.robots_num, dtype=torch.float32, device=self.device)

        num_updates = max(self.cfg.total_timesteps // self.cfg.batch_size, 1)
        recent_best_reward = -np.inf
        reach_times = 0
        v_loss = torch.zeros(1, device=self.device)
        pg_loss = torch.zeros(1, device=self.device)
        entropy_loss = torch.zeros(1, device=self.device)

        tq_bar = tqdm(range(1, num_updates + 1))
        for update in tq_bar:
            tq_bar.set_description(f"Episode [ {update} ]")

            for step in range(self.cfg.n_steps):
                global_step += self.robots_num
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    laser_datas = self.get_scan_data()
                    action, logprob, _, value = self.agent.get_action_and_value(laser_datas, next_obs)
                    values[step] = value.flatten()

                actions[step] = action
                logprobs[step] = logprob
                lasers[step] = laser_datas

                env_actions = self.agent.convert_action_for_env(action).cpu().numpy()
                env_actions[:, 0] = np.clip(env_actions[:, 0], self.cfg.min_linear_speed, self.cfg.preferred_velocity)
                next_obs_np, reward, te, tr, _ = self.env.step(env_actions)

                tra_reward += reward
                next_done = torch.logical_or(
                    torch.tensor(te, device=self.device),
                    torch.tensor(tr, device=self.device),
                ).float()

                done_indices = next_done.bool().cpu().numpy()
                if np.any(done_indices):
                    reward_deque.extend(tra_reward[done_indices].tolist())
                    tra_reward = np.where(done_indices, 0.0, tra_reward)

                reach_times += int(self.robots_num == self.env.reach_num)
                if np.all(te) or np.any(tr):
                    reach_times = 0

                if np.any(tr) or np.any(te):
                    next_obs_np, _ = self.env.reset(tr=tr, te=te)

                rewards[step] = torch.tensor(reward, dtype=torch.float32, device=self.device).view(-1)
                next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                laser_datas = self.get_scan_data()
                next_value = self.agent.get_value(laser_datas, next_obs).reshape(1, -1)
                advantages, returns = self.cal_advantage(next_value, values, rewards, dones, next_done)

            b_obs = obs.reshape((-1, self.cfg.state_dim))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1, self.env.num_actions))
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)
            b_lasers = lasers.reshape((-1, self.cfg.lidar_num))

            b_inds = np.arange(self.cfg.batch_size)
            clipfracs = []

            for _ in range(self.cfg.n_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.cfg.batch_size, self.cfg.minibatch_size):
                    end = start + self.cfg.minibatch_size
                    mb_inds = b_inds[start:end]
                    if len(mb_inds) == 0:
                        continue

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                        b_lasers[mb_inds], b_obs[mb_inds], b_actions[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    mb_advantages = b_advantages[mb_inds]
                    if self.cfg.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    with torch.inference_mode():
                        kl_mean = ((ratio - 1) - logratio).mean().item()
                        clipfracs.append(((ratio - 1.0).abs() > self.cfg.clip_range).float().mean().item())
                        if kl_mean > self.cfg.target_kl * 1.5:
                            self.learning_rate *= 0.9
                        elif kl_mean < self.cfg.target_kl * 0.5:
                            self.learning_rate *= 1.1
                        self.learning_rate = np.clip(self.learning_rate, 1e-5, 1e-3)
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.learning_rate

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - self.cfg.clip_range, 1 + self.cfg.clip_range
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    newvalue = newvalue.view(-1)
                    if self.cfg.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.cfg.clip_range,
                            self.cfg.clip_range,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.cfg.ent_coef * entropy_loss + self.cfg.vf_coef * v_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.cfg.max_grad_norm)
                    self.optimizer.step()

            y_pred = b_values.detach().cpu().numpy()
            y_true = b_returns.detach().cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            train_reward = sum(reward_deque) / len(reward_deque)
            rollout_reward = rewards.mean().item()
            recent_best_reward = max(recent_best_reward, train_reward)

            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs) if clipfracs else 0.0, global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar("charts/collision_num", self.env.collision_num, global_step)
            writer.add_scalar("charts/reach_num", self.env.reach_num, global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            writer.add_scalar("rewards/train_rewards", train_reward, global_step)
            writer.add_scalar("rewards/rollout_mean_reward", rollout_reward, global_step)

            tq_bar.set_postfix(
                {
                    "lastMeanRewards": f"{train_reward:.2f}",
                    "rolloutMean": f"{rollout_reward:.2f}",
                    "BEST": f"{recent_best_reward:.2f}",
                    "LR": f"{self.learning_rate:.2e}",
                    "Collisions": self.env.collision_num,
                }
            )

            if update % self.cfg.save_interval_updates == 0:
                torch.save(self.agent.state_dict(), self.cfg.model_dir / f"{self.agent.name}_{self.env.name}.pth")

            if reach_times >= self.cfg.early_stop_reach_times:
                break

        writer.close()


def main():
    cfg = VUCANavConfig()
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    env = VUCANavEnv(cfg)
    trainer = PPOTrainer(env, cfg)
    print(
        "train_config:",
        {
            "device": str(trainer.device),
            "robots_num": cfg.robots_num,
            "obstacle_num": cfg.obstacle_num,
            "state_dim": cfg.state_dim,
            "laser_shape": (cfg.lidar_num,),
            "batch_size": cfg.batch_size,
            "minibatch_size": cfg.minibatch_size,
            "policy_name": cfg.policy_name,
            "paper_reward_weight": cfg.paper_reward_weight,
        },
        flush=True,
    )
    trainer.train()
    env.close()


if __name__ == "__main__":
    main()
