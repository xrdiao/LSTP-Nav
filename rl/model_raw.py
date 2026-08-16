import os
import time
from datetime import datetime
from rl.normalization import RewardScaling
from rl.util_raw import *
import torch.optim as optim
from tqdm.auto import tqdm
from env_sim.argument import ROBOT_LASER_BUFFER, DISTANCE_REWARD_WEIGHT
from tensorboardX import SummaryWriter
from collections import deque
import sys

base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_path + '\\rl')


class PPO:
    def __init__(self, env, policy='AttentionAgent', cli_args=None):
        self.args = parse_args(cli_args)
        self.learning_rate = self.args.learning_rate

        # self.args.vf_coef = 0.3
        self.args.reward_scaling = True
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.args.cuda else "cpu")
        self.hooks = []

        # 实例化策略网络
        self.env = env
        agents = {
            # "Conv1dAgent": Conv1dAgent,
            # "Agent": Agent,
            # "LinearAgent": LinearAgent,
            "LstmAgent": LstmAgent,
            "AttentionAgent": AttentionAgent,
            "IJRRAgent": IJRRAgent
        }
        self.agent = agents[policy]().to(self.device)

        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.args.learning_rate, eps=1e-5)
        self.robots_num = self.env.robots_num

        self.reward_scaling = RewardScaling(1, self.args.gamma)

    def load_model(self, model_path=None):
        model_path = model_path or ('model/' + self.agent.name + '_' + self.env.name + '.pth')
        print('Loading agent', self.agent.name, 'from', model_path)
        self.agent.load_state_dict(torch.load(model_path))

    def get_scan_data(self):
        laser_datas = []
        for rob in self.env.robots:
            laser_data = list(rob.laser_buffer)
            laser_datas.append(torch.Tensor(laser_data).to(self.device))
        # laser_datas = torch.stack(laser_datas)
        return torch.stack(laser_datas)

    def cal_advantage(self, next_value, values, rewards, dones, next_done):
        advantages = torch.zeros_like(rewards).to(self.device)
        lastgaelam = 0

        for t in reversed(range(self.args.num_steps)):
            # 判断是否为轨迹终止状态
            nextnonterminal = 1-(dones[t + 1] if t < self.args.num_steps - 1 else next_done)
            
            # 选择下一状态的值估计
            nextvalues = next_value if t == self.args.num_steps - 1 else values[t + 1]
            
            # 计算TD误差
            delta = rewards[t] + self.args.gamma * nextvalues * nextnonterminal - values[t]
            
            # 更新GAE（混合当前TD误差和之前GAE）
            lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam
    
        returns = advantages + values
        return advantages, returns

    # 训练
    def train(self, random_robot: float = 3, run_name=None, model_save_path=None, show_progress=True):
        # 环境反馈的频率为 FPS / 240 Hz，FPS设置为10意味着24Hz
        time_stamp = "{0:%Y-%m-%d-%H}".format(datetime.now())
        run_name = run_name or f"runs/{time_stamp}/{self.agent.name}/{'circle'}"
        model_save_path = model_save_path or f'model/{self.agent.name}_{self.env.name}.pth'
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

        tra_reward = np.zeros(self.robots_num)
        reward_deque = deque(maxlen=100)
        reward_deque.append(0)

        # start the game
        global_step = 0
        start_time = time.time()
        obs_, _ = self.env.reset()
        next_obs = torch.Tensor(obs_).to(self.device)
        next_done = torch.zeros(self.robots_num).to(self.device)
        num_updates = self.args.total_timesteps // self.args.batch_size

        recent_best_reward = -np.inf
        reach_times = 0

        tq_bar = tqdm(range(1, num_updates + 1)) if show_progress else range(1, num_updates + 1)
        convert = self.agent.convert_action_for_env

        v_loss = torch.zeros(1)
        pg_loss = torch.zeros(1)
        entropy_loss = torch.zeros(1)

        for update in tq_bar:
            if show_progress:
                tq_bar.set_description(f'Episode [ {update} ]')

            for step in range(0, self.args.num_steps):
                global_step += 1 * self.robots_num
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    laser_datas = self.get_scan_data()
                    action, logprob, _, value = self.agent.get_action_and_value(laser_datas, next_obs[:, :4])
                    values[step] = value.flatten()

                actions[step] = action
                logprobs[step] = logprob
                lasers[step] = laser_datas

                action = convert(action).cpu().numpy()
                action[:, 0] = np.clip(action[:, 0], 0.05, 1)
                next_obs, reward, te, tr, info = self.env.step(action)

                tra_reward += reward

                # 4. 终止判断（向量化）
                next_done = torch.logical_or(
                    torch.tensor(te, device=self.device), 
                    torch.tensor(tr, device=self.device)
                ).to(self.device).float()

                # 获取需要重置的环境索引（向量化操作）
                done_indices = next_done.bool().cpu().numpy()

                if np.any(done_indices):
                    reward_deque.extend(tra_reward[done_indices].tolist())
                    tra_reward = np.where(done_indices, 0.0, tra_reward)

                # print(reward, next_obs[:, -4:-2], self.env.robots[0].cur_action, actions[step][0].cpu().numpy(),value[0].item())

                reach_times += int(self.robots_num == self.env.reach_num)
                if all(te) or any(tr):
                    reach_times = 0

                need_reset = any(tr) or any(te)
                next_obs = self.env.reset(tr=tr, te=te)[0] if need_reset else next_obs

                rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs = torch.Tensor(next_obs).to(self.device)

            # bootstrap value if not done
            with torch.no_grad():
                laser_datas = self.get_scan_data()
                next_value = self.agent.get_value(laser_datas, next_obs[:, :4]).reshape(1, -1)
                advantages, returns = self.cal_advantage(next_value, values, rewards, dones, next_done)

            # flatten the batch
            b_obs = obs[:, :, :4].reshape((-1, 4,))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + self.env.action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)
            b_lasers = lasers.reshape((-1,) + (ROBOT_LASER_BUFFER, LASER_NUM))

            # Optimizing the policy and value network
            b_inds = np.arange(self.args.batch_size)
            clipfracs = []

            # print('reach times:', reach_times, 'episode:', update)
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

                    mb_advantages = b_advantages[mb_inds]
                    if self.args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    with torch.inference_mode():
                        kl_mean = ((ratio - 1) - logratio).mean().item()
                        clipfracs += [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]

                        # 动态调整学习率（带温度系数）
                        if kl_mean > self.args.target_kl * 1.5:  # 建议target_kl=0.01
                            self.learning_rate *= 0.9
                        elif kl_mean < self.args.target_kl * 0.5:
                            self.learning_rate *= 1.1
                        self.learning_rate = np.clip(self.learning_rate, 1e-5, 1e-3)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate

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
            # writer.add_scalar("charts/a", self.agent.a.item(), global_step)

            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)

            train_reward = sum(reward_deque) / len(reward_deque)
            recent_best_reward = train_reward if train_reward > recent_best_reward else recent_best_reward
            writer.add_scalar("rewards/train rewards", train_reward, global_step)

            if show_progress:
                tq_bar.set_postfix({
                    'lastMeanRewards': f'{train_reward:.2f}',
                    'BEST': f'{recent_best_reward:.2f}',
                    'LR': f'{self.learning_rate:.2e}',
                    'Collisions': self.env.collision_num
                })
            if update % 50 == 0:
                torch.save(self.agent.state_dict(), model_save_path)
            if reach_times >= 15:
                print(
                    'distance reward:{}, random_robot={}, episodes:{}, '.format(DISTANCE_REWARD_WEIGHT, random_robot, update))
                break
        torch.save(self.agent.state_dict(), model_save_path)
        writer.close()
