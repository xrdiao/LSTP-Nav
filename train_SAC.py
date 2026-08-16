import os
from datetime import datetime
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from env_sim.argument import DISTANCE_REWARD_WEIGHT
from rl.SAC_model import SAC

from env_sim.env_util import *
from rl.replay_buffer import ReplayBuffer

base_path = os.path.dirname(os.path.abspath(__file__))
urdf_path = base_path + '/env_sim/utils/data/turtlebot.urdf'


def train():
    horizon_len = 512
    torch.backends.cudnn.deterministic = True

    time_stamp = "{0:%Y-%m-%d-%H}".format(datetime.now())
    run_name = f"runs/{time_stamp}/SAC"

    env_arg = env_args()
    env_arg.render = True
    env_arg.random_obstacles = 0
    env_arg.x_lim = 0
    env_arg.y_lim = 0.
    env_arg.boundary = 0
    env_arg.lim = 0
    env_arg.robots_num = 1

    if env_arg.safe:
        env = SafeEnv(env_arg, urdf_path=urdf_path)
    else:
        env = MyEnv(env_arg, urdf_path=urdf_path)
    env.set_max_step(3500)

    print('robot_nums:', env_arg.robots_num, 'agent: SAC')
    agent = SAC(env=env)
    state, _ = env.reset(lim=env_arg.lim)
    state = torch.Tensor(state).to(agent.device)
    agent.last_state = state.detach()
    agent.last_lasers = agent.get_scan_data()

    writer = SummaryWriter(log_dir=run_name)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(agent.args).items()])),
    )

    buffer = ReplayBuffer(
        gpu_id=0,
        num_seqs=env_arg.robots_num,
        max_size=horizon_len * 2,
        state_dim=env.observation_space.shape[0],
        action_dim=2,
        if_use_per=False,
    )
    buffer_items = agent.explore_env(env, horizon_len * 2, if_random=True, lim=env_arg.lim)
    buffer.update(buffer_items)

    max_update_step = 2e6
    tq_bar = tqdm(range(int(max_update_step / horizon_len)))

    for step in tq_bar:
        buffer_items = agent.explore_env(env, horizon_len, lim=env_arg.lim)
        buffer.update(buffer_items)

        print('episode: ', step)
        torch.set_grad_enabled(True)
        obj_critic, obj_actor, alpha = agent.update_net(buffer)
        torch.set_grad_enabled(False)

        global_step = step * horizon_len * env_arg.robots_num
        writer.add_scalar("losses/value_loss", obj_critic, global_step)
        writer.add_scalar("losses/policy_loss", obj_actor, global_step)
        writer.add_scalar("charts/collision_num", env.collision_num, global_step)
        writer.add_scalar("charts/reach_num", env.reach_num, global_step)

        train_reward = buffer_items[2].sum(0).mean().item()
        writer.add_scalar("rewards/train rewards", train_reward, global_step)
        agent.save_model()

        if agent.reach_times >= 10:
            print(
                'distance reward:{}, lim={},the robot continuously reach goal 10 times, episodes:{}'.format(
                    DISTANCE_REWARD_WEIGHT, env_arg.lim, step))
            break


if __name__ == '__main__':
    train()
