import copy
from collections import deque

import torch

from rl.model import *
import numpy as np

import os
from env_sim.my_env import MyEnv
import rvo.math as rvo_math
import pybullet as p

from env_sim.argument import *
from rvo.vector import Vector2
from rvo.simulator import Simulator

base_path = os.path.dirname(os.path.abspath(__file__))
urdf_path = base_path + '/env_sim/utils/data/turtlebot.urdf'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_preferred_velocities(simulator, goals):
    for i in range(simulator.num_agents):
        goal_vector = goals[i] - simulator.agents_[i].position_

        if rvo_math.abs_sq(goal_vector) > 1.0:
            goal_vector = rvo_math.normalize(goal_vector)

        simulator.set_agent_pref_velocity(i, goal_vector)


def obs2simulator(position: list):
    x, y = position[0], position[1]
    size = 1
    left = x - size
    right = x + size
    top = y + size
    bottom = y - size
    return [Vector2(right, bottom), Vector2(right, top), Vector2(left, top), Vector2(left, bottom)]


def evaluate(test_env, agent, steps: int = 3000, times: int = 3, lim: int = 5, debug: bool = False):
    rewards = []
    next_obs, _ = test_env.reset()
    next_obs = torch.Tensor(next_obs).to(device)

    with torch.no_grad():
        for _ in range(times):
            r = []

            for step in range(steps):
                laser_datas = []
                for rob in test_env.robots:
                    laser_data = [i for i in rob.laser_buffer]
                    laser_datas.append(torch.Tensor(laser_data).to(device))
                laser_datas = torch.stack(laser_datas)

                action, value = agent.get_deterministic_action(laser_datas, next_obs[:, -4:])
                action = action.squeeze(1)
                next_obs, reward, te, tr, info_ = test_env.step(action.cpu().numpy())
                next_obs = torch.Tensor(next_obs).to(device)
                r.append(reward)
                done = test_env.check_done(te=te, tr=tr, lim=lim)
                if debug:
                    print('action:', action, 'reward:', reward, 'value', value)

                if np.array(done).all() or step == steps - 1:
                    rewards.append(np.sum(np.array(r), axis=0))
                    break

    return np.mean(rewards)


def step_both(env, actions, simulator):
    # update pybullet
    next_obs, reward, te, tr, info = env.step(actions)
    # done = envir.check_done(te=te, tr=tr, lim=lim)

    # update simulator
    for agentNo, agent in enumerate(env.robots):
        obs_dict = agent.get_vel_and_pos()
        vel, pos = obs_dict['vel'], obs_dict['pos']
        angle = agent.get_forward_vector()[:2]

        simulator.agents_[agentNo].velocity_ = Vector2(abs(vel[0]) * angle[0], abs(vel[0]) * angle[1])
        simulator.agents_[agentNo].position_ = Vector2(pos[0], pos[1])

    return next_obs, reward, te, tr, info


def get_orca_vel(env, simulator, goals):
    set_preferred_velocities(simulator, goals)
    actions = []
    simulator.kd_tree_.build_agent_tree()

    for agentNo in range(simulator.num_agents):
        simulator.agents_[agentNo].compute_neighbors()
        simulator.agents_[agentNo].compute_new_velocity()

        action = simulator.agents_[agentNo].new_velocity_
        angle = env.robots[agentNo].follow_vector_angle([action.x, action.y])
        a = [abs(action), angle]
        actions.append(a)
    return actions


def train():
    args = parse_args()

    # --------------环境设置---------------
    render = False
    # 初始化环境
    env = MyEnv(render=render, urdf_path=urdf_path)
    test_env = MyEnv(render=False, urdf_path=urdf_path)

    simulator = Simulator()
    p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=0, cameraPitch=-89.9,
                                 cameraTargetPosition=[0, 0, 0])

    simulator.set_time_step(env.time_step)

    # 放置障碍物
    # obstacles = [[0, 0], [0, 3], [0, -3], [3, 0], [-3, 0]]
    # for obstacle in obstacles:
    #     env.place_cube(obstacle)
    #     obs = obs2simulator(obstacle)
    #     simulator.add_obstacle(obs)
    # simulator.process_obstacles()

    # 放置机器人
    robots_num = 10
    lim = 5
    for _ in range(robots_num):
        test_env.add_random_robot(lim=lim)
    for i in range(robots_num):
        env.add_random_robot(lim=lim)
    env.show_goal_point()

    simulator.set_agent_defaults(LASER_LENGTH, 10, 10.0, 10.0, ROBOT_WIDTH + 0.1, MAX_SPEED, Vector2(0.0, 0.0))
    for rob in env.robots:
        simulator.add_agent(Vector2(rob.init_pos[0], rob.init_pos[1]))

    # ---------------策略设置---------------
    agent = AttentionAgent(env).to(device)
    # agent = Agent().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ---------------添加Writer---------------
    run_name = f"pretrained_policy" + agent.name
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # ---------------训练前准备---------------
    global_step = 0
    obs = torch.zeros((args.num_steps, robots_num) + env.observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, robots_num) + env.action_space.shape).to(device)
    rewards = torch.zeros((args.num_steps, robots_num)).to(device)
    dones = torch.zeros((args.num_steps, robots_num)).to(device)
    values = torch.zeros((args.num_steps, robots_num)).to(device)
    lasers = torch.zeros((args.num_steps, robots_num) + (3, LASER_NUM)).to(device)

    obs_, _ = env.reset()
    next_obs = torch.Tensor(obs_).to(device)
    next_done = torch.zeros(robots_num).to(device)
    num_updates = args.total_timesteps // args.batch_size

    goals = []
    for rob in env.robots:
        goals.append(Vector2(rob.target_pos[0], rob.target_pos[1]))

    max_reward = -np.inf
    tq_bar = tqdm(range(1, num_updates + 1))

    # ---------------开始训练---------------
    for update in tq_bar:
        tq_bar.set_description(f'Episode [ {update} ]')

        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
            if lrnow <= 2e-5:
                args.anneal_lr = False

        for step in range(0, args.num_steps):
            global_step += 1 * robots_num
            obs[step] = next_obs
            dones[step] = next_done

            # 执行orca的策略
            with torch.no_grad():
                laser_datas = []
                for rob in env.robots:
                    laser_data = [i for i in rob.laser_buffer]
                    laser_datas.append(torch.Tensor(laser_data).to(device))
                laser_datas = torch.stack(laser_datas)
                value = agent.get_value(laser_datas, next_obs[:, -4:])
                values[step] = value.flatten()

            a = get_orca_vel(env, simulator, goals)
            actions[step] = torch.Tensor(a).to(device)
            lasers[step] = laser_datas

            next_obs, reward, te, tr, info = step_both(env, a, simulator)

            done = env.check_done(te=te, tr=tr, lim=lim)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            for i, d in enumerate(done):
                if d:
                    rob = env.robots[i]
                    goals[i] = Vector2(rob.target_pos[0], rob.target_pos[1])

        with torch.no_grad():
            laser_datas = []
            for rob in env.robots:
                laser_data = [i for i in rob.laser_buffer]
                laser_datas.append(torch.Tensor(laser_data).to(device))
            laser_datas = torch.stack(laser_datas)
            next_value = agent.get_value(laser_datas, next_obs[:, -4:]).reshape(1, -1)
            targets = torch.zeros_like(rewards).to(device)

            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                target = rewards[t] + args.gamma * nextvalues * nextnonterminal
                targets[t] = target

        # ---------------批次训练---------------
        b_obs = obs.reshape((-1,) + env.observation_space.shape)
        b_actions = actions.reshape((-1,) + env.action_space.shape)
        b_targets = targets.reshape(-1)
        b_lasers = lasers.reshape((-1,) + (3, LASER_NUM))
        b_inds = np.arange(args.batch_size)

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # 策略用监督学习的方法训练（也可以理解成模仿学习，但我不知道模仿学习是什么流程），值函数用DQN
                action, value = agent.get_deterministic_action(b_lasers[mb_inds], b_obs[mb_inds][:, -4:])
                pg_loss = F.mse_loss(action, b_actions[mb_inds])
                v_loss = F.mse_loss(value, b_targets[mb_inds])
                loss = pg_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        test_reward = evaluate(test_env=test_env, agent=agent, lim=lim, debug=False, times=1)
        writer.add_scalar("rewards/test rewards", test_reward, global_step)

        tq_bar.set_postfix({
            "bestTestReward": f'{max_reward:.2f}'
        })

        torch.save(agent.state_dict(), agent.name + '.pth')
        if max_reward < test_reward:
            max_reward = test_reward
            print('\nbest test reward:{}'.format(test_reward))


def test():
    env = MyEnv(render=True, urdf_path=urdf_path)
    p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=0, cameraPitch=-89.9,
                                 cameraTargetPosition=[0, 0, 0])
    robots_num = 5
    lim = 5
    agent = AttentionAgent(env).to(device)
    agent.load_state_dict(torch.load(agent.name+'.pth'))

    for _ in range(robots_num):
        env.add_random_robot(lim=lim)
    env.show_goal_point()
    while True:
        evaluate(env, agent)


if __name__ == '__main__':
    # train()
    test()
