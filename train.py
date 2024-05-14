from env_sim.my_env import MyEnv
from rl.model import *
import numpy as np
import pybullet as p
from PIL import Image

device = torch.device('cuda') if torch.cuda.is_available() \
    else torch.device('cpu')

num_episodes = 10000  # 总迭代次数
gamma = 0.9  # 折扣因子
actor_lr = 1e-3  # 策略网络的学习率
critic_lr = 1e-2  # 价值网络的学习率
n_hiddens = 32  # 隐含层神经元个数
return_list = []  # 保存每个回合的return

env = MyEnv(render=True)
robot_nums = 5
for i in range(robot_nums):
    goal = [i + 0.5, 10]
    env.add_robot([i, 0, 0.01, 0, 0, i, 1.5], goal)

n_states = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]

agent = PPO(n_states=n_states,  # 状态数
            n_hiddens=n_hiddens,  # 隐含层数
            n_actions=n_actions,  # 动作数
            actor_lr=actor_lr,  # 策略网络学习率
            critic_lr=critic_lr,  # 价值网络学习率
            lmbda=0.95,  # 优势函数的缩放因子
            epochs=10,  # 一组序列训练的轮次
            eps=0.2,  # PPO中截断范围的参数
            gamma=gamma,  # 折扣因子
            device=device,
            embed_dim=32,
            num_heads=2,
            if_retrain=False
            )

print('training agent')
p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=0, cameraPitch=-89.9,
                             cameraTargetPosition=[2.5, 0, 0])
# camera_img = p.getCameraImage(320, 320)
# imgs = [Image.fromarray(camera_img[2])]

for i in range(num_episodes):
    state = env.reset()  # 环境重置
    done = np.zeros_like(env.robots, dtype=bool)  # 任务完成的标记
    episode_return = np.zeros(env.robots_num)  # 累计每回合的reward

    # 构造数据集，保存每个回合的状态数据
    transition_dict = {
        'states': [],
        'actions': [],
        'next_states': [],
        'rewards': [],
        'dones': [],
    }

    while not done.all():
        action = agent.take_action(state)  # 动作选择
        next_state, reward, done, _ = env.step(action)  # 环境更新
        # 保存每个时刻的状态\动作\...
        transition_dict['states'].append(state)
        transition_dict['actions'].append(action)
        transition_dict['next_states'].append(next_state)
        transition_dict['rewards'].append(reward)
        transition_dict['dones'].append(done)
        # 更新状态
        state = next_state
        # imgs.append(Image.fromarray(camera_img[2]))

        if env.step_num > 1000:
            break

    # imgs[0].save("test_pybullet_07.gif", save_all=True, append_images=imgs[1:], duration=20, loop=0)

    # 保存每个回合的reward
    return_list.append(np.sum(transition_dict['rewards'], axis=0))
    # 模型训练
    agent.update(transition_dict)

    # 打印回合信息
    print(f'iter:{i}, return:{np.mean(return_list)}')

    if i % 1000 == 0:
        print('--------------------saving %dth model------------------' % i)
        torch.save(agent.actor.state_dict(), './model/actor.pth')
        torch.save(agent.critic.state_dict(), './model/critic.pth')

torch.save(agent.actor.state_dict(), './model/actor.pth')
torch.save(agent.critic.state_dict(), './model/critic.pth')
