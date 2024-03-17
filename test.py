from env_sim.my_env import MyEnv
from rl.model import *
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() \
    else torch.device('cpu')

# ----------------------------------------- #
# 参数设置
# ----------------------------------------- #

num_episodes = 1000  # 总迭代次数
gamma = 0.9  # 折扣因子
actor_lr = 1e-3  # 策略网络的学习率
critic_lr = 1e-2  # 价值网络的学习率
n_hiddens = 16  # 隐含层神经元个数
return_list = []  # 保存每个回合的return

env = MyEnv(render=True)
goal_1 = [0.5, 0]
goal_2 = [0.5, 1]
env.add_robot([0, 0, 0.01, 0, 0, 0, 1.5], goal_1)
env.add_robot([0, 2, 0.01, 0, 0, 0, 1], goal_2)

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
            if_retrain=True
            )

state = env.reset()  # 环境重置

done = np.zeros_like(env.robots, dtype=bool)
while not done.all():
    action = agent.take_action(state)  # 动作选择
    next_state, reward, done, _ = env.step(action)  # 环境更新
    state = next_state
