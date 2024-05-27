from env_sim.my_env import MyEnv
from rl.model import *
import numpy as np
import os
import pybullet as p

base_path = os.path.dirname(os.path.abspath(__file__))
urdf_path = base_path + '/env_sim/utils/data/turtlebot.urdf'


class SimpleAgent(nn.Module):
    def __init__(self):
        super(SimpleAgent, self).__init__()
        self.hidden_size = 4
        self.actor = nn.Sequential(nn.Linear(2, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, 2))

    def forward(self, x):
        return self.actor(x)


device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

agent = SimpleAgent().to(device)
optimizer = optim.Adam(agent.parameters())

for _ in tqdm(range(20000)):
    x = (torch.rand(2) * 2).to(device)
    y_ = agent.forward(x)
    loss = F.mse_loss(y_, x)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

render = True
env = MyEnv(render=render, urdf_path=urdf_path)
p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=0, cameraPitch=-89.9,
                             cameraTargetPosition=[0, 0, 0])

robot_nums = 1
lim = 5

print('robot_nums:', robot_nums)
for i in range(robot_nums):
    env.add_random_robot(lim=lim)

actions = np.zeros([env.robots_num, 2])
next_obs, _ = env.reset()
env.show_goal_point()

# 只收集了一个机器人的rewards
with torch.no_grad():
    for times in range(3):
        print('times:', times)
        r = []

        while True:
            for i, rob in enumerate(env.robots):
                a = rob.goto(rob.target_pos)
                vel = agent(torch.Tensor(a).to(device)).cpu().detach().numpy()
                action = [0, 0]
                action[0] = vel[0] if vel[0] < 1 else 1
                action[1] = vel[1]
                actions[i] = action
                print('action:', action, 'obs', next_obs[i], 'ref', a)

            next_obs, reward, te, tr, info_ = env.step(actions)
            r.append(reward)
            done = env.check_done(te=te, tr=tr, lim=5)
