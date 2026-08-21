from pathlib import Path
import torch
from env_sim.env_util import *
from train import create_env

from evaluation.evaluator import Evaluator

agent_id = -1

stack_laser = True
if agent_id == 1:
    stack_laser = False
if stack_laser:
    from rl.model_raw import *
else:
    from rl.model import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
env_name = ['circle', 'u', 'dumbbell', 'room']

def main(robot_num=1, obs_num=15):
    env, _ = create_env(render=False, name=env_name[0], robot_num=robot_num, obstacle_num=obs_num, radius=17) # radius=17, x_range=33, y_range=33
    env.set_max_step(20000)

    agent = [AttentionAgent, 
            #  Conv1dAgent, 
             LinearAgent, 
             LstmAgent, 
             IJRRAgent
             ]
    eva = Evaluator(env, agent[agent_id], is_PPO=True, is_pre=False, stack_laser=stack_laser)
    data = eva.evaluate(debug=False, times=500, plt_render=False)
    print(eva.agent.name)
    print(data)
    env.close()

if __name__ == '__main__':
    obstacles_num = [35]
    robot_nums = [10]
    for robot_num_ in robot_nums:
        for obs_num_ in obstacles_num:
            main(robot_num=robot_num_,obs_num=obs_num_)
    # test()
