from collections import deque
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class Recorder:
    def __init__(self, max_len=100):
        self.path = deque(maxlen=max_len)
        self.obstacles = deque(maxlen=max_len)
        self.init_point = deque(maxlen=max_len)
        self.goal_point = deque(maxlen=max_len)

    def add_env_info(self, obstacles, init_point, goal_point):
        self.obstacles.append(obstacles)
        self.init_point.append(init_point)
        self.goal_point.append(goal_point)

    def add_path(self, path):
        self.path.append(path)

    def save(self, filename):
        path = list(self.path)
        obstacles = list(self.obstacles)
        init_point = list(self.init_point)
        goal_point = list(self.goal_point)
        dict_data = {'path': path, 'obstacles': obstacles, 'init_point': init_point, 'goal_point': goal_point}

        with open(filename, 'wb') as f:
            pickle.dump(dict_data, f)
            f.close()

    def clear(self):
        self.obstacles.clear()
        self.init_point.clear()
        self.goal_point.clear()
        self.path.clear()

    @staticmethod
    def load(filename):
        file = open(filename, 'rb')
        data_dict = pickle.load(file)
        file.close()
        return data_dict

    def plot_path(self, filename, idx, x_lim=7, y_lim=7):
        data_dict = self.load(filename)
        path, obstacles, init_point, goal_point = data_dict['path'], data_dict['obstacles'], data_dict['init_point'], \
            data_dict['goal_point']

        fig = plt.figure(num=1, figsize=(x_lim, y_lim))
        axes = fig.add_subplot(1, 1, 1)
        axes.set_xlim(-x_lim, x_lim)
        axes.set_ylim(-y_lim, y_lim)

        for obstacle in obstacles[0]:
            x, y, yaw = obstacle[0], obstacle[1], obstacle[2]
            alpha = np.pi * 3 / 4 - yaw
            x = x + 0.5 * np.cos(alpha)
            y = y - 0.5 * np.sin(alpha)
            square = plt.Rectangle(xy=(x, y), width=1, height=1, angle=obstacle[2] / np.pi * 180)
            axes.add_patch(square)

        x = []
        y = []
        for point in path:
            x.append(np.array(point)[:, 0])
            y.append(np.array(point)[:, 1])
        axes.plot(x, y, color='r')

        plt.savefig('./download/path/' + agent_name[0] + '/{}.png'.format(idx))


if __name__ == '__main__':
    agent_name = ['AttentionAgent', 'Agent_Lstm', 'Agent_Conv', 'Agent_Linear', 'Agent_Lag']
    recorder = Recorder()

    for i in tqdm(range(250)):
        fn = './download/path/' + agent_name[0] + '/{}.pkl'.format(i)
        recorder.plot_path(fn, i, 8, 8)
