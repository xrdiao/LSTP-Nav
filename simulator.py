import matplotlib.pyplot as plt
import numpy as np

from agent import Agent
from obstacle import Obstacle
from vector import Vector1


class Simulator:
    def __init__(self, gui=False):
        self.agents_ = []
        self.obstacles_ = []
        self.global_time_ = 0.0
        self.time_step_ = 0.1
        self.default_agent_ = None
        self.GUI = gui

    def set_default_agent(self, neighborDist=1, maxNeighbors=10, timeHorizon=0.1, radius=0.1, maxSpeed=1,
                          velocity=Vector1(0.5, 0)):
        if self.default_agent_ is None:
            self.default_agent_ = Agent(self)

        self.default_agent_.maxSpeed_ = maxSpeed
        self.default_agent_.radius_ = radius
        self.default_agent_.velocity_ = velocity
        self.default_agent_.time_horizon_ = timeHorizon
        self.default_agent_.neighbor_dist_ = neighborDist
        self.default_agent_.max_neighbors_ = maxNeighbors

    def add_agent(self, position):
        assert self.default_agent_ is not None, 'you should set default agent first.'

        agent_ = Agent(self)
        agent_.maxSpeed_ = self.default_agent_.maxSpeed_
        agent_.radius_ = self.default_agent_.radius_
        agent_.velocity_ = self.default_agent_.velocity_
        agent_.time_horizon_ = self.default_agent_.time_horizon_
        agent_.id = len(self.agents_)
        agent_.position_ = position
        self.agents_.append(agent_)

        return agent_.id

    def add_obstacle(self, position, radius=0.5):
        """先弄个简单的，那个代码的逻辑真的妙哉"""
        obs = Obstacle()
        obs.position_ = position
        obs.radius_ = radius
        obs.id_ = len(self.obstacles_)

        self.obstacles_.append(obs)
        return obs.id_

    def step(self):
        if self.GUI:
            plt.ion()
            plt.cla()
            pos = np.array([self.agents_[i].position_[0] for i in range(len(self.agents_))])
            scatter = plt.scatter([pos[:, 0]], pos[:, 1], c=np.arange(len(self.agents_)))
            plt.legend(*(scatter.legend_elements()), loc='upper left', fontsize=8)
            plt.xlim([-2, 2])
            plt.ylim([-2, 2])
            plt.pause(0.1)
            plt.ioff()

        for agentNo in self.agents_:
            agentNo.get_newVelocity()
            agentNo.update()

        self.global_time_ += self.time_step_
        return self.global_time_

    # 这是用来保护私有变量的，相当于c里面的private
    @property
    def global_time(self):
        return self.global_time_

    @property
    def num_obstacles(self):
        return len(self.obstacles_)

    @property
    def num_agents(self):
        return len(self.agents_)
