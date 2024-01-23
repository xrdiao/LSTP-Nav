from vector import Vector1
from base_math import *


class Agent:
    def __init__(self, simulator):
        self.position_ = Vector1()
        self.velocity_ = Vector1()
        # self.pref_velocity = Vector1()
        self.id_ = 0
        self.radius_ = 0.0
        self.time_horizon_ = 0.0
        # self.time_horizon_obs_ = 0.0
        self.agent_neighbors_ = []
        self.obstacle_neighbors_ = []
        self.simulator_ = simulator
        self.maxSpeed_ = 0.0
        self.neighbor_dist_ = 0.0
        self.max_neighbors_ = 0
        self.new_velocity = Vector1()

    def find_neighbors(self):
        """find the neighbors of the current agent"""
        self.agent_neighbors_.clear()
        for i in range(len(self.simulator_.agents_)):
            if i != self.id_ and dist(self.simulator_.agents_[self.id_].position_,
                                      self.simulator_.agents_[i].position_) < self.neighbor_dist_:
                self.agent_neighbors_.append(self.simulator_.agents_[i])

    def get_newVelocity(self):
        """core code in such motion planning, for the other kind of the motion planning, the method's name should be
        'get_new_trajectory'. """
        self.find_neighbors()

        self.new_velocity = self.velocity_

    def update(self):
        self.velocity_ = self.new_velocity
        self.position_ = self.position_ + self.velocity_ * self.simulator_.time_step_
