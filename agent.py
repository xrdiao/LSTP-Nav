from vector import Vector1
from base_math import *
from bezier import Bezier, PieceBezierCurve


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
        # self.new_velocity = Vector1()
        self.trajectory_ = None
        self.glob_vel_ = None

    def find_neighbors(self):
        """find the neighbors of the current agent"""
        self.agent_neighbors_.clear()
        for i in range(len(self.simulator_.agents_)):
            if i != self.id_ and dist(self.simulator_.agents_[self.id_].position_,
                                      self.simulator_.agents_[i].position_) < self.neighbor_dist_:
                self.agent_neighbors_.append(self.simulator_.agents_[i])

    def cal_trajectory(self) -> None:
        """calculate the trajectory of the agent"""
        curves = PieceBezierCurve([1, 1])
        curves.set_all_control_points([[[0, 0], [0, 1], [1, 1.5], [2, 2]], [[2, 2], [3, 2], [4, 1], [5, 0]]])
        self.trajectory_ = curves
        self.glob_vel_ = curves.derivation()


    def get_new_velocity(self):
        """core code in such motion planning, for the other kind of the motion planning, the method's name should be
        'get_new_trajectory'. """
        new_velocity = self.glob_vel_.get_point(self.simulator_.global_time_)
        return Vector1(new_velocity[0], new_velocity[1])

    def update(self):
        self.velocity_ = self.get_new_velocity()
        self.position_ = self.position_ + self.velocity_ * self.simulator_.time_step_
