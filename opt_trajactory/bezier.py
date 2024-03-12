import math
import numpy as np
import matplotlib.pyplot as plt
from vector import Vector1


class Bezier:
    def __init__(self) -> None:
        self.__control_points_ = None
        self.d_ = 0
        self.trajectory_ = None

    def set_control_points(self, control_points):
        if isinstance(control_points, Vector1):
            self.__control_points_ = [control_points.x, control_points.y]
        else:
            self.__control_points_ = control_points
        self.__control_points_ = control_points

        self.d_ = len(control_points) - 1

    def cal_combination(self, t, k):
        return (math.factorial(self.d_) / (math.factorial(k) * math.factorial(self.d_ - k))) * t ** k * (1 - t) ** (
                self.d_ - k)

    def get_bezier_point(self, t):
        """
        :param t: time step, [0,1]
        :return bezier point, [x,y,z,...]
        """
        assert 0 <= t <= 1
        point = [np.multiply(self.__control_points_[k], self.cal_combination(t, k)) for k in
                 range(len(self.__control_points_))]
        point = np.sum(point, axis=0)
        return point

    def cal_bezier_path(self):
        trajectory = []
        for t in np.linspace(0, 1, 100):
            trajectory.append(self.get_bezier_point(t))

        self.trajectory_ = np.transpose(trajectory)
        return self.trajectory_

    def derivation(self):
        """
        :return: derivation of the bezier, Bezier class
        """
        derivation_point = np.multiply(self.d_,
                                       (np.array(self.__control_points_[1:]) - np.array(self.__control_points_[:-1])))

        derivation_ = Bezier()
        derivation_.set_control_points(derivation_point)
        return derivation_

    def get_control_points(self):
        return self.__control_points_

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.trajectory_[0], self.trajectory_[1])


class PieceBezierCurve:
    def __init__(self, time_allocations: list):
        self.time_allocations_ = time_allocations
        self.time_segment_ = np.cumsum(time_allocations)
        self.segment_num_ = len(self.time_allocations_)

        self.curves = [Bezier() for _ in range(self.segment_num_)]

    def get_point(self, t):
        """
        :param t:
        :return: point in the piece Bézier curves
        """
        assert 0 <= t <= self.time_segment_[-1]

        for i in range(len(self.time_segment_)):
            if self.time_segment_[i] >= t:
                t_l = 0 if i == 0 else self.time_segment_[i - 1]
                t_seg = (t - t_l) / self.time_allocations_[i]
                return self.curves[i].get_bezier_point(t_seg)

    def set_curve_control_points(self, control_points, index) -> None:
        self.curves[index].set_control_points(control_points)

    def set_all_control_points(self, control_points) -> None:
        assert len(control_points) == self.segment_num_
        for i in range(self.segment_num_):
            self.curves[i].set_control_points(control_points[i])
            self.curves[i].cal_bezier_path()

    def derivation(self):
        der_control_points = [self.curves[i].derivation().get_control_points() for i in range(self.segment_num_)]
        velocity = PieceBezierCurve(self.time_allocations_)
        velocity.set_all_control_points(der_control_points)

        return velocity

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        for i in range(self.segment_num_):
            self.curves[i].plot(ax)


if __name__ == '__main__':
    curves = PieceBezierCurve([0.5, 1.2])
    curves.set_all_control_points([[[0, 0], [0, 1], [1, 1.5], [2, 2]], [[2, 2], [3, 2], [4, 1], [5, 0]]])
    curves.plot()
    plt.show()

    # c = [[0, 0], [0, 1], [1, 2], [1, 5]]
    # b = Bezier()
    # b.set_control_points(c)
    # b.cal_bezier_path()
    # b.derivation()
    # b.plot()
    # plt.show()
