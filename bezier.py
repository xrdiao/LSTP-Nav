import math
import numpy as np
import matplotlib.pyplot as plt
from vector import Vector1


class Bezier:
    def __init__(self, control_points) -> None:
        if isinstance(control_points, Vector1):
            self.control_points = [control_points.x, control_points.y]
        else:
            self.control_points_ = control_points
        self.d_ = len(control_points) - 1
        self.trajectory_ = None

    def cal_combination(self, t, k):
        return (math.factorial(self.d_) / (math.factorial(k) * math.factorial(self.d_ - k))) * t ** k * (1 - t) ** (
                self.d_ - k)

    def get_bezier_point(self, t):
        """
        :param t: time step, [0,1]
        :return bezier point, [x,y,z,...]
        """
        assert 0 <= t <= 1
        point = [np.multiply(self.control_points_[k], self.cal_combination(t, k)) for k in
                 range(len(self.control_points_))]
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
                                       (np.array(self.control_points_[1:]) - np.array(self.control_points_[:-1])))

        derivation_ = Bezier(derivation_point)
        return derivation_


class PieceBezierCurve:
    def __init__(self, time_allocations: list):
        self.time_allocations_ = time_allocations
        self.time_segment_ = np.cumsum(time_allocations)
        self.segment_num_ = len(self.time_allocations_)


if __name__ == '__main__':
    c = [[0, 0], [0, 1], [1, 2], [1, 5]]
    b = Bezier(c)
    b.cal_bezier_path()
    b.derivation()
    plt.plot(b.trajectory_[0], b.trajectory_[1])
    plt.show()
