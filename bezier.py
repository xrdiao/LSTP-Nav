import math
import numpy as np
import matplotlib.pyplot as plt


class Bezier:
    def __init__(self, control_points) -> None:
        self.control_points_ = control_points
        self.d_ = len(control_points) - 1
        self.path_ = None

    def cal_combination(self, t, k):
        return (math.factorial(self.d_) / (math.factorial(k) * math.factorial(self.d_ - k))) * t ** k * (1 - t) ** (
                self.d_ - k)

    def get_bezier_point(self, t):
        assert 0 <= t <= 1
        point = [np.multiply(self.control_points_[k], self.cal_combination(t, k)) for k in
                 range(len(self.control_points_))]
        point = np.sum(point, axis=0)
        return point

    def cal_bezier_path(self):
        path = []
        for t in np.linspace(0, 1, 100):
            path.append(self.get_bezier_point(t))

        self.path_ = np.transpose(path)
        return self.path_


class PieceBezierCurve:
    def __init__(self, time_allocations: list):
        self.time_allocations_ = time_allocations
        self.time_segment_ = np.cumsum(time_allocations)
        self.segment_num_ = len(self.time_allocations_)


if __name__ == '__main__':
    c = [[0, 0], [0, 1], [1, 2],[1,5],[0,7]]
    b = Bezier(c)
    b.cal_bezier_path()
    plt.plot(b.path_[0], b.path_[1])
    plt.show()
