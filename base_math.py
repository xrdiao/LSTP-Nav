"""如无特殊说明，vector一律指Vector1"""
import math
import numpy as np


def abs_sq(vector):
    return vector @ vector


def det(vector1, vector2):
    return vector1.x * vector2.y - vector1.y * vector2.x


def norm(vector):
    return vector / abs(vector)


def dist_sq_point_line_segment(vector1, vector2, vector3):
    """点到线段的距离"""
    r = (vector3 - vector1) @ (vector2 - vector1) / abs_sq(vector2 - vector1)

    if r < 0.0:
        return abs_sq(vector3 - vector1)

    if r > 1.0:
        return abs_sq(vector3 - vector2)

    return abs_sq(vector3 - (vector1 + r * (vector2 - vector1)))


def left_of(vector1, vector2, vector3):
    """True为左，False为右，这里有个细节，行列式是有向面积，通过右手法则可以很快判断在线段左边还是在右边"""
    return det(vector1 - vector3, vector2 - vector1) > 0


def dist(vector1, vector2):
    return np.linalg.norm((vector1 - vector2)[0])
