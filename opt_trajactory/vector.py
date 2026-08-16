import math

import base_math


class Vector1:
    # 2-D dimensional vector
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y

    def __matmul__(self, other):
        assert isinstance(other, Vector1), 'the class of input should be Vector1'
        x = self.x * other.x
        y = self.y * other.y
        return Vector1(x, y)

    def __mul__(self, other):
        assert isinstance(other, float), 'the class of input should be float'
        return Vector1(self.x * other, self.y * other)

    def __add__(self, other):
        return Vector1(self.x + other.x, self.y + other.y)

    def __radd__(self, other):
        return Vector1(self.x + other.x, self.y + other.y)

    def __neg__(self):
        return Vector1(-self.x, -self.y)

    def __sub__(self, other):
        return Vector1(self.x - other.x, self.y - other.y)

    def __rsub__(self, other):
        return Vector1(other.x - self.x, other.y - self.y)

    def __abs__(self):
        return math.sqrt(base_math.abs_sq(self))

    def __getitem__(self, item):
        return [self.x, self.y]
