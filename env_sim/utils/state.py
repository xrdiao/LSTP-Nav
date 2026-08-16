import numpy as np


class State:
    def __init__(self, pos, vel, laser):
        self.position = pos
        self.velocity = vel
        self.laser = laser
