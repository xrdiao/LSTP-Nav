from simulator import Simulator
from vector import Vector1
from bezier import Bezier
import time

if __name__ == '__main__':
    s = Simulator(True)
    s.set_default_agent()
    s.add_agent(Vector1(0, 0))
    s.agents_[0].cal_trajectory()

    # s.set_default_agent(velocity=Vector1(0,0.5))
    # s.add_agent(Vector1(0, -1))
    # s.add_agent(Vector1(0, 0))
    while True:
        s.step()
        if s.global_time >= 500:
            break



