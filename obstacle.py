from vector import Vector1
class Obstacle:
    """
    Defines static obstacles in the simulation.
    """

    def __init__(self):
        self.next_ = None
        self.previous_ = None
        self.direction_ = None
        self.point_ = None
        self.id_ = 0
        self.convex_ = False

        self.position_ = Vector1()
        self.radius_ = 0.0
