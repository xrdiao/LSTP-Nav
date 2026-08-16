class AgentNode:
    def __init__(self):
        self.begin_ = None
        self.end_ = None
        self.left_ = None
        self.right_ = None
        self.agent_ = None


class ObstacleNode:
    def __init__(self):
        self.begin_ = None
        self.end_ = None
        self.left_ = None
        self.right_ = None
        self.obstacle_ = None


class Kdtree:
    def __init__(self, simulator):
        self.obstacleTree_ = None
        self.agentTree_ = None
        self.simulator = simulator

    