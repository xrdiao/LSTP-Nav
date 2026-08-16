import argparse
from distutils.util import strtobool


def env_args(argv=None):
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", type=lambda x: bool(strtobool(x)), default=True,
                        help="if true, render")
    parser.add_argument("--random-robot", type=int, default=0, nargs="?", const=True,
                        help="if random place robot")
    parser.add_argument("--robots-num", type=int, default=1, nargs="?", const=True,
                        help="robots number")
    parser.add_argument("--boundary", type=int, default=0, nargs="?", const=True,
                        help="size of the boundary, 0 means no boundary")
    parser.add_argument("--random-obstacles", type=int, default=1, nargs="?", const=True,
                        help="the random obstacles number, 0 means no obstacles")
    parser.add_argument("--x-lim", type=float, default=0.0, nargs="?", const=True,
                        help="the limitation of x of the random position of the obstacle")
    parser.add_argument("--y-lim", type=float, default=0.0, nargs="?", const=True,
                        help="the limitation of y of the random position of the obstacle")
    parser.add_argument("--x-range", type=float, default=3.0, nargs="?", const=True,
                        help="the length of the robots team")
    parser.add_argument("--y-range", type=float, default=3.0, nargs="?", const=True,
                        help="the width of the robots team")
    parser.add_argument("--radius", type=float, default=3.0, nargs="?", const=True,
                        help="radius of the circle of the robots team")
    parser.add_argument("--safe", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="safe env")
    parser.add_argument("--control-rate", type=int, default=40, nargs="?", const=True,
                        help="the control rate of the robot")
    parser.add_argument("--name", type=str, default='base', nargs="?", const=True,
                        help="the name of the environment")
    parser.add_argument("--ori-reward", type=lambda x: bool(strtobool(x)), default='True', nargs="?", const=True,
                        help="type of reward function")
    parser.add_argument("--test-mode", type=lambda x: bool(strtobool(x)), default='False', nargs="?", const=True,
                        help="the mode of the environment")
    parser.add_argument("--random-angle-obs", type=lambda x: bool(strtobool(x)), default='True', nargs="?", const=True,
                        help="type of reward function")
    parser.add_argument('--robot-camera', type=lambda x: bool(strtobool(x)), default='False', nargs="?", const=True,
                        help="if open robot camera")
    args = parser.parse_args(argv)
    return args
