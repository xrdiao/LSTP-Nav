from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "model"
RUNS_DIR = ROOT_DIR / "runs"
FIG_DIR = ROOT_DIR / "fig"
FIG_TRAJECTORY_DIR = FIG_DIR / "trajectory"
FIG_TRAJECTORY_GIF_DIR = FIG_DIR / "trajectory_gif"
RECORD_TRAJECTORY_DIR = ROOT_DIR / "record_trajectory"
PATH_DIR = ROOT_DIR / "path"
OBSTACLE_DIR = ROOT_DIR / "obstacle"
ENV_SIM_DATA_DIR = ROOT_DIR / "env_sim" / "utils" / "data"
TURTLEBOT_URDF_PATH = ENV_SIM_DATA_DIR / "turtlebot.urdf"
LASER_BUFFER_PATH = ROOT_DIR / "laser_buffer.npy"
NEUPAN_DIR = ROOT_DIR / "NeuPAN"
NEUPAN_CONVEX_DIFF_PLANNER_PATH = NEUPAN_DIR / "example" / "convex_obs" / "diff" / "planner.yaml"

DRLVO_CHECKPOINT_DIR = ROOT_DIR / "drlvo" / "checkpoints"
DRLVO_RUNS_DIR = ROOT_DIR / "drlvo" / "runs"
VUCA_CHECKPOINT_DIR = ROOT_DIR / "vuca_nav" / "checkpoints"
VUCA_RUNS_DIR = ROOT_DIR / "vuca_nav" / "runs"
