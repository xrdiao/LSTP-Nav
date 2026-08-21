from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
COMPARE_METHODS_DIR = ROOT_DIR / "compare_methods"

ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DATA_DIR = ARTIFACTS_DIR / "data"
MODEL_DIR = ARTIFACTS_DIR / "model"
RUNS_DIR = ARTIFACTS_DIR / "runs"

OUTPUTS_DIR = ROOT_DIR / "outputs"
FIG_DIR = OUTPUTS_DIR / "fig"
FIG_TRAJECTORY_DIR = FIG_DIR / "trajectory"
FIG_TRAJECTORY_GIF_DIR = FIG_DIR / "trajectory_gif"
RECORD_TRAJECTORY_DIR = OUTPUTS_DIR / "record_trajectory"
PATH_DIR = OUTPUTS_DIR / "path"
TMP_DIR = OUTPUTS_DIR / "tmp"
LASER_BUFFER_PATH = TMP_DIR / "laser_buffer.npy"

ASSETS_DIR = ROOT_DIR / "assets"
OBSTACLE_DIR = ASSETS_DIR / "obstacle"
REAL_DIR = ASSETS_DIR / "real"
ENV_SIM_DATA_DIR = ROOT_DIR / "env_sim" / "utils" / "data"
TURTLEBOT_URDF_PATH = ENV_SIM_DATA_DIR / "turtlebot.urdf"
NEUPAN_DIR = COMPARE_METHODS_DIR / "NeuPAN"
NEUPAN_CONVEX_DIFF_PLANNER_PATH = NEUPAN_DIR / "example" / "convex_obs" / "diff" / "planner.yaml"

DRLVO_DIR = COMPARE_METHODS_DIR / "drlvo"
DRLVO_CHECKPOINT_DIR = DRLVO_DIR / "checkpoints"
DRLVO_RUNS_DIR = DRLVO_DIR / "runs"
VUCA_NAV_DIR = COMPARE_METHODS_DIR / "vuca_nav"
VUCA_CHECKPOINT_DIR = VUCA_NAV_DIR / "checkpoints"
VUCA_RUNS_DIR = VUCA_NAV_DIR / "runs"
RVO_DIR = COMPARE_METHODS_DIR / "rvo"
MPC_DIR = COMPARE_METHODS_DIR / "MPC"
