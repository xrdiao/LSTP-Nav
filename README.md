# Direction-Based Obstacle Avoidance

This repository contains the codebase for **LSTP-Nav** and **PhysReplay-SimLab**.

LSTP-Nav is a lightweight and fully decentralized policy for **map-free multi-agent navigation**. It uses only local LiDAR observations, goal information, and robot velocity feedback, and combines **GRU-based temporal modeling**, **attention-based feature aggregation**, and the proposed **Heading Stability (HS) reward** to handle short-term interactions while remaining computationally efficient.

PhysReplay-SimLab is the PyBullet-based training environment used in this project. It improves training by replaying safety-critical near-failure interactions instead of always resetting the full environment, which increases the frequency of useful hard samples during policy learning.

In the reported experiments, LSTP-Nav reaches **98.6%–100.0% success rates in single-robot scenarios** and **97.8%–99.0% in 10-robot scenarios** under varying obstacle densities, while remaining robust under LiDAR degradation, nonconvex environments, and large-scale multi-robot interactions. The same policy can be transferred to real TurtleBot3 platforms without real-world fine-tuning and runs at **over 40 Hz on a Raspberry Pi 3 Model B**.

The repository also includes several comparison methods under [`compare_methods/`](./compare_methods), together with unified evaluation, plotting, and artifact management.

## Repository Layout

```text
.
├── train.py                         # Main PPO training entry
├── test.py                          # Main PPO evaluation entry
├── env_sim/                         # PyBullet simulator and robot/environment code
├── rl/                              # PPO models and training utilities
├── evaluation/                      # Evaluation and trajectory recording
├── plot/                            # Plotting scripts
├── scripts/                         # Baseline runners and utility scripts
├── compare_methods/
│   ├── NeuPAN/
│   ├── MPC/
│   ├── drlvo/
│   ├── rvo/
│   └── vuca_nav/
├── artifacts/
│   ├── data/                        # Evaluation json outputs
│   ├── model/                       # Main PPO checkpoints
│   └── runs/                        # Main PPO TensorBoard logs
├── outputs/
│   ├── fig/
│   ├── path/
│   ├── record_trajectory/
│   └── tmp/
└── assets/
    ├── obstacle/
    └── real/
```

## Installation

The repository is currently organized around a Conda environment named `lstp`.

### Option 1: Create the environment from file

```bash
conda env create -f environment.yml
conda activate lstp
```

### Option 2: Install manually into an existing environment

```bash
conda create -n lstp python=3.10
conda activate lstp

conda install -y -c pytorch -c nvidia -c conda-forge \
    pytorch=2.4.1 torchvision=0.19.1 torchaudio=2.4.1 pytorch-cuda=12.4 \
    numpy scipy matplotlib pandas tqdm pyyaml scikit-learn rich \
    gymnasium pybullet opencv tensorboard pillow ffmpeg

pip install tensorboardX thop cvxpy cvxpylayers ecos gctl==1.2 ir-sim>=2.4.0
```

Notes:

- `scripts/real_world.py` requires a ROS environment and is not covered by `environment.yml`.
- `compare_methods/NeuPAN` is used through repository-internal paths; no extra editable install is required for `scripts/neupan_bridge.py`.

## Main Workflow

### 1. Train the main PPO policy

```bash
python train.py
```

The main training entry is [`train.py`](./train.py). The default setup currently uses:

- environment: `circle`
- policy: `AttentionAgent`
- rendering: disabled

If you want to change the number of robots, obstacles, radius, or map size, edit [`create_env()` in `train.py`](./train.py).

### 2. Evaluate the main PPO policy

```bash
python test.py
```

The evaluation entry is [`test.py`](./test.py). Common things you may want to edit before running:

- `agent_id`
- `stack_laser`
- `robot_nums`
- `obstacles_num`
- `times` inside `Evaluator.evaluate(...)`

The evaluation results are written to [`artifacts/data/`](./artifacts/data).

## Comparison Methods

### ORCA baseline

```bash
python -m scripts.orca_bridge
```

### NeuPAN baseline

```bash
python -m scripts.neupan_bridge
```

### DRL-VO reproduction

```bash
python -m compare_methods.drlvo.train
python -m compare_methods.drlvo.evaluate
```

### VUCA-Nav reproduction

```bash
python -m compare_methods.vuca_nav.train
python -m compare_methods.vuca_nav.evaluate
```

### MPC utilities

The MPC-related code is located in:

- [`compare_methods/MPC/cbfnav.py`](./compare_methods/MPC/cbfnav.py)
- [`compare_methods/MPC/sicnav.py`](./compare_methods/MPC/sicnav.py)

These are library modules rather than top-level runnable entries in the current repository layout.

## Plotting and Analysis

### Compare test results

```bash
python plot/plot_test_result_std.py
```

### Plot reward trends

```bash
python -m plot.reward_trend.plot_reward
```

### Plot success-rate curves

```bash
python plot/plot_lstm_attn_success_rate_0_100.py
```

### Plot trajectories

```bash
python plot/plot_trajectory.py --help
python plot/plot_trajectory_gif.py --help
```

### Utility scripts

```bash
python -m scripts.calc_flops
python -m scripts.benchmark_policy_frequency
python -m scripts.visualize_laser_buffer --help
```

## Where to Modify Training Parameters

For the main LSTP-Nav PPO pipeline, the training-related parameters are mainly distributed across the following files.

### 1. Task setup and environment difficulty

Edit [`create_env()` in `train.py`](./train.py).

This is the first place to modify:

- `name`: environment type such as `circle`, `u`, `dumbbell`, `room`
- `robot_num`: number of robots
- `obstacle_num`: number of random obstacles
- `radius`: initial deployment radius
- `x_lim`, `y_lim`: obstacle sampling limits
- `x_range`, `y_range`: team spread / map scale
- `render`: whether to render PyBullet
- `robot_camera`: whether to enable robot camera

The current training entry uses:

```python
env, env_arg = create_env(render=False)
agent = PPO(env, policy="AttentionAgent")
```

### 2. Environment argument defaults

Edit [`env_sim/env_util.py`](./env_sim/env_util.py), inside `env_args(...)`.

This parser defines the default environment-side arguments, including:

- `--robots-num`
- `--random-obstacles`
- `--x-lim`, `--y-lim`
- `--x-range`, `--y-range`
- `--radius`
- `--control-rate`
- `--name`
- `--ori-reward`
- `--random-angle-obs`
- `--robot-camera`

Use this file when you want to change the default parser values instead of editing `train.py` directly.

### 3. PPO hyperparameters

Edit [`rl/util_raw.py`](./rl/util_raw.py), inside `parse_args(...)`.

The main PPO hyperparameters are defined there, including:

- `--learning-rate`
- `--total-timesteps`
- `--num-steps`
- `--num-minibatches`
- `--update-epochs`
- `--gamma`
- `--gae-lambda`
- `--clip-coef`
- `--ent-coef`
- `--vf-coef`
- `--max-grad-norm`
- `--target-kl`
- `--cuda`

These values are consumed by [`rl/model_raw.py`](./rl/model_raw.py) in `PPO.__init__(...)`.

### 4. Policy architecture

Edit [`rl/util_raw.py`](./rl/util_raw.py).

The main policy classes are defined there:

- `AttentionAgent`
- `LstmAgent`
- `IJRRAgent`
- `LinearAgent`

Typical architecture parameters to change are:

- GRU hidden size
- number of GRU layers
- attention head count
- actor / critic hidden dimensions
- LiDAR input dimension assumptions tied to `LASER_NUM`

If you change LiDAR dimensionality, check both:

- [`env_sim/argument.py`](./env_sim/argument.py)
- [`rl/util_raw.py`](./rl/util_raw.py)

### 5. Output, checkpoint, and log locations

Edit [`project_paths.py`](./project_paths.py) if you want to move training artifacts.

The main training pipeline currently writes to:

- checkpoints: [`artifacts/model/`](./artifacts/model)
- TensorBoard logs: [`artifacts/runs/`](./artifacts/runs)
- evaluation json outputs: [`artifacts/data/`](./artifacts/data)

### 6. Comparison method parameters

For comparison methods, use their own config files instead of `train.py`:

- DRL-VO: [`compare_methods/drlvo/config.py`](./compare_methods/drlvo/config.py)
- VUCA-Nav: [`compare_methods/vuca_nav/config.py`](./compare_methods/vuca_nav/config.py)

These files contain method-specific settings such as:

- robot / obstacle counts
- lidar size
- policy dimensions
- reward coefficients
- training horizon
- checkpoint and log directories

## Output Paths

Current output locations are:

- main evaluation json files: [`artifacts/data/`](./artifacts/data)
- main PPO checkpoints: [`artifacts/model/`](./artifacts/model)
- main PPO TensorBoard logs: [`artifacts/runs/`](./artifacts/runs)
- figures: [`outputs/fig/`](./outputs/fig)
- saved trajectories: [`outputs/record_trajectory/`](./outputs/record_trajectory)
- path snapshots: [`outputs/path/`](./outputs/path)
- temporary files such as `laser_buffer.npy`: [`outputs/tmp/`](./outputs/tmp)

Static assets are stored in:

- obstacle assets: [`assets/obstacle/`](./assets/obstacle)
- real-world assets: [`assets/real/`](./assets/real)

## Chinese Version

See [`README_zh.md`](./README_zh.md).
