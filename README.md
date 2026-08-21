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

## Configuration Notes

This repository mixes script-based experiments with code-driven configuration. Not every parameter is exposed as a CLI argument.

The main places to edit experiment settings are:

- [`train.py`](./train.py)
- [`test.py`](./test.py)
- [`compare_methods/drlvo/config.py`](./compare_methods/drlvo/config.py)
- [`compare_methods/vuca_nav/config.py`](./compare_methods/vuca_nav/config.py)
- [`plot/`](./plot) scripts for figure-specific settings

## Quick Validation

After installing the environment, the following commands are good smoke tests:

```bash
python -m scripts.calc_flops
python -m scripts.benchmark_policy_frequency
python plot/plot_test_result_std.py
python -m plot.reward_trend.plot_reward
```

## Chinese Version

See [`README_zh.md`](./README_zh.md).
