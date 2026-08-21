# Direction-Based Obstacle Avoidance 中文说明

这个仓库包含 **LSTP-Nav** 与 **PhysReplay-SimLab** 的完整代码实现。

LSTP-Nav 是一个轻量级、完全去中心化的**无地图多机器人导航**方法。它只依赖局部 LiDAR 观测、目标信息和速度反馈，并结合 **基于 GRU 的时序建模**、**基于 attention 的特征聚合** 以及提出的 **Heading Stability（HS）reward**，在保持较低计算开销的同时处理短时交互与局部避障问题。

PhysReplay-SimLab 是本项目使用的 PyBullet 训练环境。它通过回放接近失败的安全关键交互，而不是每次都重置整个环境，提升了训练中困难样本的利用率，从而改善策略学习效率。

在总结实验中，LSTP-Nav 在**单机器人场景**下取得了 **98.6%–100.0%** 的成功率，在**10 机器人场景**下取得了 **97.8%–99.0%** 的成功率，并且在 LiDAR 退化、非凸环境以及大规模多机器人交互下保持了较好的鲁棒性。同一策略还能在**不做真实世界微调**的前提下直接迁移到 TurtleBot3 平台，并在 **Raspberry Pi 3 Model B** 上达到 **40 Hz 以上**的运行频率。

仓库中还收纳了多个对比方法，统一放在 [`compare_methods/`](./compare_methods) 下，并配套了评测、绘图和输出管理流程。

## 目录结构

```text
.
├── train.py                         # 主 PPO 训练入口
├── test.py                          # 主 PPO 测试入口
├── env_sim/                         # 仿真环境与机器人实现
├── rl/                              # PPO 模型与训练工具
├── evaluation/                      # 评测与轨迹记录
├── plot/                            # 绘图脚本
├── scripts/                         # 基线运行脚本与工具脚本
├── compare_methods/                 # 对比方法
│   ├── NeuPAN/
│   ├── MPC/
│   ├── drlvo/
│   ├── rvo/
│   └── vuca_nav/
├── artifacts/                       # 数据、模型、训练日志
├── outputs/                         # 图、轨迹、临时输出
└── assets/                          # 障碍物与实机相关资源
```

## 环境安装

当前仓库建议使用 Conda 环境 `lstp`。

### 方式一：直接从环境文件创建

```bash
conda env create -f environment.yml
conda activate lstp
```

### 方式二：在已有环境里手动安装

```bash
conda create -n lstp python=3.10
conda activate lstp

conda install -y -c pytorch -c nvidia -c conda-forge \
    pytorch=2.4.1 torchvision=0.19.1 torchaudio=2.4.1 pytorch-cuda=12.4 \
    numpy scipy matplotlib pandas tqdm pyyaml scikit-learn rich \
    gymnasium pybullet opencv tensorboard pillow ffmpeg

pip install tensorboardX thop cvxpy cvxpylayers ecos gctl==1.2 ir-sim>=2.4.0
```

说明：

- [`scripts/real_world.py`](./scripts/real_world.py) 依赖 ROS，不包含在 `environment.yml` 中。
- `NeuPAN` 在当前仓库里通过内部路径调用，运行 [`scripts/neupan_bridge.py`](./scripts/neupan_bridge.py) 时不需要额外 `pip install -e`。

## 主流程

### 1. 训练主 PPO 策略

```bash
python train.py
```

当前主训练入口是 [`train.py`](./train.py)。默认配置大致为：

- 环境：`circle`
- 策略：`AttentionAgent`
- 渲染：关闭

如果你想修改机器人数量、障碍物数量、半径或地图范围，直接改 [`train.py`](./train.py) 里的 `create_env()`。

### 2. 测试主 PPO 策略

```bash
python test.py
```

测试入口是 [`test.py`](./test.py)。常见需要修改的项包括：

- `agent_id`
- `stack_laser`
- `robot_nums`
- `obstacles_num`
- `Evaluator.evaluate(...)` 里的 `times`

评测结果会写到 [`artifacts/data/`](./artifacts/data)。

## 对比方法

### ORCA

```bash
python -m scripts.orca_bridge
```

### NeuPAN

```bash
python -m scripts.neupan_bridge
```

### DRL-VO

```bash
python -m compare_methods.drlvo.train
python -m compare_methods.drlvo.evaluate
```

### VUCA-Nav

```bash
python -m compare_methods.vuca_nav.train
python -m compare_methods.vuca_nav.evaluate
```

### MPC

MPC 相关代码目前在：

- [`compare_methods/MPC/cbfnav.py`](./compare_methods/MPC/cbfnav.py)
- [`compare_methods/MPC/sicnav.py`](./compare_methods/MPC/sicnav.py)

它们目前是库代码，不是独立顶层入口脚本。

## 绘图与分析

### 测试结果对比图

```bash
python plot/plot_test_result_std.py
```

### Reward 曲线

```bash
python -m plot.reward_trend.plot_reward
```

### 成功率曲线

```bash
python plot/plot_lstm_attn_success_rate_0_100.py
```

### 轨迹可视化

```bash
python plot/plot_trajectory.py --help
python plot/plot_trajectory_gif.py --help
```

### 工具脚本

```bash
python -m scripts.calc_flops
python -m scripts.benchmark_policy_frequency
python -m scripts.visualize_laser_buffer --help
```

## 训练参数修改位置

对于主 LSTP-Nav PPO 训练链路，训练参数主要分布在下面几个位置。

### 1. 任务设置与环境难度

优先修改 [`train.py`](./train.py) 里的 `create_env()`。

这里最常改的是：

- `name`：环境类型，例如 `circle`、`u`、`dumbbell`、`room`
- `robot_num`：机器人数量
- `obstacle_num`：随机障碍物数量
- `radius`：初始部署半径
- `x_lim`、`y_lim`：障碍物采样范围
- `x_range`、`y_range`：机器人队形展开范围 / 地图尺度
- `render`：是否开启 PyBullet 渲染
- `robot_camera`：是否启用机器人相机

当前训练入口实际调用的是：

```python
env, env_arg = create_env(render=False)
agent = PPO(env, policy="AttentionAgent")
```

### 2. 环境参数默认值

修改 [`env_sim/env_util.py`](./env_sim/env_util.py) 里的 `env_args(...)`。

这个解析器定义了环境相关默认参数，包括：

- `--robots-num`
- `--random-obstacles`
- `--x-lim`、`--y-lim`
- `--x-range`、`--y-range`
- `--radius`
- `--control-rate`
- `--name`
- `--ori-reward`
- `--random-angle-obs`
- `--robot-camera`

如果你想改默认解析值，而不是直接在 `train.py` 写死，改这里更合适。

### 3. PPO 超参数

修改 [`rl/util_raw.py`](./rl/util_raw.py) 里的 `parse_args(...)`。

主 PPO 的超参数在这里定义，常改项包括：

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

这些参数会在 [`rl/model_raw.py`](./rl/model_raw.py) 的 `PPO.__init__(...)` 中被读取。

### 4. 策略网络结构

修改 [`rl/util_raw.py`](./rl/util_raw.py)。

主策略类都定义在这里：

- `AttentionAgent`
- `LstmAgent`
- `IJRRAgent`
- `LinearAgent`

常见会改的内容有：

- GRU 隐层维度
- GRU 层数
- attention head 数量
- actor / critic 隐层维度
- 与 `LASER_NUM` 绑定的 LiDAR 输入维度

如果你修改 LiDAR 维度，至少同步检查：

- [`env_sim/argument.py`](./env_sim/argument.py)
- [`rl/util_raw.py`](./rl/util_raw.py)

### 5. 输出、权重与日志位置

如果你想改训练输出位置，修改 [`project_paths.py`](./project_paths.py)。

主训练链路当前默认写到：

- 权重：[`artifacts/model/`](./artifacts/model)
- TensorBoard 日志：[`artifacts/runs/`](./artifacts/runs)
- 评测 json：[`artifacts/data/`](./artifacts/data)

### 6. 对比方法参数

对比方法不要改 `train.py`，而是改它们各自的配置文件：

- DRL-VO：[`compare_methods/drlvo/config.py`](./compare_methods/drlvo/config.py)
- VUCA-Nav：[`compare_methods/vuca_nav/config.py`](./compare_methods/vuca_nav/config.py)

这些文件通常包含：

- 机器人 / 障碍物数量
- LiDAR 维度
- 策略网络维度
- reward 系数
- 训练步长
- checkpoint 与日志目录

## 输出位置

当前主要输出目录如下：

- 评测 json：[`artifacts/data/`](./artifacts/data)
- 主 PPO 权重：[`artifacts/model/`](./artifacts/model)
- 主 PPO TensorBoard 日志：[`artifacts/runs/`](./artifacts/runs)
- 图片：[`outputs/fig/`](./outputs/fig)
- 轨迹记录：[`outputs/record_trajectory/`](./outputs/record_trajectory)
- path 快照：[`outputs/path/`](./outputs/path)
- 临时文件（如 `laser_buffer.npy`）：[`outputs/tmp/`](./outputs/tmp)

静态资源目录：

- 障碍物资源：[`assets/obstacle/`](./assets/obstacle)
- 实机资源：[`assets/real/`](./assets/real)

英文版见 [`README.md`](./README.md)。

## 引用

如果您的工作中使用了这个仓库，请引用：

```text
X. Diao, Z. Sun, J. Peng, B. -K. Zhu, B. Jia and J. Wang,
"LSTP-Nav: Lightweight Spatiotemporal Policy for Map-free Multi-agent Navigation with LiDAR,"
in IEEE Transactions on Automation Science and Engineering,
doi: 10.1109/TASE.2026.3725345.

keywords: {Deep reinforcement learning;distributed system;multi-agent;collision avoidance;map-free navigation},
```
