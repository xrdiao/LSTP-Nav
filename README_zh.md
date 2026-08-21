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

## 配置说明

这个仓库目前还是“脚本 + 源码配置”混合形式，不是所有参数都有 CLI。

主要需要改的地方有：

- [`train.py`](./train.py)
- [`test.py`](./test.py)
- [`compare_methods/drlvo/config.py`](./compare_methods/drlvo/config.py)
- [`compare_methods/vuca_nav/config.py`](./compare_methods/vuca_nav/config.py)
- [`plot/`](./plot) 下各绘图脚本

## 安装后快速验证

建议先跑下面几个命令做冒烟测试：

```bash
python -m scripts.calc_flops
python -m scripts.benchmark_policy_frequency
python plot/plot_test_result_std.py
python -m plot.reward_trend.plot_reward
```

英文版见 [`README.md`](./README.md)。
