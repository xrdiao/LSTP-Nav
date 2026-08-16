#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将多个seed的TensorBoard标量曲线聚合(均值±标准差)并画图；
同时对比两个run目录(例如 2026-03-31-13 vs 2026-03-31-18)；
平滑方式使用与TensorBoard前端一致的“指数滑动平均 EMA”。
"""

import os
import re
import glob
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ---------------------------
# 配置区：改这里就行
# ---------------------------
BASE = "/home/oem/direction_based_obstacle_avoidance/runs"

RUN_A = os.path.join(BASE, "ori_coef_0.05/Agent_Lstm_Attn")
RUN_B = os.path.join(BASE, "ori_coef_0.08/Agent_Lstm_Attn")
RUN_C = os.path.join(BASE, "ori_coef_0.01/Agent_Lstm_Attn")
RUN_D = os.path.join(BASE, "ori_coef_0.15/Agent_Lstm_Attn")


# 你要画的标量tag（去tensorboard里复制 Scalars 的 tag 名称）
SCALAR_TAG = "rewards/train_rewards"   # <- 改成你实际的tag

# 平滑强度：与TensorBoard smoothing slider一致，范围[0, 0.999...]，0表示不平滑
SMOOTHING = 0.99

# x轴用 step 还是 wall_time：通常用 step
XAXIS = "step"  # "step" or "wall_time"

# 最小公共步长：采用“共同拥有的step点”还是“插值到统一网格”
# common_steps: 只在所有seed都存在的step上做统计（最稳）
# union_interp: 对所有seed的step取并集，再对每条曲线线性插值
AGG_MODE = "union_interp"  # "common_steps" or "union_interp"

# 输出图保存路径（可选）
SAVE_FIG = None  # e.g. "compare_runs.png"
# ---------------------------

def tb_ema_smooth(y: np.ndarray, smoothing: float) -> np.ndarray:
    """
    TensorBoard前端使用的平滑本质上是指数滑动平均(EMA)：
        s_t = s_{t-1} * smoothing + (1 - smoothing) * x_t
    smoothing=0 => 原始曲线
    """
    if y.size == 0:
        return y
    if smoothing <= 0:
        return y.copy()
    s = np.empty_like(y, dtype=np.float64)
    s[0] = y[0]
    for i in range(1, len(y)):
        s[i] = s[i - 1] * smoothing + (1.0 - smoothing) * y[i]
    return s

def find_seed_dirs(run_dir: str) -> List[str]:
    """匹配 seed_1, seed-2, seed9 等目录（尽量兼容命名差异）"""
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(run_dir)
    subs = [os.path.join(run_dir, d) for d in os.listdir(run_dir)]
    subs = [d for d in subs if os.path.isdir(d)]
    pat = re.compile(r"seed[_-]?\d+$")
    seed_dirs = sorted([d for d in subs if pat.search(os.path.basename(d))])
    if not seed_dirs:
        raise RuntimeError(f"No seed dirs found under: {run_dir}")
    return seed_dirs

def load_scalar_from_seed(seed_dir: str, tag: str, xaxis: str = "step") -> Tuple[np.ndarray, np.ndarray]:
    """
    从某个seed目录中读取某个scalar tag的 (x, y)
    自动递归找到 event 文件。
    """
    # EventAccumulator 支持直接给目录，但有时目录层级复杂，递归找event文件更稳
    event_files = glob.glob(os.path.join(seed_dir, "**", "events.out.tfevents.*"), recursive=True)
    if not event_files:
        raise RuntimeError(f"No event files found in {seed_dir}")

    # 让EventAccumulator吃“包含event文件的目录”
    # 如果有多级，取event文件的共同上层不太好做；这里直接用seed_dir并增大size_guidance
    ea = EventAccumulator(
        seed_dir,
        size_guidance={
            "scalars": 0,  # 0表示尽可能全读
        },
    )
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    if tag not in tags:
        raise KeyError(f"Tag '{tag}' not found in {seed_dir}. Available: {tags[:50]} ...")

    events = ea.Scalars(tag)
    if xaxis == "step":
        x = np.array([e.step for e in events], dtype=np.int64)
    elif xaxis == "wall_time":
        x = np.array([e.wall_time for e in events], dtype=np.float64)
    else:
        raise ValueError("xaxis must be 'step' or 'wall_time'")
    y = np.array([e.value for e in events], dtype=np.float64)

    # 按x排序（以防万一）
    order = np.argsort(x)
    return x[order], y[order]

def aggregate_curves(
    curves: List[Tuple[np.ndarray, np.ndarray]],
    smoothing: float = 0.0,
    mode: str = "union_interp",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    输入：多个seed的(steps, values)
    输出：统一x轴上的 mean 和 std
    """
    # 先对每条曲线做TB同款平滑（按其自身顺序做EMA）
    smoothed = []
    for x, y in curves:
        y_s = tb_ema_smooth(y, smoothing)
        smoothed.append((x, y_s))

    if mode == "common_steps":
        # 取所有seed共同存在的x
        x_sets = [set(x.tolist()) for x, _ in smoothed]
        common = set.intersection(*x_sets)
        xs = np.array(sorted(common))
        if xs.size == 0:
            raise RuntimeError("No common steps across seeds. Try mode='union_interp'.")

        ys = []
        for x, y in smoothed:
            idx = {int(xx): i for i, xx in enumerate(x)}
            ys.append(np.array([y[idx[int(xx)]] for xx in xs], dtype=np.float64))
        Y = np.stack(ys, axis=0)
        return xs, Y.mean(axis=0), Y.std(axis=0)

    elif mode == "union_interp":
        # 取并集x，然后对每条曲线线性插值到并集上
        xs = np.unique(np.concatenate([x for x, _ in smoothed]))
        xs.sort()

        ys = []
        for x, y in smoothed:
            # np.interp要求x递增，且会对超出范围的点用边界值外推（我们避免外推：只在范围内统计）
            # 做法：对每个xs点，若在该seed的范围外则记为nan，最后用nanmean/nanstd
            y_interp = np.interp(xs, x, y)
            mask = (xs >= x[0]) & (xs <= x[-1])
            y_interp = np.where(mask, y_interp, np.nan)
            ys.append(y_interp)

        Y = np.stack(ys, axis=0)  # [nseed, nstep]
        mean = np.nanmean(Y, axis=0)
        std = np.nanstd(Y, axis=0)

        # 可选：把“所有seed都nan”的点去掉（通常只会发生在xs两端）
        valid = ~np.isnan(mean)
        return xs[valid], mean[valid], std[valid]

    else:
        raise ValueError("mode must be 'common_steps' or 'union_interp'")

@dataclass
class RunAgg:
    name: str
    x: np.ndarray
    mean: np.ndarray
    std: np.ndarray

def process_run(run_dir: str, run_name: str, tag: str) -> RunAgg:
    seed_dirs = find_seed_dirs(run_dir)
    curves = []
    for sd in seed_dirs:
        x, y = load_scalar_from_seed(sd, tag, xaxis=XAXIS)
        curves.append((x, y))

    x, mean, std = aggregate_curves(curves, smoothing=SMOOTHING, mode=AGG_MODE)
    return RunAgg(run_name, x, mean, std)

def plot_runs(runs: List[RunAgg], tag: str):
    plt.figure(figsize=(10, 5), dpi=130)

    for r in runs:
        plt.plot(r.x, r.mean, linewidth=2.0, label=f"{r.name}")
        plt.fill_between(r.x, r.mean - r.std, r.mean + r.std, alpha=0.2, linewidth=0)

    plt.xlabel("Step" if XAXIS == "step" else "Wall time")
    plt.ylabel(tag)
    plt.title(f"{tag}  | smoothing={SMOOTHING} (TensorBoard EMA) | agg={AGG_MODE}")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    if SAVE_FIG:
        plt.savefig(SAVE_FIG, bbox_inches="tight")
        print(f"Saved to: {SAVE_FIG}")
    plt.show()

def main():
    run_a = process_run(RUN_A, "ori_coef=-0.05", SCALAR_TAG)
    run_b = process_run(RUN_B, "ori_coef=-0.08", SCALAR_TAG)
    run_c = process_run(RUN_C, "ori_coef=-0.01", SCALAR_TAG)
    run_d = process_run(RUN_D, "ori_coef=0.15", SCALAR_TAG)

    # plot_runs([run_a], SCALAR_TAG)
    plot_runs([run_a, run_b, run_c, run_d], SCALAR_TAG)

if __name__ == "__main__":
    main()