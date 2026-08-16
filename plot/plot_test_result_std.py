import os
import json
from collections import defaultdict
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

METRICS = ["reach_rate", "collision_rate", "trap_rate", "avg_time", "avg_step"]
RATE_METRICS = {"reach_rate", "collision_rate", "trap_rate"}
# ...existing code...
NAME_METRIC_MAP = {"reach_rate": "SR (%)",
                   "collision_rate": "CR (%)",
                   "trap_rate": "TR (%)",
                   "avg_time": "AT (s)",
                   "avg_step": "AS"}
NAME_AGENT_MAP = {
    "Agent_IJRR": "HybridRL-CA",
    "Agent_Lstm": "TP-Net",
    "Agent_Linear": "SP-Net",
    "Agent_Lstm_Attn": "LSTP-Nav",
    "orca": "NH-ORCA",
    "neupan": "NeuPan",
    "DRLVOAgent": "DRL-VO",
    "PaperCOAAgent": "COVA",
    "CBFNav": "RCBF-NMPC",
    "SICNav": "SICNav",
}
X_OBS = [5, 10, 15, 20, 25, 30, 35]
PLOT_ROBOTS = [1, 5, 10]

def load_all_json(data_dir="./data"):
    data = []
    for fn in sorted(os.listdir(data_dir)):
        if fn.endswith(".json"):
            with open(os.path.join(data_dir, fn), "r", encoding="utf-8") as f:
                data.append(json.load(f))
    return data

def build_table(data_list, metrics=METRICS):
    """
    table[robots_num][metric][agent_name][obstacles] = value
    """
    table = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    agents = set()
    robots_set = set()

    for d in data_list:
        agent = d.get("agent_name")
        robots = d.get("robots_num")
        obs = d.get("obstacles")
        if agent is None or robots is None or obs is None:
            continue

        agents.add(agent)
        robots_set.add(robots)

        for m in metrics:
            if m in d and d[m] is not None:
                table[robots][m][agent][obs] = d[m]

    return table, sorted(robots_set), sorted(agents)

def print_data_tables(table, robots_nums, agents, metrics, x_obs):
    """打印每个机器人数和指标对应的数据表格"""
    for robots in robots_nums:
        for metric in metrics:
            print(f"\n=== Robots: {robots} | Metric: {NAME_METRIC_MAP[metric]} ===")

            data_dict = {}
            for agent in agents:
                data_dict[NAME_AGENT_MAP.get(agent, agent)] = [
                    table[robots][metric].get(agent, {}).get(obs, "N/A")
                    for obs in x_obs
                ]

            if pd is not None:
                df = pd.DataFrame(data_dict, index=x_obs)
                df.index.name = "Obstacles"
                print(df.round(4))
                continue

            header = ["Obstacles"] + list(data_dict.keys())
            print("\t".join(header))
            for idx, obs in enumerate(x_obs):
                row = [str(obs)]
                for agent_name in data_dict:
                    value = data_dict[agent_name][idx]
                    if isinstance(value, (int, float)):
                        row.append(f"{value:.4f}")
                    else:
                        row.append(str(value))
                print("\t".join(row))

import matplotlib.pyplot as plt

def plot_all_agents_one_subplot(
    table, var_table, robots_nums, agents,
    metrics,
    x_obs,
    agent_colors=None,          # 手动颜色映射: {"Agent_IJRR": "C0", ...}
    marker_size=3,              # 节点大小（建议 2~4）
    line_width=1.5,
    marker="o",
    std_alpha=0.2,              # 标准差填充透明度
):
    rows = len(robots_nums)
    cols = len(metrics)

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 4 * rows),
                             squeeze=False, sharex=True)

    # 1) 手动颜色：优先使用 agent_colors；没提供就用tab10兜底
    if agent_colors is None:
        cmap = plt.get_cmap("tab10")
        color_map = {a: cmap(i % 10) for i, a in enumerate(agents)}
    else:
        color_map = dict(agent_colors)
        # 没写到的agent用灰色兜底，避免KeyError
        for a in agents:
            color_map.setdefault(a, "0.5")  # 灰色

    var_metrics = {m: "var_" + m for m in metrics}

    for r, robots in enumerate(robots_nums):
        for c, metric in enumerate(metrics):
            ax = axes[r][c]

            for agent in agents:
                y = [table[robots][metric].get(agent, {}).get(obs, None) for obs in x_obs]
                if all(v is None for v in y):
                    continue

                scale = 100.0 if metric in RATE_METRICS else 1.0
                y = [value * scale if value is not None else None for value in y]

                # 标准差：从 var_table 读取对应 var_metric，然后取 sqrt(var)
                var_key = var_metrics[metric]
                var_y = [var_table[robots][var_key].get(agent, {}).get(obs, None) for obs in x_obs]
                std = []
                for v in var_y:
                    if v is None:
                        std.append(None)
                    else:
                        # 如果方差为负（异常数据），则当做 0 处理
                        std_val = float(v) if v >= 0 else 0.0
                        std.append(float(np.sqrt(std_val)) * scale)

                # 绘制折线
                ax.plot(
                    x_obs, y,
                    color=color_map[agent],
                    marker=marker,
                    markersize=marker_size,   # 节点大小
                    linewidth=line_width,
                    label=agent
                )

                # 如果有标准差数据，绘制填充区域
                if any(s is not None for s in std):
                    lower = [yi - si if (yi is not None and si is not None) else None for yi, si in zip(y, std)]
                    upper = [yi + si if (yi is not None and si is not None) else None for yi, si in zip(y, std)]

                    # matplotlib 的 fill_between 需要数值数组，因此需要掩盖 None 点
                    xs = []
                    lvals = []
                    uvals = []
                    for xi, lo, hi in zip(x_obs, lower, upper):
                        if lo is None or hi is None:
                            continue
                        xs.append(xi)
                        lvals.append(lo)
                        uvals.append(hi)

                    if xs:
                        ax.fill_between(xs, lvals, uvals, color=color_map[agent], alpha=std_alpha)

            ax.set_xticks(x_obs)
            ax.grid(True, alpha=0.5)
            if metric in RATE_METRICS:
                ax.set_ylim(0.0, 105.0)

            ax.tick_params(axis='both', which='major', labelsize=12)

            if r == 0:
                ax.set_title(NAME_METRIC_MAP.get(metric, metric), fontsize=14)

            if c == 0:
                ax.set_ylabel(f"Robots = {robots}", fontsize=14)

            if r == rows - 1:
                ax.set_xlabel("Obstacle Number", fontsize=14)

    # 2) 全局图例（用你手动颜色）
    handles = [
        plt.Line2D([0], [0], color=color_map[a], marker=marker,
                   markersize=marker_size, linewidth=line_width)
        for a in agents
    ]

    name_list = [NAME_AGENT_MAP.get(a, a) for a in agents]
    legend_ncol = min(len(agents), 5)
    legend_rows = int(np.ceil(len(agents) / max(legend_ncol, 1)))
    # 调这里的两个值即可控制 legend 与子图的上下距离。
    legend_top = 0.95
    layout_top = 0.91 - 0.02 * max(legend_rows - 1, 0)

    fig.legend(
        handles,
        name_list,
        loc="upper center",
        ncol=legend_ncol,
        frameon=False,
        bbox_to_anchor=(0.5, legend_top),
        fontsize=14,
        columnspacing=1.2,
        handletextpad=0.6,
    )

    plt.tight_layout(rect=(0, 0, 1, layout_top))
    # plt.show()

    fig.savefig("./fig/comparison_plot.png", dpi=300, bbox_inches="tight", pad_inches=0.05)

if __name__ == "__main__":
    # 从 history/data 读取各参数对应数值；从 data 读取方差（var_*）
    history_dir = "/home/oem/direction_based_obstacle_avoidance/history/data"
    var_dir = "/home/oem/direction_based_obstacle_avoidance/data"

    history_list = load_all_json(history_dir)
    var_list = load_all_json(var_dir)

    # data 目录中的 json 同时包含均值与方差，这里与 history/data 合并，
    # 避免某些方法只存在于 data/ 中时无法被绘制。
    table, robots_nums, agents = build_table(history_list, METRICS)
    VAR_METRICS = ["var_" + m for m in METRICS]
    var_table, _, _ = build_table(var_list, VAR_METRICS)

    selected_robots_nums = [robot_num for robot_num in PLOT_ROBOTS if robot_num in robots_nums]

    print_data_tables(table, selected_robots_nums, agents, METRICS, X_OBS)
    AGENT_COLORS = {
        "Agent_IJRR": "#8c564b",
        "Agent_Lstm": "#ffbb78",
        "Agent_Linear": "#ff7f0e",
        "Agent_Lstm_Attn": "firebrick",
        "orca": "#9467bd",
        "neupan": "#2ca02c",
        "DRLVOAgent": "#17becf",
        "PaperCOAAgent":"#1f77b4",
        "CBFNav": "#9ecae1",
        "SICNav": "#1f5074",
    }
    size = 3
    agents = agents
    agents = [
        "orca",
        "neupan",
        "DRLVOAgent",
        "PaperCOAAgent",
        "Agent_IJRR",
        "SICNav",
        "CBFNav",
        "Agent_Lstm",
        "Agent_Linear",
        "Agent_Lstm_Attn",
        ]
    plot_all_agents_one_subplot(
        table, var_table, selected_robots_nums, agents,
        metrics=METRICS,
        x_obs=X_OBS,
        agent_colors=AGENT_COLORS,
        marker_size=size+1,      # 更小的点
        line_width=size,
        marker="o",
    )
