import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
import pandas as pd

METRICS = ["reach_rate", "collision_rate", "trap_rate", "avg_time", "avg_step"]
NAME_METRIC_MAP = {"reach_rate": "成功率(%)",
                   "collision_rate": "碰撞率(%)",
                   "trap_rate": "被困率(%)",
                   "avg_time": "平均时间(s)",
                   "avg_step": "平均步数"}
NAME_AGENT_MAP = {
    "Agent_IJRR": "HybridRL-CA",
    "Agent_Lstm": "TP-Net",
    "Agent_Linear": "SP-Net",
    "Agent_Lstm_Attn": "LSTP-Net",
    "orca": "NH-ORCA",
    "neupan": "NeuPan"}
X_OBS = [5, 10, 15, 20, 25, 30, 35]

def load_all_json(data_dir="./data"):
    data = []
    for fn in os.listdir(data_dir):
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
            
            # 创建DataFrame
            data_dict = {}
            for agent in agents:
                data_dict[NAME_AGENT_MAP.get(agent, agent)] = [
                    table[robots][metric].get(agent, {}).get(obs, "N/A") 
                    for obs in x_obs
                ]
            
            df = pd.DataFrame(data_dict, index=x_obs)
            df.index.name = "Obstacles"
            print(df.round(4))  # 保留4位小数

import matplotlib.pyplot as plt

def plot_all_agents_one_subplot(
    table, robots_nums, agents,
    metrics,
    x_obs,
    agent_colors=None,          # 手动颜色映射: {"Agent_IJRR": "C0", ...}
    marker_size=3,              # 节点大小（建议 2~4）
    line_width=1.5,
    marker="o",
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

    for r, robots in enumerate(robots_nums):
        for c, metric in enumerate(metrics):
            ax = axes[r][c]

            for agent in agents:
                y = [table[robots][metric].get(agent, {}).get(obs, None) for obs in x_obs]
                if all(v is None for v in y):
                    continue

                ax.plot(
                    x_obs, y,
                    color=color_map[agent],
                    marker=marker,
                    markersize=marker_size,   # 节点大小
                    linewidth=line_width,
                    label=agent
                )

            ax.set_xticks(x_obs)
            ax.grid(True, alpha=0.5)

            ax.tick_params(axis='both', which='major', labelsize=12)

            if r == 0:
                ax.set_title(NAME_METRIC_MAP.get(metric, metric), fontsize=14)

            if c == 0:
                ax.set_ylabel(f"机器人数量={robots}", fontsize=14)

            if r == rows - 1:
                ax.set_xlabel("障碍物数量", fontsize=14)

    # 2) 全局图例（用你手动颜色）
    handles = [
        plt.Line2D([0], [0], color=color_map[a], marker=marker,
                   markersize=marker_size, linewidth=line_width)
        for a in agents
    ]

    name_list = [NAME_AGENT_MAP.get(a, a) for a in agents]
    fig.legend(handles, name_list, loc="upper center",
               ncol=min(len(agents), 6), frameon=False, bbox_to_anchor=(0.5, 0.99), fontsize=14)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    # plt.show()

    fig.savefig("./fig/comparison_plot.png", dpi=300)

if __name__ == "__main__":
    data_list = load_all_json("./data")
    table, robots_nums, agents = build_table(data_list, METRICS)
    print_data_tables(table, robots_nums, agents, METRICS, X_OBS)
    AGENT_COLORS = {
        "Agent_IJRR": "#1f77b4",
        "Agent_Lstm": "#ffbb78",
        "Agent_Linear": "#ff7f0e",
        "Agent_Lstm_Attn": "firebrick",
        "ORCA": "#9467bd",
        "neupan": "#2ca02c",
    }
    size = 3
    agents = agents
    agents = [
        "orca",
        "neupan",
        "Agent_IJRR",
        "Agent_Lstm",
        "Agent_Linear",
        "Agent_Lstm_Attn",
        ]
    plot_all_agents_one_subplot(
        table, robots_nums, agents,
        metrics=METRICS,
        x_obs=X_OBS,
        agent_colors=AGENT_COLORS,
        marker_size=size+1,      # 更小的点
        line_width=size,
        marker="o",
    )
