import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np


rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


AGENT_NAME = "Agent_Lstm_Attn"
ROBOT_COUNTS = [40, 80, 120, 160, 200]
OBSTACLE_COUNTS = [0, 100]
STYLE_MAP = {
    0: {"color": "#1f77b4", "label": "Obstacle Count = 0"},
    100: {"color": "#d62728", "label": "Obstacle Count = 100"},
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIG_DIR = os.path.join(BASE_DIR, "fig")
OUTPUT_PATH = os.path.join(FIG_DIR, "agent_lstm_attn_success_rate_dumbbell.png")


def load_record(robot_count, obstacle_count):
    file_name = f"{AGENT_NAME}_{robot_count}_{obstacle_count}.json"
    file_path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing data file: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        record = json.load(file)

    reach_rate = float(record["reach_rate"]) * 100.0
    reach_std = np.sqrt(max(float(record.get("var_reach_rate", 0.0)), 0.0)) * 100.0
    return reach_rate, reach_std


def build_data():
    data = {obstacle_count: {"rate": [], "std": []} for obstacle_count in OBSTACLE_COUNTS}
    for obstacle_count in OBSTACLE_COUNTS:
        for robot_count in ROBOT_COUNTS:
            reach_rate, reach_std = load_record(robot_count, obstacle_count)
            data[obstacle_count]["rate"].append(reach_rate)
            data[obstacle_count]["std"].append(reach_std)
    return data


def plot_dumbbell(data):
    os.makedirs(FIG_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.4, 5.8), dpi=160)
    y_positions = np.arange(len(ROBOT_COUNTS))[::-1]

    all_values = []
    for obstacle_count in OBSTACLE_COUNTS:
        rates = np.array(data[obstacle_count]["rate"], dtype=float)
        stds = np.array(data[obstacle_count]["std"], dtype=float)
        all_values.extend((rates - stds).tolist())
        all_values.extend((rates + stds).tolist())

    for idx, robot_count in enumerate(ROBOT_COUNTS):
        y = y_positions[idx]
        rate_0 = data[0]["rate"][idx]
        rate_100 = data[100]["rate"][idx]
        std_0 = data[0]["std"][idx]
        std_100 = data[100]["std"][idx]
        diff = rate_100 - rate_0

        ax.hlines(y, min(rate_0, rate_100), max(rate_0, rate_100),
                  color="#9aa0a6", linewidth=2.2, alpha=0.85, zorder=1)

        ax.errorbar(
            rate_0, y,
            xerr=std_0,
            fmt="o",
            color=STYLE_MAP[0]["color"],
            ecolor=STYLE_MAP[0]["color"],
            markersize=8,
            elinewidth=1.3,
            capsize=4,
            zorder=3,
        )
        ax.errorbar(
            rate_100, y,
            xerr=std_100,
            fmt="o",
            color=STYLE_MAP[100]["color"],
            ecolor=STYLE_MAP[100]["color"],
            markersize=8,
            elinewidth=1.3,
            capsize=4,
            zorder=3,
        )

        mid_x = (rate_0 + rate_100) / 2
        ax.text(
            mid_x,
            y + 0.18,
            f"{diff:+.1f} pts",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#4a4a4a",
        )

        ax.text(rate_0 - 1.6, y - 0.18, f"{rate_0:.1f}", ha="right", va="top",
                fontsize=8.5, color=STYLE_MAP[0]["color"])
        ax.text(rate_100 + 1.6, y - 0.18, f"{rate_100:.1f}", ha="left", va="top",
                fontsize=8.5, color=STYLE_MAP[100]["color"])

    x_min = max(0.0, min(all_values) - 6.0)
    x_max = min(100.0, max(all_values) + 6.0)

    ax.set_title("Agent_Lstm_Attn Success Rate Comparison", fontsize=15, pad=14)
    ax.set_xlabel("Success Rate (%)", fontsize=12)
    ax.set_ylabel("Number of Robots", fontsize=12)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([str(robot_count) for robot_count in ROBOT_COUNTS], fontsize=11)
    ax.set_xlim(x_min, x_max)
    ax.grid(axis="x", linestyle="--", alpha=0.28)
    ax.grid(axis="y", visible=False)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color=STYLE_MAP[0]["color"], linewidth=0,
                   markersize=8, label=STYLE_MAP[0]["label"]),
        plt.Line2D([0], [0], marker="o", color=STYLE_MAP[100]["color"], linewidth=0,
                   markersize=8, label=STYLE_MAP[100]["label"]),
        plt.Line2D([0], [0], color="#9aa0a6", linewidth=2.2, label="Gap Between Two Settings"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="lower left", fontsize=10)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main():
    data = build_data()
    plot_dumbbell(data)
    print(f"Saved figure to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
