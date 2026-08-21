import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np


rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

try:
    from project_paths import DATA_DIR, FIG_DIR
except ImportError:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from project_paths import DATA_DIR, FIG_DIR


AGENT_NAME = "Agent_Lstm_Attn"
ROBOT_COUNTS = [40, 80, 120, 160, 200]
OBSTACLE_COUNTS = [0, 100]
STYLE_MAP = {
    0: {"color": "#1f77b4", "label": "Obstacle Count = 0"},
    100: {"color": "#d62728", "label": "Obstacle Count = 100"},
}

OUTPUT_PATH = FIG_DIR / "agent_lstm_attn_success_rate_0_100.png"


def load_record(robot_count, obstacle_count):
    file_name = f"{AGENT_NAME}_{robot_count}_{obstacle_count}.json"
    file_path = DATA_DIR / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"Missing data file: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        record = json.load(file)

    if record.get("agent_name") != AGENT_NAME:
        raise ValueError(f"Unexpected agent name in {file_path}: {record.get('agent_name')}")
    if record.get("robots_num") != robot_count:
        raise ValueError(f"Unexpected robots_num in {file_path}: {record.get('robots_num')}")
    if record.get("obstacles") != obstacle_count:
        raise ValueError(f"Unexpected obstacles in {file_path}: {record.get('obstacles')}")

    reach_rate = float(record["reach_rate"]) * 100.0
    var_reach_rate = max(float(record.get("var_reach_rate", 0.0)), 0.0)
    reach_std = np.sqrt(var_reach_rate) * 100.0

    return {
        "robot_count": robot_count,
        "obstacle_count": obstacle_count,
        "reach_rate": reach_rate,
        "reach_std": reach_std,
    }


def build_series():
    series = {}
    for obstacle_count in OBSTACLE_COUNTS:
        records = [load_record(robot_count, obstacle_count) for robot_count in ROBOT_COUNTS]
        series[obstacle_count] = {
            "x": [record["robot_count"] for record in records],
            "y": [record["reach_rate"] for record in records],
            "std": [record["reach_std"] for record in records],
        }
    return series


def plot_series(series):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.8, 5.6), dpi=150)

    lower_candidates = []
    upper_candidates = []

    for obstacle_count in OBSTACLE_COUNTS:
        style = STYLE_MAP[obstacle_count]
        x = np.array(series[obstacle_count]["x"], dtype=float)
        y = np.array(series[obstacle_count]["y"], dtype=float)
        std = np.array(series[obstacle_count]["std"], dtype=float)

        lower = np.clip(y - std, 0.0, 100.0)
        upper = np.clip(y + std, 0.0, 100.0)
        lower_candidates.extend(lower.tolist())
        upper_candidates.extend(upper.tolist())

        ax.plot(
            x,
            y,
            color=style["color"],
            marker="o",
            linewidth=2.2,
            markersize=6,
            label=style["label"],
        )
        ax.fill_between(x, lower, upper, color=style["color"], alpha=0.16)
        ax.errorbar(
            x,
            y,
            yerr=std,
            fmt="none",
            ecolor=style["color"],
            elinewidth=1.2,
            capsize=4,
            alpha=0.95,
        )

    y_min = max(0.0, min(lower_candidates) - 4.0)
    y_max = min(100.0, max(upper_candidates) + 4.0)

    # ax.set_title("Agent_Lstm_Attn Success Rate vs Robot Count", fontsize=14)
    ax.set_xlabel("Number of Robots", fontsize=12)
    ax.set_ylabel("SR (%)", fontsize=12)
    ax.set_xticks(ROBOT_COUNTS)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.35, linestyle="--")
    ax.legend(frameon=False, fontsize=11)

    plt.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def print_summary(series):
    print("Agent_Lstm_Attn success rate summary")
    for obstacle_count in OBSTACLE_COUNTS:
        print(f"\nObstacle Count = {obstacle_count}")
        for robot_count, reach_rate, reach_std in zip(
            series[obstacle_count]["x"],
            series[obstacle_count]["y"],
            series[obstacle_count]["std"],
        ):
            print(
                f"robots={robot_count:>3d} | "
                f"success_rate={reach_rate:>6.2f}% | "
                f"std={reach_std:>5.2f}%"
            )


def main():
    series = build_series()
    print_summary(series)
    plot_series(series)
    print(f"\nSaved figure to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
