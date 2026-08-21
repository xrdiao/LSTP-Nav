import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon

try:
    from project_paths import FIG_TRAJECTORY_DIR, RECORD_TRAJECTORY_DIR
except ImportError:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from project_paths import FIG_TRAJECTORY_DIR, RECORD_TRAJECTORY_DIR


DEFAULT_TRAJECTORY_ROOT = RECORD_TRAJECTORY_DIR
DEFAULT_OUTPUT_ROOT = FIG_TRAJECTORY_DIR


class NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def load_trajectory(pkl_path):
    with open(pkl_path, "rb") as f:
        return NumpyCompatUnpickler(f).load()


def _episode_item(data, key):
    value = data.get(key, [])
    if len(value) == 1:
        return value[0]
    return value


def _sort_key(path):
    return (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem)


def _resolve_input_path(input_path=None):
    if input_path is None:
        if not DEFAULT_TRAJECTORY_ROOT.exists():
            raise FileNotFoundError("Trajectory root not found: {}".format(DEFAULT_TRAJECTORY_ROOT))
        return DEFAULT_TRAJECTORY_ROOT

    path = Path(input_path)
    if path.exists():
        return path

    path = DEFAULT_TRAJECTORY_ROOT / input_path
    if path.exists():
        return path

    raise FileNotFoundError("Trajectory path not found: {}".format(input_path))


def _iter_pkl_files(path):
    if path.is_file():
        if path.suffix != ".pkl":
            raise ValueError("Trajectory file must be a .pkl file: {}".format(path))
        return [path]

    return sorted(path.glob("*.pkl"), key=_sort_key)


def _iter_all_pkl_files(path):
    return sorted(path.rglob("*.pkl"), key=lambda p: (str(p.parent), _sort_key(p)))


def _path_to_array(path):
    steps = []
    for step in path:
        if step is None or len(step) == 0:
            continue
        arr = np.asarray(step, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] < 2:
            continue
        steps.append(arr[:, :2])

    if not steps:
        raise ValueError("No valid trajectory points found.")

    robot_num = min(step.shape[0] for step in steps)
    return np.stack([step[:robot_num] for step in steps], axis=0)


def _as_xy(points):
    if points is None or len(points) == 0:
        return np.empty((0, 2))
    arr = np.asarray(points, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr[:, :2]


def _normalize_obstacle(obstacle):
    if isinstance(obstacle, dict):
        return {
            "x": float(obstacle.get("x", 0.0)),
            "y": float(obstacle.get("y", 0.0)),
            "yaw": float(obstacle.get("yaw", obstacle.get("angle", 0.0))),
            "shape_type": obstacle.get("shape_type", "BOX"),
            "size_x": float(obstacle.get("size_x", 1.0)),
            "size_y": float(obstacle.get("size_y", 1.0)),
            "radius": float(obstacle.get("radius", 0.5)),
            "length": float(obstacle.get("length", 2.0)),
        }

    if len(obstacle) < 3:
        return None

    return {
        "x": float(obstacle[0]),
        "y": float(obstacle[1]),
        "yaw": float(obstacle[2]),
        "shape_type": obstacle[4] if len(obstacle) >= 5 else "BOX",
        "size_x": 1.0,
        "size_y": 1.0,
        "radius": 0.5,
        "length": 2.0,
    }


def _normalize_obstacles(obstacles):
    records = []
    for obstacle in obstacles:
        record = _normalize_obstacle(obstacle)
        if record is not None:
            records.append(record)
    return records


def _obstacles_xy(obstacles):
    xy = [[obstacle["x"], obstacle["y"]] for obstacle in obstacles]
    return np.asarray(xy, dtype=float) if xy else np.empty((0, 2))


def _rotated_box_corners(x, y, yaw, size_x=1.0, size_y=1.0):
    local = np.array([
        [-size_x / 2, -size_y / 2],
        [size_x / 2, -size_y / 2],
        [size_x / 2, size_y / 2],
        [-size_x / 2, size_y / 2],
    ])
    rotation = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)],
    ])
    return local.dot(rotation.T) + np.array([x, y])


def _draw_capsule(ax, obstacle, facecolor, edgecolor, alpha, linewidth, zorder):
    x, y, yaw = obstacle["x"], obstacle["y"], obstacle["yaw"]
    radius = obstacle["radius"]
    length = obstacle["length"]
    direction = np.array([np.cos(yaw), np.sin(yaw)])
    normal = np.array([-np.sin(yaw), np.cos(yaw)])
    half_line = max(length / 2, 0.0)

    start = np.array([x, y]) - direction * half_line
    end = np.array([x, y]) + direction * half_line
    corners = np.array([
        start + normal * radius,
        end + normal * radius,
        end - normal * radius,
        start - normal * radius,
    ])
    ax.add_patch(Polygon(corners, closed=True, facecolor=facecolor, edgecolor=edgecolor,
                         alpha=alpha, linewidth=linewidth, zorder=zorder))
    ax.add_patch(Circle(start, radius=radius, facecolor=facecolor, edgecolor=edgecolor,
                        alpha=alpha, linewidth=linewidth, zorder=zorder))
    ax.add_patch(Circle(end, radius=radius, facecolor=facecolor, edgecolor=edgecolor,
                        alpha=alpha, linewidth=linewidth, zorder=zorder))


def _draw_obstacles(ax, obstacles):
    colors = {
        "BOX": ("#f2c94c", "#7a4f00"),
        "CYLINDER": ("#9b51e0", "#3b1368"),
        "SPHERE": ("#eb5757", "#7a1616"),
        "CAPSULE": ("#27ae60", "#0b5f2a"),
    }
    alpha = 1.0
    linewidth = 1.8
    zorder = 3

    for obstacle in obstacles:
        x, y, yaw = obstacle["x"], obstacle["y"], obstacle["yaw"]
        shape_type = obstacle["shape_type"]
        facecolor, edgecolor = colors.get(shape_type, colors["BOX"])

        if shape_type in ("CYLINDER", "SPHERE"):
            patch = Circle(
                (x, y),
                radius=obstacle["radius"],
                facecolor=facecolor,
                edgecolor=edgecolor,
                alpha=alpha,
                linewidth=linewidth,
                zorder=zorder,
            )
            ax.add_patch(patch)
        elif shape_type == "CAPSULE":
            _draw_capsule(ax, obstacle, facecolor, edgecolor, alpha, linewidth, zorder)
        else:
            patch = Polygon(
                _rotated_box_corners(x, y, yaw, obstacle["size_x"], obstacle["size_y"]),
                closed=True,
                facecolor=facecolor,
                edgecolor=edgecolor,
                alpha=alpha,
                linewidth=linewidth,
                zorder=zorder,
            )
            ax.add_patch(patch)

        ax.scatter(x, y, s=12, color=edgecolor, marker="o", linewidths=0, zorder=zorder + 0.2)


def _auto_limits(ax, path_xy, init_xy, goal_xy, obstacles, pad=1.0):
    xs = [path_xy[:, :, 0].reshape(-1)]
    ys = [path_xy[:, :, 1].reshape(-1)]

    if len(init_xy):
        xs.append(init_xy[:, 0])
        ys.append(init_xy[:, 1])
    if len(goal_xy):
        xs.append(goal_xy[:, 0])
        ys.append(goal_xy[:, 1])
    if obstacles:
        obs_xy = _obstacles_xy(obstacles)
        if len(obs_xy):
            xs.append(obs_xy[:, 0])
            ys.append(obs_xy[:, 1])

    x = np.concatenate(xs)
    y = np.concatenate(ys)

    if np.isclose(x.min(), x.max()):
        ax.set_xlim(x.min() - pad, x.max() + pad)
    else:
        ax.set_xlim(x.min() - pad, x.max() + pad)

    if np.isclose(y.min(), y.max()):
        ax.set_ylim(y.min() - pad, y.max() + pad)
    else:
        ax.set_ylim(y.min() - pad, y.max() + pad)


def plot_trajectory(pkl_path, save_path=None, show=False, x_lim=None, y_lim=None, dpi=200):
    pkl_path = Path(pkl_path)
    data = load_trajectory(pkl_path)

    path_xy = _path_to_array(data.get("path", []))
    obstacles = _normalize_obstacles(_episode_item(data, "obstacles"))
    init_xy = _as_xy(_episode_item(data, "init_point"))
    goal_xy = _as_xy(_episode_item(data, "goal_point"))

    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(path_xy.shape[1], 1)))

    _draw_obstacles(ax, obstacles)

    for robot_idx in range(path_xy.shape[1]):
        xy = path_xy[:, robot_idx, :]
        color = colors[robot_idx % len(colors)]
        ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=1.6, alpha=0.95, zorder=5)
        ax.scatter(xy[-1, 0], xy[-1, 1], color=color, s=12, zorder=6)

    if len(init_xy):
        ax.scatter(
            init_xy[:, 0],
            init_xy[:, 1],
            marker="o",
            s=36,
            facecolors="white",
            edgecolors="black",
            linewidths=1.0,
            label="start",
            zorder=5,
        )
    if len(goal_xy):
        ax.scatter(
            goal_xy[:, 0],
            goal_xy[:, 1],
            marker="x",
            s=42,
            color="firebrick",
            linewidths=1.2,
            label="goal",
            zorder=5,
        )

    if x_lim is None or y_lim is None:
        _auto_limits(ax, path_xy, init_xy, goal_xy, obstacles)
    if x_lim is not None:
        ax.set_xlim(-x_lim, x_lim)
    if y_lim is not None:
        ax.set_ylim(-y_lim, y_lim)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("{} / test {}".format(pkl_path.parent.name, pkl_path.stem))
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, frameon=False, loc="best")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path


def plot_trajectory_dir(trajectory_dir, output_dir=None, show=False, x_lim=None, y_lim=None, dpi=200):
    trajectory_dir = Path(trajectory_dir)
    output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_ROOT / trajectory_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    for pkl_path in _iter_pkl_files(trajectory_dir):
        save_path = output_dir / "{}.png".format(pkl_path.stem)
        plot_trajectory(pkl_path, save_path=save_path, show=show, x_lim=x_lim, y_lim=y_lim, dpi=dpi)
        saved_files.append(save_path)

    return saved_files


def plot_all_trajectories(trajectory_root=DEFAULT_TRAJECTORY_ROOT, output_root=None, show=False, x_lim=None, y_lim=None, dpi=200):
    trajectory_root = Path(trajectory_root)
    output_root = Path(output_root) if output_root is not None else DEFAULT_OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)

    saved_files = []
    for pkl_path in _iter_all_pkl_files(trajectory_root):
        relative_parent = pkl_path.parent.relative_to(trajectory_root)
        save_path = output_root / relative_parent / "{}.png".format(pkl_path.stem)
        plot_trajectory(pkl_path, save_path=save_path, show=show, x_lim=x_lim, y_lim=y_lim, dpi=dpi)
        saved_files.append(save_path)

    return saved_files


def main():
    parser = argparse.ArgumentParser(description="Plot saved robot trajectories.")
    parser.add_argument(
        "trajectory",
        nargs="?",
        default=None,
        help="A .pkl file, a record_trajectory subfolder, or an absolute/relative path. Defaults to all record_trajectory files.",
    )
    parser.add_argument("-o", "--output", default=None, help="Output png path for a file, or output folder for a folder.")
    parser.add_argument("--show", action="store_true", help="Show the figure after plotting.")
    parser.add_argument("--x-lim", type=float, default=None, help="Set x axis range to [-x_lim, x_lim].")
    parser.add_argument("--y-lim", type=float, default=None, help="Set y axis range to [-y_lim, y_lim].")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    trajectory_path = _resolve_input_path(args.trajectory)

    if trajectory_path.is_file():
        output_path = Path(args.output) if args.output else DEFAULT_OUTPUT_ROOT / trajectory_path.parent.name / "{}.png".format(trajectory_path.stem)
        saved_files = [plot_trajectory(trajectory_path, output_path, args.show, args.x_lim, args.y_lim, args.dpi)]
    elif _iter_pkl_files(trajectory_path):
        saved_files = plot_trajectory_dir(trajectory_path, args.output, args.show, args.x_lim, args.y_lim, args.dpi)
    else:
        saved_files = plot_all_trajectories(trajectory_path, args.output, args.show, args.x_lim, args.y_lim, args.dpi)

    for saved_file in saved_files:
        print(saved_file)


if __name__ == "__main__":
    main()
