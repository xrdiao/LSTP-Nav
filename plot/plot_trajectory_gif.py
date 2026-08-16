import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plot.plot_trajectory import (
    DEFAULT_TRAJECTORY_ROOT,
    _as_xy,
    _auto_limits,
    _episode_item,
    _normalize_obstacles,
    _path_to_array,
    _resolve_input_path,
    _draw_obstacles,
    load_trajectory,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "fig" / "trajectory_gif"
DEFAULT_MAX_FRAMES = 600
DEFAULT_END_HOLD_SECONDS = 1.5


def _build_frame_indices(num_source_frames, frame_step, max_frames):
    if frame_step is not None and frame_step <= 0:
        raise ValueError("frame_step must be > 0")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be > 0")

    if frame_step is None:
        frame_step = 1

    min_step_for_max_frames = 1
    if max_frames is not None:
        min_step_for_max_frames = max(1, int(np.ceil(num_source_frames / float(max_frames))))
    effective_step = max(frame_step, min_step_for_max_frames)

    frame_indices = np.arange(0, num_source_frames, effective_step, dtype=int)
    if len(frame_indices) == 0 or frame_indices[-1] != num_source_frames - 1:
        frame_indices = np.append(frame_indices, num_source_frames - 1)

    return frame_indices, effective_step


def _make_writer(output_format, fps):
    if output_format == "gif":
        return PillowWriter(fps=fps)
    if output_format == "mp4":
        return FFMpegWriter(fps=fps, codec="libx264", bitrate=1800)
    raise ValueError("Unsupported output format: {}".format(output_format))


def save_trajectory_animation(
    pkl_path,
    save_path=None,
    x_lim=None,
    y_lim=None,
    dpi=120,
    fps=20,
    speed=1.0,
    frame_step=None,
    max_frames=DEFAULT_MAX_FRAMES,
    trail_length=None,
    end_hold_seconds=DEFAULT_END_HOLD_SECONDS,
    repeat=True,
    output_format=None,
):
    pkl_path = Path(pkl_path)
    data = load_trajectory(pkl_path)

    path_xy = _path_to_array(data.get("path", []))
    obstacles = _normalize_obstacles(_episode_item(data, "obstacles"))
    init_xy = _as_xy(_episode_item(data, "init_point"))
    goal_xy = _as_xy(_episode_item(data, "goal_point"))

    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(path_xy.shape[1], 1)))

    _draw_obstacles(ax, obstacles)

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

    line_artists = []
    point_artists = []
    for robot_idx in range(path_xy.shape[1]):
        color = colors[robot_idx % len(colors)]
        line, = ax.plot([], [], color=color, linewidth=1.8, alpha=0.95, zorder=6)
        point, = ax.plot([], [], marker="o", markersize=4.5, color=color, zorder=7)
        line_artists.append(line)
        point_artists.append(point)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, frameon=False, loc="best")

    if save_path is None:
        extension = ".{}".format(output_format or "gif")
        save_path = DEFAULT_OUTPUT_ROOT / pkl_path.parent.name / "{}{}".format(pkl_path.stem, extension)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format is None:
        output_format = save_path.suffix.lower().lstrip(".") or "gif"
    output_format = output_format.lower()

    frame_indices, effective_step = _build_frame_indices(path_xy.shape[0], frame_step, max_frames)
    output_fps = max(fps * speed, 1e-6)
    tail_frames = max(0, int(round(end_hold_seconds * output_fps)))
    total_frames = len(frame_indices) + tail_frames

    def _frame_slice(frame_idx):
        end_idx = frame_idx + 1
        if trail_length is None or trail_length <= 0:
            start_idx = 0
        else:
            start_idx = max(0, end_idx - trail_length)
        return slice(start_idx, end_idx)

    def update(frame_number):
        in_motion = frame_number < len(frame_indices)
        motion_frame_number = min(frame_number, len(frame_indices) - 1)
        real_frame_idx = frame_indices[motion_frame_number]
        visible_frame_indices = frame_indices[_frame_slice(motion_frame_number)]

        if not in_motion and len(visible_frame_indices):
            # Keep the full final frame once before gradually clearing the tail.
            trim = min(frame_number - len(frame_indices), len(visible_frame_indices))
            visible_frame_indices = visible_frame_indices[trim:]

        artists = []
        for robot_idx, (line, point) in enumerate(zip(line_artists, point_artists)):
            if len(visible_frame_indices):
                xy = path_xy[visible_frame_indices, robot_idx, :]
                line.set_data(xy[:, 0], xy[:, 1])
            else:
                line.set_data([], [])
            point.set_data([path_xy[real_frame_idx, robot_idx, 0]], [path_xy[real_frame_idx, robot_idx, 1]])
            artists.extend((line, point))
        return artists

    interval = max(1, int(round(1000.0 / output_fps)))
    animation = FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=interval,
        blit=True,
        repeat=repeat,
    )

    writer = _make_writer(output_format, output_fps)
    animation.save(save_path, writer=writer, dpi=dpi)
    plt.close(fig)
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Save one recorded trajectory as a GIF or MP4 animation.")
    parser.add_argument(
        "trajectory",
        help="A .pkl file path, or a path relative to record_trajectory.",
    )
    parser.add_argument("-o", "--output", default=None, help="Output .gif or .mp4 path.")
    parser.add_argument(
        "--format",
        choices=("gif", "mp4"),
        default=None,
        help="Output format. Defaults to the output suffix, otherwise gif.",
    )
    parser.add_argument("--x-lim", type=float, default=None, help="Set x axis range to [-x_lim, x_lim].")
    parser.add_argument("--y-lim", type=float, default=None, help="Set y axis range to [-y_lim, y_lim].")
    parser.add_argument("--dpi", type=int, default=120, help="Animation render dpi.")
    parser.add_argument("--fps", type=float, default=20.0, help="Base frames per second.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument(
        "--frame-step",
        type=int,
        default=None,
        help="Keep every Nth source frame. Default is auto, based on --max-frames.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help="Automatically increase frame stride to keep the output within this many frames.",
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=None,
        help="Only keep the latest N frames as the visible trajectory. Default keeps the full history.",
    )
    parser.add_argument(
        "--end-hold-seconds",
        type=float,
        default=DEFAULT_END_HOLD_SECONDS,
        help="After reaching the final step, keep the robots at the goal and gradually clear the trajectory tail.",
    )
    parser.add_argument("--no-repeat", action="store_true", help="Disable GIF looping.")
    args = parser.parse_args()

    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.speed <= 0:
        raise ValueError("--speed must be > 0")

    trajectory_path = _resolve_input_path(args.trajectory)
    if not trajectory_path.is_file() or trajectory_path.suffix != ".pkl":
        raise ValueError("trajectory must point to a single .pkl file: {}".format(trajectory_path))

    saved_file = save_trajectory_animation(
        trajectory_path,
        save_path=args.output,
        x_lim=args.x_lim,
        y_lim=args.y_lim,
        dpi=args.dpi,
        fps=args.fps,
        speed=args.speed,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
        trail_length=args.trail_length,
        end_hold_seconds=args.end_hold_seconds,
        repeat=not args.no_repeat,
        output_format=args.format,
    )
    print(saved_file)


if __name__ == "__main__":
    main()
