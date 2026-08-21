import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import sys
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import FIG_DIR, LASER_BUFFER_PATH

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=LASER_BUFFER_PATH, help="path to the saved laser buffer .npy file")
    parser.add_argument("--output", type=Path, default=FIG_DIR / "radar_visualization_optimized.png",
                        help="path to the output figure")
    parser.add_argument("--no-show", action="store_true", help="save the figure without opening a window")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Laser buffer not found: {args.input}")

    laser_buffer = np.load(args.input)
    num_frames, num_points = laser_buffer.shape

    cmap = LinearSegmentedColormap.from_list('radar_cmap', [
        '#003f5c', '#2f4b7c', '#665191', '#a05195',
        '#d45087', '#f95d6a', '#ff7c43', '#ffa600'
    ], N=256)

    fig = plt.figure(figsize=(14, 10), facecolor='#0f0f23')
    ax = fig.add_axes([0.05, 0.05, 0.8, 0.9], projection='3d')
    ax.view_init(elev=35, azim=-90)

    angles = np.linspace(-np.pi/2, np.pi/2, num_points, endpoint=False)
    times = np.arange(num_frames)
    verts = []
    colors = []
    max_radius = np.max(laser_buffer) * 1.2

    for i in range(num_frames):
        frame_colors = cmap(np.full(num_points, i / num_frames))
        for j in range(num_points):
            z0 = times[i]
            z1 = times[i] + 0.8
            angle = angles[j]
            radius = laser_buffer[i, j]

            verts.append([
                (radius * np.cos(angle - 0.02), radius * np.sin(angle - 0.02), z0),
                (radius * np.cos(angle + 0.02), radius * np.sin(angle + 0.02), z0),
                (radius * np.cos(angle + 0.02), radius * np.sin(angle + 0.02), z1),
                (radius * np.cos(angle - 0.02), radius * np.sin(angle - 0.02), z1)
            ])
            colors.append(frame_colors[j])

    poly = Poly3DCollection(verts, facecolors=colors, edgecolors='#ffffff22',
                            linewidths=0.8, alpha=0.85)
    ax.add_collection3d(poly)

    for i in range(num_frames):
        theta = np.linspace(-np.pi/2, np.pi/2, 50)
        x = max_radius * np.cos(theta)
        y = max_radius * np.sin(theta)
        z = np.full_like(theta, i)
        ax.plot(x, y, z, color='#ffffff66', lw=0.8, alpha=0.25)

    for j in range(0, num_points, max(1, num_points // 12)):
        theta = angles[j]
        x = max_radius * np.cos(theta)
        y = max_radius * np.sin(theta)
        for i in range(num_frames):
            ax.plot([0, x], [0, y], [i, i], color='#ffffff33', lw=0.5, alpha=0.2)

    for i in range(num_frames):
        ax.scatter([0], [0], [i], s=30, c='#ff5555', edgecolors='white', alpha=1, zorder=10)

    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(0, max_radius)
    ax.set_zlim(0, num_frames)
    ax.set_axis_off()
    ax.text(-max_radius * 0.95, 0, num_frames / 2, 'Distance (m)', color='white', fontsize=11)
    ax.text(0, max_radius * 0.95, num_frames / 2, 'Y Axis', color='white', fontsize=11)
    ax.text(0, 0, num_frames * 1.05, 'Time Step', color='white', fontsize=11)
    ax.quiver(0, 0, num_frames, max_radius * 0.5, 0, 0, color="#ff6666", arrow_length_ratio=0.1, lw=2)
    ax.text(max_radius * 0.6, 0, num_frames, "Front", color="#ff6666", fontsize=10)

    cax = fig.add_axes([0.85, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, num_frames - 1))
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Time\nSeq', rotation=0, labelpad=5, color='white', fontsize=11, y=0.5, va='center')
    cbar.ax.tick_params(colors='white', labelsize=8, length=3, width=1)
    cbar.outline.set_linewidth(0.5)
    cbar.ax.yaxis.set_ticks([])

    for i in [0, num_frames // 4, num_frames // 2, 3 * num_frames // 4, num_frames - 1]:
        rel_pos = i / (num_frames - 1)
        cax.text(0.5, rel_pos, f"T{i}", ha='center', va='center', color='white',
                 fontsize=7, transform=cax.transAxes,
                 bbox=dict(boxstyle='round,pad=0.1', facecolor='#00000088', edgecolor='none'))

    ax.grid(True, color='#3a3a5a', linestyle='--', linewidth=0.5, alpha=0.25)
    cax.set_facecolor('#00000055')
    fig.patches.extend([plt.Rectangle((0.05, 0.05), 0.8, 0.9,
                                      fill=True, color='#00000022',
                                      transform=fig.transFigure, zorder=0)])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=120, facecolor=fig.get_facecolor())
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
