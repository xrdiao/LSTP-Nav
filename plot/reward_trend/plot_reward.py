from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

try:
    from project_paths import FIG_DIR
except ImportError:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from project_paths import FIG_DIR

BASE_DIR = Path(__file__).resolve().parent
TARGET_DIRS = [
    "with HS reward",
    "without HS reward",
]
HIGHLIGHT_LABEL = "LSTP-Net"
HIGHLIGHT_COLOR = "red"
OUTPUT_DIR = FIG_DIR / "reward_trend"

def tensorboard_smooth(scalars, weight=0.6):
    """TensorBoard使用的指数加权移动平均平滑算法"""
    last = scalars[0]
    smoothed = []
    for point in scalars:
        if last is None:
            smoothed_val = point
        else:
            smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def millions_formatter(x, pos):
    """将x轴数值转换为百万(M)单位显示"""
    return f'{x*1e-6:.1f}M' if x >= 1e6 else f'{x*1e-3:.0f}K' if x >= 1e3 else f'{x:.0f}'


def sort_files_for_legend(files):
    """Ensure the highlighted series is plotted last so it appears last in the legend."""
    return sorted(
        files,
        key=lambda file_path: (
            file_path.stem == HIGHLIGHT_LABEL,
            file_path.name.lower(),
        ),
    )

def plot_separate_tensorboard_figures():
    # 仅从指定子目录读取CSV文件
    file_groups = {}
    for relative_dir in TARGET_DIRS:
        input_dir = BASE_DIR / relative_dir
        if not input_dir.is_dir():
            print(f"目录不存在，已跳过: {input_dir}")
            continue

        csv_files = sorted(path for path in input_dir.iterdir() if path.suffix == ".csv")
        if csv_files:
            file_groups[relative_dir] = sort_files_for_legend(csv_files)

    if not file_groups:
        print("没有在指定文件夹中找到任何CSV文件")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 使用tab20颜色循环
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    # 为每个子文件夹创建独立的Figure
    for group_name, files in file_groups.items():
        # 创建新Figure
        plt.figure(figsize=(14, 6))
        
        # 遍历当前子文件夹中的所有CSV文件
        for i, file in enumerate(files):
            try:
                # 读取CSV文件
                df = pd.read_csv(file)
                
                if 'Step' in df.columns and 'Value' in df.columns:
                    # 提取指标名称作为图例标签
                    label = file.stem
                    
                    # 获取当前颜色
                    color = HIGHLIGHT_COLOR if label == HIGHLIGHT_LABEL else colors[i % len(colors)]
                    
                    # 绘制原始数据（半透明）
                    plt.plot(df['Step'], df['Value'], 
                             alpha=0.15, linewidth=2, color=color)
                    
                    # 绘制平滑数据
                    smoothed = tensorboard_smooth(df['Value'], 0.98)
                    plt.plot(df['Step'], smoothed, 
                             label=label, linewidth=4, color=color)
                
            except Exception as e:
                print(f"处理文件 {file} 时出错: {str(e)}")
        
        # 设置图表标题和标签
        # title = f'TensorBoard Metrics - {os.path.basename(dirname) or "Root"}' 
        # plt.title(title, fontsize=14)
        plt.xlabel('Step', fontsize=20)
        plt.ylabel('Reward', fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=18)  # 新增这行

        # 设置x轴格式
        plt.gca().xaxis.set_major_formatter(FuncFormatter(millions_formatter))
                # 确保y轴从0开始（关键修改！）
        # 使用min_y和当前最大y值确定范围，但确保最小值是0
        # 确保x轴从0开始
        # _, current_xmax = plt.gca().get_xlim()
        # plt.xlim(left=0, right=current_xmax)

        # 添加图例和网格
        plt.legend(loc='lower right', fontsize=17, framealpha=0.9)
        plt.grid(True, alpha=0.2)
        
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_linewidth(3)  # 设置边框线宽为3（默认是0.8-1.0）

        # 调整布局
        plt.tight_layout()

        output_name = f"{group_name.lower().replace(' ', '_')}.png"
        output_path = OUTPUT_DIR / output_name
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"已保存图像: {output_path}")

if __name__ == "__main__":
    plot_separate_tensorboard_figures()
