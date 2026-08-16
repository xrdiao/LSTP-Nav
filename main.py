import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 加载数据
laser_buffer = np.load('laser_buffer.npy')
num_frames, num_points = laser_buffer.shape

# 创建自定义颜色渐变
cmap = LinearSegmentedColormap.from_list('radar_cmap', [
    '#003f5c', '#2f4b7c', '#665191', '#a05195', 
    '#d45087', '#f95d6a', '#ff7c43', '#ffa600'
], N=256)

# 创建3D图形 - 调整整体画布大小
fig = plt.figure(figsize=(14, 10), facecolor='#0f0f23')  # 缩小画布高度

# 调整坐标轴位置和大小
ax = fig.add_axes([0.05, 0.05, 0.8, 0.9], projection='3d')  # 设置坐标轴位置和大小

# 设置视角
ax.view_init(elev=35, azim=-90)  # 从侧面观察半圆

# 生成半圆角度数据 (0°-180°)
angles = np.linspace(-np.pi/2, np.pi/2, num_points, endpoint=False)
times = np.arange(num_frames)

# 创建顶点数据 - 每个柱体的4个顶点
verts = []
colors = []
max_radius = np.max(laser_buffer) * 1.2

for i in range(num_frames):
    frame_colors = cmap(np.full(num_points, i / num_frames))
    for j in range(num_points):
        # 计算柱体底部和顶部的顶点
        z0 = times[i]
        z1 = times[i] + 0.8  # 柱体高度
        
        angle = angles[j]
        radius = laser_buffer[i, j]
        
        # 创建四边形柱体的4个顶点
        v = [
            (radius * np.cos(angle - 0.02), radius * np.sin(angle - 0.02), z0),  # 左下
            (radius * np.cos(angle + 0.02), radius * np.sin(angle + 0.02), z0),  # 右下
            (radius * np.cos(angle + 0.02), radius * np.sin(angle + 0.02), z1),  # 右上
            (radius * np.cos(angle - 0.02), radius * np.sin(angle - 0.02), z1)   # 左上
        ]
        verts.append(v)
        colors.append(frame_colors[j])

# 创建柱体集合
poly = Poly3DCollection(verts, facecolors=colors, edgecolors='#ffffff22', 
                        linewidths=0.8, alpha=0.85)
ax.add_collection3d(poly)

# 添加时间平面 (半圆)
for i in range(num_frames):
    theta = np.linspace(-np.pi/2, np.pi/2, 50)  # 半圆角度范围
    x = max_radius * np.cos(theta)
    y = max_radius * np.sin(theta)
    z = np.full_like(theta, i)
    ax.plot(x, y, z, color='#ffffff66', lw=0.8, alpha=0.25)

# 添加传感器角度线 (半圆)
for j in range(0, num_points, max(1, num_points//12)):
    theta = angles[j]
    x = max_radius * np.cos(theta)
    y = max_radius * np.sin(theta)
    for i in range(num_frames):
        z = i
        ax.plot([0, x], [0, y], [z, z], color='#ffffff33', lw=0.5, alpha=0.2)

# 添加雷达原点标记
for i in range(num_frames):
    ax.scatter([0], [0], [i], s=30, c='#ff5555', edgecolors='white', alpha=1, zorder=10)

# 设置坐标轴范围
ax.set_xlim(-max_radius, max_radius)
ax.set_ylim(0, max_radius)  # 半圆只显示正Y区域
ax.set_zlim(0, num_frames)

# 隐藏坐标轴（轴线、刻度、刻度标签）
ax.set_axis_off()  # 隐藏所有坐标轴

# 添加坐标轴标签（文本形式） - 位置调整
ax.text(-max_radius*0.95, 0, num_frames/2, 'Distance (m)', color='white', fontsize=11)
ax.text(0, max_radius*0.95, num_frames/2, 'Y Axis', color='white', fontsize=11)
ax.text(0, 0, num_frames*1.05, 'Time Step', color='white', fontsize=11)


# 添加注释 - 位置调整
# ax.text2D(0.05, 0.95, f"Frames: {num_frames} | Sensors: {num_points}", 
#           transform=ax.transAxes, fontsize=11, color='#a0a0ff')
# ax.text2D(0.05, 0.90, f"Field of View: 180°", 
#           transform=ax.transAxes, fontsize=9, color='#ffa600')

# 添加方向指示器
ax.quiver(0, 0, num_frames, max_radius*0.5, 0, 0, color="#ff6666", arrow_length_ratio=0.1, lw=2)
ax.text(max_radius*0.6, 0, num_frames, "Front", color="#ff6666", fontsize=10)

# 添加窄版时间序列颜色条 - 更贴近可视化圆柱体
cbar_height = 0.7
cbar_width = 0.015  # 更窄的宽度
cbar_left = 0.85    # 更靠近圆柱体位置
cbar_bottom = 0.15

cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, num_frames-1))
cbar = fig.colorbar(sm, cax=cax)

# 窄版颜色条样式
cbar.set_label('Time\nSeq', rotation=0, labelpad=5, color='white', 
               fontsize=11, y=0.5, va='center')  # 简化标签
cbar.ax.tick_params(colors='white', labelsize=8, length=3, width=1)
cbar.outline.set_linewidth(0.5)

# 移除刻度线，使用标签替代
cbar.ax.yaxis.set_ticks([])

# 添加关键时间点标签（垂直排列节省空间）
for i in [0, num_frames//4, num_frames//2, 3*num_frames//4, num_frames-1]:
    rel_pos = i / (num_frames-1)
    cax.text(0.5, rel_pos, f"T{i}", ha='center', va='center', color='white', 
             fontsize=7, transform=cax.transAxes, 
             bbox=dict(boxstyle='round,pad=0.1', facecolor='#00000088', edgecolor='none'))

# 设置网格和样式
ax.grid(True, color='#3a3a5a', linestyle='--', linewidth=0.5, alpha=0.25)

# 添加视角调整按钮说明 - 位置调整
# ax.text2D(0.05, 0.04, "Rotate: Left Mouse\nZoom: Right Mouse", 
#           transform=ax.transAxes, fontsize=8, color='#a0a0a0', alpha=0.7)

# 添加数据统计信息 - 位置调整
# avg_distance = np.mean(laser_buffer)
# max_distance = np.max(laser_buffer)
# ax.text2D(0.05, 0.86, f"Avg: {avg_distance:.1f}m | Max: {max_distance:.1f}m", 
#           transform=ax.transAxes, fontsize=9, color='#00ccff',
#           bbox=dict(boxstyle='round,pad=0.2', facecolor='#00000055', edgecolor='none'))

# 为颜色条添加背景增强可读性
cax.set_facecolor('#00000055')

# 添加半透明背景框增强可读性
fig.patches.extend([plt.Rectangle((0.05, 0.05), 0.8, 0.9, 
                                 fill=True, color='#00000022', 
                                 transform=fig.transFigure, zorder=0)])

plt.savefig('radar_visualization_optimized.png', dpi=120, facecolor=fig.get_facecolor())
plt.show()