import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
import numpy as np
from matplotlib.path import Path

# Create figure with constrained layout
fig = plt.figure(figsize=(14, 14), dpi=100)
ax = fig.add_subplot(111)
ax.set_aspect('equal')
ax.axis('off')
plt.title('Attention-Based Navigation Network Architecture', fontsize=16, pad=20)

# Color definitions
COLORS = {
    'input': '#FFD700',       # Gold
    'lstm': '#4682B4',        # Steel blue
    'attention': '#32CD32',   # Lime green
    'residual': '#FF6347',    # Tomato
    'fusion': '#9370DB',      # Medium purple
    'actor': '#20B2AA',       # Light sea green
    'critic': '#FF4500',      # Orange red
    'output': '#8A2BE2'       # Blue violet
}

# Vertical positioning (adjusted for better spacing)
input_y = 9
gru_y = 7.5
attention_y = 5.5
fusion_y = 3.5
actor_critic_y = 1.5
details_y = 0

# Draw input layer
plt.text(0, input_y + 0.8, "Inputs", fontsize=14, ha='center', weight='bold')
ax.add_patch(Rectangle((0.5, input_y), 1, 0.7, color=COLORS['input'], alpha=0.8))
plt.text(1, input_y + 0.35, "LiDAR Scan\n(B×T×N_laser)", ha='center', fontsize=10)

ax.add_patch(Rectangle((2.5, input_y), 1, 0.7, color=COLORS['input'], alpha=0.8))
plt.text(3, input_y + 0.35, "Goal & Velocity\n(B×4)", ha='center', fontsize=10)

# Draw GRU layer
plt.text(1, gru_y + 0.8, "Temporal Encoder", fontsize=12, ha='center', style='italic')
ax.add_patch(Rectangle((0.5, gru_y), 3, 0.8, color=COLORS['lstm'], alpha=0.8))
plt.text(2, gru_y + 0.4, "2-Layer Bidirectional GRU\n(hidden_dim=256)", ha='center', fontsize=11)

# Draw attention layer
plt.text(1, attention_y + 1.2, "Multi-Head Attention", fontsize=12, ha='center', style='italic')
ax.add_patch(Rectangle((0.5, attention_y + 0.8), 1.5, 0.8, color=COLORS['attention'], alpha=0.8))
plt.text(1.25, attention_y + 1.2, "Query: Last Frame", ha='center', fontsize=10)

ax.add_patch(Rectangle((1.5, attention_y), 1.5, 0.8, color=COLORS['attention'], alpha=0.8))
plt.text(2.25, attention_y + 0.4, "Key & Value: Full Sequence", ha='center', fontsize=10)

ax.add_patch(Rectangle((1, attention_y - 0.8), 2, 0.8, color=COLORS['attention'], alpha=0.8))
plt.text(2, attention_y - 0.4, "4-Head Attention\n(Embed_dim=256)", ha='center', fontsize=11)

# Draw residual connection
plt.text(3.8, attention_y + 1.2, "Residual Encoding", fontsize=12, ha='center', style='italic')
ax.add_patch(Rectangle((3, attention_y + 0.5), 1.5, 0.7, color=COLORS['residual'], alpha=0.8))
plt.text(3.75, attention_y + 0.85, "FC(256) + ELU", ha='center', fontsize=10)

ax.add_patch(Rectangle((3, attention_y - 0.2), 1.5, 0.7, color=COLORS['residual'], alpha=0.8))
plt.text(3.75, attention_y + 0.15, "Residual Connection", ha='center', fontsize=10)

# Draw feature fusion
ax.add_patch(Rectangle((1.5, fusion_y), 2, 0.8, color=COLORS['fusion'], alpha=0.8))
plt.text(2.5, fusion_y + 0.4, "Feature Fusion\nConcat(Context + State)", 
         ha='center', fontsize=12, weight='bold')
plt.text(2.5, fusion_y - 0.2, "(Output: B×512)", ha='center', fontsize=10)

# Draw Actor network
plt.text(0.5, actor_critic_y + 1.3, "Actor Network", fontsize=14, ha='center', weight='bold')
ax.add_patch(Rectangle((0, actor_critic_y), 1, 0.7, color=COLORS['actor'], alpha=0.8))
plt.text(0.5, actor_critic_y + 0.35, "FC(256) + ELU", ha='center', fontsize=10)

ax.add_patch(Rectangle((0, actor_critic_y - 1.0), 1, 0.7, color=COLORS['actor'], alpha=0.8))
plt.text(0.5, actor_critic_y - 0.65, "FC(128) + ELU", ha='center', fontsize=10)

ax.add_patch(Rectangle((0, actor_critic_y - 2.0), 1, 0.7, color=COLORS['output'], alpha=0.8))
plt.text(0.5, actor_critic_y - 1.65, "Action (v, ω)", ha='center', fontsize=11, weight='bold')

# Draw Critic network
plt.text(4.5, actor_critic_y + 1.3, "Critic Network", fontsize=14, ha='center', weight='bold')
ax.add_patch(Rectangle((4, actor_critic_y), 1, 0.7, color=COLORS['critic'], alpha=0.8))
plt.text(4.5, actor_critic_y + 0.35, "FC(256) + ELU", ha='center', fontsize=10)

ax.add_patch(Rectangle((4, actor_critic_y - 1.0), 1, 0.7, color=COLORS['critic'], alpha=0.8))
plt.text(4.5, actor_critic_y - 0.65, "FC(128) + ELU", ha='center', fontsize=10)

ax.add_patch(Rectangle((4, actor_critic_y - 2.0), 1, 0.7, color=COLORS['output'], alpha=0.8))
plt.text(4.5, actor_critic_y - 1.65, "State Value", ha='center', fontsize=11, weight='bold')

# Draw connection arrows
def draw_arrow(start, end, color='k', linestyle='-', width=1.5):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', lw=width, 
                                color=color, linestyle=linestyle, 
                                alpha=0.7))

# Input to processing layers
draw_arrow((1, input_y), (1, gru_y + 0.8), color=COLORS['lstm'])
draw_arrow((3, input_y), (3, attention_y + 0.85), color=COLORS['residual'])

# GRU to Attention
draw_arrow((1, gru_y), (1, attention_y + 1.6), color=COLORS['attention'], linestyle='-')
draw_arrow((1, gru_y), (2.25, attention_y + 0.8), color=COLORS['attention'], linestyle='-')
draw_arrow((1, gru_y), (2.25, attention_y), color=COLORS['attention'], linestyle='-')

# Residual connection
draw_arrow((3.75, attention_y + 0.5), (3.75, attention_y - 0.2), color=COLORS['residual'], linestyle='-')
draw_arrow((3.75, attention_y - 0.2), (3.75, attention_y - 0.9), color=COLORS['residual'], linestyle='-')

# Attention fusion
draw_arrow((2, attention_y - 0.8), (2.5, fusion_y + 0.8), color=COLORS['attention'], linestyle='-')
draw_arrow((3.75, attention_y - 0.9), (3.25, fusion_y + 0.8), color=COLORS['residual'], linestyle='-')

# To Actor and Critic
draw_arrow((2.5, fusion_y), (0.5, actor_critic_y + 0.7), color=COLORS['actor'], linestyle='-')
draw_arrow((0.5, actor_critic_y), (0.5, actor_critic_y - 0.3), color=COLORS['actor'], linestyle='-')
draw_arrow((0.5, actor_critic_y - 1.0), (0.5, actor_critic_y - 1.3), color=COLORS['actor'], linestyle='-')

draw_arrow((2.5, fusion_y), (4.5, actor_critic_y + 0.7), color=COLORS['critic'], linestyle='-')
draw_arrow((4.5, actor_critic_y), (4.5, actor_critic_y - 0.3), color=COLORS['critic'], linestyle='-')
draw_arrow((4.5, actor_critic_y - 1.0), (4.5, actor_critic_y - 1.3), color=COLORS['critic'], linestyle='-')

# Add attention mechanism detail
plt.text(6.5, details_y + 3, "Attention Mechanism Detail", fontsize=12, weight='bold')
ax.add_patch(Rectangle((5.5, details_y + 1.5), 2, 1.5, fill=False, linestyle='--', edgecolor='gray'))
plt.text(6.5, details_y + 2.7, "MultiHead(Q, K, V) = Concat(head₁, ..., head₄)Wᵒ", fontsize=10)
plt.text(6.5, details_y + 2.3, "headᵢ = Softmax(QWᵢQ · KWᵢKᵀ/√dₖ) · VWᵢV", fontsize=10)
plt.text(6.5, details_y + 1.9, "dₖ = 256 (key dimension)", fontsize=10)

# Add residual connection detail
ax.add_patch(Rectangle((5.5, details_y - 0.5), 2, 1.5, fill=False, linestyle='--', edgecolor='gray'))
plt.text(6.5, details_y + 1.1, "Residual Encoding", fontsize=12, weight='bold')
draw_arrow((5.8, details_y + 0.8), (6.2, details_y + 0.8), color='k')
plt.text(6.0, details_y + 0.9, "+", fontsize=14, ha='center', va='center')
plt.text(5.7, details_y + 0.7, "Input", fontsize=9)
plt.text(6.0, details_y + 0.5, "FC(256)\nELU", fontsize=9, ha='center')
plt.text(6.3, details_y + 0.7, "Output", fontsize=9)
plt.text(6.0, details_y + 0.2, "FC(256)\n(Residual)", fontsize=9, ha='center')

# Add legend (moved to top to avoid bottom margin issues)
legend_elements = [
    plt.Line2D([0], [0], color=COLORS['input'], lw=8, label='Input Layer'),
    plt.Line2D([0], [0], color=COLORS['lstm'], lw=8, label='Temporal Encoder (GRU)'),
    plt.Line2D([0], [0], color=COLORS['attention'], lw=8, label='Attention Mechanism'),
    plt.Line2D([0], [0], color=COLORS['residual'], lw=8, label='Residual Encoding'),
    plt.Line2D([0], [0], color=COLORS['fusion'], lw=8, label='Feature Fusion'),
    plt.Line2D([0], [0], color=COLORS['actor'], lw=8, label='Actor Network'),
    plt.Line2D([0], [0], color=COLORS['critic'], lw=8, label='Critic Network'),
    plt.Line2D([0], [0], color=COLORS['output'], lw=8, label='Output Layer'),
]
ax.legend(handles=legend_elements, loc='upper center', 
          ncol=4, bbox_to_anchor=(0.5, 1.05), fontsize=10)

# Adjust layout to prevent clipping
plt.tight_layout(pad=3.0)
plt.subplots_adjust(top=0.92, bottom=0.05)

# Save figure with adequate padding
plt.savefig('network_architecture.png', bbox_inches='tight', pad_inches=0.5, dpi=300)
plt.show()