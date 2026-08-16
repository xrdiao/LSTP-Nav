import os
import torch
from thop import profile, clever_format

from rl.util_raw import *

# =========================
# 1. Load model
# =========================
agent_name = 'Agent_Lstm_Attn'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

agent = AttentionAgent().to(device)
ckpt_path = 'model/' + agent_name + '_' + 'circle' + '.pth'
agent.load_state_dict(torch.load(ckpt_path, map_location=device))
agent.eval()

# =========================
# 2. Parameter count
# =========================
total_params = sum(p.numel() for p in agent.parameters())
trainable_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)

print(f"Total params: {total_params:,}")
print(f"Trainable params: {trainable_params:,}")
print(f"Total params (M): {total_params / 1e6:.4f} M")

# =========================
# 3. Model file size
# =========================
file_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
print(f"Model file size: {file_size_mb:.2f} MB")

# =========================
# 4. Dummy input
#    Adjust shapes if needed
# =========================
B = 1
T = 5
LASER_NUM = 130
STATE_DIM = 4

laser = torch.randn(B, T, LASER_NUM).to(device)
state = torch.randn(B, STATE_DIM).to(device)

# =========================
# 5. FLOPs / MACs with THOP
# =========================
macs, params = profile(agent, inputs=(laser, state), verbose=False)

macs_str, params_str = clever_format([macs, params], "%.3f")

print(f"THOP Params: {params_str}")
print(f"MACs: {macs_str}")

# Rough conversion: FLOPs ≈ 2 * MACs
flops = macs * 2
flops_str = clever_format([flops], "%.3f")
print(f"Approx. FLOPs: {flops_str}")