import os
from pathlib import Path
import sys
import torch
from thop import profile, clever_format

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.util_raw import *

from project_paths import MODEL_DIR

def main():
    agent_name = 'Agent_Lstm_Attn'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    agent = AttentionAgent().to(device)
    ckpt_path = MODEL_DIR / f"{agent_name}_circle.pth"
    agent.load_state_dict(torch.load(ckpt_path, map_location=device))
    agent.eval()

    total_params = sum(p.numel() for p in agent.parameters())
    trainable_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)

    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Total params (M): {total_params / 1e6:.4f} M")

    file_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    print(f"Model file size: {file_size_mb:.2f} MB")

    batch_size = 1
    time_steps = 5
    laser_num = 130
    state_dim = 4

    laser = torch.randn(batch_size, time_steps, laser_num).to(device)
    state = torch.randn(batch_size, state_dim).to(device)

    macs, params = profile(agent, inputs=(laser, state), verbose=False)
    macs_str, params_str = clever_format([macs, params], "%.3f")

    print(f"THOP Params: {params_str}")
    print(f"MACs: {macs_str}")

    flops = macs * 2
    flops_str = clever_format([flops], "%.3f")
    print(f"Approx. FLOPs: {flops_str}")


if __name__ == "__main__":
    main()
