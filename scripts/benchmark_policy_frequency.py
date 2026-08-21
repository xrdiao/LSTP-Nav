import time
from pathlib import Path
import sys
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.model_raw import *  # 确保导入所有需要的Agent

def benchmark_policy_frequency(policy, device, test_duration=3.0):
    count = 0
    with torch.no_grad():
        for _ in range(10):
            state = torch.randn([1, 4], device=device)
            laser = torch.randn([1, 5, 130], device=device)
            policy.get_action_and_value(laser, state)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    while (time.perf_counter() - start_time) < test_duration:
        with torch.no_grad():
            state = torch.randn([1, 4], device=device)
            laser = torch.randn([1, 5, 130], device=device)
            policy.get_action_and_value(laser, state)
        count += 1

    if device.type == "cuda":
        torch.cuda.synchronize()
    duration = time.perf_counter() - start_time
    return count / duration


def main():
    agents = [AttentionAgent, LstmAgent, IJRRAgent]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"测试设备: {device}")

    for agent_cls in agents:
        policy = agent_cls().to(device)
        policy.eval()

        fps = benchmark_policy_frequency(policy, device=device, test_duration=3.0)
        status = "✓ 满足" if fps >= 300 else "✗ 不足"
        print(f"{agent_cls.__name__:15} | 频率: {fps:7.1f} 次/秒 | {status} 300FPS要求")


if __name__ == "__main__":
    main()
