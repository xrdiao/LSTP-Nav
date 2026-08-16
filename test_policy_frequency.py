import time
import torch
from rl.model_raw import *  # 确保导入所有需要的Agent

def test_policy_frequency(policy, test_duration=3.0):
    """改进版性能测试函数"""
    count = 0
    # 预热：避免初始化的时间影响测试
    with torch.no_grad():
        for _ in range(10):
            state = torch.randn([1, 4], device=device)
            laser = torch.randn([1, 5, 130], device=device)
            policy.get_action_and_value(laser, state)
    
    # 正式测试
    torch.cuda.synchronize()  # 确保CUDA操作同步（针对GPU）
    start_time = time.perf_counter()  # 更高精度计时器
    
    while (time.perf_counter() - start_time) < test_duration:
        with torch.no_grad():  # 禁用梯度计算
            state = torch.randn([1, 4], device=device)
            laser = torch.randn([1, 5, 130], device=device)
            policy.get_action_and_value(laser, state)
        count += 1
    
    torch.cuda.synchronize()
    duration = time.perf_counter() - start_time
    return count / duration

if __name__ == "__main__":
    agents = [AttentionAgent, LstmAgent, IJRRAgent]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"测试设备: {device}")
    
    for idx, Agent in enumerate(agents):
        policy = Agent().to(device)
        policy.eval()  # 设置为评估模式
        
        fps = test_policy_frequency(policy, test_duration=3.0)
        status = "✓ 满足" if fps >= 300 else "✗ 不足"
        print(f"{Agent.__name__:15} | 频率: {fps:7.1f} 次/秒 | {status} 300FPS要求")