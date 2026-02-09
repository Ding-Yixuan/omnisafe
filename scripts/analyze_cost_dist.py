import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_dataset(npz_path, force_horizon=1000):
    if not os.path.exists(npz_path):
        print(f"❌ 找不到文件: {npz_path}")
        return None

    print(f"\n📊 正在分析: {npz_path}")
    try:
        data = np.load(npz_path)
    except Exception as e:
        print(f"❌ 无法加载文件: {e}")
        return None
    
    # 1. 获取 Cost
    if 'cost' in data:
        costs = data['cost']
    elif 'costs' in data:
        costs = data['costs']
    else:
        print(f"❌ 找不到 cost 数据！Keys: {list(data.keys())}")
        return None

    costs = costs.squeeze()
    total_steps = len(costs)
    
    # 2. 智能切分轨迹
    # 优先尝试寻找 done 信号
    dones = None
    if 'terminals' in data: dones = data['terminals']
    elif 'dones' in data: dones = data['dones']
    elif 'done' in data: dones = data['done']
    
    if dones is not None:
        dones = dones.squeeze()
        # 检查是否全为 False (即无效的 done 信号)
        if not np.any(dones):
            print("⚠️ 检测到 done 信号全为 False，可能是采集时未记录 Reset。")
            dones = None # 废弃无效的 done
            
    # 3. 如果没有有效的 done，或者是 path_lengths 也没用，就强制切分
    traj_costs = []
    
    if dones is None or not np.any(dones):
        print(f"💡 启用强制切分模式: 每 {force_horizon} 步算一条轨迹")
        # 强制切分
        num_segments = total_steps // force_horizon
        for i in range(num_segments):
            segment = costs[i*force_horizon : (i+1)*force_horizon]
            traj_costs.append(np.sum(segment))
            
        # 处理剩余的尾巴 (如果有)
        remainder = total_steps % force_horizon
        if remainder > 0:
            print(f"⚠️ 丢弃末尾剩余的 {remainder} 步数据")
            
    else:
        # 正常切分逻辑
        current_episode_cost = 0
        for i in range(total_steps):
            current_episode_cost += costs[i]
            if dones[i] or i == total_steps - 1:
                traj_costs.append(current_episode_cost)
                current_episode_cost = 0

    traj_costs = np.array(traj_costs)
    
    # 4. 统计
    num_trajs = len(traj_costs)
    if num_trajs == 0: return None

    safe_trajs = np.sum(traj_costs == 0)
    unsafe_trajs = num_trajs - safe_trajs
    total_cost = np.sum(traj_costs)
    
    print("-" * 40)
    print(f"🔹 轨迹总数: {num_trajs}")
    print(f"✅ 安全轨迹 (Cost=0): {safe_trajs} ({safe_trajs/num_trajs*100:.1f}%)")
    print(f"❌ 碰撞轨迹 (Cost>0): {unsafe_trajs} ({unsafe_trajs/num_trajs*100:.1f}%)")
    print(f"💥 总 Cost 数: {total_cost:.0f}")
    print(f"📉 平均 Cost: {total_cost/num_trajs:.2f} / traj")
    print("-" * 40)
    
    if unsafe_trajs > 0:
        print("💀 最危险的 5 条轨迹:")
        sorted_indices = np.argsort(traj_costs)[::-1]
        for k in range(min(5, unsafe_trajs)):
            idx = sorted_indices[k]
            print(f"   Traj #{idx:<4} | Cost: {traj_costs[idx]:.0f}")
            
    return traj_costs

def main():
    # 填入你的路径
    file1 = "./runs/PPOLag-{SafetyPointGoal1-v0}/seed-000-2026-02-07-19-09-38/safety_gym_26dim_data.npz"
    file2 = "./runs/PPOLag-{SafetyPointGoal1-v0}/seed-000-2026-02-07-16-45-27/safety_gym_26dim_data.npz"
    
    print("="*60)
    analyze_dataset(file1)
    analyze_dataset(file2)

if __name__ == "__main__":
    main()