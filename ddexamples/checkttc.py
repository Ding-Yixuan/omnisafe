import numpy as np
import matplotlib.pyplot as plt

def observe_safety_correlation(file_path):
    # 1. 加载数据
    data = np.load(file_path)
    is_safe = data['is_safe']   # 基于 TTC < 1.0 的预警 (0为危险)
    env_cost = data['env_cost'] # 环境自带的碰撞判定 (通常 > 0 表示碰撞)
    
    total_steps = len(is_safe)
    
    # 2. 定义状态
    # 逻辑：is_safe=0 是预警，env_cost > 0 是真实碰撞
    alert_indices = (is_safe == 0)
    safe_indices = (is_safe == 1)
    collision_indices = (env_cost > 0)
    no_collision_indices = (env_cost == 0)

    # 3. 计算关键统计量
    # A. 提前预警成功 (预测危险且确实碰撞)
    true_positive = np.sum(alert_indices & collision_indices)
    
    # B. 虚警 (预测危险但没碰撞 - 这通常是“灵敏度”的体现)
    false_positive = np.sum(alert_indices & no_collision_indices)
    
    # C. 漏报 (预测安全但撞了 - 这种情况最危险)
    false_negative = np.sum(safe_indices & collision_indices)

    # 4. 打印分析报告
    print(f"📊 --- 安全相关性分析报告 ({file_path}) ---")
    print(f"总样本数: {total_steps}")
    print("-" * 40)
    print(f"🚩 TTC 预警次数 (is_safe=0): {np.sum(alert_indices)}")
    print(f"💥 真实碰撞次数 (env_cost>0): {np.sum(collision_indices)}")
    print("-" * 40)
    
    if np.sum(alert_indices) > 0:
        sync_rate = (true_positive / np.sum(alert_indices)) * 100
        print(f"✅ 同步命中: {true_positive} 次 (预警时确实发生了碰撞)")
        print(f"🚀 灵敏预警: {false_positive} 次 (预警了但环境还没判定碰撞 -> 实现了提前量)")
        print(f"同步率 (True Positive Rate): {sync_rate:.2f}%")
    
    if false_negative > 0:
        print(f"❌ 漏报危险: {false_negative} 次 (TTC 觉得安全但环境判定撞了！可能需要调大 TTC 阈值)")
    else:
        print(f"🛡️  完美覆盖: 0 次漏报 (所有的真实碰撞都被你的 TTC 提前预警到了)")

    # 5. 可视化观察
    plt.figure(figsize=(10, 6))
    
    # 绘制时间轴局部切片（比如前500步）查看信号重叠情况
    slice_idx = 500
    time_axis = np.arange(slice_idx)
    
    plt.plot(time_axis, 1 - is_safe[:slice_idx], label='TTC Alert (1=Active)', color='red', alpha=0.7, linewidth=2)
    plt.fill_between(time_axis, 0, env_cost[:slice_idx], color='orange', alpha=0.3, label='Real Environment Cost')
    
    plt.title(f"TTC Alert vs Real Cost (First {slice_idx} steps)")
    plt.xlabel("Step")
    plt.ylabel("Signal")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

if __name__ == "__main__":
    # 请确保路径正确
    observe_safety_correlation('./data_pro/ppolag_测试data.npz')