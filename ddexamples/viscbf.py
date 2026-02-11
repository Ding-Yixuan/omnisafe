import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from train_cbf import CBFNetwork  # 👈 引用你的 CBF 定义

# =================================================================
# 1. 简易虚拟雷达 (修正版：与 SafetyGymnasium 物理引擎对齐)
# =================================================================
def get_virtual_lidar(agent_pos, obstacles, num_bins=16, max_dist=3.0):
    """
    计算从 agent_pos 发出的雷达射线，撞到 obstacles 的距离
    obstacles: list of [x, y]
    """
    # 默认是 0 (Safety Gym 中 exp(-inf) = 0)，代表周围没东西
    # 注意：Safety Gym 的 lidar 默认输出是 exp(-dist)，所以空旷处是 0
    lidar = np.zeros(num_bins) 
    
    angles = np.linspace(0, 2*np.pi, num_bins, endpoint=False)
    
    agent_radius = 0.1
    hazard_radius = 0.2
    
    for i, angle in enumerate(angles):
        # 射线方向向量
        ray_dir = np.array([np.cos(angle), np.sin(angle)])
        
        closest_dist = float('inf') # 记录这条射线上最近的障碍物距离
        
        for obs in obstacles:
            obs = np.array(obs)
            rel_pos = obs - agent_pos
            
            # 投影长度 (圆心在射线上的投影)
            proj = np.dot(rel_pos, ray_dir)
            
            if proj > 0: # 障碍物在前方
                # 垂距
                dist_to_ray = np.linalg.norm(rel_pos - proj * ray_dir)
                
                # 如果射线穿过障碍物圆柱
                if dist_to_ray < hazard_radius:
                    # 计算表面距离 (勾股定理)
                    # d = 投影长 - 半弦长 - agent半径
                    half_chord = np.sqrt(hazard_radius**2 - dist_to_ray**2)
                    d = proj - half_chord - agent_radius
                    
                    if d < closest_dist:
                        closest_dist = d
        
        # 🔥【关键修正】与训练数据的物理意义对齐
        # 训练数据用的 patched_obs 里调用了 self._obs_lidar
        # SafetyGymnasium 的 _obs_lidar 默认逻辑是：exp(-distance)
        if closest_dist < max_dist:
            # 距离越近，值越大 (接近 1)
            # 距离越远，值越小 (接近 0)
            lidar[i] = np.exp(-closest_dist)
        else:
            lidar[i] = 0.0 # 超出射程或无障碍
            
    return lidar
# =================================================================
# 2. 可视化主程序
# =================================================================
def visualize_landscape():
    device = 'cuda:0'
    
    # --- 配置 ---
    CBF_PATH = './看cbf数据/ppolag_测试data/best_cbf_model.pt'
    NORM_PATH = './看cbf数据/ppolag_测试data/cbf_normalization.npz'
    
    # 自定义障碍物位置 (上帝视角)
    OBSTACLES = [[-0.5, 0.5], [0.5, -0.5]] 
    GOAL = [1.0, 1.0]
    
    # --- 加载模型 ---
    model = CBFNetwork(obs_dim=26).to(device)
    model.load_state_dict(torch.load(CBF_PATH, map_location=device))
    model.eval()
    
    # --- 加载归一化参数 ---
    norm_data = np.load(NORM_PATH)
    mins = torch.from_numpy(norm_data['mins']).float().to(device)
    maxs = torch.from_numpy(norm_data['maxs']).float().to(device)

    # --- 生成网格 ---
    # x_range = np.linspace(-1.5, 1.5, 10)
    # y_range = np.linspace(-1.5, 1.5, 10)
    # X, Y = np.meshgrid(x_range, y_range)
    resolution = 100

    x_range = np.linspace(-1.5, 1.5, resolution) 
    y_range = np.linspace(-1.5, 1.5, resolution)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X) # 存放 CBF 值

    print("🚀 计算全图 CBF 值...")
    
    for i in range(resolution):
        for j in range(resolution):
            agent_pos = np.array([X[i, j], Y[i, j]])
            
            # 1. 构造 Lidar (16)
            lidar = get_virtual_lidar(agent_pos, OBSTACLES)
            
            # 2. 构造 Goal (3)
            # 简化的相对坐标计算
            vec = GOAL - agent_pos
            cx, cy = vec[0], vec[1]
            z_complex = cx + 1j * cy
            dist = np.exp(-np.abs(z_complex)) # 保持 exp 距离
            angle = np.angle(z_complex)
            goal_vec = np.array([dist, np.cos(angle), np.sin(angle)])
            
            # 3. 构造 Sensor (7)
            # 假设静止状态，速度加速度为0
            sensor_vec = np.zeros(7) 
            
            # 4. 拼装
            obs = np.concatenate([sensor_vec, goal_vec, lidar])
            
            # 5. 归一化 & 预测
            obs_tensor = torch.from_numpy(obs).float().to(device)
            # Normalize
            obs_norm = (obs_tensor - mins) / (maxs - mins)
            obs_norm = 2 * obs_norm - 1
            obs_norm = torch.clamp(obs_norm, -1.0, 1.0)
            
            with torch.no_grad():
                cbf_val = model(obs_norm.unsqueeze(0)).item()
                Z[i, j] = cbf_val

    # --- 画图 ---
    plt.figure(figsize=(10, 8))
    
    # 热力图
    # cmap: RdBu (红=负/危险, 蓝=正/安全)
    plt.contourf(X, Y, Z, levels=50, cmap='RdBu', alpha=0.8)
    plt.colorbar(label='CBF Value (Safety)')
    
    # 安全边界 (CBF = 0)
    plt.contour(X, Y, Z, levels=[0], colors='black', linewidths=2, linestyles='--')
    
    # 画障碍物
    for obs in OBSTACLES:
        circle = plt.Circle(obs, 0.18, color='black', alpha=0.5)
        plt.gca().add_patch(circle)
        plt.text(obs[0], obs[1], 'OBS', ha='center', color='white')
        
    # 画终点
    plt.scatter(GOAL[0], GOAL[1], marker='*', s=200, c='yellow', edgecolors='black', label='Goal')
    
    plt.title("CBF Safety Landscape (Red=Unsafe, Blue=Safe)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('cbf_landscape.png')
    print("✅ 可视化已保存至 cbf_landscape.png")
    plt.show()

if __name__ == '__main__':
    visualize_landscape()