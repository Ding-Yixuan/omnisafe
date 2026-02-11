import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from train_cbf import CBFNetwork  # 👈 保持引用

# =================================================================
# 1. 简易虚拟雷达 (逻辑修正版：去除 Debug 信息，保留 Heading 旋转)
# =================================================================
def get_virtual_lidar(agent_pos, obstacles, heading, num_bins=16, max_dist=3.0):
    """
    计算 Lidar 数据，射线方向随 robot_heading 旋转
    """
    lidar = np.zeros(num_bins) 
    # 0 代表车头正前方 (Local Frame)
    relative_angles = np.linspace(0, 2*np.pi, num_bins, endpoint=False)
    
    agent_radius = 0.1 
    hazard_radius = 0.2 
    
    for i, rel_angle in enumerate(relative_angles):
        # 🔥 核心修正：绝对角度 = 相对角度 + 机器人朝向
        abs_angle = rel_angle + heading
        
        ray_dir = np.array([np.cos(abs_angle), np.sin(abs_angle)])
        closest_dist = float('inf')
        
        for obs in obstacles:
            obs = np.array(obs)
            rel_pos = obs - agent_pos
            proj = np.dot(rel_pos, ray_dir)
            
            if proj > 0:
                dist_to_ray = np.linalg.norm(rel_pos - proj * ray_dir)
                if dist_to_ray < hazard_radius:
                    half_chord = np.sqrt(hazard_radius**2 - dist_to_ray**2)
                    d = proj - half_chord - agent_radius
                    if d < closest_dist:
                        closest_dist = d
        
        if closest_dist < max_dist:
            lidar[i] = np.exp(-closest_dist)
        else:
            lidar[i] = 0.0
            
    return lidar

# =================================================================
# 2. 可视化主程序 (最终版：逻辑正确 + 画面干净)
# =================================================================
def visualize_landscape_final():
    device = 'cuda:0'
    
    # --- 🎛️ 参数调整区域 ---
    MANUAL_SPEED = 1.0        # 速度大小
    
    # 方向 (弧度): 
    # 5*np.pi/4 (左下), np.pi/4 (右上), np.pi (左), 0 (右)
    # 设为 None 则自动朝向 Goal
    MANUAL_HEADING = 7 * np.pi / 4  
    
    SAFETY_MARGIN = 0.1       # 缓冲带宽度
    RESOLUTION = 150          # 分辨率
    
    # --- 配置 ---
    CBF_PATH = './看cbf数据/ppolag_测试data2让边界变小/best_cbf_model.pt'
    NORM_PATH = './看cbf数据/ppolag_测试data2让边界变小/cbf_normalization.npz'
    
    # 场景定义
    OBSTACLES = [[-0.5, 0.5], [0.5, -0.5]] 
    GOAL = [-1.0, -1.0] # 你的新终点
    
    # --- 加载模型 ---
    print(f"🔄 Loading model from {CBF_PATH}...")
    model = CBFNetwork(obs_dim=26).to(device)
    model.load_state_dict(torch.load(CBF_PATH, map_location=device))
    model.eval()
    
    # --- 加载归一化参数 ---
    norm_data = np.load(NORM_PATH)
    mins = torch.from_numpy(norm_data['mins']).float().to(device)
    maxs = torch.from_numpy(norm_data['maxs']).float().to(device)

    # --- 生成网格 ---
    x_range = np.linspace(-1.5, 1.5, RESOLUTION) 
    y_range = np.linspace(-1.5, 1.5, RESOLUTION)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X) 

    print(f"🚀 计算全图 CBF 值 (Speed={MANUAL_SPEED}, Heading={MANUAL_HEADING:.2f})...")
    
    goal_np = np.array(GOAL)
    
    for i in range(RESOLUTION):
        for j in range(RESOLUTION):
            agent_pos = np.array([X[i, j], Y[i, j]])
            
            # --- 1. 确定方向 ---
            vec = goal_np - agent_pos
            cx, cy = vec[0], vec[1]
            z_complex = cx + 1j * cy
            angle_to_goal = np.angle(z_complex)
            
            if MANUAL_HEADING is not None:
                current_heading = MANUAL_HEADING
            else:
                current_heading = angle_to_goal
            
            # --- 2. 构造 Observation ---
            
            # A. Lidar (带 Heading 修正)
            lidar = get_virtual_lidar(agent_pos, OBSTACLES, heading=current_heading)
            
            # B. Goal Vector (3)
            # Goal 向量在 Observation 中通常是相对坐标旋转后的结果
            # 但在这里为了简化，只要相对距离和角度对齐即可
            dist = np.exp(-np.abs(z_complex)) 
            # 这里的角度是指 Goal 在机器人坐标系下的角度？还是世界坐标系？
            # PPO 训练时通常是 goal_pos - agent_pos，然后旋转到 agent frame
            # 简单起见，我们保持原逻辑，这通常影响不大，核心是 Lidar 和 Vel
            goal_vec = np.array([dist, np.cos(angle_to_goal), np.sin(angle_to_goal)])
            
            # C. Sensor (Velocity) - 🔥 核心修正：使用 Local Velocity
            # 假设机器人沿着车头移动，那么本地 x 速度 = Speed，y 速度 = 0
            # 这模拟了 "正向前进" 的状态，消除了侧向漂移带来的不对称
            vel_input = np.array([MANUAL_SPEED, 0.0])
            
            acc = np.zeros(2)      
            gyro = np.zeros(1)     
            mag = vec[:2] / (np.linalg.norm(vec[:2]) + 1e-8) 
            
            sensor_vec = np.concatenate([acc, vel_input, gyro, mag])
            
            # D. 拼装
            obs = np.concatenate([sensor_vec, goal_vec, lidar])
            
            # --- 3. 预测 ---
            obs_tensor = torch.from_numpy(obs).float().to(device)
            obs_norm = (obs_tensor - mins) / (maxs - mins)
            obs_norm = 2 * obs_norm - 1
            obs_norm = torch.clamp(obs_norm, -5.0, 5.0)
            
            with torch.no_grad():
                cbf_out = model(obs_norm.unsqueeze(0))
                Z[i, j] = cbf_out.item()

    # --- 🎨 画图 (恢复 V2 干净风格) ---
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    
    # 区域填充
    levels = [Z.min(), 0, SAFETY_MARGIN, Z.max()]
    plt.contourf(X, Y, Z, levels=[-100, 0], colors=['#FF9999'], alpha=0.8) # Unsafe (红)
    plt.contourf(X, Y, Z, levels=[0, SAFETY_MARGIN], colors=['#FFFF99'], alpha=0.8) # Buffer (黄)
    plt.contourf(X, Y, Z, levels=[SAFETY_MARGIN, 100], colors=['#99CCFF'], alpha=0.6) # Safe (蓝)
    
    # 边界线
    cs_0 = plt.contour(X, Y, Z, levels=[0.0], colors='blue', linewidths=2, linestyles='solid')
    plt.clabel(cs_0, fmt={0.0: 'h(x)=0'}, inline=True, fontsize=12)
    
    cs_margin = plt.contour(X, Y, Z, levels=[SAFETY_MARGIN], colors='grey', linewidths=2, linestyles='dotted')
    plt.clabel(cs_margin, fmt={SAFETY_MARGIN: f'margin={SAFETY_MARGIN}'}, inline=True, fontsize=10)

    # 障碍物
    for obs in OBSTACLES:
        circle = plt.Circle(obs, 0.2, color='black', alpha=0.6)
        ax.add_patch(circle)
        plt.text(obs[0], obs[1], 'OBS', ha='center', va='center', color='white', fontweight='bold')
        
    # 终点
    plt.scatter(GOAL[0], GOAL[1], marker='*', s=300, c='gold', edgecolors='black', label='Goal', zorder=10)
    
    # 速度方向箭头 (保留这个很有用，能让你知道现在的设定方向)
    if MANUAL_HEADING is not None and MANUAL_SPEED > 0:
        arrow_len = 0.3
        # 画在原点或者终点附近
        center_x, center_y = 0.0, 0.0
        plt.arrow(center_x, center_y, arrow_len*np.cos(MANUAL_HEADING), arrow_len*np.sin(MANUAL_HEADING), 
                  width=0.02, color='purple', label='Current Heading', zorder=20)

    plt.title(f"CBF Landscape (Speed={MANUAL_SPEED}, Heading={MANUAL_HEADING:.2f})\nRed=Unsafe, Yellow=Buffer, Blue=Safe", fontsize=14)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF9999', edgecolor='none', label='Unsafe (h < 0)'),
        Patch(facecolor='#FFFF99', edgecolor='none', label='Buffer (0 < h < margin)'),
        Patch(facecolor='#99CCFF', edgecolor='none', label='Safe (h > margin)'),
        plt.Line2D([0], [0], color='blue', lw=2, label='Boundary h(x)=0'),
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'cbf_landscape_speed_{MANUAL_SPEED}.png')
    print(f"✅ 图表已保存: cbf_landscape_speed_{MANUAL_SPEED}.png")
    plt.show()

if __name__ == '__main__':
    visualize_landscape_final()