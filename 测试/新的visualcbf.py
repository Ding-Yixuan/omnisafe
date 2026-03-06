import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

# =================================================================
# 0. 避坑：直接粘贴网络定义，彻底解决 ModuleNotFoundError
# =================================================================
import torch.nn as nn
class CBFNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        return self.net(x)

# =================================================================
# 1. 简易虚拟雷达 (逻辑修正版：去除 Debug 信息，保留 Heading 旋转)
# =================================================================
def get_virtual_lidar(agent_pos, obstacles, heading, num_bins=16, max_dist=3.0):
    """
    计算 Lidar 数据，射线方向随 robot_heading 旋转
    """
    lidar = np.zeros(num_bins) 
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
# 2. 可视化主程序 (最终版：逻辑正确 + 画面干净 + 适配 Action-CBF)
# =================================================================
def visualize_landscape_final():
    device = 'cuda:0'
    
    # --- 🎛️ 参数调整区域 ---
    MANUAL_SPEED = 1.0        # 速度大小
    
    # 方向 (弧度): 5*np.pi/4 (左下), np.pi/4 (右上), np.pi (左), 0 (右)
    MANUAL_HEADING = 7 * np.pi / 4  
    
    SAFETY_MARGIN = 0.1       # 缓冲带宽度
    RESOLUTION = 150          # 分辨率
    
    # --- ⚠️ 核心配置：确保指向你最新的“混合数据”模型路径 ---
    # 如果你想看旧的，记得把这里的路径和下面的 act_dim 改回去
    CBF_PATH = './看cbf数据/混合数据集_ActionCBF/best_cbf_model.pt' 
    NORM_PATH = './看cbf数据/混合数据集_ActionCBF/cbf_normalization.npz'
    
    # 场景定义
    OBSTACLES = [[-0.5, 0.5], [0.5, -0.5]] 
    GOAL = [-1.0, -1.0] # 你的新终点
    
    # --- 加载模型 (🔥 适配了 28维输入) ---
    print(f"🔄 Loading model from {CBF_PATH}...")
    model = CBFNetwork(obs_dim=26, act_dim=2).to(device)
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
    
    # 🔥 定义当前网络假设正在执行的动作 (假设不转弯，只保持当前速度前进)
    action_vec = np.array([MANUAL_SPEED, 0.0])
    
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
            dist = np.exp(-np.abs(z_complex)) 
            goal_vec = np.array([dist, np.cos(angle_to_goal), np.sin(angle_to_goal)])
            
            # C. Sensor (Velocity) - 使用 Local Velocity
            vel_input = np.array([MANUAL_SPEED, 0.0])
            acc = np.zeros(2)      
            gyro = np.zeros(1)     
            mag = vec[:2] / (np.linalg.norm(vec[:2]) + 1e-8) 
            sensor_vec = np.concatenate([acc, vel_input, gyro, mag])
            
            # D. 🔥 拼装：拼接成 28 维特征 (26维 obs + 2维 act)
            inputs = np.concatenate([sensor_vec, goal_vec, lidar, action_vec])
            
            # --- 3. 预测 ---
            inputs_tensor = torch.from_numpy(inputs).float().to(device)
            inputs_norm = (inputs_tensor - mins) / (maxs - mins)
            inputs_norm = 2 * inputs_norm - 1
            inputs_norm = torch.clamp(inputs_norm, -5.0, 5.0)
            
            with torch.no_grad():
                cbf_out = model(inputs_norm.unsqueeze(0))
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
    
    # 速度方向箭头
    if MANUAL_HEADING is not None and MANUAL_SPEED > 0:
        arrow_len = 0.3
        center_x, center_y = 0.0, 0.0
        plt.arrow(center_x, center_y, arrow_len*np.cos(MANUAL_HEADING), arrow_len*np.sin(MANUAL_HEADING), 
                  width=0.02, color='purple', label='Current Heading', zorder=20)

    plt.title(f"Action-CBF Landscape (Speed={MANUAL_SPEED}, Heading={MANUAL_HEADING:.2f})\nRed=Unsafe, Yellow=Buffer, Blue=Safe", fontsize=14)
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