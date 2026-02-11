import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import gymnasium
import safety_gymnasium
from safety_gymnasium.assets.geoms import Hazards
from safety_gymnasium.tasks.safe_navigation.goal.goal_level1 import GoalLevel1
from safety_gymnasium.tasks.safe_navigation.goal.goal_level0 import GoalLevel0

# =================================================================
# 1. 简易 Patch (只为了初始化环境拿坐标，不需要 patch obs)
# =================================================================
def patched_init(self, config):
    self.lidar_num_bins = 16
    self.lidar_max_dist = 3.0
    self.sensors_obs = ['accelerometer', 'velocimeter', 'gyro', 'magnetometer']
    self.task_name = 'GoalLevel1_Reproduction'
    config.update({'lidar_num_bins': 16, 'lidar_max_dist': 3.0, 
                   'sensors_obs': self.sensors_obs, 'task_name': self.task_name})
    GoalLevel0.__init__(self, config=config)
    self.placements_conf.extents = [-1.5, -1.5, 1.5, 1.5]
    self._add_geoms(Hazards(num=2, keepout=0.18))

GoalLevel1.__init__ = patched_init

# =================================================================
# 2. CBF 网络定义
# =================================================================
class CBFNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x): return self.net(x)

# =================================================================
# 3. 核心工具：纯数学构造 Observation (God Mode)
# =================================================================
def synthesize_obs(x, y, goal_pos, hazards_pos, lidar_num_bins=16, max_dist=3.0):
    """
    不依赖 MuJoCo，直接根据几何关系计算 Observation (26维)
    """
    # --- A. Sensors (7维) ---
    # 假设静态绘图，速度加速度均为0，朝向(Heading)固定为 0 (正东)
    # [acc_x, acc_y, vel_x, vel_y, gyro_z, mag_x, mag_y]
    # mag 在 heading=0 时通常指向 (1, 0) 或者根据环境北极。这里设为默认值。
    sensor_vec = np.zeros(7, dtype=np.float32)
    sensor_vec[5] = 1.0 # mag_x
    
    # --- B. Goal (3维) ---
    # 向量计算
    dx = goal_pos[0] - x
    dy = goal_pos[1] - y
    # 因为假设 robot heading=0，所以旋转矩阵是单位矩阵，直接用 dx, dy
    # 复数变换 (论文同款)
    z = dx + 1j * dy
    dist = np.abs(z)
    dist_enc = np.exp(-dist) 
    angle = np.angle(z)
    goal_vec = np.array([dist_enc, np.cos(angle), np.sin(angle)], dtype=np.float32)
    
    # --- C. Lidar (16维) ---
    # 模拟 Safety Gymnasium 的 Lidar 逻辑
    lidar_vec = np.zeros(lidar_num_bins, dtype=np.float32)
    bin_size = 2 * np.pi / lidar_num_bins
    
    for hz_pos in hazards_pos:
        # 相对位置
        hz_dx = hz_pos[0] - x
        hz_dy = hz_pos[1] - y
        hz_dist = np.sqrt(hz_dx**2 + hz_dy**2)
        
        # 如果超出最大探测距离，忽略
        if hz_dist > max_dist:
            continue
            
        # 计算角度 (相对于机器人朝向 0)
        hz_angle = np.arctan2(hz_dy, hz_dx) 
        # 归一化到 [0, 2pi]
        hz_angle = hz_angle % (2 * np.pi)
        
        # 确定分箱
        bin_idx = int(hz_angle / bin_size) % lidar_num_bins
        
        # 计算强度 exp(-dist)
        intensity = np.exp(-hz_dist)
        
        # Safety Gym 逻辑：取该 bin 中最大的强度 (最近的障碍物)
        if intensity > lidar_vec[bin_idx]:
            lidar_vec[bin_idx] = intensity
            
        # *可选优化*：为了防止 aliasing，可以将强度分散到相邻 bin，
        # 但 Point 机器人的标准 Lidar 通常是 binary binning。
            
    # --- D. 拼接 ---
    return np.concatenate([sensor_vec, goal_vec, lidar_vec])

# =================================================================
# 4. 绘图主程序
# =================================================================
def plot_god_mode():
    # --- 配置 ---
    model_path = './cbf_checkpoints/cbf_v1/best_cbf_model.pt'
    norm_path = './cbf_checkpoints/cbf_v1/cbf_normalization.npz'
    save_path = './cbf_checkpoints/cbf_v1/final_cbf_map.png'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # 1. 加载参数
    print("📉 Loading Stats...")
    norm_data = np.load(norm_path)
    mins = torch.from_numpy(norm_data['mins']).float().to(device)
    maxs = torch.from_numpy(norm_data['maxs']).float().to(device)

    # 2. 加载模型
    print("🧠 Loading Model...")
    cbf_net = CBFNetwork(26).to(device)
    try:
        cbf_net.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    except TypeError:
        cbf_net.load_state_dict(torch.load(model_path, map_location=device))
    cbf_net.eval()

    # 3. 初始化环境 (仅用于提取 Goal 和 Hazard 的真实坐标)
    print("🌍 Reading Map Config...")
    env = gymnasium.make('SafetyPointGoal1-v0')
    env.reset()
    
    hazards_pos = env.task.hazards.pos.copy()
    goal_pos = env.task.goal.pos.copy()
    
    print(f"📍 Hazards True Pos: \n{hazards_pos}")
    print(f"📍 Goal True Pos: {goal_pos}")

    # 4. 扫描网格 (纯数学计算，速度极快)
    res = 200 
    x = np.linspace(-1.5, 1.5, res)
    y = np.linspace(-1.5, 1.5, res)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    print("📸 Scanning (God Mode: Mathematical Synthesis)...")
    
    # 批量处理或逐点处理，这里为了清晰逐点处理
    obs_batch = []
    indices = []
    
    for i in range(res):
        for j in range(res):
            obs = synthesize_obs(X[i, j], Y[i, j], goal_pos, hazards_pos)
            obs_batch.append(obs)
            indices.append((i, j))
            
    # 转为 Tensor 批量预测 (提速)
    obs_tensor = torch.tensor(np.array(obs_batch), dtype=torch.float32).to(device)
    
    # 归一化
    obs_norm = (obs_tensor - mins) / (maxs - mins)
    obs_norm = 2 * obs_norm - 1
    obs_norm = torch.clamp(obs_norm, -1.0, 1.0)
    
    with torch.no_grad():
        preds = cbf_net(obs_norm).cpu().numpy().flatten()
        
    # 填回 Z
    for k, (i, j) in enumerate(indices):
        Z[i, j] = preds[k]

    # 5. 绘图
    print(f"📊 Stats: Min={Z.min():.4f}, Max={Z.max():.4f}")
    
    fig, ax = plt.subplots(figsize=(9, 9))
    
    # 绘制背景
    # 使用 RdYlGn (红-黄-绿)，以 0 为中心
    # 使用 TwoSlopeNorm 来确保 0 对应白色或黄色，负数红，正数绿
    import matplotlib.colors as mcolors
    divnorm = mcolors.TwoSlopeNorm(vmin=Z.min(), vcenter=0., vmax=Z.max())
    
    im = ax.imshow(Z, extent=[-1.5, 1.5, -1.5, 1.5], origin='lower', 
                   cmap='RdYlGn', norm=divnorm, alpha=0.6)
    plt.colorbar(im, label='CBF Value h(x)')

    # 绘制边界线
    width = 0.1
    # 绘制 h(x)=0 (决策边界)
    ax.contour(X, Y, Z, levels=[0], colors='blue', linewidths=2.5, linestyles='solid')
    # 绘制 h(x)=+/-0.1 (缓冲区)
    ax.contour(X, Y, Z, levels=[-width, width], colors='grey', linewidths=1.5, linestyles='dashed')

    # 绘制真实障碍物
    for hz in hazards_pos:
        # 物理体积 (Keepout 区域)
        circle = plt.Circle((hz[0], hz[1]), 0.18, color='red', alpha=0.5, label='Hazard')
        ax.add_patch(circle)
        # 轮廓
        circle_edge = plt.Circle((hz[0], hz[1]), 0.18, color='black', fill=False, linewidth=2)
        ax.add_patch(circle_edge)
    
    ax.plot(goal_pos[0], goal_pos[1], 'g*', markersize=18, markeredgecolor='k', label='Goal')

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title(f"CBF Safety Landscape\nBlue Line: Learned Boundary (h=0)")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ Success! Map saved to {save_path}")
    plt.show()

if __name__ == '__main__':
    plot_god_mode()