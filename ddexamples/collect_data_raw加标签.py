import torch
import torch.nn as nn
import numpy as np
import os
import safety_gymnasium
import gymnasium
from safety_gymnasium.assets.geoms import Hazards
from safety_gymnasium.tasks.safe_navigation.goal.goal_level1 import GoalLevel1
from safety_gymnasium.tasks.safe_navigation.goal.goal_level0 import GoalLevel0

# =================================================================
# 1. 环境定义 (保持不变)
# =================================================================
def patched_init(self, config):
    self.lidar_num_bins = 16
    self.lidar_max_dist = 3.0
    self.sensors_obs = ['accelerometer', 'velocimeter', 'gyro', 'magnetometer']
    self.task_name = 'GoalLevel1_Reproduction'
    config.update({'lidar_num_bins': 16, 'lidar_max_dist': 3.0, 'sensors_obs': self.sensors_obs, 'task_name': self.task_name})
    GoalLevel0.__init__(self, config=config)
    self.placements_conf.extents = [-1.5, -1.5, 1.5, 1.5]
    self._add_geoms(Hazards(num=2, keepout=0.2))

def patched_obs(self):
    # 原始物理观测
    lidar_vec = self._obs_lidar(self.hazards.pos, self.hazards.group)
    acc = self.agent.get_sensor('accelerometer')[:2]
    vel = self.agent.get_sensor('velocimeter')[:2]
    gyro = self.agent.get_sensor('gyro')[-1:]
    mag = self.agent.get_sensor('magnetometer')[:2]
    sensor_vec = np.concatenate([acc, vel, gyro, mag])
    vec = (self.goal.pos - self.agent.pos) @ self.agent.mat
    x, y = vec[0], vec[1]
    z = x + 1j * y
    dist = np.exp(-np.abs(z)) 
    angle = np.angle(z)
    goal_vec = np.array([dist, np.cos(angle), np.sin(angle)])
    return np.concatenate([sensor_vec, goal_vec, lidar_vec]).astype(np.float32)

GoalLevel1.__init__ = patched_init
GoalLevel1.obs = patched_obs

def calculate_ttc(env_task, agent_pos, agent_vel):
    """
    计算 TTC (Time-To-Collision)
    基于脚本 verify_geometry 确认的真实物理参数：
    - Agent (Point): 半径 0.10m
    - Hazard: 半径 0.20m
    """
    min_ttc = float('inf')
    
    # 1. 获取 Hazard 位置
    try:
        hazards_pos = env_task.hazards.pos
        # 【基于脚本实测】虽然 keepout=0.18，但物理半径实测为 0.20
        hazards_radius = 0.20 
    except:
        # 兼容旧代码结构
        if hasattr(env_task, '_geoms') and 'hazards' in env_task._geoms:
            hazards_pos = env_task._geoms['hazards'].pos
        else:
             # 最后的 fallback，防止报错
            hazards_pos = []
        hazards_radius = 0.20
    
    # 2. 【基于脚本实测】Robot 物理半径
    agent_radius = 0.10
    
    # 3. 接触阈值 (圆心距)
    # 0.10 + 0.20 = 0.30m
    collision_threshold = agent_radius + hazards_radius
    
    if len(hazards_pos) == 0:
        return float('inf')

    for h_pos in hazards_pos:
        # 取前两维 (x, y)
        h_pos_2d = h_pos[:2] 
        
        rel_pos = h_pos_2d - agent_pos
        dist_center = np.linalg.norm(rel_pos)
        
        # --- 核心：表面距离计算 ---
        dist_surface = dist_center - collision_threshold
        
        # 已经碰撞 (重叠)
        if dist_surface <= 0: 
            return 0.0 
        
        # 计算速度投影
        if dist_center > 1e-6: 
            direction = rel_pos / dist_center
        else: 
            direction = np.zeros(2)
            
        v_proj = np.dot(agent_vel, direction)
        
        # 只有在靠近 (v > 0) 时才计算 TTC
        # 设定一个极小的速度阈值，过滤静止抖动
        if v_proj > 1e-4: 
            ttc = dist_surface / v_proj
            if ttc < min_ttc:
                min_ttc = ttc
        
    return min_ttc

# =================================================================
# 2. 手动重建 PPO Agent (针对 'dict' 只有权重的情况)
# =================================================================
class PPO_Inference_Agent(nn.Module):
    def __init__(self, obs_dim=26, act_dim=2, hidden_sizes=[64, 64]):
        super().__init__()
        
        # 1. 观测归一化器 (Obs Normalizer)
        self.obs_mean = nn.Parameter(torch.zeros(obs_dim), requires_grad=False)
        self.obs_var = nn.Parameter(torch.ones(obs_dim), requires_grad=False)
        
        # 2. 策略网络 (Policy Network - Actor)
        # OmniSafe 默认结构: Linear -> Tanh -> Linear -> Tanh -> Linear
        layers = []
        sizes = [obs_dim] + hidden_sizes + [act_dim]
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:
                layers.append(nn.Tanh()) # OmniSafe 默认激活函数
        self.net = nn.Sequential(*layers)
        
        # OmniSafe 的 pi 可能包含 log_std，但在 eval 时只需要 mean
        
    def load_from_dict(self, ckpt):
        """ 从字典加载权重 """
        print("🔧 手动加载权重...")
        
        # A. 加载 Normalizer
        if 'obs_normalizer' in ckpt:
            norm_state = ckpt['obs_normalizer']
            # OmniSafe normalizer 通常存的是 'mean' 和 'var' 或 'running_mean'
            # 我们需要打印看看 key 长什么样，这里做泛化处理
            print(f"   找到 Obs Normalizer, Keys: {norm_state.keys()}")
            if 'mean' in norm_state:
                self.obs_mean.data = norm_state['mean'].cpu()
                self.obs_var.data = norm_state['var'].cpu()
            elif 'running_mean' in norm_state: # 兼容 torch.nn.BatchNorm 风格
                self.obs_mean.data = norm_state['running_mean'].cpu()
                self.obs_var.data = norm_state['running_var'].cpu()
            print("   ✅ Normalizer 参数加载完毕")
            
        # B. 加载 Actor (Pi)
        if 'pi' in ckpt:
            pi_state = ckpt['pi']
            # 尝试直接加载 state_dict
            try:
                # 过滤掉 log_std (如果网络结构里没有定义)
                # 通常 pi 的 key 是 'net.0.weight', 'net.0.bias' 等
                # 我们定义的 self.net 直接对应
                new_state_dict = {}
                for k, v in pi_state.items():
                    # OmniSafe 经常叫 'mean.net.0.weight' 或者直接 'net.0.weight'
                    if 'mean' in k or 'net' in k: 
                        # 简化 key，去掉前缀
                        clean_k = k.replace('mean.', '').replace('net.', '') 
                        # 这是一个非常简单的映射尝试，假设结构是 [0, 2, 4] (层索引)
                        # 如果你的 hidden_size 不是 [64, 64]，这里可能会报错
                        pass
                
                # 最稳妥的方法：不猜 Key，直接按顺序赋值权重 (Weight Surgery)
                print("   正在进行权重手术 (Weight Surgery)...")
                layer_idx = 0
                for name, param in pi_state.items():
                    # 只提取权重和偏置
                    if 'weight' in name and len(param.shape) == 2: # Linear Weight
                         if layer_idx < len(self.net):
                             while not isinstance(self.net[layer_idx], nn.Linear):
                                 layer_idx += 1
                             print(f"   Setting Layer {layer_idx} weight: {param.shape}")
                             self.net[layer_idx].weight.data = param.cpu()
                    elif 'bias' in name and len(param.shape) == 1: # Linear Bias
                         if layer_idx < len(self.net):
                             while not isinstance(self.net[layer_idx], nn.Linear):
                                 layer_idx += 1
                             print(f"   Setting Layer {layer_idx} bias: {param.shape}")
                             self.net[layer_idx].bias.data = param.cpu()
                             layer_idx += 1 # 只有 bias 设置完才算过了一层
                print("   ✅ Policy 权重加载完毕")
                
            except Exception as e:
                print(f"   ❌ 加载 Policy 失败: {e}")
                print("   ⚠️ 建议使用方案一：寻找 pyt_save/model.pt")
                exit()
        else:
            print("❌ 字典里没有 'pi' Key")
            exit()

    def step(self, raw_obs):
        """ 输入 Raw Obs -> Normalize -> Actor -> Action """
        # 1. Normalize
        # clip raw_obs (optional, omnisafe usually clips to [-5, 5] before norm? No, after.)
        # Formula: (x - mean) / sqrt(var + epsilon)
        obs_norm = (torch.tensor(raw_obs) - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8)
        
        # 2. Clip Obs (通常 OmniSafe 会把归一化后的值 clip 到 [-5, 5])
        obs_norm = torch.clamp(obs_norm, -5.0, 5.0)
        
        # 3. Forward
        action = self.net(obs_norm)
        return action.detach().numpy()

# =================================================================
# 3. 采集主程序
# =================================================================
def collect():
    # ================= 配置 =================
    AGENT_PATH = './runs/PPOLag-{SafetyPointGoal1-v0}/seed-000-2026-02-10-21-13-01/torch_save/epoch-500.pt'
    SAVE_PATH = './data_pro/ppolag_256.npz'
    MAX_STEPS = 50000
    TTC_THRESHOLD = 1.0  # 安全阈值
    
    # 1. 加载 Agent
    print(f"🔄 手动组装 Agent from {AGENT_PATH}...")
    ckpt = torch.load(AGENT_PATH, map_location='cpu')
    agent = PPO_Inference_Agent(obs_dim=26, act_dim=2, hidden_sizes=[256, 256])
    agent.load_from_dict(ckpt)
    
    # 2. 创建环境
    env = safety_gymnasium.make('SafetyPointGoal1-v0')
    
    # --- 初始化增强型 Buffer ---
    dataset = {
        'obs': [], 'act': [], 'next_obs': [], 'rew': [], 'env_cost': [], 
        'done': [], 'ttc': [], 'is_safe': [], 'goal_pos': [], 
        'agent_pos': [], 'segment_id': []
    }
    
    current_segment = 0
    total_steps = 0
    o, _ = env.reset()
    
    print("🚀 Start collecting ENHANCED data (Code 1 Framework + Code 2 Content)...")
    
    while total_steps < MAX_STEPS:
        # A. 获取当前 Raw Obs (代码 1 特有补丁)
        raw_obs_numpy = env.task.obs() 
        
        # B. 决策
        action = agent.step(raw_obs_numpy)

        # C. 执行环境步
        next_o, reward, cost, done, trunc, info = env.step(action)
        
        # --- D. 物理信息提取 (集成自代码 2) ---
        # 获取机器人和目标的实时物理位置
        agent_pos = env.task.agent.pos[:2].copy()
        agent_vel = env.task.agent.vel[:2].copy()
        goal_pos = env.task.goal.pos[:2].copy()
        
        # 计算 TTC (直接在循环内调用计算逻辑)
        ttc_val = calculate_ttc(env.task, agent_pos, agent_vel)
        is_safe = 1 if ttc_val > TTC_THRESHOLD else 0
        
        # E. 存储到 Dataset
        dataset['obs'].append(raw_obs_numpy)
        dataset['act'].append(action)
        dataset['next_obs'].append(next_o) # 环境标准的 next_obs
        dataset['rew'].append(reward)
        dataset['env_cost'].append(cost)
        dataset['done'].append(done or trunc)
        dataset['ttc'].append(ttc_val)
        dataset['is_safe'].append(is_safe)
        dataset['goal_pos'].append(goal_pos)
        dataset['agent_pos'].append(agent_pos)
        dataset['segment_id'].append(current_segment)
        
        total_steps += 1
        if total_steps % 1000 == 0:
            print(f"Collected {total_steps}/{MAX_STEPS} steps... TTC Mean: {np.mean(dataset['ttc'][-100:]):.2f}")

        if done or trunc:
            o, _ = env.reset()
            current_segment += 1
        else:
            o = next_o

    # F. 清洗并保存
    print(f"💾 Saving ENHANCED data to {SAVE_PATH}...")
    # 将列表转换为 Numpy 数组并压缩保存
    final_data = {k: np.array(v) for k, v in dataset.items()}
    np.savez_compressed(SAVE_PATH, **final_data)
    print("🎉 Done!")

# 注意：你需要把代码 2 中的 calculate_ttc 函数复制到代码 1 中，放在 collect 函数上方。
if __name__ == '__main__':
    collect()