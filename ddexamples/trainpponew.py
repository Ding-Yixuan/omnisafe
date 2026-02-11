import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import gymnasium
import safety_gymnasium
# 引入你的环境 Patch，用于 Evaluation
from safety_gymnasium.tasks.safe_navigation.goal.goal_level1 import GoalLevel1
from safety_gymnasium.tasks.safe_navigation.goal.goal_level0 import GoalLevel0
from safety_gymnasium.assets.geoms import Hazards

# =================================================================
# 1. 环境 Patch (必须与采集时一致)
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
    self._add_geoms(Hazards(num=2, keepout=0.2)) # 0.2

def patched_obs(self):
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

# =================================================================
# 2. 配置参数
# =================================================================
CONFIG = {
    'dataset_path': './data_pro/ppolag_xinde8ge.npz', # 👈 确保路径对
    'horizon': 64,
    'obs_dim': 26,
    'act_dim': 2,
    'hidden_dim': 256,
    'train_steps': 100000,
    'batch_size': 256,
    'lr': 2e-4,
    'device': 'cuda:0',
    'save_dir': './diffuser_checkpoints/best_auto_save', # 👈 新路径
    'eval_freq': 5000,      # 每 5000 步评估一次
    'eval_episodes': 10,    # 每次评估跑 10 个回合
}

# =================================================================
# 3. 数据集与网络 (保持不变)
# =================================================================
class TrajectoryDataset(Dataset):
    def __init__(self, data_path, horizon=64):
        print(f"📂 Loading data from {data_path}...")
        raw_data = np.load(data_path)
        self.obs = raw_data['obs'].astype(np.float32)
        self.act = raw_data['act'].astype(np.float32)
        # 兼容 segment_id 或 episode_done
        if 'segment_id' in raw_data:
            self.segment_ids = raw_data['segment_id']
        else:
            # 简单的备选方案：如果没有 segment_id，假设只有一条长轨迹 (不推荐)
            self.segment_ids = np.zeros(len(self.obs))
            
        self.mins = np.concatenate([self.obs.min(axis=0), self.act.min(axis=0)])
        self.maxs = np.concatenate([self.obs.max(axis=0), self.act.max(axis=0)])
        self.maxs[self.maxs == self.mins] += 1.0 # 防除零

        self.indices = []
        total_steps = len(self.obs)
        print("✂️  Slicing trajectories...")
        for i in range(total_steps - horizon + 1):
            if self.segment_ids[i] == self.segment_ids[i + horizon - 1]:
                self.indices.append(i)
        print(f"✅ Created {len(self.indices)} sequences.")
        
    def normalize(self, x):
        x_norm = (x - self.mins) / (self.maxs - self.mins)
        return 2 * x_norm - 1

    def unnormalize(self, x):
        x_01 = (x + 1) / 2
        return x_01 * (self.maxs - self.mins) + self.mins
        
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        start = self.indices[idx]
        end = start + CONFIG['horizon']
        traj = np.concatenate([self.obs[start:end], self.act[start:end]], axis=-1)
        return torch.tensor(self.normalize(traj), dtype=torch.float32)

# --- U-Net Components ---
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class TemporalUnet(nn.Module):
    def __init__(self, transition_dim, dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim), nn.Linear(dim, dim * 4), nn.Mish(), nn.Linear(dim * 4, dim),
        )
        self.down1 = nn.Sequential(nn.Conv1d(transition_dim, dim, 3, padding=1), nn.Mish())
        self.down2 = nn.Sequential(nn.Conv1d(dim, dim * 2, 3, padding=1), nn.Mish())
        self.down3 = nn.Sequential(nn.Conv1d(dim * 2, dim * 4, 3, padding=1), nn.Mish())
        self.up1 = nn.Sequential(nn.Conv1d(dim * 4, dim * 2, 3, padding=1), nn.Mish())
        self.up2 = nn.Sequential(nn.Conv1d(dim * 2, dim, 3, padding=1), nn.Mish())
        self.final_conv = nn.Conv1d(dim, transition_dim, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t).unsqueeze(-1)
        x1 = self.down1(x) + t_emb
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x = self.up1(x3) + x2
        x = self.up2(x) + x1
        return self.final_conv(x)

# --- Diffusion Manager ---
class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, transition_dim, n_timesteps=100): # 👈 100步推理够快了
        super().__init__()
        self.model = model
        self.horizon = horizon
        self.transition_dim = transition_dim
        self.n_timesteps = n_timesteps
        betas = torch.linspace(1e-4, 2e-2, n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('betas', betas)
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1. / alphas))
        self.register_buffer('posterior_variance', betas * (1. - torch.cat([torch.tensor([1.]), alphas_cumprod[:-1]])) / (1. - alphas_cumprod))

    def compute_loss(self, x_0):
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x_0.device).long()
        noise = torch.randn_like(x_0)
        coef1 = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1)
        coef2 = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1)
        x_t = coef1 * x_0 + coef2 * noise
        noise_pred = self.model(x_t.permute(0, 2, 1), t).permute(0, 2, 1)
        return nn.functional.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(self, cond_obs):
        """ 简单的采样逻辑，用于 Eval """
        batch_size = cond_obs.shape[0]
        device = cond_obs.device
        
        # 从纯噪声开始
        # Shape: [Batch, Horizon, Dim]
        x = torch.randn((batch_size, self.horizon, self.transition_dim), device=device)
        
        # DDPM 逆向采样
        for i in reversed(range(self.n_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            
            # 预测噪声
            noise_pred = self.model(x.permute(0, 2, 1), t).permute(0, 2, 1)
            
            # 计算 x_{t-1}
            # mean = 1/sqrt(alpha) * (x_t - beta/sqrt(1-alpha_bar) * noise_pred)
            alpha_t = self.sqrt_recip_alphas[i] ** (-2) # recover alpha from sqrt_recip
            # 这里简化公式，直接用 standard DDPM update
            beta_t = self.betas[i]
            sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[i]
            sqrt_recip_alpha_t = self.sqrt_recip_alphas[i]
            
            mean = sqrt_recip_alpha_t * (x - beta_t / sqrt_one_minus_alpha_cumprod_t * noise_pred)
            
            if i > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(self.posterior_variance[i])
                x = mean + sigma * noise
            else:
                x = mean
                
            # 【重要】每次去噪后，强制把第一步的 Observation 设为当前的真实 Obs
            # 这叫 "In-painting" 技巧，保证规划从当前状态开始
            # x[:, 0, :26] = cond_obs # 这里需要 normalized obs
            # 为了简单，我们在 loss 里不 mask obs，但在采样时并不强制替换 (Open-loop planning)
            # 或者我们在 unnormalize 后只取第一个 Action
            
        return x

# =================================================================
# 4. 评估函数 (Evaluation) - 自动打分
# =================================================================
# =================================================================
# 4. 评估函数 (Evaluation) - 自动打分 [修复版]
# =================================================================
def evaluate(diffusion_model, dataset, eval_episodes=10):
    """ 
    在真实环境中跑 N 个回合。
    策略：每一步调用 Diffuser 生成一条轨迹，执行第一个动作。
    """
    # 【修复1】：加上 disable_env_checker=True，防止 Gym 报错说返回值数量不对
    env = gymnasium.make('SafetyPointGoal1-v0', disable_env_checker=True).unwrapped
    device = CONFIG['device']
    
    total_collisions = 0
    total_success = 0
    total_reward = 0
    
    # 归一化参数
    mins = torch.tensor(dataset.mins, device=device)
    maxs = torch.tensor(dataset.maxs, device=device)
    
    print(f"\n🧪 Evaluating (Episodes={eval_episodes})...")
    
    # 临时设置为 eval 模式
    diffusion_model.model.eval()
    
    for ep in range(eval_episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        
        while not done and step < 200: 
            # (1) 随机初始化 x
            x = torch.randn((1, CONFIG['horizon'], CONFIG['obs_dim'] + CONFIG['act_dim']), device=device)
            
            # (2) 归一化当前 Obs
            curr_obs_norm = (obs - dataset.mins[:26]) / (dataset.maxs[:26] - dataset.mins[:26])
            curr_obs_norm = 2 * curr_obs_norm - 1
            curr_obs_torch = torch.tensor(curr_obs_norm, device=device, dtype=torch.float32)
            
            # (3) 逆向去噪
            with torch.no_grad():
                for i in reversed(range(diffusion_model.n_timesteps)):
                    t = torch.full((1,), i, device=device, dtype=torch.long)
                    
                    # In-painting: 强制修正第 0 步的 Obs 为当前真实 Obs
                    x[:, 0, :26] = curr_obs_torch 
                    
                    noise_pred = diffusion_model.model(x.permute(0, 2, 1), t).permute(0, 2, 1)
                    
                    beta_t = diffusion_model.betas[i]
                    sqrt_one_minus_alpha_cumprod_t = diffusion_model.sqrt_one_minus_alphas_cumprod[i]
                    sqrt_recip_alpha_t = diffusion_model.sqrt_recip_alphas[i]
                    
                    mean = sqrt_recip_alpha_t * (x - beta_t / sqrt_one_minus_alpha_cumprod_t * noise_pred)
                    
                    if i > 0:
                        x = mean + torch.sqrt(diffusion_model.posterior_variance[i]) * torch.randn_like(x)
                    else:
                        x = mean
                
            # (4) 采样结束，反归一化
            # 加了 .detach() 防止报错
            traj = x[0].detach().cpu().numpy() # [H, 28]
            traj = (traj + 1) / 2 * (dataset.maxs - dataset.mins) + dataset.mins
            
            # 取出第一个动作
            action = traj[0, 26:] 
            
            # 【修复2】：兼容 5 或 6 个返回值的解包逻辑
            step_result = env.step(action)
            
            if len(step_result) == 6:
                # Safety Gym 标准: obs, reward, cost, terminated, truncated, info
                obs, reward, cost, terminated, truncated, info = step_result
                done = terminated or truncated
            elif len(step_result) == 5:
                # Standard Gym: obs, reward, terminated, truncated, info
                obs, reward, terminated, truncated, info = step_result
                cost = info.get('cost', 0.0) # 尝试从 info 拿 cost
                done = terminated or truncated
            else:
                raise ValueError(f"Env step returned {len(step_result)} values, expected 5 or 6.")
            
            # 统计
            if cost > 0: total_collisions += 1
            
            total_reward += reward
            step += 1
            
    # 恢复训练模式
    diffusion_model.model.train()
            
    avg_collision = total_collisions / eval_episodes
    avg_reward = total_reward / eval_episodes
    
    # 打分公式
    score = avg_reward - 10 * avg_collision
    
    print(f"📊 Eval Result: Reward={avg_reward:.2f}, Collisions={avg_collision:.2f} | Score={score:.2f}")
    return score
# =================================================================
# 5. 主训练循环
# =================================================================
def train():
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    device = torch.device(CONFIG['device'])
    
    # 1. 准备数据
    dataset = TrajectoryDataset(CONFIG['dataset_path'], horizon=CONFIG['horizon'])
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
    np.savez(os.path.join(CONFIG['save_dir'], 'normalization.npz'), mins=dataset.mins, maxs=dataset.maxs)

    # 2. 初始化模型
    transition_dim = CONFIG['obs_dim'] + CONFIG['act_dim']
    unet = TemporalUnet(transition_dim=transition_dim, dim=CONFIG['hidden_dim']).to(device)
    # 注意：训练时 Timesteps 可以大一点 (100)，推理时再决定
    diffusion = GaussianDiffusion(unet, CONFIG['horizon'], transition_dim, n_timesteps=100).to(device)
    
    optimizer = torch.optim.Adam(unet.parameters(), lr=CONFIG['lr'])
    
    best_score = -float('inf') # 初始最低分
    
    print(f"🚀 Start Training... Steps: {CONFIG['train_steps']}")
    
    step = 0
    while step < CONFIG['train_steps']:
        for batch_traj in dataloader:
            batch_traj = batch_traj.to(device)
            loss = diffusion.compute_loss(batch_traj)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            step += 1
            if step % 100 == 0:
                print(f"Step {step} | Loss: {loss.item():.6f}")

            # --- 核心：定期评估并保存最佳 ---
            if step % CONFIG['eval_freq'] == 0:
                # 评估
                current_score = evaluate(diffusion, dataset, eval_episodes=CONFIG['eval_episodes'])
                
                # 记录 Best
                if current_score > best_score:
                    best_score = current_score
                    save_path = os.path.join(CONFIG['save_dir'], 'best_model.pt')
                    torch.save(unet.state_dict(), save_path)
                    print(f"🌟 New Best Model Found! Score: {best_score:.2f} -> Saved to {save_path}")
                else:
                    print(f"   (Current Score {current_score:.2f} < Best {best_score:.2f}, skip save)")
            
            if step >= CONFIG['train_steps']: break

    print("🎉 Training Finished!")

if __name__ == '__main__':
    train()