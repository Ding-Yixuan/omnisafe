# import os
# import torch
# import torch.nn as nn
# import numpy as np
# from torch.utils.data import Dataset, DataLoader

# # =================================================================
# # 1. 配置参数
# # =================================================================
# CONFIG = {
#     # 🔴 训练 PPO 数据时用这组：
#     # 'dataset_path': './data_pro/data_ppo_exp.npz',
#     # 'save_dir': './diffuser_models/ppo_exp',
    
#     # 🟢 训练 PPOLag 数据时用这组 (先把这组注释放开跑)：
#     'dataset_path': './data_pro/data_ppolag_exp.npz',
#     'save_dir': './diffuser_models/ppolag_exp',

#     'horizon': 64,          # 预测未来 64 步
#     'obs_dim': 26,          # 你的 26 维感知
#     'act_dim': 2,           # 2 维动作
#     'hidden_dim': 256,      # U-Net 隐藏层维度
#     'train_steps': 50000,   # 训练步数 (通常 5 万步足够收敛)
#     'batch_size': 256,     
#     'lr': 2e-4,             
#     'device': 'cuda:0',     
# }

# # =================================================================
# # 2. 数据集加载与归一化 (引入 99% 分位数防抖动)
# # =================================================================
# class TrajectoryDataset(Dataset):
#     def __init__(self, data_path, horizon=64):
#         print(f"Loading data from {data_path}...")
#         raw_data = np.load(data_path)
        
#         self.obs = raw_data['obs'].astype(np.float32)
#         self.act = raw_data['act'].astype(np.float32)
#         self.segment_ids = raw_data['segment_id']
        
#         # 【核心优化】使用 1% 和 99% 分位数代替绝对最大最小值，过滤极端异常点
#         self.mins = np.concatenate([np.percentile(self.obs, 1, axis=0), np.percentile(self.act, 1, axis=0)])
#         self.maxs = np.concatenate([np.percentile(self.obs, 99, axis=0), np.percentile(self.act, 99, axis=0)])
        
#         # 防止分母为 0
#         self.maxs[self.maxs == self.mins] += 1e-6
        
#         # 构建轨迹切片索引
#         self.indices = []
#         total_steps = len(self.obs)
#         for i in range(total_steps - horizon + 1):
#             if self.segment_ids[i] == self.segment_ids[i + horizon - 1]:
#                 self.indices.append(i)
                
#         print(f"✅ 构建了 {len(self.indices)} 条有效轨迹 (长度 {horizon})")
        
#     def normalize(self, x):
#         x_norm = (x - self.mins) / (self.maxs - self.mins)
#         return 2 * x_norm - 1 # 映射到 [-1, 1]

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, idx):
#         start_t = self.indices[idx]
#         end_t = start_t + CONFIG['horizon']
        
#         obs_seq = self.obs[start_t:end_t]
#         act_seq = self.act[start_t:end_t]
#         traj = np.concatenate([obs_seq, act_seq], axis=-1)
#         traj_norm = self.normalize(traj)
        
#         return torch.tensor(traj_norm, dtype=torch.float32)

# # =================================================================
# # 3. 强化版 U-Net 架构 (带 GroupNorm 和深层 Time Emb)
# # =================================================================
# class SinusoidalPosEmb(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim
#     def forward(self, x):
#         device = x.device
#         half_dim = self.dim // 2
#         emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
#         emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
#         emb = x[:, None] * emb[None, :]
#         return torch.cat((emb.sin(), emb.cos()), dim=-1)

# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv1d(in_channels, out_channels, 3, padding=1),
#             nn.GroupNorm(8, out_channels), 
#             nn.Mish()
#         )
#     def forward(self, x):
#         return self.block(x)

# class TemporalUnet(nn.Module):
#     def __init__(self, transition_dim, dim=256):
#         super().__init__()
#         self.time_mlp = nn.Sequential(SinusoidalPosEmb(dim), nn.Linear(dim, dim * 4), nn.Mish())
        
#         self.time_proj1 = nn.Linear(dim * 4, dim)
#         self.time_proj2 = nn.Linear(dim * 4, dim * 2)
#         self.time_proj3 = nn.Linear(dim * 4, dim * 4)

#         self.down1 = ConvBlock(transition_dim, dim)
#         self.down2 = ConvBlock(dim, dim * 2)
#         self.down3 = ConvBlock(dim * 2, dim * 4)

#         self.up1 = ConvBlock(dim * 4, dim * 2)
#         self.up2 = ConvBlock(dim * 2, dim)
#         self.final_conv = nn.Conv1d(dim, transition_dim, 1)

#     def forward(self, x, t):
#         t_emb = self.time_mlp(t) 
#         t1 = self.time_proj1(t_emb).unsqueeze(-1)
#         t2 = self.time_proj2(t_emb).unsqueeze(-1)
#         t3 = self.time_proj3(t_emb).unsqueeze(-1)
        
#         x1 = self.down1(x) + t1
#         x2 = self.down2(x1) + t2
#         x3 = self.down3(x2) + t3
        
#         x = self.up1(x3) + x2
#         x = self.up2(x) + x1
#         return self.final_conv(x)

# # =================================================================
# # 4. 扩散过程 GaussianDiffusion
# # =================================================================
# class GaussianDiffusion(nn.Module):
#     def __init__(self, model, n_timesteps=100): # 100 步足以拟合低维控制
#         super().__init__()
#         self.model = model
#         self.n_timesteps = n_timesteps
#         betas = torch.linspace(1e-4, 2e-2, n_timesteps)
#         alphas = 1. - betas
#         alphas_cumprod = torch.cumprod(alphas, dim=0)
        
#         self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
#         self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

#     def compute_loss(self, x_0):
#         batch_size = x_0.shape[0]
#         t = torch.randint(0, self.n_timesteps, (batch_size,), device=x_0.device).long()
#         noise = torch.randn_like(x_0)
        
#         coef1 = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1)
#         coef2 = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1)
#         x_t = coef1 * x_0 + coef2 * noise
        
#         x_t_in = x_t.permute(0, 2, 1) 
#         noise_pred = self.model(x_t_in, t).permute(0, 2, 1)
        
#         return nn.functional.mse_loss(noise_pred, noise)

# # =================================================================
# # 5. 训练主循环
# # =================================================================
# def train():
#     os.makedirs(CONFIG['save_dir'], exist_ok=True)
#     device = torch.device(CONFIG['device'])
    
#     dataset = TrajectoryDataset(CONFIG['dataset_path'], horizon=CONFIG['horizon'])
#     dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    
#     # 【非常重要】保存归一化参数，推理/评估时需要！
#     np.savez(os.path.join(CONFIG['save_dir'], 'normalization.npz'), mins=dataset.mins, maxs=dataset.maxs)
    
#     transition_dim = CONFIG['obs_dim'] + CONFIG['act_dim']
#     unet = TemporalUnet(transition_dim=transition_dim, dim=CONFIG['hidden_dim']).to(device)
#     diffusion = GaussianDiffusion(unet).to(device)
    
#     optimizer = torch.optim.Adam(unet.parameters(), lr=CONFIG['lr'])
    
#     print(f"\n🚀 开始训练 Diffuser... (目标步数: {CONFIG['train_steps']})")
#     step = 0
    
#     while step < CONFIG['train_steps']:
#         for batch_traj in dataloader:
#             batch_traj = batch_traj.to(device)
#             loss = diffusion.compute_loss(batch_traj)
            
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             step += 1
#             if step % 500 == 0:
#                 print(f"Step {step:05d}/{CONFIG['train_steps']} | MSE Loss: {loss.item():.6f}")
                
#             if step % 10000 == 0 or step == CONFIG['train_steps']:
#                 torch.save(unet.state_dict(), os.path.join(CONFIG['save_dir'], f'unet_step_{step}.pt'))
                
#             if step >= CONFIG['train_steps']:
#                 break

#     print(f"🎉 训练完成！模型保存在: {CONFIG['save_dir']}")

# if __name__ == '__main__':
#     train()


import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# =================================================================
# 1. 配置参数
# =================================================================
CONFIG = {
    # 🔴 请确认你的数据路径和想要保存模型的路径
    'dataset_path': './data_pro/ppo_测试.npz',
    'save_dir': './diffuser_models/ppo_测试',

    'horizon': 64,          # 预测未来 64 步
    'obs_dim': 26,          # 26 维感知
    'act_dim': 2,           # 2 维动作
    'hidden_dim': 256,      # U-Net 隐藏层维度
    'train_steps': 50000,   # 训练步数 (建议 5 万步)
    'batch_size': 256,     
    'lr': 2e-4,             
    'device': 'cuda:0',     
}

# =================================================================
# 2. 数据集加载与归一化 (引入 99% 分位数防抖动)
# =================================================================
class TrajectoryDataset(Dataset):
    def __init__(self, data_path, horizon=64):
        print(f"Loading data from {data_path}...")
        raw_data = np.load(data_path)
        
        self.obs = raw_data['obs'].astype(np.float32)
        self.act = raw_data['act'].astype(np.float32)
        self.segment_ids = raw_data['segment_id']
        
        # 【核心优化】使用 1% 和 99% 分位数代替绝对最大最小值，过滤极端异常点
        self.mins = np.concatenate([np.percentile(self.obs, 1, axis=0), np.percentile(self.act, 1, axis=0)])
        self.maxs = np.concatenate([np.percentile(self.obs, 99, axis=0), np.percentile(self.act, 99, axis=0)])
        
        # 防止分母为 0
        self.maxs[self.maxs == self.mins] += 1e-6
        
        # 构建轨迹切片索引
        self.indices = []
        total_steps = len(self.obs)
        for i in range(total_steps - horizon + 1):
            if self.segment_ids[i] == self.segment_ids[i + horizon - 1]:
                self.indices.append(i)
                
        print(f"✅ 构建了 {len(self.indices)} 条有效轨迹 (长度 {horizon})")
        
    def normalize(self, x):
        x_norm = (x - self.mins) / (self.maxs - self.mins)
        return 2 * x_norm - 1 # 映射到 [-1, 1]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_t = self.indices[idx]
        end_t = start_t + CONFIG['horizon']
        
        obs_seq = self.obs[start_t:end_t]
        act_seq = self.act[start_t:end_t]
        traj = np.concatenate([obs_seq, act_seq], axis=-1)
        traj_norm = self.normalize(traj)
        
        return torch.tensor(traj_norm, dtype=torch.float32)

# =================================================================
# 3. 强化版 U-Net 架构 (带 GroupNorm 和深层 Time Emb)
# =================================================================
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

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels), 
            nn.Mish()
        )
    def forward(self, x):
        return self.block(x)

class TemporalUnet(nn.Module):
    def __init__(self, transition_dim, dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(dim), nn.Linear(dim, dim * 4), nn.Mish())
        
        self.time_proj1 = nn.Linear(dim * 4, dim)
        self.time_proj2 = nn.Linear(dim * 4, dim * 2)
        self.time_proj3 = nn.Linear(dim * 4, dim * 4)

        self.down1 = ConvBlock(transition_dim, dim)
        self.down2 = ConvBlock(dim, dim * 2)
        self.down3 = ConvBlock(dim * 2, dim * 4)

        self.up1 = ConvBlock(dim * 4, dim * 2)
        self.up2 = ConvBlock(dim * 2, dim)
        self.final_conv = nn.Conv1d(dim, transition_dim, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t) 
        t1 = self.time_proj1(t_emb).unsqueeze(-1)
        t2 = self.time_proj2(t_emb).unsqueeze(-1)
        t3 = self.time_proj3(t_emb).unsqueeze(-1)
        
        x1 = self.down1(x) + t1
        x2 = self.down2(x1) + t2
        x3 = self.down3(x2) + t3
        
        x = self.up1(x3) + x2
        x = self.up2(x) + x1
        return self.final_conv(x)

# =================================================================
# 4. 扩散过程 GaussianDiffusion
# =================================================================
class GaussianDiffusion(nn.Module):
    def __init__(self, model, n_timesteps=100): # 100 步足以拟合低维控制
        super().__init__()
        self.model = model
        self.n_timesteps = n_timesteps
        betas = torch.linspace(1e-4, 2e-2, n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

    def compute_loss(self, x_0):
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x_0.device).long()
        noise = torch.randn_like(x_0)
        
        coef1 = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1)
        coef2 = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1)
        x_t = coef1 * x_0 + coef2 * noise
        
        x_t_in = x_t.permute(0, 2, 1) 
        noise_pred = self.model(x_t_in, t).permute(0, 2, 1)
        
        return nn.functional.mse_loss(noise_pred, noise)

# =================================================================
# 5. 训练主循环
# =================================================================
# =================================================================
# 5. 训练主循环 (带有 Loss 曲线绘制)
# =================================================================
def train():
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    device = torch.device(CONFIG['device'])
    
    dataset = TrajectoryDataset(CONFIG['dataset_path'], horizon=CONFIG['horizon'])
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    
    np.savez(os.path.join(CONFIG['save_dir'], 'normalization.npz'), mins=dataset.mins, maxs=dataset.maxs)
    
    transition_dim = CONFIG['obs_dim'] + CONFIG['act_dim']
    unet = TemporalUnet(transition_dim=transition_dim, dim=CONFIG['hidden_dim']).to(device)
    diffusion = GaussianDiffusion(unet).to(device)
    
    optimizer = torch.optim.Adam(unet.parameters(), lr=CONFIG['lr'])
    
    print(f"\n🚀 开始训练 Diffuser... (目标步数: {CONFIG['train_steps']})")
    step = 0
    
    # 📝 新增：用于记录 Loss 的小本本
    loss_history = []
    
    while step < CONFIG['train_steps']:
        for batch_traj in dataloader:
            batch_traj = batch_traj.to(device)
            loss = diffusion.compute_loss(batch_traj)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录当前步的 Loss
            loss_history.append(loss.item())
            step += 1
            
            if step % 500 == 0:
                print(f"Step {step:05d}/{CONFIG['train_steps']} | MSE Loss: {loss.item():.6f}")
                
            if step % 10000 == 0 or step == CONFIG['train_steps']:
                torch.save(unet.state_dict(), os.path.join(CONFIG['save_dir'], f'unet_step_{step}.pt'))
                
            if step >= CONFIG['train_steps']:
                break

    print(f"🎉 训练完成！模型保存在: {CONFIG['save_dir']}")
    
    # 🎨 新增：画出极其舒适的 Loss 曲线
    print("📈 正在绘制并保存 Loss 曲线...")
    plt.figure(figsize=(10, 5))
    
    # 1. 画出原始震荡的浅色曲线
    plt.plot(loss_history, alpha=0.3, color='royalblue', label='Raw Step Loss')
    
    # 2. 画出平滑后的深色曲线 (滑动窗口设为 100)
    window_size = 100
    if len(loss_history) > window_size:
        smoothed_loss = np.convolve(loss_history, np.ones(window_size)/window_size, mode='valid')
        plt.plot(np.arange(window_size-1, len(loss_history)), smoothed_loss, color='darkorange', linewidth=2, label=f'Smoothed Loss (window={window_size})')
        
    plt.xlabel('Training Steps')
    plt.ylabel('MSE Loss')
    plt.title('Diffuser Training Loss Curve')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 保存图片
    loss_fig_path = os.path.join(CONFIG['save_dir'], 'loss_curve.png')
    plt.savefig(loss_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Loss 曲线已保存至: {loss_fig_path}")

if __name__ == '__main__':
    train()