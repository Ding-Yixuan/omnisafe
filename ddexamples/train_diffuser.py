import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader


# =================================================================
# 1. 配置参数
# =================================================================
CONFIG = {
    'dataset_path': './data_pro/data_ppolag_exp.npz',
    'horizon': 64,          # 一次生成64步 (H)
    'obs_dim': 26,          # 观测维度
    'act_dim': 2,           # 动作维度
    'hidden_dim': 256,      # 网络隐藏层维度
    'train_steps': 50000,  # 训练步数 Gradient Steps
    'batch_size': 256,     
    'lr': 2e-4,             
    'device': 'cuda:0',     
    'save_dir': './diffuser_models/ppolag_exp',
}

# =================================================================
# 2. 数据集 (TrajectoryDataset)
# =================================================================
class TrajectoryDataset(Dataset):
    def __init__(self, data_path, horizon=64):
        print(f"Loading data from {data_path}...")
        raw_data = np.load(data_path)
        
        # 提取核心数据
        self.obs = raw_data['obs'].astype(np.float32)
        self.act = raw_data['act'].astype(np.float32)
        self.segment_ids = raw_data['segment_id']
        
        # 1. 归一化 (MinMax Normalization to [-1, 1])
        # Diffuser 对数据范围非常敏感，必须归一化
        self.mins = np.concatenate([self.obs.min(axis=0), self.act.min(axis=0)])
        self.maxs = np.concatenate([self.obs.max(axis=0), self.act.max(axis=0)])
        
        # 防止分母为0 (如果某维数据没变过)
        self.maxs[self.maxs == self.mins] += 1.0
        
        # 2. 构建索引 (找出所有合法的切片起点)
        # 我们不能跨越 segment_id，也不能切出长度小于 horizon 的片段
        self.indices = []
        total_steps = len(self.obs)
        
        print("✂️  Slicing trajectories...")
        for i in range(total_steps - horizon + 1):
            # 检查起点和终点是否在同一个 Segment 内
            if self.segment_ids[i] == self.segment_ids[i + horizon - 1]:
                self.indices.append(i)
                
        print(f"✅ Created {len(self.indices)} sequences (Horizon={horizon})")
        
    def normalize(self, x):
        """ [0, 1] -> [-1, 1] """
        # 先归一化到 [0, 1]
        x_norm = (x - self.mins) / (self.maxs - self.mins)
        # 再映射到 [-1, 1]
        return 2 * x_norm - 1

    def unnormalize(self, x):
        """ [-1, 1] -> original """
        x_01 = (x + 1) / 2
        return x_01 * (self.maxs - self.mins) + self.mins

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_t = self.indices[idx]
        end_t = start_t + CONFIG['horizon']
        
        # 取出这一段的 obs 和 act
        obs_seq = self.obs[start_t:end_t]
        act_seq = self.act[start_t:end_t]
        
        # 拼接成 Joint Trajectory: [H, obs_dim + act_dim]
        traj = np.concatenate([obs_seq, act_seq], axis=-1)
        
        # 归一化
        traj_norm = self.normalize(traj)
        return torch.tensor(traj_norm, dtype=torch.float32)

# =================================================================
# 3. 网络架构 (Temporal U-Net Block)
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
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class TemporalUnet(nn.Module):
    def __init__(self, transition_dim, dim=128):
        super().__init__()
        
        # 时间步编码 (Time Embedding)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        # 编码器 (Downsample)
        self.down1 = nn.Sequential(nn.Conv1d(transition_dim, dim, 3, padding=1), nn.Mish())
        self.down2 = nn.Sequential(nn.Conv1d(dim, dim * 2, 3, padding=1), nn.Mish())
        self.down3 = nn.Sequential(nn.Conv1d(dim * 2, dim * 4, 3, padding=1), nn.Mish())

        # 解码器 (Upsample)
        self.up1 = nn.Sequential(nn.Conv1d(dim * 4, dim * 2, 3, padding=1), nn.Mish())
        self.up2 = nn.Sequential(nn.Conv1d(dim * 2, dim, 3, padding=1), nn.Mish())
        self.final_conv = nn.Conv1d(dim, transition_dim, 1)

    def forward(self, x, t):
        # x: [Batch, Dim, Horizon] (注意 Conv1d 需要 Dim 在中间)
        # t: [Batch]
        
        t_emb = self.time_mlp(t).unsqueeze(-1) # [Batch, Dim, 1]
        
        # Down
        x1 = self.down1(x) + t_emb
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        
        # Up (简单的 U-Net 连接)
        x = self.up1(x3) + x2
        x = self.up2(x) + x1
        x = self.final_conv(x)
        return x

# =================================================================
# 4. 扩散过程管理 (GaussianDiffusion)
# =================================================================
class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, transition_dim, n_timesteps=1000):
        super().__init__()
        self.model = model
        self.horizon = horizon
        self.transition_dim = transition_dim
        self.n_timesteps = n_timesteps

        # 定义 Beta Schedule (Linear)
        betas = torch.linspace(1e-4, 2e-2, n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # 注册 buffer，这样它们会自动转到 GPU
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

    def compute_loss(self, x_0):
        """ 计算去噪 Loss """
        batch_size = x_0.shape[0]
        
        # 1. 随机采样时间步 t
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x_0.device).long()
        
        # 2. 生成噪声 noise
        noise = torch.randn_like(x_0)
        
        # 3. 加噪 (Forward Process): x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * noise
        # 提取对应 t 的系数
        coef1 = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1)
        coef2 = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1)
        
        x_t = coef1 * x_0 + coef2 * noise
        
        # 4. 模型预测噪声 (Model Prediction)
        # Conv1d 需要 [Batch, Dim, Horizon]，所以我们要 permute
        x_t_in = x_t.permute(0, 2, 1) 
        noise_pred = self.model(x_t_in, t)
        noise_pred = noise_pred.permute(0, 2, 1) # 转回 [Batch, Horizon, Dim]
        
        # 5. 计算 MSE Loss
        return nn.functional.mse_loss(noise_pred, noise)

# =================================================================
# 5. 主训练循环
# =================================================================
def train():
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    device = torch.device(CONFIG['device'])
    
    # 1. 准备数据
    dataset = TrajectoryDataset(CONFIG['dataset_path'], horizon=CONFIG['horizon'])
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    
    # 保存归一化参数，推理时要用
    np.savez(os.path.join(CONFIG['save_dir'], 'normalization.npz'), mins=dataset.mins, maxs=dataset.maxs)
    print("✅ Normalization params saved.")

    # 2. 初始化模型
    transition_dim = CONFIG['obs_dim'] + CONFIG['act_dim']
    unet = TemporalUnet(transition_dim=transition_dim, dim=CONFIG['hidden_dim']).to(device)
    diffusion = GaussianDiffusion(unet, CONFIG['horizon'], transition_dim).to(device)
    
    optimizer = torch.optim.Adam(unet.parameters(), lr=CONFIG['lr'])
    
    print(f"🚀 Start Training Diffuser... Steps: {CONFIG['train_steps']}")
    
    step = 0
    loss_history = []
    
    while step < CONFIG['train_steps']:
        for batch_traj in dataloader:
            batch_traj = batch_traj.to(device) # [Batch, Horizon, Dim]
            
            # 计算 Loss
            loss = diffusion.compute_loss(batch_traj)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            step += 1
            loss_history.append(loss.item())
            
            if step % 100 == 0:
                print(f"Step {step}/{CONFIG['train_steps']} | Loss: {loss.item():.6f}")
                
            if step % 5000 == 0:
                # 保存模型
                save_path = os.path.join(CONFIG['save_dir'], f'diffuser_step_{step}.pt')
                torch.save(unet.state_dict(), save_path)
                print(f"💾 Model saved to {save_path}")
                
                # 画 Loss 曲线
                plt.figure()
                plt.plot(loss_history)
                plt.title("Diffusion Training Loss")
                plt.xlabel("Steps")
                plt.ylabel("MSE Loss")
                plt.savefig(os.path.join(CONFIG['save_dir'], 'loss_curve.png'))
                plt.close()
                
            if step >= CONFIG['train_steps']:
                break

    print("🎉 Training Finished!")

if __name__ == '__main__':
    train()