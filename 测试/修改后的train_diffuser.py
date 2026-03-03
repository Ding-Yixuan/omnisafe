import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader

# =================================================================
# 1. 配置参数 (🔥 修改: 引入 T_o 和 T_p，抛弃统一的 horizon)
# =================================================================
CONFIG = {
    'dataset_path': './data_pro/ppolag_测试data.npz',
    'obs_horizon': 2,       # 👈 新增: 每次观测历史 2 步 (T_o)
    'pred_horizon': 16,     # 👈 新增: 每次规划未来 16 步 (T_p)
    'obs_dim': 26,          
    'act_dim': 2,           
    'hidden_dim': 256,      
    'train_steps': 50000,  
    'batch_size': 256,     
    'lr': 2e-4,             
    'device': 'cuda:0',     
    'save_dir': './diffuser_models/新的ppolag', # 建议换个新目录存新模型
}

# =================================================================
# 2. 数据集 (🔥 修改: 解耦 obs 和 act 的切片逻辑)
# =================================================================
class TrajectoryDataset(Dataset):
    def __init__(self, data_path, obs_horizon=2, pred_horizon=16):
        print(f"Loading data from {data_path}...")
        raw_data = np.load(data_path)
        
        self.obs = raw_data['obs'].astype(np.float32)
        self.act = raw_data['act'].astype(np.float32)
        self.segment_ids = raw_data['segment_id']
        
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        
        # 为了兼容你后期的推理脚本，保留统一的 mins 和 maxs 保存格式
        self.mins = np.concatenate([self.obs.min(axis=0), self.act.min(axis=0)])
        self.maxs = np.concatenate([self.obs.max(axis=0), self.act.max(axis=0)])
        self.maxs[self.maxs == self.mins] += 1.0
        
        self.indices = []
        total_steps = len(self.obs)
        
        print("✂️  Slicing trajectories...")
        # 确保切片的最长部分（通常是 pred_horizon）在同一个 segment 内
        max_len = max(obs_horizon, pred_horizon)
        for i in range(total_steps - max_len + 1):
            if self.segment_ids[i] == self.segment_ids[i + max_len - 1]:
                self.indices.append(i)
                
        print(f"✅ Created {len(self.indices)} sequences (Obs H={obs_horizon}, Pred H={pred_horizon})")
        
    def normalize(self, x, mins, maxs):
        """ 通用归一化到 [-1, 1] """
        x_norm = (x - mins) / (maxs - mins)
        return 2 * x_norm - 1

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_t = self.indices[idx]
        
        # 🔥 修改: 分别取出 obs 和 act，不再 concat
        # obs 取前 obs_horizon 步作为条件
        obs_seq = self.obs[start_t : start_t + self.obs_horizon]
        # act 取前 pred_horizon 步作为要生成的预测目标
        act_seq = self.act[start_t : start_t + self.pred_horizon]
        
        # 分别归一化
        obs_norm = self.normalize(obs_seq, self.mins[:26], self.maxs[:26])
        act_norm = self.normalize(act_seq, self.mins[26:], self.maxs[26:])
        
        return torch.tensor(obs_norm, dtype=torch.float32), torch.tensor(act_norm, dtype=torch.float32)

# =================================================================
# 3. 网络架构 (🔥 修改: 引入条件编码器，将输入维度改为纯动作)
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
    # 🔥 修改: 接收 act_dim 和 obs 相关参数
    def __init__(self, act_dim, obs_dim, obs_horizon, dim=256):
        super().__init__()
        
        # 👈 新增: 观测条件编码器 (Observation Encoder)
        # 将 (T_o * obs_dim) 的张量压平后提取特征
        obs_feat_dim = obs_dim * obs_horizon
        self.obs_mlp = nn.Sequential(
            nn.Linear(obs_feat_dim, dim),
            nn.Mish(),
            nn.Linear(dim, dim)
        )

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        # 🔥 修改: U-Net 的最外层输入和输出通道数变为了 act_dim (2)，不再是 28
        self.down1 = nn.Sequential(nn.Conv1d(act_dim, dim, 3, padding=1), nn.Mish())
        self.down2 = nn.Sequential(nn.Conv1d(dim, dim * 2, 3, padding=1), nn.Mish())
        self.down3 = nn.Sequential(nn.Conv1d(dim * 2, dim * 4, 3, padding=1), nn.Mish())

        self.up1 = nn.Sequential(nn.Conv1d(dim * 4, dim * 2, 3, padding=1), nn.Mish())
        self.up2 = nn.Sequential(nn.Conv1d(dim * 2, dim, 3, padding=1), nn.Mish())
        self.final_conv = nn.Conv1d(dim, act_dim, 1)

    # 🔥 修改: 前向传播新增 obs_cond 参数
    def forward(self, x, t, obs_cond):
        # x: [Batch, act_dim, pred_horizon]
        # obs_cond: [Batch, obs_horizon, obs_dim]
        
        batch_size = x.shape[0]
        
        # 1. 提取观测特征 [Batch, Dim] -> [Batch, Dim, 1] 匹配空间维度
        obs_cond_flat = obs_cond.view(batch_size, -1)
        obs_emb = self.obs_mlp(obs_cond_flat).unsqueeze(-1)
        
        # 2. 提取时间特征
        t_emb = self.time_mlp(t).unsqueeze(-1) 
        
        # 3. 融合条件 (基于论文 FiLM 思路的简化加法注入)
        cond_emb = t_emb + obs_emb 
        
        # Down
        x1 = self.down1(x) + cond_emb # 👈 将融合后的条件注入网络
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        
        # Up
        x = self.up1(x3) + x2
        x = self.up2(x) + x1
        x = self.final_conv(x)
        return x

# =================================================================
# 4. 扩散过程管理 (🔥 修改: 只对 action 加噪，透传 obs_cond)
# =================================================================
class GaussianDiffusion(nn.Module):
    def __init__(self, model, n_timesteps=1000):
        super().__init__()
        self.model = model
        self.n_timesteps = n_timesteps

        betas = torch.linspace(1e-4, 2e-2, n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

    # 🔥 修改: 计算 Loss 时分离条件和目标
    def compute_loss(self, obs_cond, act_target):
        batch_size = act_target.shape[0]
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=act_target.device).long()
        
        # 🔥 修改: 只对 action (act_target) 生成噪声
        noise = torch.randn_like(act_target)
        
        coef1 = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1)
        coef2 = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1)
        
        x_t = coef1 * act_target + coef2 * noise
        
        x_t_in = x_t.permute(0, 2, 1) 
        
        # 🔥 修改: 把干净的 obs_cond 传给模型作为指导
        noise_pred = self.model(x_t_in, t, obs_cond)
        noise_pred = noise_pred.permute(0, 2, 1) 
        
        # 纯比较 action 维度的预测误差
        return nn.functional.mse_loss(noise_pred, noise)

# =================================================================
# 5. 主训练循环 (🔥 修改: 获取并传递解耦后的数据)
# =================================================================
def train():
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    device = torch.device(CONFIG['device'])
    
    # 初始化 Dataset
    dataset = TrajectoryDataset(
        CONFIG['dataset_path'], 
        obs_horizon=CONFIG['obs_horizon'], 
        pred_horizon=CONFIG['pred_horizon']
    )
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    
    np.savez(os.path.join(CONFIG['save_dir'], 'normalization.npz'), mins=dataset.mins, maxs=dataset.maxs)
    print("✅ Normalization params saved.")

    # 🔥 修改: U-Net 初始化参数更新
    unet = TemporalUnet(
        act_dim=CONFIG['act_dim'], 
        obs_dim=CONFIG['obs_dim'], 
        obs_horizon=CONFIG['obs_horizon'], 
        dim=CONFIG['hidden_dim']
    ).to(device)
    
    diffusion = GaussianDiffusion(unet).to(device)
    optimizer = torch.optim.Adam(unet.parameters(), lr=CONFIG['lr'])
    
    print(f"🚀 Start Training Conditioned Diffuser... Steps: {CONFIG['train_steps']}")
    
    step = 0
    loss_history = []
    
    while step < CONFIG['train_steps']:
        # 🔥 修改: 这里 unpack 出两个张量
        for obs_batch, act_batch in dataloader:
            obs_batch = obs_batch.to(device) # [Batch, obs_horizon, obs_dim]
            act_batch = act_batch.to(device) # [Batch, pred_horizon, act_dim]
            
            # 🔥 修改: 传入观测和动作
            loss = diffusion.compute_loss(obs_cond=obs_batch, act_target=act_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            step += 1
            loss_history.append(loss.item())
            
            if step % 100 == 0:
                print(f"Step {step}/{CONFIG['train_steps']} | Loss: {loss.item():.6f}")
                
            if step % 5000 == 0:
                save_path = os.path.join(CONFIG['save_dir'], f'unet_step_{step}.pt')
                torch.save(unet.state_dict(), save_path)
                print(f"💾 Model saved to {save_path}")
                
                plt.figure()
                plt.plot(loss_history)
                plt.title("Conditional Diffusion Training Loss")
                plt.xlabel("Steps")
                plt.ylabel("MSE Loss")
                plt.savefig(os.path.join(CONFIG['save_dir'], 'loss_curve.png'))
                plt.close()
                
            if step >= CONFIG['train_steps']:
                break

    print("🎉 Training Finished!")

if __name__ == '__main__':
    train()