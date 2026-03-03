import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# =================================================================
# 1. 配置参数
# =================================================================
CONFIG = {
    # Diffuser 训练数据
    'dataset_path': './data_pro/ppolag_测试data.npz',  
    'save_dir': './diffuser_models/cbf_diffuser_lagrangian',
    
    # 🔴 CBF 模型路径 (从你提供的代码 1 中提取的保存路径)
    'cbf_model_path': './看cbf数据/ppolag_测试data2让边界变小/best_cbf_model.pt',
    'cbf_norm_path': './看cbf数据/ppolag_测试data2让边界变小/cbf_normalization.npz',

    'horizon': 64,          
    'obs_dim': 26,          
    'act_dim': 2,           
    'hidden_dim': 256,      
    'train_steps': 50000,   
    'batch_size': 256,     
    
    # 拉格朗日超参数
    'diff_lr': 2e-4,        # U-Net 学习率
    'lam_lr': 1e-3,         # Lambda 学习率
    'cost_limit': 0.05,     # 允许的微小安全误差阈值
    
    'device': 'cuda:0',     
}

# =================================================================
# 2. CBF 网络定义 (直接复用你的代码)
# =================================================================
class CBFNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # 输出 h(x)
        )
    def forward(self, x):
        return self.net(x)

# =================================================================
# 3. 数据集与 Diffuser 网络架构 (沿用你验证过好用的老版)
# =================================================================
class TrajectoryDataset(Dataset):
    def __init__(self, data_path, horizon=64):
        print(f"Loading data from {data_path}...")
        raw_data = np.load(data_path)
        self.obs = raw_data['obs'].astype(np.float32)
        self.act = raw_data['act'].astype(np.float32)
        self.segment_ids = raw_data['segment_id']
        
        # 使用你原本好用的绝对 Min-Max 归一化
        self.mins = np.concatenate([self.obs.min(axis=0), self.act.min(axis=0)])
        self.maxs = np.concatenate([self.obs.max(axis=0), self.act.max(axis=0)])
        self.maxs[self.maxs == self.mins] += 1e-6
        
        self.indices = []
        total_steps = len(self.obs)
        for i in range(total_steps - horizon + 1):
            if self.segment_ids[i] == self.segment_ids[i + horizon - 1]:
                self.indices.append(i)
                
    def normalize(self, x):
        x_norm = (x - self.mins) / (self.maxs - self.mins)
        return 2 * x_norm - 1

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_t = self.indices[idx]
        end_t = start_t + CONFIG['horizon']
        traj = np.concatenate([self.obs[start_t:end_t], self.act[start_t:end_t]], axis=-1)
        return torch.tensor(self.normalize(traj), dtype=torch.float32)

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

# 老版简易 U-Net
class TemporalUnet(nn.Module):
    def __init__(self, transition_dim, dim=256): 
        super().__init__()
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(dim), nn.Linear(dim, dim * 4), nn.Mish(), nn.Linear(dim * 4, dim))
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

# =================================================================
# 4. 融合 CBF 惩罚的扩散过程 (🔥 本次最核心的修改)
# =================================================================
class GaussianDiffusionCBF(nn.Module):
    def __init__(self, model, cbf_net, diff_mins, diff_maxs, cbf_mins, cbf_maxs, obs_dim=26, n_timesteps=100):
        super().__init__()
        self.model = model
        self.cbf_net = cbf_net
        self.obs_dim = obs_dim
        self.n_timesteps = n_timesteps
        
        # 预计算扩散系数
        betas = torch.linspace(1e-4, 2e-2, n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        
        # 注册归一化参数，用于尺度转换
        self.register_buffer('diff_mins', torch.tensor(diff_mins[:obs_dim]))
        self.register_buffer('diff_maxs', torch.tensor(diff_maxs[:obs_dim]))
        self.register_buffer('cbf_mins', torch.tensor(cbf_mins))
        self.register_buffer('cbf_maxs', torch.tensor(cbf_maxs))

    def compute_losses(self, x_0):
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # 1. 正常的扩散加噪与预测
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=device).long()
        noise = torch.randn_like(x_0)
        
        coef1 = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1)
        coef2 = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1)
        x_t = coef1 * x_0 + coef2 * noise
        
        x_t_in = x_t.permute(0, 2, 1) 
        noise_pred = self.model(x_t_in, t).permute(0, 2, 1)
        
        diff_mse_loss = nn.functional.mse_loss(noise_pred, noise)
        
        # 2. 🔥 CBF 逻辑：从预测的噪声反推生成的轨迹 x_0
        # 公式: x_0 = (x_t - sqrt(1 - a_bar) * noise) / sqrt(a_bar)
        inv_coef1 = (1.0 / coef1)
        inv_coef2 = (coef2 / coef1)
        pred_x_0 = inv_coef1 * x_t - inv_coef2 * noise_pred
        
        # 提取观测维度
        pred_obs_normed = pred_x_0[:, :, :self.obs_dim]
        
        # 桥梁 A：把 Diffuser 生成的 [-1, 1] 还原回物理绝对尺度
        pred_obs_raw = (pred_obs_normed + 1.0) / 2.0 * (self.diff_maxs - self.diff_mins) + self.diff_mins
        
        # 桥梁 B：把物理绝对尺度转换成 CBF 训练时用的 [-1, 1] (带 Clip)
        cbf_input = (pred_obs_raw - self.cbf_mins) / (self.cbf_maxs - self.cbf_mins)
        cbf_input = 2.0 * cbf_input - 1.0
        cbf_input = torch.clamp(cbf_input, -5.0, 5.0)
        
        # 3. 计算 CBF 惩罚
        h_val = self.cbf_net(cbf_input) # 输出形状 [batch, horizon, 1]
        
        # 我们期望 h_val >= 0 (安全)。如果 h_val < 0，用 ReLU 产生惩罚
        # h_val 前面加负号，意味着不安全程度
        cbf_penalty = torch.mean(torch.nn.functional.relu(-h_val))
        
        return diff_mse_loss, cbf_penalty

# =================================================================
# 5. 主训练循环 (双重优化：UNet + 拉格朗日乘子)
# =================================================================
def train():
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    device = torch.device(CONFIG['device'])
    
    # 1. 加载 Diffuser 数据
    dataset = TrajectoryDataset(CONFIG['dataset_path'], horizon=CONFIG['horizon'])
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
    np.savez(os.path.join(CONFIG['save_dir'], 'normalization.npz'), mins=dataset.mins, maxs=dataset.maxs)
    
    # 2. 加载冻结的预训练 CBF 网络
    print(f"🔄 加载 CBF 模型与归一化参数...")
    cbf_norm = np.load(CONFIG['cbf_norm_path'])
    cbf_mins, cbf_maxs = cbf_norm['mins'], cbf_norm['maxs']
    
    cbf_net = CBFNetwork(CONFIG['obs_dim'], 256).to(device)
    cbf_net.load_state_dict(torch.load(CONFIG['cbf_model_path'], map_location=device))
    cbf_net.eval() # 冻结 CBF 网络
    for param in cbf_net.parameters():
        param.requires_grad = False
        
    # 3. 初始化模型
    transition_dim = CONFIG['obs_dim'] + CONFIG['act_dim']
    unet = TemporalUnet(transition_dim=transition_dim, dim=CONFIG['hidden_dim']).to(device)
    diffusion = GaussianDiffusionCBF(
        model=unet, cbf_net=cbf_net, 
        diff_mins=dataset.mins, diff_maxs=dataset.maxs,
        cbf_mins=cbf_mins, cbf_maxs=cbf_maxs,
        obs_dim=CONFIG['obs_dim']
    ).to(device)
    
    # 4. 🔥 拉格朗日乘子初始化
    # 使用 log_lam 保证 lambda 永远是正数
    log_lam = torch.tensor(np.log(1.0), requires_grad=True, device=device)
    
    # 双优化器
    opt_unet = torch.optim.Adam(unet.parameters(), lr=CONFIG['diff_lr'])
    opt_lam = torch.optim.Adam([log_lam], lr=CONFIG['lam_lr'])
    
    loss_history = []
    cbf_history = []
    lam_history = []
    
    print(f"\n🚀 开始拉格朗日 CBF-Diffuser 训练...")
    step = 0
    
    while step < CONFIG['train_steps']:
        for batch_traj in dataloader:
            batch_traj = batch_traj.to(device)
            
            # 前向传播：拿到重构误差和 CBF 惩罚
            diff_loss, cbf_penalty = diffusion.compute_losses(batch_traj)
            
            # 当前的 Lambda 值
            lam = torch.exp(log_lam)
            
            # A. 更新 U-Net: 最小化 (Diff Loss + Lam * CBF_Penalty)
            total_loss = diff_loss + lam.detach() * cbf_penalty
            
            opt_unet.zero_grad()
            total_loss.backward()
            opt_unet.step()
            
            # B. 更新 Lambda: 最大化 Lambda * (CBF_Penalty - Limit)
            # 所以 loss_lam = - Lam * (CBF_Penalty - Limit)
            loss_lam = - lam * (cbf_penalty.detach() - CONFIG['cost_limit'])
            
            opt_lam.zero_grad()
            loss_lam.backward()
            opt_lam.step()
            
            # 记录日志
            loss_history.append(diff_loss.item())
            cbf_history.append(cbf_penalty.item())
            lam_history.append(lam.item())
            step += 1
            
            if step % 500 == 0:
                print(f"Step {step:05d} | MSE: {diff_loss.item():.4f} | CBF Penalty: {cbf_penalty.item():.4f} | Lambda: {lam.item():.4f}")
                
            if step % 10000 == 0 or step == CONFIG['train_steps']:
                torch.save(unet.state_dict(), os.path.join(CONFIG['save_dir'], f'unet_step_{step}.pt'))
                
            if step >= CONFIG['train_steps']:
                break

    print("📈 保存训练图表...")
    plt.figure(figsize=(15, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(loss_history, alpha=0.6)
    plt.title('Diffuser MSE Loss')
    
    plt.subplot(1, 3, 2)
    plt.plot(cbf_history, color='red', alpha=0.6)
    plt.title('CBF Penalty (Should go down)')
    
    plt.subplot(1, 3, 3)
    plt.plot(lam_history, color='green', alpha=0.6)
    plt.title('Lagrangian Multiplier $\lambda$')
    
    plt.savefig(os.path.join(CONFIG['save_dir'], 'cbf_training_logs.png'))
    plt.close()
    
    print(f"🎉 训练完成！模型保存在: {CONFIG['save_dir']}")

if __name__ == '__main__':
    train()