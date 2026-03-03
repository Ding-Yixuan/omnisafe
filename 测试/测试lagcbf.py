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
    'dataset_path': './data_pro/ppolag_测试data.npz',  
    'save_dir': './diffuser_models/cbf_diffuser_pdf_2',
    
    # CBF 模型路径 (确保路径正确)
    'cbf_model_path': './看cbf数据/ppolag_测试data2让边界变小/best_cbf_model.pt',
    'cbf_norm_path': './看cbf数据/ppolag_测试data2让边界变小/cbf_normalization.npz',

    'horizon': 64,          
    'obs_dim': 26,          
    'act_dim': 2,           
    'hidden_dim': 256,      
    'train_steps': 50000,   
    'batch_size': 256,     
    

    'diff_lr': 2e-4,        
    'nu_lr': 1e-3,          
    'safety_threshold': 0.3,
    
    'device': 'cuda:0',     
}

# =================================================================
# 2. CBF 网络定义 & 简易 U-Net 架构 (保持你的老版本)
# =================================================================
class CBFNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1) # [cite: 56]
        )
    def forward(self, x):
        return self.net(x)

class TrajectoryDataset(Dataset):
    def __init__(self, data_path, horizon=64):
        raw_data = np.load(data_path)
        self.obs, self.act, self.segment_ids = raw_data['obs'].astype(np.float32), raw_data['act'].astype(np.float32), raw_data['segment_id']
        self.mins = np.concatenate([self.obs.min(axis=0), self.act.min(axis=0)])
        self.maxs = np.concatenate([self.obs.max(axis=0), self.act.max(axis=0)])
        self.maxs[self.maxs == self.mins] += 1e-6
        self.indices = [i for i in range(len(self.obs) - horizon + 1) if self.segment_ids[i] == self.segment_ids[i + horizon - 1]]
                
    def normalize(self, x):
        return 2 * ((x - self.mins) / (self.maxs - self.mins)) - 1

    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        start_t = self.indices[idx]
        traj = np.concatenate([self.obs[start_t:start_t+CONFIG['horizon']], self.act[start_t:start_t+CONFIG['horizon']]], axis=-1)
        return torch.tensor(self.normalize(traj), dtype=torch.float32)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        emb = torch.exp(torch.arange(self.dim // 2, device=x.device) * -(torch.log(torch.tensor(10000.0)) / (self.dim // 2 - 1)))
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

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
        # ✅ 这里的 def forward 必须和上面的 def __init__ 严格左对齐！
        x1 = self.down1(x) + t_emb
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x = self.up1(x3) + x2
        x = self.up2(x) + x1
        return self.final_conv(x)
# =================================================================
# 3. 严格参照 PDF 重写的 Diffusion 过程
# =================================================================
class GaussianDiffusionCBF(nn.Module):
    def __init__(self, model, cbf_net, diff_mins, diff_maxs, cbf_mins, cbf_maxs, obs_dim=26, n_timesteps=100):
        super().__init__()
        self.model = model
        self.cbf_net = cbf_net # PDF [cite: 208]
        self.obs_dim = obs_dim
        self.n_timesteps = n_timesteps
        
        betas = torch.linspace(1e-4, 2e-2, n_timesteps)
        alphas_cumprod = torch.cumprod(1. - betas, dim=0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        
        self.register_buffer('diff_mins', torch.tensor(diff_mins[:obs_dim]))
        self.register_buffer('diff_maxs', torch.tensor(diff_maxs[:obs_dim]))
        self.register_buffer('cbf_mins', torch.tensor(cbf_mins))
        self.register_buffer('cbf_maxs', torch.tensor(cbf_maxs))

    def p_losses(self, x_start): # 参照 PDF 的 p_losses 命名 [cite: 159]
        batch_size = x_start.shape[0]
        device = x_start.device
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=device).long()
        noise = torch.randn_like(x_start)
        
        coef1 = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1)
        coef2 = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1)
        x_noisy = coef1 * x_start + coef2 * noise
        
        model_out = self.model(x_noisy.permute(0, 2, 1), t).permute(0, 2, 1)
        
        # 1. 原始 Diffusion Loss [cite: 164]
        loss_diff = nn.functional.mse_loss(model_out, noise)
        
        # 2. 利用预测的噪声反推 x_0 [cite: 168]
        x_0_pred = (x_noisy - coef2 * model_out) / coef1 
        
        # 3. 提取状态并反归一化 (参照 PDF 的 phys_state 计算) [cite: 201]
        pred_obs_normed = x_0_pred[:, :, :self.obs_dim]
        phys_state = (pred_obs_normed + 1.0) / 2.0 * (self.diff_maxs - self.diff_mins) + self.diff_mins
        
        # 4. 转换到 CBF 需要的尺度 [cite: 85]
        cbf_input = 2.0 * ((phys_state - self.cbf_mins) / (self.cbf_maxs - self.cbf_mins)) - 1.0
        cbf_input = torch.clamp(cbf_input, -5.0, 5.0)
        
        # 5. 送入冻结的 CBF 网络求 h_val [cite: 208]
        h_val = self.cbf_net(cbf_input) 
        
        # 6. 计算违规程度 (参照 PDF 的 violation 公式) 
        # violation = torch.relu(CONFIG['safety_threshold'] - h_val) 
        # loss_barrier = torch.mean(violation) # [cite: 213]
        
        # return loss_diff, loss_barrier
        # # 6. 计算违规程度 (参照 PDF 的 violation 公式)
        # violation = torch.relu(CONFIG['safety_threshold'] - h_val) 
        
        # # 🔥【核心修复：时间步信任衰减】
        # # coef1 就是 sqrt(alphas_cumprod)，所以 (coef1 ** 2) 就是 alpha_bar。
        # # 当 t 很大时（全是噪声），alpha_bar 趋近于 0，自动屏蔽掉 CBF 对乱码的疯狂惩罚！
        # # 当 t 很小时（轨迹成型），alpha_bar 趋近于 1，CBF 正常发挥拉格朗日约束作用！
        # alpha_bar = coef1 ** 2
        # weighted_violation = violation * alpha_bar
        
        # loss_barrier = torch.mean(weighted_violation)
        
        # return loss_diff, loss_barrier

        # 6. 计算违规程度
        violation = torch.relu(CONFIG['safety_threshold'] - h_val) 
        
        mask = (t < 30).float().view(-1, 1, 1)
        
        weighted_violation = violation * mask
        
        # 只有被 mask 选中的有效样本才参与求均值计算，防止被 0 稀释
        if mask.sum() > 0:
            loss_barrier = weighted_violation.sum() / mask.sum()
        else:
            loss_barrier = torch.tensor(0.0, device=device)
        
        return loss_diff, loss_barrier
        

# =================================================================
# 4. 主训练循环 (参照 PDF 的手动乘子更新逻辑)
# =================================================================
def train():
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    device = torch.device(CONFIG['device'])
    
    dataset = TrajectoryDataset(CONFIG['dataset_path'], horizon=CONFIG['horizon'])
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
    np.savez(os.path.join(CONFIG['save_dir'], 'normalization.npz'), mins=dataset.mins, maxs=dataset.maxs)
    
    # 加载 CBF
    print(f"🔄 加载 CBF 模型...")
    cbf_norm = np.load(CONFIG['cbf_norm_path'])
    cbf_net = CBFNetwork(CONFIG['obs_dim'], 256).to(device)
    cbf_net.load_state_dict(torch.load(CONFIG['cbf_model_path'], map_location=device))
    cbf_net.eval()
    for param in cbf_net.parameters():
        param.requires_grad = False # 冻结网络 [cite: 231]
        
    unet = TemporalUnet(transition_dim=CONFIG['obs_dim'] + CONFIG['act_dim'], dim=CONFIG['hidden_dim']).to(device)
    diffusion = GaussianDiffusionCBF(
        model=unet, cbf_net=cbf_net, 
        diff_mins=dataset.mins, diff_maxs=dataset.maxs,
        cbf_mins=cbf_norm['mins'], cbf_maxs=cbf_norm['maxs']
    ).to(device)
    
    # 🔥 参照 PDF: Lagrangian setup (直接定义 Tensor，不用优化器) [cite: 226]
    nu = torch.tensor(0.01, device=device, requires_grad=False) 
    
    opt_unet = torch.optim.Adam(unet.parameters(), lr=CONFIG['diff_lr'])
    
    loss_history, barrier_history, nu_history = [], [], []
    
    print(f"\n🚀 开始纯正版 PDF-Lagrangian CBF-Diffuser 训练...")
    step = 0
    
    while step < CONFIG['train_steps']:
        for batch_traj in dataloader:
            batch_traj = batch_traj.to(device)
            
            # 1. 计算双 Loss [cite: 240]
            loss_diff, loss_barrier = diffusion.p_losses(batch_traj)
            
            # 2. 联合 Loss: L_total = L_diff + nu * L_barrier [cite: 152]
            loss = loss_diff + nu * loss_barrier 
            
            # 3. 更新模型
            opt_unet.zero_grad()
            loss.backward()
            opt_unet.step()
            
            # 4. 🔥 参照 PDF: 纯手工更新拉格朗日乘子 nu 
            # 没有动量，只有朴实无华的梯度上升，绝不撕裂梯度！
            nu.data += CONFIG['nu_lr'] * loss_barrier.item()
            nu.data = torch.max(nu.data, torch.tensor(1e-5, device=device)) # 
            
            loss_history.append(loss_diff.item())
            barrier_history.append(loss_barrier.item())
            nu_history.append(nu.item())
            step += 1
            
            if step % 500 == 0:
                print(f"Step {step:05d} | Diff MSE: {loss_diff.item():.4f} | CBF Barrier: {loss_barrier.item():.4f} | Nu (λ): {nu.item():.4f}")
                
            if step % 10000 == 0 or step == CONFIG['train_steps']:
                torch.save(unet.state_dict(), os.path.join(CONFIG['save_dir'], f'unet_step_{step}.pt'))
                
            if step >= CONFIG['train_steps']: break

    # 保存图表
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.plot(loss_history, alpha=0.6)
    plt.title('Diffuser MSE Loss')
    plt.subplot(1, 3, 2)
    plt.plot(barrier_history, color='red', alpha=0.6)
    plt.title('CBF Barrier Loss')
    plt.subplot(1, 3, 3)
    plt.plot(nu_history, color='green', alpha=0.6)
    plt.title('Lagrangian Multiplier $\\nu$')
    plt.savefig(os.path.join(CONFIG['save_dir'], 'cbf_pdf_training_logs.png'))
    plt.close()
    
    print(f"🎉 训练完成！模型保存在: {CONFIG['save_dir']}")

if __name__ == '__main__':
    train()