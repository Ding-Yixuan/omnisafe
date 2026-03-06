import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader

# =================================================================
# 0. 引入 Action-Value CBF 网络结构
# =================================================================
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
# 1. 配置参数 (🔥 新增: CBF 和 拉格朗日超参数)
# =================================================================
CONFIG = {
    'dataset_path': './data_pro/ppolag_cost10_combined.npz',
    'obs_horizon': 2,       
    'pred_horizon': 16,     
    'obs_dim': 26,          
    'act_dim': 2,           
    'hidden_dim': 256,      
    'train_steps': 50000,  
    'batch_size': 256,     
    'lr': 2e-4,             
    'device': 'cuda:0',     
    'save_dir': './diffuser_models/cost10_safe', # 换个新名字
    
    # 🔥 新增: CBF 相关的路径
    'cbf_model_path': './看cbf数据/混合数据集_ActionCBF/best_cbf_model.pt',
    'cbf_norm_path': './看cbf数据/混合数据集_ActionCBF/cbf_normalization.npz',
    
    # 🔥 新增: 拉格朗日参数
    'lambda_lr': 5e-4,       # lambda 的更新步长
    'cbf_margin': 0.0,       # 安全裕度 (希望 cbf_val > 0.1)
    'warmup_steps': 15000     # 👈 核心: 前 5000 步纯学动作，不加安全约束！
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
# 4. 扩散过程管理 (🔥 核心修改: 引入 CBF 计算梯度的 x_0 Trick)
# =================================================================
class GaussianDiffusion(nn.Module):
    def __init__(self, model, cbf_model, c_mins, c_maxs, d_mins, d_maxs, n_timesteps=1000):
        super().__init__()
        self.model = model
        self.n_timesteps = n_timesteps
        
        # 冻结 CBF 模型，不更新它的梯度
        self.cbf_model = cbf_model
        for param in self.cbf_model.parameters():
            param.requires_grad = False
            
        # 保存两套极值用于归一化转换
        self.c_mins = c_mins
        self.c_maxs = c_maxs
        self.d_mins = d_mins
        self.d_maxs = d_maxs

        betas = torch.linspace(1e-4, 2e-2, n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

    # 🔥 史诗级修改: 算 MSE 的同时，推演预测动作算 CBF 惩罚！
    def compute_loss(self, obs_cond, act_target, lagrange_lambda, step):
        batch_size = act_target.shape[0]
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=act_target.device).long()
        noise = torch.randn_like(act_target)
        
        coef1 = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1)
        coef2 = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1)
        
        x_t = coef1 * act_target + coef2 * noise
        
        # 1. U-Net 预测噪声
        x_t_in = x_t.permute(0, 2, 1) 
        noise_pred = self.model(x_t_in, t, obs_cond).permute(0, 2, 1) 
        
        # 基础损失: 预测噪声 vs 真实噪声
        loss_mse = nn.functional.mse_loss(noise_pred, noise)
        
        # 如果还在 Warm-up 阶段，直接返回 MSE，不搞惩罚
        if step < CONFIG['warmup_steps']:
            return loss_mse, 0.0, 0.0

        # ==========================================
        # 🔥 The x_0 Trick: 算出模型当前认为的“干净动作”
        # 公式: x_0 = (x_t - sqrt(1-alpha_bar) * epsilon) / sqrt(alpha_bar)
        # 绝对不能加 detach()，梯度就是通过 x_0_pred 流回 U-Net 的！
        # ==========================================
        x_0_pred = (x_t - coef2 * noise_pred) / coef1 
        
        # 我们只约束第一步的动作 [Batch, 2]
        act_step0 = x_0_pred[:, 0, :] 
        
        # 拿到当前的观测状态 [Batch, 26] (取 obs_horizon 的最新一帧)
        obs_curr = obs_cond[:, -1, :] 
        
        # --- 维度与归一化对齐大乱炖 ---
        # 1. 先把 Diffuser [-1, 1] 的数值还原成物理世界的真实数值
        obs_raw = (obs_curr + 1) / 2 * (self.d_maxs[:26] - self.d_mins[:26]) + self.d_mins[:26]
        act_raw = (act_step0 + 1) / 2 * (self.d_maxs[26:] - self.d_mins[26:]) + self.d_mins[26:]
        
        # 2. 拼接成 28 维
        raw_inputs = torch.cat([obs_raw, act_raw], dim=-1)
        
        # 3. 按照 CBF 的规矩，重新归一化到 [-5, 5]
        cbf_inputs_norm = (raw_inputs - self.c_mins) / (self.c_maxs - self.c_mins)
        cbf_inputs_norm = 2 * cbf_inputs_norm - 1
        cbf_inputs_norm = torch.clamp(cbf_inputs_norm, -5.0, 5.0)
        
        # ==========================================
        # 🔥 计算 CBF 安全分与拉格朗日违规项
        # ==========================================
        safety_scores = self.cbf_model(cbf_inputs_norm) # [Batch, 1]
        
        # 违规程度: 当 score < margin 时，产生正向惩罚；否则惩罚为 0
        violation = torch.relu(CONFIG['cbf_margin'] - safety_scores)
        
        # 违规均值 (用于拉格朗日更新)
        mean_violation = torch.mean(violation)
        
        # 最终拉格朗日 Loss 项
        loss_cbf = lagrange_lambda * mean_violation
        
        loss_total = loss_mse + loss_cbf
        
        return loss_total, loss_cbf.item(), mean_violation.item()

# =================================================================
# 5. 主训练循环 (🔥 新增: 拉格朗日参数 Dual Gradient Descent)
# =================================================================
def train():
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    device = torch.device(CONFIG['device'])
    
    # 1. 准备 Diffuser 数据集
    dataset = TrajectoryDataset(
        CONFIG['dataset_path'], 
        obs_horizon=CONFIG['obs_horizon'], 
        pred_horizon=CONFIG['pred_horizon']
    )
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    
    # 获取 Diffuser 的极值
    d_mins = torch.from_numpy(dataset.mins).float().to(device)
    d_maxs = torch.from_numpy(dataset.maxs).float().to(device)
    np.savez(os.path.join(CONFIG['save_dir'], 'normalization.npz'), mins=dataset.mins, maxs=dataset.maxs)

    # 2. 🔥 加载预训练的 CBF 模型与它的极值
    print("🔄 Loading pretrained Action-Value CBF...")
    cbf_model = CBFNetwork(obs_dim=26, act_dim=2).to(device)
    cbf_model.load_state_dict(torch.load(CONFIG['cbf_model_path'], map_location=device))
    cbf_model.eval() # 锁死为推理模式
    
    cbf_norm_data = np.load(CONFIG['cbf_norm_path'])
    c_mins = torch.from_numpy(cbf_norm_data['mins']).float().to(device)
    c_maxs = torch.from_numpy(cbf_norm_data['maxs']).float().to(device)

    # 3. 初始化 U-Net 和 扩散过程
    unet = TemporalUnet(
        act_dim=CONFIG['act_dim'], 
        obs_dim=CONFIG['obs_dim'], 
        obs_horizon=CONFIG['obs_horizon'], 
        dim=CONFIG['hidden_dim']
    ).to(device)
    
    diffusion = GaussianDiffusion(unet, cbf_model, c_mins, c_maxs, d_mins, d_maxs).to(device)
    optimizer = torch.optim.Adam(unet.parameters(), lr=CONFIG['lr'])
    
    # 🔥 初始化拉格朗日乘子 (Lambda)
    lagrange_lambda = 0.0
    
    print(f"🚀 Start Training Safe Conditioned Diffuser... Steps: {CONFIG['train_steps']}")
    
    step = 0
    history = {'total_loss': [], 'cbf_loss': [], 'lambda': [], 'violation': []}
    
    while step < CONFIG['train_steps']:
        for obs_batch, act_batch in dataloader:
            obs_batch = obs_batch.to(device) 
            act_batch = act_batch.to(device) 
            
            # 🔥 传进 lambda 和 step，算出带约束的 Loss
            loss_total, loss_cbf_item, mean_violation = diffusion.compute_loss(
                obs_cond=obs_batch, 
                act_target=act_batch,
                lagrange_lambda=lagrange_lambda,
                step=step
            )
            
            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            
            # 🔥 拉格朗日对偶更新 (Dual Ascent): 违规越多，惩罚权重越大
            if step >= CONFIG['warmup_steps']:
                # lagrange_lambda = max(0.0, lagrange_lambda + CONFIG['lambda_lr'] * mean_violation)
                # lagrange_lambda = min(2.0, max(0.0, lagrange_lambda * 0.999 + CONFIG['lambda_lr'] * mean_violation))
                lagrange_lambda = min(5.0, max(0.0, lagrange_lambda * 0.999 + CONFIG['lambda_lr'] * mean_violation))
            
            step += 1
            
            # 记录数据
            history['total_loss'].append(loss_total.item())
            history['cbf_loss'].append(loss_cbf_item)
            history['lambda'].append(lagrange_lambda)
            history['violation'].append(mean_violation)
            
            if step % 100 == 0:
                print(f"Step {step:5d} | Total: {loss_total.item():.4f} | CBF Loss: {loss_cbf_item:.4f} | Viol: {mean_violation:.4f} | Lambda: {lagrange_lambda:.4f}")
                
            if step % 5000 == 0:
                torch.save(unet.state_dict(), os.path.join(CONFIG['save_dir'], f'unet_step_{step}.pt'))
                
                # 画两张图：Loss 图 和 Lambda-违规 监控图
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 2, 1)
                plt.plot(history['total_loss'], label='Total Loss', alpha=0.7)
                plt.plot(history['cbf_loss'], label='CBF Loss', alpha=0.7)
                plt.title("Training Losses")
                plt.legend()
                
                plt.subplot(1, 2, 2)
                plt.plot(history['lambda'], label='Lagrange Lambda', color='red')
                plt.twinx()
                plt.plot(history['violation'], label='Mean Violation', color='orange', alpha=0.5)
                plt.title("Lambda vs Violation")
                plt.savefig(os.path.join(CONFIG['save_dir'], 'safe_training_curve.png'))
                plt.close()
                
            if step >= CONFIG['train_steps']:
                break

    print("🎉 Safe Training Finished!")

if __name__ == '__main__':
    train()