import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt

# =================================================================
# 1. 配置参数
# =================================================================
CONFIG = {
    'dataset_path': './data_pro/ppolag_测试data.npz',  # 👈 确保文件名对
    'obs_dim': 26,
    'hidden_dim': 256,
    'lr': 1e-3,              # 稍微调小一点，更稳定
    'batch_size': 256,       # 128 Safe + 128 Unsafe
    'train_steps': 20000,    # 👈 修正：2万步足够了 (约100个Epoch)，不需要200万步！
    'eval_freq': 500,        # 每500步验证一次
    'device': 'cuda:0',
    'save_dir': './看cbf数据/ppolag_测试data2让边界变小'
}

# =================================================================
# 2. CBF 网络定义 (保持不变)
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
# 3. 数据集 (增加了 Train/Val 划分 & 平衡采样)
# =================================================================
class BalancedCBFDataset:
    def __init__(self, data_path, device):
        print(f"📂 Loading data from {data_path}...")
        raw_data = np.load(data_path)
        
        # 加载数据
        full_obs = torch.from_numpy(raw_data['obs']).float().to(device)
        full_lbl = torch.from_numpy(raw_data['is_safe']).float().to(device)
        
        # 简单的划分逻辑：前 90% 训练，后 10% 验证
        total_len = len(full_lbl)
        split_idx = int(total_len * 0.9)
        
        self.train_obs = full_obs[:split_idx]
        self.train_lbl = full_lbl[:split_idx]
        
        self.val_obs = full_obs[split_idx:]
        self.val_lbl = full_lbl[split_idx:]
            
        # --- 关键：分离索引用于平衡采样 (仅针对训练集) ---
        self.safe_indices = (self.train_lbl == 1).nonzero(as_tuple=True)[0]
        self.unsafe_indices = (self.train_lbl == 0).nonzero(as_tuple=True)[0]
        
        print(f"📊 Training Stats:")
        print(f"   - Safe samples: {len(self.safe_indices)}")
        print(f"   - Unsafe samples: {len(self.unsafe_indices)}")
        
        if len(self.unsafe_indices) == 0:
            raise ValueError("❌ 训练集中没有不安全样本！无法训练有效边界！")

    def get_train_batch(self, batch_size):
        """ 强制 50% Safe, 50% Unsafe """
        half = batch_size // 2
        
        # 随机采样 Safe
        idx_safe = self.safe_indices[torch.randint(0, len(self.safe_indices), (half,))]
        
        # 随机采样 Unsafe (允许重复，因为 Unsafe 样本通常很少)
        idx_unsafe = self.unsafe_indices[torch.randint(0, len(self.unsafe_indices), (half,))]
        
        batch_obs = torch.cat([self.train_obs[idx_safe], self.train_obs[idx_unsafe]])
        batch_labels = torch.cat([self.train_lbl[idx_safe], self.train_lbl[idx_unsafe]])
        
        return batch_obs, batch_labels

    def get_val_batch(self, batch_size):
        """ 验证集不需要平衡，随机取即可 """
        idx = torch.randint(0, len(self.val_lbl), (batch_size,))
        return self.val_obs[idx], self.val_lbl[idx]

# =================================================================
# 4. 主训练循环
# =================================================================
def train():
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    device = torch.device(CONFIG['device'])
    
    # 1. 准备数据
    dataset = BalancedCBFDataset(CONFIG['dataset_path'], device)
    
    # 2. 计算并应用归一化 (使用训练集统计量)
    # 注意：我们要保存这个 normalization 参数，画图时必须要用！
    obs_cpu = dataset.train_obs.cpu().numpy()
    mins = torch.from_numpy(obs_cpu.min(axis=0)).to(device)
    maxs = torch.from_numpy(obs_cpu.max(axis=0)).to(device)
    maxs[maxs == mins] += 1.0 # 防除零
    
    # 保存归一化参数
    np.savez(os.path.join(CONFIG['save_dir'], 'cbf_normalization.npz'), 
             mins=mins.cpu().numpy(), maxs=maxs.cpu().numpy())
    print("✅ Normalization params saved.")
    
    # In-place 归一化函数
    def normalize_tensor(tensor):
        normed = (tensor - mins) / (maxs - mins)
        normed = 2 * normed - 1
        return torch.clamp(normed, -5.0, 5.0) # Clip 掉异常值
    
    dataset.train_obs = normalize_tensor(dataset.train_obs)
    dataset.val_obs = normalize_tensor(dataset.val_obs)
    print("✅ Data Normalized.")

    # 3. 模型与优化器
    cbf_net = CBFNetwork(CONFIG['obs_dim'], CONFIG['hidden_dim']).to(device)
    optimizer = optim.Adam(cbf_net.parameters(), lr=CONFIG['lr'])
    criterion = nn.MSELoss() # 回归 Loss (逼近 +1/-1)

    best_val_loss = float('inf')
    loss_history = []
    val_history = []

    print(f"🚀 Start Training CBF...")
    
    for step in range(CONFIG['train_steps']):
        # --- Training ---
        cbf_net.train()
        batch_obs, batch_labels = dataset.get_train_batch(CONFIG['batch_size'])
        
        # Label 映射: 0 -> -1 (Unsafe), 1 -> +1 (Safe)
        target_h = 2 * batch_labels - 1 
        target_h = target_h.unsqueeze(1)
        
        pred_h = cbf_net(batch_obs)
        loss = criterion(pred_h, target_h)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # --- Validation & Logging ---
        if step % CONFIG['eval_freq'] == 0:
            cbf_net.eval()
            with torch.no_grad():
                val_obs, val_lbl = dataset.get_val_batch(CONFIG['batch_size'])
                val_target = 2 * val_lbl - 1
                val_pred = cbf_net(val_obs)
                val_loss = criterion(val_pred, val_target.unsqueeze(1))
                
                loss_history.append(loss.item())
                val_history.append(val_loss.item())
                
                print(f"Step {step:5d} | Train Loss: {loss.item():.5f} | Val Loss: {val_loss.item():.5f}")
                
                # 保存最佳模型
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    torch.save(cbf_net.state_dict(), os.path.join(CONFIG['save_dir'], 'best_cbf_model.pt'))
                    print(f"  🌟 New Best Model Saved! (Val Loss: {best_val_loss:.5f})")

    # 画图
    plt.figure(figsize=(10,5))
    plt.plot(loss_history, label='Train Loss')
    plt.plot(val_history, label='Val Loss')
    plt.legend()
    plt.title("CBF Training Curve")
    plt.savefig(os.path.join(CONFIG['save_dir'], 'cbf_loss_curve.png'))
    print("🏁 Training Finished.")

if __name__ == '__main__':
    train()