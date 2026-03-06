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
    # 👈 修改：变成一个包含两个数据集路径的列表
    'dataset_paths': [
        './data_pro/ppolag_cost5_combined.npz',
        './data_pro/ppolag_cost0_combined.npz'
    ],  
    'obs_dim': 26,
    'act_dim': 2,            
    'hidden_dim': 256,
    'lr': 1e-3,              
    'batch_size': 256,       
    'train_steps': 20000,    
    'eval_freq': 500,        
    'device': 'cuda:0',
    'save_dir': './看cbf数据/混合数据集_ActionCBF' 
}

# =================================================================
# 2. CBF 网络定义 (保持不变)
# =================================================================
class CBFNetwork(nn.Module):
    # 👈 1. 这里加上 act_dim
    def __init__(self, obs_dim, act_dim, hidden_dim=256): 
        super().__init__()
        self.net = nn.Sequential(
            # 👈 2. 这里大门拓宽：obs_dim + act_dim (26 + 2 = 28)
            nn.Linear(obs_dim + act_dim, hidden_dim), 
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
    def __init__(self, data_paths, device): # 👈 注意参数变成了 data_paths (列表)
        
        all_obs, all_act, all_lbl = [], [], []
        
        # 1. 遍历读取所有数据集并收集
        for path in data_paths:
            print(f"📂 Loading data from {path}...")
            raw_data = np.load(path)
            all_obs.append(raw_data['obs'])
            all_act.append(raw_data['act'])
            all_lbl.append(raw_data['is_safe'])
            
        # 2. 暴力拼接！把 PPOLag 和 Diffuser 的数据融为一体
        full_obs_np = np.concatenate(all_obs, axis=0)
        full_act_np = np.concatenate(all_act, axis=0)
        full_lbl_np = np.concatenate(all_lbl, axis=0)
        
        print(f"🌍 Mixed Dataset Total Size: {len(full_lbl_np)} samples")
        
        # 3. 转换为 Tensor 并放到 GPU
        raw_obs = torch.from_numpy(full_obs_np).float().to(device) # [N, 26]
        raw_lbl = torch.from_numpy(full_lbl_np).float().to(device) # [N]
        raw_act = torch.from_numpy(full_act_np).float().to(device) # [N, 16, 2] 或 [N, 2]
        
        # 4. 提取第 0 步动作 (如果 act 是序列的话)
        if raw_act.dim() == 3:
            raw_act_step0 = raw_act[:, 0, :] # 变成 [N, 2]
        else:
            raw_act_step0 = raw_act
            
        # 5. 🔥 核心：拼接状态和动作，变成 28 维的 Action-Value 特征！
        full_inputs = torch.cat([raw_obs, raw_act_step0], dim=-1) # [N, 28]
        
        # 6. 划分训练集和验证集 (前 90% 训练，后 10% 验证)
        total_len = len(raw_lbl)
        split_idx = int(total_len * 0.9)
        
        # ⚠️ 注意这里全部改名叫 inputs 了，代表 obs+act
        self.train_inputs = full_inputs[:split_idx] 
        self.train_lbl = raw_lbl[:split_idx]
        
        self.val_inputs = full_inputs[split_idx:]
        self.val_lbl = raw_lbl[split_idx:]
            
        # 7. 平衡采样逻辑 (保持不变，只改变量名)
        self.safe_indices = (self.train_lbl == 1).nonzero(as_tuple=True)[0]
        self.unsafe_indices = (self.train_lbl == 0).nonzero(as_tuple=True)[0]
        
        print(f"📊 Training Stats (After Mix):")
        print(f"   - Safe samples: {len(self.safe_indices)}")
        print(f"   - Unsafe samples: {len(self.unsafe_indices)}")
        
        if len(self.unsafe_indices) == 0:
            raise ValueError("❌ 训练集中没有不安全样本！无法训练有效边界！")

    # 👇 下面的 get_train_batch 和 get_val_batch 里
    # 记得把所有的 self.train_obs 替换成 self.train_inputs 即可！
    def get_train_batch(self, batch_size):
        """ 强制 50% Safe, 50% Unsafe """
        half = batch_size // 2
        
        # 随机采样 Safe
        idx_safe = self.safe_indices[torch.randint(0, len(self.safe_indices), (half,))]
        
        # 随机采样 Unsafe
        idx_unsafe = self.unsafe_indices[torch.randint(0, len(self.unsafe_indices), (half,))]
        
        # 👈 修改：把 train_obs 改成 train_inputs
        batch_inputs = torch.cat([self.train_inputs[idx_safe], self.train_inputs[idx_unsafe]])
        batch_labels = torch.cat([self.train_lbl[idx_safe], self.train_lbl[idx_unsafe]])
        
        return batch_inputs, batch_labels

    def get_val_batch(self, batch_size):
        """ 验证集不需要平衡，随机取即可 """
        idx = torch.randint(0, len(self.val_lbl), (batch_size,))
        
        # 👈 修改：把 val_obs 改成 val_inputs
        return self.val_inputs[idx], self.val_lbl[idx]

# =================================================================
# 4. 主训练循环
# =================================================================
def train():
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    device = torch.device(CONFIG['device'])
    
    # 1. 准备数据
    dataset = BalancedCBFDataset(CONFIG['dataset_paths'], device)
    
    # 2. 计算并应用归一化 (使用训练集统计量)
    # 注意：我们要保存这个 normalization 参数，画图时必须要用！
# 2. 计算并应用归一化 (使用训练集统计量)
    # 把 train_obs 全部改成 train_inputs
    inputs_cpu = dataset.train_inputs.cpu().numpy() 
    mins = torch.from_numpy(inputs_cpu.min(axis=0)).to(device)
    maxs = torch.from_numpy(inputs_cpu.max(axis=0)).to(device)
    maxs[maxs == mins] += 1.0 # 防除零
    
    # 保存归一化参数 (名字也可以顺便改成 cbf_normalization.npz)
    np.savez(os.path.join(CONFIG['save_dir'], 'cbf_normalization.npz'), 
             mins=mins.cpu().numpy(), maxs=maxs.cpu().numpy())
    print("✅ Normalization params saved.")
    
    # In-place 归一化函数保持不变
    def normalize_tensor(tensor):
        normed = (tensor - mins) / (maxs - mins)
        normed = 2 * normed - 1
        return torch.clamp(normed, -5.0, 5.0) # Clip 掉异常值
    
    # 把 train_obs 和 val_obs 改成 train_inputs 和 val_inputs
    dataset.train_inputs = normalize_tensor(dataset.train_inputs)
    dataset.val_inputs = normalize_tensor(dataset.val_inputs)
    print("✅ Data Normalized.")

    # 3. 模型与优化器
    cbf_net = CBFNetwork(CONFIG['obs_dim'], CONFIG['act_dim'], CONFIG['hidden_dim']).to(device)
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