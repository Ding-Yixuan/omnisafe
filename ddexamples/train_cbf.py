# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import os
# import matplotlib.pyplot as plt

# # =================================================================
# # 1. 配置参数
# # =================================================================
# CONFIG = {
#     'dataset_path': './data_pro/ppolag_zuida.npz',  # 👈 确保这里指向你最新的 raw 数据
#     'obs_dim': 26,
#     'hidden_dim': 256,
#     'lr': 1e-3,
#     'batch_size': 256,      # 每次训练取 128个安全 + 128个不安全
#     'train_steps': 2000000,   # 训练步数
#     'device': 'cuda:0',
#     'save_dir': './cbf_checkpoints/cbf2'
# }

# # =================================================================
# # 2. CBF 网络定义
# # =================================================================
# class CBFNetwork(nn.Module):
#     def __init__(self, obs_dim, hidden_dim=256):
#         super().__init__()
#         # 一个简单的 MLP，输出 1 维标量
#         self.net = nn.Sequential(
#             nn.Linear(obs_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1) # 输出 h(x)
#         )
        
#     def forward(self, x):
#         return self.net(x)

# # =================================================================
# # 3. 数据集与平衡采样器 (关键！)
# # =================================================================
# class BalancedCBFDataset:
#     def __init__(self, data_path, device):
#         print(f"📂 Loading data from {data_path}...")
#         raw_data = np.load(data_path)
        
#         self.obs = torch.from_numpy(raw_data['obs']).float().to(device)
#         # is_safe: 1=Safe, 0=Unsafe
#         self.labels = torch.from_numpy(raw_data['is_safe']).float().to(device)
        
#         # 分离安全和不安全数据的索引
#         # 注意: TTC > Threshold 是安全 (1), 否则是不安全 (0)
#         self.safe_indices = (self.labels == 1).nonzero(as_tuple=True)[0]
#         self.unsafe_indices = (self.labels == 0).nonzero(as_tuple=True)[0]
        
#         print(f"📊 Data Statistics:")
#         print(f"   - Total: {len(self.labels)}")
#         print(f"   - Safe samples: {len(self.safe_indices)}")
#         print(f"   - Unsafe samples: {len(self.unsafe_indices)}")
        
#         if len(self.unsafe_indices) == 0:
#             print("❌ 严重警告：数据集中没有不安全样本！CBF 无法训练边界！")
#             print("💡 建议：在采集时调大 TTC_THRESHOLD (比如 1.5 或 2.0)，或者让机器人稍微'浪'一点。")
            
#     def get_batch(self, batch_size):
#         """ 每次从两堆数据里各取一半 """
#         half_batch = batch_size // 2
        
#         # 随机采样索引
#         idx_safe = self.safe_indices[torch.randint(0, len(self.safe_indices), (half_batch,))]
        
#         # 如果不安全样本太少，允许重复采样
#         idx_unsafe = self.unsafe_indices[torch.randint(0, len(self.unsafe_indices), (half_batch,))]
        
#         batch_obs = torch.cat([self.obs[idx_safe], self.obs[idx_unsafe]])
#         batch_labels = torch.cat([self.labels[idx_safe], self.labels[idx_unsafe]])
        
#         return batch_obs, batch_labels

# # =================================================================
# # 4. 归一化工具 (必须和 Diffuser 保持一致)
# # =================================================================
# # CBF 也需要归一化输入，我们直接计算并在训练前处理
# def normalize_data(obs, mins, maxs):
#     # [0, 1]
#     x_norm = (obs - mins) / (maxs - mins)
#     # [-1, 1]
#     return 2 * x_norm - 1

# # =================================================================
# # 5. 主训练循环
# # =================================================================
# def train():
#     os.makedirs(CONFIG['save_dir'], exist_ok=True)
#     device = torch.device(CONFIG['device'])
    
#     # 1. 准备数据
#     dataset = BalancedCBFDataset(CONFIG['dataset_path'], device)
    
#     # 计算归一化参数 (使用全部数据计算)
#     # 注意：这里我们重新计算一遍，或者直接加载 Diffuser 的 normalization.npz 也可以
#     # 为了独立性，我们这里重新算一遍并保存
#     all_obs_cpu = dataset.obs.cpu().numpy()
#     mins = torch.from_numpy(all_obs_cpu.min(axis=0)).to(device)
#     maxs = torch.from_numpy(all_obs_cpu.max(axis=0)).to(device)
#     # 防止除零
#     maxs[maxs == mins] += 1.0
    
#     np.savez(os.path.join(CONFIG['save_dir'], 'cbf_normalization.npz'), 
#              mins=mins.cpu().numpy(), maxs=maxs.cpu().numpy())
    
#     # 2. 归一化整个数据集 (In-place)
#     dataset.obs = (dataset.obs - mins) / (maxs - mins)
#     dataset.obs = 2 * dataset.obs - 1
#     # 强制 Clip 防止极端值
#     dataset.obs = torch.clamp(dataset.obs, -1.0, 1.0)
    
#     print("✅ Data Normalized and Ready.")

#     # 3. 初始化模型
#     cbf_net = CBFNetwork(CONFIG['obs_dim'], CONFIG['hidden_dim']).to(device)
#     optimizer = optim.Adam(cbf_net.parameters(), lr=CONFIG['lr'])
    
#     # Loss 函数
#     # 我们希望 Safe -> +1, Unsafe -> -1
#     # 所以我们将 label (0, 1) 映射到 (-1, 1)
#     criterion = nn.MSELoss()

#     print(f"🚀 Start Training CBF...")
    
#     loss_history = []
    
#     for step in range(CONFIG['train_steps']):
#         # 获取平衡 Batch
#         batch_obs, batch_labels = dataset.get_batch(CONFIG['batch_size'])
        
#         # 将 Label 从 {0, 1} 转换为 {-1, 1}
#         # 0 -> -1 (Unsafe)
#         # 1 -> +1 (Safe)
#         target_h = 2 * batch_labels - 1
#         target_h = target_h.unsqueeze(1) # [Batch, 1]
        
#         # Forward
#         pred_h = cbf_net(batch_obs)
        
#         # Loss
#         loss = criterion(pred_h, target_h)
        
#         # Backward
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         if step % 100 == 0:
#             loss_history.append(loss.item())
#             print(f"Step {step}/{CONFIG['train_steps']} | Loss: {loss.item():.6f}")
            
#     # 保存模型
#     save_path = os.path.join(CONFIG['save_dir'], 'cbf_model.pt')
#     torch.save(cbf_net.state_dict(), save_path)
#     print(f"💾 CBF Model saved to {save_path}")
    
#     # 画图
#     plt.plot(loss_history)
#     plt.title("CBF Training Loss")
#     plt.savefig(os.path.join(CONFIG['save_dir'], 'cbf_loss.png'))

#     # 简单测试一下
#     print("\n🔬 Testing Prediction:")
#     with torch.no_grad():
#         test_obs, test_lbl = dataset.get_batch(10)
#         preds = cbf_net(test_obs)
#         for i in range(10):
#             gt = "Safe (+1)" if test_lbl[i] > 0.5 else "Unsafe (-1)"
#             print(f"   GT: {gt} | Pred: {preds[i].item():.4f}")

# if __name__ == '__main__':
#     train()


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
    'lr': 3e-4,              # 稍微调小一点，更稳定
    'batch_size': 256,       # 128 Safe + 128 Unsafe
    'train_steps': 30000,    # 👈 修正：3万步足够了 (约150个Epoch)
    'eval_freq': 1000,       # 每1000步验证一次
    'device': 'cuda:0',
    'save_dir': './看cbf数据/ppolag_测试data'
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
# 3. 数据集 (增加了 Train/Val 划分)
# =================================================================
class BalancedCBFDataset:
    def __init__(self, data_path, device, mode='train', val_ratio=0.1):
        print(f"📂 Loading data from {data_path}...")
        raw_data = np.load(data_path)
        
        full_obs = torch.from_numpy(raw_data['obs']).float().to(device)
        full_lbl = torch.from_numpy(raw_data['is_safe']).float().to(device)
        
        # 简单的划分逻辑：取后 10% 做验证
        total_len = len(full_lbl)
        split_idx = int(total_len * (1 - val_ratio))
        
        if mode == 'train':
            self.obs = full_obs[:split_idx]
            self.labels = full_lbl[:split_idx]
        else:
            self.obs = full_obs[split_idx:]
            self.labels = full_lbl[split_idx:]
            
        # 分离索引用于平衡采样
        self.safe_indices = (self.labels == 1).nonzero(as_tuple=True)[0]
        self.unsafe_indices = (self.labels == 0).nonzero(as_tuple=True)[0]
        
        print(f"[{mode.upper()}] Safe: {len(self.safe_indices)} | Unsafe: {len(self.unsafe_indices)}")
        
        if len(self.unsafe_indices) == 0 and mode == 'train':
            raise ValueError("❌ 训练集中没有不安全样本！无法训练！")

    def get_batch(self, batch_size):
        half = batch_size // 2
        # 随机采样，允许重复 (with replacement) 以应对样本少的情况
        idx_safe = self.safe_indices[torch.randint(0, len(self.safe_indices), (half,))]
        idx_unsafe = self.unsafe_indices[torch.randint(0, len(self.unsafe_indices), (half,))]
        
        batch_obs = torch.cat([self.obs[idx_safe], self.obs[idx_unsafe]])
        batch_labels = torch.cat([self.labels[idx_safe], self.labels[idx_unsafe]])
        
        return batch_obs, batch_labels

# =================================================================
# 4. 主训练循环
# =================================================================
def train():
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    device = torch.device(CONFIG['device'])
    
    # 1. 准备数据
    train_set = BalancedCBFDataset(CONFIG['dataset_path'], device, mode='train')
    val_set = BalancedCBFDataset(CONFIG['dataset_path'], device, mode='val')
    
    # 2. 计算并应用归一化 (使用训练集统计量)
    # 注意：Validation Set 也要用 Train Set 的均值方差来归一化，严谨！
    obs_cpu = train_set.obs.cpu().numpy()
    mins = torch.from_numpy(obs_cpu.min(axis=0)).to(device)
    maxs = torch.from_numpy(obs_cpu.max(axis=0)).to(device)
    maxs[maxs == mins] += 1.0 # 防除零
    
    # 保存归一化参数供后续使用
    np.savez(os.path.join(CONFIG['save_dir'], 'cbf_normalization.npz'), 
             mins=mins.cpu().numpy(), maxs=maxs.cpu().numpy())
    
    # In-place 归一化函数
    def apply_norm(dataset_obj):
        dataset_obj.obs = (dataset_obj.obs - mins) / (maxs - mins)
        dataset_obj.obs = 2 * dataset_obj.obs - 1
        dataset_obj.obs = torch.clamp(dataset_obj.obs, -1.0, 1.0)
    
    apply_norm(train_set)
    apply_norm(val_set)
    print("✅ Data Normalized.")

    # 3. 模型与优化器
    cbf_net = CBFNetwork(CONFIG['obs_dim'], CONFIG['hidden_dim']).to(device)
    optimizer = optim.Adam(cbf_net.parameters(), lr=CONFIG['lr'])
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    loss_history = []
    val_history = []

    print(f"🚀 Start Training...")
    
    for step in range(CONFIG['train_steps']):
        # --- Training ---
        cbf_net.train()
        batch_obs, batch_labels = train_set.get_batch(CONFIG['batch_size'])
        target_h = 2 * batch_labels - 1 # [0,1] -> [-1,1]
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
                # 验证集也采样一个 Batch 看看 Loss
                val_obs, val_lbl = val_set.get_batch(CONFIG['batch_size'])
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