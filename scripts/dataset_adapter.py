####################1

# import torch
# import numpy as np
# from torch.utils.data import Dataset
# from collections import namedtuple

# # 定义 Batch 结构
# Batch = namedtuple('Batch', 'trajectories conditions')

# class SafetyGymDataset(Dataset):
#     def __init__(self, data_path, horizon=64, normalizer='LimitsNormalizer'):
#         self.horizon = horizon
        
#         # 加载数据
#         data = np.load(data_path)
        
#         # 确保数据是 float32
#         # self.observations = data['observations'].astype(np.float32)
#         # self.actions = data['actions'].astype(np.float32)
#         raw_obs = data['observations'].astype(np.float32)
#         print(f"Dataset Obs Original Range: Min={raw_obs.min():.2f}, Max={raw_obs.max():.2f}")
        
#         # 强制截断观测值！去除物理引擎错误的尖刺 (通常 +-10 就很大了)
#         self.observations = np.clip(raw_obs, -10.0, 10.0)
#         print(f"Dataset Obs Clipped Range: [-10.0, 10.0]")
        
#         # 2. 处理动作 (Actions) - 【保持之前的修复】
#         raw_actions = data['actions'].astype(np.float32)
#         print(f"Dataset Act Original Range: Min={raw_actions.min():.2f}, Max={raw_actions.max():.2f}")
#         self.actions = np.clip(raw_actions, -1.0, 1.0)
        
#         # 【核心修复 1】添加 Trainer 所需的维度属性
#         self.observation_dim = self.observations.shape[1]
#         self.action_dim = self.actions.shape[1]
        
#         # 【核心修复 2】添加 normalizer 属性指向自己
#         # Trainer 会调用 self.dataset.normalizer.unnormalize()
#         self.normalizer = self
        
#         # 处理路径长度
#         if 'path_lengths' in data:
#             self.path_lengths = data['path_lengths']
#         else:
#             print("⚠️ Warning: path_lengths not found, assuming fixed length 1000")
#             N = len(self.observations)
#             self.path_lengths = [1000] * (N // 1000)
        
#         # 预计算滑动窗口索引
#         self.indices = []
#         ptr = 0
#         for length in self.path_lengths:
#             if length >= horizon:
#                 self.indices.extend(range(ptr, ptr + length - horizon + 1))
#             ptr += length
            
#         print(f"Dataset loaded: {len(self.indices)} sliding windows available.")
        
#         # 初始化归一化参数
#         self.obs_min = self.observations.min(axis=0)
#         self.obs_max = self.observations.max(axis=0)
#         self.act_min = self.actions.min(axis=0)
#         self.act_max = self.actions.max(axis=0)
        
#         # 避免除以 0
#         self.obs_max[self.obs_max == self.obs_min] += 1e-6
#         self.act_max[self.act_max == self.act_min] += 1e-6

#     def normalize(self, x, key):
#         if key == 'observations':
#             return 2 * (x - self.obs_min) / (self.obs_max - self.obs_min) - 1
#         elif key == 'actions':
#             return 2 * (x - self.act_min) / (self.act_max - self.act_min) - 1
#         return x

#     def unnormalize(self, x, key):
#         # Trainer 可能会传入 tensor，先转 numpy
#         if torch.is_tensor(x):
#             x = x.cpu().detach().numpy()
            
#         if key == 'observations':
#             return (x + 1) * (self.obs_max - self.obs_min) / 2 + self.obs_min
#         elif key == 'actions':
#             return (x + 1) * (self.act_max - self.act_min) / 2 + self.act_min
#         return x

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, idx):
#         start = self.indices[idx]
#         end = start + self.horizon
        
#         obs_seq = self.observations[start:end]
#         act_seq = self.actions[start:end]
        
#         norm_obs = self.normalize(obs_seq, 'observations')
#         norm_act = self.normalize(act_seq, 'actions')
        
#         trajectory = np.concatenate([norm_obs, norm_act], axis=-1)
#         condition = {0: norm_obs[0]} 
        
#         return Batch(trajectory, condition)


# #####################2gaussian
# import numpy as np
# import torch
# from collections import namedtuple

# # 1. 定义 Batch 结构 (修复 _fields 报错)
# Batch = namedtuple('Batch', 'trajectories conditions')

# class SafetyGymDataset(torch.utils.data.Dataset):
#     def __init__(self, data_path, horizon=64, normalizer='Gaussian'):
#         self.horizon = horizon
        
#         # --- 加载数据 ---
#         print(f"Loading data from {data_path}")
#         data = np.load(data_path)
        
#         # --- 观测处理 (截断) ---
#         raw_obs = data['observations'].astype(np.float32)
#         print(f"Original Obs Range: [{raw_obs.min():.2f}, {raw_obs.max():.2f}]")
#         self.observations = np.clip(raw_obs, -10.0, 10.0)
        
#         # --- 动作处理 (截断) ---
#         raw_actions = data['actions'].astype(np.float32)
#         print(f"Original Act Range: [{raw_actions.min():.2f}, {raw_actions.max():.2f}]")
#         self.actions = np.clip(raw_actions, -1.0, 1.0)
#         print(f"Clipped Actions to [-1.0, 1.0]")
        
#         # --- 计算高斯统计量 (Mean / Std) ---
#         self.obs_mean = self.observations.mean(axis=0)
#         self.obs_std = self.observations.std(axis=0)
#         self.obs_std[self.obs_std < 1e-6] = 1.0 # 防止除零
        
#         self.act_mean = self.actions.mean(axis=0)
#         self.act_std = self.actions.std(axis=0)
#         self.act_std[self.act_std < 1e-6] = 1.0
        
#         print("✅ Gaussian Normalizer Initialized (Mean/Std)")

#         # 【核心修复 3】 将 normalizer 指向自己，让 Trainer 能找到它
#         self.normalizer = self

#         # --- 归一化数据 ---
#         self.norm_observations = self.normalize(self.observations, 'observations')
#         self.norm_actions = self.normalize(self.actions, 'actions')
        
#         # --- 构建索引 ---
#         if 'path_lengths' in data:
#             self.path_lengths = data['path_lengths']
#         else:
#             self.path_lengths = [len(self.observations)]
            
#         self.indices = []
#         ctr = 0
#         for length in self.path_lengths:
#             if length >= horizon:
#                 self.indices.extend(range(ctr, ctr + length - horizon + 1))
#             ctr += length
            
#         self.observation_dim = self.observations.shape[1]
#         self.action_dim = self.actions.shape[1]

#     def normalize(self, x, key):
#         if key == 'observations':
#             mean, std = self.obs_mean, self.obs_std
#         elif key == 'actions':
#             mean, std = self.act_mean, self.act_std
#         else:
#             raise ValueError(f"Unknown key: {key}")
        
#         if torch.is_tensor(x):
#             mean = torch.tensor(mean, device=x.device, dtype=x.dtype)
#             std = torch.tensor(std, device=x.device, dtype=x.dtype)
        
#         return (x - mean) / std

#     def unnormalize(self, x, key):
#         if key == 'observations':
#             mean, std = self.obs_mean, self.obs_std
#         elif key == 'actions':
#             mean, std = self.act_mean, self.act_std
#         else:
#             raise ValueError(f"Unknown key: {key}")
            
#         if torch.is_tensor(x):
#             mean = torch.tensor(mean, device=x.device, dtype=x.dtype)
#             std = torch.tensor(std, device=x.device, dtype=x.dtype)
            
#         return x * std + mean

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, idx):
#         start_idx = self.indices[idx]
#         end_idx = start_idx + self.horizon
        
#         # 获取窗口片段
#         obs_batch = self.norm_observations[start_idx:end_idx]
#         act_batch = self.norm_actions[start_idx:end_idx]
        
#         # 【核心修复 2】 拼接成 (Horizon, 28) 的轨迹
#         trajectories = np.concatenate([obs_batch, act_batch], axis=-1)
        
#         # 【核心修复 2】 构造条件字典 (Diffuser 期望 cond 是个 dict)
#         conditions = {0: obs_batch[0]}
        
#         return Batch(trajectories, conditions)

# ######################3改成预测动作
# import numpy as np
# import torch
# from collections import namedtuple

# # Batch 结构
# Batch = namedtuple('Batch', 'trajectories conditions')

# class SafetyGymDataset(torch.utils.data.Dataset):
#     def __init__(self, data_path, horizon=64):
#         self.horizon = horizon
        
#         # 1. 加载数据
#         print(f"Loading data from {data_path}")
#         data = np.load(data_path)
        
#         # 2. 读取数据 (强制 float32)
#         self.observations = data['observations'].astype(np.float32)
#         self.actions = data['actions'].astype(np.float32)
        
#         # 3. 【观测处理】使用 Robust Gaussian
#         # 计算分位数 (Numpy 默认返回 float64，必须转 float32)
#         q_low = np.quantile(self.observations, 0.01, axis=0).astype(np.float32)
#         q_high = np.quantile(self.observations, 0.99, axis=0).astype(np.float32)
        
#         # 截断观测值
#         self.observations = np.clip(self.observations, q_low, q_high)
        
#         # 计算均值和方差 (强制 float32)
#         self.obs_mean = self.observations.mean(axis=0).astype(np.float32)
#         self.obs_std = self.observations.std(axis=0).astype(np.float32)
#         self.obs_std[self.obs_std < 1e-2] = 1.0 
        
#         print("✅ Observation Normalizer: Robust Gaussian (float32)")
        
#         # 4. 【动作处理】Min-Max
#         self.act_min = -1.0 
#         self.act_max = 1.0
        
#         print("✅ Action Normalizer: Min-Max [-1, 1]")

#         # 5. 设置 normalizer 引用
#         self.normalizer = self

#         # 6. 归一化数据
#         self.norm_observations = self.normalize(self.observations, 'observations')
#         self.norm_actions = self.normalize(self.actions, 'actions')
        
#         # 7. 构建索引
#         if 'path_lengths' in data:
#             self.path_lengths = data['path_lengths']
#         else:
#             self.path_lengths = [len(self.observations)]
            
#         self.indices = []
#         ctr = 0
#         for length in self.path_lengths:
#             if length >= horizon:
#                 self.indices.extend(range(ctr, ctr + length - horizon + 1))
#             ctr += length
            
#         self.observation_dim = self.observations.shape[1]
#         self.action_dim = self.actions.shape[1]

#     def normalize(self, x, key):
#         """ 混合归一化逻辑 """
#         if key == 'observations':
#             mean, std = self.obs_mean, self.obs_std
#             if torch.is_tensor(x):
#                 # 确保 tensor 也是 float32
#                 mean = torch.tensor(mean, device=x.device, dtype=torch.float32)
#                 std = torch.tensor(std, device=x.device, dtype=torch.float32)
#                 # 如果输入 x 是 double，强制转 float
#                 if x.dtype == torch.float64: x = x.float()
#             return (x - mean) / std
            
#         elif key == 'actions':
#             if torch.is_tensor(x):
#                 if x.dtype == torch.float64: x = x.float()
#                 return torch.clamp(x, -1.0, 1.0)
#             return np.clip(x, -1.0, 1.0)
        
#         else:
#             raise ValueError(f"Unknown key: {key}")

#     def unnormalize(self, x, key):
#         """ 反归一化逻辑 """
#         if key == 'observations':
#             mean, std = self.obs_mean, self.obs_std
#             if torch.is_tensor(x):
#                 mean = torch.tensor(mean, device=x.device, dtype=torch.float32)
#                 std = torch.tensor(std, device=x.device, dtype=torch.float32)
#                 if x.dtype == torch.float64: x = x.float()
#             return x * std + mean
            
#         elif key == 'actions':
#             if torch.is_tensor(x):
#                 if x.dtype == torch.float64: x = x.float()
#                 return torch.clamp(x, -1.0, 1.0)
#             return np.clip(x, -1.0, 1.0)
            
#         else:
#             raise ValueError(f"Unknown key: {key}")

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, idx):
#         start_idx = self.indices[idx]
#         end_idx = start_idx + self.horizon
        
#         obs_batch = self.norm_observations[start_idx:end_idx]
#         act_batch = self.norm_actions[start_idx:end_idx]
        
#         # 【关键】拼接后再次确保是 float32
#         trajectories = np.concatenate([obs_batch, act_batch], axis=-1).astype(np.float32)
        
#         # Condition 也要转 float32
#         conditions = {0: obs_batch[0].astype(np.float32)}
        
#         return Batch(trajectories, conditions)


######################4改goal权重
import numpy as np
import torch
from collections import namedtuple

# Batch 结构
Batch = namedtuple('Batch', 'trajectories conditions')

class SafetyGymDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, horizon=64, goal_weight=1.0):
        self.horizon = horizon
        
        # 1. 加载数据
        print(f"Loading data from {data_path}")
        data = np.load(data_path)
        
        # 2. 读取数据 (强制 float32)
        self.observations = data['observations'].astype(np.float32)
        self.actions = data['actions'].astype(np.float32)
        
        # 3. 【观测处理】使用 Robust Gaussian
        # 计算分位数 (Numpy 默认返回 float64，必须转 float32)
        q_low = np.quantile(self.observations, 0.01, axis=0).astype(np.float32)
        q_high = np.quantile(self.observations, 0.99, axis=0).astype(np.float32)
        
        # 截断观测值
        self.observations = np.clip(self.observations, q_low, q_high)
        
        # 计算均值和方差 (强制 float32)
        self.obs_mean = self.observations.mean(axis=0).astype(np.float32)
        self.obs_std = self.observations.std(axis=0).astype(np.float32)
        self.obs_std[self.obs_std < 1e-2] = 1.0 
        
        print("✅ Observation Normalizer: Robust Gaussian (float32)")
        
        # 4. 【动作处理】Min-Max
        self.act_min = -1.0 
        self.act_max = 1.0
        
        print("✅ Action Normalizer: Min-Max [-1, 1]")

        # 5. 设置 normalizer 引用
        self.normalizer = self

        # 6. 归一化数据
        self.norm_observations = self.normalize(self.observations, 'observations')
        self.norm_actions = self.normalize(self.actions, 'actions')

        if goal_weight != 1.0:
            print(f"🔥 applying goal weight: {goal_weight}x")
            # 注意：这里直接修改 norm_observations
            # 这样模型看到的 Goal 数值会比正常值大 goal_weight 倍
            self.norm_observations[:, 7:10] *= goal_weight
        
        # 7. 构建索引
        if 'path_lengths' in data:
            self.path_lengths = data['path_lengths']
        else:
            self.path_lengths = [len(self.observations)]
            
        self.indices = []
        ctr = 0
        for length in self.path_lengths:
            if length >= horizon:
                self.indices.extend(range(ctr, ctr + length - horizon + 1))
            ctr += length
            
        self.observation_dim = self.observations.shape[1]
        self.action_dim = self.actions.shape[1]

    def normalize(self, x, key):
        """ 混合归一化逻辑 """
        if key == 'observations':
            mean, std = self.obs_mean, self.obs_std
            if torch.is_tensor(x):
                # 确保 tensor 也是 float32
                mean = torch.tensor(mean, device=x.device, dtype=torch.float32)
                std = torch.tensor(std, device=x.device, dtype=torch.float32)
                # 如果输入 x 是 double，强制转 float
                if x.dtype == torch.float64: x = x.float()
            return (x - mean) / std
            
        elif key == 'actions':
            if torch.is_tensor(x):
                if x.dtype == torch.float64: x = x.float()
                return torch.clamp(x, -1.0, 1.0)
            return np.clip(x, -1.0, 1.0)
        
        else:
            raise ValueError(f"Unknown key: {key}")

    def unnormalize(self, x, key):
        """ 反归一化逻辑 """
        if key == 'observations':
            mean, std = self.obs_mean, self.obs_std
            if torch.is_tensor(x):
                mean = torch.tensor(mean, device=x.device, dtype=torch.float32)
                std = torch.tensor(std, device=x.device, dtype=torch.float32)
                if x.dtype == torch.float64: x = x.float()
            return x * std + mean
            
        elif key == 'actions':
            if torch.is_tensor(x):
                if x.dtype == torch.float64: x = x.float()
                return torch.clamp(x, -1.0, 1.0)
            return np.clip(x, -1.0, 1.0)
            
        else:
            raise ValueError(f"Unknown key: {key}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        end_idx = start_idx + self.horizon
        
        obs_batch = self.norm_observations[start_idx:end_idx]
        act_batch = self.norm_actions[start_idx:end_idx]
        
        # 【关键】拼接后再次确保是 float32
        trajectories = np.concatenate([obs_batch, act_batch], axis=-1).astype(np.float32)
        
        # Condition 也要转 float32
        conditions = {0: obs_batch[0].astype(np.float32)}
        
        return Batch(trajectories, conditions)