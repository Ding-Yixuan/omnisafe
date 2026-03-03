import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import safety_gymnasium
import gymnasium
from safety_gymnasium.assets.geoms import Hazards
from safety_gymnasium.tasks.safe_navigation.goal.goal_level1 import GoalLevel1
from safety_gymnasium.tasks.safe_navigation.goal.goal_level0 import GoalLevel0

# =================================================================
# 1. 环境定义 (严格保持与训练数据一致的 exp 模式)
# =================================================================
def patched_init(self, config):
    self.lidar_num_bins = 16
    self.lidar_max_dist = 3.0
    self.sensors_obs = ['accelerometer', 'velocimeter', 'gyro', 'magnetometer']
    self.task_name = 'GoalLevel1_Reproduction'
    config.update({'lidar_num_bins': 16, 'lidar_max_dist': 3.0, 'sensors_obs': self.sensors_obs, 'task_name': self.task_name})
    GoalLevel0.__init__(self, config=config)
    self.placements_conf.extents = [-1.5, -1.5, 1.5, 1.5]
    self._add_geoms(Hazards(num=2, keepout=0.2))

def patched_obs(self):
    lidar_vec = self._obs_lidar(self.hazards.pos, self.hazards.group)
    acc = self.agent.get_sensor('accelerometer')[:2]
    vel = self.agent.get_sensor('velocimeter')[:2]
    gyro = self.agent.get_sensor('gyro')[-1:]
    mag = self.agent.get_sensor('magnetometer')[:2]
    sensor_vec = np.concatenate([acc, vel, gyro, mag])
    vec = (self.goal.pos - self.agent.pos) @ self.agent.mat
    x, y = vec[0], vec[1]
    
    # 🔥 严格使用 exp，与你的训练数据保持一致
    z = x + 1j * y
    dist = np.exp(-np.abs(z)) 
    angle = np.angle(z)
    goal_vec = np.array([dist, np.cos(angle), np.sin(angle)])
    return np.concatenate([sensor_vec, goal_vec, lidar_vec]).astype(np.float32)

GoalLevel1.__init__ = patched_init
GoalLevel1.obs = patched_obs

# =================================================================
# 2. 模型架构 (换回你的老版简易 U-Net 结构)
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

# 注意：老版本没有 ConvBlock，直接用 nn.Sequential
class TemporalUnet(nn.Module):
    def __init__(self, transition_dim, dim=256): 
        super().__init__()
        # 老版本的 time_mlp 是 4 层
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(dim), nn.Linear(dim, dim * 4), nn.Mish(), nn.Linear(dim * 4, dim))
        
        self.down1 = nn.Sequential(nn.Conv1d(transition_dim, dim, 3, padding=1), nn.Mish())
        self.down2 = nn.Sequential(nn.Conv1d(dim, dim * 2, 3, padding=1), nn.Mish())
        self.down3 = nn.Sequential(nn.Conv1d(dim * 2, dim * 4, 3, padding=1), nn.Mish())
        self.up1 = nn.Sequential(nn.Conv1d(dim * 4, dim * 2, 3, padding=1), nn.Mish())
        self.up2 = nn.Sequential(nn.Conv1d(dim * 2, dim, 3, padding=1), nn.Mish())
        self.final_conv = nn.Conv1d(dim, transition_dim, 1)

    def forward(self, x, t):
        # 老版本只在最外层加了一次时间注入
        t_emb = self.time_mlp(t).unsqueeze(-1)
        x1 = self.down1(x) + t_emb
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x = self.up1(x3) + x2
        x = self.up2(x) + x1
        x = self.final_conv(x)
        return x
    
####################################新版本    

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
    
# =================================================================
# 3. 扩散采样器 (🔥 包含正确的 Inpainting 修复)
# =================================================================
class DiffusionSampler:
    def __init__(self, model, normalization_path, device='cuda:0', horizon=64, obs_dim=26, act_dim=2):
        self.model = model
        self.device = device
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_timesteps = 100
        
        norm_data = np.load(normalization_path)
        self.mins = torch.from_numpy(norm_data['mins']).to(device)
        self.maxs = torch.from_numpy(norm_data['maxs']).to(device)
        
        betas = torch.linspace(1e-4, 2e-2, self.n_timesteps).to(device)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_variance = betas * (1. - torch.cat([torch.tensor([1.0]).to(device), alphas_cumprod[:-1]])) / (1. - alphas_cumprod)

    def normalize(self, x):
        x_norm = (x - self.mins) / (self.maxs - self.mins)
        x_norm = 2 * x_norm - 1
        return torch.clamp(x_norm, -1.0, 1.0)

    def unnormalize(self, x):
        x_01 = (x + 1) / 2
        return x_01 * (self.maxs - self.mins) + self.mins

    @torch.no_grad()
    def sample(self, current_obs):
        batch_size = 1
        shape = (batch_size, self.horizon, self.obs_dim + self.act_dim)
        
        # 1. 准备干净起始观测
        curr_obs_tensor = torch.from_numpy(current_obs).float().to(self.device)
        dummy_input = torch.zeros(self.obs_dim + self.act_dim).to(self.device)
        dummy_input[:self.obs_dim] = curr_obs_tensor
        norm_start = self.normalize(dummy_input)[:self.obs_dim]

        # 2. 从纯噪声开始
        x = torch.randn(shape, device=self.device)
        
        # 3. 去噪循环
        for i in reversed(range(0, self.n_timesteps)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            
            # 🔥 正确的 Inpainting: 加噪覆盖
            noise_t = torch.randn_like(norm_start)
            coef1 = self.sqrt_alphas_cumprod[i]
            coef2 = self.sqrt_one_minus_alphas_cumprod[i]
            noisy_start = coef1 * norm_start + coef2 * noise_t
            x[:, 0, :self.obs_dim] = noisy_start

            x_in = x.permute(0, 2, 1) 
            noise_pred = self.model(x_in, t).permute(0, 2, 1)
            
            beta_t = 1 - (self.sqrt_recip_alphas[i] ** -2)
            coeff = beta_t / self.sqrt_one_minus_alphas_cumprod[i]
            mean = self.sqrt_recip_alphas[i] * (x - coeff * noise_pred)
            
            if i > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(self.posterior_variance[i])
                x = mean + sigma * noise
            else:
                x = mean
                
        # 最后一步，强制注入无损观测
        x[:, 0, :self.obs_dim] = norm_start

        traj = self.unnormalize(x)
        action = traj[0, 0, self.obs_dim:] 
        return action.cpu().numpy()

# =================================================================
# 4. 批量评估循环
# =================================================================
def evaluate_diffuser(model_path, norm_path, label_name, num_episodes=50, device='cuda:0'):
    print(f"\n🚀 开始评估 Diffuser 模型: {label_name}")
    print(f"📂 模型路径: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型文件: {model_path}")
        return None, []

    # 加载模型和采样器
    model = TemporalUnet(transition_dim=28, dim=256).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    sampler = DiffusionSampler(model, norm_path, device=device)

    # 初始化纯净无渲染环境（跑得快）
    env = safety_gymnasium.make('SafetyPointGoal1-v0')
    
    ep_returns = []
    ep_costs = []
    details_list = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_ret, ep_cost = 0.0, 0.0
        
        for step in range(1000): 
            # Diffuser 在线采样（Receding Horizon Control）
            action = sampler.sample(obs)
            next_obs, reward, cost, done, trunc, info = env.step(action)
            
            ep_ret += reward
            ep_cost += cost.item() if hasattr(cost, 'item') else cost
            obs = next_obs
            
            # 由于 Diffuser 采样较慢，打印内部进度防焦虑
            if (step + 1) % 100 == 0:
                print(f"    - Env step {step+1}/1000...")

            if done or trunc:
                break
                
        ep_returns.append(ep_ret)
        ep_costs.append(ep_cost)
        
        details_list.append({
            'Model': f"Diffuser ({label_name})",
            'Episode': ep + 1,
            'Reward': ep_ret,
            'Cost': ep_cost
        })
        
        print(f"  Episode {ep+1:02d}/{num_episodes} | Return: {ep_ret:6.2f} | Cost: {ep_cost:6.2f}")

    mean_ret, std_ret = np.mean(ep_returns), np.std(ep_returns)
    mean_cost, std_cost = np.mean(ep_costs), np.std(ep_costs)

    print("\n" + "="*50)
    print(f"📊 Diffuser ({label_name}) 评估结果 ({num_episodes} Episodes):")
    print(f"  🏆 Reward: {mean_ret:.2f} ± {std_ret:.2f}")
    print(f"  ⚠️ Cost:   {mean_cost:.2f} ± {std_cost:.2f}")
    print("="*50)
    
    summary_data = {
        'Model': f"Diffuser ({label_name})",
        'Episodes': num_episodes,
        'Reward_Mean': mean_ret, 'Reward_Std': std_ret,
        'Cost_Mean': mean_cost, 'Cost_Std': std_cost
    }
    return summary_data, details_list

# =================================================================
# 5. 主程序：跑两组 Diffuser 并存入表格
# # =================================================================
# if __name__ == '__main__':
#     # 🔴🔴🔴 请根据你实际保存的文件名修改下面的路径 🔴🔴🔴
#     # 你的文件可能叫 diffuser_step_50000.pt 或者 unet_step_50000.pt
    
#     MODELS_TO_EVALUATE = {
#         # 格式: "标签名": ("模型路径", "归一化参数路径")
#         # "PPOLag_Expert": (
#         #     './diffuser_models/ppolag_测试data/unet_step_50000.pt', 
#         #     './diffuser_models/ppolag_测试data/normalization.npz'
#         # ),
#         # "PPO_Expert": (
#         #     './diffuser_models/ppo_测试/unet_step_50000.pt', 
#         #     './diffuser_models/ppo_测试/normalization.npz'
#         # )
#         "PPOLag_Expert": (
#             './看loss曲线/ppolag_测试data/diffuser_step_50000.pt', 
#             './看loss曲线/ppolag_测试data/normalization.npz'
#         ),
#         "PPO_Expert": (
#             './看loss曲线/ppo_测试/diffuser_step_50000.pt', 
#             './看loss曲线/ppo_测试/normalization.npz'
#         )
#     }
    
#     NUM_EPISODES = 50 
    
#     all_summaries = []
#     all_details = []
    
#     for label, (model_path, norm_path) in MODELS_TO_EVALUATE.items():
#         summary, details = evaluate_diffuser(model_path, norm_path, label, num_episodes=NUM_EPISODES)
#         if summary:
#             all_summaries.append(summary)
#             all_details.extend(details)

#     # 保存至 CSV，准备写论文！
#     if all_summaries:
#         df_summary = pd.DataFrame(all_summaries)
#         df_summary['Reward (Mean ± Std)'] = df_summary.apply(lambda row: f"{row['Reward_Mean']:.2f} ± {row['Reward_Std']:.2f}", axis=1)
#         df_summary['Cost (Mean ± Std)'] = df_summary.apply(lambda row: f"{row['Cost_Mean']:.2f} ± {row['Cost_Std']:.2f}", axis=1)
        
#         df_summary.to_csv('diffuser_eval_summary.csv', index=False)
#         pd.DataFrame(all_details).to_csv('diffuser_eval_details.csv', index=False)
        
#         print("\n✅ 所有 Diffuser 模型评估完成！")
#         print("💾 最终的均值方差对比表格已保存至: diffuser_eval_summary.csv")


# =================================================================
# 5. 主程序：只评估带 CBF 护盾的 Diffuser
# =================================================================
if __name__ == '__main__':
    # 🔴 这里换成了你刚刚跑完 CBF 训练的保存路径
    MODELS_TO_EVALUATE = {
        "Diffuser_CBF (PPOLag)": (
            './diffuser_models/cbf_diffuser_pdf_2/unet_step_50000.pt', 
            './diffuser_models/cbf_diffuser_pdf_2/normalization.npz'
        )
    }
    
    NUM_EPISODES = 50 
    
    all_summaries = []
    all_details = []
    
    for label, (model_path, norm_path) in MODELS_TO_EVALUATE.items():
        summary, details = evaluate_diffuser(model_path, norm_path, label, num_episodes=NUM_EPISODES)
        if summary:
            all_summaries.append(summary)
            all_details.extend(details)

    # 保存至 CSV
    if all_summaries:
        df_summary = pd.DataFrame(all_summaries)
        df_summary['Reward (Mean ± Std)'] = df_summary.apply(lambda row: f"{row['Reward_Mean']:.2f} ± {row['Reward_Std']:.2f}", axis=1)
        df_summary['Cost (Mean ± Std)'] = df_summary.apply(lambda row: f"{row['Cost_Mean']:.2f} ± {row['Cost_Std']:.2f}", axis=1)
        
        # 换个名字保存，防止覆盖你之前的纯净版表格
        df_summary.to_csv('cbf_diffuser_eval_summarychange2.csv', index=False)
        pd.DataFrame(all_details).to_csv('cbf_diffuser_eval_detailschange2.csv', index=False)
        
        print("\n✅ CBF-Diffuser 模型评估完成！")
        print("💾 最终结果已单独保存至: cbf_diffuser_eval_summarychange2.csv")