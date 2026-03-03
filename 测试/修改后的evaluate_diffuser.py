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
from collections import deque # 👈 新增：用于 RHC 历史观测缓冲

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
# 2. 模型架构 (🔥 修改：切换为纯动作预测、观测条件的 U-Net)
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

class TemporalUnet(nn.Module):
    def __init__(self, act_dim=2, obs_dim=26, obs_horizon=2, dim=256):
        super().__init__()
        
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
            nn.Linear(dim * 4, dim)
        )
        
        self.down1 = nn.Sequential(nn.Conv1d(act_dim, dim, 3, padding=1), nn.Mish())
        self.down2 = nn.Sequential(nn.Conv1d(dim, dim * 2, 3, padding=1), nn.Mish())
        self.down3 = nn.Sequential(nn.Conv1d(dim * 2, dim * 4, 3, padding=1), nn.Mish())
        self.up1 = nn.Sequential(nn.Conv1d(dim * 4, dim * 2, 3, padding=1), nn.Mish())
        self.up2 = nn.Sequential(nn.Conv1d(dim * 2, dim, 3, padding=1), nn.Mish())
        self.final_conv = nn.Conv1d(dim, act_dim, 1)

    def forward(self, x, t, obs_cond):
        batch_size = x.shape[0]
        obs_cond_flat = obs_cond.view(batch_size, -1)
        obs_emb = self.obs_mlp(obs_cond_flat).unsqueeze(-1)
        
        t_emb = self.time_mlp(t).unsqueeze(-1)
        cond_emb = t_emb + obs_emb 
        
        x1 = self.down1(x) + cond_emb
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x = self.up1(x3) + x2
        x = self.up2(x) + x1
        x = self.final_conv(x)
        return x

# =================================================================
# 3. 扩散采样器 (🔥 修改：纯动作扩散，移除 Inpainting)
# =================================================================
class DiffusionSampler:
    def __init__(self, model, normalization_path, device='cuda:0', obs_horizon=2, pred_horizon=16, obs_dim=26, act_dim=2):
        self.model = model
        self.device = device
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_timesteps = 100
        
        norm_data = np.load(normalization_path)
        self.obs_mins = torch.from_numpy(norm_data['mins'][:obs_dim]).to(device)
        self.obs_maxs = torch.from_numpy(norm_data['maxs'][:obs_dim]).to(device)
        self.act_mins = torch.from_numpy(norm_data['mins'][obs_dim:]).to(device)
        self.act_maxs = torch.from_numpy(norm_data['maxs'][obs_dim:]).to(device)
        
        betas = torch.linspace(1e-4, 2e-2, self.n_timesteps).to(device)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        self.posterior_variance = betas * (1. - torch.cat([torch.tensor([1.0]).to(device), alphas_cumprod[:-1]])) / (1. - alphas_cumprod)

    @torch.no_grad()
    def sample(self, obs_hist):
        batch_size = 1
        
        obs_tensor = torch.from_numpy(obs_hist).float().to(self.device)
        obs_norm = 2 * (obs_tensor - self.obs_mins) / (self.obs_maxs - self.obs_mins) - 1
        obs_norm = torch.clamp(obs_norm, -1.0, 1.0).unsqueeze(0) 

        shape = (batch_size, self.pred_horizon, self.act_dim)
        x = torch.randn(shape, device=self.device)
        
        for i in reversed(range(0, self.n_timesteps)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            
            x_in = x.permute(0, 2, 1) 
            noise_pred = self.model(x_in, t, obs_norm).permute(0, 2, 1)
            
            beta_t = 1 - (self.sqrt_recip_alphas[i] ** -2)
            coeff = beta_t / self.sqrt_one_minus_alphas_cumprod[i]
            
            mean = self.sqrt_recip_alphas[i] * (x - coeff * noise_pred)
            if i > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(self.posterior_variance[i])
                x = mean + sigma * noise
            else:
                x = mean
            
        x_01 = (x + 1) / 2
        actions = x_01 * (self.act_maxs - self.act_mins) + self.act_mins
        return actions[0].cpu().numpy()

# =================================================================
# 4. 批量评估循环 (🔥 修改：应用 RHC 收缩视界，处理 Reward 和 Cost)
# =================================================================
def evaluate_diffuser(model_path, norm_path, label_name, num_episodes=50, device='cuda:0'):
    print(f"\n🚀 开始评估 Diffuser 模型: {label_name}")
    print(f"📂 模型路径: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型文件: {model_path}")
        return None, []

    obs_horizon = 2
    pred_horizon = 16
    exec_horizon = 8

    # 实例化新的条件网络
    model = TemporalUnet(act_dim=2, obs_dim=26, obs_horizon=obs_horizon, dim=256).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    sampler = DiffusionSampler(model, norm_path, device=device, obs_horizon=obs_horizon, pred_horizon=pred_horizon)

    env = safety_gymnasium.make('SafetyPointGoal1-v0')
    
    ep_returns = []
    ep_costs = []
    details_list = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_ret, ep_cost = 0.0, 0.0
        
        # 初始化观测缓冲队列
        obs_deque = deque(maxlen=obs_horizon)
        for _ in range(obs_horizon):
            obs_deque.append(obs)
            
        global_step = 0
        done_flag = False

        while global_step < 1000 and not done_flag: 
            # 1. 拿历史 2 步观测去规划未来 16 步动作
            obs_hist = np.array(obs_deque)
            action_seq = sampler.sample(obs_hist)
            
            # 2. 连续执行前 8 步
            for i in range(exec_horizon):
                action = action_seq[i]
                next_obs, reward, cost, done, trunc, info = env.step(action)
                
                # 更新队列并累加指标
                obs_deque.append(next_obs)
                ep_ret += reward
                ep_cost += cost.item() if hasattr(cost, 'item') else cost
                global_step += 1
                
                if done or trunc:
                    done_flag = True
                    break # 跳出 8 步执行循环
                    
            if global_step % 100 < exec_horizon:
                print(f"    - Env step {global_step}/1000...")
                
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
# 5. 主程序：保持你原有的逻辑不变
# =================================================================
if __name__ == '__main__':
    MODELS_TO_EVALUATE = {
        "Diffuser_CBF (PPOLag)": (
            './diffuser_models/新的ppolag/unet_step_50000.pt', 
            './diffuser_models/新的ppolag/normalization.npz'
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

    if all_summaries:
        df_summary = pd.DataFrame(all_summaries)
        df_summary['Reward (Mean ± Std)'] = df_summary.apply(lambda row: f"{row['Reward_Mean']:.2f} ± {row['Reward_Std']:.2f}", axis=1)
        df_summary['Cost (Mean ± Std)'] = df_summary.apply(lambda row: f"{row['Cost_Mean']:.2f} ± {row['Cost_Std']:.2f}", axis=1)
        
        df_summary.to_csv('cbf_diffuser_eval_summarychange4.csv', index=False)
        pd.DataFrame(all_details).to_csv('cbf_diffuser_eval_detailschange4.csv', index=False)
        
        print("\n✅ CBF-Diffuser 模型评估完成！")
        print("💾 最终结果已单独保存至: cbf_diffuser_eval_summarychange4.csv")