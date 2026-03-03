import os
os.environ['MUJOCO_GL'] = 'egl'

import torch
import torch.nn as nn
import numpy as np
import safety_gymnasium
import gymnasium
from safety_gymnasium.assets.geoms import Hazards
from safety_gymnasium.tasks.safe_navigation.goal.goal_level1 import GoalLevel1
from safety_gymnasium.tasks.safe_navigation.goal.goal_level0 import GoalLevel0
import imageio
from collections import deque  # 👈 新增: 用于存储历史观测

# =================================================================
# 1. 环境定义 (Monkey Patch) - 保持不变
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
    z = x + 1j * y
    dist = np.abs(z)
    angle = np.angle(z)
    goal_vec = np.array([dist, np.cos(angle), np.sin(angle)])
    return np.concatenate([sensor_vec, goal_vec, lidar_vec]).astype(np.float32)

GoalLevel1.__init__ = patched_init
GoalLevel1.obs = patched_obs

# =================================================================
# 2. 模型架构 (🔥 修改: 替换为训练时的条件版 U-Net)
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
# 3. 扩散采样器 (🔥 修改: 纯动作生成，观测作为外部条件传入)
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
        
        # 加载归一化参数并分离 obs 和 act 的 mins/maxs
        norm_data = np.load(normalization_path)
        self.obs_mins = torch.from_numpy(norm_data['mins'][:obs_dim]).to(device)
        self.obs_maxs = torch.from_numpy(norm_data['maxs'][:obs_dim]).to(device)
        self.act_mins = torch.from_numpy(norm_data['mins'][obs_dim:]).to(device)
        self.act_maxs = torch.from_numpy(norm_data['maxs'][obs_dim:]).to(device)
        
        # 预计算
        betas = torch.linspace(1e-4, 2e-2, self.n_timesteps).to(device)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        self.posterior_variance = betas * (1. - torch.cat([torch.tensor([1.0]).to(device), alphas_cumprod[:-1]])) / (1. - alphas_cumprod)

    @torch.no_grad()
    def sample(self, obs_hist):
        """
        obs_hist: numpy array, shape [obs_horizon, obs_dim]
        返回: numpy array, shape [pred_horizon, act_dim]
        """
        batch_size = 1
        
        # 1. 准备并归一化条件观测
        obs_tensor = torch.from_numpy(obs_hist).float().to(self.device)
        obs_norm = 2 * (obs_tensor - self.obs_mins) / (self.obs_maxs - self.obs_mins) - 1
        obs_norm = torch.clamp(obs_norm, -1.0, 1.0).unsqueeze(0) # 加上 batch 维度 [1, 2, 26]

        # 2. 纯动作噪声初始化 [1, 16, 2]
        shape = (batch_size, self.pred_horizon, self.act_dim)
        x = torch.randn(shape, device=self.device)
        
        # 3. 去噪循环
        for i in reversed(range(0, self.n_timesteps)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            
            x_in = x.permute(0, 2, 1) # [Batch, Dim, Horizon]
            # 🔥 将 obs_norm 作为条件传给 U-Net
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
            
            # 🔥 删除了强制 Inpainting 修正状态的代码！因为现在生成的就是纯动作！

        # 4. 反归一化动作
        x_01 = (x + 1) / 2
        actions = x_01 * (self.act_maxs - self.act_mins) + self.act_mins
        return actions[0].cpu().numpy() # [pred_horizon, act_dim]

# =================================================================
# 4. 主程序 (🔥 修改: 引入双端队列和 RHC 收缩视界执行)
# =================================================================
if __name__ == '__main__':
    # 配置你的新模型路径
    MODEL_PATH = './diffuser_models/ppolag_exp_cond/unet_step_50000.pt'
    NORM_PATH = './diffuser_models/ppolag_exp_cond/normalization.npz'
    VIDEO_PATH = './diffuser_models/ppolag_exp_cond/diffuser_godview_rhc.mp4'
    
    device = 'cuda:0'
    obs_horizon = 2
    pred_horizon = 16
    exec_horizon = 8  # 👈 RHC: 每次执行前 8 步
    
    print("正在加载模型...")
    model = TemporalUnet(act_dim=2, obs_dim=26, obs_horizon=obs_horizon, dim=256).to(device)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("✅ 模型加载成功")
    else:
        print(f"❌ 找不到模型: {MODEL_PATH}")
        exit()
        
    sampler = DiffusionSampler(model, NORM_PATH, device=device, obs_horizon=obs_horizon, pred_horizon=pred_horizon)

    print("正在初始化环境 (God View)...")
    env = safety_gymnasium.make('SafetyPointGoal1-v0', 
                                render_mode='rgb_array', 
                                camera_name='fixedfar',  
                                width=1024, 
                                height=1024)
    
    print(f"🚀 开始 Diffuser RHC 控制测试...")
    obs, _ = env.reset()
    frames = []
    
    # 👈 新增: 初始化历史观测队列 (存 2 步)
    obs_deque = deque(maxlen=obs_horizon)
    for _ in range(obs_horizon):
        obs_deque.append(obs) # 第一步没有历史，用当前帧填满历史
    
    global_step = 0
    done_flag = False
    
    try:
        while global_step < 1000 and not done_flag: 
            if global_step % 50 == 0: print(f"Global Step {global_step}/1000...")

            # A. 规划 (给模型历史 2 步，吐出未来 16 步)
            obs_hist = np.array(obs_deque)
            action_seq = sampler.sample(obs_hist)
            
            if global_step % 40 == 0:
                print(f"   [Plan] 生成轨迹长度: {len(action_seq)} | 下一步动作: v={action_seq[0, 0]:.2f}, w={action_seq[0, 1]:.2f}")

            # B. 执行 (RHC: 提取前 8 步连续执行)
            for i in range(exec_horizon):
                action = action_seq[i]
                next_obs, reward, cost, done, trunc, info = env.step(action)
                
                # 更新历史队列
                obs_deque.append(next_obs)
                frames.append(env.render())
                
                global_step += 1
                
                # 如果在这 8 步里撞线或者结束了，立刻跳出内层循环
                if done or trunc:
                    print(f"✨ Episode Finished at step {global_step}")
                    done_flag = True
                    break
                
    except KeyboardInterrupt:
        print("手动停止...")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if len(frames) > 0:
            print(f"💾 正在保存视频 ({len(frames)} 帧) -> {VIDEO_PATH} ...")
            imageio.mimsave(VIDEO_PATH, frames, fps=30)
            print("✅ 视频保存完毕！请下载查看。")