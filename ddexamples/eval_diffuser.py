import os
# 1. 强制使用 EGL 后端 (解决服务器无显示器报错)
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

# =================================================================
# 1. 环境定义 (Monkey Patch) - 必须与训练完全一致！
# =================================================================
def patched_init(self, config):
    self.lidar_num_bins = 16
    self.lidar_max_dist = 3.0
    self.sensors_obs = ['accelerometer', 'velocimeter', 'gyro', 'magnetometer']
    self.task_name = 'GoalLevel1_Reproduction'
    config.update({'lidar_num_bins': 16, 'lidar_max_dist': 3.0, 'sensors_obs': self.sensors_obs, 'task_name': self.task_name})
    GoalLevel0.__init__(self, config=config)
    # 扩大一点地图范围，防止刷在墙外
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
    # ✅ 必须确认：训练时用的是 np.exp 还是 np.abs？
    # 如果你最后用 np.exp 训练的数据，这里必须是 np.exp
    dist = np.exp(-np.abs(z)) 
    angle = np.angle(z)
    goal_vec = np.array([dist, np.cos(angle), np.sin(angle)])
    return np.concatenate([sensor_vec, goal_vec, lidar_vec]).astype(np.float32)

GoalLevel1.__init__ = patched_init
GoalLevel1.obs = patched_obs

# =================================================================
# 2. 模型架构 (必须与训练一致)
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
        x = self.final_conv(x)
        return x

# =================================================================
# 3. 扩散采样器 (修复转圈问题)
# =================================================================
class DiffusionSampler:
    def __init__(self, model, normalization_path, device='cuda:0', horizon=64, obs_dim=26, act_dim=2):
        self.model = model
        self.device = device
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_timesteps = 100
        
        # 加载归一化参数
        norm_data = np.load(normalization_path)
        self.mins = torch.from_numpy(norm_data['mins']).to(device)
        self.maxs = torch.from_numpy(norm_data['maxs']).to(device)
        
        # 预计算
        betas = torch.linspace(1e-4, 2e-2, self.n_timesteps).to(device)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        self.posterior_variance = betas * (1. - torch.cat([torch.tensor([1.0]).to(device), alphas_cumprod[:-1]])) / (1. - alphas_cumprod)

    def normalize(self, x):
        x_norm = (x - self.mins) / (self.maxs - self.mins)
        x_norm = 2 * x_norm - 1
        # 🔥【关键修复】强制 Clip 到 [-1, 1]，防止 OOD 导致转圈
        return torch.clamp(x_norm, -1.0, 1.0)

    def unnormalize(self, x):
        x_01 = (x + 1) / 2
        return x_01 * (self.maxs - self.mins) + self.mins

    @torch.no_grad()
    def sample(self, current_obs):
        batch_size = 1
        shape = (batch_size, self.horizon, self.obs_dim + self.act_dim)
        
        # 1. 准备当前观测
        curr_obs_tensor = torch.from_numpy(current_obs).float().to(self.device)
        dummy_input = torch.zeros(self.obs_dim + self.act_dim).to(self.device)
        dummy_input[:self.obs_dim] = curr_obs_tensor
        
        # 归一化 + Clip
        norm_start = self.normalize(dummy_input)[:self.obs_dim]

        # 2. 从噪声开始
        x = torch.randn(shape, device=self.device)
        
        # 3. 去噪循环
        for i in reversed(range(0, self.n_timesteps)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
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
            
            # 🔥 Inpainting: 强制修正当前状态
            x[:, 0, :self.obs_dim] = norm_start

        traj = self.unnormalize(x)
        action = traj[0, 0, self.obs_dim:] 
        return action.cpu().numpy()

# =================================================================
# 4. 主程序 (上帝视角 + 视频保存)
# =================================================================
if __name__ == '__main__':
    # 配置
    MODEL_PATH = './看loss曲线/ppolag_256/diffuser_step_50000.pt'
    NORM_PATH = './看loss曲线/ppolag_256/normalization.npz'
    VIDEO_PATH = './看loss曲线/ppolag_256/diffuser_godview.mp4'
    
    device = 'cuda:0'
    
    # 1. 加载模型
    print("正在加载模型...")
    model = TemporalUnet(transition_dim=28, dim=256).to(device)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("✅ 模型加载成功")
    else:
        print(f"❌ 找不到模型: {MODEL_PATH}")
        exit()
        
    sampler = DiffusionSampler(model, NORM_PATH, device=device)

    # 2. 环境初始化 (上帝视角)
    print("正在初始化环境 (God View)...")
    # 🔥🔥🔥 关键修改：camera_name='fixedfar' (上帝视角)
    # 🔥🔥🔥 增大分辨率 width=1024, height=1024 让视频更清楚
    env = safety_gymnasium.make('SafetyPointGoal1-v0', 
                                render_mode='rgb_array', 
                                camera_name='fixedfar',  # 👈 上帝视角
                                width=1024, 
                                height=1024)
    
    print(f"🚀 开始 Diffuser 控制测试...")
    obs, _ = env.reset()
    frames = []
    
    try:
        for step in range(1000): 
            if step % 50 == 0: print(f"Step {step}/500...")

            # A. 规划
            action = sampler.sample(obs)
            
            # 🔍 调试打印：看看是不是一直在转圈 (Point机器人的动作是 [v, omega])
            # 如果 omega (第2维) 很大且 v (第1维) 很小，就是在原地转
            if step % 20 == 0:
                print(f"   Action: v={action[0]:.2f}, w={action[1]:.2f}")

            # B. 执行
            next_obs, reward, cost, done, trunc, info = env.step(action)
            
            # C. 渲染
            frame = env.render()
            frames.append(frame)
            
            obs = next_obs
            
            if done or trunc:
                print(f"✨ Episode Finished at step {step}")
                obs, _ = env.reset()
                
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