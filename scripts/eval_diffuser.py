# # import os
# # import torch
# # import numpy as np
# # import omnisafe
# # import imageio
# # import sys
# # import time
# # import pickle
# # import gymnasium
# # from collections import namedtuple
# # # /home/lqz27/anaconda3/envs/omnisafedd/bin/python /home/lqz27/dyx_ws/omnisafe/scripts/eval_diffuser.py
# # # ================= 路径设置 =================
# # current_dir = os.path.dirname(os.path.abspath(__file__))
# # project_root = os.path.dirname(current_dir)
# # if project_root not in sys.path: sys.path.append(project_root)
# # sys.path.append(current_dir) 

# # from diffuser.models.diffusion import GaussianDiffusion
# # from diffuser.models.temporal import TemporalUnet
# # from dataset_adapter import SafetyGymDataset

# # # ================= Monkey Patch (保持环境一致) =================
# # import safety_gymnasium
# # from safety_gymnasium.assets.geoms import Hazards
# # from safety_gymnasium.tasks.safe_navigation.goal.goal_level1 import GoalLevel1
# # from safety_gymnasium.tasks.safe_navigation.goal.goal_level0 import GoalLevel0

# # def patched_init(self, config):
# #     self.lidar_num_bins = 16
# #     self.lidar_max_dist = 3.0
# #     self.sensors_obs = ['accelerometer', 'velocimeter', 'gyro', 'magnetometer']
# #     self.task_name = 'GoalLevel1_Reproduction'
# #     config.update({'lidar_num_bins': 16, 'lidar_max_dist': 3.0, 
# #                    'sensors_obs': self.sensors_obs, 'task_name': self.task_name})
# #     GoalLevel0.__init__(self, config=config)
# #     self.placements_conf.extents = [-1.5, -1.5, 1.5, 1.5]
# #     self._add_geoms(Hazards(num=2, keepout=0.18))

# # def patched_build_observation_space(self):
# #     self.observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32)

# # def patched_obs(self):
# #     lidar_vec = self._obs_lidar(self.hazards.pos, self.hazards.group) 
# #     acc = self.agent.get_sensor('accelerometer')[:2]
# #     vel = self.agent.get_sensor('velocimeter')[:2]
# #     gyro = self.agent.get_sensor('gyro')[-1:]
# #     mag = self.agent.get_sensor('magnetometer')[:2]
# #     sensor_vec = np.concatenate([acc, vel, gyro, mag])
# #     vec = (self.goal.pos - self.agent.pos) @ self.agent.mat
# #     x, y = vec[0], vec[1]
# #     z = x + 1j * y
# #     dist = np.exp(-np.abs(z)) 
# #     angle = np.angle(z)
# #     goal_vec = np.array([dist, np.cos(angle), np.sin(angle)])
# #     return np.concatenate([sensor_vec, goal_vec, lidar_vec]).astype(np.float32)

# # GoalLevel1.__init__ = patched_init
# # GoalLevel1.build_observation_space = patched_build_observation_space
# # GoalLevel1.obs = patched_obs

# # # ================= 配置区域 =================
# # DATASET_NAME = 'dataset_raw.npz' # 使用 Raw 训练的模型
# # # DATASET_NAME = 'dataset_safe_only.npz' 

# # CHECKPOINT_DIR = os.path.join(project_root, 'diffuser_checkpoints', DATASET_NAME.replace(".npz", ""))
# # CHECKPOINT_NAME = None
# # DATASET_PATH = os.path.join(project_root, 'datasets', DATASET_NAME)

# # DEVICE = 'cuda:0'
# # HORIZON = 64
# # NUM_EPISODES = 5 # 跑几轮测试

# # # ================= 策略封装 (仿照你的 Policy 类) =================
# # class DiffusionPolicy:
# #     def __init__(self, diffusion_model, normalizer):
# #         self.diffusion = diffusion_model
# #         self.normalizer = normalizer
# #         self.action_dim = diffusion_model.action_dim
# #         self.observation_dim = diffusion_model.observation_dim

# #     def __call__(self, obs, batch_size=1):
# #         # 1. 归一化观测
# #         # obs: (26,) -> (1, 26)
# #         obs_in = obs[None, :]
# #         norm_obs = self.normalizer.normalize(obs_in, 'observations')
        
# #         # 2. 构造条件
# #         conditions = {0: torch.tensor(norm_obs, device=DEVICE)}
        
# #         # 3. 生成轨迹
# #         start_t = time.time()
# #         # with torch.no_grad():
# #         #     # samples: (B, H, Obs+Act)
# #         #     samples = self.diffusion.conditional_sample(conditions)
# #         #     samples = samples.cpu().numpy()
# #         with torch.no_grad():
# #             samples = self.diffusion.conditional_sample(conditions, return_diffusion=False, verbose=False)
# #             if isinstance(samples, tuple):
# #                 samples = samples[0]
                
# #             samples = samples.cpu().numpy()
# #         end_t = time.time()
        
# #         # 4. 提取动作 (Trajectory Optimization)
# #         # 提取第一步的动作部分 [0, 0, 26:28]
# #         norm_action = samples[0, 0, self.observation_dim:] 
        
# #         # 5. 反归一化动作
# #         action = self.normalizer.unnormalize(norm_action[None, :], 'actions')
# #         action = action[0]

# #         if np.random.rand() < 0.05: # 5% 的概率打印，抽查
# #             print(f"\n🔍 [诊断] Step 检查:")
# #             print(f"  > 模型输出 (Norm): {norm_action} (范围应在 -1~1 之间)")
# #             print(f"  > 还原动作 (Real): {action} (Point机器人通常在 -1~1 之间)")
# #             print(f"  > 数据集动作范围: Min={self.normalizer.act_min}, Max={self.normalizer.act_max}")
        
# #         return action, samples, (end_t - start_t)

# # # ================= 主逻辑 =================
# # def main():
# #     # 1. 加载数据集 (Stats)
# #     print(f"Loading dataset stats: {DATASET_PATH}")
# #     dataset = SafetyGymDataset(DATASET_PATH, horizon=HORIZON)
    
# #     # 2. 加载模型
# #     model = TemporalUnet(
# #         horizon=HORIZON,
# #         transition_dim=26 + 2,
# #         cond_dim=26,
# #         dim=256,
# #         dim_mults=(1, 2, 4)
# #     ).to(DEVICE)

# #     diffusion = GaussianDiffusion(
# #         model=model,
# #         horizon=HORIZON,
# #         observation_dim=26,
# #         action_dim=2,
# #         n_timesteps=20, 
# #         loss_type='l2',
# #         clip_denoised=True,
# #         predict_epsilon=True,
# #     ).to(DEVICE)
# #     diffusion.normalizer = dataset

# #     # 3. 加载权重
# #     if CHECKPOINT_NAME is None:
# #         # 获取目录下所有文件
# #         all_files = os.listdir(CHECKPOINT_DIR)
# #         # 筛选出 "state_xxx.pt" 格式的文件
# #         ckpt_files = [f for f in all_files if f.startswith('state_') and f.endswith('.pt')]
        
# #         if not ckpt_files:
# #             raise FileNotFoundError(f"❌ 在 {CHECKPOINT_DIR} 下没找到任何 state_*.pt 模型文件！请先训练。")
        
# #         # 提取步数并排序 (state_1000.pt -> 1000)
# #         # key 逻辑: 把 "state_" 和 ".pt" 去掉，剩下的转 int
# #         ckpt_files.sort(key=lambda x: int(x.replace('state_', '').replace('.pt', '')))
        
# #         # 取最后一个 (步数最大的)
# #         latest_ckpt = ckpt_files[-1]
# #         print(f"🔄 自动检测到最新模型: {latest_ckpt}")
# #         ckpt_path = os.path.join(CHECKPOINT_DIR, latest_ckpt)
# #     else:
# #         # 如果指定了文件名，就用指定的
# #         ckpt_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)

# #     print(f"Loading checkpoint: {ckpt_path}")
# #     state_dict = torch.load(ckpt_path, map_location=DEVICE)
# #     # diffusion.model.load_state_dict(state_dict['model']) # 或者 ['ema']
# #     diffusion.load_state_dict(state_dict['model'])
# #     diffusion.eval()
    
# #     # 初始化策略
# #     policy = DiffusionPolicy(diffusion, dataset)

# #     # 4. 🔥 预热 GPU (来自 planlast.py 的灵感)
# #     print("🔥 Warming up GPU...")
# #     dummy_obs = np.zeros(26, dtype=np.float32)
# #     for _ in range(2):
# #         policy(dummy_obs)
# #     print("✅ Warmup done.")

# #     # 5. 环境
# #     # evaluator = omnisafe.Evaluator(render_mode='rgb_array')
# #     # env = evaluator._env 
# #     env = safety_gymnasium.make('SafetyPointGoal1-v0', render_mode='rgb_array', camera_name='fixedfar', width=256, height=256)
    
# #     # 6. 评估循环
# #     results = []
    
# #     for ep in range(NUM_EPISODES):
# #         print(f"\n=== Episode {ep+1}/{NUM_EPISODES} ===")
# #         obs, _ = env.reset()
# #         done = False
# #         total_rew = 0
# #         total_cost = 0
# #         step = 0
        
# #         # 视频录制
# #         video_path = f'eval_ep{ep}.mp4'
# #         writer = imageio.get_writer(video_path, fps=30)
        
# #         traj_data = {'observations': [], 'actions': [], 'costs': []}
        
# #         while not done and step < 1000:
# #             # Plan
# #             action, plan_traj, plan_time = policy(obs)
            
# #             # Step
# #             act_tensor = torch.as_tensor(action, dtype=torch.float32)
# #             next_obs, reward, cost, terminated, truncated, _ = env.step(act_tensor)
            
# #             # Record
# #             frame = env.render()
# #             writer.append_data(frame)
            
# #             traj_data['observations'].append(obs)
# #             traj_data['actions'].append(action)
# #             traj_data['costs'].append(cost)
            
# #             total_rew += reward
# #             total_cost += cost
# #             obs = next_obs
# #             step += 1
            
# #             if step % 50 == 0:
# #                 print(f"Step {step} | Reward: {reward:.3f} | Cost: {cost:.0f}")
                
# #             if terminated or truncated:
# #                 done = True
                
# #         writer.close()
# #         print(f"Episode Finished. Return: {total_rew:.2f}, Cost: {total_cost}")
        
# #         # 保存这一轮的数据 (参考 planlast 的 all_raw_results)
# #         results.append({
# #             'episode': ep,
# #             'return': total_rew,
# #             'cost': total_cost,
# #             'length': step,
# #             'trajectory': np.array(traj_data['observations'])
# #         })

# #     # 7. 保存最终结果到 PKL
# #     save_pkl = os.path.join(project_root, f'eval_results_{DATASET_NAME[:-4]}.pkl')
# #     with open(save_pkl, 'wb') as f:
# #         pickle.dump(results, f)
# #     print(f"\n✅ 所有测试完成，数据已保存至: {save_pkl}")

# # if __name__ == '__main__':
# #     main()

# import os
# # 【核心修复 1】强制使用 EGL 后端进行无头渲染 (必须放在 import imageio 之前)
# os.environ['MUJOCO_GL'] = 'egl' 

# import torch
# import numpy as np
# import imageio
# import sys
# import time
# import pickle
# import gymnasium
# from collections import namedtuple

# # ================= 路径设置 =================
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_dir)
# if project_root not in sys.path: sys.path.append(project_root)
# sys.path.append(current_dir) 

# from diffuser.models.diffusion import GaussianDiffusion
# from diffuser.models.temporal import TemporalUnet
# from scripts.dataset_adapter import SafetyGymDataset # 确保路径正确

# # ================= Monkey Patch (保持环境一致) =================
# import safety_gymnasium
# from safety_gymnasium.assets.geoms import Hazards
# from safety_gymnasium.tasks.safe_navigation.goal.goal_level1 import GoalLevel1
# from safety_gymnasium.tasks.safe_navigation.goal.goal_level0 import GoalLevel0

# def patched_init(self, config):
#     self.lidar_num_bins = 16
#     self.lidar_max_dist = 3.0
#     self.sensors_obs = ['accelerometer', 'velocimeter', 'gyro', 'magnetometer']
#     self.task_name = 'GoalLevel1_Reproduction'
#     config.update({'lidar_num_bins': 16, 'lidar_max_dist': 3.0, 
#                    'sensors_obs': self.sensors_obs, 'task_name': self.task_name})
#     GoalLevel0.__init__(self, config=config)
#     self.placements_conf.extents = [-1.5, -1.5, 1.5, 1.5]
#     self._add_geoms(Hazards(num=2, keepout=0.18))

# def patched_build_observation_space(self):
#     self.observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32)

# def patched_obs(self):
#     lidar_vec = self._obs_lidar(self.hazards.pos, self.hazards.group) 
#     acc = self.agent.get_sensor('accelerometer')[:2]
#     vel = self.agent.get_sensor('velocimeter')[:2]
#     gyro = self.agent.get_sensor('gyro')[-1:]
#     mag = self.agent.get_sensor('magnetometer')[:2]
#     sensor_vec = np.concatenate([acc, vel, gyro, mag])
#     vec = (self.goal.pos - self.agent.pos) @ self.agent.mat
#     x, y = vec[0], vec[1]
#     z = x + 1j * y
#     dist = np.exp(-np.abs(z)) 
#     angle = np.angle(z)
#     goal_vec = np.array([dist, np.cos(angle), np.sin(angle)])
#     return np.concatenate([sensor_vec, goal_vec, lidar_vec]).astype(np.float32)

# GoalLevel1.__init__ = patched_init
# GoalLevel1.build_observation_space = patched_build_observation_space
# GoalLevel1.obs = patched_obs

# # ================= 配置区域 =================
# DATASET_NAME = 'dataset_raw.npz' # 使用 Raw 训练的模型
# # DATASET_NAME = 'dataset_safe_only.npz' 

# CHECKPOINT_DIR = os.path.join(project_root, 'diffuser_checkpoints', DATASET_NAME.replace(".npz", ""))
# # 设置为 None 表示自动寻找步数最大的模型
# CHECKPOINT_NAME = None 
# DATASET_PATH = os.path.join(project_root, 'datasets', DATASET_NAME)

# DEVICE = 'cuda:0'
# HORIZON = 64
# NUM_EPISODES = 5 # 跑几轮测试

# # ================= 策略封装 =================
# class DiffusionPolicy:
#     def __init__(self, diffusion_model, normalizer):
#         self.diffusion = diffusion_model
#         self.normalizer = normalizer
#         self.action_dim = diffusion_model.action_dim
#         self.observation_dim = diffusion_model.observation_dim

#     def __call__(self, obs, batch_size=1):
#         # 1. 归一化观测 (使用 Gaussian Mean/Std)
#         # obs: (26,) -> (1, 26)
#         obs_in = obs[None, :]
#         norm_obs = self.normalizer.normalize(obs_in, 'observations')
        
#         norm_obs = np.clip(norm_obs, -5.0, 5.0)
        
#         # 2. 构造条件
#         # 注意：现在 dataset_adapter 里的条件是 {0: ...}
#         conditions = {0: torch.tensor(norm_obs, device=DEVICE)}
        
#         # 3. 生成轨迹
#         start_t = time.time()
#         with torch.no_grad():
#             # 这里的 verbose=False 很重要，防止进度条刷屏卡死
#             samples = self.diffusion.conditional_sample(conditions, return_diffusion=False, verbose=False)
            
#             if isinstance(samples, tuple):
#                 samples = samples[0]
                
#             samples = samples.cpu().numpy()
#         end_t = time.time()
        
#         # 4. 提取动作 (Trajectory Optimization)
#         # 提取第一步的动作部分 [0, 0, 26:28] (前26是Obs, 后2是Act)
#         # 注意：现在 samples 的维度是 (Batch, Horizon, Obs+Act) = (1, 64, 28)
#         norm_action = samples[0, 0, self.observation_dim:] 
        
#         # 5. 反归一化动作
#         action = self.normalizer.unnormalize(norm_action[None, :], 'actions')
#         action = action[0]

#         # # 诊断打印 (Gaussian 版)
#         # if np.random.rand() < 0.1: # 5% 的概率打印，抽查
#         #     print(f"\n🔍 [诊断] Step 检查:")
#         #     # 新增：打印观测值的前3位，看看是不是也在变
#         #     print(f"  > 输入观测 (Norm Top3): {norm_obs[0, :3]}") 
#         #     print(f"  > 模型输出 (Norm): {norm_action} (Gaussian: 通常在 -3~3 之间)")
#         #     print(f"  > 还原动作 (Real): {action} (Point机器人通常在 -1~1 之间)")
#         #     # 这里的 act_min/max 其实是截断后的边界，仅作参考
#         #     # print(f"  > 数据集动作边界: Min={self.normalizer.act_min}, Max={self.normalizer.act_max}")
#         if np.random.rand() < 0.1: 
#             print(f"\n🔍 [诊断] Step 检查:")
#             # 打印观测值的第 20~23 位 (通常 Goal 信息在这里)
#             # 根据 patched_obs: [sensor(16), goal(3), lidar(16)] -> wait, let's check index
#             # sensor_vec = 4 (acc+vel+gyro+mag) ?? No.
#             # 让我们直接打印 norm_obs 的方差，如果全是 0 就完了
            
#             print(f"  > 输入观测 (Norm Min/Max): {norm_obs.min():.3f} / {norm_obs.max():.3f}")
#             # PointGoal1 的 obs 结构通常是: 
#             # 0-3: 传感器
#             # 4-6: Goal (Dist, Cos, Sin) <--- 重点看这里！
#             # 7-22: Lidar
#             print(f"  > 目标信号 (Obs[7:10]): {norm_obs[0, 7:10]}") 
            
#             print(f"  > 模型输出 (Norm): {norm_action}")
#             print(f"  > 还原动作 (Real): {action}")
#         return action, samples, (end_t - start_t)

# # ================= 主逻辑 =================
# def main():
#     # 1. 加载数据集 (Stats)
#     print(f"Loading dataset stats: {DATASET_PATH}")
#     # 这里会自动进行 Gaussian 统计量的计算和 Clip 操作
#     dataset = SafetyGymDataset(DATASET_PATH, horizon=HORIZON)
    
#     # 2. 加载模型
#     model = TemporalUnet(
#         horizon=HORIZON,
#         transition_dim=26 + 2, # Obs + Act
#         cond_dim=26,           # Obs
#         dim=256,
#         dim_mults=(1, 2, 4)
#     ).to(DEVICE)

#     diffusion = GaussianDiffusion(
#         model=model,
#         horizon=HORIZON,
#         observation_dim=26,
#         action_dim=2,
#         n_timesteps=100, 
#         loss_type='l2',
#         clip_denoised=True,
#         predict_epsilon=False,
#     ).to(DEVICE)
#     # 绑定 normalizer
#     diffusion.normalizer = dataset

#     # 3. 加载权重 (自动寻找最新)
#     if CHECKPOINT_NAME is None:
#         if not os.path.exists(CHECKPOINT_DIR):
#              raise FileNotFoundError(f"❌ 目录不存在: {CHECKPOINT_DIR}")
        
#         # 获取目录下所有文件
#         all_files = os.listdir(CHECKPOINT_DIR)
#         # 筛选出 "state_xxx.pt" 格式的文件
#         ckpt_files = [f for f in all_files if f.startswith('state_') and f.endswith('.pt')]
        
#         if not ckpt_files:
#             raise FileNotFoundError(f"❌ 在 {CHECKPOINT_DIR} 下没找到任何 state_*.pt 模型文件！请先训练。")
        
#         # 提取步数并排序 (state_1000.pt -> 1000)
#         ckpt_files.sort(key=lambda x: int(x.replace('state_', '').replace('.pt', '')))
        
#         # 取最后一个 (步数最大的)
#         latest_ckpt = ckpt_files[-1]
#         print(f"🔄 自动检测到最新模型: {latest_ckpt}")
#         ckpt_path = os.path.join(CHECKPOINT_DIR, latest_ckpt)
#     else:
#         # 如果指定了文件名，就用指定的
#         ckpt_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)

#     print(f"Loading checkpoint: {ckpt_path}")
#     state_dict = torch.load(ckpt_path, map_location=DEVICE)
    
#     # 优先加载 EMA 权重 (如果有)
#     # if 'ema' in state_dict:
#     #     print("✨ Loading EMA weights (Better performance)...")
#     #     diffusion.load_state_dict(state_dict['ema'])
#     # else:
#     #     print("⚠️ No EMA weights found, loading standard weights...")
#     #     diffusion.load_state_dict(state_dict['model'])
#     print("⚠️ Force loading standard weights for debugging...")
#     diffusion.load_state_dict(state_dict['model'])
        
#     diffusion.eval()
    
#     # 初始化策略
#     policy = DiffusionPolicy(diffusion, dataset)

#     # 4. 🔥 预热 GPU
#     print("🔥 Warming up GPU...")
#     dummy_obs = np.zeros(26, dtype=np.float32)
#     for _ in range(2):
#         policy(dummy_obs)
#     print("✅ Warmup done.")

#     # 5. 环境 (直接 make，应用 Patch)
#     print("Creating environment...")
#     env = safety_gymnasium.make('SafetyPointGoal1-v0', render_mode='rgb_array', camera_name='fixedfar', width=256, height=256)
    
#     # 6. 评估循环
#     results = []
    
#     for ep in range(NUM_EPISODES):
#         print(f"\n=== Episode {ep+1}/{NUM_EPISODES} ===")
#         obs, _ = env.reset()
#         done = False
#         total_rew = 0
#         total_cost = 0
#         step = 0
        
#         # 视频录制
#         video_path = f'eval_ep{ep}.mp4'
#         # 使用 imageio 的 ffmpeg writer，指定 pixel format 以兼容大多数播放器
#         writer = imageio.get_writer(video_path, fps=30, macro_block_size=None)
        
#         traj_data = {'observations': [], 'actions': [], 'costs': []}
        
#         while not done and step < 1000:
#             # Plan
#             action, plan_traj, plan_time = policy(obs)
            
#             # Step
#             # 再次强制 Clip 动作，防止物理引擎炸裂
#             action = np.clip(action, -1.0, 1.0)
            
#             act_tensor = torch.as_tensor(action, dtype=torch.float32)
#             # Safety Gym 接口变更: reset 返回 info, step 返回 terminated, truncated
#             # next_obs, reward, cost, terminated, truncated, info
#             step_result = env.step(act_tensor)
            
#             if len(step_result) == 6: # New Gym API
#                  next_obs, reward, cost, terminated, truncated, _ = step_result
#             elif len(step_result) == 5: # Old Gym API (safety gym 可能会有变种)
#                  next_obs, reward, cost, done, _ = step_result
#                  terminated = done
#                  truncated = False
#             elif len(step_result) == 4: # Standard Gym
#                  next_obs, reward, terminated, truncated = step_result
#                  cost = 0 # 没有 Cost

#             # Record
#             try:
#                 frame = env.render()
#                 writer.append_data(frame)
#             except Exception as e:
#                 if step == 0: print(f"⚠️ Render failed: {e}")

#             traj_data['observations'].append(obs)
#             traj_data['actions'].append(action)
#             traj_data['costs'].append(cost)
            
#             total_rew += reward
#             total_cost += cost
#             obs = next_obs
#             step += 1
            
#             if step % 50 == 0:
#                 print(f"Step {step} | Reward: {reward:.3f} | Cost: {cost:.0f}")
                
#             if terminated or truncated:
#                 done = True
                
#         writer.close()
#         print(f"Episode Finished. Return: {total_rew:.2f}, Cost: {total_cost}")
        
#         # 保存这一轮的数据
#         results.append({
#             'episode': ep,
#             'return': total_rew,
#             'cost': total_cost,
#             'length': step,
#             'trajectory': np.array(traj_data['observations'])
#         })

#     # 7. 保存最终结果到 PKL
#     save_pkl = os.path.join(project_root, f'eval_results_{DATASET_NAME[:-4]}.pkl')
#     with open(save_pkl, 'wb') as f:
#         pickle.dump(results, f)
#     print(f"\n✅ 所有测试完成，数据已保存至: {save_pkl}")

# if __name__ == '__main__':
#     main()



import os
# 【核心修复 1】强制使用 EGL 后端进行无头渲染 (必须放在 import imageio 之前)
os.environ['MUJOCO_GL'] = 'egl' 

import torch
import numpy as np
import imageio
import sys
import time
import pickle
import gymnasium
import argparse  # 🔥【新增】参数解析
from collections import namedtuple

# ================= 路径设置 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path: sys.path.append(project_root)
sys.path.append(current_dir) 

from diffuser.models.diffusion import GaussianDiffusion
from diffuser.models.temporal import TemporalUnet
from scripts.dataset_adapter import SafetyGymDataset 

# ================= Monkey Patch (保持环境一致) =================
import safety_gymnasium
from safety_gymnasium.assets.geoms import Hazards
from safety_gymnasium.tasks.safe_navigation.goal.goal_level1 import GoalLevel1
from safety_gymnasium.tasks.safe_navigation.goal.goal_level0 import GoalLevel0

def patched_init(self, config):
    self.lidar_num_bins = 16
    self.lidar_max_dist = 3.0
    self.sensors_obs = ['accelerometer', 'velocimeter', 'gyro', 'magnetometer']
    self.task_name = 'GoalLevel1_Reproduction'
    config.update({'lidar_num_bins': 16, 'lidar_max_dist': 3.0, 
                   'sensors_obs': self.sensors_obs, 'task_name': self.task_name})
    GoalLevel0.__init__(self, config=config)
    self.placements_conf.extents = [-1.5, -1.5, 1.5, 1.5]
    self._add_geoms(Hazards(num=2, keepout=0.2))

def patched_build_observation_space(self):
    self.observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32)

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
    dist = np.exp(-np.abs(z)) 
    angle = np.angle(z)
    goal_vec = np.array([dist, np.cos(angle), np.sin(angle)])
    return np.concatenate([sensor_vec, goal_vec, lidar_vec]).astype(np.float32)

GoalLevel1.__init__ = patched_init
GoalLevel1.build_observation_space = patched_build_observation_space
GoalLevel1.obs = patched_obs

# ================= 策略封装 =================
class DiffusionPolicy:
    def __init__(self, diffusion_model, normalizer, goal_weight=1.0):
        self.diffusion = diffusion_model
        self.normalizer = normalizer
        self.action_dim = diffusion_model.action_dim
        self.observation_dim = diffusion_model.observation_dim
        self.goal_weight = goal_weight # 🔥【新增】

    def __call__(self, obs, batch_size=1):
        # 1. 归一化观测
        obs_in = obs[None, :]
        norm_obs = self.normalizer.normalize(obs_in, 'observations')
        
        # 🔥【关键】评估时也要乘权重！
        if self.goal_weight != 1.0:
            # 假设 Goal 是 7, 8, 9 维 (Dist, Cos, Sin)
            norm_obs[:, 7:10] *= self.goal_weight

        # 截断保护
        norm_obs = np.clip(norm_obs, -5.0, 5.0)
        
        # 2. 构造条件
        DEVICE = next(self.diffusion.parameters()).device
        conditions = {0: torch.tensor(norm_obs, device=DEVICE)}
        
        # 3. 生成轨迹
        start_t = time.time()
        with torch.no_grad():
            samples = self.diffusion.conditional_sample(conditions, return_diffusion=False, verbose=False)
            if isinstance(samples, tuple): samples = samples[0]
            samples = samples.cpu().numpy()
        end_t = time.time()
        
        # 4. 提取动作
        norm_action = samples[0, 0, self.observation_dim:] 
        
        # 5. 反归一化
        action = self.normalizer.unnormalize(norm_action[None, :], 'actions')
        action = action[0]

        # 诊断打印
        if np.random.rand() < 0.05: 
            print(f"\n🔍 [诊断] Step 检查:")
            print(f"  > 输入观测 (Norm Min/Max): {norm_obs.min():.3f} / {norm_obs.max():.3f}")
            print(f"  > 目标信号 (Obs[7:10] * {self.goal_weight}x): {norm_obs[0, 7:10]}") 
            print(f"  > 还原动作 (Real): {action}")
            
        return action, samples, (end_t - start_t)

# ================= 主逻辑 =================
def main():
    # 🔥【新增】参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dataset_raw.npz')
    parser.add_argument('--model_path', type=str, default='./diffuser_checkpoints/default_run', 
                        help='模型文件夹路径 (例如 ./diffuser_checkpoints/checkpoints_weighted)')
    parser.add_argument('--goal_weight', type=float, default=1.0, 
                        help='评估时使用的目标权重 (必须和训练时一致！)')
    parser.add_argument('--num_episodes', type=int, default=5)
    args = parser.parse_args()

    DATASET_PATH = os.path.join(project_root, 'datasets', args.dataset)
    CHECKPOINT_DIR = args.model_path
    DEVICE = 'cuda:0'
    HORIZON = 64

    # 1. 加载数据集 (Stats)
    print(f"Loading dataset stats: {DATASET_PATH}")
    # 注意：这里的 goal_weight 参数给 Dataset 是没用的（Eval时不加载数据），但为了保持一致性可以传
    # 关键是在 Policy.__call__ 里手动乘
    dataset = SafetyGymDataset(DATASET_PATH, horizon=HORIZON)
    
    # 2. 加载模型结构
    model = TemporalUnet(
        horizon=HORIZON,
        transition_dim=26 + 2,
        cond_dim=26,
        dim=256,
        dim_mults=(1, 2, 4)
    ).to(DEVICE)

    diffusion = GaussianDiffusion(
        model=model,
        horizon=HORIZON,
        observation_dim=26,
        action_dim=2,
        n_timesteps=100, 
        loss_type='l2',
        clip_denoised=True,
        predict_epsilon=False,
    ).to(DEVICE)
    diffusion.normalizer = dataset

    # 3. 加载权重
    if not os.path.exists(CHECKPOINT_DIR):
         raise FileNotFoundError(f"❌ 目录不存在: {CHECKPOINT_DIR}")
    
    all_files = os.listdir(CHECKPOINT_DIR)
    ckpt_files = [f for f in all_files if f.startswith('state_') and f.endswith('.pt')]
    
    if not ckpt_files:
        raise FileNotFoundError(f"❌ 在 {CHECKPOINT_DIR} 下没找到任何 state_*.pt 模型文件！")
    
    ckpt_files.sort(key=lambda x: int(x.replace('state_', '').replace('.pt', '')))
    latest_ckpt = ckpt_files[-1]
    
    print(f"🔄 自动检测到最新模型: {latest_ckpt}")
    ckpt_path = os.path.join(CHECKPOINT_DIR, latest_ckpt)

    print(f"Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    diffusion.load_state_dict(state_dict['model'])
    diffusion.eval()
    
    # 初始化策略 (传入 goal_weight)
    print(f"🔥 Evaluation Goal Weight: {args.goal_weight}x")
    policy = DiffusionPolicy(diffusion, dataset, goal_weight=args.goal_weight)

    # 4. 预热 GPU
    print("🔥 Warming up GPU...")
    dummy_obs = np.zeros(26, dtype=np.float32)
    for _ in range(2): policy(dummy_obs)
    print("✅ Warmup done.")

    # 5. 环境
    print("Creating environment...")
    env = safety_gymnasium.make('SafetyPointGoal1-v0', render_mode='rgb_array', camera_name='fixedfar', width=256, height=256)
    
    # 6. 评估循环
    for ep in range(args.num_episodes):
        print(f"\n=== Episode {ep+1}/{args.num_episodes} ===")
        obs, _ = env.reset()
        done = False
        total_rew = 0
        step = 0
        video_path = f'eval_ep{ep}.mp4'
        writer = imageio.get_writer(video_path, fps=30, macro_block_size=None)
        
        while not done and step < 1000:
            action, _, _ = policy(obs)
            action = np.clip(action, -1.0, 1.0)
            
            # Step
            step_result = env.step(action)
            if len(step_result) == 6:
                 next_obs, reward, cost, terminated, truncated, _ = step_result
            else:
                 next_obs, reward, cost, terminated, truncated = step_result[0], step_result[1], 0, step_result[2], step_result[3]

            try:
                frame = env.render()
                writer.append_data(frame)
            except: pass

            total_rew += reward
            obs = next_obs
            step += 1
            
            if terminated or truncated: done = True
                
        writer.close()
        print(f"Episode Finished. Return: {total_rew:.2f}")

if __name__ == '__main__':
    main()