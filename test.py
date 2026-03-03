# import numpy as np
# import safety_gymnasium
# import torch
# import os

# # 1. 加载训练数据
# DATA_PATH = './datasets/dataset_raw.npz' # 确保路径对
# print(f"正在读取训练数据: {DATA_PATH}")
# data = np.load(DATA_PATH)
# train_obs = data['observations']

# # 计算训练数据的每一维的均值和方差
# train_mean = np.mean(train_obs, axis=0)
# train_std = np.std(train_obs, axis=0)
# print(f"训练数据 Obs 维度: {train_obs.shape}")

# # 2. 创建评估环境 (使用你的 Patch 逻辑)
# from safety_gymnasium.tasks.safe_navigation.goal.goal_level1 import GoalLevel1
# from safety_gymnasium.tasks.safe_navigation.goal.goal_level0 import GoalLevel0
# from safety_gymnasium.assets.geoms import Hazards
# import gymnasium

# # --- 复制你的 Patch 代码 ---
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

# def patched_obs(self):
#     # 你的手动拼接逻辑
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
# GoalLevel1.obs = patched_obs
# # -------------------------

# print("正在初始化评估环境...")
# env = safety_gymnasium.make('SafetyPointGoal1-v0', render_mode='rgb_array', camera_name='fixedfar', width=256, height=256)

# # 3. 采样一些环境数据
# env_obs_list = []
# obs, _ = env.reset()
# for _ in range(1000):
#     action = env.action_space.sample()
#     obs, _, _, _, _, _ = env.step(action)
#     env_obs_list.append(obs)

# env_obs = np.array(env_obs_list)
# env_mean = np.mean(env_obs, axis=0)
# env_std = np.std(env_obs, axis=0)

# # 4. 对比打印
# print("\n" + "="*60)
# print(f"{'Dim':<5} | {'Train Mean':<12} | {'Eval Mean':<12} | {'Diff':<10} | {'判断'}")
# print("-" * 60)

# mismatch_count = 0
# for i in range(26):
#     t_m = train_mean[i]
#     e_m = env_mean[i]
#     diff = abs(t_m - e_m)
    
#     # 简单的启发式判断
#     status = "✅"
#     if diff > 0.5: status = "❌ 偏差大"
#     if i >= 10 and i < 26: # 雷达区域
#         if t_m < 0.1 and e_m > 0.5: status = "❓ 只有环境有障碍?"
#         if t_m > 0.5 and e_m < 0.1: status = "❓ 只有训练有障碍?"
        
#     print(f"{i:<5} | {t_m:12.4f} | {e_m:12.4f} | {diff:10.4f} | {status}")
#     if diff > 1.0: mismatch_count += 1

# print("="*60)
# if mismatch_count > 3:
#     print("💀 严重警告：观测分布差异巨大！你的 patched_obs 顺序很可能写错了！")
#     print("建议：检查 patched_obs 里的拼接顺序，是不是把 Goal 和 Lidar 搞反了？")
# else:
#     print("✨ 分布看起来基本一致，观测顺序应该没问题。问题可能在 Action 归一化。")

# import numpy as np
# import os

# # 指向你正在用的那个文件
# DATA_PATH = '/home/lqz27/dyx_ws/omnisafe/datasets/dataset_raw.npz'

# print(f"正在检查文件: {DATA_PATH}")

# if not os.path.exists(DATA_PATH):
#     print("❌ 文件不存在！路径错了吗？")
# else:
#     data = np.load(DATA_PATH)
#     actions = data['actions']
#     obs = data['observations']
    
#     print(f"数据量 (Steps): {actions.shape[0]}")
    
#     # 1. 检查动作方差
#     act_std = np.std(actions, axis=0)
#     print(f"📊 动作方差 (Action Std): {act_std}")
    
#     if np.mean(act_std) < 0.1:
#         print("🚨 【实锤了】这是旧数据！方差极低，机器人在画圆或不动。")
#         print("   -> 请检查 preprocess 脚本是否真的把 v2 数据写进去了。")
#     else:
#         print("✅ 动作方差正常，确实是新数据。")

#     # 2. 检查观测方差 (导致 5.000 爆炸的原因)
#     obs_std = np.std(obs, axis=0)
#     print(f"📊 观测方差 (Obs Std Min/Max): {np.min(obs_std):.6f} / {np.max(obs_std):.6f}")
    
#     # 检查是否有极小方差
#     low_var_dims = np.where(obs_std < 1e-4)[0]
#     if len(low_var_dims) > 0:
#         print(f"⚠️ 警告: 第 {low_var_dims} 维度的方差极小！")
#         print("   这意味着训练集里这些维度几乎没变过，但在测试时一变就会导致归一化爆炸。")

# import numpy as np
# # 换成你的 dataset_v2_raw 路径
# data = np.load('/home/lqz27/dyx_ws/omnisafe/datasets/dataset_raw.npz') 
# actions = data['actions']

# print(f"动作均值: {np.mean(actions, axis=0)}")
# # 如果 Action[1] (第二项) 是负数（比如 -0.4），说明专家本身就喜欢右转。
# # 如果 Action[1] 接近 0 (比如 -0.05)，说明数据没问题，是模型还没练好。

# import os
# import torch
# import numpy as np
# import omnisafe
# import safety_gymnasium
# import gymnasium
# from safety_gymnasium.assets.geoms import Hazards
# from safety_gymnasium.tasks.safe_navigation.goal.goal_level1 import GoalLevel1
# from safety_gymnasium.tasks.safe_navigation.goal.goal_level0 import GoalLevel0

# # =================================================================
# # 1. Monkey Patch (必须加，因为 checkpoint 是 26 维的)
# # =================================================================
# def patched_init(self, config):
#     self.lidar_num_bins = 16
#     self.lidar_max_dist = 3.0
#     self.sensors_obs = ['accelerometer', 'velocimeter', 'gyro', 'magnetometer']
#     self.task_name = 'GoalLevel1_Reproduction'
#     config.update({'lidar_num_bins': 16, 'lidar_max_dist': 3.0, 'sensors_obs': self.sensors_obs, 'task_name': self.task_name})
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
# print("✅ 环境 Patch 已应用 (26维)")

# # ================= 配置 =================
# LOG_DIR = './runs/PPOLag-{SafetyPointGoal1-v0}/seed-000-2026-02-09-14-10-04'

# # ================= 辅助函数 =================
# def find_actor(obj, depth=0):
#     if depth > 4: return None
#     if hasattr(obj, 'predict') and callable(getattr(obj, 'predict')):
#         if not isinstance(obj, omnisafe.Evaluator): return obj
#     for attr_name in dir(obj):
#         if attr_name.startswith('__'): continue
#         try:
#             attr_obj = getattr(obj, attr_name)
#             res = find_actor(attr_obj, depth + 1)
#             if res: return res
#         except: continue
#     return None

# def verify_ppo_performance():
#     print(f"🔍 正在加载 PPO 模型...")
#     evaluator = omnisafe.Evaluator()
    
#     try:
#         evaluator.load_saved(save_dir=LOG_DIR, model_name='epoch-100.pt')
#         print("✅ 加载了 epoch-500.pt")
#     except:
#         try:
#             evaluator.load_saved(save_dir=LOG_DIR, model_name='model.pt')
#             print("✅ 加载了 model.pt")
#         except:
#             print("❌ 加载失败，请检查路径")
#             return

#     agent = find_actor(evaluator)
#     if agent is None:
#         print("❌ 无法找到策略网络！")
#         return
        
#     env = evaluator._env
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     if hasattr(agent, 'to'): agent.to(device)

#     print(f"✅ 模型加载成功！Env Dim: {env.observation_space.shape}")
    
#     # 跑 5 个 Episode 看看平均分
#     num_episodes = 5
#     total_rewards = []
    
#     for i in range(num_episodes):
#         obs, _ = env.reset()
#         done = False
#         step = 0
#         ep_ret = 0
#         ep_cost = 0
        
#         first_actions = []
        
#         while not done and step < 1000:
#             with torch.no_grad():
#                 # 确保 obs 是 Tensor (Batch=1)
#                 if isinstance(obs, np.ndarray):
#                     obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(device).unsqueeze(0)
#                 else:
#                     obs_tensor = obs.to(device).unsqueeze(0)
                
#                 # 预测动作 (Tensor on GPU) -> Shape通常是 (1, 2)
#                 act = agent.predict(obs_tensor, deterministic=True)
                
#                 if step < 10: 
#                     first_actions.append(act.squeeze(0).cpu().numpy())

#             # 🔥【修复关键点】
#             # 1. 把动作搬回 CPU
#             # 2. 压缩维度 (1, 2) -> (2,)
#             act_cpu_tensor = act.cpu().squeeze(0) 
            
#             # 诊断：如果是第一步，打印一下形状确认
#             if step == 0 and i == 0:
#                 print(f"🔍 动作形状检查: Original={act.shape}, Squeezed={act_cpu_tensor.shape}")
            
#             res = env.step(act_cpu_tensor)
            
#             if len(res) == 6:
#                 obs, reward, cost, terminated, truncated, _ = res
#             elif len(res) == 5:
#                 obs, reward, cost, terminated, truncated = res
            
#             # 处理返回值类型
#             if isinstance(reward, torch.Tensor): reward = reward.item()
#             if isinstance(cost, torch.Tensor): cost = cost.item()
#             if isinstance(terminated, torch.Tensor): terminated = bool(terminated.item())
#             if isinstance(truncated, torch.Tensor): truncated = bool(truncated.item())
            
#             ep_ret += reward
#             ep_cost += cost
#             step += 1
            
#             if terminated or truncated:
#                 done = True
        
#         total_rewards.append(ep_ret)
        
#         avg_act = np.mean(first_actions, axis=0)
#         print(f"Episode {i+1}: Reward={ep_ret:.2f}, Cost={ep_cost}, Steps={step}")
#         print(f"   开局平均动作: {avg_act}")
#         if avg_act[1] < -0.1 or avg_act[1] > 0.1:
#             print("   ⚠️  警告: 明显在转圈！")

#     print(f"="*30)
#     avg_score = np.mean(total_rewards)
#     print(f"📊 平均得分: {avg_score:.2f}")
    
#     if avg_score < 5.0:
#         print("❌ 结论：实锤了！PPO 模型根本没训练好。")
#         print("   -> 赶紧重训 PPO 吧，别浪费时间了。")
#     else:
#         print("✅ 结论：PPO 是好的！")
#         print("   -> 之前的数据采集脚本有 Bug，我给你新的。")

# if __name__ == '__main__':
#     verify_ppo_performance()

# import pandas as pd

# # 读取日志
# df = pd.read_csv('./runs/PPOLag-{SafetyPointGoal1-v0}/seed-000-2026-02-09-17-49-50/progress.csv')

# # 1. 筛选安全达标的 epoch (假设 Cost Limit 是 0，考虑一点波动可以设为 5)
# safe_df = df[df['Metrics/EpCost'] <= 5.0]

# if not safe_df.empty:
#     # 2. 找奖励最高的
#     best_epoch_row = safe_df.loc[safe_df['Metrics/EpRet'].idxmax()]
#     best_epoch = int(best_epoch_row['Train/Epoch'])
#     print(f"🏆 最佳模型在 Epoch: {best_epoch}")
#     print(f"   - Reward: {best_epoch_row['Metrics/EpRet']}")
#     print(f"   - Cost:   {best_epoch_row['Metrics/EpCost']}")
# else:
#     print("没有找到安全的 Epoch，建议放宽 Cost 筛选条件或检查训练。")

# import numpy as np

# # 1. 加载数据 (确保路径对)
# data_path = './data_pro/ppolag_zuida.npz'
# print(f"📂 正在读取: {data_path}")
# data = np.load(data_path)

# # 2. 获取 segment_id
# seg_ids = data['segment_id']
# unique_segs = np.unique(seg_ids)

# print(f"\n📊 总共发现 {len(unique_segs)} 条轨迹片段")
# print("="*40)
# print(f"{'ID':<5} | {'Length (Steps)':<15} | {'Status'}")
# print("-" * 40)

# # 3. 循环打印每一条的长度
# lengths = []
# for seg_id in unique_segs:
#     # 计算当前 segment 的长度
#     seg_len = np.sum(seg_ids == seg_id)
#     lengths.append(seg_len)
    
#     # 简单的评价
#     status = ""
#     if seg_len < 50: status = "⚡️ 极速"
#     elif seg_len > 1000: status = "🐢 超时/徘徊"
#     elif seg_len > 500: status = "🤔 较慢"
    
#     print(f"{seg_id:<5} | {seg_len:<15} | {status}")

# print("="*40)
# print(f"平均轨迹长度: {np.mean(lengths):.2f} 步")
# print(f"最短: {np.min(lengths)} 步")
# print(f"最长: {np.max(lengths)} 步")

# import torch
# import numpy as np
# import os
# import safety_gymnasium
# import gymnasium
# from safety_gymnasium.assets.geoms import Hazards
# from safety_gymnasium.tasks.safe_navigation.goal.goal_level1 import GoalLevel1
# from safety_gymnasium.tasks.safe_navigation.goal.goal_level0 import GoalLevel0

# # ==========================================
# # 1. 环境定义 (必须和你现在的一样)
# # ==========================================
# def patched_init(self, config):
#     self.lidar_num_bins = 16
#     self.lidar_max_dist = 3.0
#     self.sensors_obs = ['accelerometer', 'velocimeter', 'gyro', 'magnetometer']
#     self.task_name = 'GoalLevel1_Reproduction'
#     config.update({'lidar_num_bins': 16, 'lidar_max_dist': 3.0, 'sensors_obs': self.sensors_obs, 'task_name': self.task_name})
#     GoalLevel0.__init__(self, config=config)
#     self.placements_conf.extents = [-1.5, -1.5, 1.5, 1.5]
#     self._add_geoms(Hazards(num=2, keepout=0.18))

# def patched_obs(self):
#     lidar_vec = self._obs_lidar(self.hazards.pos, self.hazards.group)
#     acc = self.agent.get_sensor('accelerometer')[:2]
#     vel = self.agent.get_sensor('velocimeter')[:2]
#     gyro = self.agent.get_sensor('gyro')[-1:]
#     mag = self.agent.get_sensor('magnetometer')[:2]
    
#     # 传感器部分 (7维)
#     sensor_vec = np.concatenate([acc, vel, gyro, mag])
    
#     # 目标部分 (3维)
#     vec = (self.goal.pos - self.agent.pos) @ self.agent.mat
#     x, y = vec[0], vec[1]
#     z = x + 1j * y
#     dist = np.exp(-np.abs(z)) 
#     angle = np.angle(z)
#     goal_vec = np.array([dist, np.cos(angle), np.sin(angle)])
    
#     # 拼接: [Sensor(7), Goal(3), Lidar(16)] = 26维
#     return np.concatenate([sensor_vec, goal_vec, lidar_vec]).astype(np.float32)

# GoalLevel1.__init__ = patched_init
# GoalLevel1.obs = patched_obs

# # ==========================================
# # 2. 诊断主程序
# # ==========================================
# if __name__ == '__main__':
#     NORM_PATH = './diffuser_checkpoints/normalization.npz'
#     DATA_PATH = './data_pro/ppolag_best.npz' # 你的原始训练数据路径
    
#     print("============== 🩺 诊断报告 ==============")
    
#     # --- 检查 1: 归一化参数 ---
#     if not os.path.exists(NORM_PATH):
#         print(f"❌ 错误: 找不到 {NORM_PATH}")
#         exit()
    
#     norm_data = np.load(NORM_PATH)
#     mins = norm_data['mins']
#     maxs = norm_data['maxs']
    
#     print(f"1. 归一化参数检查:")
#     print(f"   维度: {mins.shape} (预期 28: 26 obs + 2 act)")
#     print(f"   Obs Mins (前5位): {mins[:5]}")
#     print(f"   Obs Maxs (前5位): {maxs[:5]}")
#     print(f"   Lidar Range (最后16位): Min={mins[-16:].min():.4f}, Max={maxs[-16:].max():.4f}")
    
#     if np.allclose(mins, maxs):
#         print("   ❌ 严重警告: mins 和 maxs 完全相同！这将导致除零错误或全零输入。")
#     else:
#         print("   ✅ 参数分布看起来有数值。")

#     # --- 检查 2: 原始数据分布 ---
#     if os.path.exists(DATA_PATH):
#         raw_data = np.load(DATA_PATH)
#         obs_data = raw_data['obs']
#         print(f"\n2. 原始训练数据检查 ({DATA_PATH}):")
#         print(f"   Obs Shape: {obs_data.shape}")
#         print(f"   Last 16 dims (Lidar) mean: {obs_data[:, -16:].mean():.4f}")
#         if obs_data.shape[1] != 26:
#             print(f"   ❌ 维度警告: 训练数据是 {obs_data.shape[1]} 维，但代码期望 26 维！")
#     else:
#         print(f"\n2. 原始训练数据未找到，跳过检查。")

#     # --- 检查 3: 实时环境数值 ---
#     print(f"\n3. 实时环境数值检查:")
#     env = safety_gymnasium.make('SafetyPointGoal1-v0')
#     obs, _ = env.reset()
    
#     # 模拟走到障碍物附近
#     print("   正在移动机器人以获取非零观测...")
#     for _ in range(10):
#         obs, _, _, _, _, _ = env.step(np.array([1.0, 0.0]))
    
#     print(f"   当前 Obs (Total 26 dims):")
#     print(f"   -> Sensor (0-6): {obs[:7]}")
#     print(f"   -> Goal   (7-9): {obs[7:10]}")
#     print(f"   -> Lidar  (10-25): {obs[10:]}")
    
#     # 归一化模拟
#     obs_norm = (obs - mins[:26]) / (maxs[:26] - mins[:26])
#     obs_norm = 2 * obs_norm - 1
    
#     print(f"\n4. 归一化后的 Obs (送入网络的值):")
#     print(f"   -> Range: [{obs_norm.min():.4f}, {obs_norm.max():.4f}]")
#     print(f"   -> Lidar Norm: {obs_norm[10:]}")
    
#     if obs_norm.max() > 5.0 or obs_norm.min() < -5.0:
#         print("   ❌ 警告: 输入值极其巨大！说明归一化参数 min/max 和当前环境观测不匹配。")
#         print("      可能原因: 传感器顺序搞反了，或者单位不一致。")
#     elif np.allclose(obs_norm[10:], -1.0, atol=0.1):
#         print("   ⚠️ 警告: 雷达归一化后全是 -1。说明机器人以为周围全是空的，或者雷达没开。")
#     else:
#         print("   ✅ 输入值在合理范围 (-1 到 1 附近)。")

#     print("\n============== 诊断结束 ==============")

# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# def check_data_radius():
#     # 1. 加载数据
#     data_path = './data_pro/ppolag_zuida.npz' # 👈 换成你最新的数据集文件名
#     try:
#         data = np.load(data_path)
#     except FileNotFoundError:
#         print(f"❌ 找不到文件: {data_path}")
#         return

#     obs = data['obs']
#     labels = data['is_safe']  # 1=Safe, 0=Unsafe
    
#     print(f"📊 数据总量: {len(labels)}")
#     print(f"   安全样本: {np.sum(labels == 1)}")
#     print(f"   不安全样本: {np.sum(labels == 0)}")

#     # 2. 提取雷达的最大值 (代表离最近障碍物的距离)
#     # Lidar 是 obs 的最后 16 维
#     lidar_data = obs[:, -16:] 
#     max_lidar = np.max(lidar_data, axis=1)

#     # 3. 分析：在什么雷达强度下，标签变成了 Unsafe？
#     # Lidar = exp(-dist). 
#     # dist = 0 (贴脸) -> Lidar = 1.0
#     # dist = large -> Lidar = 0.0
    
#     plt.figure(figsize=(12, 6))
    
#     # 画分布图
#     sns.histplot(max_lidar[labels==1], color='green', label='Safe Samples', kde=False, bins=50, alpha=0.5, stat='density')
#     sns.histplot(max_lidar[labels==0], color='red', label='Unsafe Samples', kde=False, bins=50, alpha=0.5, stat='density')
    
#     plt.xlabel('Max Lidar Value (1.0 = Touching Hazard Surface)')
#     plt.ylabel('Density')
#     plt.title('Distribution of Safe/Unsafe Labels vs. Lidar Reading')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
    
#     # 找出“边界”：Unsafe 样本通常从哪个 Lidar 值开始出现？
#     unsafe_lidars = max_lidar[labels==0]
#     if len(unsafe_lidars) > 0:
#         threshold_estimate = np.percentile(unsafe_lidars, 5) # 取 5% 分位点作为边界
#         plt.axvline(threshold_estimate, color='black', linestyle='--', label=f'Estimated Boundary (Lidar={threshold_estimate:.2f})')
#         print(f"\n🔍 诊断结果：")
#         print(f"   Unsafe 标签开始大量出现的雷达阈值约为: {threshold_estimate:.2f}")
        
#         # 反推物理含义
#         # Lidar = exp(-dist_surface) => dist_surface = -ln(Lidar)
#         dist_surface = -np.log(threshold_estimate + 1e-6)
#         print(f"   这意味着：当离障碍物表面约 {dist_surface:.2f} 米时，数据被标记为不安全。")
        
#         if dist_surface < 0.05:
#             print("   ⚠️ 警告：你的数据可能真的只有在‘非常贴近’时才标记为不安全。")
#             print("   👉 建议：训练时不需要改数据，但在使用 CBF 时，可以通过调节 h(x) 的阈值来通过补偿。")
#         else:
#             print("   ✅ 放心：你的数据在接触前已经留有余量 (Buffer)，包含了物理半径的影响。")
            
#     plt.savefig('./cbf_checkpoints/data_check.png')
#     print("✅ 图表已保存至 ./cbf_checkpoints/data_check.png")
#     plt.show()

# if __name__ == '__main__':
#     check_data_radius()

# import gymnasium
# import safety_gymnasium
# import numpy as np
# import mujoco

# def get_mujoco_model(env):
#     """鲁棒地查找 MuJoCo Model"""
#     candidates = [
#         (env.unwrapped, "model"),
#         (env.unwrapped, "_model"),
#         (getattr(env.unwrapped, "task", None), "model"),
#         (getattr(env.unwrapped, "mujoco", None), "model")
#     ]
#     for obj, attr in candidates:
#         if obj is not None and hasattr(obj, attr):
#             return getattr(obj, attr)
#     raise AttributeError("❌ 无法找到 MuJoCo 模型，封装层级太复杂！")

# def verify_geometry():
#     print("🌍 正在初始化环境...")
#     env = gymnasium.make('SafetyPointGoal1-v0')
#     env.reset()
    
#     # 1. 获取底层模型
#     try:
#         model = get_mujoco_model(env)
#         print("✅ 成功获取 MuJoCo Model")
#     except Exception as e:
#         print(e)
#         return

#     print("=" * 40)
#     print("🔍 MuJoCo 物理参数核查")
#     print("=" * 40)

#     # 2. 查找机器人的 Geom Size
#     # 在 MuJoCo 中，Point 机器人的 Geom 通常叫 'robot' 或 'point'
#     try:
#         # 尝试名字 'robot'
#         geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'robot')
#         if geom_id == -1:
#             # 尝试名字 'agent'
#              geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'agent')
        
#         if geom_id != -1:
#             # MuJoCo 的 geom_size 是一个数组 [size_x, size_y, size_z]
#             # 对于球体 (sphere)，第一个值就是半径
#             robot_size = model.geom_size[geom_id][0]
#             print(f"🤖 Robot Geom ID: {geom_id}")
#             print(f"📏 Robot Radius (Geom Size): 【 {robot_size:.6f} 米 】")
#         else:
#             print("❌ 未找到名为 'robot' 或 'agent' 的 Geom，打印所有 Geom 名字：")
#             for i in range(model.ngeom):
#                 name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
#                 print(f"   - ID {i}: {name}, Size: {model.geom_size[i][0]}")

#     except Exception as e:
#         print(f"❌ 读取 Geom 失败: {e}")

#     # 3. 验证 Hazards 大小
#     print("-" * 40)
#     try:
#         # 尝试直接读取 SafetyGym 配置
#         if hasattr(env.task, 'hazards'):
#              print(f"⚠️ Hazards Config Size: 【 {env.task.hazards.size:.6f} 米 】")
#         elif hasattr(env.task, '_geoms') and 'hazards' in env.task._geoms:
#              # 旧版本兼容
#              print(f"⚠️ Hazards Config Size: 【 {env.task._geoms['hazards'].size:.6f} 米 】")
#     except:
#         print("⚠️ 无法读取 Hazard 配置对象")

#     print("=" * 40)

# if __name__ == '__main__':
#     verify_geometry()

# import safety_gymnasium
# import pprint

# def deep_inspect():
#     print("正在初始化环境...")
#     env = safety_gymnasium.make('SafetyPointGoal1-v0')
    
#     # 1. 解包到最底层
#     task = env.unwrapped.task
#     agent = task.agent
    
#     print("\n" + "="*40)
#     print("🕵️‍♂️ 开始深度搜索 'lidar' 参数...")
#     print("="*40)

#     # -------------------------------------------------
#     # 方法 A: 检查 Agent 的配置对象 (Common in SafetyGymnasium)
#     # -------------------------------------------------
#     if hasattr(agent, 'conf'):
#         print("\n[Location A] Found 'agent.conf':")
#         # 遍历 conf 里的属性
#         for key in dir(agent.conf):
#             if 'lidar' in key and not key.startswith('__'):
#                 val = getattr(agent.conf, key)
#                 print(f"  - agent.conf.{key} = {val}")

#     # -------------------------------------------------
#     # 方法 B: 暴力遍历 Agent 的所有属性
#     # -------------------------------------------------
#     print("\n[Location B] Scanning all 'agent' attributes:")
#     found = False
#     for attr in dir(agent):
#         if 'lidar' in attr.lower():
#             try:
#                 val = getattr(agent, attr)
#                 # 过滤掉函数，只看数值
#                 if not callable(val):
#                     print(f"  - agent.{attr} = {val}")
#                     found = True
#             except:
#                 pass
    
#     if not found:
#         print("  (None found directly on agent)")

#     # -------------------------------------------------
#     # 方法 C: 检查 Task 级别的配置
#     # -------------------------------------------------
#     print("\n[Location C] Scanning 'task' attributes:")
#     for attr in dir(task):
#         if 'lidar' in attr.lower():
#             try:
#                 val = getattr(task, attr)
#                 if not callable(val):
#                     print(f"  - task.{attr} = {val}")
#             except:
#                 pass

# if __name__ == "__main__":
#     deep_inspect()

# import safety_gymnasium
# import inspect
# import numpy as np

# def inspect_goal_compass():
#     print("="*60)
#     print("��️‍♂️ 正在寻找官方 'Goal Compass' (_obs_compass) 的源码...")
#     print("="*60)
    
#     # 1. 初始化一个环境
#     env = safety_gymnasium.make('SafetyPointGoal1-v0')
#     task = env.unwrapped.task
    
#     # 2. 尝试获取 _obs_compass 的源码
#     try:
#         # _obs_compass 通常定义在 BaseTask 或其父类中
#         if hasattr(task, '_obs_compass'):
#             func = task._obs_compass
#             code = inspect.getsource(func)
#             print(f"✅ 找到了！函数名: {func.__name__}")
#             print("-" * 40)
#             print(code)
#             print("-" * 40)
#         else:
#             print("❌ 在 task 下没找到 _obs_compass，可能改名了？")
            
#     except Exception as e:
#         print(f"❌ 读取源码失败: {e}")

#     print("\n�� 解读：")
#     print("如果代码里是 'pos - self.agent.pos' (减法)，那就是线性的 -> �� 聪明")
#     print("如果代码里有 'np.exp' (指数)，那就是非线性的 -> �� 蠢笨")
#     print("="*60)

# if __name__ == "__main__":
#     inspect_goal_compass()

# import os
# import torch
# import numpy as np
# import imageio
# import omnisafe
# import safety_gymnasium
# import gymnasium
# from safety_gymnasium.assets.geoms import Hazards
# # 引入原始类
# from safety_gymnasium.tasks.safe_navigation.goal.goal_level1 import GoalLevel1
# from safety_gymnasium.tasks.safe_navigation.goal.goal_level0 import GoalLevel0

# # =================================================================
# # 1. 【核心】重新植入 Patch (必须与训练时完全一致！)
# # =================================================================

# def patched_init(self, config):
#     """复现训练时的环境设置"""
#     self.lidar_num_bins = 16
#     self.lidar_max_dist = 3.0
#     self.sensors_obs = ['accelerometer', 'velocimeter', 'gyro', 'magnetometer']
#     self.task_name = 'GoalLevel1_Reproduction'
    
#     config.update({
#         'lidar_num_bins': 16,
#         'lidar_max_dist': 3.0,
#         'sensors_obs': self.sensors_obs,
#         'task_name': self.task_name
#     })
    
#     GoalLevel0.__init__(self, config=config)
    
#     # 修改环境: 2 Hazards
#     self.placements_conf.extents = [-1.5, -1.5, 1.5, 1.5]
#     self._add_geoms(Hazards(num=2, keepout=0.2))

# def patched_build_observation_space(self):
#     self.observation_space = gymnasium.spaces.Box(
#         low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32
#     )

# def patched_obs(self):
#     """
#     【关键】这里必须是你刚才改过的 Linear 版本
#     """
#     # 1. Hazard Lidar (16维)
#     lidar_vec = self._obs_lidar(self.hazards.pos, self.hazards.group) 
    
#     # 2. Sensors (7维)
#     acc = self.agent.get_sensor('accelerometer')[:2]
#     vel = self.agent.get_sensor('velocimeter')[:2]
#     gyro = self.agent.get_sensor('gyro')[-1:]
#     mag = self.agent.get_sensor('magnetometer')[:2]
#     sensor_vec = np.concatenate([acc, vel, gyro, mag])

#     # 3. Goal (3维) - 使用线性距离！
#     vec = (self.goal.pos - self.agent.pos) @ self.agent.mat
#     x, y = vec[0], vec[1]
    
#     # --- 你的 Linear 修改 ---
#     z = x + 1j * y
#     dist = np.abs(z) / 10.0  # 线性！
#     angle = np.angle(z)
    
#     goal_vec = np.array([dist, np.cos(angle), np.sin(angle)])

#     # 4. 拼接 (26维)
#     flat_obs = np.concatenate([sensor_vec, goal_vec, lidar_vec]).astype(np.float32)
#     return flat_obs

# # 应用补丁
# GoalLevel1.__init__ = patched_init
# GoalLevel1.build_observation_space = patched_build_observation_space
# GoalLevel1.obs = patched_obs
# print("✅ Monkey Patch 已就绪 (26维 Linear模式)")

# # =================================================================
# # 2. 配置路径
# # =================================================================

# # �� 请修改这里！指向你刚才训练结束的文件夹
# # 比如: runs/PPOLag-{SafetyPointGoal1-v0}/seed-000-2023-10-xx-xx-xx
# LOG_DIR = './runs/PPOLag-{SafetyPointGoal1-v0}/seed-000-2026-02-17-19-59-08' 

# # 视频保存名字
# VIDEO_NAME = 'linear_goal_result.mp4'

# # =================================================================
# # 3. 开始录制
# # =================================================================

# def main():
#     # 1. 加载模型
#     print(f"正在加载模型: {LOG_DIR}")
#     evaluator = omnisafe.Evaluator(render_mode='rgb_array')
    
#     # 注意：确保 model.pt 存在，或者改成 epoch-499.pt / epoch-500.pt
#     # evaluator.load_saved 会把模型加载到 evaluator._actor 中
#     try:
#         evaluator.load_saved(save_dir=LOG_DIR, model_name='model.pt', camera_name='fixedfar')
#     except Exception as e:
#         print(f"⚠️ model.pt 加载失败 ({e})，尝试加载 epoch-499.pt...")
#         evaluator.load_saved(save_dir=LOG_DIR, model_name='epoch-500.pt', camera_name='fixedfar')
    
#     # =========================================================
#     # ��️ 修复点：正确获取 Actor 网络
#     # =========================================================
#     actor = None
    
#     # 1. 尝试直接获取私有属性 _actor (OmniSafe 标准位置)
#     if hasattr(evaluator, '_actor') and evaluator._actor is not None:
#         print("✅ 成功在 evaluator._actor 找到策略网络")
#         actor = evaluator._actor
        
#     # 2. 尝试获取 actor 公共属性
#     elif hasattr(evaluator, 'actor') and evaluator.actor is not None:
#         print("✅ 成功在 evaluator.actor 找到策略网络")
#         actor = evaluator.actor
        
#     # 3. 暴力递归查找 (作为最后的备选)
#     else:
#         print("⚠️ 属性未直接找到，尝试递归搜索...")
#         def find_actor_recursive(obj, depth=0):
#             if depth > 3: return None
#             # 必须包含 predict 或 act 方法
#             if (hasattr(obj, 'predict') or hasattr(obj, 'act')) and not isinstance(obj, omnisafe.Evaluator):
#                 return obj
#             for key in dir(obj):
#                 if key.startswith('__'): continue
#                 try:
#                     val = getattr(obj, key)
#                     # 排除基础类型
#                     if isinstance(val, (int, float, str, bool, list, dict)): continue
#                     res = find_actor_recursive(val, depth + 1)
#                     if res: return res
#                 except: pass
#             return None
            
#         actor = find_actor_recursive(evaluator)

#     if actor is None:
#         # 如果还是找不到，可能是 load_saved 没成功
#         raise RuntimeError("❌ 无法找到 Actor 网络！请检查 LOG_DIR 路径下是否有 .pt 模型文件。")

#     # =========================================================
    
#     env = evaluator._env
#     print(f"环境维度: {env.observation_space.shape}")
    
#     # 再次确认维度
#     assert env.observation_space.shape == (26,), f"❌ 维度不对！期望 (26,) 但得到 {env.observation_space.shape}"

#     # 3. 录制循环
#     print(f"开始录制视频 -> {VIDEO_NAME}")
#     obs, _ = env.reset()
#     frames = []
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # 确保 actor 在正确的设备上 (有些版本 load_saved 后在 cpu)
#     if hasattr(actor, 'to'):
#         actor.to(device)
    
#     total_cost = 0
    
#     # 跑 1000 步
#     for step in range(1000):
#         try:
#             frames.append(env.render())
#         except:
#             pass # 有些环境 headless 渲染会报错，跳过
        
#         # 预测动作
#         # 预测动作
#         with torch.no_grad():
#             obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            
#             # (自动兼容单/多返回值的逻辑保持不变)
#             if hasattr(actor, 'predict'):
#                 raw_output = actor.predict(obs_t, deterministic=True)
#             elif hasattr(actor, 'act'):
#                 raw_output = actor.act(obs_t, deterministic=True)
#             else:
#                 raw_output = actor(obs_t)
            
#             if isinstance(raw_output, tuple):
#                 action = raw_output[0]
#             else:
#                 action = raw_output

#             # =========================================================
#             # ��️ 修复点：不要转 .numpy()！保持 Tensor！
#             # =========================================================
#             # ❌ 删除这行 (或注释掉):
#             # if isinstance(action, torch.Tensor):
#             #     action = action.cpu().numpy().squeeze()

#             # ✅ 改为这样：只去掉 batch 维度，但保持是 Tensor
#             if isinstance(action, torch.Tensor):
#                 action = action.squeeze(0).cpu()  # (1, dim) -> (dim)
#                 # 注意：wrapper 里的参数通常和 actor 在同一个 device 上
#                 # 如果报错 "Expected all tensors to be on the same device"，
#                 # 请尝试加上 .cpu()，即: action = action.squeeze(0).cpu()
#                 # 但根据你的报错，它想要 Tensor，所以先只做 squeeze
#             # =========================================================

            
#         # 执行
#         obs, reward, cost, terminated, truncated, info = env.step(action)
        
#         if hasattr(cost, 'item'): cost = cost.item()
#         total_cost += cost
        
#         if step % 100 == 0:
#             print(f"Step {step}: Reward={reward:.4f}, Cost={cost:.4f}, TotalCost={total_cost:.1f}")
            
#         if terminated or truncated:
#             print("回合结束")
#             break
            
#     # 4. 保存视频
#     print("正在编码 MP4...")
#     imageio.mimsave(VIDEO_NAME, frames, fps=30)
#     print(f"✅ 视频已保存: {os.path.abspath(VIDEO_NAME)}")

# if __name__ == '__main__':
#     main()


import safety_gymnasium
import inspect
from safety_gymnasium.tasks.safe_navigation.goal.goal_level0 import GoalLevel0

def check_reward_code():
    print("="*60)
    print("🔍 正在读取官方 Goal 任务的 Reward (奖励) 源代码...")
    print("="*60)
    
    try:
        # 在 Safety Gymnasium 中，奖励计算通常在 calculate_reward 方法里
        source_code = inspect.getsource(GoalLevel0.calculate_reward)
        print(source_code)
    except Exception as e:
        print(f"读取失败: {e}")

    print("="*60)
    print("💡 重点寻找：last_dist_goal (上一步距离) 和 dist_goal() (当前距离)")
    print("="*60)

if __name__ == "__main__":
    check_reward_code()