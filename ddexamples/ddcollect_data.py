# import os
# import torch
# import numpy as np
# import omnisafe
# import safety_gymnasium
# import gymnasium
# from safety_gymnasium.assets.geoms import Hazards
# # 引入原始类
# from safety_gymnasium.tasks.safe_navigation.goal.goal_level1 import GoalLevel1
# from safety_gymnasium.tasks.safe_navigation.goal.goal_level0 import GoalLevel0

# # =================================================================
# # 1. 【核心必须】植入 Monkey Patch
# # =================================================================

# def patched_init(self, config):
#     """替换 GoalLevel1.__init__"""
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
#     self._add_geoms(Hazards(num=2, keepout=0.18))
#     print("【Patch】环境地图已修改: 2 Hazards")

# def patched_build_observation_space(self):
#     """替换 build_observation_space"""
#     self.observation_space = gymnasium.spaces.Box(
#         low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32
#     )

# def patched_obs(self):
#     """替换 obs 方法 (确保 26 维)"""
#     # 1. Hazard Lidar (16维)
#     lidar_vec = self._obs_lidar(self.hazards.pos, self.hazards.group) 
    
#     # 2. Sensors (7维)
#     acc = self.agent.get_sensor('accelerometer')[:2]
#     vel = self.agent.get_sensor('velocimeter')[:2]
#     gyro = self.agent.get_sensor('gyro')[-1:]
#     mag = self.agent.get_sensor('magnetometer')[:2]
#     sensor_vec = np.concatenate([acc, vel, gyro, mag])

#     # 3. Goal (3维)
#     vec = (self.goal.pos - self.agent.pos) @ self.agent.mat
#     x, y = vec[0], vec[1]
#     z = x + 1j * y
#     dist = np.abs(z)
#     dist = np.exp(-dist) 
#     angle = np.angle(z)
#     goal_vec = np.array([dist, np.cos(angle), np.sin(angle)])

#     # 4. 拼接
#     flat_obs = np.concatenate([sensor_vec, goal_vec, lidar_vec]).astype(np.float32)
#     return flat_obs

# # 应用补丁
# GoalLevel1.__init__ = patched_init
# GoalLevel1.build_observation_space = patched_build_observation_space
# GoalLevel1.obs = patched_obs
# print("✅ 成功应用环境 Monkey Patch (26维模式)")

# # =================================================================
# # 2. 配置部分
# # =================================================================
# # ⚠️ 请确认此路径是正确的模型路径
# LOG_DIR = './runs/PPOLag-{SafetyPointGoal1-v0}/seed-000-2026-02-07-19-09-38'

# SAVE_NAME = 'safety_gym_26dim_data.npz'
# NUM_SAMPLES = 50000 
# ENV_ID = 'SafetyPointGoal1-v0'

# # =================================================================
# # 3. 辅助函数
# # =================================================================
# def find_actor(obj, depth=0):
#     """递归搜索 Actor (策略网络)"""
#     if depth > 4: return None
#     if hasattr(obj, 'predict') and callable(getattr(obj, 'predict')):
#         if not isinstance(obj, omnisafe.Evaluator):
#             return obj
#     for attr_name in dir(obj):
#         if attr_name.startswith('__'): continue
#         try:
#             attr_obj = getattr(obj, attr_name)
#             res = find_actor(attr_obj, depth + 1)
#             if res: return res
#         except:
#             continue
#     return None

# def convert_to_numpy(data):
#     """【核心修复】通用转换函数：处理 Tensor、Device、维度"""
#     # 如果是 Tensor，先转 CPU 再转 Numpy
#     if isinstance(data, torch.Tensor):
#         data = data.detach().cpu().numpy()
    
#     # 如果是 numpy 数组
#     if isinstance(data, np.ndarray):
#         # 处理 batch 维度 (1, N) -> (N,)
#         if data.ndim > 1 and data.shape[0] == 1:
#             data = data.squeeze(0)
#         # 如果是 0 维数组 (scalar)，转为 python scalar
#         if data.ndim == 0:
#             data = data.item()
            
#     return data

# def clean_data_dict(dataset):
#     """最终清理，将 list 堆叠为大 numpy array"""
#     cleaned_data = {}
#     for k, v in dataset.items():
#         # 这里假设 v 中的元素已经是处理好的 numpy array 或 scalar
#         arr = np.array(v)
        
#         if k in ['terminals', 'timeouts']:
#             arr = arr.astype(np.bool_)
#         else:
#             arr = arr.astype(np.float32)
            
#         cleaned_data[k] = arr
#     return cleaned_data

# if __name__ == '__main__':
#     # =================================================================
#     # 4. 加载模型与环境
#     # =================================================================
#     evaluator = omnisafe.Evaluator()
#     # 自动尝试加载模型
#     model_files = ['model.pt', 'epoch-500.pt', 'epoch-10.pt']
#     loaded = False
#     for mf in model_files:
#         try:
#             evaluator.load_saved(save_dir=LOG_DIR, model_name=mf)
#             print(f"✅ 成功加载模型: {mf}")
#             loaded = True
#             break
#         except Exception as e:
#             continue
    
#     if not loaded:
#         print(f"❌ 无法在 {LOG_DIR} 中找到模型文件: {model_files}")
#         # 如果你需要强行继续（比如只有 epoch-100.pt），请手动修改上面的列表或这里
#         # raise FileNotFoundError("Model file not found")

#     actor = find_actor(evaluator)
#     if actor is None:
#         raise RuntimeError("❌ 无法找到 Actor 网络，请检查模型加载路径。")
    
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     actor.to(device)
#     actor.eval()
    
#     env = evaluator._env
#     print(f"当前环境观测空间: {env.observation_space.shape}")
#     assert env.observation_space.shape == (26,), "❌ 环境维度依然不对！Patch 未生效。"

#     # =================================================================
#     # 5. 采集循环
#     # =================================================================
#     dataset = {
#         'observations': [],
#         'actions': [],
#         'next_observations': [], 
#         'rewards': [],
#         'costs': [],             
#         'terminals': [],
#         'timeouts': []
#     }

#     obs, _ = env.reset()
#     episode_step = 0
#     collected_steps = 0

#     print(f"开始采集 {NUM_SAMPLES} 步数据...")

#     while collected_steps < NUM_SAMPLES:
#         # 模型预测
#         with torch.no_grad():
#             # 确保 obs 是 Tensor 且在 device 上
#             obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(device)
#             if obs_tensor.ndim == 1:
#                 obs_tensor = obs_tensor.unsqueeze(0) # (1, 26)
            
#             act = actor.predict(obs_tensor, deterministic=False) 
            
#         act_cpu = act.squeeze(0).cpu()
        
#         # 环境步进
#         # 注意：env.step 返回的 reward/cost 可能是 Tensor 也可能是 float
#         next_obs, reward, cost, terminated, truncated, info = env.step(act_cpu)
        
#         # 【关键修复】存入前立刻转为 Numpy/Scalar
#         # 避免后续出现 "only one element tensors can be converted..." 错误
#         safe_obs = convert_to_numpy(obs)
#         safe_act = convert_to_numpy(act_cpu)
#         safe_next_obs = convert_to_numpy(next_obs)
#         safe_reward = convert_to_numpy(reward)
#         safe_cost = convert_to_numpy(cost)
        
#         dataset['observations'].append(safe_obs)
#         dataset['actions'].append(safe_act)
#         dataset['next_observations'].append(safe_next_obs)
#         dataset['rewards'].append(safe_reward)
#         dataset['costs'].append(safe_cost)
#         dataset['terminals'].append(terminated)
#         dataset['timeouts'].append(truncated)
        
#         obs = next_obs
#         episode_step += 1
#         collected_steps += 1
        
#         if terminated or truncated:
#             obs, _ = env.reset()
#             episode_step = 0
            
#         if collected_steps % 5000 == 0:
#             print(f"进度: {collected_steps} / {NUM_SAMPLES}")

#     # =================================================================
#     # 6. 保存数据
#     # =================================================================
#     print("正在处理并保存数据...")
#     final_data = clean_data_dict(dataset)
    
#     save_path = os.path.join(LOG_DIR, SAVE_NAME)
#     np.savez(save_path, **final_data)
    
#     print(f"✅ 数据采集完成！")
#     print(f"保存路径: {save_path}")
#     print(f"观测维度: {final_data['observations'].shape}")
#     print(f"成本总数(Unsafe steps): {np.sum(final_data['costs'] > 0)}")



##########################2

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
# # 1. Monkey Patch (确保和你训练PPO时的环境一致！)
# # =================================================================
# # ⚠️ 重要提示：如果你的 PPO 模型是在"默认环境"下训练的，
# # 请注释掉下面这个 Patch，否则 PPO 会变傻！
# # 如果你的 PPO 确实是在 26维 环境下训练的，请保留。
# # =================================================================

# def patched_init(self, config):
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

# # =================================================================
# # 2. 配置
# # =================================================================
# LOG_DIR = './runs/PPOLag-{SafetyPointGoal1-v0}/seed-000-2026-02-07-19-09-38'
# SAVE_NAME = 'safety_gym_26dim_data_v2.npz' # 改个名字，v2
# NUM_SAMPLES = 200000  # 【增加数据量】 到 20万
# ENV_ID = 'SafetyPointGoal1-v0'

# # =================================================================
# # 3. 辅助函数
# # =================================================================
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

# def convert_to_numpy(data):
#     if isinstance(data, torch.Tensor):
#         data = data.detach().cpu().numpy()
#     if isinstance(data, np.ndarray):
#         if data.ndim > 1 and data.shape[0] == 1:
#             data = data.squeeze(0)
#         if data.ndim == 0:
#             data = data.item()
#     return data

# def clean_data_dict(dataset):
#     cleaned_data = {}
#     for k, v in dataset.items():
#         arr = np.array(v)
#         if k in ['terminals', 'timeouts']:
#             arr = arr.astype(np.bool_)
#         else:
#             arr = arr.astype(np.float32)
#         cleaned_data[k] = arr
#     return cleaned_data

# if __name__ == '__main__':
#     # 加载模型
#     evaluator = omnisafe.Evaluator()
#     model_loaded = False
#     # 优先加载训练最久的 epoch-500
#     for mf in ['epoch-500.pt', 'model.pt']:
#         try:
#             evaluator.load_saved(save_dir=LOG_DIR, model_name=mf)
#             print(f"✅ 成功加载模型: {mf}")
#             model_loaded = True
#             break
#         except: continue
    
#     if not model_loaded:
#         raise FileNotFoundError(f"❌ 在 {LOG_DIR} 没找到模型文件")

#     actor = find_actor(evaluator)
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     actor.to(device)
#     actor.eval()
    
#     env = evaluator._env
#     print(f"当前环境 Obs Dim: {env.observation_space.shape}")

#     # 数据容器
#     dataset = {'observations': [], 'actions': [], 'next_observations': [], 
#                'rewards': [], 'costs': [], 'terminals': [], 'timeouts': []}

#     obs, _ = env.reset()
#     collected_steps = 0
    
#     # 诊断变量
#     episode_reward = 0
#     success_count = 0
#     episode_count = 0

#     print(f"\n 开始采集 {NUM_SAMPLES} 步数据...")
#     print("⚠️  请观察下方的 [专家诊断] 输出，确保 PPO 正常工作！")

#     while collected_steps < NUM_SAMPLES:
        
#         with torch.no_grad():
#             obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(device)
#             if obs_tensor.ndim == 1: obs_tensor = obs_tensor.unsqueeze(0)
            
#             # 【改为 True】使用确定性策略，避免随机噪声导致画圆
#             act = actor.predict(obs_tensor, deterministic=True)
            
#         act_cpu = act.squeeze(0).cpu()
        
#         next_obs, reward, cost, terminated, truncated, info = env.step(act_cpu)
#         if collected_steps % 100 == 0:
#             print(f"\n [诊断 Step {collected_steps}]")
#             print(f"   Obs (Raw?): {obs[:5]} ... (看数值大小)")
#             print(f"   Act (Output): {act_cpu.numpy()} ... (看是不是卡死在边界或0)")
#             print(f"   Reward: {reward} | Cost: {cost}")
        
#         # 存储
#         dataset['observations'].append(convert_to_numpy(obs))
#         dataset['actions'].append(convert_to_numpy(act_cpu))
#         dataset['next_observations'].append(convert_to_numpy(next_obs))
#         dataset['rewards'].append(convert_to_numpy(reward))
#         dataset['costs'].append(convert_to_numpy(cost))
#         dataset['terminals'].append(terminated)
#         dataset['timeouts'].append(truncated)
        
#         obs = next_obs
#         collected_steps += 1
#         episode_reward += reward

#         if terminated or truncated:
#             # 统计成功率 (Goal 任务 Reward > 0 通常意味着靠近目标，具体看 Cost)
#             episode_count += 1
#             if cost == 0: # 简单粗暴判断：没死就算一次完整尝试
#                  pass 
            
#             obs, _ = env.reset()
#             episode_reward = 0
            
#         if collected_steps % 10000 == 0:
#             print(f"进度: {collected_steps} / {NUM_SAMPLES} | 当前 Episode Reward: {episode_reward:.2f}")

#     # 保存
#     print("\n 正在保存...")
#     final_data = clean_data_dict(dataset)
#     save_path = os.path.join(LOG_DIR, SAVE_NAME)
#     np.savez(save_path, **final_data)
    
#     #  最终诊断报告
#     total_cost = np.sum(final_data['costs'])
#     print(f"="*40)
#     print(f"✅ 采集完成: {save_path}")
#     print(f" 数据集体检报告:")
#     print(f"   - 总步数: {len(final_data['observations'])}")
#     print(f"   - 发生碰撞的总步数 (Total Cost): {total_cost}")
#     print(f"   - 动作均值: {np.mean(final_data['actions'], axis=0)}")
#     print(f"   - 动作方差: {np.std(final_data['actions'], axis=0)}")
#     print(f"="*40)
    
#     if np.std(final_data['actions']) < 0.05:
#         print("⚠️ 严重警告：采集到的动作方差极低！PPO 老师可能在‘装死’或‘画圆’。")
#         print("   原因可能是：训练时的环境(60维)和采集时的环境(Patch 26维)不匹配！")




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
# # 1. Monkey Patch (保持不变)
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
#     # 这是生成 Raw Obs 的核心函数
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

# # =================================================================
# # 2. 配置
# # =================================================================
# LOG_DIR = './runs/PPOLag-{SafetyPointGoal1-v0}/seed-000-2026-02-07-19-09-38'
# SAVE_NAME = 'safety_gym_raw_26dim.npz' # ⚠️ 改名！标记为 raw，避免混淆
# NUM_SAMPLES = 200000 
# ENV_ID = 'SafetyPointGoal1-v0'

# # =================================================================
# # 3. 辅助函数
# # =================================================================
# def convert_to_numpy(data):
#     if isinstance(data, torch.Tensor): data = data.detach().cpu().numpy()
#     if isinstance(data, np.ndarray):
#         if data.ndim > 1 and data.shape[0] == 1: data = data.squeeze(0)
#         if data.ndim == 0: data = data.item()
#     return data

# if __name__ == '__main__':
#     # 加载模型
#     evaluator = omnisafe.Evaluator()
#     evaluator.load_saved(save_dir=LOG_DIR, model_name='epoch-500.pt')
    
#     # 这里的 env 是被 Wrap 过的，输出的是 Normalized Obs
#     env = evaluator._env 
    
#     # 获取 Agent (Actor)
#     # 注意：我们使用 evaluator.agent.predict，它会自动处理归一化输入
#     agent = evaluator.agent

#     # 数据容器
#     dataset = {'observations': [], 'actions': [], 'costs': [], 'terminals': [], 'timeouts': []}

#     # Reset 拿到的是 Normalized Obs
#     norm_obs, _ = env.reset()
#     collected_steps = 0
    
#     print(f"\n 开始采集 {NUM_SAMPLES} 步 真·原始数据 (Raw Data)...")

#     while collected_steps < NUM_SAMPLES:
        
#         # 1. 策略决策 (PPO 需要 Normalized Obs)
#         obs_tensor = torch.as_tensor(norm_obs, dtype=torch.float32).to(agent.device)
#         act, _ = agent.predict(obs_tensor, deterministic=True)
#         act_cpu = act.squeeze(0).cpu()
        
#         # 2. 【核心修改】获取 Raw Obs 用于存储 
#         # 调用最底层环境的 obs() 方法，绕过所有 Wrapper
#         # 这样存下来的才是真实的物理数值
#         raw_obs_data = env.unwrapped.obs() 
        
#         # 3. 环境步进 (返回的是 Normalized Next Obs)
#         next_norm_obs, reward, cost, terminated, truncated, info = env.step(act_cpu)
        
#         # 诊断打印
#         if collected_steps % 5000 == 0:
#             print(f"进度: {collected_steps}/{NUM_SAMPLES}")
#             # 可以对比一下 raw 和 norm 的区别
#             # print(f"Raw: {raw_obs_data[:3]} | Norm: {norm_obs[:3]}")

#         # 4. 存储 (存 Raw Obs !)
#         dataset['observations'].append(convert_to_numpy(raw_obs_data)) 
#         dataset['actions'].append(convert_to_numpy(act_cpu))
#         dataset['costs'].append(convert_to_numpy(cost))
#         dataset['terminals'].append(terminated)
#         dataset['timeouts'].append(truncated)
        
#         norm_obs = next_norm_obs
#         collected_steps += 1
        
#         if terminated or truncated:
#             norm_obs, _ = env.reset()

#     # 保存
#     print("\n 正在保存...")
#     final_data = {k: np.array(v) for k, v in dataset.items()}
#     save_path = os.path.join(LOG_DIR, SAVE_NAME)
#     np.savez(save_path, **final_data)
    
#     #  最终体检
#     obs_std = np.std(final_data['observations'], axis=0)
#     print(f"="*40)
#     print(f"✅ 采集完成: {save_path}")
#     print(f" 新数据体检报告:")
#     print(f"   - 观测方差范围: {np.min(obs_std):.4f} / {np.max(obs_std):.4f}")
    
#     # 这一次，方差应该参差不齐，而不是都在 1.0 附近
#     if np.max(obs_std) < 0.5:
#         print("⚠️ 警告: 方差依然很小，请检查是否成功获取到了 Raw Obs。")
#     else:
#         print(" 完美！方差分布正常，这才是真正的 Raw Data。")

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
# # 1. Monkey Patch (保持不变)
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
#     # 这是生成 Raw Obs 的核心函数
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

# # =================================================================
# # 2. 配置
# # =================================================================
# LOG_DIR = './runs/PPOLag-{SafetyPointGoal1-v0}/seed-000-2026-02-07-19-09-38'
# SAVE_NAME = 'safety_gym_raw_26dim.npz' 
# NUM_SAMPLES = 200000 

# # =================================================================
# # 3. 辅助函数
# # =================================================================
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

# def convert_to_numpy(data):
#     if isinstance(data, torch.Tensor): data = data.detach().cpu().numpy()
#     if isinstance(data, np.ndarray):
#         if data.ndim > 1 and data.shape[0] == 1: data = data.squeeze(0)
#         if data.ndim == 0: data = data.item()
#     return data

# # 🔥 【核心修复】增强版环境挖掘函数 🔥
# def get_base_env(env):
#     """能够识别 SafetyGym .task 结构的深度挖掘机"""
#     current = env
#     depth = 0
#     print("\n🔍 正在挖掘底层环境结构...")
    
#     while True:
#         env_type = type(current).__name__
#         print(f"   [Layer {depth}] Type: {env_type}")
        
#         # 1. 检查是否有 .task (SafetyGym 特有结构)
#         if hasattr(current, 'task'):
#             print(f"     ✅ 发现 .task 属性! 尝试访问 task.obs()...")
#             if hasattr(current.task, 'obs'):
#                 return current.task # 找到了！
        
#         # 2. 检查本身是否有 .obs (我们 Patch 的就是这个)
#         # 注意：有些 Wrapper 也会转发 .obs，所以我们要确保它是绑定的方法
#         if hasattr(current, 'obs') and callable(getattr(current, 'obs')):
#              # 排除掉只是 getattr 转发的情况，确认一下
#              return current
        
#         # 3. 继续向下挖
#         if hasattr(current, '_env'):
#             current = current._env
#         elif hasattr(current, 'env'):
#             current = current.env
#         else:
#             # 挖到底了
#             print(f"   ❌ Layer {depth} 是最底层，但没有发现 .obs() 或 .task")
#             print(f"      Available attrs: {[a for a in dir(current) if not a.startswith('_')]}")
#             raise AttributeError("❌ 无法找到底层环境！请截图发给助手分析。")
        
#         depth += 1
#         if depth > 20: # 防止死循环
#             raise RecursionError("环境包装层级过深 (>20)！")

# if __name__ == '__main__':
#     # 1. 加载 PPO
#     evaluator = omnisafe.Evaluator()
#     model_loaded = False
#     for mf in ['epoch-500.pt', 'model.pt']:
#         try:
#             evaluator.load_saved(save_dir=LOG_DIR, model_name=mf)
#             print(f"✅ 成功加载模型: {mf}")
#             model_loaded = True
#             break
#         except: continue
    
#     if not model_loaded:
#         raise FileNotFoundError(f"❌ 在 {LOG_DIR} 没找到模型文件")
    
#     actor = find_actor(evaluator)
#     if actor is None:
#         try: actor = evaluator.actor
#         except: raise RuntimeError("无法找到 Actor 网络")

#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     if hasattr(actor, 'to'): actor.to(device)
    
#     env = evaluator._env 
    
#     # 🔥 挖掘底层环境
#     base_env = get_base_env(env)
#     print(f"✅ 成功锁定底层环境: {base_env}")

#     # 2. 准备采集
#     dataset = {'observations': [], 'actions': [], 'costs': [], 'terminals': [], 'timeouts': []}
    
#     norm_obs, _ = env.reset()
#     collected_steps = 0
    
#     print(f"\n🚀 开始采集 {NUM_SAMPLES} 步 真·原始数据 (Raw Data)...")
    
#     while collected_steps < NUM_SAMPLES:
        
#         # A. PPO 决策 (使用 Normalized Obs)
#         with torch.no_grad():
#             obs_tensor = torch.as_tensor(norm_obs, dtype=torch.float32).to(device)
#             if obs_tensor.ndim == 1: obs_tensor = obs_tensor.unsqueeze(0)
#             act = actor.predict(obs_tensor, deterministic=True)
        
#         act_cpu = act.squeeze(0).cpu()
        
#         # B. 🔥 获取 Raw Obs 🔥
#         # 如果 base_env 是 task 对象，直接调用
#         # 如果 base_env 是环境对象，也直接调用
#         raw_obs_data = base_env.obs() 
        
#         # C. 环境步进
#         next_norm_obs, reward, cost, terminated, truncated, info = env.step(act_cpu)
        
#         # D. 存储
#         dataset['observations'].append(convert_to_numpy(raw_obs_data)) 
#         dataset['actions'].append(convert_to_numpy(act_cpu))
#         dataset['costs'].append(convert_to_numpy(cost))
#         dataset['terminals'].append(terminated)
#         dataset['timeouts'].append(truncated)
        
#         norm_obs = next_norm_obs
#         collected_steps += 1
        
#         if terminated or truncated:
#             norm_obs, _ = env.reset()
            
#         if collected_steps % 10000 == 0:
#             print(f"进度: {collected_steps}/{NUM_SAMPLES}")

#     # 3. 保存
#     print("\n💾 正在保存...")
#     final_data = {k: np.array(v) for k, v in dataset.items()}
#     save_path = os.path.join(LOG_DIR, SAVE_NAME)
#     np.savez(save_path, **final_data)
    
#     print(f"✅ 采集完成: {save_path}")
    
#     # 4. 再次体检
#     obs_std = np.std(final_data['observations'], axis=0)
#     print(f"📊 新数据观测方差范围: {np.min(obs_std):.4f} / {np.max(obs_std):.4f}")
    
#     if np.max(obs_std) < 0.5:
#         print("⚠️ 警告: 方差依然很小，可能没拿到真数据！")
#     else:
#         print("🎉 完美！方差分布参差不齐，这才是真正的 Raw Data。")



import os
import torch
import numpy as np
import omnisafe
import safety_gymnasium
import gymnasium
from safety_gymnasium.assets.geoms import Hazards
from safety_gymnasium.tasks.safe_navigation.goal.goal_level1 import GoalLevel1
from safety_gymnasium.tasks.safe_navigation.goal.goal_level0 import GoalLevel0

# =================================================================
# 1. Monkey Patch (必须加，确保环境是 26 维)
# =================================================================
def patched_init(self, config):
    self.lidar_num_bins = 16
    self.lidar_max_dist = 3.0
    self.sensors_obs = ['accelerometer', 'velocimeter', 'gyro', 'magnetometer']
    self.task_name = 'GoalLevel1_Reproduction'
    config.update({'lidar_num_bins': 16, 'lidar_max_dist': 3.0, 'sensors_obs': self.sensors_obs, 'task_name': self.task_name})
    GoalLevel0.__init__(self, config=config)
    self.placements_conf.extents = [-1.5, -1.5, 1.5, 1.5]
    self._add_geoms(Hazards(num=2, keepout=0.18))

def patched_build_observation_space(self):
    self.observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32)

def patched_obs(self):
    # 这是生成 Raw Obs 的核心函数
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
print("✅ 环境 Patch 已应用 (26维)")

# =================================================================
# 2. 配置
# =================================================================
LOG_DIR = './runs/PPOLag-{SafetyPointGoal1-v0}/seed-000-2026-02-07-19-09-38'
SAVE_NAME = 'safety_gym_raw_26dim.npz' 
NUM_SAMPLES = 200000 

# =================================================================
# 3. 辅助函数
# =================================================================
def find_actor(obj, depth=0):
    if depth > 4: return None
    if hasattr(obj, 'predict') and callable(getattr(obj, 'predict')):
        if not isinstance(obj, omnisafe.Evaluator): return obj
    for attr_name in dir(obj):
        if attr_name.startswith('__'): continue
        try:
            attr_obj = getattr(obj, attr_name)
            res = find_actor(attr_obj, depth + 1)
            if res: return res
        except: continue
    return None

def convert_to_numpy(data):
    if isinstance(data, torch.Tensor): data = data.detach().cpu().numpy()
    if isinstance(data, np.ndarray):
        if data.ndim > 1 and data.shape[0] == 1: data = data.squeeze(0)
        if data.ndim == 0: data = data.item()
    return data

# 🔥 挖掘底层环境结构 (为了获取 Raw Obs)
def get_base_env(env):
    current = env
    depth = 0
    print("\n🔍 正在挖掘底层环境结构...")
    while True:
        # 1. 检查是否有 .task
        if hasattr(current, 'task') and hasattr(current.task, 'obs'):
            return current.task
        # 2. 检查是否有 .obs (Patch 的情况)
        if hasattr(current, 'obs') and callable(getattr(current, 'obs')):
             return current
        # 3. 向下挖
        if hasattr(current, '_env'): current = current._env
        elif hasattr(current, 'env'): current = current.env
        else:
            raise AttributeError("❌ 无法找到底层环境！")
        depth += 1
        if depth > 20: raise RecursionError("环境层级过深")

if __name__ == '__main__':
    # 1. 加载 PPO
    evaluator = omnisafe.Evaluator()
    model_loaded = False
    for mf in ['epoch-500.pt', 'model.pt']:
        try:
            evaluator.load_saved(save_dir=LOG_DIR, model_name=mf)
            print(f"✅ 成功加载模型: {mf}")
            model_loaded = True
            break
        except: continue
    
    if not model_loaded:
        raise FileNotFoundError(f"❌ 在 {LOG_DIR} 没找到模型文件")
    
    actor = find_actor(evaluator)
    if actor is None:
        raise RuntimeError("无法找到 Actor 网络")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if hasattr(actor, 'to'): actor.to(device)
    
    env = evaluator._env 
    base_env = get_base_env(env)
    print(f"✅ 成功锁定底层环境: {base_env}")

    # 2. 准备采集
    dataset = {'observations': [], 'actions': [], 'costs': [], 'terminals': [], 'timeouts': []}
    
    norm_obs, _ = env.reset()
    collected_steps = 0
    
    print(f"\n🚀 开始采集 {NUM_SAMPLES} 步 真·原始数据 (Raw Data)...")
    
    while collected_steps < NUM_SAMPLES:
        
        # A. PPO 决策
        with torch.no_grad():
            if isinstance(norm_obs, np.ndarray):
                obs_tensor = torch.as_tensor(norm_obs, dtype=torch.float32).to(device).unsqueeze(0)
            else:
                obs_tensor = norm_obs.to(device).unsqueeze(0)
                
            act = actor.predict(obs_tensor, deterministic=True)
        
        # 🔥【关键修复】准备传给环境的动作
        # 1. .cpu(): 解决 Device Mismatch
        # 2. .squeeze(0): 解决 Dimension Mismatch (1,2) -> (2,)
        act_env = act.cpu().squeeze(0)
        
        # B. 获取 Raw Obs
        raw_obs_data = base_env.obs() 
        
        # C. 环境步进 (传入修复后的 act_env)
        res = env.step(act_env)
        
        # 解包返回值
        if len(res) == 6:
            next_norm_obs, reward, cost, terminated, truncated, _ = res
        elif len(res) == 5:
            next_norm_obs, reward, cost, terminated, truncated = res
        
        # D. 存储 (存 act_env 的 numpy 版本)
        dataset['observations'].append(convert_to_numpy(raw_obs_data)) 
        dataset['actions'].append(convert_to_numpy(act_env))
        dataset['costs'].append(convert_to_numpy(cost))
        dataset['terminals'].append(convert_to_numpy(terminated))
        dataset['timeouts'].append(convert_to_numpy(truncated))
        
        norm_obs = next_norm_obs
        collected_steps += 1
        
        if terminated or truncated:
            norm_obs, _ = env.reset()
            
        if collected_steps % 10000 == 0:
            print(f"进度: {collected_steps}/{NUM_SAMPLES}")

    # 3. 保存
    print("\n💾 正在保存...")
    final_data = {k: np.array(v) for k, v in dataset.items()}
    save_path = os.path.join(LOG_DIR, SAVE_NAME)
    np.savez(save_path, **final_data)
    
    print(f"✅ 采集完成: {save_path}")
    
    # 4. 体检
    obs_std = np.std(final_data['observations'], axis=0)
    print(f"📊 新数据观测方差范围: {np.min(obs_std):.4f} / {np.max(obs_std):.4f}")
    
    # 统计成功率
    terminals = final_data['terminals']
    timeouts = final_data['timeouts']
    costs = final_data['costs']
    success_rate = np.sum(terminals) / (np.sum(terminals) + np.sum(timeouts))
    print(f"📊 采集轨迹成功率: {success_rate*100:.1f}%")
    if success_rate < 0.05:
        print("⚠️ 警告: 成功率极低！虽然代码跑通了，但采集的数据全是‘转圈圈’的数据。")
        print("   -> 建议: 重训 PPO 模型 (我可以给你重训脚本)。")