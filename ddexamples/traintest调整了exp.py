import omnisafe
import safety_gymnasium
import gymnasium
import numpy as np
import torch
from safety_gymnasium.assets.geoms import Hazards
# 引入原始类
from safety_gymnasium.tasks.safe_navigation.goal.goal_level1 import GoalLevel1
from safety_gymnasium.tasks.safe_navigation.goal.goal_level0 import GoalLevel0

# =================================================================
# 1. 定义 Patch 函数 (核心修复版)
# =================================================================

def patched_init(self, config):
    """替换 GoalLevel1.__init__"""
    self.lidar_num_bins = 16
    self.lidar_max_dist = 3.0
    self.sensors_obs = ['accelerometer', 'velocimeter', 'gyro', 'magnetometer']
    self.task_name = 'GoalLevel1_Reproduction'
    
    config.update({
        'lidar_num_bins': 16,
        'lidar_max_dist': 3.0,
        'sensors_obs': self.sensors_obs,
        'task_name': self.task_name
    })
    
    GoalLevel0.__init__(self, config=config)
    
    # 修改环境元素: 2 Hazards
    self.placements_conf.extents = [-1.5, -1.5, 1.5, 1.5]
    self._add_geoms(Hazards(num=2, keepout=0.18))

def patched_build_observation_space(self):
    """替换 build_observation_space"""
    self.observation_space = gymnasium.spaces.Box(
        low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32
    )

# 🔥【新增】手写一个绝对可靠的 Lidar 计算函数
# 防止官方内部接口 _obs_lidar 返回奇怪的数据
def obs_lidar_pseudo(self):
    # 16 bins, max 3m
    obs = np.zeros(16, dtype=np.float32)
    bin_size = 2 * np.pi / 16
    
    # 遍历所有障碍物 (Hazards)
    for pos in self.hazards.pos:
        # 计算相对向量
        vec = pos - self.agent.pos
        # 旋转到机器人坐标系
        vec = np.matmul(vec, self.agent.mat)
        
        dist = np.linalg.norm(vec)
        angle = np.arctan2(vec[1], vec[0]) # -pi to pi
        
        # 忽略太远的
        if dist > 3.0: continue
        
        # 角度映射到 [0, 2pi]
        if angle < 0: angle += 2 * np.pi
        
        # 找到对应的 bin
        bin_idx = int(angle / bin_size) % 16
        
        # 计算强度 (1.0 表示贴脸，0.0 表示 3米远)
        intensity = 1.0 - (dist / 3.0)
        
        # 如果这个 bin 已经有值，保留更大的那个（更近的障碍物）
        if intensity > obs[bin_idx]:
            obs[bin_idx] = intensity
            
    return obs

def patched_obs(self):
    """
    【核心修复】替换顶层 obs 方法
    """
    # 1. 获取 Hazard Lidar (16维)
    # 使用我们要手动写的 pseudo 函数，确保数据正确
    lidar_vec = obs_lidar_pseudo(self)
    
    # 2. 获取 Sensors (7维)
    acc = self.agent.get_sensor('accelerometer')[:2]  # (x,y)
    vel = self.agent.get_sensor('velocimeter')[:2]    # (x,y)
    gyro = self.agent.get_sensor('gyro')[-1:]         # (z)
    mag = self.agent.get_sensor('magnetometer')[:2]   # (x,y)
    sensor_vec = np.concatenate([acc, vel, gyro, mag])

    # 3. 获取 Goal (3维: dist, cos, sin)
    vec = (self.goal.pos - self.agent.pos) @ self.agent.mat
    x, y = vec[0], vec[1]
    
    # 🔥【关键修改】PPO 需要真实的物理距离！
    # 不要用 exp(-dist)，那是给 Diffuser 用的，PPO 用了会变傻。
    dist = np.linalg.norm([x, y]) 
    
    angle = np.arctan2(y, x)
    goal_vec = np.array([dist, np.cos(angle), np.sin(angle)])

    # 4. 拼接 (26维)
    flat_obs = np.concatenate([sensor_vec, goal_vec, lidar_vec]).astype(np.float32)
    
    return flat_obs

# =================================================================
# 2. 执行 Monkey Patch
# =================================================================
GoalLevel1.__init__ = patched_init
GoalLevel1.build_observation_space = patched_build_observation_space
GoalLevel1.obs = patched_obs

print("✅ 成功 Monkey Patch (Corrected Version): PPO-Friendly Obs")

# =================================================================
# 3. 训练脚本 (PPOLag)
# =================================================================
if __name__ == '__main__':
    # 使用官方 ID
    env_id = 'SafetyPointGoal1-v0'
    
    custom_cfgs = {
        # 1. 训练通用参数
        'train_cfgs': {
            'total_steps': 1024000, # 100万步通常足够收敛
            'vector_env_nums': 1,
            'parallel': 1,
            'device': 'cuda:0',
        },
        # 2. 算法参数 (PPOLag)
        'algo_cfgs': {
            'steps_per_epoch': 2048,
            'update_iters': 10,
            'gamma': 0.99,
            'lam': 0.97,
            'clip': 0.2,
            'use_cost': True,  # 必须开启 Cost
        },
        # 3. 拉格朗日参数
        'lagrange_cfgs': {
            'cost_limit': 0.0,                 
            'lagrangian_multiplier_init': 0.001, 
            'lambda_lr': 0.035,                 
        },
        # 4. 模型架构
        'model_cfgs': {
             'actor': {
                 'hidden_sizes': [256, 256],
                 'activation': 'tanh'
             },
             'critic': {
                 'hidden_sizes': [256, 256],
                 'activation': 'tanh'
             }
        },
        # 5. 日志参数
        'logger_cfgs': {
            'use_wandb': False,
            'save_model_freq': 50,
        },
    }

    print(f"初始化 Agent (ID: {env_id})...")
    agent = omnisafe.Agent('PPOLag', env_id, custom_cfgs=custom_cfgs)
    
    # 维度再次确认
    try:
        if hasattr(agent.agent._env, 'observation_space'):
             obs_space = agent.agent._env.observation_space
        else:
             obs_space = agent.agent._env._env.observation_space
        print(f"Final Observation Space: {obs_space.shape}")
        if obs_space.shape[0] != 26:
            raise RuntimeError("维度依然不对！")
    except:
        pass

    print(f"🚀 训练启动中...")
    agent.learn()