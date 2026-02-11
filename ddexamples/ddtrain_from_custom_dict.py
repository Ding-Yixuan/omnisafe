import omnisafe
import safety_gymnasium
import gymnasium
import numpy as np
from safety_gymnasium.assets.geoms import Hazards
# 引入原始类
from safety_gymnasium.tasks.safe_navigation.goal.goal_level1 import GoalLevel1
from safety_gymnasium.tasks.safe_navigation.goal.goal_level0 import GoalLevel0

# =================================================================
# 1. 定义 Patch 函数
# =================================================================

def patched_init(self, config):
    """替换 GoalLevel1.__init__"""
    # A. 预定义属性
    self.lidar_num_bins = 16
    self.lidar_max_dist = 3.0
    self.sensors_obs = ['accelerometer', 'velocimeter', 'gyro', 'magnetometer']
    self.task_name = 'GoalLevel1_Reproduction'
    
    # B. 强制更新 Config
    config.update({
        'lidar_num_bins': 16,
        'lidar_max_dist': 3.0,
        'sensors_obs': self.sensors_obs,
        'task_name': self.task_name
    })
    
    # C. 调用父类构造
    GoalLevel0.__init__(self, config=config)
    
    # D. 修改环境元素
    self.placements_conf.extents = [-1.5, -1.5, 1.5, 1.5]
    self._add_geoms(Hazards(num=2, keepout=0.2))

def patched_build_observation_space(self):
    """替换 build_observation_space"""
    self.observation_space = gymnasium.spaces.Box(
        low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32
    )

def patched_obs(self):
    """
    【核心修复】替换顶层 obs 方法
    """
    # ------------------------------------------------------------------
    # 1. 获取 Hazard Lidar (16维)
    # 修复点：_obs_lidar 需要传入 (positions, group) 两个参数
    # ------------------------------------------------------------------
    lidar_vec = self._obs_lidar(self.hazards.pos, self.hazards.group) 
    
    # 2. 获取 Sensors (7维)
    acc = self.agent.get_sensor('accelerometer')[:2]  # (x,y)
    vel = self.agent.get_sensor('velocimeter')[:2]    # (x,y)
    gyro = self.agent.get_sensor('gyro')[-1:]         # (z)
    mag = self.agent.get_sensor('magnetometer')[:2]   # (x,y)
    sensor_vec = np.concatenate([acc, vel, gyro, mag])

    # 3. 获取 Goal (3维: dist, cos, sin)
    vec = (self.goal.pos - self.agent.pos) @ self.agent.mat
    x, y = vec[0], vec[1]
    
    # 论文特有的变换
    z = x + 1j * y
    dist = np.abs(z)
    dist = np.exp(-dist) 
    angle = np.angle(z)
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

print("成功 Monkey Patch: __init__, build_observation_space, obs")

# =================================================================
# 3. 训练脚本
# =================================================================
env_id = 'SafetyPointGoal1-v0'

###########################ppo
# if __name__ == '__main__':
#     custom_cfgs = {
#         'train_cfgs': {
#             'total_steps': 1024000,
#             'vector_env_nums': 1,
#             'parallel': 1,
#             'device': 'cuda:0',
#         },
#         'algo_cfgs': {
#             'steps_per_epoch': 2048,
#             'update_iters': 10,
#             'use_cost': False, 
#         },
#         'logger_cfgs': {
#             'use_wandb': False,
#             'save_model_freq': 50,
#         },
#         'model_cfgs': {
#              'actor': {
#                  'hidden_sizes': [256, 256],
#                  'activation': 'tanh'
#              },
#              'critic': {
#                  'hidden_sizes': [256, 256],
#                  'activation': 'tanh'
#              }
#         },
#     }


###########################ppolag
if __name__ == '__main__':
    # 使用官方 ID (我们已经 Patch 了它的底层逻辑)
    env_id = 'SafetyPointGoal1-v0'
    
    custom_cfgs = {
        # 1. 训练通用参数
        'train_cfgs': {
            'total_steps': 1024000,
            'vector_env_nums': 1,
            'parallel': 1,
            'device': 'cuda:0',
        },
        # 2. 算法参数
        'algo_cfgs': {
            'steps_per_epoch': 2048,
            'update_iters': 10,
            'gamma': 0.99,
            'lam': 0.97,
            'clip': 0.2,
            'use_cost': True,  # 【关键】PPOLag 必须开启 Cost
        },
        # 3. 拉格朗日参数 (严格复现论文 Table I)
        'lagrange_cfgs': {
            'cost_limit': 0,                 # 论文  设为 0
            'lagrangian_multiplier_init': 1.0, # 论文  Table I
            'lambda_lr': 0.01,                 # 论文  Table I
        },
        # 4. 模型架构 (可选：复现论文网络结构)
        'model_cfgs': {
             'actor': {
                 'hidden_sizes': [256, 256],   # 论文 
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
    # agent = omnisafe.Agent('PPO', env_id, custom_cfgs=custom_cfgs)
    agent = omnisafe.Agent('PPOLag', env_id, custom_cfgs=custom_cfgs)
    
    # =================================================================
    # 4. 最终验证
    # =================================================================
    try:
        if hasattr(agent.agent._env, 'observation_space'):
             obs_space = agent.agent._env.observation_space
        else:
             obs_space = agent.agent._env._env.observation_space
    except:
        # Fallback for debug
        obs_space = gymnasium.spaces.Box(-np.inf, np.inf, (0,))

    obs_shape = obs_space.shape
    print("\n" + "="*50)
    print(f"Final Agent Observation Space: {obs_shape}")
    
    if obs_shape == (26,):
        print("✅ SUCCESS: Environment is strictly aligned to 26 dims.")
    else:
        print("❌ FAIL Details:")
        print(f"  - Actual Shape: {obs_shape}")
        # 如果还是错的，说明 build_observation_space 没生效
        raise RuntimeError(f"❌ FAIL: Expected (26,) but got {obs_shape}.")
    print("="*50 + "\n")

    print(f"训练启动中...")
    agent.learn()