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
# 1. 环境定义 (严格保持 exp 补丁，确保模型能认出环境)
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
    
    # 严格保持 exp 设定
    z = x + 1j * y
    dist = np.exp(-np.abs(z)) 
    angle = np.angle(z)
    goal_vec = np.array([dist, np.cos(angle), np.sin(angle)])
    return np.concatenate([sensor_vec, goal_vec, lidar_vec]).astype(np.float32)

GoalLevel1.__init__ = patched_init
GoalLevel1.build_observation_space = patched_build_observation_space
GoalLevel1.obs = patched_obs

# =================================================================
# 2. TTC 物理计算逻辑 (你的优秀设计，保留！)
# =================================================================
def calculate_ttc(env_task, agent_pos, agent_vel):
    min_ttc = float('inf')
    try:
        hazards_pos = env_task.hazards.pos
    except:
        if hasattr(env_task, '_geoms') and 'hazards' in env_task._geoms:
            hazards_pos = env_task._geoms['hazards'].pos
        else:
            hazards_pos = []
            
    hazards_radius = 0.20
    agent_radius = 0.10
    collision_threshold = agent_radius + hazards_radius
    
    if len(hazards_pos) == 0:
        return float('inf')

    for h_pos in hazards_pos:
        h_pos_2d = h_pos[:2] 
        rel_pos = h_pos_2d - agent_pos
        dist_center = np.linalg.norm(rel_pos)
        dist_surface = dist_center - collision_threshold
        
        if dist_surface <= 0: 
            return 0.0 
        
        if dist_center > 1e-6: 
            direction = rel_pos / dist_center
        else: 
            direction = np.zeros(2)
            
        v_proj = np.dot(agent_vel, direction)
        
        if v_proj > 1e-4: 
            ttc = dist_surface / v_proj
            if ttc < min_ttc:
                min_ttc = ttc
    return min_ttc

# =================================================================
# 3. 稳健的数据采集主程序
# =================================================================
def collect_dataset(log_dir, save_path, total_steps=50000):
    print(f"\n🎥 开始采集增强数据集: {save_path}")
    
    # 1. 借用官方 Evaluator，最安全的加载方式！
    evaluator = omnisafe.Evaluator(render_mode='rgb_array')
    try:
        evaluator.load_saved(save_dir=log_dir, model_name='model.pt')
    except:
        evaluator.load_saved(save_dir=log_dir, model_name='epoch-500.pt')

    actor = evaluator._actor if hasattr(evaluator, '_actor') else evaluator.actor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    actor.to(device)
    
    # 拿到包含完整 Wrapper 的环境
    env = evaluator._env

    # 2. 初始化你的超强 Buffer
    dataset = {
        'obs': [], 'act': [], 'next_obs': [], 'rew': [], 'env_cost': [], 
        'done': [], 'ttc': [], 'is_safe': [], 'goal_pos': [], 
        'agent_pos': [], 'segment_id': []
    }
    
    TTC_THRESHOLD = 1.0
    current_segment = 0
    step_count = 0
    
    obs, _ = env.reset()
    
    while step_count < total_steps:
        # A. 动作预测
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            
            # deterministic=False 让它带有一点点随机性，这对 Diffuser 学习多样性轨迹至关重要
            if hasattr(actor, 'predict'):
                raw_out = actor.predict(obs_t, deterministic=False)
            else:
                raw_out = actor.act(obs_t, deterministic=False)
            
            action = raw_out[0] if isinstance(raw_out, tuple) else raw_out
            action_tensor = action.squeeze(0).cpu() # 保持为 Tensor 给环境

        # 获取底层 Task (用于提取物理坐标)
        # OmniSafe 的 env 包装得很深，通常 .unwrapped 才能拿到 GoalLevel1 对象
        curr_env = env
        while hasattr(curr_env, '_env'):
            curr_env = curr_env._env  # 一层一层往里剥
        env_task = curr_env.unwrapped.task  # 拿到最底层的物理 Task

        # B. 物理信息提取
        agent_pos = env_task.agent.pos[:2].copy()
        agent_vel = env_task.agent.vel[:2].copy()
        goal_pos = env_task.goal.pos[:2].copy()
        
        ttc_val = calculate_ttc(env_task, agent_pos, agent_vel)
        is_safe = 1 if ttc_val > TTC_THRESHOLD else 0

        # C. 执行环境
        next_obs, reward, cost, terminated, truncated, info = env.step(action_tensor)
        done = terminated or truncated

        # D. 存储 (这里我们把 action 存为 numpy，供后续 Diffuser 使用)
        # dataset['obs'].append(obs.copy())
        # dataset['act'].append(action_tensor.numpy().copy())
        # dataset['next_obs'].append(next_obs.copy())
        # dataset['rew'].append(reward)
        # dataset['env_cost'].append(cost.item() if hasattr(cost, 'item') else cost)
        # D. 存储 (自动处理 Tensor 转 Numpy)
        obs_np = obs.cpu().numpy().copy() if isinstance(obs, torch.Tensor) else obs.copy()
        next_obs_np = next_obs.cpu().numpy().copy() if isinstance(next_obs, torch.Tensor) else next_obs.copy()
        rew_val = reward.item() if hasattr(reward, 'item') else reward

        dataset['obs'].append(obs_np)
        dataset['act'].append(action_tensor.numpy().copy())
        dataset['next_obs'].append(next_obs_np)
        dataset['rew'].append(rew_val)
        dataset['env_cost'].append(cost.item() if hasattr(cost, 'item') else cost)
        dataset['done'].append(done)
        dataset['ttc'].append(ttc_val)
        dataset['is_safe'].append(is_safe)
        dataset['goal_pos'].append(goal_pos)
        dataset['agent_pos'].append(agent_pos)
        dataset['segment_id'].append(current_segment)

        step_count += 1
        
        if step_count % 5000 == 0:
            print(f"  已采集 {step_count}/{total_steps} 步... TTC Mean(最近百步): {np.mean(dataset['ttc'][-100:]):.2f}")

        # E. 循环管理
        if done:
            obs, _ = env.reset()
            current_segment += 1
        else:
            obs = next_obs

    # F. 压缩保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    final_data = {k: np.array(v) for k, v in dataset.items()}
    np.savez_compressed(save_path, **final_data)
    print(f"🎉 完美！数据已保存至: {save_path} (共 {current_segment+1} 条轨迹)")

# =================================================================
# 4. 执行
# =================================================================
if __name__ == "__main__":
    # 🔴 换成你刚才跑评估用的那个 runs 路径
    PPO_DIR = './runs/PPO-{SafetyPointGoal1-v0}/seed-000-2026-02-11-21-17-42'
    PPOLAG_DIR = './runs/PPOLag-{SafetyPointGoal1-v0}/seed-000-2026-02-27-18-28-38'
    
    # 采集量：10万步 足够训出一个极好的 Diffuser 了
    TOTAL_STEPS = 100000 
    
    if os.path.exists(PPO_DIR):
        collect_dataset(PPO_DIR, './data_pro/data_ppo_exp.npz', total_steps=TOTAL_STEPS)
    else:
        print(f"找不到 PPO 模型路径: {PPO_DIR}")
        
    if os.path.exists(PPOLAG_DIR):
        collect_dataset(PPOLAG_DIR, './data_pro/data_ppolag_exp.npz', total_steps=TOTAL_STEPS)
    else:
        print(f"找不到 PPOLag 模型路径: {PPOLAG_DIR}")