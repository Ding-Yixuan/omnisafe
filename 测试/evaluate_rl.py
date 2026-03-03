import os
import torch
import numpy as np
import pandas as pd
import omnisafe
import safety_gymnasium
import gymnasium
from safety_gymnasium.assets.geoms import Hazards
from safety_gymnasium.tasks.safe_navigation.goal.goal_level1 import GoalLevel1
from safety_gymnasium.tasks.safe_navigation.goal.goal_level0 import GoalLevel0

# =================================================================
# 1. 严格对齐训练时的 Monkey Patch (保持 exp 设定)
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
    
    # 严格保持 exp
    z = x + 1j * y
    dist = np.abs(z)
    dist = np.exp(-dist) 
    angle = np.angle(z)
    goal_vec = np.array([dist, np.cos(angle), np.sin(angle)])

    return np.concatenate([sensor_vec, goal_vec, lidar_vec]).astype(np.float32)

GoalLevel1.__init__ = patched_init
GoalLevel1.build_observation_space = patched_build_observation_space
GoalLevel1.obs = patched_obs
print("✅ 评估环境 Monkey Patch 成功 (26维 exp 模式)")

# =================================================================
# 2. 评估函数 (返回所有回合的明细数据)
# =================================================================
def evaluate_policy(log_dir, model_name="PPO", num_episodes=50):
    print(f"\n🚀 开始评估模型: {model_name} -> {log_dir}")
    
    evaluator = omnisafe.Evaluator(render_mode='rgb_array')
    
    try:
        evaluator.load_saved(save_dir=log_dir, model_name='model.pt')
    except Exception as e:
        print(f"⚠️ 找不到 model.pt，尝试加载 epoch-500.pt... ({e})")
        evaluator.load_saved(save_dir=log_dir, model_name='epoch-500.pt')

    actor = None
    if hasattr(evaluator, '_actor') and evaluator._actor is not None:
        actor = evaluator._actor
    elif hasattr(evaluator, 'actor') and evaluator.actor is not None:
        actor = evaluator.actor
    else:
        raise RuntimeError("❌ 无法找到 Actor 网络！")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    actor.to(device)
    env = evaluator._env

    ep_returns = []
    ep_costs = []
    
    # 用于记录明细的列表
    details_list = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_ret, ep_cost = 0.0, 0.0
        
        for step in range(1000): 
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                
                if hasattr(actor, 'predict'):
                    raw_out = actor.predict(obs_t, deterministic=True)
                else:
                    raw_out = actor.act(obs_t, deterministic=True)
                
                action = raw_out[0] if isinstance(raw_out, tuple) else raw_out
                action_tensor = action.squeeze(0).cpu()

            obs, reward, cost, terminated, truncated, info = env.step(action_tensor) # ✅ 传入 Tensor
            
            ep_ret += reward
            ep_cost += cost.item() if hasattr(cost, 'item') else cost
            
            if terminated or truncated:
                break
                
        ep_returns.append(ep_ret)
        ep_costs.append(ep_cost)
        
        # 存入明细
        details_list.append({
            'Model': model_name,
            'Episode': ep + 1,
            'Reward': ep_ret,
            'Cost': ep_cost
        })
        
        print(f"  Episode {ep+1:02d}/{num_episodes} | Return: {ep_ret:6.2f} | Cost: {ep_cost:6.2f}")

    mean_ret, std_ret = np.mean(ep_returns), np.std(ep_returns)
    mean_cost, std_cost = np.mean(ep_costs), np.std(ep_costs)

    print("\n" + "="*50)
    print(f"📊 {model_name} 评估结果 ({num_episodes} Episodes):")
    print(f"  🏆 Reward: {mean_ret:.2f} ± {std_ret:.2f}")
    print(f"  ⚠️ Cost:   {mean_cost:.2f} ± {std_cost:.2f}")
    print("="*50)
    
    # 汇总数据字典
    summary_data = {
        'Model': model_name,
        'Episodes': num_episodes,
        'Reward_Mean': mean_ret,
        'Reward_Std': std_ret,
        'Cost_Mean': mean_cost,
        'Cost_Std': std_cost
    }
    
    return summary_data, details_list

# =================================================================
# 3. 主程序：批量评估并存入 CSV
# =================================================================
if __name__ == "__main__":
    MODELS_TO_EVALUATE = {
        "PPO": "./runs/PPO-{SafetyPointGoal1-v0}/seed-000-2026-02-11-21-17-42",
        "PPOLag": "./runs/PPOLag-{SafetyPointGoal1-v0}/seed-000-2026-02-27-18-28-38"
    }
    
    NUM_EPISODES = 50 
    
    all_summaries = []
    all_details = []
    
    for model_name, path in MODELS_TO_EVALUATE.items():
        if os.path.exists(path):
            summary, details = evaluate_policy(log_dir=path, model_name=model_name, num_episodes=NUM_EPISODES)
            all_summaries.append(summary)
            all_details.extend(details)
        else:
            print(f"\n❌ 路径不存在，跳过 {model_name}: {path}")

    # ==========================================
    # 保存到 CSV
    # ==========================================
    if all_summaries:
        # 1. 保存汇总表格 (用于画论文里的表格)
        df_summary = pd.DataFrame(all_summaries)
        # 格式化一下，方便直接复制：添加 "Mean ± Std" 列
        df_summary['Reward (Mean ± Std)'] = df_summary.apply(lambda row: f"{row['Reward_Mean']:.2f} ± {row['Reward_Std']:.2f}", axis=1)
        df_summary['Cost (Mean ± Std)'] = df_summary.apply(lambda row: f"{row['Cost_Mean']:.2f} ± {row['Cost_Std']:.2f}", axis=1)
        
        summary_file = 'rl_eval_summary.csv'
        df_summary.to_csv(summary_file, index=False)
        print(f"\n✅ 汇总统计已保存至: {summary_file}")
        
        # 2. 保存明细表格 (用于深度分析或画箱线图)
        df_details = pd.DataFrame(all_details)
        details_file = 'rl_eval_details.csv'
        df_details.to_csv(details_file, index=False)
        print(f"✅ 回合明细已保存至: {details_file}\n")
    else:
        print("\n⚠️ 没有找到任何有效模型，未生成 CSV。请检查路径配置！")