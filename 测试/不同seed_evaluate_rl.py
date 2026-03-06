import os
import glob
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
# 2. 单个模型的评估函数 (🔥 修改：只返回原始列表，不在内部算均值)
# =================================================================
def evaluate_single_seed(log_dir, model_name, seed_name, num_episodes=50):
    print(f"\n🚀 正在评估: {model_name} (Seed: {seed_name}) -> {log_dir}")
    
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

            obs, reward, cost, terminated, truncated, info = env.step(action_tensor) 
            
            ep_ret += reward
            ep_cost += cost.item() if hasattr(cost, 'item') else cost
            
            if terminated or truncated:
                break
                
        ep_returns.append(ep_ret)
        ep_costs.append(ep_cost)
        
        # 存入明细，增加 Seed 标识
        details_list.append({
            'Model': model_name,
            'Seed_Dir': seed_name,
            'Episode': ep + 1,
            'Reward': ep_ret,
            'Cost': ep_cost
        })
        
    return ep_returns, ep_costs, details_list

# =================================================================
# 3. 主程序：自动发现同组 Seed 并批量评估
# =================================================================
if __name__ == "__main__":
    # 🔥 设置包含所有 seed 文件夹的根目录
    BASE_DIR = "./runs/PPOLag-{SafetyPointGoal1-v0}"
    
    # 🔥 设置你要评估的 cost limits
    COST_LIMITS = [0, 3, 5, 10]
    
    NUM_EPISODES_PER_SEED = 50 
    
    all_summaries = []
    all_details = []
    
    for cost in COST_LIMITS:
        model_name = f"PPOLag_Cost{cost}"
        
        # 自动匹配包含该 costlimit 的文件夹 
        # 例如: seed-*-costlimit3_* search_pattern = os.path.join(BASE_DIR, f"*costlimit{cost}_*")
        search_pattern = os.path.join(BASE_DIR, f"*costlimit{cost}_*")
        matched_dirs = glob.glob(search_pattern)
        
        if not matched_dirs:
            print(f"\n⚠️ 找不到 {model_name} 的任何 seed 文件夹，跳过。")
            continue
            
        print(f"\n" + "="*50)
        print(f"🎯 找到 {model_name} 的 {len(matched_dirs)} 个 Seed 文件夹。开始聚合评估...")
        print("="*50)
        
        combined_returns = []
        combined_costs = []
        
        # 遍历同一个 cost limit 下的所有 seed 文件夹
        for log_dir in matched_dirs:
            seed_name = os.path.basename(log_dir).split('-')[1] # 提取如 "000"
            
            ep_rets, ep_costs, details = evaluate_single_seed(
                log_dir=log_dir, 
                model_name=model_name, 
                seed_name=seed_name,
                num_episodes=NUM_EPISODES_PER_SEED
            )
            
            combined_returns.extend(ep_rets)
            combined_costs.extend(ep_costs)
            all_details.extend(details)
            
        # 聚合所有的 seed 数据计算总的均值和方差
        mean_ret, std_ret = np.mean(combined_returns), np.std(combined_returns)
        mean_cost, std_cost = np.mean(combined_costs), np.std(combined_costs)

        print("\n" + "*"*50)
        print(f"📊 {model_name} 总体评估结果 ({len(matched_dirs)} Seeds x {NUM_EPISODES_PER_SEED} Eps = {len(combined_returns)} 汇总):")
        print(f"  🏆 Reward: {mean_ret:.2f} ± {std_ret:.2f}")
        print(f"  ⚠️ Cost:   {mean_cost:.2f} ± {std_cost:.2f}")
        print("*"*50)
        
        all_summaries.append({
            'Model': model_name,
            'Total_Seeds': len(matched_dirs),
            'Total_Episodes': len(combined_returns),
            'Reward_Mean': mean_ret,
            'Reward_Std': std_ret,
            'Cost_Mean': mean_cost,
            'Cost_Std': std_cost
        })

    # ==========================================
    # 保存到 CSV
    # ==========================================
    if all_summaries:
        df_summary = pd.DataFrame(all_summaries)
        df_summary['Reward (Mean ± Std)'] = df_summary.apply(lambda row: f"{row['Reward_Mean']:.2f} ± {row['Reward_Std']:.2f}", axis=1)
        df_summary['Cost (Mean ± Std)'] = df_summary.apply(lambda row: f"{row['Cost_Mean']:.2f} ± {row['Cost_Std']:.2f}", axis=1)
        
        summary_file = 'rl_eval_summary_combined.csv'
        df_summary.to_csv(summary_file, index=False)
        print(f"\n✅ 聚合汇总统计已保存至: {summary_file}")
        
        df_details = pd.DataFrame(all_details)
        details_file = 'rl_eval_details_combined.csv'
        df_details.to_csv(details_file, index=False)
        print(f"✅ 所有 Seed 的回合明细已保存至: {details_file}\n")
    else:
        print("\n⚠️ 未生成 CSV，请检查 BASE_DIR 路径是否正确。")