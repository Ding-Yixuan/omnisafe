import omnisafe
import torch
import numpy as np
import os
import safety_gymnasium
import gymnasium
from safety_gymnasium.assets.geoms import Hazards
from safety_gymnasium.tasks.safe_navigation.goal.goal_level1 import GoalLevel1
from safety_gymnasium.tasks.safe_navigation.goal.goal_level0 import GoalLevel0

# =================================================================
# 1. Monkey Patch (环境 26 维补丁 - 保持线性距离)
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
    lidar_vec = self._obs_lidar(self.hazards.pos, self.hazards.group) 
    acc = self.agent.get_sensor('accelerometer')[:2]
    vel = self.agent.get_sensor('velocimeter')[:2]
    gyro = self.agent.get_sensor('gyro')[-1:]
    mag = self.agent.get_sensor('magnetometer')[:2]
    sensor_vec = np.concatenate([acc, vel, gyro, mag])
    vec = (self.goal.pos - self.agent.pos) @ self.agent.mat
    x, y = vec[0], vec[1]
    z = x + 1j * y
    # dist = np.abs(z)  # ✅ 保持线性距离
    dist = np.exp(-np.abs(z)) 
    angle = np.angle(z)
    goal_vec = np.array([dist, np.cos(angle), np.sin(angle)])
    return np.concatenate([sensor_vec, goal_vec, lidar_vec]).astype(np.float32)

GoalLevel1.__init__ = patched_init
GoalLevel1.build_observation_space = patched_build_observation_space
GoalLevel1.obs = patched_obs
print("✅ 环境 Patch 已应用 (26维)")

# =================================================================
# 2. 辅助函数
# =================================================================

def calculate_ttc(env_task, agent_pos, agent_vel):
    """计算 TTC (修复维度问题)"""
    min_ttc = float('inf')
    try:
        hazards_pos = env_task.hazards.pos
        hazards_radius = 0.18
    except:
        hazards_pos = env_task._geoms['hazards'].pos
        hazards_radius = 0.18
    agent_radius = 0.05
    
    for h_pos in hazards_pos:
        # ✅ 修复点：h_pos 是 (3,)，我们只取前两维 (2,)
        h_pos_2d = h_pos[:2] 
        
        rel_pos = h_pos_2d - agent_pos
        dist_center = np.linalg.norm(rel_pos)
        dist_surface = dist_center - agent_radius - hazards_radius
        
        if dist_surface <= 0: return 0.0 # 已经碰撞
        
        if dist_center > 1e-6: 
            direction = rel_pos / dist_center
        else: 
            direction = np.zeros(2)
            
        v_proj = np.dot(agent_vel, direction)
        
        if v_proj > 0: # 正在靠近
            ttc = dist_surface / v_proj
        else: # 正在远离
            ttc = float('inf')
            
        if ttc < min_ttc:
            min_ttc = ttc
            
    return min_ttc
def get_real_task(adapter_env):
    """剥洋葱：找到底层 SafetyGymnasium 环境"""
    current = adapter_env
    for _ in range(10): 
        if hasattr(current, 'task'): return current.task
        elif hasattr(current, '_env'): current = current._env
        elif hasattr(current, 'env'): current = current.env
        elif hasattr(current, 'unwrapped'): current = current.unwrapped
        else: break
    if hasattr(current, 'envs'): return current.envs[0].task
    raise AttributeError("无法找到底层的 SafetyGymnasium Task")

def collect_split_trajectories(agent, num_episodes=500, save_path="./data_pro"):
    eval_env = agent.agent._env
    base_env = get_real_task(eval_env)
    
    print(f"✅ 成功连接到底层环境: {base_env.__class__.__name__}")
    
    # 1. 获取 Device
    if hasattr(agent.agent, '_device'):
        device = agent.agent._device
    else:
        try:
            device = next(agent.agent.actor.parameters()).device
        except:
            device = 'cuda:0'
    print(f"🖥️  正在使用计算设备: {device}")

    # 2. 获取 Actor 网络
    if hasattr(agent.agent, '_actor_critic'):
        actor_net = agent.agent._actor_critic.actor
    elif hasattr(agent.agent, 'actor_critic'):
        actor_net = agent.agent.actor_critic.actor
    else:
        raise AttributeError("❌ 无法在 collect 函数中找到 Actor 网络")
        
    # 3. 获取 Normalizer
    normalizer = agent.agent.obs_normalizer
    if hasattr(normalizer, 'eval'):
        normalizer.eval()

    dataset = {
        'obs': [], 'act': [], 'next_obs': [], 'rew': [], 'done': [],
        'ttc': [], 'is_safe': [], 'env_cost': [], 
        'goal_pos': [], 'agent_pos': [], 'segment_id': []
    }
    
    TTC_THRESHOLD = 1.0 
    global_segment_counter = 0

    print(f"🚀 开始采集 {num_episodes} 条长序列...")
    
    for ep in range(num_episodes):
        o, _ = eval_env.reset()
        current_goal_pos = base_env.goal.pos[:2].copy()
        
        done = False
        trunc = False
        
        while not (done or trunc):
            # -------------------------------------------------
            # 核心预测逻辑
            # -------------------------------------------------
            with torch.no_grad():
                obs_tensor = torch.as_tensor(o, dtype=torch.float32).to(device)
                if obs_tensor.ndim == 1: obs_tensor = obs_tensor.unsqueeze(0)
                if normalizer is not None: obs_tensor = normalizer.normalize(obs_tensor)
                
                act_tensor = actor_net.predict(obs_tensor, deterministic=True)
                action_for_env = act_tensor 
                # 立即转 CPU Numpy 备用
                action_for_save = act_tensor.cpu().numpy().flatten()
            # -------------------------------------------------
            
            # 环境交互
            step_result = eval_env.step(action_for_env)
            if len(step_result) == 6:
                next_o, r, c, done, trunc, info = step_result
            elif len(step_result) == 5:
                next_o, r, done, trunc, info = step_result
                c = 0.0
            else:
                raise ValueError(f"环境返回了 {len(step_result)} 个值")
            
            # 物理信息
            agent_pos = base_env.agent.pos[:2]
            agent_vel = base_env.agent.vel[:2]
            real_goal_pos = base_env.goal.pos[:2]
            
            if np.linalg.norm(real_goal_pos - current_goal_pos) > 1e-4:
                global_segment_counter += 1
                current_goal_pos = real_goal_pos.copy()

            ttc_val = calculate_ttc(base_env, agent_pos, agent_vel)
            is_safe = 1 if ttc_val > TTC_THRESHOLD else 0
            
            # === 存入列表 (先不处理，最后统一洗) ===
            dataset['obs'].append(o)
            dataset['act'].append(action_for_save)
            dataset['next_obs'].append(next_o)
            dataset['rew'].append(r)
            dataset['env_cost'].append(c)
            dataset['done'].append(done or trunc)
            dataset['ttc'].append(ttc_val)
            dataset['is_safe'].append(is_safe)
            dataset['goal_pos'].append(real_goal_pos)
            dataset['agent_pos'].append(agent_pos)
            dataset['segment_id'].append(global_segment_counter)
            
            o = next_o
            
        if (ep+1) % 10 == 0:
            print(f"   Episode {ep+1}/{num_episodes} Finished. Total Segments: {global_segment_counter}")

    # =========================================================
    # 🧹 强力清洗函数：处理一切 Tensor/Cuda/List 问题
    # =========================================================
    def to_numpy_safe(data_list):
        cleaned = []
        for item in data_list:
            # 如果是 PyTorch Tensor (无论 CPU 还是 GPU)
            if isinstance(item, torch.Tensor):
                item = item.detach().cpu().numpy()
            # 如果是 Numpy 标量 (例如 float32)
            if isinstance(item, (np.floating, float)):
                item = float(item)
            if isinstance(item, (np.integer, int)):
                item = int(item)
            # 如果是 Numpy 数组，检查维度
            if isinstance(item, np.ndarray):
                if item.ndim == 2 and item.shape[0] == 1:
                    item = item.squeeze(0)
            
            cleaned.append(item)
        return np.array(cleaned)

    print("\n🧹 正在清洗并压缩数据...")
    final_dataset = {}
    
    for k, v in dataset.items():
        try:
            # 对每一列都执行清洗
            final_dataset[k] = to_numpy_safe(v)
        except Exception as e:
            print(f"⚠️ 警告: 字段 {k} 清洗失败: {e}")

    os.makedirs(save_path, exist_ok=True)
    file_name = os.path.join(save_path, "ppolag_shicbf.npz")
    np.savez_compressed(file_name, **final_dataset)
    
    print(f"\n🎉 恭喜! 数据采集完成，已保存至: {file_name}")
    print(f"    - obs shape: {final_dataset['obs'].shape}")
    print(f"    - act shape: {final_dataset['act'].shape}")
    print(f"    - done shape: {final_dataset['done'].shape}")
    print(f"    - is_safe mean: {np.mean(final_dataset['is_safe']):.2%}")

# =================================================================
# 3. 主程序入口
# =================================================================
if __name__ == '__main__':
    import os
    # 🎯 目标路径
    base_dir = "/home/lqz27/dyx_ws/omnisafe/runs/PPOLag-{SafetyPointGoal1-v0}/seed-000-2026-02-07-19-09-38/torch_save"
    path_337 = os.path.join(base_dir, "epoch-337.pt")
    path_500 = os.path.join(base_dir, "epoch-500.pt")
    
    if os.path.exists(path_337):
        SPECIFIC_MODEL_PATH = path_337
        print(f"🎯 锁定最佳模型: epoch-337.pt")
    elif os.path.exists(path_500):
        SPECIFIC_MODEL_PATH = path_500
        print(f"⚠️ 未找到 epoch-337，降级使用: epoch-500.pt")
    else:
        print(f"❌ 错误：在 {base_dir} 下没找到模型文件！")
        exit()

    env_id = 'SafetyPointGoal1-v0'
    custom_cfgs = {
        'train_cfgs': {
            'vector_env_nums': 1,
            'parallel': 1,
            'device': 'cuda:0',
        },
        'model_cfgs': {
             'actor': {
                 'hidden_sizes': [256, 256], 
                 'activation': 'tanh'
             }
        }
    }

    print("\nStep 1: 初始化 Agent...")
    agent = omnisafe.Agent('PPOLag', env_id, custom_cfgs=custom_cfgs)

    # 🔧 关键修复：手动注入 Normalizer
    if not hasattr(agent.agent, 'obs_normalizer') or agent.agent.obs_normalizer is None:
        print("🔧 检测到 Normalizer 未初始化，正在手动注入...")
        from omnisafe.common.normalizer import Normalizer
        
        if hasattr(agent.agent, '_device'):
            device = agent.agent._device
        else:
            device = 'cuda:0' 
            
        # 先初始化 (不传 device)
        agent.agent.obs_normalizer = Normalizer(shape=(26,))
        # 再移动
        if hasattr(agent.agent.obs_normalizer, 'to'):
            agent.agent.obs_normalizer.to(device)
            
        print(f"🔧 手动注入完成 (Device: {device})。")

    print(f"Step 2: 加载权重 -> {SPECIFIC_MODEL_PATH}")
    model_params = torch.load(SPECIFIC_MODEL_PATH, map_location='cuda:0')

    try:
        # 1. 寻找 actor_critic
        if hasattr(agent.agent, '_actor_critic'):
            model_interface = agent.agent._actor_critic
        else:
            model_interface = agent.agent.actor_critic

        # 2. 加载 Actor
        if 'pi' in model_params:
            model_interface.actor.load_state_dict(model_params['pi'])
            print("✅ Actor 权重加载成功")
        
        # 3. 加载 Normalizer
        if 'obs_normalizer' in model_params:
            if hasattr(agent.agent, 'obs_normalizer'):
                agent.agent.obs_normalizer.load_state_dict(model_params['obs_normalizer'])
                print("✅ Normalizer (agent-level) 加载成功")
            else:
                print("❌ 严重警告：模型包含 Normalizer 参数，但无法找到加载位置！")
        
    except Exception as e:
        print(f"❌ 加载出错: {e}")
        import traceback
        traceback.print_exc()
        exit()

    print("\nStep 3: 开始采集...")
    collect_split_trajectories(agent, num_episodes=500, save_path="./data_pro")