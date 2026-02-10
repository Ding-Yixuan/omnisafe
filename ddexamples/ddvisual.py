import os
import torch
import numpy as np
import imageio
import omnisafe
import safety_gymnasium
import gymnasium
from safety_gymnasium.assets.geoms import Hazards
# 引入原始类
from safety_gymnasium.tasks.safe_navigation.goal.goal_level1 import GoalLevel1
from safety_gymnasium.tasks.safe_navigation.goal.goal_level0 import GoalLevel0

# =================================================================
# 1. 【核心必须】植入 Monkey Patch (与训练/采集完全一致)
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
    
    # 修改环境: 2 Hazards
    self.placements_conf.extents = [-1.5, -1.5, 1.5, 1.5]
    self._add_geoms(Hazards(num=2, keepout=0.18))
    print("【Patch】环境地图已修改: 2 Hazards")

def patched_build_observation_space(self):
    """替换 build_observation_space"""
    self.observation_space = gymnasium.spaces.Box(
        low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32
    )

def patched_obs(self):
    """替换 obs 方法 (确保 26 维)"""
    # 1. Hazard Lidar (16维)
    lidar_vec = self._obs_lidar(self.hazards.pos, self.hazards.group) 
    
    # 2. Sensors (7维)
    acc = self.agent.get_sensor('accelerometer')[:2]
    vel = self.agent.get_sensor('velocimeter')[:2]
    gyro = self.agent.get_sensor('gyro')[-1:]
    mag = self.agent.get_sensor('magnetometer')[:2]
    sensor_vec = np.concatenate([acc, vel, gyro, mag])

    # 3. Goal (3维)
    vec = (self.goal.pos - self.agent.pos) @ self.agent.mat
    x, y = vec[0], vec[1]
    z = x + 1j * y
    dist = np.abs(z)
    dist = np.exp(-dist) 
    angle = np.angle(z)
    goal_vec = np.array([dist, np.cos(angle), np.sin(angle)])

    # 4. 拼接
    flat_obs = np.concatenate([sensor_vec, goal_vec, lidar_vec]).astype(np.float32)
    return flat_obs

# 应用补丁
GoalLevel1.__init__ = patched_init
GoalLevel1.build_observation_space = patched_build_observation_space
GoalLevel1.obs = patched_obs
print("✅ 成功应用环境 Monkey Patch (26维模式)")

# =================================================================
# 2. 辅助函数
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

# ================= 配置区域 =================
# 请确保这里指向你训练好的模型文件夹
LOG_DIR = './runs/PPOLag-{SafetyPointGoal1-v0}/seed-000-2026-02-09-17-49-50'
# runs/PPOLag-{SafetyPointGoal1-v0}/seed-000-2026-02-07-19-09-38
VIDEO_FILENAME = "safe_navigation_26dim.mp4"
MAX_STEPS = 2000  
CAMERA_NAME = 'fixedfar' 
# ===========================================

def main():
    print(f"正在初始化视频录制环境...")
    
    # 初始化评估器
    evaluator = omnisafe.Evaluator(render_mode='rgb_array')
    
    # 尝试加载模型
    model_name = 'model.pt' 
    try:
        evaluator.load_saved(save_dir=LOG_DIR, model_name=model_name, camera_name=CAMERA_NAME)
    except:
        print(f"⚠️ {model_name} 不存在，尝试 epoch-10.pt")
        evaluator.load_saved(save_dir=LOG_DIR, model_name='epoch-500.pt', camera_name=CAMERA_NAME)
    
    env = evaluator._env
    
    # 验证维度 
    print(f"环境观测维度: {env.observation_space.shape}")
    assert env.observation_space.shape == (26,), "❌ 维度错误！Patch 未生效，模型将无法运行。"

    actor = find_actor(evaluator)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    actor.to(device)
    actor.eval()

    obs, _ = env.reset()
    
    print(f"\n=== 开始录制 MP4 (保存为 {VIDEO_FILENAME}) ===")
    
    with imageio.get_writer(VIDEO_FILENAME, fps=30) as writer:
        step = 0
        total_cost = 0
        while step < MAX_STEPS:
            # 1. 模型预测
            with torch.no_grad():
                # 处理 numpy -> tensor
                obs_t = torch.as_tensor(obs, dtype=torch.float32).to(device)
                if obs_t.ndim == 1: obs_t = obs_t.unsqueeze(0)
                
                # 可视化使用确定性策略
                act = actor.predict(obs_t, deterministic=True)
            
            # 2. 渲染画面
            try:
                frame = env.render()
                writer.append_data(frame)
            except Exception as e:
                print(f"渲染帧失败: {e}")

            # 3. 执行动作 (【核心修复】保持 Tensor 但转到 CPU)
            # act 是 (1, action_dim) 的 Tensor
            # .squeeze(0) -> (action_dim,)
            # .cpu() -> 移动到 CPU，因为 OmniSafe Wrapper 的参数在 CPU 上
            act_step = act.squeeze(0).cpu() 
            
            obs, reward, cost, terminated, truncated, info = env.step(act_step)
            
            # 2. 累加 Cost
            cost_val = cost.item() if hasattr(cost, 'item') else cost
            total_cost += cost_val  # 只要撞了就 +1

            # 3. 打印时显示【当前瞬时Cost】和【累计总Cost】
            if step % 100 == 0:
                print(f"录制进度: {step}/{MAX_STEPS} | 当前Cost: {cost_val:.0f} | 累计Cost: {total_cost:.0f}")
                # 如果累计Cost在猛增，说明中间一直在撞

            step += 1
            
            if terminated or truncated:
                print(f"--- 回合结束，本回合总撞击次数: {total_cost:.0f} ---")
                obs, _ = env.reset()
                # total_cost = 0 # 如果你想看跨回合的总数就别重置，想看单回合的就在这里重置

    print(f"✅ 视频保存成功！文件位置: {os.path.abspath(VIDEO_FILENAME)}")

if __name__ == "__main__":
    main()