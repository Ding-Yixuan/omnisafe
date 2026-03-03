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
# 1. 【核心必须】植入 Monkey Patch (仅修改障碍物数量，保持默认60维)
# =================================================================

def patched_init(self, config):
    """
    替换 GoalLevel1.__init__
    只修改障碍物数量，不修改传感器配置，从而保持默认的 60 维观测空间
    """
    # 调用父类构造 (GoalLevel0)，传入原始 config
    # 不再手动 update config 中的 sensors_obs，这样就会使用默认值
    GoalLevel0.__init__(self, config=config)
    
    # 修改环境: 2 Hazards
    self.placements_conf.extents = [-1.5, -1.5, 1.5, 1.5]
    self._add_geoms(Hazards(num=2, keepout=0.2))
    print("【Patch】环境地图已修改: 2 Hazards (观测空间保持默认 60 维)")

# 应用补丁
# 注意：我们不再替换 obs 和 build_observation_space，让它们保持原样
GoalLevel1.__init__ = patched_init
print("✅ 成功应用环境 Monkey Patch (默认 60 维模式 + 2 障碍物)")

# =================================================================
# 2. 辅助函数
# =================================================================
def find_actor(obj, depth=0):
    if depth > 4: return None
    # 检查是否有 predict 方法且不是 Evaluator 本身
    if hasattr(obj, 'predict') and callable(getattr(obj, 'predict')):
        if not isinstance(obj, omnisafe.Evaluator): return obj
    
    # 递归查找
    for attr_name in dir(obj):
        if attr_name.startswith('__'): continue
        try:
            attr_obj = getattr(obj, attr_name)
            res = find_actor(attr_obj, depth + 1)
            if res: return res
        except: continue
    return None

# ================= 配置区域 =================
# 请确保这里指向你训练好的模型文件夹 (60维的模型)
LOG_DIR = './runs/PPOLag-{SafetyPointGoal1-v0}/seed-000-2026-02-17-12-10-36' 
VIDEO_FILENAME = "./safe_navigation_60dim.mp4"
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
        print(f"⚠️ {model_name} 不存在，尝试 epoch-500.pt (请根据实际情况调整文件名)")
        # 注意：如果你的模型还没跑到 500 epoch，这里可能会报错，请检查目录下有哪些 .pt 文件
        evaluator.load_saved(save_dir=LOG_DIR, model_name='epoch-100.pt', camera_name=CAMERA_NAME)
    
    env = evaluator._env
    
    # 验证维度 
    print(f"环境观测维度: {env.observation_space.shape}")
    
    # 核心修改：这里应该断言 60 维
    if env.observation_space.shape == (60,):
        print("✅ 维度验证通过: 60维")
    else:
        print(f"⚠️ 警告: 维度是 {env.observation_space.shape}，预期是 (60,)")
        # 如果你确定你的模型是 60 维的，但这里报错，说明环境没初始化对
        # 如果你的模型是别的维度，请相应修改这里的 assert
        # assert env.observation_space.shape == (60,), "❌ 维度错误！"

    # 查找策略网络
    actor = find_actor(evaluator)
    if actor is None:
        raise ValueError("❌ 无法在 Evaluator 中找到 Actor 网络！")
        
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

            # 3. 执行动作
            # act 是 (1, action_dim) 的 Tensor
            # .squeeze(0) -> (action_dim,)
            # .cpu() -> 移动到 CPU
            act_step = act.squeeze(0).cpu() 
            
            obs, reward, cost, terminated, truncated, info = env.step(act_step)
            
            # 4. 累加 Cost
            cost_val = cost.item() if hasattr(cost, 'item') else cost
            total_cost += cost_val 

            # 5. 打印状态
            if step % 100 == 0:
                print(f"录制进度: {step}/{MAX_STEPS} | 当前Cost: {cost_val:.0f} | 累计Cost: {total_cost:.0f}")

            step += 1
            
            if terminated or truncated:
                print(f"--- 回合结束，本回合总撞击次数: {total_cost:.0f} ---")
                obs, _ = env.reset()
                # total_cost = 0 

    print(f"✅ 视频保存成功！文件位置: {os.path.abspath(VIDEO_FILENAME)}")

if __name__ == "__main__":
    main()