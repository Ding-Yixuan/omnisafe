import os
import torch
import numpy as np
import omnisafe
import imageio
from datetime import datetime

# ================= 配置区域 =================
# 指向你新训练的原生 PPO 文件夹 (请修改这里！)
LOG_DIR = './runs/PPOLag-{SafetyPointGoal1-v0}/seed-000-2026-02-09-14-37-17' 

# 想要保存视频的名称
VIDEO_NAME = 'check_ppo_performance.mp4'
NUM_EPISODES = 3
# ===========================================

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

def main():
    # 1. 设置无头渲染 (Headless Rendering)
    os.environ['MUJOCO_GL'] = 'egl' 

    print(f"🔍 正在加载模型: {LOG_DIR}")
    evaluator = omnisafe.Evaluator()
    
    try:
        # 优先加载最终模型，如果没有就加载 epoch-xxx
        evaluator.load_saved(save_dir=LOG_DIR, model_name='model.pt')
    except:
        print("⚠️ 没找到 model.pt，尝试寻找最新的 epoch 模型...")
        # 自动找最大的 epoch
        files = os.listdir(os.path.join(LOG_DIR, 'torch_save'))
        epochs = [f for f in files if 'epoch' in f]
        if not epochs:
            raise FileNotFoundError("❌ 没找到任何模型文件！")
        latest = sorted(epochs, key=lambda x: int(x.split('-')[1].split('.')[0]))[-1]
        print(f"✅ 加载: {latest}")
        evaluator.load_saved(save_dir=LOG_DIR, model_name=latest)

    # 2. 获取策略和环境
    agent = find_actor(evaluator)
    # 强制开启 render_mode='rgb_array' 以便录像
    env = evaluator._env
    # 有些 wrapper 比较深，需要重新 make 一个用于录像的环境
    # 为了简单，我们直接用 evaluator 的 env，如果不支 render 我们再想办法
    
    print(f"✅ 模型加载成功！环境维度: {env.observation_space.shape}")
    print("🎥 开始录制视频...")

    frames = []
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    agent.to(device)

    total_success = 0

    for ep in range(NUM_EPISODES):
        obs, _ = env.reset()
        done = False
        step = 0
        ep_ret = 0
        
        while not done and step < 1000:
            # 渲染一帧
            try:
                # OmniSafe 的 env 可能包裹了很多层，尝试调用 render
                if hasattr(env, 'render'):
                    frame = env.render() # 假如它返回 rgb_array
                    if frame is not None: 
                        frames.append(frame)
            except Exception as e:
                pass # 如果渲染失败先不管，主要看 log

            # 决策
            with torch.no_grad():
                if isinstance(obs, np.ndarray):
                    t_obs = torch.as_tensor(obs, dtype=torch.float32).to(device).unsqueeze(0)
                else:
                    t_obs = obs.to(device).unsqueeze(0)
                act = agent.predict(t_obs, deterministic=True)
                act_cpu = act.squeeze(0).cpu()

            # 步进
            res = env.step(act_cpu)
            if len(res) == 6: next_obs, reward, cost, terminated, truncated, _ = res
            else: next_obs, reward, cost, terminated, truncated = res
            
            ep_ret += reward
            step += 1
            obs = next_obs
            
            if terminated or truncated:
                done = True
                if terminated: # 真正到达目标
                    print(f"🎉 Episode {ep+1}: 成功到达目标！Steps={step}, Reward={ep_ret:.2f}")
                    total_success += 1
                else: # 超时
                    print(f"⏳ Episode {ep+1}: 超时 (Timeout). Steps={step}, Reward={ep_ret:.2f}")

    # 保存视频
    if len(frames) > 0:
        imageio.mimsave(VIDEO_NAME, frames, fps=30)
        print(f"\n✅ 视频已保存至: {VIDEO_NAME}")
        print("📥 请下载视频查看，确认机器人是否走直线。")
    else:
        print("\n⚠️ 无法渲染视频 (可能是环境 Render 设置问题)，但请看上面的文字 Log。")
        print("如果显示 '成功到达目标'，那就没问题！")

if __name__ == '__main__':
    main()