import numpy as np
import os

# ================= 配置 =================
# 指向你采集到的那个大文件
SOURCE_DATA = './runs/PPOLag-{SafetyPointGoal1-v0}/seed-000-2026-02-07-19-09-38/safety_gym_raw_26dim.npz'
# SOURCE_DATA = './runs/PPOLag-{SafetyPointGoal1-v0}/seed-000-2026-02-07-19-09-38/safety_gym_26dim_data_v2.npz'
# runs/PPOLag-{SafetyPointGoal1-v0}/seed-000-2026-02-07-19-09-38
OUTPUT_DIR = './datasets'
# =======================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def split_into_episodes(data):
    """将扁平数据切分为轨迹列表"""
    episodes = []
    # 假设 timeouts 或 terminals 标记了结束
    ends = np.logical_or(data['terminals'], data['timeouts'])
    end_indices = np.where(ends)[0] + 1
    start_indices = np.concatenate([[0], end_indices[:-1]])

    for start, end in zip(start_indices, end_indices):
        # 排除过短的轨迹
        if end - start < 10: continue
        
        episodes.append({
            'obs': data['observations'][start:end],
            'act': data['actions'][start:end],
            'cost': data['costs'][start:end],
        })
    return episodes

def save_dataset(episodes, filename):
    """保存为 Diffuser 友好的格式"""
    # 拼接回大数组
    data_dict = {
        'observations': np.concatenate([e['obs'] for e in episodes], 0),
        'actions': np.concatenate([e['act'] for e in episodes], 0),
        'costs': np.concatenate([e['cost'] for e in episodes], 0),
        # 生成对应的索引区间 [start, end]
    }
    
    # 还需要保存每条轨迹的长度信息，方便 Loader 读取
    path_lengths = [len(e['obs']) for e in episodes]
    data_dict['path_lengths'] = np.array(path_lengths)
    
    save_path = os.path.join(OUTPUT_DIR, filename)
    np.savez(save_path, **data_dict)
    print(f"✅ Saved {filename}: {len(episodes)} trajectories, {sum(path_lengths)} steps")

def main():
    print(f"正在读取: {SOURCE_DATA}")
    raw = np.load(SOURCE_DATA)
    all_episodes = split_into_episodes(raw)
    
    # 1. Dataset Raw (所有数据)
    save_dataset(all_episodes, 'dataset_raw.npz')
    
    # 2. Dataset Truncated (截断碰撞后)
    trunc_episodes = []
    for eps in all_episodes:
        # 找第一个 cost > 0 的位置
        crash_idx = np.where(eps['cost'] > 0)[0]
        if len(crash_idx) > 0:
            idx = crash_idx[0]
            if idx > 10: # 至少保留一点
                trunc_episodes.append({
                    'obs': eps['obs'][:idx],
                    'act': eps['act'][:idx],
                    'cost': eps['cost'][:idx]
                })
        else:
            trunc_episodes.append(eps)
    save_dataset(trunc_episodes, 'dataset_truncated.npz')
    
    # 3. Dataset Safe Only (只保留完全无碰撞的)
    safe_episodes = [e for e in all_episodes if np.sum(e['cost']) == 0]
    save_dataset(safe_episodes, 'dataset_safe_only.npz')

if __name__ == '__main__':
    main()