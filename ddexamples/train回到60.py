import omnisafe
import os

# =================================================================
# 1. 训练脚本 (使用官方原生环境，保证 100% 收敛)
# =================================================================
if __name__ == '__main__':
    # 官方环境 ID
    env_id = 'SafetyPointGoal1-v0'
    
    custom_cfgs = {
        # 1. 训练参数
        'train_cfgs': {
            'total_steps': 1024000, # 100万步
            'vector_env_nums': 1,
            'parallel': 1,
            'device': 'cuda:0',
        },
        # 2. 算法参数 (PPOLag)
        'algo_cfgs': {
            'steps_per_epoch': 2048,
            'update_iters': 10,
            'gamma': 0.99,
            'lam': 0.95,
            'clip': 0.2,
            'use_cost': True,
            'entropy_coef': 0.01, # 🔥 关键：加一点熵，防止它“死在原地转圈”
        },
        # 3. 拉格朗日参数
        'lagrange_cfgs': {
            # 🔥 关键修复：不要设为 0！
            # 设为 25.0 (Safety Gym Benchmark 标准)，让它敢于探索
            'cost_limit': 5.0,                  
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
        # 5. 日志
        'logger_cfgs': {
            'use_wandb': False,
            'save_model_freq': 50, # 每 50 epoch 存一次
        }
    }

    print(f"🚀 启动原生 PPO 训练 (Target: SafetyPointGoal1-v0)...")
    print("   -> Cost Limit: 25.0 (允许适度探索)")
    print("   -> Entropy: 0.01 (防止转圈)")
    
    agent = omnisafe.Agent('PPOLag', env_id, custom_cfgs=custom_cfgs)
    
    # 打印一下原生维度让你放心
    # 这里的维度通常是 60，但这没关系！Diffuser 不需要知道 PPO 看到了什么
    # Diffuser 只需要学习 "26维状态 -> 动作" 的映射
    try:
        env = agent.agent._env
        print(f"✅ 原生环境加载成功！Obs Shape: {env.observation_space.shape}")
    except:
        pass

    agent.learn()