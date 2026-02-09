# import torch
# import numpy as np
# import os
# import sys
# # MUJOCO_GL=egl /home/lqz27/anaconda3/envs/omnisafedd/bin/python /home/lqz27/dyx_ws/omnisafe/scripts/eval_diffuser.py

# # =================================================================
# # 1. 路径设置 (确保 Python 能找到 diffuser 包)
# # =================================================================
# # 获取当前脚本所在目录 (.../omnisafe/scripts)
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # 获取项目根目录 (.../omnisafe)
# project_root = os.path.dirname(current_dir)

# # 将项目根目录加入 sys.path
# if project_root not in sys.path:
#     sys.path.append(project_root)

# print(f"✅ 已添加项目根路径: {project_root}")

# # 引入你的 adapter (假设它在 scripts 目录下)
# sys.path.append(current_dir) 

# try:
#     from dataset_adapter import SafetyGymDataset
# except ImportError:
#     print("⚠️ 在当前目录下找不到 dataset_adapter，尝试从上一级查找...")
#     from scripts.dataset_adapter import SafetyGymDataset

# # =================================================================
# # 2. 【核心修复】正确的导入方式
# # =================================================================
# # ❌ 不要用: from diffusion import GaussianDiffusion (这是报错的原因)
# # ✅ 必须用: from diffuser.models.diffusion import ...

# try:
#     from diffuser.models.diffusion import GaussianDiffusion 
#     from diffuser.utils.training import Trainer 
# except ImportError as e:
#     print("\n❌ 导入失败！请检查文件结构是否如下：")
#     print(f"   {project_root}/diffuser/models/diffusion.py (原 diffusion初始版本.py)")
#     print(f"   {project_root}/diffuser/utils/training.py")
#     print(f"报错信息: {e}")
#     sys.exit(1)

# # 一个简单的 Dummy Renderer
# class DummyRenderer:
#     def composite(self, savepath, observations):
#         pass 

# def main():
#     # ================= 配置 =================
#     dataset_name = 'dataset_raw.npz' # 根据需要修改: 'dataset_truncated.npz' 或 'dataset_safe_only.npz'
#     dataset_path = os.path.join(project_root, 'datasets', dataset_name)
#     save_dir = os.path.join(project_root, 'diffuser_checkpoints', dataset_name.replace(".npz", ""))
#     os.makedirs(save_dir, exist_ok=True)
#     device = 'cuda:0'
#     horizon = 64
#     n_diffusion_steps = 100 # 论文通常用 20 或 100，训练可以先设小一点跑通流程
#     batch_size = 256
#     n_train_steps = 100000
    
#     # ================= 1. 加载数据 =================
#     if not os.path.exists(dataset_path):
#         raise FileNotFoundError(f"找不到数据集: {dataset_path}\n请先运行 1_preprocess_data.py 生成数据！")

#     dataset = SafetyGymDataset(dataset_path, horizon=horizon)
#     renderer = DummyRenderer()
    
#     print(f"Observation Dim: 26")
#     print(f"Action Dim: 2")
    
#     # ================= 2. 构建模型 (Temporal U-Net) =================
#     from diffuser.models.temporal import TemporalUnet 
    
#     model = TemporalUnet(
#         horizon=horizon,
#         transition_dim=26 + 2, # obs + act
#         cond_dim=26,
#         dim=256,
#         dim_mults=(1, 2, 4)
#     ).to(device)
    
#     # ================= 3. 构建 Diffuser =================
#     diffusion = GaussianDiffusion(
#         model=model,
#         horizon=horizon,
#         observation_dim=26,
#         action_dim=2,
#         n_timesteps=n_diffusion_steps,
#         loss_type='l2',
#         clip_denoised=True,
#         predict_epsilon=False, 
#         action_weight=10.0,   
#     ).to(device)
    
#     # 注入 normalizer (Trainer 需要用到)
#     diffusion.normalizer = dataset 
    
#     # ================= 4. 开始训练 =================
#     trainer = Trainer(
#         diffusion_model=diffusion,
#         dataset=dataset,
#         renderer=renderer,
#         train_batch_size=batch_size,
#         train_lr=2e-5,
#         results_folder=save_dir,
#         save_freq=1000,
#         label_freq=1000,
#         log_freq=100,
#     )
    
#     print(f"=== 开始训练: {dataset_name} ===")
#     trainer.train(n_train_steps)

# if __name__ == '__main__':
#     main()

import torch
import numpy as np
import os
import sys
import argparse  # 🔥【新增】引入参数解析库

# =================================================================
# 1. 路径设置
# =================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

print(f"✅ 已添加项目根路径: {project_root}")
sys.path.append(current_dir)

try:
    from dataset_adapter import SafetyGymDataset
except ImportError:
    print("⚠️ 在当前目录下找不到 dataset_adapter，尝试从上一级查找...")
    from scripts.dataset_adapter import SafetyGymDataset

# =================================================================
# 2. 导入 Diffuser 模块
# =================================================================
try:
    from diffuser.models.diffusion import GaussianDiffusion
    from diffuser.utils.training import Trainer
    from diffuser.models.temporal import TemporalUnet
except ImportError as e:
    print("\n❌ 导入失败！请检查文件结构。")
    print(f"报错信息: {e}")
    sys.exit(1)

class DummyRenderer:
    def composite(self, savepath, observations):
        pass

def main():
    # ================= 🔥【修改】参数解析 =================
    parser = argparse.ArgumentParser()
    
    # 1. 数据集名称 (默认指向 v2_raw)
    parser.add_argument('--dataset', type=str, default='dataset_v2_raw.npz', 
                        help='dataset file name in ./datasets/')
    
    # 2. 模型保存路径 (允许自定义，不再写死)
    parser.add_argument('--save_path', type=str, default='./diffuser_checkpoints/default_run',
                        help='path to save checkpoints')
    
    # 3. 目标权重 (默认 1.0，想加强 Goal 信号就设大，比如 5.0 或 10.0)
    parser.add_argument('--goal_weight', type=float, default=1.0, 
                        help='multiply goal signal by this weight')
    
    # 其他训练参数
    parser.add_argument('--n_train_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=256)
    
    args = parser.parse_args()

    # ================= 配置路径 =================
    dataset_path = os.path.join(project_root, 'datasets', args.dataset)
    save_dir = args.save_path  # 🔥 直接使用传入的路径
    
    os.makedirs(save_dir, exist_ok=True)
    
    device = 'cuda:0'
    horizon = 64
    n_diffusion_steps = 100 

    # ================= 1. 加载数据 =================
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"找不到数据集: {dataset_path}\n请确保数据在 datasets 文件夹下！")

    print(f"Loading Dataset: {args.dataset}")
    print(f"Goal Weight: {args.goal_weight}x")
    
    # 🔥【修改】传入 goal_weight 参数
    # 注意：这需要你的 dataset_adapter.py __init__ 已经支持 goal_weight 参数
    dataset = SafetyGymDataset(
        dataset_path, 
        horizon=horizon, 
        goal_weight=args.goal_weight 
    )
    
    renderer = DummyRenderer()
    
    print(f"Observation Dim: 26")
    print(f"Action Dim: 2")
    
    # ================= 2. 构建模型 =================
    model = TemporalUnet(
        horizon=horizon,
        transition_dim=26 + 2,
        cond_dim=26,
        dim=256,
        dim_mults=(1, 2, 4)
    ).to(device)
    
    # ================= 3. 构建 Diffuser =================
    diffusion = GaussianDiffusion(
        model=model,
        horizon=horizon,
        observation_dim=26,
        action_dim=2,
        n_timesteps=n_diffusion_steps,
        loss_type='l2',
        clip_denoised=True,
        predict_epsilon=False, # 坚持使用 False (Predict X_Start)
        action_weight=10.0,   
    ).to(device)
    
    diffusion.normalizer = dataset
    
    # ================= 4. 开始训练 =================
    trainer = Trainer(
        diffusion_model=diffusion,
        dataset=dataset,
        renderer=renderer,
        train_batch_size=args.batch_size,
        train_lr=2e-5,
        
        # 🔥【修改】结果保存路径
        results_folder=save_dir,
        
        save_freq=1000,
        label_freq=1000,
        log_freq=100,
    )
    
    print(f"=== 开始训练 ===")
    print(f"   Dataset: {args.dataset}")
    print(f"   Save Path: {save_dir}")
    print(f"   Goal Weight: {args.goal_weight}")
    
    trainer.train(args.n_train_steps)

if __name__ == '__main__':
    main()