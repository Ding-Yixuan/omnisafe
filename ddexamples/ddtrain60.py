# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example of training a policy from custom dict with OmniSafe."""

import omnisafe
import safety_gymnasium
from safety_gymnasium.assets.geoms import Hazards
from safety_gymnasium.tasks.safe_navigation.goal.goal_level1 import GoalLevel1
from safety_gymnasium.tasks.safe_navigation.goal.goal_level0 import GoalLevel0

# =================================================================
# 1. 定义 Patch 函数 (仅修改环境配置，不改 obs)
# =================================================================

def patched_init(self, config):
    """替换 GoalLevel1.__init__，仅修改障碍物数量"""
    # 这里我们只修改环境生成的配置，不强制修改 sensors_obs 或 lidar 参数
    # 这样 OmniSafe/SafetyGymnasium 会使用默认的观测空间 (通常是 60 维)
    
    # 调用父类构造 (GoalLevel0)
    GoalLevel0.__init__(self, config=config)
    
    # 修改环境元素：限制范围并设置 2 个障碍物
    self.placements_conf.extents = [-1.5, -1.5, 1.5, 1.5]
    self._add_geoms(Hazards(num=2, keepout=0.2))

# =================================================================
# 2. 执行 Monkey Patch
# =================================================================
# 只替换 __init__，保持 obs 和 build_observation_space 为原版
GoalLevel1.__init__ = patched_init

print("成功 Monkey Patch: __init__ (障碍物改为2个，观测维度保持默认)")


if __name__ == '__main__':
    env_id = 'SafetyPointGoal1-v0'
    custom_cfgs = {
        'train_cfgs': {
            'total_steps': 1024000,
            'vector_env_nums': 1,
            'parallel': 1,
        },
        'algo_cfgs': {
            'steps_per_epoch': 2048,
            'update_iters': 1,
        },
        'logger_cfgs': {
            'use_wandb': False,
        },
    }

    agent = omnisafe.Agent('PPOLag', env_id, custom_cfgs=custom_cfgs)
    agent.learn()

    agent.plot(smooth=1)
    # render 需要 monitor 支持，如果在 headless 服务器上可能需要去掉或改为 rgb_array 保存
    # agent.render(num_episodes=1, render_mode='rgb_array', width=256, height=256)
    agent.evaluate(num_episodes=1)