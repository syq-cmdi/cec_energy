# 基于强化学习的云边能量管理系统

该项目实现了一个基于强化学习的云边协同能量管理系统，通过智能决策优化边缘节点与云端储能之间的能量分配，提高自给率，减少电网依赖。

## 项目结构

```
├── models.py              # 神经网络模型定义
├── replay_buffer.py       # 经验回放缓冲区
├── td3_agent.py           # TD3算法代理实现
├── rl_environment.py      # 强化学习环境
├── training.py            # 训练和评估函数
├── visualization.py       # 结果可视化工具
├── integration.py         # 集成到原有系统
├── main.py                # 主程序入口
└── requirements.txt       # 依赖库
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 训练模式

训练一个新的强化学习模型：

```bash
python main.py --mode train --nodes 8 --hours 168 --cloud_capacity 200 --episodes 1000
```

参数说明：
- `--nodes`: 边缘节点数量
- `--hours`: 仿真时长(小时)
- `--cloud_capacity`: 云储能容量(kWh)
- `--episodes`: 训练周期数

### 评估模式

评估一个已训练模型的性能，并与基线策略比较：

```bash
python main.py --mode evaluate --model_path models/best_model --nodes 8 --hours 168
```

参数说明：
- `--model_path`: 训练好的模型路径

### 优化模式

使用训练好的模型来优化能量管理：

```bash
python main.py --mode optimize --model_path models/best_model --nodes 8 --hours 168
```

## 系统特点

1. **智能决策** - 使用Twin Delayed DDPG (TD3)算法进行连续动作空间的优化
2. **多目标优化** - 同时优化自给率、能量共享和电网互动
3. **可视化监控** - 全面的可视化工具展示系统性能
4. **灵活配置** - 支持不同节点数量和储能容量的配置

## 性能指标

系统通过以下关键指标评估性能：

1. **自给率** - 系统不依赖电网的能量比例
2. **能量共享量** - 节点间共享的能量总量
3. **电网互动** - 从电网购电和向电网售电的总量
4. **超循环特性** - 电池充放电循环的深度和频率

## 集成到现有系统

可以通过`integration.py`模块将训练好的强化学习模型集成到现有能量管理系统：

```python
from integration import integrate_rl_agent_with_energy_manager

# 加载已有能量管理器
energy_manager = EnergyManager(...)

# 集成RL代理
enhanced_manager = integrate_rl_agent_with_energy_manager(
    energy_manager, 
    "path/to/trained/model"
)

# 运行优化的仿真
results = enhanced_manager.run_simulation()
```

## 算法说明

该系统使用TD3（Twin Delayed DDPG）算法，它是一种适用于连续动作空间的深度强化学习算法，具有以下特点：

1. **双Q网络** - 减少Q值过估计
2. **延迟策略更新** - 提高训练稳定性
3. **目标策略平滑** - 增加对动作噪声的鲁棒性
4. **经验回放** - 打破样本相关性，提高数据效率

## 与原有系统的比较

与基于启发式规则的基线能量管理策略相比，基于RL的云边协同方案通常能实现：

- **提高自给率** - 减少对电网的依赖
- **增加能量共享** - 最大化节点间的能量共享利用
- **优化电池使用** - 降低电池循环深度，延长寿命
- **适应性强** - 更好地应对负载和发电的变化和不确定性

## 后续工作

可以在以下方向进一步改进系统：

1. **多智能体强化学习** - 为每个边缘节点和云储能各自配备独立的智能体
2. **分布式训练** - 实现大规模边缘节点环境的分布式训练
3. **预测整合** - 结合负载和光伏发电预测模型
4. **用户偏好** - 考虑用户舒适度和偏好的多目标优化

## 许可证

本项目基于MIT许可证开源。

## 引用

如果您在研究中使用了本项目，请引用：

```
@misc{rl-cloud-edge-energy,
  author = {Your Name},
  title = {基于强化学习的云边能量管理系统},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/rl-cloud-edge-energy}
}
```