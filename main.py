import os
import random
import numpy as np
import torch
from datetime import datetime
import argparse
import sys

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 打补丁到能量管理器
from energy_manager_patch import patch_energy_manager
EnergyManager, EdgeNode, CloudStorage = patch_energy_manager()

from training import train_rl_energy_manager, evaluate_and_compare
from visualization import visualize_results

def main():
    """主程序入口"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="强化学习云边协同能量管理系统")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'optimize'],
                      help='运行模式: train (训练), evaluate (评估), optimize (优化)')
    parser.add_argument('--nodes', type=int, default=8, help='边缘节点数量')
    parser.add_argument('--hours', type=int, default=168, help='仿真时长 (小时)')
    parser.add_argument('--cloud_capacity', type=float, default=200, help='云储能容量 (kWh)')
    parser.add_argument('--episodes', type=int, default=300, help='训练周期数')
    parser.add_argument('--model_path', type=str, default=None, help='模型路径 (用于评估和优化)')
    parser.add_argument('--base_dir', type=str, default=None, help='基本输出目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 设置边缘节点电池容量
    edge_battery_capacities = [30, 25, 40, 20, 35, 50, 60, 15]
    if args.nodes != 8:
        # 调整容量列表以匹配节点数量
        if args.nodes < 8:
            edge_battery_capacities = edge_battery_capacities[:args.nodes]
        else:
            # 扩展列表以覆盖额外节点
            avg_capacity = sum(edge_battery_capacities) / len(edge_battery_capacities)
            extra_capacities = [avg_capacity + random.uniform(-5, 5) for _ in range(args.nodes - 8)]
            edge_battery_capacities.extend(extra_capacities)
    
    # 设置基本目录
    if args.base_dir is None:
        args.base_dir = f"rl_energy_manager_{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    # 创建必要的目录
    os.makedirs(args.base_dir, exist_ok=True)
    
    # 运行选择的模式
    if args.mode == 'train':
        run_training_mode(args, edge_battery_capacities)
    elif args.mode == 'evaluate':
        run_evaluation_mode(args, edge_battery_capacities)
    elif args.mode == 'optimize':
        run_optimization_mode(args, edge_battery_capacities)
    else:
        print(f"未知模式: {args.mode}")

def run_training_mode(args, edge_battery_capacities):
    """运行训练模式"""
    print("====== 强化学习云边协同能源管理系统 - 训练模式 ======")
    print(f"节点数量: {args.nodes}")
    print(f"仿真时长: {args.hours}小时")
    print(f"边缘电池容量: {edge_battery_capacities}")
    print(f"云储能容量: {args.cloud_capacity} kWh")
    print(f"训练周期: {args.episodes}")
    print("=======================================")
    
    # 设置目录
    model_dir = os.path.join(args.base_dir, "models")
    results_dir = os.path.join(args.base_dir, "results")
    comparison_dir = os.path.join(args.base_dir, "comparison")
    
    # 运行训练
    print("\n开始训练...")
    agent, metrics = train_rl_energy_manager(
        num_nodes=args.nodes,
        simulation_hours=args.hours,
        edge_battery_capacities=edge_battery_capacities,
        cloud_storage_capacity=args.cloud_capacity,
        episodes=args.episodes,
        model_dir=model_dir,
        results_dir=results_dir
    )
    
    # 评估和比较
    print("\n开始评估和比较...")
    comparison_results = evaluate_and_compare(
        num_nodes=args.nodes,
        simulation_hours=args.hours,
        edge_battery_capacities=edge_battery_capacities,
        cloud_storage_capacity=args.cloud_capacity,
        agent=agent,
        save_dir=comparison_dir
    )
    
    # 保存最终结果概述
    with open(os.path.join(args.base_dir, "summary.txt"), 'w') as f:
        f.write("强化学习云边协同能源管理系统 - 结果概述\n")
        f.write("=====================================\n\n")
        
        f.write(f"系统配置:\n")
        f.write(f"  节点数量: {args.nodes}\n")
        f.write(f"  仿真时长: {args.hours}小时\n")
        f.write(f"  边缘电池容量: {edge_battery_capacities}\n")
        f.write(f"  云储能容量: {args.cloud_capacity} kWh\n\n")
        
        f.write(f"训练信息:\n")
        f.write(f"  训练周期: {args.episodes}\n")
        
        if 'performance_diff' in comparison_results:
            diff = comparison_results['performance_diff']
            f.write(f"\n性能改进:\n")
            f.write(f"  自给率: {diff['self_sufficiency']:.2f}%\n")
            f.write(f"  能量共享: {diff['energy_sharing']:.2f} kWh\n")
            f.write(f"  电网购电: {diff['grid_import']:.2f} kWh\n")
            f.write(f"  电网售电: {diff['grid_export']:.2f} kWh\n")
    
    print(f"\n所有结果已保存至: {args.base_dir}")
    print("强化学习云边协同能源管理系统运行完成!")

def run_evaluation_mode(args, edge_battery_capacities):
    """运行评估模式"""
    print("====== 强化学习云边协同能源管理系统 - 评估模式 ======")
    print(f"节点数量: {args.nodes}")
    print(f"仿真时长: {args.hours}小时")
    print(f"边缘电池容量: {edge_battery_capacities}")
    print(f"云储能容量: {args.cloud_capacity} kWh")
    print(f"模型路径: {args.model_path}")
    print("=======================================")
    
    if args.model_path is None:
        print("错误: 评估模式需要指定模型路径 (--model_path)")
        return
        
    # 设置评估目录
    eval_dir = os.path.join(args.base_dir, "evaluation")
    
    # 运行评估
    print("\n开始评估...")
    comparison_results = evaluate_and_compare(
        num_nodes=args.nodes,
        simulation_hours=args.hours,
        edge_battery_capacities=edge_battery_capacities,
        cloud_storage_capacity=args.cloud_capacity,
        model_path=args.model_path,
        save_dir=eval_dir
    )
    
    # 输出评估结果概述
    if 'performance_diff' in comparison_results:
        diff = comparison_results['performance_diff']
        print(f"\n性能对比概述:")
        print(f"  自给率改进: {diff['self_sufficiency']:.2f}%")
        print(f"  能量共享增加: {diff['energy_sharing']:.2f} kWh")
        print(f"  电网购电变化: {diff['grid_import']:.2f} kWh")
    
    print(f"\n评估结果已保存至: {eval_dir}")

def run_optimization_mode(args, edge_battery_capacities):
    """运行优化模式 - 创建和使用RL优化的能量管理器"""
    print("====== 强化学习云边协同能源管理系统 - 优化模式 ======")
    print(f"节点数量: {args.nodes}")
    print(f"仿真时长: {args.hours}小时")
    print(f"边缘电池容量: {edge_battery_capacities}")
    print(f"云储能容量: {args.cloud_capacity} kWh")
    print(f"模型路径: {args.model_path}")
    print("=======================================")
    
    if args.model_path is None:
        print("错误: 优化模式需要指定训练好的模型路径 (--model_path)")
        return
    
    # 设置优化输出目录
    opt_dir = os.path.join(args.base_dir, "optimization")
    os.makedirs(opt_dir, exist_ok=True)
    
    # 创建优化的能量管理器
    print("\n创建优化的能量管理器...")
    
    # 导入必要的模块
    from td3_agent import TD3Agent
    from rl_environment import RLEnergyManager
    
    # 创建RL环境
    env = RLEnergyManager(
        num_nodes=args.nodes,
        simulation_hours=args.hours,
        edge_battery_capacities=edge_battery_capacities,
        cloud_storage_capacity=args.cloud_capacity,
        start_time=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    )
    
    # 加载RL代理
    agent = TD3Agent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        max_action=1.0,
        hidden_dim=128
    )
    agent.load(args.model_path)
    
    # 运行一个完整的仿真
    print("\n运行RL优化的仿真...")
    rl_reward, metrics, trajectory = env.run_episode(agent, exploration=False)
    
    # 报告性能指标
    print("\n优化模式运行结果:")
    print(f"  自给率: {metrics['self_sufficiency_percentage']:.2f}%")
    print(f"  能量共享: {metrics['energy_sharing_total']:.2f} kWh")
    print(f"  电网购电: {metrics['total_grid_import']:.2f} kWh")
    print(f"  电网售电: {metrics['total_grid_export']:.2f} kWh")
    
    # 可视化结果
    from visualization import plot_evaluation_results
    plot_evaluation_results(env, trajectory, 'RL Optimized', opt_dir)
    
    # 保存结果概述
    with open(os.path.join(opt_dir, "optimization_summary.txt"), 'w') as f:
        f.write("RL优化结果概述\n")
        f.write("================\n\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\n优化结果已保存至: {opt_dir}")

if __name__ == "__main__":
    main()
