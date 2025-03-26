import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def plot_training_results(
    rewards, 
    critic_losses, 
    actor_losses, 
    self_sufficiency, 
    energy_sharing,
    save_dir
):
    """
    绘制训练结果
    
    Args:
        rewards: 奖励历史
        critic_losses: Critic损失历史
        actor_losses: Actor损失历史
        self_sufficiency: 自给率历史
        energy_sharing: 能量共享历史
        save_dir: 保存目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 绘制奖励曲线
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'rewards_{timestamp}.png'))
    plt.close()
    
    # 2. 绘制损失曲线
    if critic_losses and actor_losses:
        plt.figure(figsize=(10, 6))
        plt.plot(critic_losses, label='Critic Loss')
        
        # 处理actor_losses中的None值
        valid_actor_losses = [loss for loss in actor_losses if loss is not None]
        valid_indices = [i for i, loss in enumerate(actor_losses) if loss is not None]
        
        if valid_actor_losses:
            plt.plot(valid_indices, valid_actor_losses, label='Actor Loss')
            
        plt.title('Training Losses')
        plt.xlabel('Update Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'losses_{timestamp}.png'))
        plt.close()
    
    # 3. 绘制自给率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(self_sufficiency)
    plt.title('Self-Sufficiency Percentage')
    plt.xlabel('Episode')
    plt.ylabel('Self-Sufficiency (%)')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'self_sufficiency_{timestamp}.png'))
    plt.close()
    
    # 4. 绘制能量共享曲线
    plt.figure(figsize=(10, 6))
    plt.plot(energy_sharing)
    plt.title('Energy Sharing')
    plt.xlabel('Episode')
    plt.ylabel('Energy Shared (kWh)')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'energy_sharing_{timestamp}.png'))
    plt.close()
    
    # 5. 绘制平滑后的奖励曲线（使用移动平均）
    window_size = min(20, len(rewards))
    if window_size > 0:
        smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        
        plt.figure(figsize=(10, 6))
        plt.plot(smoothed_rewards)
        plt.title('Smoothed Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'smoothed_rewards_{timestamp}.png'))
        plt.close()
    
    # 保存原始数据
    np.save(os.path.join(save_dir, f'rewards_{timestamp}.npy'), rewards)
    np.save(os.path.join(save_dir, f'self_sufficiency_{timestamp}.npy'), self_sufficiency)
    np.save(os.path.join(save_dir, f'energy_sharing_{timestamp}.npy'), energy_sharing)
    
    if critic_losses:
        np.save(os.path.join(save_dir, f'critic_losses_{timestamp}.npy'), critic_losses)
    if actor_losses:
        # 保存时过滤掉None值
        np.save(os.path.join(save_dir, f'actor_losses_{timestamp}.npy'), 
                np.array([loss if loss is not None else np.nan for loss in actor_losses]))

def plot_evaluation_results(env, trajectory, title, save_path):
    """
    绘制评估结果
    
    Args:
        env: RL环境
        trajectory: 轨迹数据
        title: 图表标题
        save_path: 保存路径
    """
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 提取数据
    states = trajectory['states']
    actions = trajectory['actions']
    rewards = trajectory['rewards']
    
    # 时间轴
    time_steps = list(range(len(rewards)))
    
    # 1. 累计奖励曲线
    plt.figure(figsize=(10, 6))
    cumulative_rewards = np.cumsum(rewards)
    plt.plot(time_steps, cumulative_rewards)
    plt.title(f'{title} - Cumulative Reward')
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Reward')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'cumulative_reward.png'))
    plt.close()
    
    # 2. 动作随时间的变化
    plt.figure(figsize=(12, 8))
    actions_np = np.array(actions)
    for i in range(actions_np.shape[1]):
        if i < env.num_nodes:
            label = f'Node {i+1}'
        else:
            label = 'Cloud Storage'
        plt.plot(time_steps, actions_np[:, i], label=label)
    
    plt.title(f'{title} - Actions Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Action Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'actions.png'))
    plt.close()
    
    # 3. 边缘节点的SOC变化
    try:
        plt.figure(figsize=(12, 8))
        states_np = np.array(states)
        
        # 提取节点SOC (从状态向量的相应位置)
        for i in range(env.num_nodes):
            node_soc = states_np[:, i+1]  # 状态中的SOC位置
            plt.plot(time_steps, node_soc, label=f'Node {i+1}')
        
        plt.title(f'{title} - Edge Node SOC')
        plt.xlabel('Time Step')
        plt.ylabel('State of Charge')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_path, 'node_soc.png'))
        plt.close()
        
        # 4. 云储能SOC变化
        plt.figure(figsize=(10, 6))
        cloud_soc = states_np[:, env.num_nodes+1]  # 状态中的云储能SOC位置
        plt.plot(time_steps, cloud_soc)
        plt.title(f'{title} - Cloud Storage SOC')
        plt.xlabel('Time Step')
        plt.ylabel('State of Charge')
        plt.grid(True)
        plt.savefig(os.path.join(save_path, 'cloud_soc.png'))
        plt.close()
    except Exception as e:
        print(f"绘制SOC图表时出错: {str(e)}")
    
    # 5. 每步奖励
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, rewards)
    plt.title(f'{title} - Rewards per Step')
    plt.xlabel('Time Step')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'step_rewards.png'))
    plt.close()
    
    # 保存轨迹数据
    try:
        np.save(os.path.join(save_path, 'states.npy'), np.array(states))
        np.save(os.path.join(save_path, 'actions.npy'), np.array(actions))
        np.save(os.path.join(save_path, 'rewards.npy'), np.array(rewards))
    except Exception as e:
        print(f"保存轨迹数据时出错: {str(e)}")

def plot_comparison(results, save_path):
    """
    绘制RL与基线策略的比较图
    
    Args:
        results: 评估结果字典
        save_path: 保存路径
    """
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 提取数据
    rl_metrics = results['rl_policy']['metrics']
    baseline_metrics = results['baseline_policy']['metrics']
    
    # 比较指标
    metrics_to_compare = [
        'self_sufficiency_percentage',
        'energy_sharing_total',
        'total_grid_import',
        'total_grid_export'
    ]
    
    metric_labels = {
        'self_sufficiency_percentage': 'Self-Sufficiency (%)',
        'energy_sharing_total': 'Energy Sharing (kWh)',
        'total_grid_import': 'Grid Import (kWh)',
        'total_grid_export': 'Grid Export (kWh)'
    }
    
    # 1. 并排条形图比较
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(metrics_to_compare))
    width = 0.35
    
    rl_values = [rl_metrics[metric] for metric in metrics_to_compare]
    baseline_values = [baseline_metrics[metric] for metric in metrics_to_compare]
    
    plt.bar(x - width/2, rl_values, width, label='RL Policy')
    plt.bar(x + width/2, baseline_values, width, label='Baseline')
    
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.title('RL vs Baseline Policy Comparison')
    plt.xticks(x, [metric_labels[m] for m in metrics_to_compare])
    plt.legend()
    
    # 添加数值标签
    for i, v in enumerate(rl_values):
        plt.text(i - width/2, v + 0.5, f'{v:.1f}', ha='center')
    
    for i, v in enumerate(baseline_values):
        plt.text(i + width/2, v + 0.5, f'{v:.1f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'metrics_comparison.png'))
    plt.close()
    
    # 2. 性能改进百分比图
    improvements = {}
    for metric in metrics_to_compare:
        if metric in ['self_sufficiency_percentage', 'energy_sharing_total']:
            # 这些指标是越高越好
            improvements[metric] = (rl_metrics[metric] - baseline_metrics[metric]) / max(baseline_metrics[metric], 1e-5) * 100
        else:
            # 这些指标是越低越好（如电网购电）
            improvements[metric] = (baseline_metrics[metric] - rl_metrics[metric]) / max(baseline_metrics[metric], 1e-5) * 100
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(improvements)), list(improvements.values()))
    
    # 设置颜色：正值为绿色（改进），负值为红色（下降）
    for i, bar in enumerate(bars):
        if bar.get_height() >= 0:
            bar.set_color('green')
        else:
            bar.set_color('red')
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Metrics')
    plt.ylabel('Improvement (%)')
    plt.title('RL Policy Performance Improvement over Baseline')
    plt.xticks(range(len(improvements)), [metric_labels[m] for m in improvements.keys()])
    
    # 添加数值标签
    for i, v in enumerate(improvements.values()):
        plt.text(i, v + (1 if v >= 0 else -5), f'{v:.1f}%', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'performance_improvement.png'))
    plt.close()
    
    # 3. 自给率和电网依赖性饼图比较
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # RL策略饼图
    rl_labels = ['Self-Sufficient', 'Grid Dependent']
    rl_sizes = [rl_metrics['self_sufficiency_percentage'], 
                rl_metrics['grid_dependency_percentage']]
    ax1.pie(rl_sizes, labels=rl_labels, autopct='%1.1f%%', startangle=90)
    ax1.set_title('RL Policy')
    
    # 基线策略饼图
    baseline_labels = ['Self-Sufficient', 'Grid Dependent']
    baseline_sizes = [baseline_metrics['self_sufficiency_percentage'], 
                      baseline_metrics['grid_dependency_percentage']]
    ax2.pie(baseline_sizes, labels=baseline_labels, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Baseline Policy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'self_sufficiency_comparison.png'))
    plt.close()
    
    # 保存比较数据
    comparison_data = {
        'rl_metrics': rl_metrics,
        'baseline_metrics': baseline_metrics,
        'improvements': improvements
    }
    
    # 生成文本报告
    with open(os.path.join(save_path, 'comparison_report.txt'), 'w') as f:
        f.write("RL vs Baseline Policy Comparison Report\n")
        f.write("=====================================\n\n")
        
        f.write("Metrics:\n")
        for metric in metrics_to_compare:
            f.write(f"{metric_labels[metric]}:\n")
            f.write(f"  RL Policy: {rl_metrics[metric]:.2f}\n")
            f.write(f"  Baseline: {baseline_metrics[metric]:.2f}\n")
            f.write(f"  Difference: {rl_metrics[metric] - baseline_metrics[metric]:.2f}\n")
            f.write(f"  Improvement: {improvements[metric]:.2f}%\n\n")

def visualize_results(results, save_dir):
    """
    将原始能量管理器的可视化方法包装为兼容RL比较
    
    Args:
        results: 能量管理系统的结果字典
        save_dir: 保存目录
    """
    # 创建目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 绘制基本能量交互图
    try:
        timestamps = list(range(len(results['grid_interaction_history']['draw'])))
        
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, results['grid_interaction_history']['draw'], 'r-', label='Grid Draw')
        plt.plot(timestamps, results['grid_interaction_history']['feed'], 'g-', label='Grid Feed')
        plt.title('Grid Interactions Over Time')
        plt.xlabel('Time (hours)')
        plt.ylabel('Energy (kWh)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'grid_interactions.png'))
        plt.close()
        
        # 绘制云储能能量水平
        plt.figure(figsize=(10, 6))
        energy_levels = results['cloud_storage']['energy_level_history'][:-1]  # 移除多余的点
        plt.plot(timestamps, energy_levels[:len(timestamps)], 'b-')  # 确保长度匹配
        plt.title('Cloud Storage Energy Level')
        plt.xlabel('Time (hours)')
        plt.ylabel('Energy (kWh)')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'cloud_storage_level.png'))
        plt.close()
        
        # 绘制节点SOC
        plt.figure(figsize=(10, 6))
        for node_id, node_data in results['edge_nodes'].items():
            soc_history = node_data['soc_history'][:-1]  # 移除多余的点
            plt.plot(timestamps, soc_history[:len(timestamps)], label=f'Node {node_id}')  # 确保长度匹配
        plt.title('Edge Node SOC')
        plt.xlabel('Time (hours)')
        plt.ylabel('State of Charge')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'node_soc.png'))
        plt.close()
    except Exception as e:
        print(f"绘制图表时出错: {str(e)}")
    
    # 绘制自给率饼图
    plt.figure(figsize=(8, 8))
    labels = ['Self-Sufficient', 'Grid Dependent']
    sizes = [results['metrics']['self_sufficiency_percentage'], 
             results['metrics']['grid_dependency_percentage']]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('Energy Self-Sufficiency')
    plt.savefig(os.path.join(save_dir, 'self_sufficiency.png'))
    plt.close()
    
    # 保存摘要信息
    with open(os.path.join(save_dir, 'summary.txt'), 'w') as f:
        f.write("Energy Management System Results\n")
        f.write("=============================\n\n")
        f.write(f"Total Load: {results['metrics']['total_load']:.2f} kWh\n")
        f.write(f"Total PV Generation: {results['metrics']['total_pv_generation']:.2f} kWh\n")
        f.write(f"Grid Dependency: {results['metrics']['grid_dependency_percentage']:.2f}%\n")
        f.write(f"Self-Sufficiency: {results['metrics']['self_sufficiency_percentage']:.2f}%\n")
        f.write(f"Energy Sharing: {results['metrics']['energy_sharing_total']:.2f} kWh\n")