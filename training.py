import os
import random
import numpy as np
import torch
from datetime import datetime
import sys

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 应用补丁到能量管理器
from energy_manager_patch import patch_energy_manager
EnergyManager, EdgeNode, CloudStorage = patch_energy_manager()

from replay_buffer import ReplayBuffer
from td3_agent import TD3Agent
from rl_environment import RLEnergyManager
from visualization import plot_training_results, plot_evaluation_results, plot_comparison

def train_rl_energy_manager(
    num_nodes=8,
    simulation_hours=168,  # 7天
    edge_battery_capacities=None,
    cloud_storage_capacity=200,
    episodes=1000,
    batch_size=64,  
    replay_buffer_size=100000,
    save_interval=100,
    eval_interval=50,
    model_dir='models',
    results_dir='rl_results'
):
    """
    训练强化学习能量管理器
    
    Args:
        num_nodes: 边缘节点数量
        simulation_hours: 每次仿真的时长（小时）
        edge_battery_capacities: 各节点电池容量
        cloud_storage_capacity: 云储能容量
        episodes: 训练周期数
        batch_size: 批次大小
        replay_buffer_size: 经验回放缓冲区大小
        save_interval: 模型保存间隔
        eval_interval: 评估间隔
        model_dir: 模型保存目录
        results_dir: 结果保存目录
    """
    # 设置默认电池容量（如果未指定）
    if edge_battery_capacities is None:
        edge_battery_capacities = [30] * num_nodes
    
    # 创建环境
    env = RLEnergyManager(
        num_nodes=num_nodes,
        simulation_hours=simulation_hours,
        edge_battery_capacities=edge_battery_capacities,
        cloud_storage_capacity=cloud_storage_capacity,
        start_time=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    )
    
    # 创建RL代理
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    # 使用TD3算法，并调整超参数
    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=1.0,
        hidden_dim=256,  # 增大网络容量
        actor_lr=1e-4,   # 调整学习率
        critic_lr=1e-4,  # 调整学习率
        policy_noise=0.2, # 增加政策噪声
        noise_clip=0.5,   # 增加噪声裁剪范围
        tau=0.01         # 增加目标网络更新速率
    )
    
    # 创建经验回放缓冲区
    replay_buffer = ReplayBuffer(replay_buffer_size)
    
    # 创建结果目录
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # 记录训练指标
    rewards_history = []
    critic_losses = []
    actor_losses = []
    self_sufficiency_history = []
    energy_sharing_history = []
    
    # 最佳模型记录
    best_reward = -float('inf')
    best_self_sufficiency = 0
    
    print(f"开始训练RL能量管理器: {num_nodes}个节点，{simulation_hours}小时仿真")
    
    # 预热阶段：随机填充经验池
    print("预热阶段：随机填充经验池...")
    warmup_episodes = min(10, episodes // 10)  # 减少预热周期，更快开始学习
    for episode in range(warmup_episodes):
        state = env.reset()
        
        for _ in range(simulation_hours):
            # 使用均匀分布的随机动作，而不是纯高斯噪声
            action = np.random.uniform(-1, 1, size=env.action_dim)
            next_state, reward, done, _ = env.step(action)
            
            # 保存经验
            replay_buffer.add(state, action, reward, next_state, done)
            
            if done:
                break
                
            state = next_state
                
        print(f"预热周期 {episode+1}/{warmup_episodes} 完成")
    
    print(f"预热完成，经验池大小: {len(replay_buffer)}")
    
    # 主训练循环
    for episode in range(1, episodes + 1):
        # 使用指数衰减的探索噪声，保持更长探索期
        exploration_noise = max(0.4 * np.exp(-episode / (episodes / 5)), 0.05)
        
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # 选择动作
            action = agent.select_action(state, add_noise=True, noise_scale=exploration_noise)
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            # 保存经验
            replay_buffer.add(state, action, reward, next_state, done)
            
            # 更新状态
            state = next_state
            
            # 训练智能体 - 每步进行多次更新
            if len(replay_buffer) > batch_size * 5:
                for _ in range(4):  # 每步环境交互进行多次网络更新，加速学习
                    critic_loss, actor_loss = agent.update(replay_buffer, batch_size)
                    if critic_loss is not None:
                        critic_losses.append(critic_loss)
                    if actor_loss is not None:
                        actor_losses.append(actor_loss)
        
        # 记录训练指标
        metrics = env._calculate_final_metrics()
        rewards_history.append(episode_reward)
        self_sufficiency_history.append(metrics['self_sufficiency_percentage'])
        energy_sharing_history.append(metrics['energy_sharing_total'])
        
        # 打印训练进度
        if episode % 5 == 0 or episode == 1:  # 更频繁地显示进度
            avg_reward = np.mean(rewards_history[-10:]) if episode > 10 else episode_reward
            print(f"Episode {episode}/{episodes} - 奖励: {episode_reward:.2f}, 平均奖励: {avg_reward:.2f}")
            print(f"  自给率: {metrics['self_sufficiency_percentage']:.2f}%, 能量共享: {metrics['energy_sharing_total']:.2f} kWh")
            print(f"  电网购电: {metrics['total_grid_import']:.2f} kWh, 电网售电: {metrics['total_grid_export']:.2f} kWh")
            print(f"  探索噪声: {exploration_noise:.2f}")
        
        # 保存模型检查点
        if episode % save_interval == 0 or episode == episodes:
            checkpoint_dir = os.path.join(model_dir, f"checkpoint_{episode}")
            agent.save(checkpoint_dir)
            print(f"模型已保存至: {checkpoint_dir}")
        
        # 保存最佳模型
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(os.path.join(model_dir, "best_reward_model"))
            print(f"新的最佳奖励模型已保存: {best_reward:.2f}")
        
        if metrics['self_sufficiency_percentage'] > best_self_sufficiency:
            best_self_sufficiency = metrics['self_sufficiency_percentage']
            agent.save(os.path.join(model_dir, "best_self_sufficiency_model"))
            print(f"新的最佳自给率模型已保存: {best_self_sufficiency:.2f}%")
        
        # 评估模型（无探索）
        if episode % eval_interval == 0 or episode == episodes:
            eval_reward, eval_metrics, _ = env.run_episode(agent, exploration=False)
            print(f"\n评估结果 (Episode {episode}):")
            print(f"  奖励: {eval_reward:.2f}")
            print(f"  自给率: {eval_metrics['self_sufficiency_percentage']:.2f}%")
            print(f"  能量共享: {eval_metrics['energy_sharing_total']:.2f} kWh")
            print(f"  电网购电: {eval_metrics['total_grid_import']:.2f} kWh")
            print(f"  电网售电: {eval_metrics['total_grid_export']:.2f} kWh\n")
    
    # 训练结束后绘制结果
    plot_training_results(
        rewards_history, 
        critic_losses, 
        actor_losses, 
        self_sufficiency_history, 
        energy_sharing_history,
        results_dir
    )
    
    print("RL能量管理器训练完成！")
    print(f"最佳奖励: {best_reward:.2f}")
    print(f"最佳自给率: {best_self_sufficiency:.2f}%")
    
    # 返回训练好的代理和最后的评估结果
    return agent, eval_metrics

def evaluate_and_compare(
    num_nodes=8,
    simulation_hours=168,
    edge_battery_capacities=None,
    cloud_storage_capacity=200,
    agent=None,
    model_path=None,
    save_dir='comparison_results'
):
    """
    评估并比较RL策略与基线策略
    
    Args:
        num_nodes: 节点数量
        simulation_hours: 仿真时长
        edge_battery_capacities: 电池容量列表
        cloud_storage_capacity: 云储能容量
        agent: 预训练的RL代理（如果有）
        model_path: 模型路径（如果无预训练代理）
        save_dir: 结果保存目录
    
    Returns:
        比较结果字典
    """
    # 设置默认电池容量（如果未指定）
    if edge_battery_capacities is None:
        edge_battery_capacities = [30] * num_nodes
    
    # 创建结果目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建环境
    env = RLEnergyManager(
        num_nodes=num_nodes,
        simulation_hours=simulation_hours,
        edge_battery_capacities=edge_battery_capacities,
        cloud_storage_capacity=cloud_storage_capacity,
        start_time=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    )
    
    # 加载RL代理
    if agent is None and model_path is not None:
        agent = TD3Agent(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            max_action=1.0,
            hidden_dim=256  # 使用与训练相同的网络大小
        )
        agent.load(model_path)
    
    results = {}
    
    # 运行RL策略
    if agent is not None:
        print("评估RL策略...")
        rl_reward, rl_metrics, rl_trajectory = env.run_episode(agent, exploration=False)
        print(f"RL策略评估结果:")
        print(f"  奖励: {rl_reward:.2f}")
        print(f"  自给率: {rl_metrics['self_sufficiency_percentage']:.2f}%")
        print(f"  能量共享: {rl_metrics['energy_sharing_total']:.2f} kWh")
        print(f"  电网购电: {rl_metrics['total_grid_import']:.2f} kWh")
        
        # 可视化RL策略结果
        plot_evaluation_results(
            env, 
            rl_trajectory, 
            'RL Policy', 
            os.path.join(save_dir, f'rl_policy_{timestamp}')
        )
        
        results['rl_policy'] = {
            'reward': rl_reward,
            'metrics': rl_metrics,
            'trajectory': rl_trajectory
        }
    
    # 运行基线策略（原始能量管理逻辑）
    print("\n运行基线策略...")
    # 导入可视化函数
    from visualization import visualize_results
    
    original_energy_manager = EnergyManager(
        num_nodes=num_nodes,
        simulation_hours=simulation_hours,
        edge_battery_capacities=edge_battery_capacities,
        cloud_storage_capacity=cloud_storage_capacity,
        load_profiles=env.energy_manager.load_profiles,
        pv_profiles=env.energy_manager.pv_profiles,
        start_time=env.energy_manager.start_time
    )
    
    # 运行原始仿真
    baseline_results = original_energy_manager.run_simulation()
    
    # 计算基线指标
    baseline_metrics = {
        'total_load': baseline_results['metrics']['total_load'],
        'total_pv_generation': baseline_results['metrics']['total_pv_generation'],
        'total_grid_import': baseline_results['grid_draw_total'],
        'total_grid_export': baseline_results['grid_feed_total'],
        'self_sufficiency_percentage': baseline_results['metrics']['self_sufficiency_percentage'],
        'grid_dependency_percentage': baseline_results['metrics']['grid_dependency_percentage'],
        'energy_sharing_total': baseline_results['metrics']['energy_sharing_total']
    }
    
    print(f"基线策略评估结果:")
    print(f"  自给率: {baseline_metrics['self_sufficiency_percentage']:.2f}%")
    print(f"  能量共享: {baseline_metrics['energy_sharing_total']:.2f} kWh")
    print(f"  电网购电: {baseline_metrics['total_grid_import']:.2f} kWh")
    
    # 可视化基线结果
    visualize_results(baseline_results, save_dir=os.path.join(save_dir, f'baseline_{timestamp}'))
    
    results['baseline_policy'] = {
        'metrics': baseline_metrics,
        'results': baseline_results
    }
    
    # 比较结果并绘制
    if 'rl_policy' in results and 'baseline_policy' in results:
        # 计算性能差异
        performance_diff = {
            'self_sufficiency': results['rl_policy']['metrics']['self_sufficiency_percentage'] - 
                              results['baseline_policy']['metrics']['self_sufficiency_percentage'],
            'energy_sharing': results['rl_policy']['metrics']['energy_sharing_total'] - 
                            results['baseline_policy']['metrics']['energy_sharing_total'],
            'grid_import': results['rl_policy']['metrics']['total_grid_import'] - 
                          results['baseline_policy']['metrics']['total_grid_import'],
            'grid_export': results['rl_policy']['metrics']['total_grid_export'] - 
                          results['baseline_policy']['metrics']['total_grid_export'],
        }
        
        # 打印比较结果
        print("\n性能比较 (RL vs. 基线):")
        print(f"  自给率差异: {performance_diff['self_sufficiency']:.2f}% " + 
              ("(改进)" if performance_diff['self_sufficiency'] > 0 else "(下降)"))
        print(f"  能量共享差异: {performance_diff['energy_sharing']:.2f} kWh " +
              ("(增加)" if performance_diff['energy_sharing'] > 0 else "(减少)"))
        print(f"  电网购电差异: {performance_diff['grid_import']:.2f} kWh " +
              ("(增加)" if performance_diff['grid_import'] > 0 else "(减少)"))
        
        # 保存比较结果
        plot_comparison(results, os.path.join(save_dir, f'comparison_{timestamp}'))
        
        results['performance_diff'] = performance_diff
    
    return results