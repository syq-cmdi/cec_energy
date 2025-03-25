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
    batch_size=64,  # 较小的批次可能更稳定
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
    print("训练前配置:")
    print(f"节点数: {num_nodes}, 仿真时长: {simulation_hours}小时")
    print(f"电池容量: {edge_battery_capacities}")
    print(f"云储能容量: {cloud_storage_capacity}")
    
    # 设置默认电池容量（如果未指定）
    if edge_battery_capacities is None:
        edge_battery_capacities = [30] * num_nodes
    
    # 创建环境
    try:
        env = RLEnergyManager(
            num_nodes=num_nodes,
            simulation_hours=simulation_hours,
            edge_battery_capacities=edge_battery_capacities,
            cloud_storage_capacity=cloud_storage_capacity,
            start_time=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        )
    except Exception as e:
        print(f"创建环境时出错: {e}")
        raise
    
    # 打印状态和动作空间信息
    print(f"状态空间维度: {env.state_dim}")
    print(f"动作空间维度: {env.action_dim}")
    
    # 获取一个初始状态样本进行检查
    state_sample = env.get_state()
    print(f"状态样本形状: {state_sample.shape}, 类型: {type(state_sample)}")
    print(f"状态样本值: {state_sample}")
    
    # 创建RL代理
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    # 使用TD3算法，降低学习率以提高稳定性
    try:
        agent = TD3Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=1.0,
            hidden_dim=128,
            actor_lr=5e-5,  # 降低学习率
            critic_lr=5e-5   # 降低学习率
        )
    except Exception as e:
        print(f"创建TD3代理时出错: {e}")
        raise
    
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
    warmup_episodes = min(100, episodes // 10)
    for _ in range(warmup_episodes):
        state = env.reset()
        
        done = False
        while not done:
            # 完全随机动作
            action = np.random.uniform(-1, 1, size=env.action_dim)
            next_state, reward, done, _ = env.step(action)
            
            # 保存经验
            replay_buffer.add(state, action, reward, next_state, done)
            
            state = next_state
    
    print(f"预热完成，经验池大小: {len(replay_buffer)}")
    
    # 主训练循环
    for episode in range(1, episodes + 1):
        # 设置探索噪声 - 使用指数衰减获得更好的探索策略
        exploration_noise = max(0.3 * np.exp(-episode / (episodes / 3)), 0.05)
        
        # 运行一个回合
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        while not done:
            # 选择动作 - 添加错误处理
            try:
                action = agent.select_action(state, add_noise=True, noise_scale=exploration_noise)
            except Exception as e:
                print(f"选择动作时出错: {e}")
                print(f"状态形状: {np.array(state).shape}, 值: {state[:5]}...")
                # 使用随机动作作为备选
                action = np.random.uniform(-1, 1, size=env.action_dim)
            
            # 执行动作
            try:
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                episode_steps += 1
            except Exception as e:
                print(f"执行动作时出错: {e}")
                # 如果环境出错，终止当前回合
                break
            
            # 保存经验
            replay_buffer.add(state, action, reward, next_state, done)
            
            # 更新状态
            state = next_state
            
            # 训练智能体 - 更保守的更新策略
            if len(replay_buffer) > batch_size * 5:  # 确保足够的样本
                # 减少更新次数以防止过拟合
                update_count = 1
                for _ in range(update_count):
                    try:
                        critic_loss, actor_loss = agent.update(replay_buffer, batch_size)
                        if critic_loss is not None:
                            critic_losses.append(critic_loss)
                        if actor_loss is not None:
                            actor_losses.append(actor_loss)
                    except Exception as e:
                        print(f"更新网络时出错: {e}")
        
        # 计算最终指标
        try:
            metrics = env._calculate_final_metrics()
        except Exception as e:
            print(f"计算指标时出错: {e}")
            # 创建默认指标
            metrics = {
                'self_sufficiency_percentage': 0,
                'energy_sharing_total': 0,
                'total_grid_import': 0,
                'total_grid_export': 0
            }
        
        # 记录训练指标
        rewards_history.append(episode_reward)
        self_sufficiency_history.append(metrics['self_sufficiency_percentage'])
        energy_sharing_history.append(metrics['energy_sharing_total'])
        
        # 打印训练进度
        if episode % 10 == 0 or episode == 1:
            avg_reward = np.mean(rewards_history[-10:]) if episode > 10 else episode_reward
            print(f"Episode {episode}/{episodes} - 奖励: {episode_reward:.2f}, 平均奖励: {avg_reward:.2f}")
            print(f"  自给率: {metrics['self_sufficiency_percentage']:.2f}%, 能量共享: {metrics['energy_sharing_total']:.2f} kWh")
            print(f"  电网购电: {metrics['total_grid_import']:.2f} kWh, 电网售电: {metrics['total_grid_export']:.2f} kWh")
            print(f"  探索噪声: {exploration_noise:.2f}")
            
            # 如果有损失数据，打印平均损失
            if critic_losses:
                avg_critic_loss = np.mean(critic_losses[-100:])
                print(f"  平均Critic损失: {avg_critic_loss:.4f}")
            if actor_losses:
                # 过滤掉None值
                valid_losses = [loss for loss in actor_losses[-100:] if loss is not None]
                if valid_losses:
                    avg_actor_loss = np.mean(valid_losses)
                    print(f"  平均Actor损失: {avg_actor_loss:.4f}")
        
        # 保存模型检查点
        if episode % save_interval == 0 or episode == episodes:
            try:
                checkpoint_dir = os.path.join(model_dir, f"checkpoint_{episode}")
                agent.save(checkpoint_dir)
                print(f"模型已保存至: {checkpoint_dir}")
            except Exception as e:
                print(f"保存模型时出错: {e}")
        
        # 保存最佳模型
        if episode_reward > best_reward:
            best_reward = episode_reward
            try:
                agent.save(os.path.join(model_dir, "best_reward_model"))
                print(f"新的最佳奖励模型已保存: {best_reward:.2f}")
            except Exception as e:
                print(f"保存最佳奖励模型时出错: {e}")
        
        if metrics['self_sufficiency_percentage'] > best_self_sufficiency:
            best_self_sufficiency = metrics['self_sufficiency_percentage']
            try:
                agent.save(os.path.join(model_dir, "best_self_sufficiency_model"))
                print(f"新的最佳自给率模型已保存: {best_self_sufficiency:.2f}%")
            except Exception as e:
                print(f"保存最佳自给率模型时出错: {e}")
        
        # 评估模型（无探索）
        if episode % eval_interval == 0 or episode == episodes:
            try:
                eval_reward, eval_metrics, _ = env.run_episode(agent, exploration=False)
                print(f"\n评估结果 (Episode {episode}):")
                print(f"  奖励: {eval_reward:.2f}")
                print(f"  自给率: {eval_metrics['self_sufficiency_percentage']:.2f}%")
                print(f"  能量共享: {eval_metrics['energy_sharing_total']:.2f} kWh")
                print(f"  电网购电: {eval_metrics['total_grid_import']:.2f} kWh")
                print(f"  电网售电: {eval_metrics['total_grid_export']:.2f} kWh\n")
            except Exception as e:
                print(f"评估模型时出错: {e}")
                print("继续训练...")
    
    # 训练结束后绘制结果
    try:
        plot_training_results(
            rewards_history, 
            critic_losses, 
            actor_losses, 
            self_sufficiency_history, 
            energy_sharing_history,
            results_dir
        )
    except Exception as e:
        print(f"绘制训练结果时出错: {e}")
    
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
    try:
        env = RLEnergyManager(
            num_nodes=num_nodes,
            simulation_hours=simulation_hours,
            edge_battery_capacities=edge_battery_capacities,
            cloud_storage_capacity=cloud_storage_capacity,
            start_time=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        )
    except Exception as e:
        print(f"创建评估环境时出错: {e}")
        raise
    
    # 加载RL代理
    if agent is None and model_path is not None:
        try:
            agent = TD3Agent(
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                max_action=1.0,
                hidden_dim=128
            )
            agent.load(model_path)
        except Exception as e:
            print(f"加载模型时出错: {e}")
            raise
    
    results = {}
    
    # 运行RL策略
    if agent is not None:
        print("评估RL策略...")
        try:
            rl_reward, rl_metrics, rl_trajectory = env.run_episode(agent, exploration=False)
            print(f"RL策略评估结果:")
            print(f"  奖励: {rl_reward:.2f}")
            print(f"  自给率: {rl_metrics['self_sufficiency_percentage']:.2f}%")
            print(f"  能量共享: {rl_metrics['energy_sharing_total']:.2f} kWh")
            print(f"  电网购电: {rl_metrics['total_grid_import']:.2f} kWh")
            
            # 可视化RL策略结果
            try:
                plot_evaluation_results(
                    env, 
                    rl_trajectory, 
                    'RL Policy', 
                    os.path.join(save_dir, f'rl_policy_{timestamp}')
                )
            except Exception as e:
                print(f"可视化RL结果时出错: {e}")
            
            results['rl_policy'] = {
                'reward': rl_reward,
                'metrics': rl_metrics,
                'trajectory': rl_trajectory
            }
        except Exception as e:
            print(f"评估RL策略时出错: {e}")
    
    # 运行基线策略（原始能量管理逻辑）
    print("\n运行基线策略...")
    # 导入可视化函数
    from visualization import visualize_results
    
    try:
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
        try:
            visualize_results(baseline_results, save_dir=os.path.join(save_dir, f'baseline_{timestamp}'))
        except Exception as e:
            print(f"可视化基线结果时出错: {e}")
        
        results['baseline_policy'] = {
            'metrics': baseline_metrics,
            'results': baseline_results
        }
    except Exception as e:
        print(f"评估基线策略时出错: {e}")
    
    # 比较结果并绘制
    if 'rl_policy' in results and 'baseline_policy' in results:
        try:
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
            try:
                plot_comparison(results, os.path.join(save_dir, f'comparison_{timestamp}'))
            except Exception as e:
                print(f"绘制比较结果时出错: {e}")
            
            results['performance_diff'] = performance_diff
        except Exception as e:
            print(f"计算性能差异时出错: {e}")
    
    return results
