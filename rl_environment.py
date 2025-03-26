import numpy as np
from datetime import datetime
import sys
import os
import types

# Add parent directory to path to import test.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the original components
from test import EdgeNode, CloudStorage, EnergyManager

class RLEnergyManager:
    """
    基于强化学习的能量管理系统
    """
    def __init__(
        self,
        num_nodes,
        simulation_hours,
        edge_battery_capacities,
        cloud_storage_capacity,
        load_profiles=None,
        pv_profiles=None,
        start_time=None
    ):
        """
        初始化基于RL的能量管理环境
        
        Args:
            num_nodes: 边缘节点数量
            simulation_hours: 仿真时长（小时）
            edge_battery_capacities: 每个节点的电池容量列表
            cloud_storage_capacity: 云储能容量
            load_profiles: 可选的预定义负载曲线
            pv_profiles: 可选的预定义光伏发电曲线
            start_time: 仿真开始时间
        """
        self.energy_manager = EnergyManager(
            num_nodes=num_nodes,
            simulation_hours=simulation_hours,
            edge_battery_capacities=edge_battery_capacities,
            cloud_storage_capacity=cloud_storage_capacity,
            load_profiles=load_profiles,
            pv_profiles=pv_profiles,
            start_time=start_time
        )
        
        self.num_nodes = num_nodes
        self.simulation_hours = simulation_hours
        self.current_hour = 0
        self.done = False
        
        # 设置状态空间和动作空间维度
        # 状态: [当前小时/总时长, 各节点SOC, 云储能SOC, 各节点净负载/电池容量]
        self.state_dim = 1 + num_nodes + 1 + num_nodes
        
        # 动作: [各节点电池放电倾向, 云端储能放电倾向]
        # 正值表示倾向于保留能量，负值表示倾向于放电/共享能量
        self.action_dim = num_nodes + 1
        
        # 记录每步的性能指标
        self.step_rewards = []
        self.grid_imports = []
        self.grid_exports = []
        self.energy_sharing = []
        self.shortfalls = []
        self.action_history = []
        
        # 上一步的指标，用于计算差值奖励
        self.last_grid_draw = 0
        self.last_energy_sharing = 0
        self.last_shortfall = 0
        
        # 保存原始_process_timestep方法，并替换为增强版本
        self.original_process_timestep = self.energy_manager._process_timestep
        self.energy_manager._process_timestep = types.MethodType(self._enhanced_process_timestep, self.energy_manager)
        
    def _enhanced_process_timestep(self, energy_manager):
        """
        增强版的_process_timestep方法，接受并应用RL动作
        
        Args:
            energy_manager: 能量管理器实例
        """
        # 首先应用当前的RL决策到各节点和云储能
        if hasattr(energy_manager, 'current_rl_actions'):
            actions = energy_manager.current_rl_actions
            
            # 应用节点动作
            for i, node in enumerate(energy_manager.edge_nodes):
                if i < len(actions) - 1:
                    # 设置节点放电修饰器
                    # 将[-1,1]范围的动作转换为[0,1]范围的放电修饰器
                    # 值越低，越倾向于放电/共享能量
                    discharge_modifier = (1.0 + actions[i]) / 2.0
                    node.rl_discharge_modifier = discharge_modifier
            
            # 应用云储能动作
            if len(actions) > 0:
                cloud_action = actions[-1]
                energy_manager.cloud_storage.rl_action = cloud_action
        
        # 调用原始方法
        self.original_process_timestep()
        
    def get_state(self):
        """
        获取当前环境状态
        
        Returns:
            state: 当前状态向量
        """
        # 归一化当前小时 (0-1)
        hour_norm = self.current_hour / self.simulation_hours
        
        # 获取各节点SOC
        edge_socs = [node.soc for node in self.energy_manager.edge_nodes]
        
        # 云储能SOC
        cloud_soc = (self.energy_manager.cloud_storage.energy_stored /
                     self.energy_manager.cloud_storage.capacity if 
                     self.energy_manager.cloud_storage.capacity > 0 else 0)
        
        # 获取当前和预测的净负载 (正值为缺口，负值为过剩)
        net_loads = []
        for node in self.energy_manager.edge_nodes:
            node_id = node.id
            if self.current_hour < self.simulation_hours:
                load = self.energy_manager.load_profiles[node_id][self.current_hour]
                pv = self.energy_manager.pv_profiles[node_id][self.current_hour]
                # 归一化净负载
                net_load = (load - pv) / max(node.battery_capacity, 1.0)
                net_loads.append(net_load)
            else:
                # 最后一步使用前一个时间步的数据
                net_loads.append(0)
            
        # 组合状态
        state = [hour_norm] + edge_socs + [cloud_soc] + net_loads
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        """
        执行一步仿真并返回下一状态、奖励和终止标志
        
        Args:
            action: 代理生成的动作向量
            
        Returns:
            next_state: 下一个状态
            reward: 当前步骤的奖励
            done: 是否结束
            info: 额外信息
        """
        if self.done:
            return self.get_state(), 0, True, {}
            
        # 保存动作历史
        self.action_history.append(action.copy())
        
        # 设置当前RL动作
        self.energy_manager.current_rl_actions = action
        
        # 记录操作前的状态
        pre_grid_draw_total = sum(self.energy_manager.grid_draw_history) if self.energy_manager.grid_draw_history else 0
        pre_grid_feed_total = sum(self.energy_manager.grid_feed_history) if self.energy_manager.grid_feed_history else 0
        pre_energy_sharing = sum(node.energy_received_from_sharing for node in self.energy_manager.edge_nodes)
        
        # 执行一步仿真
        t = self.current_hour
        self.energy_manager._process_timestep()
        
        # 计算这一步的实际指标变化
        grid_draw_total = sum(self.energy_manager.grid_draw_history)
        grid_feed_total = sum(self.energy_manager.grid_feed_history)
        energy_sharing = sum(node.energy_received_from_sharing for node in self.energy_manager.edge_nodes)
        
        # 计算本步的差值
        grid_draw = grid_draw_total - pre_grid_draw_total
        grid_feed = grid_feed_total - pre_grid_feed_total
        energy_shared = energy_sharing - pre_energy_sharing
        
        # 计算负载短缺量
        shortfall = sum(node.shortfall_history[-1] for node in self.energy_manager.edge_nodes) if t < len(self.energy_manager.edge_nodes[0].shortfall_history) else 0
        
        # 记录指标
        self.grid_imports.append(grid_draw)
        self.grid_exports.append(grid_feed)
        self.energy_sharing.append(energy_shared)
        self.shortfalls.append(shortfall)
        
        # 计算奖励 (多目标优化)
        reward = self._calculate_reward(grid_draw, grid_feed, energy_shared, shortfall)
        self.step_rewards.append(reward)
        
        # 更新当前小时
        self.current_hour += 1
        
        # 检查是否结束
        self.done = (self.current_hour >= self.simulation_hours)
        
        # 更新上一步指标
        self.last_grid_draw = grid_draw
        self.last_energy_sharing = energy_shared
        self.last_shortfall = shortfall
        
        # 获取下一个状态
        next_state = self.get_state()
        
        info = {
            'grid_draw': grid_draw,
            'grid_feed': grid_feed,
            'energy_shared': energy_shared,
            'shortfall': shortfall
        }
        
        return next_state, reward, self.done, info
    
    def _calculate_reward(self, grid_draw, grid_feed, energy_shared, shortfall):
        """
        计算多目标奖励函数
        
        Args:
            grid_draw: 从电网购电量
            grid_feed: 向电网售电量
            energy_shared: 节点间共享的能量
            shortfall: 负载缺口
            
        Returns:
            reward: 综合奖励
        """
        # 奖励权重 - 显著提高能量共享和减少短缺的权重
        w_grid_draw = -2.0    # 减少从电网购电
        w_grid_feed = -0.5    # 适度减少向电网售电 (损失能量)
        w_energy_shared = 4.0 # 强烈鼓励节点间能量共享
        w_shortfall = -8.0    # 严重惩罚负载缺口
        
        # 归一化各项指标
        # 使用更合理的基准值进行归一化
        total_battery_capacity = sum(node.battery_capacity for node in self.energy_manager.edge_nodes)
        
        # 避免除零错误
        if total_battery_capacity == 0:
            total_battery_capacity = 1.0
            
        norm_grid_draw = grid_draw / (total_battery_capacity / 5)  # 使用总电池容量的20%作为参考
        norm_grid_feed = grid_feed / (total_battery_capacity / 5)
        norm_energy_shared = energy_shared / (total_battery_capacity / 10)  # 使用总电池容量的10%作为参考
        norm_shortfall = shortfall / (total_battery_capacity / 20)  # 使用总电池容量的5%作为参考
        
        # 计算综合奖励，不同组件的范围更均衡
        reward = (w_grid_draw * norm_grid_draw + 
                  w_grid_feed * norm_grid_feed + 
                  w_energy_shared * norm_energy_shared + 
                  w_shortfall * norm_shortfall)
        
        # 添加额外奖励以激励能量共享增长
        if energy_shared > self.last_energy_sharing:
            reward += 1.0  # 额外奖励能量共享增加的行为
        
        # 打印调试信息，了解奖励构成
        if self.current_hour % 20 == 0:
            print(f"Reward breakdown: grid_draw={w_grid_draw * norm_grid_draw:.2f}, "
                  f"grid_feed={w_grid_feed * norm_grid_feed:.2f}, "
                  f"energy_shared={w_energy_shared * norm_energy_shared:.2f}, "
                  f"shortfall={w_shortfall * norm_shortfall:.2f}")
        
        return reward
    
    def reset(self):
        """
        重置环境到初始状态
        
        Returns:
            state: 初始状态
        """
        # 重置EnergyManager - 使用相同的配置创建新实例
        self.energy_manager = EnergyManager(
            num_nodes=self.num_nodes,
            simulation_hours=self.simulation_hours,
            edge_battery_capacities=[node.battery_capacity for node in self.energy_manager.edge_nodes],
            cloud_storage_capacity=self.energy_manager.cloud_storage.capacity,
            load_profiles=self.energy_manager.load_profiles,
            pv_profiles=self.energy_manager.pv_profiles,
            start_time=self.energy_manager.start_time
        )
        
        # 保存原始_process_timestep方法，并替换为增强版本
        self.original_process_timestep = self.energy_manager._process_timestep
        self.energy_manager._process_timestep = types.MethodType(self._enhanced_process_timestep, self.energy_manager)
        
        # 重置内部状态
        self.current_hour = 0
        self.done = False
        
        # 清空记录
        self.step_rewards = []
        self.grid_imports = []
        self.grid_exports = []
        self.energy_sharing = []
        self.shortfalls = []
        self.action_history = []
        
        # 重置上一步指标
        self.last_grid_draw = 0
        self.last_energy_sharing = 0
        self.last_shortfall = 0
        
        # 返回初始状态
        return self.get_state()
    
    def run_episode(self, agent, exploration=True):
        """
        使用指定代理运行一个完整的仿真周期
        
        Args:
            agent: RL代理
            exploration: 是否执行探索
            
        Returns:
            total_reward: 总奖励
            info: 额外信息
        """
        state = self.reset()
        total_reward = 0
        
        all_states = []
        all_actions = []
        all_rewards = []
        
        while not self.done:
            # 选择动作
            action = agent.select_action(state, add_noise=exploration, 
                                        noise_scale=0.3 if exploration else 0.0)
            
            # 执行动作
            next_state, reward, done, info = self.step(action)
            
            # 累计奖励
            total_reward += reward
            
            # 保存轨迹
            all_states.append(state)
            all_actions.append(action)
            all_rewards.append(reward)
            
            # 更新状态
            state = next_state
        
        # 计算最终性能指标
        final_metrics = self._calculate_final_metrics()
        
        return total_reward, final_metrics, {
            'states': all_states,
            'actions': all_actions,
            'rewards': all_rewards
        }
    
    def _calculate_final_metrics(self):
        """
        计算最终性能指标
        
        Returns:
            metrics: 指标字典
        """
        # 总负载
        total_load = sum(sum(loads) for loads in self.energy_manager.load_profiles.values())
        
        # 总光伏发电
        total_pv = sum(sum(pvs) for pvs in self.energy_manager.pv_profiles.values())
        
        # 总从电网购电
        total_grid_import = sum(self.grid_imports)
        
        # 总向电网售电
        total_grid_export = sum(self.grid_exports)
        
        # 计算自给率
        grid_dependency = total_grid_import / total_load * 100 if total_load > 0 else 0
        self_sufficiency = 100 - grid_dependency
        
        # 总能量共享
        total_energy_shared = sum(self.energy_sharing)
        
        # 总负载缺口
        total_shortfall = sum(self.shortfalls)
        
        metrics = {
            'total_load': total_load,
            'total_pv_generation': total_pv,
            'total_grid_import': total_grid_import,
            'total_grid_export': total_grid_export,
            'self_sufficiency_percentage': self_sufficiency,
            'grid_dependency_percentage': grid_dependency,
            'energy_sharing_total': total_energy_shared,
            'total_shortfall': total_shortfall
        }
        
        return metrics