import types
import torch
import numpy as np

from td3_agent import TD3Agent

def integrate_rl_agent_with_energy_manager(energy_manager, agent_path):
    """
    将训练好的RL代理与现有能量管理器集成
    
    Args:
        energy_manager: 原有的能量管理器实例
        agent_path: RL代理模型路径
        
    Returns:
        enhanced_manager: 增强的能量管理器
    """
    # 创建RL代理
    num_nodes = energy_manager.num_nodes
    simulation_hours = energy_manager.simulation_hours
    
    # 计算状态和动作维度
    state_dim = 1 + num_nodes + 1 + num_nodes  # [当前小时, 各节点SOC, 云储能SOC, 各节点净负载]
    action_dim = num_nodes + 1  # [各节点与云端交互比例, 云端与电网交互比例]
    
    # 初始化代理
    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=1.0,
        hidden_dim=128
    )
    
    # 加载预训练模型
    agent.load(agent_path)
    
    # 为energy_manager添加RL相关属性
    energy_manager.rl_agent = agent
    energy_manager.use_rl = True
    
    # 创建构建状态的辅助方法
    def get_rl_state(self, t):
        """构建当前状态向量用于RL代理"""
        # 归一化当前小时 (0-1)
        hour_norm = t / self.simulation_hours
        
        # 获取各节点SOC
        edge_socs = [node.soc for node in self.edge_nodes]
        
        # 云储能SOC
        cloud_soc = (self.cloud_storage.energy_stored /
                     self.cloud_storage.capacity if 
                     self.cloud_storage.capacity > 0 else 0)
        
        # 获取各节点当前净负载
        net_loads = []
        for node in self.edge_nodes:
            node_id = node.id
            load = self.load_profiles[node_id][t]
            pv = self.pv_profiles[node_id][t]
            net_load = (load - pv) / max(node.battery_capacity, 1.0)  # 归一化
            net_loads.append(net_load)
        
        # 组合状态
        state = [hour_norm] + edge_socs + [cloud_soc] + net_loads
        
        return np.array(state, dtype=np.float32)
    
    # 创建增强的_process_timestep方法
    original_process_timestep = energy_manager._process_timestep
    
    def enhanced_process_timestep(self):
        """RL增强的时间步处理方法"""
        t = self.current_time
        
        # 使用RL代理优化决策
        if hasattr(self, 'use_rl') and self.use_rl:
            # 构建当前状态
            state = self.get_rl_state(t)
            
            # 使用RL代理预测动作
            with torch.no_grad():
                action = self.rl_agent.select_action(state, add_noise=False)
            
            # 应用RL决策到能量管理系统
            _apply_rl_actions_to_energy_manager(self, action, t)
        
        # 调用原始处理逻辑
        original_process_timestep(self)
    
    # 将辅助方法添加到energy_manager
    energy_manager.get_rl_state = types.MethodType(get_rl_state, energy_manager)
    
    # 替换方法
    energy_manager._process_timestep = types.MethodType(enhanced_process_timestep, energy_manager)
    
    return energy_manager

def _apply_rl_actions_to_energy_manager(energy_manager, actions, current_time):
    """
    将RL动作应用到能量管理系统
    
    Args:
        energy_manager: 能量管理器实例
        actions: RL动作
        current_time: 当前时间步
    """
    # 提取边缘节点动作和云端动作
    edge_actions = actions[:energy_manager.num_nodes]
    cloud_action = actions[-1]
    
    # 影响节点行为 - 这里使用动作值调整节点的交互策略
    # 正值表示更倾向于本地储能/充电，负值表示更倾向于能量共享/放电
    for i, action in enumerate(edge_actions):
        if i < len(energy_manager.edge_nodes):
            node = energy_manager.edge_nodes[i]
            
            # 示例: 调整节点的能量共享意愿或本地储能优先级
            # 这里只是设置一个属性，实际使用需要在能量管理器中添加相应逻辑
            node.rl_sharing_bias = action
            
            # 更多具体实现：
            # 1. 调整节点电池的放电阈值
            if hasattr(node, 'discharge_threshold'):
                # 正值增加放电阈值（更保守），负值降低放电阈值（更激进）
                node.discharge_threshold = 0.2 + 0.1 * action  # 例如：0.1~0.3
            
            # 2. 调整节点的共享比例
            if hasattr(node, 'sharing_ratio'):
                # 正值减少共享比例，负值增加共享比例
                node.sharing_ratio = 0.5 - 0.3 * action  # 例如：0.2~0.8
    
    # 影响云端行为 - 调整云储能和电网交互策略
    # 例如调整云储能的充放电阈值
    energy_manager.cloud_storage.rl_grid_bias = cloud_action
    
    # 更多具体实现：
    # 1. 调整云储能的电网交互阈值
    if hasattr(energy_manager.cloud_storage, 'grid_interaction_threshold'):
        # 正值增加与电网交互阈值（减少交互），负值降低阈值（增加交互）
        energy_manager.cloud_storage.grid_interaction_threshold = 0.3 + 0.2 * cloud_action
    
    # 2. 调整云储能优先级分配
    if hasattr(energy_manager, 'cloud_priority'):
        # 正值优先服务有盈余的节点，负值优先服务有缺口的节点
        energy_manager.cloud_priority = cloud_action  # -1~1范围

def create_optimized_energy_manager(
    num_nodes, 
    simulation_hours, 
    edge_battery_capacities, 
    cloud_storage_capacity,
    rl_model_path, 
    load_profiles=None, 
    pv_profiles=None, 
    start_time=None
):
    """
    创建一个带有RL优化的能量管理器
    
    Args:
        num_nodes: 节点数量
        simulation_hours: 仿真时长
        edge_battery_capacities: 电池容量列表
        cloud_storage_capacity: 云储能容量
        rl_model_path: RL模型路径
        load_profiles: 负载曲线
        pv_profiles: 光伏曲线
        start_time: 开始时间
    
    Returns:
        optimized_manager: RL优化的能量管理器
    """
    # 导入能量管理器
    from energy_manager import EnergyManager
    
    # 创建基本能量管理器
    base_manager = EnergyManager(
        num_nodes=num_nodes,
        simulation_hours=simulation_hours,
        edge_battery_capacities=edge_battery_capacities,
        cloud_storage_capacity=cloud_storage_capacity,
        load_profiles=load_profiles,
        pv_profiles=pv_profiles,
        start_time=start_time
    )
    
    # 应用RL优化
    optimized_manager = integrate_rl_agent_with_energy_manager(
        base_manager, 
        rl_model_path
    )
    
    return optimized_manager
