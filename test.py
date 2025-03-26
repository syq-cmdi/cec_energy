import numpy as np  
import random
import torch

import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Tuple, Any
from datetime import datetime, timedelta
import os
import matplotlib.dates as mdates

# ------------------------------
# 边缘节点类，增加了电池充放电效率参数
# ------------------------------
class EdgeNode:
    def __init__(self, id: int, battery_capacity: float, initial_soc: float = 0.5, battery_efficiency: float = 0.95):
        """
        初始化具有电池储能功能的边缘节点

        Args:
            id: 节点唯一标识
            battery_capacity: 电池最大储能容量 (kWh)
            initial_soc: 初始充电状态 (0~1)
            battery_efficiency: 电池充放电效率（例如 0.95 表示 95% 效率）
        """
        self.id = id
        self.battery_capacity = battery_capacity
        self.soc = initial_soc  # 充电状态
        self.energy_stored = initial_soc * battery_capacity
        self.battery_efficiency = battery_efficiency

        # 用于跟踪节点性能指标
        self.energy_drawn_from_battery = 0
        self.energy_charged_to_battery = 0
        self.energy_received_from_sharing = 0
        self.energy_contributed_to_sharing = 0
        self.shortfall_history = []
        self.surplus_history = []
        self.soc_history = [initial_soc]
        
        # 新增超循环特征
        self.charge_discharge_cycles = 0
        self.cycle_depth_history = []
        self.cycle_timestamps = []
        self.last_soc_direction = None
        self.last_cycle_peak = initial_soc
        self.last_cycle_valley = initial_soc

    def local_balance(self, net_load: float, timestamp: datetime) -> float:
        """
        尝试利用本地电池平衡节点负载

        Args:
            net_load: 正值表示负载大于发电（缺口），负值表示有多余发电
            timestamp: 当前时间戳，用于记录循环的时间
        Returns:
            remaining_net_load: 经本地平衡后剩余的负载 (完全平衡时为 0)
        """
        # 保存旧SOC用于检测循环特征
        old_soc = self.soc
        
        # 当负载为正（缺口）时，从电池放电（考虑放电效率）
        if net_load > 0:
            available_energy = self.energy_stored * self.battery_efficiency
            max_discharge = min(available_energy, net_load)
            # 更新电池能量：扣除放电时需要反推效率
            self.energy_stored -= max_discharge / self.battery_efficiency
            self.energy_drawn_from_battery += max_discharge
            remaining_net_load = net_load - max_discharge

            # 记录短缺情况
            self.shortfall_history.append(remaining_net_load if remaining_net_load > 0 else 0)
            self.surplus_history.append(0)
            
        # 当负载为负（剩余能量）时，对电池进行充电（考虑充电效率）
        elif net_load < 0:
            surplus = -net_load  # 转为正值
            available_capacity = self.battery_capacity - self.energy_stored
            # 充电时：输入能量会有一部分损失，所以能存入的能量为 surplus * efficiency
            energy_to_store = min(available_capacity, surplus * self.battery_efficiency)
            self.energy_stored += energy_to_store
            self.energy_charged_to_battery += energy_to_store
            # 实际吸收的 surplus = energy_to_store / efficiency
            remaining_surplus = surplus - energy_to_store / self.battery_efficiency
            remaining_net_load = -remaining_surplus

            self.shortfall_history.append(0)
            self.surplus_history.append(remaining_surplus)
            
        # 完美平衡情况
        else:
            remaining_net_load = 0
            self.shortfall_history.append(0)
            self.surplus_history.append(0)
            
        # 更新 SOC
        self.soc = self.energy_stored / self.battery_capacity
        self.soc_history.append(self.soc)
        
        # 超循环特征检测
        self._detect_charge_cycles(old_soc, timestamp)
        
        return remaining_net_load

    def _detect_charge_cycles(self, old_soc: float, timestamp: datetime) -> None:
        """
        检测并记录充放电循环
        
        Args:
            old_soc: 更新前的SOC值
            timestamp: 当前时间戳
        """
        # 检测SOC方向变化
        current_direction = None
        if self.soc > old_soc:
            current_direction = "charging"
        elif self.soc < old_soc:
            current_direction = "discharging"
        else:
            # SOC没有变化，保持前一个方向
            current_direction = self.last_soc_direction
        
        # 如果这是第一次设置方向
        if self.last_soc_direction is None:
            self.last_soc_direction = current_direction
            return
            
        # 检测循环拐点
        if current_direction != self.last_soc_direction:
            if current_direction == "charging" and self.last_soc_direction == "discharging":
                # 从放电转为充电，记录循环谷值
                self.last_cycle_valley = old_soc
            elif current_direction == "discharging" and self.last_soc_direction == "charging":
                # 从充电转为放电，记录循环峰值并计算循环深度
                self.last_cycle_peak = old_soc
                cycle_depth = self.last_cycle_peak - self.last_cycle_valley
                
                # 只记录有意义的循环（深度大于阈值）
                if cycle_depth > 0.05:  # 5%的SOC变化作为最小循环深度
                    self.charge_discharge_cycles += 1
                    self.cycle_depth_history.append(cycle_depth)
                    self.cycle_timestamps.append(timestamp)
        
        # 更新方向
        self.last_soc_direction = current_direction

    def receive_shared_energy(self, amount: float) -> None:
        """记录通过能量共享接收到的能量"""
        self.energy_received_from_sharing += amount
        
    def contribute_shared_energy(self, amount: float) -> None:
        """记录通过能量共享贡献的能量"""
        self.energy_contributed_to_sharing += amount


# ------------------------------
# 云级储能类
# ------------------------------
class CloudStorage:
    def __init__(self, capacity: float, initial_energy: float = 0):
        """
        初始化云级储能

        Args:
            capacity: 最大储能容量 (kWh)
            initial_energy: 初始储能 (kWh)
        """
        self.capacity = capacity
        self.energy_stored = initial_energy
        
        # 跟踪储能变化
        self.energy_charged_history = []
        self.energy_discharged_history = []
        self.energy_level_history = [initial_energy]
        
        # 超循环特征
        self.charge_discharge_cycles = 0
        self.cycle_depth_history = []
        self.cycle_timestamps = []
        self.last_direction = None
        self.last_cycle_peak = initial_energy / capacity if capacity > 0 else 0
        self.last_cycle_valley = self.last_cycle_peak
        
    def charge(self, energy_amount: float, timestamp: datetime) -> float:
        """
        为云储能充电

        Args:
            energy_amount: 待充电能量
            timestamp: 当前时间戳
        Returns:
            amount_stored: 实际存储的能量（可能受容量限制）
        """
        old_level = self.energy_stored / self.capacity if self.capacity > 0 else 0
        
        available_capacity = self.capacity - self.energy_stored
        amount_stored = min(energy_amount, available_capacity)
        self.energy_stored += amount_stored
        self.energy_charged_history.append(amount_stored)
        self.energy_level_history.append(self.energy_stored)
        
        # 检测循环
        new_level = self.energy_stored / self.capacity if self.capacity > 0 else 0
        self._detect_charge_cycles(old_level, new_level, timestamp)
        
        return amount_stored
    
    def discharge(self, energy_request: float, timestamp: datetime) -> float:
        """
        从云储能放电

        Args:
            energy_request: 请求放电的能量
            timestamp: 当前时间戳
        Returns:
            amount_discharged: 实际放出的能量（可能受储量限制）
        """
        old_level = self.energy_stored / self.capacity if self.capacity > 0 else 0
        
        amount_discharged = min(energy_request, self.energy_stored)
        self.energy_stored -= amount_discharged
        self.energy_discharged_history.append(amount_discharged)
        self.energy_level_history.append(self.energy_stored)
        
        # 检测循环
        new_level = self.energy_stored / self.capacity if self.capacity > 0 else 0
        self._detect_charge_cycles(old_level, new_level, timestamp)
        
        return amount_discharged
    
    def _detect_charge_cycles(self, old_level: float, new_level: float, timestamp: datetime) -> None:
        """
        检测并记录云储能的充放电循环
        
        Args:
            old_level: 更新前的能量水平（归一化为0-1）
            new_level: 更新后的能量水平（归一化为0-1）
            timestamp: 当前时间戳
        """
        # 检测方向变化
        current_direction = None
        if new_level > old_level:
            current_direction = "charging"
        elif new_level < old_level:
            current_direction = "discharging"
        else:
            # 能量水平没有变化，保持前一个方向
            current_direction = self.last_direction
        
        # 如果这是第一次设置方向
        if self.last_direction is None:
            self.last_direction = current_direction
            return
            
        # 检测循环拐点
        if current_direction != self.last_direction:
            if current_direction == "charging" and self.last_direction == "discharging":
                # 从放电转为充电，记录循环谷值
                self.last_cycle_valley = old_level
            elif current_direction == "discharging" and self.last_direction == "charging":
                # 从充电转为放电，记录循环峰值并计算循环深度
                self.last_cycle_peak = old_level
                cycle_depth = self.last_cycle_peak - self.last_cycle_valley
                
                # 只记录有意义的循环（深度大于阈值）
                if cycle_depth > 0.05:  # 5%的容量变化作为最小循环深度
                    self.charge_discharge_cycles += 1
                    self.cycle_depth_history.append(cycle_depth)
                    self.cycle_timestamps.append(timestamp)
        
        # 更新方向
        self.last_direction = current_direction


# ------------------------------
# 能量管理器：处理边缘与云储能的调度和能量共享
# ------------------------------
class EnergyManager:
    def __init__(self, 
                 num_nodes: int, 
                 simulation_hours: int,
                 edge_battery_capacities: List[float],
                 cloud_storage_capacity: float,
                 load_profiles: Dict[int, List[float]] = None,
                 pv_profiles: Dict[int, List[float]] = None,
                 start_time: datetime = None):
        """
        初始化能量管理系统

        Args:
            num_nodes: 边缘节点数量
            simulation_hours: 仿真时长（小时）
            edge_battery_capacities: 每个节点的电池容量列表
            cloud_storage_capacity: 云储能容量
            load_profiles: 可选的预定义负载曲线
            pv_profiles: 可选的预定义光伏发电曲线
            start_time: 仿真开始时间，如果为None则使用当前时间
        """
        self.num_nodes = num_nodes
        self.simulation_hours = simulation_hours
        self.current_time = 0
        
        # 设置时间轴
        if start_time is None:
            self.start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            self.start_time = start_time
        self.timestamps = [self.start_time + timedelta(hours=i) for i in range(simulation_hours)]
        
        # 初始化边缘节点
        self.edge_nodes = []
        for i in range(num_nodes):
            node_id = i + 1
            battery_capacity = edge_battery_capacities[i]
            self.edge_nodes.append(EdgeNode(node_id, battery_capacity))
        
        # 初始化云储能
        self.cloud_storage = CloudStorage(cloud_storage_capacity)
        
        # 如果预定义了负载和光伏曲线，则直接使用；否则自动生成
        if load_profiles and pv_profiles:
            self.load_profiles = load_profiles
            self.pv_profiles = pv_profiles
        else:
            self.load_profiles, self.pv_profiles = self._generate_profiles()
            
        # 用于记录系统性能
        self.grid_feed_history = []
        self.grid_draw_history = []
        
    def _generate_profiles(self) -> Tuple[Dict[int, List[float]], Dict[int, List[float]]]:
        """
        为每个节点生成合成的负载和光伏发电曲线

        Returns:
            load_profiles: 节点负载曲线字典
            pv_profiles: 节点光伏发电曲线字典
        """
        load_profiles = {}
        pv_profiles = {}
        
        for i in range(self.num_nodes):
            node_id = i + 1
            
            # 生成具有日周期和随机波动的负载曲线
            base_load = np.random.uniform(5, 15)
            daily_pattern = np.sin(np.linspace(0, 2*np.pi, 24)) * 0.5 + 0.5
            daily_pattern = np.tile(daily_pattern, self.simulation_hours // 24 + 1)[:self.simulation_hours]
            noise = np.random.normal(0, 0.1, self.simulation_hours)
            load = base_load * (daily_pattern + noise)
            load = np.maximum(load, 0)
            load_profiles[node_id] = load.tolist()
            
            # 生成具有日周期和随机波动的光伏发电曲线
            pv_capacity = np.random.uniform(5, 20)
            solar_pattern = np.sin(np.linspace(0, np.pi, 12)) ** 2
            solar_pattern = np.concatenate([np.zeros(6), solar_pattern, np.zeros(6)])
            solar_pattern = np.tile(solar_pattern, self.simulation_hours // 24 + 1)[:self.simulation_hours]
            solar_noise = np.random.normal(0, 0.1, self.simulation_hours)
            solar_noise = np.maximum(solar_noise, -0.3)
            pv = pv_capacity * (solar_pattern + solar_noise)
            pv = np.maximum(pv, 0)
            pv_profiles[node_id] = pv.tolist()
            
        return load_profiles, pv_profiles
    
    def run_simulation(self) -> Dict[str, Any]:
        """
        运行边缘-云能量共享的完整仿真

        Returns:
            results: 包含仿真结果和各项指标的字典
        """
        for t in range(self.simulation_hours):
            self.current_time = t
            self._process_timestep()
            
        return self._calculate_results()
    
    def _process_timestep(self) -> None:
        """处理仿真中的单个时间步"""
        t = self.current_time
        current_timestamp = self.timestamps[t]
        
        # 获取各节点当前时刻的负载和光伏数据
        node_net_loads = {}
        for node in self.edge_nodes:
            node_id = node.id
            load = self.load_profiles[node_id][t]
            pv = self.pv_profiles[node_id][t]
            net_load = load - pv  # 正值表示缺口，负值表示盈余
            node_net_loads[node_id] = net_load
        
        # 每个节点进行本地平衡
        node_remaining_loads = {}
        for node in self.edge_nodes:
            remaining_load = node.local_balance(node_net_loads[node.id], current_timestamp)
            node_remaining_loads[node.id] = remaining_load
        
        # 汇总系统总缺口与盈余
        total_deficit = 0
        total_surplus = 0
        deficit_nodes = []
        surplus_nodes = []
        
        for node in self.edge_nodes:
            remaining_load = node_remaining_loads[node.id]
            if remaining_load > 0:
                total_deficit += remaining_load
                deficit_nodes.append((node, remaining_load))
            elif remaining_load < 0:
                surplus_amount = -remaining_load
                total_surplus += surplus_amount
                surplus_nodes.append((node, surplus_amount))
        
        grid_draw = 0
        grid_feed = 0

        # 定义能量共享过程中的传输效率（例如 95%）
        SHARING_EFFICIENCY = 0.95

        if total_surplus >= total_deficit:
            # 情况1：盈余足以覆盖缺口
            if total_deficit > 0:
                for deficit_node, deficit_amount in deficit_nodes:
                    for surplus_node, surplus_amount in surplus_nodes:
                        contribution = (surplus_amount / total_surplus) * deficit_amount
                        # 考虑共享传输过程的损耗
                        effective_contribution = contribution * SHARING_EFFICIENCY
                        surplus_node.contribute_shared_energy(contribution)
                        deficit_node.receive_shared_energy(effective_contribution)
            remaining_surplus = total_surplus - total_deficit
            cloud_charge = self.cloud_storage.charge(remaining_surplus, current_timestamp)
            grid_feed = remaining_surplus - cloud_charge
            for node, _ in deficit_nodes:
                node_remaining_loads[node.id] = 0
                
        else:
            # 情况2：盈余不足以覆盖所有缺口
            if total_surplus > 0:
                for surplus_node, surplus_amount in surplus_nodes:
                    for deficit_node, deficit_amount in deficit_nodes:
                        contribution = (deficit_amount / total_deficit) * surplus_amount
                        effective_contribution = contribution * SHARING_EFFICIENCY
                        surplus_node.contribute_shared_energy(contribution)
                        deficit_node.receive_shared_energy(effective_contribution)
                    # 更新各缺口节点的剩余负载（此处为简单近似）
                    node_remaining_loads[deficit_node.id] -= contribution
            remaining_deficit = total_deficit - total_surplus
            cloud_discharge = self.cloud_storage.discharge(remaining_deficit, current_timestamp)
            if cloud_discharge > 0:
                for deficit_node, deficit_amount in deficit_nodes:
                    allocation = (deficit_amount / total_deficit) * cloud_discharge
                    deficit_node.receive_shared_energy(allocation)
                    node_remaining_loads[deficit_node.id] -= allocation
            grid_draw = remaining_deficit - cloud_discharge

        self.grid_feed_history.append(grid_feed)
        self.grid_draw_history.append(grid_draw)
    
    def _calculate_results(self) -> Dict[str, Any]:
        """计算并整理仿真结果和各项指标"""
        results = {
            'grid_feed_total': sum(self.grid_feed_history),
            'grid_draw_total': sum(self.grid_draw_history),
            'grid_interaction_history': {
                'feed': self.grid_feed_history,
                'draw': self.grid_draw_history
            },
            'cloud_storage': {
                'final_energy': self.cloud_storage.energy_stored,
                'energy_level_history': self.cloud_storage.energy_level_history,
                'energy_charged_history': self.cloud_storage.energy_charged_history,
                'energy_discharged_history': self.cloud_storage.energy_discharged_history,
                'charge_discharge_cycles': self.cloud_storage.charge_discharge_cycles,
                'cycle_depth_history': self.cloud_storage.cycle_depth_history,
                'cycle_timestamps': self.cloud_storage.cycle_timestamps
            },
            'edge_nodes': {},
            'load_profiles': self.load_profiles,
            'pv_profiles': self.pv_profiles,
            'timestamps': self.timestamps
        }
        
        for node in self.edge_nodes:
            node_results = {
                'id': node.id,
                'final_soc': node.soc,
                'soc_history': node.soc_history,
                'energy_drawn_from_battery': node.energy_drawn_from_battery,
                'energy_charged_to_battery': node.energy_charged_to_battery,
                'energy_received_from_sharing': node.energy_received_from_sharing,
                'energy_contributed_to_sharing': node.energy_contributed_to_sharing,
                'shortfall_history': node.shortfall_history,
                'surplus_history': node.surplus_history,
                'battery_capacity': node.battery_capacity,
                'charge_discharge_cycles': node.charge_discharge_cycles,
                'cycle_depth_history': node.cycle_depth_history,
                'cycle_timestamps': node.cycle_timestamps
            }
            results['edge_nodes'][node.id] = node_results
            
        # 计算系统自给率指标
        total_load = 0
        total_pv = 0
        for node_id in range(1, self.num_nodes + 1):
            total_load += sum(self.load_profiles[node_id])
            total_pv += sum(self.pv_profiles[node_id])
        
        grid_dependency = results['grid_draw_total'] / total_load * 100 if total_load > 0 else 0
        self_sufficiency = 100 - grid_dependency
        
        # 计算超循环相关指标
        total_cycles = sum(node.charge_discharge_cycles for node in self.edge_nodes) + self.cloud_storage.charge_discharge_cycles
        avg_cycle_depth = np.mean([depth for node in self.edge_nodes for depth in node.cycle_depth_history] + 
                                  self.cloud_storage.cycle_depth_history) if total_cycles > 0 else 0
        
        results['metrics'] = {
            'total_load': total_load,
            'total_pv_generation': total_pv,
            'grid_dependency_percentage': grid_dependency,
            'self_sufficiency_percentage': self_sufficiency,
            'energy_sharing_total': sum(node.energy_received_from_sharing for node in self.edge_nodes),
            'total_cycles': total_cycles,
            'avg_cycle_depth': avg_cycle_depth
        }
        
        return results

# ------------------------------
# 结果可视化函数（更新为单幅图形并按本地时间保存）
# ------------------------------
def visualize_results(results: Dict[str, Any], save_dir: str = "plots") -> None:
    """
    可视化仿真结果，每张图单独展示并按本地时间保存

    Args:
        results: 仿真结果字典
        save_dir: 结果保存目录
    """
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    timestamps = results['timestamps']
    hours = list(range(len(timestamps)))  # 使用小时序号而非datetime对象避免matplotlib的日期转换问题
    
    # 1. 网格交互
    plt.figure(figsize=(12, 6))
    plt.plot(hours, results['grid_interaction_history']['draw'], 'r-', label='Grid Draw')
    plt.plot(hours, results['grid_interaction_history']['feed'], 'g-', label='Grid Feed')
    plt.title('Grid Interactions Over Time')
    plt.xlabel('Time (hours)')
    plt.ylabel('Energy (kWh)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'grid_interactions.png'))
    plt.close()
    
    # 2. 云储能能量水平
    plt.figure(figsize=(12, 6))
    plt.plot(hours, results['cloud_storage']['energy_level_history'][:-1], 'b-')  # 去掉多余的点
    plt.title('Cloud Storage Energy Level')
    plt.xlabel('Time (hours)')
    plt.ylabel('Energy (kWh)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cloud_storage_energy.png'))
    plt.close()
    
    # 3. 云储能循环深度分布
    if results['cloud_storage']['charge_discharge_cycles'] > 0:
        plt.figure(figsize=(10, 6))
        plt.hist(results['cloud_storage']['cycle_depth_history'], bins=10, alpha=0.7)
        plt.title('Cloud Storage Cycle Depth Distribution')
        plt.xlabel('Cycle Depth')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'cloud_cycle_distribution.png'))
        plt.close()
    
    # 4. 节点电池 SOC 变化
    plt.figure(figsize=(12, 6))
    for node_id, node_data in results['edge_nodes'].items():
        plt.plot(hours, node_data['soc_history'][:-1], label=f'Node {node_id}')  # 去掉多余的点
    plt.title('Edge Node Battery SOC')
    plt.xlabel('Time (hours)')
    plt.ylabel('State of Charge')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'edge_node_soc.png'))
    plt.close()
    
    # 5. 能量共享情况
    node_ids = list(results['edge_nodes'].keys())
    contributed = [results['edge_nodes'][node_id]['energy_contributed_to_sharing'] for node_id in node_ids]
    received = [results['edge_nodes'][node_id]['energy_received_from_sharing'] for node_id in node_ids]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(node_ids))
    width = 0.35
    plt.bar(x - width/2, contributed, width, label='Contributed')
    plt.bar(x + width/2, received, width, label='Received')
    plt.title('Energy Sharing by Node')
    plt.xlabel('Node ID')
    plt.ylabel('Energy (kWh)')
    plt.xticks(x, [f'Node {id}' for id in node_ids])
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'energy_sharing.png'))
    plt.close()
    
    # 6. 关键性能指标
    plt.figure(figsize=(10, 6))
    metrics = [
        results['metrics']['self_sufficiency_percentage'],
        100 - results['metrics']['self_sufficiency_percentage'],
        results['metrics']['energy_sharing_total'] / results['metrics']['total_load'] * 100,
        results['metrics']['avg_cycle_depth'] * 100  # 转换为百分比显示
    ]
    labels = ['Self-Sufficiency', 'Grid Dependency', 'Energy Sharing', 'Avg Cycle Depth']
    plt.bar(labels, metrics)
    plt.title('Key Performance Metrics')
    plt.ylabel('Percentage (%)')
    plt.ylim(0, 100)
    for i, v in enumerate(metrics):
        plt.text(i, v + 1, f'{v:.1f}%', ha='center')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'key_metrics.png'))
    plt.close()
    
    # 7. 整体能量平衡饼图
    plt.figure(figsize=(8, 8))
    energy_sources = [
        results['metrics']['total_pv_generation'] - results['grid_feed_total'],  # 本地消耗的 PV 能量
        results['grid_draw_total'],  # 网格输入能量
    ]
    labels = ['PV Consumed', 'Grid Import']
    plt.pie(energy_sources, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('Energy Sources')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'energy_sources.png'))
    plt.close()
    
    # 8. 边缘节点循环数统计
    plt.figure(figsize=(10, 6))
    cycles = [results['edge_nodes'][node_id]['charge_discharge_cycles'] for node_id in node_ids]
    plt.bar(range(len(node_ids)), cycles, color='purple')
    plt.title('Number of Charge-Discharge Cycles by Node')
    plt.xlabel('Node ID')
    plt.ylabel('Number of Cycles')
    plt.xticks(range(len(node_ids)), [f'Node {id}' for id in node_ids])
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'node_cycles.png'))
    plt.close()
    
    # 9. 循环深度分布 - 所有节点合并
    all_depths = []
    for node_id in node_ids:
        all_depths.extend(results['edge_nodes'][node_id]['cycle_depth_history'])
    
    if all_depths:  # 确保有循环数据
        plt.figure(figsize=(10, 6))
        plt.hist(all_depths, bins=10, alpha=0.7, color='green')
        plt.title('Edge Nodes Cycle Depth Distribution')
        plt.xlabel('Cycle Depth')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'edge_cycle_distribution.png'))
        plt.close()
    
    # 10. 循环时间分布热图
    all_cycle_times = []
    all_node_ids = []
    all_depths = []
    
    for node_id in node_ids:
        node_times = results['edge_nodes'][node_id]['cycle_timestamps']
        node_depths = results['edge_nodes'][node_id]['cycle_depth_history']
        if node_times:
            all_cycle_times.extend(node_times)
            all_node_ids.extend([node_id] * len(node_times))
            all_depths.extend(node_depths)
    
    if all_cycle_times:  # 确保有循环数据
        plt.figure(figsize=(12, 6))
        # 转换为小时
        hours = [ts.hour for ts in all_cycle_times]
        plt.scatter(hours, all_node_ids, c=all_depths, cmap='viridis', 
                   s=100, alpha=0.7, edgecolors='black')
        plt.colorbar(label='Cycle Depth')
        plt.title('Charge-Discharge Cycles by Time of Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Node ID')
        plt.yticks(node_ids)
        plt.xlim(0, 24)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'cycle_time_distribution.png'))
        plt.close()
    
    # 11. 循环深度随时间变化
    if all_cycle_times:
        plt.figure(figsize=(12, 6))
        for node_id in node_ids:
            times = results['edge_nodes'][node_id]['cycle_timestamps']
            depths = results['edge_nodes'][node_id]['cycle_depth_history']
            if times:
                # 使用小时序号替代时间戳，避免matplotlib日期转换问题
                time_hours = [(ts - results['timestamps'][0]).total_seconds() / 3600 for ts in times]
                plt.plot(time_hours, depths, 'o-', label=f'Node {node_id}')
        
        plt.title('Cycle Depth Over Time')
        plt.xlabel('Time (hours from simulation start)')
        plt.ylabel('Cycle Depth')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'cycle_depth_time.png'))
        plt.close()

def visualize_node_impact(results: Dict[str, Any], save_dir: str = "plots") -> None:
    """
    可视化各边缘节点对系统性能的影响，按单独图形展示并保存

    Args:
        results: 仿真结果字典
        save_dir: 结果保存目录
    """
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    node_ids = list(results['edge_nodes'].keys())
    
    # 1. 节点能量净贡献（正值表示贡献，负值表示消费）
    plt.figure(figsize=(10, 6))
    net_contributions = []
    for node_id in node_ids:
        node = results['edge_nodes'][node_id]
        net_contribution = node['energy_contributed_to_sharing'] - node['energy_received_from_sharing']
        net_contributions.append(net_contribution)
    colors = ['g' if nc >= 0 else 'r' for nc in net_contributions]
    plt.bar(range(len(node_ids)), net_contributions, color=colors)
    plt.title('Net Energy Contribution by Node')
    plt.xlabel('Node ID')
    plt.ylabel('Net Energy Contribution (kWh)')
    plt.xticks(range(len(node_ids)), [f'Node {id}' for id in node_ids])
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, axis='y')
    for i, v in enumerate(net_contributions):
        plt.text(i, v + (1 if v >= 0 else -3), f'{v:.1f}', ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'net_energy_contribution.png'))
    plt.close()
    
    # 2. 节点负载与光伏发电
    plt.figure(figsize=(10, 6))
    node_loads = []
    node_pv_gens = []
    for node_id in node_ids:
        node_loads.append(sum(results['load_profiles'][node_id]))
        node_pv_gens.append(sum(results['pv_profiles'][node_id]))
    x = np.arange(len(node_ids))
    width = 0.35
    plt.bar(x - width/2, node_loads, width, label='Load')
    plt.bar(x + width/2, node_pv_gens, width, label='PV Generation')
    plt.title('Load vs PV Generation by Node')
    plt.xlabel('Node ID')
    plt.ylabel('Energy (kWh)')
    plt.xticks(x, [f'Node {id}' for id in node_ids])
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'load_vs_pv.png'))
    plt.close()
    
    # 3. 节点自给率
    plt.figure(figsize=(10, 6))
    node_self_sufficiency = []
    for node_id in node_ids:
        load = sum(results['load_profiles'][node_id])
        pv = sum(results['pv_profiles'][node_id])
        sharing_received = results['edge_nodes'][node_id]['energy_received_from_sharing']
        pv_used_locally = min(load, pv)
        self_sufficiency = (pv_used_locally + sharing_received) / load * 100 if load > 0 else 100
        node_self_sufficiency.append(self_sufficiency)
    plt.bar(range(len(node_ids)), node_self_sufficiency, color='skyblue')
    plt.title('Self-Sufficiency by Node')
    plt.xlabel('Node ID')
    plt.ylabel('Self-Sufficiency (%)')
    plt.xticks(range(len(node_ids)), [f'Node {id}' for id in node_ids])
    plt.ylim(0, 110)
    plt.grid(True, axis='y')
    for i, v in enumerate(node_self_sufficiency):
        plt.text(i, v + 2, f'{v:.1f}%', ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'node_self_sufficiency.png'))
    plt.close()
        
    # 4. 移除节点对系统自给率的影响分析
    plt.figure(figsize=(10, 6))
    system_self_sufficiency = results['metrics']['self_sufficiency_percentage']
    node_removal_impact = []
    total_load = results['metrics']['total_load']
    for node_id in node_ids:
        node = results['edge_nodes'][node_id]
        node_load = sum(results['load_profiles'][node_id])
        net_contribution = node['energy_contributed_to_sharing'] - node['energy_received_from_sharing']
        remaining_load = total_load - node_load
        energy_from_grid = results['grid_draw_total']
        if net_contribution > 0:
            estimated_new_grid_draw = energy_from_grid + net_contribution
        else:
            estimated_new_grid_draw = max(0, energy_from_grid + net_contribution)
        new_self_sufficiency = (1 - estimated_new_grid_draw / remaining_load) * 100 if remaining_load > 0 else 0
        impact = new_self_sufficiency - system_self_sufficiency
        node_removal_impact.append(impact)
    colors = ['g' if impact >= 0 else 'r' for impact in node_removal_impact]
    plt.bar(range(len(node_ids)), node_removal_impact, color=colors)
    plt.title('Impact on System Self-Sufficiency if Node is Removed')
    plt.xlabel('Node ID')
    plt.ylabel('Change in Self-Sufficiency (%)')
    plt.xticks(range(len(node_ids)), [f'Node {id}' for id in node_ids])
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, axis='y')
    for i, v in enumerate(node_removal_impact):
        plt.text(i, v + (0.5 if v >= 0 else -1.5), f'{v:.1f}%', ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'node_removal_impact.png'))
    plt.close()
    
    # 5. 节点循环性能与电池容量关系
    plt.figure(figsize=(10, 6))
    capacities = [results['edge_nodes'][node_id]['battery_capacity'] for node_id in node_ids]
    cycles = [results['edge_nodes'][node_id]['charge_discharge_cycles'] for node_id in node_ids]
    
    # 创建散点图
    plt.scatter(capacities, cycles, c=range(len(node_ids)), cmap='viridis', s=100, alpha=0.7)
    
    # 为每个点添加标签
    for i, node_id in enumerate(node_ids):
        plt.annotate(f'Node {node_id}', (capacities[i], cycles[i]), 
                   textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.title('Battery Capacity vs Number of Cycles')
    plt.xlabel('Battery Capacity (kWh)')
    plt.ylabel('Number of Cycles')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'capacity_vs_cycles.png'))
    plt.close()
    
    # 6. 节点平均循环深度分布
    plt.figure(figsize=(10, 6))
    avg_depths = []
    for node_id in node_ids:
        depths = results['edge_nodes'][node_id]['cycle_depth_history']
        avg_depth = np.mean(depths) if depths else 0
        avg_depths.append(avg_depth)
    
    plt.bar(range(len(node_ids)), avg_depths, color='orange')
    plt.title('Average Cycle Depth by Node')
    plt.xlabel('Node ID')
    plt.ylabel('Average Cycle Depth')
    plt.xticks(range(len(node_ids)), [f'Node {id}' for id in node_ids])
    plt.grid(True, axis='y')
    for i, v in enumerate(avg_depths):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'avg_cycle_depth.png'))
    plt.close()

def analyze_node_impact(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    分析各边缘节点对整体系统性能的影响，增加超循环相关分析

    Args:
        results: 仿真结果字典
    Returns:
        node_impact: 包含各节点指标的字典
    """
    node_ids = list(results['edge_nodes'].keys())
    node_impact = {}
    
    for node_id in node_ids:
        node = results['edge_nodes'][node_id]
        node_load = sum(results['load_profiles'][node_id])
        node_pv = sum(results['pv_profiles'][node_id])
        pv_used_locally = min(node_load, node_pv)
        isolated_self_sufficiency = pv_used_locally / node_load * 100 if node_load > 0 else 100
        sharing_received = node['energy_received_from_sharing']
        connected_self_sufficiency = (pv_used_locally + sharing_received) / node_load * 100 if node_load > 0 else 100
        net_contribution = node['energy_contributed_to_sharing'] - node['energy_received_from_sharing']
        contribution_to_peak_shaving = node['energy_contributed_to_sharing']
        battery_capacity = node.get('battery_capacity', 0)
        if battery_capacity > 0:
            energy_throughput = node['energy_drawn_from_battery'] + node['energy_charged_to_battery']
            battery_utilization = energy_throughput / (battery_capacity * 2 * len(results['grid_interaction_history']['draw'])) * 100
        else:
            battery_utilization = 0
        
        # 超循环分析
        cycle_count = node['charge_discharge_cycles']
        avg_cycle_depth = np.mean(node['cycle_depth_history']) if node['cycle_depth_history'] else 0
        max_cycle_depth = max(node['cycle_depth_history']) if node['cycle_depth_history'] else 0
        
        # 计算每kWh电池容量的循环数
        cycles_per_kwh = cycle_count / battery_capacity if battery_capacity > 0 else 0
        
        # 电池容量利用率（通过循环深度）
        capacity_utilization = avg_cycle_depth * cycle_count * battery_capacity if battery_capacity > 0 else 0
            
        node_impact[node_id] = {
            'load': node_load,
            'pv_generation': node_pv,
            'isolated_self_sufficiency': isolated_self_sufficiency,
            'connected_self_sufficiency': connected_self_sufficiency,
            'self_sufficiency_improvement': connected_self_sufficiency - isolated_self_sufficiency,
            'net_contribution': net_contribution,
            'energy_contributed': node['energy_contributed_to_sharing'],
            'energy_received': node['energy_received_from_sharing'],
            'contribution_to_peak_shaving': contribution_to_peak_shaving,
            'battery_utilization': battery_utilization,
            'cycle_count': cycle_count,
            'avg_cycle_depth': avg_cycle_depth,
            'max_cycle_depth': max_cycle_depth,
            'cycles_per_kwh': cycles_per_kwh,
            'capacity_utilization': capacity_utilization
        }
    
    return node_impact

# ------------------------------
# 运行示例仿真
# ------------------------------
def run_example_simulation():
    """运行一个示例仿真，并进行节点影响分析，增加超循环分析显示"""
    num_nodes = 8
    simulation_hours = 168  # 7 天
    edge_battery_capacities = [30, 25, 40, 20, 35, 50, 60, 15]
    cloud_storage_capacity = 1000
    
    # 设置仿真开始时间为当前时间
    start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    energy_manager = EnergyManager(
        num_nodes=num_nodes,
        simulation_hours=simulation_hours,
        edge_battery_capacities=edge_battery_capacities,
        cloud_storage_capacity=cloud_storage_capacity,
        start_time=start_time
    )
    
    print("Running simulation...")
    results = energy_manager.run_simulation()
    
    print("\nSimulation Results:")
    print(f"Total Load: {results['metrics']['total_load']:.2f} kWh")
    print(f"Total PV Generation: {results['metrics']['total_pv_generation']:.2f} kWh")
    print(f"Grid Dependency: {results['metrics']['grid_dependency_percentage']:.2f}%")
    print(f"Self-Sufficiency: {results['metrics']['self_sufficiency_percentage']:.2f}%")
    print(f"Total Energy Sharing: {results['metrics']['energy_sharing_total']:.2f} kWh")
    print(f"Total Cycles: {results['metrics']['total_cycles']}")
    print(f"Average Cycle Depth: {results['metrics']['avg_cycle_depth']:.3f}")
    
    # 创建保存目录
    save_dir = f"simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 生成可视化结果
    visualize_results(results, save_dir=save_dir)
    
    node_impact = analyze_node_impact(results)
    
    print("\nNode Impact Analysis:")
    for node_id, impact in node_impact.items():
        print(f"\nNode {node_id}:")
        print(f"  Load: {impact['load']:.2f} kWh")
        print(f"  PV Generation: {impact['pv_generation']:.2f} kWh")
        print(f"  Self-Sufficiency (Isolated): {impact['isolated_self_sufficiency']:.2f}%")
        print(f"  Self-Sufficiency (Connected): {impact['connected_self_sufficiency']:.2f}%")
        print(f"  Self-Sufficiency Improvement: {impact['self_sufficiency_improvement']:.2f}%")
        print(f"  Net Energy Contribution: {impact['net_contribution']:.2f} kWh")
        print(f"  Energy Contributed to Sharing: {impact['energy_contributed']:.2f} kWh")
        print(f"  Energy Received from Sharing: {impact['energy_received']:.2f} kWh")
        print(f"  Battery Utilization: {impact['battery_utilization']:.2f}%")
        print(f"  Charge-Discharge Cycles: {impact['cycle_count']}")
        print(f"  Average Cycle Depth: {impact['avg_cycle_depth']:.3f}")
        print(f"  Maximum Cycle Depth: {impact['max_cycle_depth']:.3f}")
        print(f"  Cycles per kWh: {impact['cycles_per_kwh']:.3f}")
    
    visualize_node_impact(results, save_dir=save_dir)
    
    return results, node_impact

# ------------------------------
# 运行不同配置的比较仿真
# ------------------------------
def run_comparative_simulation():
    """运行不同边缘节点配置下的仿真，并比较其影响"""
    num_nodes = 8
    simulation_hours = 168  # 7 天
    cloud_storage_capacity = 200
    
    # 设置仿真开始时间为当前时间
    start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    configurations = {
        "Balanced": {
            "battery_capacities": [30, 30, 30, 30, 30, 30, 30, 30],
            "description": "所有节点均等电池容量"
        },
        "Leader-Follower": {
            "battery_capacities": [100, 30, 30, 30, 30, 30, 30, 30],
            "description": "一个节点具有更大电池容量"
        },
        "Progressive": {
            "battery_capacities": [10, 20, 30, 40, 50, 60, 70, 80],
            "description": "电池容量逐渐增加"
        },
        "Diverse": {
            "battery_capacities": [15, 60, 5, 40, 25, 100, 150, 30],
            "description": "电池容量差异较大"
        }
    }
    
    all_results = {}
    all_node_impacts = {}
    
    # 使用参考仿真生成负载和 PV 曲线
    ref_energy_manager = EnergyManager(
        num_nodes=num_nodes,
        simulation_hours=simulation_hours,
        edge_battery_capacities=[30, 30, 30, 30, 30, 30, 30, 30],
        cloud_storage_capacity=cloud_storage_capacity,
        start_time=start_time
    )
    load_profiles = ref_energy_manager.load_profiles
    pv_profiles = ref_energy_manager.pv_profiles
    
    print("Running comparative simulations...")
    
    for config_name, config in configurations.items():
        print(f"\nSimulating {config_name} configuration: {config['description']}")
        energy_manager = EnergyManager(
            num_nodes=num_nodes,
            simulation_hours=simulation_hours,
            edge_battery_capacities=config["battery_capacities"],
            cloud_storage_capacity=cloud_storage_capacity,
            load_profiles=load_profiles,
            pv_profiles=pv_profiles,
            start_time=start_time
        )
        
        results = energy_manager.run_simulation()
        node_impact = analyze_node_impact(results)
        
        print(f"  Self-Sufficiency: {results['metrics']['self_sufficiency_percentage']:.2f}%")
        print(f"  Grid Dependency: {results['metrics']['grid_dependency_percentage']:.2f}%")
        print(f"  Total Energy Sharing: {results['metrics']['energy_sharing_total']:.2f} kWh")
        print(f"  Total Cycles: {results['metrics']['total_cycles']}")
        print(f"  Average Cycle Depth: {results['metrics']['avg_cycle_depth']:.3f}")
        
        all_results[config_name] = results
        all_node_impacts[config_name] = node_impact
    
    # 创建保存目录
    save_dir = f"comparative_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    visualize_comparative_results(all_results, all_node_impacts, save_dir)
    
    return all_results, all_node_impacts

def visualize_comparative_results(all_results, all_node_impacts, save_dir="comparative_plots"):
    """
    可视化不同边缘节点配置下的仿真比较结果，保存为单独的图形

    Args:
        all_results: 各配置下的仿真结果字典
        all_node_impacts: 各配置下的节点影响分析字典
        save_dir: 结果保存目录
    """
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 1. 系统自给率比较
    plt.figure(figsize=(10, 6))
    config_names = list(all_results.keys())
    self_sufficiency = [results['metrics']['self_sufficiency_percentage'] for results in all_results.values()]
    bars = plt.bar(config_names, self_sufficiency, color='skyblue')
    plt.title('System Self-Sufficiency by Configuration')
    plt.xlabel('Configuration')
    plt.ylabel('Self-Sufficiency (%)')
    plt.ylim(0, 100)
    plt.grid(True, axis='y')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'self_sufficiency_comparison.png'))
    plt.close()
    
    # 2. 网格交互比较
    plt.figure(figsize=(10, 6))
    grid_draws = [results['grid_draw_total'] for results in all_results.values()]
    grid_feeds = [results['grid_feed_total'] for results in all_results.values()]
    x = np.arange(len(config_names))
    width = 0.35
    plt.bar(x - width/2, grid_draws, width, label='Grid Draw', color='r')
    plt.bar(x + width/2, grid_feeds, width, label='Grid Feed', color='g')
    plt.title('Grid Interactions by Configuration')
    plt.xlabel('Configuration')
    plt.ylabel('Energy (kWh)')
    plt.xticks(x, config_names)
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'grid_interactions_comparison.png'))
    plt.close()
    
    # 3. 能量共享总量比较
    plt.figure(figsize=(10, 6))
    energy_sharing = [results['metrics']['energy_sharing_total'] for results in all_results.values()]
    bars = plt.bar(config_names, energy_sharing, color='orange')
    plt.title('Total Energy Sharing by Configuration')
    plt.xlabel('Configuration')
    plt.ylabel('Energy Shared (kWh)')
    plt.grid(True, axis='y')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'energy_sharing_comparison.png'))
    plt.close()
    
    # 4. 超循环比较
    plt.figure(figsize=(10, 6))
    total_cycles = [results['metrics'].get('total_cycles', 0) for results in all_results.values()]
    avg_depths = [results['metrics'].get('avg_cycle_depth', 0) for results in all_results.values()]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Total Cycles', color=color)
    ax1.bar(x - width/2, total_cycles, width, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Average Cycle Depth', color=color)
    ax2.bar(x + width/2, avg_depths, width, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Cycle Performance by Configuration')
    plt.xticks(x, config_names)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cycle_performance_comparison.png'))
    plt.close()
    
    # 5. 云储能利用情况
    plt.figure(figsize=(12, 6))
    for config_name, results in all_results.items():
        cloud_energy = results['cloud_storage']['energy_level_history']
        time_hours = list(range(len(cloud_energy)))  # 使用小时序号
        plt.plot(time_hours, cloud_energy, label=config_name)
    plt.title('Cloud Storage Energy Level Over Time')
    plt.xlabel('Time (hours)')
    plt.ylabel('Energy (kWh)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cloud_storage_comparison.png'))
    plt.close()
    
    # 6. 节点自给率提升分布
    plt.figure(figsize=(10, 6))
    ss_improvement_data = []
    for config_name, node_impacts in all_node_impacts.items():
        improvements = [impact['self_sufficiency_improvement'] for impact in node_impacts.values()]
        ss_improvement_data.append(improvements)
    plt.boxplot(ss_improvement_data, labels=config_names)
    plt.title('Distribution of Node Self-Sufficiency Improvements')
    plt.xlabel('Configuration')
    plt.ylabel('Self-Sufficiency Improvement (%)')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'self_sufficiency_distribution.png'))
    plt.close()
    
    # 7. 电池利用效率
    plt.figure(figsize=(10, 6))
    avg_battery_utilization = []
    for config_name, node_impacts in all_node_impacts.items():
        util = np.mean([impact['battery_utilization'] for impact in node_impacts.values()])
        avg_battery_utilization.append(util)
    bars = plt.bar(config_names, avg_battery_utilization, color='purple')
    plt.title('Average Battery Utilization Efficiency')
    plt.xlabel('Configuration')
    plt.ylabel('Utilization (%)')
    plt.grid(True, axis='y')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'battery_utilization.png'))
    plt.close()
    
    # 8. 平均循环深度比较
    plt.figure(figsize=(10, 6))
    avg_cycle_depth_by_config = []
    for config_name, node_impacts in all_node_impacts.items():
        avg_depth = np.mean([impact['avg_cycle_depth'] for impact in node_impacts.values() if impact['cycle_count'] > 0])
        if np.isnan(avg_depth):
            avg_depth = 0
        avg_cycle_depth_by_config.append(avg_depth)
    
    bars = plt.bar(config_names, avg_cycle_depth_by_config, color='green')
    plt.title('Average Cycle Depth by Configuration')
    plt.xlabel('Configuration')
    plt.ylabel('Average Cycle Depth')
    plt.grid(True, axis='y')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'avg_cycle_depth_comparison.png'))
    plt.close()
    
    # 第二组图: 详细节点分析
    
    # 9. 各配置下每个节点的净能量贡献
    # 动态获取节点数量（取第一个配置下的节点数量）
    first_config_key = next(iter(all_node_impacts))
    num_nodes_dynamic = len(all_node_impacts[first_config_key])
    bar_width = 0.15
    index = np.arange(num_nodes_dynamic)
    
    plt.figure(figsize=(12, 6))
    for i, (config_name, node_impacts) in enumerate(all_node_impacts.items()):
        net_contributions = [impact['net_contribution'] for node_id, impact in sorted(node_impacts.items())]
        position = index + i * bar_width
        plt.bar(position, net_contributions, bar_width, label=config_name)
    plt.title('Net Energy Contribution by Node for Each Configuration')
    plt.xlabel('Node ID')
    plt.ylabel('Net Contribution (kWh)')
    plt.xticks(index + bar_width * (len(all_node_impacts) - 1) / 2, [f'Node {i+1}' for i in range(num_nodes_dynamic)])
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'net_contribution_by_node.png'))
    plt.close()
    
    # 10. 各配置下每个节点的自给率提升
    plt.figure(figsize=(12, 6))
    for i, (config_name, node_impacts) in enumerate(all_node_impacts.items()):
        improvements = [impact['self_sufficiency_improvement'] for node_id, impact in sorted(node_impacts.items())]
        position = index + i * bar_width
        plt.bar(position, improvements, bar_width, label=config_name)
    plt.title('Self-Sufficiency Improvement by Node for Each Configuration')
    plt.xlabel('Node ID')
    plt.ylabel('Improvement (%)')
    plt.xticks(index + bar_width * (len(all_node_impacts) - 1) / 2, [f'Node {i+1}' for i in range(num_nodes_dynamic)])
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'self_sufficiency_improvement_by_node.png'))
    plt.close()
    
    # 11. 各配置下能量流向堆叠图
    plt.figure(figsize=(10, 6))
    energy_data = []
    for config_name, results in all_results.items():
        local_consumption = results['metrics']['total_pv_generation'] - results['grid_feed_total']
        sharing = results['metrics']['energy_sharing_total']
        grid_import = results['grid_draw_total']
        energy_data.append([local_consumption, sharing, grid_import])
    energy_data = np.array(energy_data).T
    bottom = np.zeros(len(config_names))
    patterns = ['', '///', '\\\\\\']
    labels = ['Local PV Consumption', 'Energy Sharing', 'Grid Import']
    for i, data_label in enumerate(labels):
        plt.bar(config_names, energy_data[i], bottom=bottom, label=data_label, hatch=patterns[i])
        bottom += energy_data[i]
    plt.title('Energy Flow Breakdown by Configuration')
    plt.xlabel('Configuration')
    plt.ylabel('Energy (kWh)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'energy_flow_breakdown.png'))
    plt.close()
    
    # 12. 各配置下节点电池利用率
    plt.figure(figsize=(12, 6))
    for i, (config_name, node_impacts) in enumerate(all_node_impacts.items()):
        utilization = [impact['battery_utilization'] for node_id, impact in sorted(node_impacts.items())]
        position = index + i * bar_width
        plt.bar(position, utilization, bar_width, label=config_name)
    plt.title('Battery Utilization by Node for Each Configuration')
    plt.xlabel('Node ID')
    plt.ylabel('Utilization (%)')
    plt.xticks(index + bar_width * (len(all_node_impacts) - 1) / 2, [f'Node {i+1}' for i in range(num_nodes_dynamic)])
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'battery_utilization_by_node.png'))
    plt.close()
    
    # 13. 各配置下节点循环次数
    plt.figure(figsize=(12, 6))
    for i, (config_name, node_impacts) in enumerate(all_node_impacts.items()):
        cycle_counts = [impact['cycle_count'] for node_id, impact in sorted(node_impacts.items())]
        position = index + i * bar_width
        plt.bar(position, cycle_counts, bar_width, label=config_name)
    plt.title('Cycle Count by Node for Each Configuration')
    plt.xlabel('Node ID')
    plt.ylabel('Number of Cycles')
    plt.xticks(index + bar_width * (len(all_node_impacts) - 1) / 2, [f'Node {i+1}' for i in range(num_nodes_dynamic)])
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cycle_count_by_node.png'))
    plt.close()
    
    # 14. 能量需求与电池容量的比例
    plt.figure(figsize=(10, 6))
    for config_name, results in all_results.items():
        ratios = []
        for node_id, node in results['edge_nodes'].items():
            node_load = sum(results['load_profiles'][int(node_id)])
            battery_capacity = node['battery_capacity']
            ratio = node_load / battery_capacity if battery_capacity > 0 else 0
            ratios.append(ratio)
        plt.plot(range(1, len(ratios)+1), ratios, 'o-', label=config_name)
    
    plt.title('Load-to-Battery Capacity Ratio by Node')
    plt.xlabel('Node ID')
    plt.ylabel('Load/Capacity Ratio')
    plt.xticks(range(1, len(ratios)+1))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'load_capacity_ratio.png'))
    plt.close()
    
    # 15. 循环深度与电池容量的关系
    plt.figure(figsize=(10, 6))
    
    for config_name, results in all_results.items():
        capacities = []
        depths = []
        node_ids_text = []
        
        for node_id, node in results['edge_nodes'].items():
            capacity = node['battery_capacity']
            if node['cycle_depth_history']:
                avg_depth = np.mean(node['cycle_depth_history'])
                capacities.append(capacity)
                depths.append(avg_depth)
                node_ids_text.append(str(node_id))
                
        if capacities:  # 确保有数据
            plt.scatter(capacities, depths, label=config_name, alpha=0.7, s=80)
            
            # 添加节点ID标签
            for i, txt in enumerate(node_ids_text):
                plt.annotate(f'N{txt}', (capacities[i], depths[i]), 
                           xytext=(5, 5), textcoords='offset points')
    
    plt.title('Average Cycle Depth vs Battery Capacity')
    plt.xlabel('Battery Capacity (kWh)')
    plt.ylabel('Average Cycle Depth')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cycle_depth_vs_capacity.png'))
    plt.close()
    
    # 16. 保存配置对比概要信息
    with open(os.path.join(save_dir, 'config_summary.txt'), 'w') as f:
        f.write("Configuration Summary\n")
        f.write("=====================\n\n")
        
        # 表格头部
        f.write(f"{'Configuration':<20} {'Self-Suff(%)':<15} {'Grid Dep(%)':<15} {'Energy Shared':<15} {'Total Cycles':<15} {'Avg Depth':<15}\n")
        f.write(f"{'-'*20:<20} {'-'*15:<15} {'-'*15:<15} {'-'*15:<15} {'-'*15:<15} {'-'*15:<15}\n")
        
        # 填充数据
        for config_name, results in all_results.items():
            metrics = results['metrics']
            f.write(f"{config_name:<20} "
                  f"{metrics['self_sufficiency_percentage']:<15.2f} "
                  f"{metrics['grid_dependency_percentage']:<15.2f} "
                  f"{metrics['energy_sharing_total']:<15.2f} "
                  f"{metrics.get('total_cycles', 0):<15} "
                  f"{metrics.get('avg_cycle_depth', 0):<15.3f}\n")
        
        f.write("\n\nDetailed Node Analysis\n")
        f.write("=====================\n\n")
        
        for config_name, node_impacts in all_node_impacts.items():
            f.write(f"\n{config_name} Configuration:\n")
            f.write(f"{'Node ID':<10} {'Load':<10} {'PV Gen':<10} {'Self-Suff(%)':<15} {'Contrib':<10} {'Recv':<10} {'Cycles':<10} {'Avg Depth':<10}\n")
            f.write(f"{'-'*10:<10} {'-'*10:<10} {'-'*10:<10} {'-'*15:<15} {'-'*10:<10} {'-'*10:<10} {'-'*10:<10} {'-'*10:<10}\n")
            
            for node_id, impact in sorted(node_impacts.items()):
                f.write(f"{node_id:<10} "
                      f"{impact['load']:<10.2f} "
                      f"{impact['pv_generation']:<10.2f} "
                      f"{impact['connected_self_sufficiency']:<15.2f} "
                      f"{impact['energy_contributed']:<10.2f} "
                      f"{impact['energy_received']:<10.2f} "
                      f"{impact['cycle_count']:<10} "
                      f"{impact['avg_cycle_depth']:<10.3f}\n")
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
    
    # 使用TD3算法
    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=1.0,
        hidden_dim=128
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
    
    for episode in range(1, episodes + 1):
        # 设置探索噪声
        exploration_noise = max(0.5 * (1 - episode / episodes), 0.1)
        
        # 运行一个回合
        total_reward, metrics, trajectory = env.run_episode(agent, exploration=True)
        
        # 存储经验
        for i in range(len(trajectory['states'])):
            state = trajectory['states'][i]
            action = trajectory['actions'][i]
            reward = trajectory['rewards'][i]
            
            # 获取下一个状态和是否结束
            next_state = trajectory['states'][i+1] if i < len(trajectory['states'])-1 else None
            done = (i == len(trajectory['states'])-1)
            
            if next_state is not None:
                replay_buffer.add(state, action, reward, next_state, done)
        
        # 只有当经验池足够大时才开始训练
        if len(replay_buffer) > batch_size * 10:
            # 执行多次更新
            for _ in range(simulation_hours // 10):  # 训练步数与仿真时长相关
                critic_loss, actor_loss = agent.update(replay_buffer, batch_size)
                if critic_loss is not None:
                    critic_losses.append(critic_loss)
                if actor_loss is not None:
                    actor_losses.append(actor_loss)
        
        # 记录训练指标
        rewards_history.append(total_reward)
        self_sufficiency_history.append(metrics['self_sufficiency_percentage'])
        energy_sharing_history.append(metrics['energy_sharing_total'])
        
        # 打印训练进度
        if episode % 10 == 0 or episode == 1:
            avg_reward = np.mean(rewards_history[-10:]) if episode > 10 else total_reward
            print(f"Episode {episode}/{episodes} - 奖励: {total_reward:.2f}, 平均奖励: {avg_reward:.2f}")
            print(f"  自给率: {metrics['self_sufficiency_percentage']:.2f}%, 能量共享: {metrics['energy_sharing_total']:.2f} kWh")
            print(f"  电网购电: {metrics['total_grid_import']:.2f} kWh, 电网售电: {metrics['total_grid_export']:.2f} kWh")
            print(f"  探索噪声: {exploration_noise:.2f}")
        
        # 保存模型检查点
        if episode % save_interval == 0 or episode == episodes:
            checkpoint_dir = os.path.join(model_dir, f"checkpoint_{episode}")
            agent.save(checkpoint_dir)
            print(f"模型已保存至: {checkpoint_dir}")
        
        # 保存最佳模型
        if total_reward > best_reward:
            best_reward = total_reward
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
        
        # 因为Actor更新不如Critic频繁，需要对齐
        actor_indices = np.linspace(0, len(critic_losses)-1, len(actor_losses)).astype(int)
        aligned_actor_losses = np.zeros_like(critic_losses)
        aligned_actor_losses.fill(np.nan)
        aligned_actor_losses[actor_indices] = actor_losses
        
        plt.plot(aligned_actor_losses, label='Actor Loss')
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
        np.save(os.path.join(save_dir, f'actor_losses_{timestamp}.npy'), actor_losses)

# ==============================
# 评估与比较RL与基线策略
# ==============================

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
            hidden_dim=128
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
    if 'rl_policy' in results:
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

def plot_evaluation_results(env, trajectory, title, save_path):
    """
    绘制评估结果
    
    Args:
        env: RL环境
        trajectory: 轨迹数据
        title: 图表标题
        save_path: 保存路径
    """
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
    np.save(os.path.join(save_path, 'states.npy'), states_np)
    np.save(os.path.join(save_path, 'actions.npy'), actions_np)
    np.save(os.path.join(save_path, 'rewards.npy'), rewards)

def plot_comparison(results, save_path):
    """
    绘制RL与基线策略的比较图
    
    Args:
        results: 评估结果字典
        save_path: 保存路径
    """
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
    
    np.save(os.path.join(save_path, 'comparison_data.npy'), comparison_data)
    
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

# ==============================
# 主程序
# ==============================

def main():
    """主程序入口"""
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 系统配置
    num_nodes = 8
    simulation_hours = 168  # 7天
    edge_battery_capacities = [30, 25, 40, 20, 35, 50, 60, 15]
    cloud_storage_capacity = 200
    
    # 训练参数
    train_episodes = 1000
    batch_size = 64
    replay_buffer_size = 100000
    
    # 目录设置
    base_dir = f"rl_energy_manager_{datetime.now().strftime('%Y%m%d_%H%M')}"
    model_dir = os.path.join(base_dir, "models")
    results_dir = os.path.join(base_dir, "results")
    comparison_dir = os.path.join(base_dir, "comparison")
    
    os.makedirs(base_dir, exist_ok=True)
    
    # 打印系统信息
    print("====== 强化学习云边协同能源管理系统 ======")
    print(f"节点数量: {num_nodes}")
    print(f"仿真时长: {simulation_hours}小时")
    print(f"边缘电池容量: {edge_battery_capacities}")
    print(f"云储能容量: {cloud_storage_capacity} kWh")
    print("=======================================")
    
    # 运行训练
    print("\n开始训练...")
    agent, _ = train_rl_energy_manager(
        num_nodes=num_nodes,
        simulation_hours=simulation_hours,
        edge_battery_capacities=edge_battery_capacities,
        cloud_storage_capacity=cloud_storage_capacity,
        episodes=train_episodes,
        batch_size=batch_size,
        replay_buffer_size=replay_buffer_size,
        model_dir=model_dir,
        results_dir=results_dir
    )
    
    # 评估和比较
    print("\n开始评估和比较...")
    comparison_results = evaluate_and_compare(
        num_nodes=num_nodes,
        simulation_hours=simulation_hours,
        edge_battery_capacities=edge_battery_capacities,
        cloud_storage_capacity=cloud_storage_capacity,
        agent=agent,
        save_dir=comparison_dir
    )
    
    # 保存最终结果概述
    with open(os.path.join(base_dir, "summary.txt"), 'w') as f:
        f.write("强化学习云边协同能源管理系统 - 结果概述\n")
        f.write("=====================================\n\n")
        
        f.write(f"系统配置:\n")
        f.write(f"  节点数量: {num_nodes}\n")
        f.write(f"  仿真时长: {simulation_hours}小时\n")
        f.write(f"  边缘电池容量: {edge_battery_capacities}\n")
        f.write(f"  云储能容量: {cloud_storage_capacity} kWh\n\n")
        
        f.write(f"训练信息:\n")
        f.write(f"  训练周期: {train_episodes}\n")
        f.write(f"  批次大小: {batch_size}\n")
        f.write(f"  经验回放大小: {replay_buffer_size}\n\n")
        
        if 'performance_diff' in comparison_results:
            diff = comparison_results['performance_diff']
            f.write(f"性能改进:\n")
            f.write(f"  自给率: {diff['self_sufficiency']:.2f}%\n")
            f.write(f"  能量共享: {diff['energy_sharing']:.2f} kWh\n")
            f.write(f"  电网购电: {diff['grid_import']:.2f} kWh\n")
            f.write(f"  电网售电: {diff['grid_export']:.2f} kWh\n")
    
    print(f"\n所有结果已保存至: {base_dir}")
    print("强化学习云边协同能源管理系统运行完成!")

# ==============================
# 集成到原有能量管理器的接口
# ==============================

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
    
    # 创建增强的_process_timestep方法
    original_process_timestep = energy_manager._process_timestep
    
    def enhanced_process_timestep(self):
        """RL增强的时间步处理方法"""
        t = self.current_time
        
        # 构建当前状态
        hour_norm = t / self.simulation_hours
        edge_socs = [node.soc for node in self.edge_nodes]
        cloud_soc = self.cloud_storage.energy_stored / self.cloud_storage.capacity if self.cloud_storage.capacity > 0 else 0
        
        # 获取各节点当前净负载
        net_loads = []
        for node in self.edge_nodes:
            node_id = node.id
            load = self.load_profiles[node_id][t]
            pv = self.pv_profiles[node_id][t]
            net_load = (load - pv) / max(node.battery_capacity, 1.0)  # 归一化
            net_loads.append(net_load)
        
        # 组合状态
        state = np.array([hour_norm] + edge_socs + [cloud_soc] + net_loads, dtype=np.float32)
        
        # 使用RL代理预测动作
        with torch.no_grad():
            action = agent.select_action(state, add_noise=False)
        
        # 应用RL决策到能量管理系统
        _apply_rl_actions_to_energy_manager(self, action, t)
        
        # 调用原始处理逻辑
        original_process_timestep(self)
    
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
            
            # 调整节点倾向性 - 实际实现取决于能量管理系统设计
            # 可以影响_process_timestep中的决策逻辑
            
            # 示例: 调整节点的能量共享意愿或本地储能优先级
            # 这里只是设置一个属性，实际使用需要在能量管理器中添加相应逻辑
            node.rl_sharing_bias = action
    
    # 影响云端行为 - 调整云储能和电网交互策略
    # 例如调整云储能的充放电阈值
    energy_manager.cloud_storage.rl_grid_bias = cloud_action

# 如果直接运行脚本则执行main()
if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
from datetime import datetime
import os

# ==============================
# 强化学习相关类定义
# ==============================

# 定义经验回放存储
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity):
        """初始化缓冲区
        
        Args:
            capacity: 缓冲区最大容量
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """添加经验到缓冲区"""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """随机采样一批经验
        
        Returns:
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones
        """
        experiences = random.sample(self.buffer, batch_size)
        
        # 将经验拆分为对应的分量
        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.FloatTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.FloatTensor([e.done for e in experiences])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """返回缓冲区当前大小"""
        return len(self.buffer)

class Actor(nn.Module):
    """Actor网络：决定在给定状态下采取什么行动（策略）"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=64, max_action=1.0):
        """初始化Actor网络
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_dim: 隐藏层维度
            max_action: 动作幅度上限
        """
        super(Actor, self).__init__()
        self.max_action = max_action
        
        self.layer1 = nn.Linear(state_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        """前向传播
        
        Args:
            state: 当前状态
        
        Returns:
            action: 模型生成的动作
        """
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        action = torch.tanh(self.layer3(x)) * self.max_action
        return action

class Critic(nn.Module):
    """Critic网络：评估在给定状态下采取特定动作的价值（Q值）"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        """初始化Critic网络
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_dim: 隐藏层维度
        """
        super(Critic, self).__init__()
        
        # Q1 架构
        self.layer1_1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.layer2_1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3_1 = nn.Linear(hidden_dim, 1)
        
        # Q2 架构 (双Q学习可以减少过估计问题)
        self.layer1_2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.layer2_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3_2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, action):
        """前向传播，计算两个Q值
        
        Args:
            state: 当前状态
            action: 当前动作
            
        Returns:
            q1_value, q2_value: 两个网络的Q值
        """
        sa = torch.cat([state, action], 1)
        
        # Q1计算
        q1 = F.relu(self.layer1_1(sa))
        q1 = F.relu(self.layer2_1(q1))
        q1 = self.layer3_1(q1)
        
        # Q2计算
        q2 = F.relu(self.layer1_2(sa))
        q2 = F.relu(self.layer2_2(q2))
        q2 = self.layer3_2(q2)
        
        return q1, q2
    
    def Q1(self, state, action):
        """只计算Q1值，用于策略更新
        
        Returns:
            q1_value: 第一个网络的Q值
        """
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.layer1_1(sa))
        q1 = F.relu(self.layer2_1(q1))
        q1 = self.layer3_1(q1)
        
        return q1

class TD3Agent:
    """Twin Delayed DDPG (TD3) 代理实现"""
    
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action=1.0,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        actor_lr=3e-4,
        critic_lr=3e-4,
        hidden_dim=64,
        device="cpu"
    ):
        """初始化TD3代理
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            max_action: 动作幅度上限
            discount: 折扣因子gamma
            tau: 目标网络软更新系数
            policy_noise: 目标策略平滑的噪声标准差
            noise_clip: 目标策略平滑噪声裁剪范围
            policy_freq: 策略网络更新频率
            actor_lr: Actor学习率
            critic_lr: Critic学习率
            hidden_dim: 隐藏层维度
            device: 使用的设备(cpu/cuda)
        """
        self.device = device
        
        # 初始化Actor网络
        self.actor = Actor(state_dim, action_dim, hidden_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # 初始化Critic网络
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # 设置超参数
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        
        self.total_it = 0  # 更新次数计数
    
    def select_action(self, state, add_noise=True, noise_scale=0.1):
        """根据当前状态选择动作
        
        Args:
            state: 当前状态
            add_noise: 是否添加探索噪声
            noise_scale: 噪声比例
            
        Returns:
            action: 选择的动作
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            action = self.actor(state_tensor).cpu().numpy()
            
            # 添加探索噪声
            if add_noise:
                noise = np.random.normal(0, noise_scale, size=action.shape)
                action = action + noise
                action = np.clip(action, -self.max_action, self.max_action)
                
        return action
    
    def update(self, replay_buffer, batch_size=256):
        """更新神经网络
        
        Args:
            replay_buffer: 经验回放缓冲区
            batch_size: 批次大小
            
        Returns:
            critic_loss: Critic损失
            actor_loss: Actor损失（如果更新了Actor则返回，否则返回None）
        """
        self.total_it += 1
        
        # 从回放缓冲区采样
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # 计算目标Q值
        with torch.no_grad():
            # 使用目标Actor选择下一个动作
            next_actions = self.actor_target(next_states)
            
            # 添加目标策略平滑噪声
            noise = torch.randn_like(next_actions) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_actions = torch.clamp(next_actions + noise, -self.max_action, self.max_action)
            
            # 使用目标Critic计算目标Q值
            target_Q1, target_Q2 = self.critic_target(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)  # 取较小值减轻过估计
            target_Q = rewards + (1 - dones) * self.discount * target_Q
        
        # 计算当前Q值
        current_Q1, current_Q2 = self.critic(states, actions)
        
        # 计算Critic损失
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 延迟策略更新
        actor_loss = None
        if self.total_it % self.policy_freq == 0:
            # 计算Actor损失 (最大化Q值)
            actor_loss = -self.critic.Q1(states, self.actor(states)).mean()
            
            # 更新Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 软更新目标网络
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return critic_loss.item(), actor_loss.item() if actor_loss is not None else None
    
    def save(self, directory):
        """保存模型"""
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        torch.save(self.actor.state_dict(), os.path.join(directory, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(directory, "critic.pth"))
        
    def load(self, directory):
        """加载模型"""
        self.actor.load_state_dict(torch.load(os.path.join(directory, "actor.pth")))
        self.critic.load_state_dict(torch.load(os.path.join(directory, "critic.pth")))

# ==============================
# 基于强化学习的能量管理环境
# ==============================

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
        # 状态: [当前小时, 各节点SOC, 云储能SOC, 各节点净负载]
        self.state_dim = 1 + num_nodes + 1 + num_nodes
        
        # 动作: [各节点与云端交互比例, 云端与电网交互比例]
        # 正值表示充电/购电，负值表示放电/售电
        self.action_dim = num_nodes + 1
        
        # 记录每步的性能指标
        self.step_rewards = []
        self.grid_imports = []
        self.grid_exports = []
        self.energy_sharing = []
        self.shortfalls = []
        self.action_history = []
        
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
        
        # 预测下一小时的净负载 (正值为缺口，负值为过剩)
        net_loads = []
        for node in self.energy_manager.edge_nodes:
            node_id = node.id
            if self.current_hour < self.simulation_hours - 1:
                load = self.energy_manager.load_profiles[node_id][self.current_hour]
                pv = self.energy_manager.pv_profiles[node_id][self.current_hour]
            else:
                # 对于最后一小时，使用前一小时的数据
                load = self.energy_manager.load_profiles[node_id][self.current_hour-1]
                pv = self.energy_manager.pv_profiles[node_id][self.current_hour-1]
            
            # 归一化净负载
            net_load = (load - pv) / max(node.battery_capacity, 1.0)
            net_loads.append(net_load)
            
        # 组合状态
        state = [hour_norm] + edge_socs + [cloud_soc] + net_loads
        
        return np.array(state, dtype=np.float32)
    
    def process_rl_action(self, action):
        """
        处理强化学习代理生成的动作
        
        Args:
            action: 代理生成的动作向量 [-1, 1]范围内
            
        Returns:
            edge_commands: 各节点的控制命令
            cloud_command: 云端的控制命令
        """
        edge_commands = []
        
        # 前num_nodes个元素对应边缘节点控制
        for i in range(self.num_nodes):
            edge_commands.append(action[i])
            
        # 最后一个元素对应云端控制
        cloud_command = action[-1]
        
        return edge_commands, cloud_command
    
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
        self.action_history.append(action.copy())
        
        # 处理RL动作
        edge_commands, cloud_command = self.process_rl_action(action)
        
        # 将RL策略转换为能量管理系统的控制决策
        self._apply_rl_decisions(edge_commands, cloud_command)
        
        # 执行一步仿真
        t = self.current_hour
        self.energy_manager._process_timestep()
        
        # 计算奖励
        grid_draw = self.energy_manager.grid_draw_history[t]
        grid_feed = self.energy_manager.grid_feed_history[t]
        
        # 计算能量共享量
        energy_shared = sum(node.energy_received_from_sharing for node in self.energy_manager.edge_nodes)
        
        # 计算负载短缺量
        shortfall = sum(node.shortfall_history[-1] for node in self.energy_manager.edge_nodes)
        
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
        
        # 获取下一个状态
        next_state = self.get_state()
        
        info = {
            'grid_draw': grid_draw,
            'grid_feed': grid_feed,
            'energy_shared': energy_shared,
            'shortfall': shortfall
        }
        
        return next_state, reward, self.done, info
    
    def _apply_rl_decisions(self, edge_commands, cloud_command):
        """
        应用RL决策到能量管理系统
        
        Args:
            edge_commands: 边缘节点控制命令
            cloud_command: 云端控制命令
        """
        # 这里添加自定义逻辑来影响EnergyManager的行为
        # 例如，可以调整节点之间优先级以及与云储能交互的策略
        
        # 此处只是一个示例，实际实现会基于系统设计调整
        # 例如可以设置云储能的充放电阈值，或节点间的能量交换优先级
        t = self.current_hour
        for i, node in enumerate(self.energy_manager.edge_nodes):
            # 命令范围在[-1, 1]，这里转换为节点策略
            # 正值表示倾向于充电/保存能量，负值表示倾向于放电/共享能量
            # 这可以通过影响节点内平衡逻辑或能量共享逻辑来实现
            
            # 这里是一个简化示例，实际实现可能更复杂
            node_id = node.id
            if t < self.simulation_hours:
                load = self.energy_manager.load_profiles[node_id][t]
                pv = self.energy_manager.pv_profiles[node_id][t]
                net_load = load - pv
                
                # 基于RL命令调整节点行为
                # 例如，设置节点愿意共享的能量比例
                # 或者调整节点优先考虑自用还是共享的策略
                # 实际实现会依赖于系统设计
                
                # 在此示例中，我们不做实际修改，只是记录命令以便后续实现
                pass
                
        # 类似地，处理云端命令
        # 正值可能表示更积极从电网购电，负值表示更积极向电网售电
        # 实际实现依赖于系统设计
        pass
    
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
        # 奖励权重
        w_grid_draw = -1.0  # 减少从电网购电
        w_grid_feed = -0.2  # 适度减少向电网售电 (损失能量)
        w_energy_shared = 2.0  # 鼓励节点间能量共享
        w_shortfall = -5.0  # 严重惩罚负载缺口
        
        # 归一化各项指标
        # 假设的参考值，实际应根据系统规模调整
        norm_grid_draw = grid_draw / (sum(node.battery_capacity for node in self.energy_manager.edge_nodes) / 10)
        norm_grid_feed = grid_feed / (sum(node.battery_capacity for node in self.energy_manager.edge_nodes) / 10)
        norm_energy_shared = energy_shared / (sum(node.battery_capacity for node in self.energy_manager.edge_nodes) / 10)
        norm_shortfall = shortfall / (sum(node.battery_capacity for node in self.energy_manager.edge_nodes) / 20)
        
        # 计算综合奖励
        reward = (w_grid_draw * norm_grid_draw + 
                  w_grid_feed * norm_grid_feed + 
                  w_energy_shared * norm_energy_shared + 
                  w_shortfall * norm_shortfall)
        
        return reward
    
    def reset(self):
        """
        重置环境到初始状态
        
        Returns:
            state: 初始状态
        """
        # 重置EnergyManager
        self.energy_manager = EnergyManager(
            num_nodes=self.num_nodes,
            simulation_hours=self.simulation_hours,
            edge_battery_capacities=[node.battery_capacity for node in self.energy_manager.edge_nodes],
            cloud_storage_capacity=self.energy_manager.cloud_storage.capacity,
            load_profiles=self.energy_manager.load_profiles,
            pv_profiles=self.energy_manager.pv_profiles,
            start_time=self.energy_manager.start_time
        )
        
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
        
        # 返回初始状态
        return self.get_state()
    
    def render(self, mode='human'):
        """
        渲染当前环境状态，用于可视化
        """
        # 这里可以实现可视化逻辑
        pass
    
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
            action = agent.select_action(state, add_noise=exploration)
            
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

# ==============================
# 强化学习训练主函数
# ==============================

def train_rl_energy_manager(
    num_nodes=8,
    simulation_hours=168,  # 7天
    edge_battery_capacities=None,
    cloud_storage_capacity=200,
    episodes=1000,
    batch_size=256,
    replay_buffer_size=1000000,
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
    env = RLEnergy
# ------------------------------
# 主程序入口
# ------------------------------
if __name__ == "__main__":
    # 运行单一示例仿真
    results, node_impact = run_example_simulation()
    
    # 运行不同配置下的比较仿真
    # all_results, all_node_impacts = run_comparative_simulation()