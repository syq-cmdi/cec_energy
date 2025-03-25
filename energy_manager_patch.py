"""
本文件用于为test.py中的EnergyManager添加RL支持
该文件应与test.py放在同一目录下
"""

import sys
import os
import types

# 获取当前文件目录
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 导入原始的EnergyManager类
from test import EdgeNode, CloudStorage, EnergyManager

def patch_energy_manager():
    """给EnergyManager打补丁以支持RL决策"""
    
    # 保存原始的_process_timestep方法
    original_process_timestep = EnergyManager._process_timestep
    
    # 创建增强的_process_timestep方法
    def enhanced_process_timestep(self):
        """RL增强的能量处理方法"""
        t = self.current_time
        
        # 获取各节点当前时刻的负载和光伏数据
        node_net_loads = {}
        for node in self.edge_nodes:
            node_id = node.id
            load = self.load_profiles[node_id][t]
            pv = self.pv_profiles[node_id][t]
            net_load = load - pv  # 正值表示缺口，负值表示盈余
            node_net_loads[node_id] = net_load
        
        # 检查节点是否有RL决策修改器
        for node in self.edge_nodes:
            if hasattr(node, 'rl_discharge_modifier'):
                # 修改器会影响节点放电行为
                discharge_mod = node.rl_discharge_modifier
                
                # 存储到node对象，以便在local_balance方法中使用
                node.discharge_threshold_modifier = discharge_mod
        
        # 检查云储能是否有RL决策修改器
        if hasattr(self.cloud_storage, 'rl_action'):
            cloud_action = self.cloud_storage.rl_action
            # 作为示例，云储能修改器可以改变充放电偏好
            self.cloud_storage.charge_preference = 0.5 + 0.5 * cloud_action  # 映射到0-1
        
        # 调用原始_process_timestep逻辑
        original_process_timestep(self)
    
    # 修改EdgeNode的local_balance方法以支持RL决策
    original_local_balance = EdgeNode.local_balance
    
    def enhanced_local_balance(self, net_load, timestamp=None):
        """增强的local_balance方法，考虑RL的决策修改"""
        # 考虑RL决策影响
        if hasattr(self, 'discharge_threshold_modifier') and net_load > 0:
            # 如果节点有RL修改器，调整放电行为
            discharge_mod = self.discharge_threshold_modifier
            
            # 降低discharge_mod会鼓励更多放电
            if self.soc < 0.2 + 0.3 * discharge_mod:
                # SOC过低时保留能量
                # discharge_mod越高，保留电量的阈值越高
                return net_load
            
            # 否则正常放电，但会根据RL动作调整放电量
            # discharge_mod越低，放电比例越大
            if net_load > 0 and self.soc > 0.1:
                discharge_ratio = 1.0 - 0.5 * discharge_mod  # 映射到0.5-1.0
                available_energy = self.energy_stored * self.battery_efficiency
                max_discharge = min(available_energy, net_load * discharge_ratio)
                
                # 更新电池能量
                self.energy_stored -= max_discharge / self.battery_efficiency
                self.energy_drawn_from_battery += max_discharge
                remaining_net_load = net_load - max_discharge

                # 记录短缺情况
                self.shortfall_history.append(remaining_net_load if remaining_net_load > 0 else 0)
                self.surplus_history.append(0)
                
                # 更新SOC
                self.soc = self.energy_stored / self.battery_capacity
                self.soc_history.append(self.soc)
                
                return remaining_net_load
        
        # 如果没有RL修改器或不需要修改，调用原始方法
        return original_local_balance(self, net_load, timestamp)
    
    # 修改CloudStorage的charge和discharge方法
    original_cloud_charge = CloudStorage.charge
    original_cloud_discharge = CloudStorage.discharge
    
    def enhanced_cloud_charge(self, energy_amount, timestamp=None):
        """增强的charge方法，考虑RL的决策修改"""
        # 考虑RL决策影响
        if hasattr(self, 'charge_preference'):
            # charge_preference越高，接受越多的能量
            charge_ratio = self.charge_preference  # 0-1范围
            modified_energy = energy_amount * charge_ratio
            
            # 调用原始方法，保持timestamp参数
            return original_cloud_charge(self, modified_energy, timestamp)
        else:
            # 无修改，调用原始方法
            return original_cloud_charge(self, energy_amount, timestamp)
    
    def enhanced_cloud_discharge(self, energy_request, timestamp=None):
        """增强的discharge方法，考虑RL的决策修改"""
        # 考虑RL决策影响
        if hasattr(self, 'charge_preference'):
            # charge_preference越低，愿意放出越多的能量
            discharge_ratio = 1.0 - self.charge_preference  # 映射到1-0范围
            modified_request = energy_request * (0.5 + 0.5 * discharge_ratio)  # 至少放出一半
            
            # 调用原始方法，保持timestamp参数
            return original_cloud_discharge(self, modified_request, timestamp)
        else:
            # 无修改，调用原始方法
            return original_cloud_discharge(self, energy_request, timestamp)
    
    # 应用补丁到类方法
    EnergyManager._process_timestep = enhanced_process_timestep
    EdgeNode.local_balance = enhanced_local_balance
    CloudStorage.charge = enhanced_cloud_charge
    CloudStorage.discharge = enhanced_cloud_discharge
    
    print("Energy Manager 已成功打补丁，增加了强化学习支持")
    
    return EnergyManager, EdgeNode, CloudStorage

# 当直接运行此脚本时，应用补丁
if __name__ == "__main__":
    patch_energy_manager()
