import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """Actor网络：决定在给定状态下采取什么行动（策略）"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128, max_action=1.0):
        """初始化Actor网络
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_dim: 隐藏层维度
            max_action: 动作幅度上限
        """
        super(Actor, self).__init__()
        self.max_action = max_action
        
        # 使用更深的网络结构
        self.layer1 = nn.Linear(state_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.layer4 = nn.Linear(hidden_dim // 2, action_dim)
        
        # 添加批归一化层以稳定训练
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm3 = nn.BatchNorm1d(hidden_dim // 2)
        
        # 初始化权重以稳定训练
        self._init_weights()
        
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用改良的正交初始化
                nn.init.orthogonal_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, state):
        """前向传播
        
        Args:
            state: 当前状态
        
        Returns:
            action: 模型生成的动作
        """
        # 对单个样本的批归一化处理
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        x = F.relu(self.batch_norm1(self.layer1(state)))
        x = F.relu(self.batch_norm2(self.layer2(x)))
        x = F.relu(self.batch_norm3(self.layer3(x)))
        action = torch.tanh(self.layer4(x)) * self.max_action
        return action

class Critic(nn.Module):
    """Critic网络：评估在给定状态下采取特定动作的价值（Q值）"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """初始化Critic网络
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_dim: 隐藏层维度
        """
        super(Critic, self).__init__()
        
        # Q1 架构 - 使用更深的网络
        self.q1_layer1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_layer3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.q1_layer4 = nn.Linear(hidden_dim // 2, 1)
        
        # 批归一化层
        self.q1_batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.q1_batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.q1_batch_norm3 = nn.BatchNorm1d(hidden_dim // 2)
        
        # Q2 架构 (双Q学习可以减少过估计问题)
        self.q2_layer1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_layer3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.q2_layer4 = nn.Linear(hidden_dim // 2, 1)
        
        # 批归一化层
        self.q2_batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.q2_batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.q2_batch_norm3 = nn.BatchNorm1d(hidden_dim // 2)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state, action):
        """前向传播，计算两个Q值
        
        Args:
            state: 当前状态
            action: 当前动作
            
        Returns:
            q1_value, q2_value: 两个网络的Q值
        """
        # 处理输入维度
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
            
        sa = torch.cat([state, action], 1)
        
        # Q1计算
        q1 = F.relu(self.q1_batch_norm1(self.q1_layer1(sa)))
        q1 = F.relu(self.q1_batch_norm2(self.q1_layer2(q1)))
        q1 = F.relu(self.q1_batch_norm3(self.q1_layer3(q1)))
        q1 = self.q1_layer4(q1)
        
        # Q2计算
        q2 = F.relu(self.q2_batch_norm1(self.q2_layer1(sa)))
        q2 = F.relu(self.q2_batch_norm2(self.q2_layer2(q2)))
        q2 = F.relu(self.q2_batch_norm3(self.q2_layer3(q2)))
        q2 = self.q2_layer4(q2)
        
        return q1, q2
    
    def Q1(self, state, action):
        """只计算Q1值，用于策略更新
        
        Args:
            state: 当前状态
            action: 当前动作
            
        Returns:
            q1_value: 第一个网络的Q值
        """
        # 处理输入维度
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
            
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.q1_batch_norm1(self.q1_layer1(sa)))
        q1 = F.relu(self.q1_batch_norm2(self.q1_layer2(q1)))
        q1 = F.relu(self.q1_batch_norm3(self.q1_layer3(q1)))
        q1 = self.q1_layer4(q1)
        
        return q1