import torch
import random
import numpy as np
from collections import deque, namedtuple

# 定义经验回放存储的数据结构
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
        # 确保所有数据是numpy数组格式
        state = np.array(state, dtype=np.float32)
        action = np.array(action, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """随机采样一批经验
        
        Returns:
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones
        """
        experiences = random.sample(self.buffer, batch_size)
        
        # 预先创建numpy数组然后转换为张量，避免警告
        states = np.vstack([e.state for e in experiences])
        actions = np.vstack([e.action.reshape(1, -1) for e in experiences])
        rewards = np.vstack([np.array([e.reward], dtype=np.float32) for e in experiences])
        next_states = np.vstack([e.next_state for e in experiences])
        dones = np.vstack([np.array([float(e.done)], dtype=np.float32) for e in experiences])
        
        # 转换为张量
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """返回缓冲区当前大小"""
        return len(self.buffer)
