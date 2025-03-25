import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from models import Actor, Critic

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
        hidden_dim=128,  # 增加隐藏层维度
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
    # 确保状态是适当形状的张量
    if isinstance(state, np.ndarray):
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
    else:
        state_tensor = torch.FloatTensor(state).reshape(1, -1).to(self.device)
        
    # 设置为评估模式以避免BatchNorm/Dropout问题
    self.actor.eval()
        
    with torch.no_grad():
        action = self.actor(state_tensor).cpu().data.numpy().flatten()
    
    # 恢复到训练模式
    self.actor.train()
        
    # 添加探索噪声
    if add_noise:
        # 使用OU噪声代替高斯噪声以获得更好的探索性能
        noise = self._generate_ou_noise(action.shape, noise_scale)
        action = action + noise
        action = np.clip(action, -self.max_action, self.max_action)
            
    return action

    # 添加OU噪声生成方法
def _generate_ou_noise(self, shape, scale=0.1, theta=0.15, dt=1e-2):
    """生成Ornstein-Uhlenbeck噪声，比简单的高斯噪声更适合连续控制
    
    Args:
        shape: 噪声形状
        scale: 噪声幅度
        theta: 均值回归速率
        dt: 时间步长
        
    Returns:
        noise: OU噪声
    """
    if not hasattr(self, 'noise'):
        self.noise = np.zeros(shape)
    
    x = self.noise
    dx = theta * (0 - x) * dt + scale * np.random.randn(*shape) * np.sqrt(dt)
    self.noise = x + dx
    
    return self.noise
    
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
        rewards = rewards.unsqueeze(1).to(self.device)  # 修改为 [batch_size, 1]
        next_states = next_states.to(self.device)
        dones = dones.unsqueeze(1).to(self.device)  # 修改为 [batch_size, 1]
        
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
        
        # 计算Critic损失 - 修改为确保维度匹配
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=2.0)  # 梯度裁剪
        self.critic_optimizer.step()
        
        # 延迟策略更新
        actor_loss = None
        if self.total_it % self.policy_freq == 0:
            # 计算Actor损失 (最大化Q值)
            actor_loss = -self.critic.Q1(states, self.actor(states)).mean()
            
            # 更新Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0)  # 梯度裁剪
            self.actor_optimizer.step()
            
            # 软更新目标网络
            with torch.no_grad():
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                    
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # 返回损失值，用于监控训练进度
        critic_loss_item = critic_loss.item()
        actor_loss_item = actor_loss.item() if actor_loss is not None else None
        
        return critic_loss_item, actor_loss_item
    
    def save(self, directory):
        """保存模型"""
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        torch.save(self.actor.state_dict(), os.path.join(directory, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(directory, "critic.pth"))
        
    def load(self, directory):
        """加载模型"""
        actor_path = os.path.join(directory, "actor.pth")
        critic_path = os.path.join(directory, "critic.pth")
        
        if os.path.exists(actor_path):
            self.actor.load_state_dict(torch.load(actor_path))
            self.actor_target.load_state_dict(self.actor.state_dict())
            
        if os.path.exists(critic_path):
            self.critic.load_state_dict(torch.load(critic_path))
            self.critic_target.load_state_dict(self.critic.state_dict())