"""
简单的模型测试脚本，用于验证模型能否处理单个样本
"""

import torch
import numpy as np
from models import Actor, Critic

def test_actor_model():
    """测试Actor模型能否处理单个样本"""
    print("测试Actor模型...")
    
    # 创建一个测试状态
    state_dim = 18  # 与实际环境匹配
    action_dim = 9  # 与实际环境匹配
    
    # 初始化模型
    actor = Actor(state_dim, action_dim)
    
    # 测试训练模式
    actor.train()
    
    # 测试单个样本（一维）
    state1 = torch.rand(state_dim)
    print(f"测试单个一维样本，形状: {state1.shape}")
    try:
        action1 = actor(state1)
        print(f"  成功! 输出动作形状: {action1.shape}")
    except Exception as e:
        print(f"  失败: {e}")
    
    # 测试单个样本（二维，带批次维度）
    state2 = torch.rand(1, state_dim)
    print(f"测试单个二维样本，形状: {state2.shape}")
    try:
        action2 = actor(state2)
        print(f"  成功! 输出动作形状: {action2.shape}")
    except Exception as e:
        print(f"  失败: {e}")
    
    # 测试小批量
    state3 = torch.rand(5, state_dim)
    print(f"测试小批量，形状: {state3.shape}")
    try:
        action3 = actor(state3)
        print(f"  成功! 输出动作形状: {action3.shape}")
    except Exception as e:
        print(f"  失败: {e}")
        
    # 测试评估模式
    actor.eval()
    print("测试评估模式下的行为...")
    try:
        action4 = actor(state1)
        print(f"  成功! 评估模式下单样本输出形状: {action4.shape}")
    except Exception as e:
        print(f"  失败: {e}")
    
    print("Actor测试完成!\n")

def test_critic_model():
    """测试Critic模型能否处理单个样本"""
    print("测试Critic模型...")
    
    # 创建测试数据
    state_dim = 18  # 与实际环境匹配
    action_dim = 9  # 与实际环境匹配
    
    # 初始化模型
    critic = Critic(state_dim, action_dim)
    
    # 测试训练模式
    critic.train()
    
    # 测试单个样本
    state = torch.rand(state_dim)
    action = torch.rand(action_dim)
    print(f"测试单个样本，状态形状: {state.shape}, 动作形状: {action.shape}")
    try:
        q1, q2 = critic(state, action)
        print(f"  成功! Q值形状: {q1.shape}, {q2.shape}")
    except Exception as e:
        print(f"  失败: {e}")
    
    # 测试小批量
    state_batch = torch.rand(5, state_dim)
    action_batch = torch.rand(5, action_dim)
    print(f"测试小批量，状态形状: {state_batch.shape}, 动作形状: {action_batch.shape}")
    try:
        q1, q2 = critic(state_batch, action_batch)
        print(f"  成功! Q值形状: {q1.shape}, {q2.shape}")
    except Exception as e:
        print(f"  失败: {e}")
    
    # 测试Q1函数
    print("测试Q1函数...")
    try:
        q1 = critic.Q1(state, action)
        print(f"  成功! Q1形状: {q1.shape}")
    except Exception as e:
        print(f"  失败: {e}")
    
    print("Critic测试完成!")

if __name__ == "__main__":
    test_actor_model()
    test_critic_model()