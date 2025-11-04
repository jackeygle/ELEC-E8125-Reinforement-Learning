""" This file define some components for DDPG algorithm, including:
    - Actor network (Policy)
    - Critic network
    - Replay buffer

The student should complete the code in the middle for these components.

Hint: refer to the course exercises."""

from collections import namedtuple, defaultdict
import pickle, os, random
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal, Independent

Batch = namedtuple('Batch', ['state', 'action', 'next_state', 'reward', 'not_done', 'extra'])


class Policy(nn.Module):
    """Actor网络 - 输出确定性动作"""
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action
        
        # 定义神经网络结构
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        
    def forward(self, state):
        # 前向传播
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # tanh将输出限制在[-1, 1]
        
        # 缩放到实际动作范围
        return x * self.max_action


class Critic(nn.Module):
    """Critic网络 - Q(s,a)值函数"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        # 定义神经网络结构
        # 将state和action连接后输入网络
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        
    def forward(self, state, action):
        # 将state和action连接
        x = torch.cat([state, action], dim=1)
        
        # 前向传播
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        
        return q_value


class ReplayBuffer(object):
    """经验回放缓冲区"""
    def __init__(self, state_shape: tuple, action_dim: int, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0  # 当前指针位置
        self.size = 0  # 当前缓冲区大小
        
        # 初始化缓冲区数组
        self.state = np.zeros((max_size, *state_shape), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, *state_shape), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((max_size, 1), dtype=np.float32)
        
    def add(self, state, action, next_state, reward, done):
        """添加一条经验"""
        # 处理 state
        if isinstance(state, np.ndarray):
            self.state[self.ptr] = state
        else:
            self.state[self.ptr] = np.array(state)
        
        # 处理 action - 关键修复！
        if torch.is_tensor(action):
            self.action[self.ptr] = action.cpu().numpy()
        elif isinstance(action, np.ndarray):
            self.action[self.ptr] = action
        else:
            self.action[self.ptr] = np.array(action)
        
        # 处理 next_state
        if isinstance(next_state, np.ndarray):
            self.next_state[self.ptr] = next_state
        else:
            self.next_state[self.ptr] = np.array(next_state)
        
        # 处理 reward 和 done
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done  # not_done标志
        
        # 更新指针和大小
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def sample(self, batch_size, device='cpu'):
        """随机采样一个batch"""
        # 随机选择索引
        ind = np.random.randint(0, self.size, size=batch_size)
        
        # 转换为torch张量
        batch = Batch(
            state=torch.FloatTensor(self.state[ind]).to(device),
            action=torch.FloatTensor(self.action[ind]).to(device),
            next_state=torch.FloatTensor(self.next_state[ind]).to(device),
            reward=torch.FloatTensor(self.reward[ind]).to(device),
            not_done=torch.FloatTensor(self.not_done[ind]).to(device),
            extra={}
        )
        
        return batch
    
    def __len__(self):
        return self.size