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

# Actor-critic agent
# class Policy(nn.Module):
#     def __init__(self, state_dim, action_dim, max_action):
#         super().__init__()
#         self.max_action = max_action
        

#     def forward(self, state):
#         return 


# class Critic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super().__init__()
#         True

#     def forward(self, state, action):
#         return

# class ReplayBuffer(object):
#     def __init__(self, state_shape:tuple, action_dim: int, max_size=int(1e6)):
#         True
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q_net(x)


class ReplayBuffer(object):
    def __init__(self, state_shape: tuple, action_dim: int, max_size=int(1e6)):
        ##### Your code starts here. #####
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        
        self.state = np.zeros((self.max_size, *state_shape), dtype=np.float32)
        self.action = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((self.max_size, *state_shape), dtype=np.float32)
        self.reward = np.zeros((self.max_size, 1), dtype=np.float32)
        self.done = np.zeros((self.max_size, 1), dtype=np.float32)
        ##### Your code ends here. #####

    def add(self, state, action, next_state, reward, done):
        ##### Your code starts here. #####
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        ##### Your code ends here. #####

    def sample(self, batch_size):
        ##### Your code starts here. #####
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = (
            self.state[idx],
            self.action[idx],
            self.next_state[idx],
            self.reward[idx],
            self.done[idx],
        )
        return batch
        ##### Your code ends here. #####
