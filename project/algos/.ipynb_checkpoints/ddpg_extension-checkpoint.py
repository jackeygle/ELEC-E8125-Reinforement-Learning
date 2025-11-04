import copy, time
from pathlib import Path
import utils.common_utils as cu
import numpy as np
import torch
import torch.nn.functional as F

from .agent_base import BaseAgent
from .ddpg_utils import Policy, Critic, ReplayBuffer
from .ddpg_agent import DDPGAgent

def to_numpy(tensor):
    return tensor.cpu().numpy().flatten()

class DDPGExtension(DDPGAgent):

    ## Your code starts here. ######
    # You need to override the update method to implement the DDPG with extensions.
    # You can modify other functions of the base class if needed.



    def __init__(self, config=None):
        super(DDPGExtension, self).__init__(config)
        
        # SIL超参数
        self.sil_coef = 0.1  # SIL损失的权重系数
        self.sil_update_freq = 4  # 每4次常规更新做1次SIL更新
        self.update_counter = 0
        
        # 用于存储episode的轨迹
        self.episode_buffer = []
        self.best_episodes = []  # 存储高回报的episodes
        self.max_best_episodes = 100  # 最多保存100个好的episodes
        self.return_percentile = 70  # 只保留前30%的好episodes
        
    def record(self, state, action, next_state, reward, done):
        """保存transition - 同时记录到episode buffer"""
        # 调用父类方法保存到replay buffer
        super().record(state, action, next_state, reward, done)
        
        # 保存到当前episode buffer
        self.episode_buffer.append({
            'state': state.copy() if isinstance(state, np.ndarray) else state,
            'action': action.clone() if torch.is_tensor(action) else action.copy(),
            'next_state': next_state.copy() if isinstance(next_state, np.ndarray) else next_state,
            'reward': reward,
            'done': done
        })
    
    def compute_returns(self, rewards, gamma=None):
        """计算discounted returns"""
        if gamma is None:
            gamma = self.gamma
        
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        return returns
    
    def store_good_episode(self):
        """将当前episode存储到best_episodes中（如果足够好）"""
        if len(self.episode_buffer) == 0:
            return
        
        # 计算episode的总回报
        rewards = [t['reward'] for t in self.episode_buffer]
        episode_return = sum(rewards)
        
        # 计算每个transition的return
        returns = self.compute_returns(rewards)
        
        # 为每个transition添加return信息
        for i, transition in enumerate(self.episode_buffer):
            transition['return'] = returns[i]
        
        # 将episode添加到best_episodes
        self.best_episodes.append({
            'transitions': self.episode_buffer.copy(),
            'episode_return': episode_return
        })
        
        # 保持best_episodes的大小限制，只保留高回报的episodes
        if len(self.best_episodes) > self.max_best_episodes:
            # 按episode_return排序，保留top episodes
            self.best_episodes.sort(key=lambda x: x['episode_return'], reverse=True)
            threshold_idx = int(len(self.best_episodes) * self.return_percentile / 100)
            self.best_episodes = self.best_episodes[:threshold_idx]
        
        # 清空当前episode buffer
        self.episode_buffer = []
    
    def sample_good_transitions(self, batch_size):
        """从好的episodes中采样transitions"""
        if len(self.best_episodes) == 0:
            return None
        
        # 收集所有好的transitions
        all_good_transitions = []
        for episode in self.best_episodes:
            all_good_transitions.extend(episode['transitions'])
        
        if len(all_good_transitions) < batch_size:
            return None
        
        # 随机采样
        indices = np.random.choice(len(all_good_transitions), batch_size, replace=False)
        sampled = [all_good_transitions[i] for i in indices]
        
        # 转换为batch格式
        states = torch.FloatTensor(np.array([t['state'] for t in sampled])).to(self.device)
        actions = torch.stack([t['action'] if torch.is_tensor(t['action']) 
                               else torch.FloatTensor(t['action']) for t in sampled]).to(self.device)
        returns = torch.FloatTensor([t['return'] for t in sampled]).reshape(-1, 1).to(self.device)
        
        return states, actions, returns
    
    def update(self):
        """更新策略 - 加入SIL"""
        info = {}
        update_iter = self.buffer_ptr - self.buffer_head
        
        if self.buffer_ptr > self.random_transition:
            for _ in range(update_iter):
                self.update_counter += 1
                batch = self.buffer.sample(self.batch_size, device=self.device)
                
                # ===== 标准DDPG更新Critic =====
                with torch.no_grad():
                    next_action = self.pi_target(batch.next_state)
                    target_q = self.q_target(batch.next_state, next_action)
                    target_q = batch.reward + batch.not_done * self.gamma * target_q
                
                current_q = self.q(batch.state, batch.action)
                q_loss = F.mse_loss(current_q, target_q)
                
                self.q_optimizer.zero_grad()
                q_loss.backward()
                self.q_optimizer.step()
                
                # ===== 标准DDPG更新Actor =====
                pi_loss = -self.q(batch.state, self.pi(batch.state)).mean()
                
                self.pi_optimizer.zero_grad()
                pi_loss.backward()
                self.pi_optimizer.step()
                
                # ===== SIL更新 =====
                # 每隔sil_update_freq次更新进行一次SIL更新
                if self.update_counter % self.sil_update_freq == 0:
                    sil_data = self.sample_good_transitions(self.batch_size // 2)
                    
                    if sil_data is not None:
                        sil_states, sil_actions, sil_returns = sil_data
                        
                        # 当前策略生成的动作
                        current_actions = self.pi(sil_states)
                        
                        # 计算当前Q值和优势
                        with torch.no_grad():
                            # 过去好动作的Q值
                            past_q = self.q(sil_states, sil_actions)
                            # 当前策略动作的Q值
                            current_q_sil = self.q(sil_states, current_actions)
                            # 优势函数：过去的动作比当前的好多少
                            advantage = past_q - current_q_sil
                            # 也可以使用return作为优势的替代
                            # advantage = sil_returns - current_q_sil
                        
                        # 只有当advantage > 0时才进行模仿学习
                        # 这意味着只模仿那些比当前策略更好的动作
                        positive_mask = (advantage > 0).float()
                        
                        if positive_mask.sum() > 0:
                            # 计算模仿损失（让当前策略的动作接近过去的好动作）
                            sil_loss = (positive_mask * F.mse_loss(
                                current_actions, 
                                sil_actions, 
                                reduction='none'
                            ).mean(dim=1, keepdim=True)).sum() / (positive_mask.sum() + 1e-8)
                            
                            # 加权后的总损失
                            weighted_sil_loss = self.sil_coef * sil_loss
                            
                            # 反向传播SIL损失
                            self.pi_optimizer.zero_grad()
                            weighted_sil_loss.backward()
                            self.pi_optimizer.step()
                            
                            info['sil_loss'] = sil_loss.item()
                            info['sil_positive_ratio'] = (positive_mask.sum() / len(positive_mask)).item()
                
                # ===== 软更新目标网络 =====
                cu.soft_update_params(self.q, self.q_target, self.tau)
                cu.soft_update_params(self.pi, self.pi_target, self.tau)
        
        self.buffer_head = self.buffer_ptr
        return info
    
