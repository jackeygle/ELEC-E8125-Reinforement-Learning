"""
DDPG Agent - 完整实现版本
"""

from pathlib import Path
import copy, time
import utils.common_utils as cu
import torch
import numpy as np
import torch.nn.functional as F

from .agent_base import BaseAgent
from .ddpg_utils import Policy, Critic, ReplayBuffer

def to_numpy(tensor):
    return tensor.cpu().numpy().flatten()

class DDPGAgent(BaseAgent):
    def __init__(self, config=None):
        super(DDPGAgent, self).__init__(config)
        self.device = self.cfg.device
        self.name = 'ddpg'
        state_dim = self.observation_space_dim
        self.action_dim = self.action_space_dim
        self.max_action = self.cfg.max_action
        self.lr = float(self.cfg.lr)  # 确保lr是浮点数
      
        # 初始化经验回放缓冲区
        self.buffer = ReplayBuffer((state_dim,), self.action_dim, max_size=int(float(self.cfg.buffer_size)))
        
        self.batch_size = self.cfg.batch_size
        self.gamma = self.cfg.gamma
        self.tau = self.cfg.tau
        
        # 初始化Actor和Critic网络
        self.pi = Policy(state_dim, self.action_dim, self.max_action).to(self.device)
        self.pi_target = copy.deepcopy(self.pi)
        self.pi_optimizer = torch.optim.Adam(self.pi.parameters(), lr=self.lr)
        
        self.q = Critic(state_dim, self.action_dim).to(self.device)
        self.q_target = copy.deepcopy(self.q)
        self.q_optimizer = torch.optim.Adam(self.q.parameters(), lr=self.lr)
        
        # 用于计数
        self.buffer_ptr = 0
        self.buffer_head = 0 
        self.random_transition = 5000  # 前5000步随机探索
        self.max_episode_steps = self.cfg.max_episode_steps
    

    def update(self):
        """ 每次收集完一条轨迹后，更新策略和Q函数 """
        info = {}
        update_iter = self.buffer_ptr - self.buffer_head
        
        # 只有当缓冲区有足够数据时才开始更新
        if self.buffer_ptr > self.random_transition:
            # 对轨迹中的每个transition都更新一次
            for _ in range(update_iter):
                # 从缓冲区采样一个batch
                batch = self.buffer.sample(self.batch_size, device=self.device)
                
                # ===== 更新Critic网络 =====
                with torch.no_grad():
                    # 计算目标Q值: y = r + γ * Q_target(s', π_target(s'))
                    next_action = self.pi_target(batch.next_state)
                    target_q = self.q_target(batch.next_state, next_action)
                    target_q = batch.reward + batch.not_done * self.gamma * target_q
                
                # 当前Q值
                current_q = self.q(batch.state, batch.action)
                
                # Critic损失 (MSE)
                q_loss = F.mse_loss(current_q, target_q)
                
                # 更新Critic
                self.q_optimizer.zero_grad()
                q_loss.backward()
                self.q_optimizer.step()
                
                # ===== 更新Actor网络 =====
                # Actor损失: -Q(s, π(s)) (最大化Q值)
                pi_loss = -self.q(batch.state, self.pi(batch.state)).mean()
                
                # 更新Actor
                self.pi_optimizer.zero_grad()
                pi_loss.backward()
                self.pi_optimizer.step()
                
                # ===== 软更新目标网络 =====
                cu.soft_update_params(self.q, self.q_target, self.tau)
                cu.soft_update_params(self.pi, self.pi_target, self.tau)
        
        # 更新buffer_head
        self.buffer_head = self.buffer_ptr
        
        # 返回空字典，保持与原始代码一致
        return info

    
    @torch.no_grad()
    def get_action(self, observation, evaluation=False):
        """获取动作"""
        
        # 前5000步随机探索
        if self.buffer_ptr < self.random_transition and not evaluation:
            action = torch.randn(self.action_dim) * self.max_action
        else:
            # 将观察转换为张量
            state = torch.FloatTensor(observation.reshape(1, -1)).to(self.device)
            
            # 从策略网络获取动作
            action = self.pi(state).cpu().squeeze()
            
            # 添加探索噪声（训练时）
            if not evaluation:
                noise = torch.randn_like(action) * self.max_action * 0.1
                action = action + noise
                # 限制在动作范围内
                action = torch.clamp(action, -self.max_action, self.max_action)
        
        return action, {}

    def record(self, state, action, next_state, reward, done):
        """ 保存transition到缓冲区 """
        self.buffer_ptr += 1
        self.buffer.add(state, action, next_state, reward, done)
    
    def train_iteration(self):
        # 运行一个episode
        reward_sum, timesteps, done = 0, 0, False
        obs, _ = self.env.reset()
        
        while not done:
            # 从策略采样动作
            action, _ = self.get_action(obs)

            # 执行动作
            next_obs, reward, done, _, _ = self.env.step(to_numpy(action))

            # 存储transition
            done_bool = float(done) if timesteps < self.max_episode_steps else 0 
            self.record(obs, action, next_obs, reward, done_bool)
                
            # 累计奖励
            reward_sum += reward
            timesteps += 1
            
            if timesteps >= self.max_episode_steps:
                done = True
            
            # 更新观察
            obs = next_obs.copy()

        # episode结束后更新策略
        info = self.update()
        
        # 返回训练统计信息
        info.update({
            'episode_length': timesteps,
            'ep_reward': reward_sum,
        })
        
        return info
        
    def train(self):
        if self.cfg.save_logging:
            L = cu.Logger() # create a simple logger to record stats
        start = time.perf_counter()
        total_step=0
        run_episode_reward=[]
        log_count=0
        
        for ep in range(self.cfg.train_episodes + 1):
            # collect data and update the policy
            train_info = self.train_iteration()
            train_info.update({'episodes': ep})
            total_step+=train_info['episode_length']
            train_info.update({'total_step': total_step})
            run_episode_reward.append(train_info['ep_reward'])
            
            if total_step>self.cfg.log_interval*log_count:
                average_return=sum(run_episode_reward)/len(run_episode_reward)
                if not self.cfg.silent:
                    print(f"Episode {ep} Step {total_step} finished. Average episode return: {average_return}")
                if self.cfg.save_logging:
                    train_info.update({'average_return':average_return})
                    L.log(**train_info)
                run_episode_reward=[]
                log_count+=1

        if self.cfg.save_model:
            self.save_model()
            
        logging_path = str(self.logging_dir)+'/logs'   
        if self.cfg.save_logging:
            L.save(logging_path, self.seed)
        self.env.close()

        end = time.perf_counter()
        train_time = (end-start)/60
        print('------ Training Finished ------')
        print(f'Total traning time is {train_time}mins')
        
    def load_model(self):
        filepath = str(self.model_dir) + '/model_parameters_' + str(self.seed) + '.pt'
        
        d = torch.load(filepath)
        self.q.load_state_dict(d['q'])
        self.q_target.load_state_dict(d['q_target'])
        self.pi.load_state_dict(d['pi'])
        self.pi_target.load_state_dict(d['pi_target'])
    
    def save_model(self):
        filepath = str(self.model_dir) + '/model_parameters_' + str(self.seed) + '.pt'
        
        torch.save({
            'q': self.q.state_dict(),
            'q_target': self.q_target.state_dict(),
            'pi': self.pi.state_dict(),
            'pi_target': self.pi_target.state_dict()
        }, filepath)
        print("Saved model to", filepath, "...")