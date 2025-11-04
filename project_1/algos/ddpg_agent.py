"""
DDPG Agent

This file implements the DDPG algorithm. Students need to complete the code in the update() and get_action() functions.
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
        self.device = self.cfg.device  # ""cuda" if torch.cuda.is_available() else "cpu"
        self.name = 'ddpg'
        state_dim = self.observation_space_dim
        self.action_dim = self.action_space_dim
        self.max_action = self.cfg.max_action
        self.lr=self.cfg.lr
      
        self.buffer = ReplayBuffer((state_dim,), self.action_dim, max_size=int(float(self.cfg.buffer_size)))
        
        self.batch_size = self.cfg.batch_size
        self.gamma = self.cfg.gamma
        self.tau = self.cfg.tau
        
        # used to count number of transitions in a trajectory
        self.buffer_ptr = 0
        self.buffer_head = 0 
        self.random_transition = 5000 # collect 5k random data for better exploration
        self.max_episode_steps=self.cfg.max_episode_steps
        
        
                # Initialize networks
        self.pi = Policy(state_dim, self.action_dim, self.max_action).to(self.device)
        self.pi_target = copy.deepcopy(self.pi)
  

        self.q = Critic(state_dim, self.action_dim).to(self.device)
        self.q_target = copy.deepcopy(self.q)

        self.pi_optimizer = torch.optim.Adam(self.pi.parameters(), lr=float(self.lr))
        self.q_optimizer = torch.optim.Adam(self.q.parameters(), lr=float(self.lr))
        # Exploration noise parameters
        self.noise_scale = self.cfg.expl_noise if hasattr(self.cfg, 'expl_noise') else 0.1
        self.noise_decay = 0.9999
        self.min_noise_scale = 0.01
        self.total_updates = 0

    def update(self,):
        """ After collecting one trajectory, update the pi and q for #transition times: """
        info = {}
        
        ###### Your code starts here. ######
        # 1. Update the network once per transition
        # 2. Update once you have enough data in the buffer

        # Hints: 1. compute the Q target with the q_target and pi_target networks
        #        2. compute the critic loss and update the q's parameters
        #        3. compute actor loss and update the pi's parameters
        #        4. update the target q and pi using cu.soft_update_params() (See the DQN code)
                # Basic checks: do not update until we have enough samples
        if self.buffer.size() < max(self.batch_size, 1):
            # no update possible
            return info

        # Number of gradient steps: we follow "update once per transition" semantic from template
        # We'll perform `buffer_head` updates if buffer_head recorded, otherwise at least one update.
        n_updates = max(1, int(self.buffer_head)) if hasattr(self, 'buffer_head') and self.buffer_head>0 else 1

        # But to avoid huge bursts, cap updates to buffer size // batch
        max_updates = max(1, int(self.buffer.size() // max(1, self.batch_size)))
        n_updates = min(n_updates, max_updates)

        q_losses = []
        pi_losses = []

        for _ in range(n_updates):
            # Sample a batch from replay buffer
            batch = self.buffer.sample(self.batch_size)
            # ReplayBuffer.sample might return a dict or tuple; handle common formats
            if isinstance(batch, dict):
                s = batch['obs']
                a = batch['act']
                s2 = batch['next_obs']
                r = batch['rew']
                d = batch['done']
            else:
                # assume tuple: (s, a, s2, r, d)
                try:
                    s, a, s2, r, d = batch
                except Exception:
                    # try alternative ordering
                    s = batch[0]; a = batch[1]; s2 = batch[2]; r = batch[3]; d = batch[4]

            # convert to tensors
            s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
            a = torch.as_tensor(a, dtype=torch.float32, device=self.device)
            s2 = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
            r = torch.as_tensor(r, dtype=torch.float32, device=self.device).unsqueeze(-1)
            d = torch.as_tensor(d, dtype=torch.float32, device=self.device).unsqueeze(-1)

            # Q target: r + gamma * (1 - done) * q_target(next_state, pi_target(next_state))
            with torch.no_grad():
                a2 = self.pi_target(s2)
                q_target_next = self.q_target(s2, a2)
                y = r + (1.0 - d) * (self.gamma * q_target_next)

            # current Q
            q_val = self.q(s, a)
            critic_loss = F.mse_loss(q_val, y)

            # update critic
            self.q_optimizer.zero_grad()
            critic_loss.backward()
            # gradient clipping optional
            torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
            self.q_optimizer.step()

            # actor loss: maximize Q(s, pi(s)) -> minimize -Q
            pi_action = self.pi(s)
            actor_loss = -self.q(s, pi_action).mean()

            self.pi_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
            self.pi_optimizer.step()

            # soft update targets
            cu.soft_update_params(self.q, self.q_target, self.tau)
            cu.soft_update_params(self.pi, self.pi_target, self.tau)

            # record stats
            q_losses.append(critic_loss.item())
            pi_losses.append(actor_loss.item())

            # noise decay
            self.noise_scale = max(self.min_noise_scale, self.noise_scale * self.noise_decay)
            self.total_updates += 1

        # update buffer head (reset)
        self.buffer_head = 0

        info.update({
            'q_loss': float(np.mean(q_losses)) if len(q_losses)>0 else 0.0,
            'pi_loss': float(np.mean(pi_losses)) if len(pi_losses)>0 else 0.0,
            'noise_scale': float(self.noise_scale),
            'updates': n_updates
        })
        # Remember to update the buffer head

        
        return info



    
    @torch.no_grad()
    def get_action(self, observation, evaluation=False):
        
        ##### Your code starts here. ######
        # 1. Get action from the policy network
        # 2. Add noise for exploration (if evaluation == False) i.e. select random action
        # 
        # 检查 self.pi 是否存在
        if not hasattr(self, 'pi'):
            raise AttributeError("self.pi is not defined. Ensure it is initialized in DDPGAgent or external script.")

        # 将 observation 转换为 tensor
        if isinstance(observation, np.ndarray):
            obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        else:
            obs = observation.to(self.device).float()
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        # 从策略网络获得动作
        action_tensor = self.pi(obs)  # 假设 self.pi 输出形状为 (batch, 2)
        action = to_numpy(action_tensor.squeeze(0))  # 转换为 numpy.ndarray, 形状 (2,)

        # 训练时添加噪声
        if not evaluation:
            noise_scale = getattr(self, 'noise_scale', 0.2)  # 默认噪声标准差
            noise = np.random.normal(0, noise_scale, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)  # 动作范围匹配 SandingEnv 的 [-1, 1]

        # 调试输出
        print(f"DEBUG: get_action: action shape = {action.shape}, action = {action}")
        return action # just return a positional value
        # return action, {} # just return a positional value

    def record(self, state, action, next_state, reward, done):
        """ Save transitions to the buffer. """
        self.buffer_ptr += 1
        self.buffer.add(state, action, next_state, reward, done)
    
    def train_iteration(self):
        #start = time.perf_counter()
        # Run actual training        
        reward_sum, timesteps, done = 0, 0, False
        # Reset the environment and observe the initial state
        obs, _ = self.env.reset()
        while not done:
            
            # Sample action from policy
            action = self.get_action(obs)
            print(f"DEBUG: action type = {type(action)}, action = {action}")  # 加这行调试
            # Perform the action on the environment, get new state and reward
            next_obs, reward, done, _, _ = self.env.step(to_numpy(action))

            # Store action's outcome (so that the agent can improve its policy)        
            
            done_bool = float(done) if timesteps < self.max_episode_steps else 0 
            self.record(obs, action, next_obs, reward, done_bool)
                
            # Store total episode reward
            reward_sum += reward
            timesteps += 1
            
            if timesteps >= self.max_episode_steps:
                done = True
            # update observation
            obs = next_obs.copy()

        # update the policy after one episode
        #s = time.perf_counter()
        info = self.update()
        #e = time.perf_counter()
        
        # Return stats of training
        info.update({
                    'episode_length': timesteps,
                    'ep_reward': reward_sum,
                    })
        
        end = time.perf_counter()
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
        # define the save path, do not modify
        filepath=str(self.model_dir)+'/model_parameters_'+str(self.seed)+'.pt'
        
        d = torch.load(filepath)
        self.q.load_state_dict(d['q'])
        self.q_target.load_state_dict(d['q_target'])
        self.pi.load_state_dict(d['pi'])
        self.pi_target.load_state_dict(d['pi_target'])
    
    def save_model(self):   
        # define the save path, do not modify
        filepath=str(self.model_dir)+'/model_parameters_'+str(self.seed)+'.pt'
        
        torch.save({
            'q': self.q.state_dict(),
            'q_target': self.q_target.state_dict(),
            'pi': self.pi.state_dict(),
            'pi_target': self.pi_target.state_dict()
        }, filepath)
        print("Saved model to", filepath, "...")
        
        
