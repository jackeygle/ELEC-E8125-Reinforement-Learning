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

#     ## Your code starts here. ######
#     # You need to override the update method to implement the DDPG with extensions.
#     # You can modify other functions of the base class if needed.
#     def __init__(self, config=None):
#         super(DDPGExtension, self).__init__(config)
#         # extension-specific hyperparams
#         self.policy_delay = getattr(self.cfg, 'policy_delay', 2)  # update actor every policy_delay steps
#         self.target_policy_noise = getattr(self.cfg, 'target_policy_noise', 0.2)  # smoothing noise std
#         self.target_noise_clip = getattr(self.cfg, 'target_noise_clip', 0.5)
#         # running stats for reward normalization
#         self.ret_rms_eps = 1e-6
#         self.ret_mean = 0.0
#         self.ret_var = 1.0
#         self.ret_count = 1e-4
#         # prioritized replay placeholder - not implemented fully
#         # self.prioritized = getattr(self.cfg, 'prioritized', False)

#         # ensure some attributes exist from parent
#         if not hasattr(self, 'total_updates'):
#             self.total_updates = 0

#     def _update_reward_stats(self, rewards):
#         # a simple running mean/var for reward normalization
#         r = np.asarray(rewards, dtype=np.float64).ravel()
#         if r.size == 0:
#             return
#         batch_mean = r.mean()
#         batch_var = r.var()
#         batch_count = r.size

#         # Welford-style update
#         delta = batch_mean - self.ret_mean
#         tot_count = self.ret_count + batch_count
#         new_mean = self.ret_mean + delta * (batch_count / tot_count)
#         m_a = self.ret_var * (self.ret_count)
#         m_b = batch_var * (batch_count)
#         M2 = m_a + m_b + delta**2 * (self.ret_count * batch_count / tot_count)
#         new_var = M2 / tot_count

#         self.ret_mean = new_mean
#         self.ret_var = new_var
#         self.ret_count = tot_count

#     def update(self,):
#         """
#         DDPG extension update:
#          - target policy smoothing (add clipped noise to target actions, like TD3)
#          - delayed policy update (update actor less frequently)
#          - reward normalization (simple running mean/var)
#         """
#         info = {}

#         if self.buffer.size() < max(self.batch_size, 1):
#             return info

#         # control update count similar to base
#         n_updates = max(1, int(self.buffer_head)) if hasattr(self, 'buffer_head') and self.buffer_head>0 else 1
#         max_updates = max(1, int(self.buffer.size() // max(1, self.batch_size)))
#         n_updates = min(n_updates, max_updates)

#         q_losses = []
#         pi_losses = []
#         for _u in range(n_updates):
#             batch = self.buffer.sample(self.batch_size)
#             if isinstance(batch, dict):
#                 s = batch['obs']
#                 a = batch['act']
#                 s2 = batch['next_obs']
#                 r = batch['rew']
#                 d = batch['done']
#             else:
#                 try:
#                     s, a, s2, r, d = batch
#                 except Exception:
#                     s = batch[0]; a = batch[1]; s2 = batch[2]; r = batch[3]; d = batch[4]

#             # reward normalization: update running stats and normalize r
#             self._update_reward_stats(r)
#             r = (r - self.ret_mean) / (np.sqrt(self.ret_var) + self.ret_rms_eps)

#             # convert to tensors
#             s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
#             a = torch.as_tensor(a, dtype=torch.float32, device=self.device)
#             s2 = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
#             r = torch.as_tensor(r, dtype=torch.float32, device=self.device).unsqueeze(-1)
#             d = torch.as_tensor(d, dtype=torch.float32, device=self.device).unsqueeze(-1)

#             # target action with smoothing noise
#             with torch.no_grad():
#                 a2 = self.pi_target(s2)
#                 # add clipped noise (target policy smoothing)
#                 noise = torch.randn_like(a2) * self.target_policy_noise
#                 noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
#                 a2 = (a2 + noise).clamp(-self.max_action, self.max_action)

#                 q_target_next = self.q_target(s2, a2)
#                 y = r + (1.0 - d) * (self.gamma * q_target_next)

#             # critic update
#             q_val = self.q(s, a)
#             critic_loss = F.mse_loss(q_val, y)

#             self.q_optimizer.zero_grad()
#             critic_loss.backward()
#             torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
#             self.q_optimizer.step()

#             q_losses.append(critic_loss.item())

#             # delayed policy update: only update actor every policy_delay steps
#             if (self.total_updates % self.policy_delay) == 0:
#                 pi_action = self.pi(s)
#                 actor_loss = -self.q(s, pi_action).mean()
#                 self.pi_optimizer.zero_grad()
#                 actor_loss.backward()
#                 torch.nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
#                 self.pi_optimizer.step()
#                 pi_losses.append(actor_loss.item())
#             else:
#                 # record a placeholder if actor not updated
#                 pi_losses.append(0.0)

#             # soft update targets
#             cu.soft_update_params(self.q, self.q_target, self.tau)
#             cu.soft_update_params(self.pi, self.pi_target, self.tau)

#             # noise decay for exploration (inherited noise_scale)
#             self.noise_scale = max(self.min_noise_scale, self.noise_scale * self.noise_decay)
#             self.total_updates += 1

#         # reset buffer head
#         self.buffer_head = 0

#         info.update({
#             'q_loss': float(np.mean(q_losses)) if len(q_losses)>0 else 0.0,
#             'pi_loss': float(np.mean(pi_losses)) if len(pi_losses)>0 else 0.0,
#             'noise_scale': float(self.noise_scale),
#             'updates': n_updates,
#             'ret_mean': float(self.ret_mean),
#             'ret_var': float(self.ret_var)
#         })
#         return info   
    # pass