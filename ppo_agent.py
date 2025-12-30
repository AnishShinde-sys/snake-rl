import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import json

# #region agent log
_DEBUG_LOG_PATH = "/Users/anishshinde/snake-rl/.cursor/debug.log"
def _debug_log(hyp, loc, msg, data):
    with open(_DEBUG_LOG_PATH, "a") as f:
        f.write(json.dumps({"hypothesisId": hyp, "location": loc, "message": msg, "data": data, "timestamp": __import__('time').time()}) + "\n")
# #endregion


class ActorCritic(nn.Module):
    """Neural network for PPO with SEPARATE actor-critic architecture.
    
    Key fix: Actor and critic have completely separate networks.
    This prevents critic gradients from dominating shared layers.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        # SEPARATE Actor network (policy) - no shared layers!
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),  # Tanh for actor (bounded outputs, better gradient flow)
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # SEPARATE Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize actor with orthogonal init for better policy gradient learning
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better learning"""
        for module in self.actor:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        # Last layer of actor should have smaller weights for better exploration
        nn.init.orthogonal_(self.actor[-2].weight, gain=0.01)
        
        for module in self.critic:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state):
        """Forward pass through network"""
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value
    
    def act(self, state):
        """Select action given state"""
        action_probs, state_value = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.item(), action_logprob, state_value


class PPOAgent:
    """PPO Agent for training"""
    
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, 
                 eps_clip=0.2, k_epochs=10, device='cpu'):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.device = device
        
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.lr = lr
        self.min_lr = lr * 0.1  # Don't go below 10% of initial LR
        self.actor_optimizer = optim.Adam(self.policy.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.policy.critic.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.mse_loss = nn.MSELoss()
        self.update_count = 0  # Track number of updates for LR decay
        
        # Memory
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.state_values = []
    
    def select_action(self, state, action_mask=None):
        """Select action using old policy with optional action masking"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            action_probs, state_value = self.policy_old(state_tensor)
            
            # Apply action mask - zero out probabilities for unsafe actions
            if action_mask is not None:
                mask_tensor = torch.FloatTensor(action_mask).to(self.device)
                # Mask: multiply probs by mask, then renormalize
                masked_probs = action_probs * mask_tensor
                # Handle edge case where all are masked (shouldn't happen with our mask logic)
                if masked_probs.sum() > 0:
                    masked_probs = masked_probs / masked_probs.sum()
                else:
                    masked_probs = action_probs  # Fallback to original
                action_probs_to_use = masked_probs
            else:
                action_probs_to_use = action_probs
            
            dist = Categorical(action_probs_to_use)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            action = action.item()
            
            # #region agent log
            probs = action_probs.cpu().numpy().tolist()
            masked = action_probs_to_use.cpu().numpy().tolist() if action_mask else probs
            # Log every 500 steps
            if len(self.actions) % 500 == 0:
                _debug_log("H_MASK", "ppo_agent.py:115", "ACTION_SELECT", {
                    "action": action, "raw_probs": [round(p, 3) for p in probs],
                    "masked_probs": [round(p, 3) for p in masked],
                    "mask": action_mask, "value": float(state_value.item())
                })
            # #endregion
        
        self.states.append(state_tensor)
        self.actions.append(action)
        self.logprobs.append(action_logprob)
        self.state_values.append(state_value)
        
        return action
    
    def store_transition(self, reward, is_terminal):
        """Store reward and terminal flag"""
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)
    
    def update(self):
        """Update policy using PPO"""
        # #region agent log
        _debug_log("H3", "ppo_agent.py:108", "UPDATE_START", {
            "num_transitions": len(self.rewards),
            "raw_rewards_sample": self.rewards[:10] if len(self.rewards) > 10 else self.rewards
        })
        # #endregion
        
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.rewards), reversed(self.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Convert rewards to tensor
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        
        # Convert list to tensor
        old_states = torch.stack(self.states).detach()
        old_actions = torch.tensor(self.actions).to(self.device)
        old_logprobs = torch.stack(self.logprobs).detach()
        old_state_values = torch.stack(self.state_values).squeeze().detach()
        
        # Calculate advantages (GAE would be better, but this is simpler)
        advantages = rewards - old_state_values
        
        # #region agent log
        _debug_log("H3", "ppo_agent.py:130", "ADVANTAGES_BEFORE_NORM", {
            "adv_mean": float(advantages.mean()), "adv_std": float(advantages.std()),
            "adv_min": float(advantages.min()), "adv_max": float(advantages.max()),
            "rewards_mean": float(rewards.mean()), "values_mean": float(old_state_values.mean())
        })
        # #endregion
        
        # FIX: Scale advantages instead of normalizing to zero-mean
        # This keeps the policy gradient signal strong
        if len(advantages) > 1:
            advantages = advantages / (advantages.std() + 1e-7)  # Scale but keep mean!
        
        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            # Evaluating old actions and values
            action_probs, state_values = self.policy(old_states)
            dist = Categorical(action_probs)
            action_logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(action_logprobs - old_logprobs)
            
            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Final loss of clipped objective PPO
            # SEPARATE losses for actor and critic (key fix!)
            value_loss = self.mse_loss(state_values.squeeze(), rewards)
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy_bonus = dist_entropy.mean()  # Positive = encourage exploration
            
            # Actor loss: policy gradient + entropy bonus
            # Higher entropy (0.05) to keep exploring better strategies
            actor_loss = policy_loss - 0.05 * entropy_bonus
            
            # #region agent log
            if _ == 0 or _ == self.k_epochs - 1:  # Log first AND last epoch
                _debug_log("H3", "ppo_agent.py:165", "LOSS_VALUES", {
                    "epoch": _, "policy_loss": float(policy_loss), "value_loss": float(value_loss),
                    "entropy": float(entropy_bonus), "actor_loss": float(actor_loss),
                    "ratio_mean": float(ratios.mean()), "ratio_std": float(ratios.std())
                })
            # #endregion
            
            # Update actor (policy network) - no gradient clipping, let it learn!
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optimizer.step()
            
            # Update critic (value network)
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), 0.5)
            self.critic_optimizer.step()
        
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # LR decay: gradually reduce LR to help convergence
        self.update_count += 1
        if self.update_count % 50 == 0:  # Every 50 updates, decay LR
            new_lr = max(self.min_lr, self.lr * (0.995 ** (self.update_count // 50)))
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = new_lr
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = new_lr
        
        # Clear memory
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.state_values = []
    
    def save(self, filepath):
        """Save model weights"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath):
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_old.load_state_dict(checkpoint['policy_state_dict'])
        if 'actor_optimizer_state_dict' in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

