import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """Neural network for PPO with actor-critic architecture"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state):
        """Forward pass through network"""
        shared_features = self.shared(state)
        action_probs = self.actor(shared_features)
        state_value = self.critic(shared_features)
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
    
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.95, 
                 eps_clip=0.2, k_epochs=10, device='cpu'):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.device = device
        
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.mse_loss = nn.MSELoss()
        
        # Memory
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.state_values = []
    
    def select_action(self, state):
        """Select action using old policy"""
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob, state_value = self.policy_old.act(state)
        
        self.states.append(state)
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
        
        # Normalize advantages (not rewards) for better stability
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        
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
            # Use rewards (not normalized) for value loss, advantages (normalized) for policy loss
            value_loss = self.mse_loss(state_values.squeeze(), rewards)
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -dist_entropy.mean()
            
            loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
            
            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
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
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath):
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_old.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

