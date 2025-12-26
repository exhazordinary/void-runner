"""
VOID_RUNNER - Curiosity-Driven PPO Agent
=========================================
PPO agent enhanced with intrinsic motivation for exploration.

Key innovations:
1. Dual value functions (intrinsic + extrinsic)
2. Different discount factors for curiosity (higher gamma_int encourages long-term exploration)
3. Combined reward signal that fades extrinsic weight over time
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque

from .networks import PolicyNetwork, RunningMeanStd
from .curiosity import CuriosityModule


@dataclass
class RolloutBuffer:
    """Storage for PPO rollout data."""
    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    log_probs: List[np.ndarray] = field(default_factory=list)
    rewards_ext: List[np.ndarray] = field(default_factory=list)
    rewards_int: List[np.ndarray] = field(default_factory=list)
    values_ext: List[np.ndarray] = field(default_factory=list)
    values_int: List[np.ndarray] = field(default_factory=list)
    dones: List[np.ndarray] = field(default_factory=list)
    next_observations: List[np.ndarray] = field(default_factory=list)

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards_ext.clear()
        self.rewards_int.clear()
        self.values_ext.clear()
        self.values_int.clear()
        self.dones.clear()
        self.next_observations.clear()

    def __len__(self):
        return len(self.observations)


class VoidRunnerAgent:
    """
    VOID_RUNNER: A curiosity-driven exploration agent.

    Combines PPO with Random Network Distillation (RND) for
    intrinsic motivation. Excels in sparse reward environments
    by generating its own curiosity-based reward signal.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        # PPO hyperparameters
        lr: float = 3e-4,
        gamma_ext: float = 0.999,      # Extrinsic reward discount
        gamma_int: float = 0.99,        # Intrinsic reward discount (lower = short-term curiosity)
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.1,
        entropy_coef: float = 0.001,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        # Curiosity hyperparameters
        int_coef: float = 1.0,          # Intrinsic reward coefficient
        ext_coef: float = 2.0,          # Extrinsic reward coefficient
        curiosity_lr: float = 1e-4,
        # Training hyperparameters
        n_epochs: int = 4,
        batch_size: int = 256,
        n_envs: int = 1,
        # Device
        device: str = "cpu",
        is_image: bool = False,
        continuous: bool = False,
    ):
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_envs = n_envs
        self.is_image = is_image
        self.continuous = continuous

        # Discount factors
        self.gamma_ext = gamma_ext
        self.gamma_int = gamma_int
        self.gae_lambda = gae_lambda

        # PPO hyperparameters
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # Reward coefficients
        self.int_coef = int_coef
        self.ext_coef = ext_coef

        # Initialize policy network
        self.policy = PolicyNetwork(
            obs_dim, action_dim,
            hidden_dim=256,
            is_image=is_image,
            continuous=continuous
        ).to(device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Initialize curiosity module
        self.curiosity = CuriosityModule(
            obs_dim,
            feature_dim=512,
            learning_rate=curiosity_lr,
            is_image=is_image,
            device=device
        )

        # Reward normalization
        self.reward_rms = RunningMeanStd(shape=())
        self.obs_rms = RunningMeanStd(shape=(obs_dim,) if not is_image else (1,))

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # Statistics tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.intrinsic_rewards = deque(maxlen=100)
        self.update_count = 0

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Select action using current policy."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            if len(obs_tensor.shape) == 1:
                obs_tensor = obs_tensor.unsqueeze(0)

            action, log_prob, value_ext, value_int = self.policy.get_action(
                obs_tensor, deterministic
            )

            return (
                action.cpu().numpy(),
                log_prob.cpu().numpy(),
                value_ext.cpu().numpy(),
                value_int.cpu().numpy()
            )

    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        log_prob: np.ndarray,
        reward_ext: np.ndarray,
        value_ext: np.ndarray,
        value_int: np.ndarray,
        done: np.ndarray,
        next_obs: np.ndarray
    ):
        """Store transition in buffer and compute intrinsic reward."""
        # Compute intrinsic reward
        with torch.no_grad():
            next_obs_tensor = torch.FloatTensor(next_obs).to(self.device)
            if len(next_obs_tensor.shape) == 1:
                next_obs_tensor = next_obs_tensor.unsqueeze(0)

            reward_int = self.curiosity.compute_intrinsic_reward(
                next_obs_tensor, update=False
            ).cpu().numpy()

        # Normalize intrinsic reward
        reward_int = self.curiosity.normalize_reward(reward_int)

        self.buffer.observations.append(obs)
        self.buffer.actions.append(action)
        self.buffer.log_probs.append(log_prob)
        self.buffer.rewards_ext.append(reward_ext)
        self.buffer.rewards_int.append(reward_int)
        self.buffer.values_ext.append(value_ext)
        self.buffer.values_int.append(value_int)
        self.buffer.dones.append(done)
        self.buffer.next_observations.append(next_obs)

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: np.ndarray,
        gamma: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation."""
        n_steps = len(rewards)
        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * self.gae_lambda * next_non_terminal * last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, next_obs: np.ndarray) -> Dict[str, float]:
        """Update policy using collected rollout."""
        if len(self.buffer) == 0:
            return {}

        # Convert buffer to arrays
        observations = np.array(self.buffer.observations)
        actions = np.array(self.buffer.actions)
        old_log_probs = np.array(self.buffer.log_probs)
        rewards_ext = np.array(self.buffer.rewards_ext)
        rewards_int = np.array(self.buffer.rewards_int)
        values_ext = np.array(self.buffer.values_ext)
        values_int = np.array(self.buffer.values_int)
        dones = np.array(self.buffer.dones)
        next_observations = np.array(self.buffer.next_observations)

        # Get next values for GAE
        with torch.no_grad():
            next_obs_tensor = torch.FloatTensor(next_obs).to(self.device)
            if len(next_obs_tensor.shape) == 1:
                next_obs_tensor = next_obs_tensor.unsqueeze(0)
            _, next_value_ext, next_value_int = self.policy(next_obs_tensor)
            next_value_ext = next_value_ext.cpu().numpy().flatten()
            next_value_int = next_value_int.cpu().numpy().flatten()

        # Compute GAE for both reward streams
        advantages_ext, returns_ext = self.compute_gae(
            rewards_ext.flatten(), values_ext.flatten(),
            dones.flatten(), next_value_ext, self.gamma_ext
        )

        advantages_int, returns_int = self.compute_gae(
            rewards_int.flatten(), values_int.flatten(),
            dones.flatten(), next_value_int, self.gamma_int
        )

        # Combined advantages
        advantages = self.ext_coef * advantages_ext + self.int_coef * advantages_int
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Flatten arrays
        observations = observations.reshape(-1, self.obs_dim) if not self.is_image else observations.reshape(-1, *observations.shape[2:])
        actions = actions.flatten() if not self.continuous else actions.reshape(-1, self.action_dim)
        old_log_probs = old_log_probs.flatten()

        # Update RND predictor
        with torch.no_grad():
            next_obs_tensor = torch.FloatTensor(next_observations.reshape(-1, self.obs_dim) if not self.is_image else next_observations.reshape(-1, *next_observations.shape[2:])).to(self.device)
        self.curiosity.compute_intrinsic_reward(next_obs_tensor, update=True)

        # PPO update
        n_samples = len(observations)
        indices = np.arange(n_samples)

        total_loss = 0
        policy_loss_sum = 0
        value_loss_sum = 0
        entropy_sum = 0
        n_updates = 0

        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                # Get batch data
                obs_batch = torch.FloatTensor(observations[batch_indices]).to(self.device)
                action_batch = torch.LongTensor(actions[batch_indices]).to(self.device) if not self.continuous else torch.FloatTensor(actions[batch_indices]).to(self.device)
                old_log_prob_batch = torch.FloatTensor(old_log_probs[batch_indices]).to(self.device)
                advantage_batch = torch.FloatTensor(advantages[batch_indices]).to(self.device)
                return_ext_batch = torch.FloatTensor(returns_ext[batch_indices]).to(self.device)
                return_int_batch = torch.FloatTensor(returns_int[batch_indices]).to(self.device)

                # Evaluate actions
                log_prob, entropy, value_ext, value_int = self.policy.evaluate_actions(
                    obs_batch, action_batch
                )

                # Policy loss (PPO clipped objective)
                ratio = torch.exp(log_prob - old_log_prob_batch)
                surr1 = ratio * advantage_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value losses (for both value heads)
                value_loss_ext = 0.5 * ((value_ext.squeeze() - return_ext_batch) ** 2).mean()
                value_loss_int = 0.5 * ((value_int.squeeze() - return_int_batch) ** 2).mean()
                value_loss = value_loss_ext + value_loss_int

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss +
                    self.value_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                entropy_sum += entropy.mean().item()
                n_updates += 1

        self.buffer.clear()
        self.update_count += 1

        return {
            'total_loss': total_loss / n_updates,
            'policy_loss': policy_loss_sum / n_updates,
            'value_loss': value_loss_sum / n_updates,
            'entropy': entropy_sum / n_updates,
            'avg_reward_int': rewards_int.mean(),
            'avg_reward_ext': rewards_ext.mean(),
            'avg_curiosity': self.curiosity.get_avg_curiosity(),
        }

    def save(self, path: str):
        """Save agent state."""
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'update_count': self.update_count,
        }, path)
        self.curiosity.save(path.replace('.pt', '_curiosity.pt'))

    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.update_count = checkpoint['update_count']
        self.curiosity.load(path.replace('.pt', '_curiosity.pt'))
