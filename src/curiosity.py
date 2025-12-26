"""
VOID_RUNNER - Curiosity Module
==============================
Random Network Distillation (RND) for intrinsic motivation.

The core idea is beautifully simple:
1. A random network maps observations to embeddings (never trained)
2. A predictor network learns to match these embeddings
3. Prediction error = curiosity = intrinsic reward

High error means: "I haven't seen this before" -> Explore!
Low error means: "I know this place" -> Move on to novelty

Reference: Burda et al., "Exploration by Random Network Distillation" (2018)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional
from .networks import RNDTargetNetwork, RNDPredictorNetwork, RunningMeanStd


class CuriosityModule:
    """
    Random Network Distillation curiosity module.

    Generates intrinsic rewards based on how surprising/novel
    an observation is to the agent.
    """

    def __init__(
        self,
        input_dim: int,
        feature_dim: int = 512,
        learning_rate: float = 1e-4,
        is_image: bool = False,
        device: str = "cpu",
        update_proportion: float = 0.25,  # Fraction of experiences used for RND update
    ):
        self.device = device
        self.update_proportion = update_proportion
        self.is_image = is_image

        # Initialize target (frozen) and predictor (trainable) networks
        self.target = RNDTargetNetwork(
            input_dim, feature_dim, is_image
        ).to(device)

        self.predictor = RNDPredictorNetwork(
            input_dim, feature_dim, is_image
        ).to(device)

        self.optimizer = optim.Adam(
            self.predictor.parameters(),
            lr=learning_rate
        )

        # Running statistics for reward normalization
        self.reward_rms = RunningMeanStd(shape=())
        self.obs_rms = RunningMeanStd(shape=(input_dim,) if not is_image else (1,))

        # Track curiosity statistics
        self.total_curiosity = 0
        self.curiosity_count = 0

    def compute_intrinsic_reward(
        self,
        next_obs: torch.Tensor,
        update: bool = True
    ) -> torch.Tensor:
        """
        Compute intrinsic reward as prediction error.

        Args:
            next_obs: Next observation(s) [batch_size, *obs_shape]
            update: Whether to update the predictor network

        Returns:
            Intrinsic rewards [batch_size]
        """
        with torch.no_grad():
            target_features = self.target(next_obs)

        predictor_features = self.predictor(next_obs)

        # MSE between predictor and target features
        intrinsic_reward = torch.mean(
            (predictor_features - target_features) ** 2,
            dim=-1
        )

        # Update predictor to reduce error (this is the learning!)
        if update:
            # Only update on a subset to prevent over-fitting
            mask = torch.rand(intrinsic_reward.shape[0]) < self.update_proportion
            if mask.sum() > 0:
                loss = intrinsic_reward[mask.to(self.device)].mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
                self.optimizer.step()

        # Track statistics
        self.total_curiosity += intrinsic_reward.detach().mean().item()
        self.curiosity_count += 1

        return intrinsic_reward.detach()

    def normalize_reward(self, reward: np.ndarray) -> np.ndarray:
        """Normalize intrinsic rewards using running statistics."""
        self.reward_rms.update(reward.flatten())
        return reward / np.sqrt(self.reward_rms.var + 1e-8)

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observations using running statistics."""
        if not self.is_image:
            self.obs_rms.update(obs)
            return self.obs_rms.normalize(obs)
        return obs / 255.0

    def get_avg_curiosity(self) -> float:
        """Get average curiosity since last reset."""
        if self.curiosity_count == 0:
            return 0.0
        return self.total_curiosity / self.curiosity_count

    def reset_stats(self):
        """Reset curiosity statistics."""
        self.total_curiosity = 0
        self.curiosity_count = 0

    def save(self, path: str):
        """Save predictor network."""
        torch.save({
            'predictor': self.predictor.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'reward_rms_mean': self.reward_rms.mean,
            'reward_rms_var': self.reward_rms.var,
            'obs_rms_mean': self.obs_rms.mean,
            'obs_rms_var': self.obs_rms.var,
        }, path)

    def load(self, path: str):
        """Load predictor network."""
        checkpoint = torch.load(path, map_location=self.device)
        self.predictor.load_state_dict(checkpoint['predictor'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.reward_rms.mean = checkpoint['reward_rms_mean']
        self.reward_rms.var = checkpoint['reward_rms_var']
        self.obs_rms.mean = checkpoint['obs_rms_mean']
        self.obs_rms.var = checkpoint['obs_rms_var']


class ICMModule:
    """
    Intrinsic Curiosity Module (ICM) - Alternative curiosity formulation.

    Uses forward dynamics prediction error as curiosity signal.
    Learns in a feature space to focus on controllable aspects.

    Reference: Pathak et al., "Curiosity-driven Exploration by
               Self-Supervised Prediction" (2017)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        feature_dim: int = 256,
        learning_rate: float = 1e-4,
        beta: float = 0.2,  # Weight for forward vs inverse loss
        device: str = "cpu"
    ):
        self.device = device
        self.beta = beta
        self.action_dim = action_dim

        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        ).to(device)

        # Inverse model: predict action from (s, s')
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        ).to(device)

        # Forward model: predict s' from (s, a)
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        ).to(device)

        params = (
            list(self.encoder.parameters()) +
            list(self.inverse_model.parameters()) +
            list(self.forward_model.parameters())
        )
        self.optimizer = optim.Adam(params, lr=learning_rate)

    def compute_intrinsic_reward(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        update: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute intrinsic reward as forward prediction error.

        Returns:
            intrinsic_reward: Prediction error in feature space
            info: Dict with losses for logging
        """
        # Encode observations
        phi_s = self.encoder(obs.float())
        phi_s_next = self.encoder(next_obs.float())

        # One-hot encode discrete actions
        if len(action.shape) == 1:
            action_onehot = torch.zeros(action.shape[0], self.action_dim).to(self.device)
            action_onehot.scatter_(1, action.unsqueeze(1).long(), 1)
        else:
            action_onehot = action

        # Forward model prediction
        forward_input = torch.cat([phi_s, action_onehot], dim=-1)
        phi_s_next_pred = self.forward_model(forward_input)

        # Inverse model prediction
        inverse_input = torch.cat([phi_s, phi_s_next], dim=-1)
        action_pred = self.inverse_model(inverse_input)

        # Losses
        forward_loss = 0.5 * torch.mean((phi_s_next_pred - phi_s_next.detach()) ** 2, dim=-1)
        inverse_loss = nn.functional.cross_entropy(action_pred, action.long(), reduction='none')

        # Intrinsic reward is forward prediction error
        intrinsic_reward = forward_loss.detach()

        if update:
            total_loss = (1 - self.beta) * inverse_loss.mean() + self.beta * forward_loss.mean()
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) +
                list(self.inverse_model.parameters()) +
                list(self.forward_model.parameters()),
                1.0
            )
            self.optimizer.step()

        return intrinsic_reward, {
            'forward_loss': forward_loss.mean().item(),
            'inverse_loss': inverse_loss.mean().item()
        }
