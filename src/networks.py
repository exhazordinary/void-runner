"""
VOID_RUNNER - Neural Network Architectures
===========================================
Core networks for curiosity-driven exploration using Random Network Distillation.

The key insight: Curiosity = Prediction Error
- A fixed random target network encodes observations
- A trainable predictor network tries to match the target's output
- High prediction error = novel state = intrinsic reward
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np


class ConvEncoder(nn.Module):
    """Convolutional encoder for image-based observations."""

    def __init__(self, input_channels: int = 4, feature_dim: int = 512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(0.2),
            nn.Flatten()
        )

        # Calculate flattened size (assuming 84x84 input)
        self.feature_dim = feature_dim
        self.fc = nn.Linear(64 * 7 * 7, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize to [0, 1]
        x = x.float() / 255.0
        features = self.conv(x)
        return self.fc(features)


class MLPEncoder(nn.Module):
    """MLP encoder for vector-based observations."""

    def __init__(self, input_dim: int, feature_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, feature_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class RNDTargetNetwork(nn.Module):
    """
    Random Network Distillation - Target Network

    This network is NEVER trained. Its random weights create a fixed
    embedding space. The predictor's job is to learn this random function.
    Novel states produce embeddings the predictor hasn't learned to match.
    """

    def __init__(self, input_dim: int, output_dim: int = 512, is_image: bool = False):
        super().__init__()
        self.is_image = is_image

        if is_image:
            self.encoder = ConvEncoder(input_dim, output_dim)
        else:
            self.encoder = MLPEncoder(input_dim, output_dim)

        # Initialize with specific random distribution for stability
        self._init_weights()

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class RNDPredictorNetwork(nn.Module):
    """
    Random Network Distillation - Predictor Network

    This network IS trained to predict the target network's output.
    Prediction error serves as the intrinsic reward signal.

    The predictor is intentionally smaller/simpler than the target,
    creating a learning challenge that correlates with state novelty.
    """

    def __init__(self, input_dim: int, output_dim: int = 512, is_image: bool = False):
        super().__init__()
        self.is_image = is_image

        if is_image:
            self.encoder = ConvEncoder(input_dim, output_dim)
        else:
            self.encoder = MLPEncoder(input_dim, output_dim)

        # Additional prediction head
        self.predictor_head = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.predictor_head(features)


class PolicyNetwork(nn.Module):
    """
    Actor-Critic Policy Network

    Separate value heads for intrinsic and extrinsic rewards,
    enabling different discount factors (gamma_i vs gamma_e).
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        is_image: bool = False,
        continuous: bool = False
    ):
        super().__init__()
        self.continuous = continuous
        self.is_image = is_image

        # Shared feature extractor
        if is_image:
            self.encoder = ConvEncoder(input_dim, hidden_dim)
        else:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )

        # Actor head
        if continuous:
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.actor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )

        # Separate critics for intrinsic and extrinsic values
        self.critic_ext = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.critic_int = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encoder(x.float())

        if self.continuous:
            action_mean = self.actor_mean(features)
            action_std = torch.exp(self.actor_log_std)
            action_logits = (action_mean, action_std)
        else:
            action_logits = self.actor(features)

        value_ext = self.critic_ext(features)
        value_int = self.critic_int(features)

        return action_logits, value_ext, value_int

    def get_action(self, x: torch.Tensor, deterministic: bool = False):
        """Sample action from policy."""
        action_logits, value_ext, value_int = self.forward(x)

        if self.continuous:
            mean, std = action_logits
            if deterministic:
                action = mean
            else:
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
            log_prob = torch.distributions.Normal(mean, std).log_prob(action).sum(-1)
        else:
            probs = F.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            if deterministic:
                action = probs.argmax(dim=-1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)

        return action, log_prob, value_ext, value_int

    def evaluate_actions(self, x: torch.Tensor, actions: torch.Tensor):
        """Evaluate log probability and entropy of actions."""
        action_logits, value_ext, value_int = self.forward(x)

        if self.continuous:
            mean, std = action_logits
            dist = torch.distributions.Normal(mean, std)
            log_prob = dist.log_prob(actions).sum(-1)
            entropy = dist.entropy().sum(-1)
        else:
            probs = F.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(actions)
            entropy = dist.entropy()

        return log_prob, entropy, value_ext, value_int


class RunningMeanStd:
    """
    Welford's online algorithm for computing running mean and variance.
    Used for observation and reward normalization.
    """

    def __init__(self, shape: Tuple[int, ...], epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / np.sqrt(self.var + 1e-8)
