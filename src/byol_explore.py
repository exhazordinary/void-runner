"""
VOID_RUNNER - BYOL-Explore: Self-Supervised Curiosity
======================================================
Implementation based on DeepMind's BYOL-Explore paper.

Key insight: Use self-supervised learning as BOTH:
1. World model learning
2. Curiosity signal

The same prediction loss that trains representations
also drives exploration. Elegant unification!

BYOL (Bootstrap Your Own Latent) predicts an older copy
of its own latent representation - no negative samples needed.

BYOL-Explore extends this:
- Predict next state's latent from current state + action
- Prediction error = intrinsic reward
- Bootstrapping prevents representation collapse

Reference: Guo et al., "BYOL-Explore: Exploration by Bootstrapped Prediction"
https://arxiv.org/abs/2206.08332
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional
import copy


class BYOLEncoder(nn.Module):
    """
    BYOL-style encoder with projector.

    Online network: Updated via gradient descent
    Target network: Exponential moving average of online
    """

    def __init__(
        self,
        obs_dim: int,
        repr_dim: int = 256,
        proj_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # Encoder: obs -> representation
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, repr_dim),
        )

        # Projector: representation -> projection (for loss computation)
        self.projector = nn.Sequential(
            nn.Linear(repr_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

        self.repr_dim = repr_dim
        self.proj_dim = proj_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (representation, projection)."""
        representation = self.encoder(x)
        projection = self.projector(representation)
        return representation, projection

    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class BYOLPredictor(nn.Module):
    """
    Predictor network: predicts target projection from online projection.

    This asymmetry (predictor only on online) prevents collapse.
    """

    def __init__(self, proj_dim: int = 128, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WorldModelTransition(nn.Module):
    """
    Latent space transition model: z_{t+1} = f(z_t, a_t)

    Predicts next latent state from current latent and action.
    """

    def __init__(
        self,
        repr_dim: int = 256,
        action_dim: int = 4,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(repr_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, repr_dim),
        )

        self.action_dim = action_dim

    def forward(
        self,
        latent: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        # One-hot encode discrete actions
        if action.dim() == 1:
            action = F.one_hot(action.long(), self.action_dim).float()

        x = torch.cat([latent, action], dim=-1)
        return self.net(x)


class BYOLExplore:
    """
    BYOL-Explore: Self-supervised curiosity.

    Unifies world model learning and exploration in one objective.

    Architecture:
    - Online encoder f_θ: obs -> representation -> projection
    - Target encoder f_ξ: EMA of online (not trained)
    - Predictor q_θ: online projection -> target projection prediction
    - Transition model: latent(t) + action -> latent(t+1)

    Loss = MSE(predictor(online_proj), target_proj)
    Intrinsic reward = this same loss!

    Key insight: High prediction error means:
    1. Novel observation (representation learning happening)
    2. Should explore here more (curiosity signal)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        repr_dim: int = 256,
        proj_dim: int = 128,
        hidden_dim: int = 256,
        learning_rate: float = 1e-4,
        ema_tau: float = 0.99,  # EMA decay for target network
        device: str = "cpu",
    ):
        self.device = device
        self.ema_tau = ema_tau
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Online network (trained)
        self.online_encoder = BYOLEncoder(
            obs_dim, repr_dim, proj_dim, hidden_dim
        ).to(device)

        # Target network (EMA of online - NOT trained)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Predictor (asymmetric - only on online side)
        self.predictor = BYOLPredictor(proj_dim, hidden_dim).to(device)

        # Transition model for next-state prediction
        self.transition = WorldModelTransition(
            repr_dim, action_dim, hidden_dim
        ).to(device)

        # Optimizer for all trainable components
        self.optimizer = optim.Adam(
            list(self.online_encoder.parameters()) +
            list(self.predictor.parameters()) +
            list(self.transition.parameters()),
            lr=learning_rate
        )

        # Statistics
        self.total_loss = 0.0
        self.update_count = 0

    @torch.no_grad()
    def _update_target(self):
        """Update target network with EMA."""
        for online_param, target_param in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            target_param.data = (
                self.ema_tau * target_param.data +
                (1 - self.ema_tau) * online_param.data
            )

    def _byol_loss(
        self,
        online_proj: torch.Tensor,
        target_proj: torch.Tensor
    ) -> torch.Tensor:
        """
        BYOL loss: cosine similarity after prediction.

        Loss = 2 - 2 * cos_sim(predictor(online), target)
        """
        predicted = self.predictor(online_proj)

        # L2 normalize
        predicted = F.normalize(predicted, dim=-1)
        target = F.normalize(target_proj, dim=-1)

        # Cosine similarity loss
        loss = 2 - 2 * (predicted * target).sum(dim=-1)
        return loss

    def compute_intrinsic_reward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        update: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute intrinsic reward as BYOL prediction error.

        The same loss that trains the world model = curiosity signal.
        """
        # Get online representations and projections
        online_repr, online_proj = self.online_encoder(obs)

        # Get target projections for next_obs (no grad)
        with torch.no_grad():
            _, target_proj_next = self.target_encoder(next_obs)

        # Predict next latent state
        predicted_next_repr = self.transition(online_repr, action)

        # Get projection of predicted next state
        predicted_next_proj = self.online_encoder.projector(predicted_next_repr)

        # BYOL loss between predicted next and actual next (target)
        byol_loss = self._byol_loss(predicted_next_proj, target_proj_next)

        # Intrinsic reward = prediction error
        intrinsic_reward = byol_loss.detach()

        if update:
            # Full BYOL loss for training
            loss = byol_loss.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.online_encoder.parameters()) +
                list(self.predictor.parameters()) +
                list(self.transition.parameters()),
                1.0
            )
            self.optimizer.step()

            # Update target network
            self._update_target()

            self.total_loss += loss.item()
            self.update_count += 1

        info = {
            'byol_loss': byol_loss.mean().item(),
            'intrinsic_reward': intrinsic_reward.mean().item(),
        }

        return intrinsic_reward, info

    def get_representation(self, obs: torch.Tensor) -> torch.Tensor:
        """Get learned representation for downstream use."""
        return self.online_encoder.get_representation(obs)

    def get_avg_loss(self) -> float:
        if self.update_count == 0:
            return 0.0
        return self.total_loss / self.update_count

    def reset_stats(self):
        self.total_loss = 0.0
        self.update_count = 0


class WorldModelCuriosity:
    """
    World Model-based curiosity with uncertainty estimation.

    Uses an ensemble of transition models to estimate epistemic
    uncertainty. High disagreement = high uncertainty = curious!

    This is related to BYOL-Explore but uses explicit ensembles
    rather than bootstrapped prediction.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_models: int = 5,
        learning_rate: float = 1e-4,
        device: str = "cpu"
    ):
        self.device = device
        self.n_models = n_models

        # Ensemble of world models
        self.models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, obs_dim)
            ).to(device)
            for _ in range(n_models)
        ])

        self.optimizers = [
            optim.Adam(model.parameters(), lr=learning_rate)
            for model in self.models
        ]

        self.action_dim = action_dim

    def predict(
        self,
        obs: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next state with uncertainty.

        Returns:
            mean_prediction: Average prediction across ensemble
            uncertainty: Variance across ensemble (epistemic uncertainty)
        """
        if action.dim() == 1:
            action = F.one_hot(action.long(), self.action_dim).float()

        x = torch.cat([obs, action], dim=-1)

        predictions = torch.stack([model(x) for model in self.models])

        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.var(dim=0).mean(dim=-1)

        return mean_pred, uncertainty

    def compute_intrinsic_reward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        update: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Intrinsic reward = ensemble disagreement (epistemic uncertainty).
        """
        _, uncertainty = self.predict(obs, action)

        if update:
            if action.dim() == 1:
                action_oh = F.one_hot(action.long(), self.action_dim).float()
            else:
                action_oh = action

            x = torch.cat([obs, action_oh], dim=-1)

            for model, optimizer in zip(self.models, self.optimizers):
                pred = model(x)
                loss = F.mse_loss(pred, next_obs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        info = {
            'uncertainty': uncertainty.mean().item(),
        }

        return uncertainty.detach(), info
