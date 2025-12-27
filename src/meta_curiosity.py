"""
VOID_RUNNER - Meta-Curiosity: Learning to be Curious
======================================================
Meta-learning for intrinsic motivation.

Three Key Questions:
-------------------
1. WHAT to be curious about? (Curiosity Transfer)
2. WHEN to be curious? (Adaptive Scheduling)
3. HOW curious to be? (Automatic Coefficient Tuning)

This module explores:
- Transfer of curiosity across environments
- Meta-learning curiosity hyperparameters
- Adaptive curiosity that responds to learning progress
- Universal curiosity features that generalize

Hypothesis: Good curiosity is learnable and transferable.
"Corners are interesting" might generalize across all mazes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from collections import deque
from dataclasses import dataclass
import copy


class UniversalCuriosityEncoder(nn.Module):
    """
    Encoder that learns universal curiosity features.

    Goal: Learn representations that capture "interestingness"
    across different environments.

    Architecture:
    - Shared encoder backbone
    - Curiosity head (predicts novelty)
    - Domain-specific adapters (fine-tune per environment)
    """

    def __init__(
        self,
        input_dim: int,
        feature_dim: int = 128,
        hidden_dim: int = 256,
        n_adapters: int = 4,
    ):
        super().__init__()

        # Shared backbone (frozen after meta-training)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        # Domain adapters (small networks that adapt to new environments)
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, 32),
                nn.ReLU(),
                nn.Linear(32, feature_dim),
            )
            for _ in range(n_adapters)
        ])

        # Adapter selection (which adapter to use)
        self.adapter_selector = nn.Linear(feature_dim, n_adapters)

        # Curiosity prediction head
        self.curiosity_head = nn.Linear(feature_dim, 1)

        self.feature_dim = feature_dim
        self.current_adapter_idx = 0

    def forward(
        self,
        obs: torch.Tensor,
        use_adapter: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: (features, curiosity_score)
        """
        # Backbone features
        features = self.backbone(obs)

        if use_adapter:
            # Soft adapter selection
            adapter_weights = F.softmax(self.adapter_selector(features), dim=-1)

            # Weighted combination of adapters
            adapted = torch.zeros_like(features)
            for i, adapter in enumerate(self.adapters):
                adapted += adapter_weights[:, i:i+1] * adapter(features)

            features = features + adapted  # Residual connection

        curiosity = self.curiosity_head(features)
        return features, curiosity.squeeze(-1)

    def freeze_backbone(self):
        """Freeze backbone after meta-training."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def add_new_adapter(self):
        """Add a new adapter for a new environment."""
        new_adapter = nn.Sequential(
            nn.Linear(self.feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.feature_dim),
        )
        self.adapters.append(new_adapter)
        return len(self.adapters) - 1


class CuriosityTransfer:
    """
    Transfer curiosity across environments.

    Training Protocol:
    1. Meta-train on diverse source environments
    2. Learn universal features of "interestingness"
    3. Fine-tune adapters on target environment

    The key insight: Curiosity features might generalize.
    - "Unexplored areas are interesting"
    - "Novel object configurations are interesting"
    - "State transitions that are hard to predict are interesting"

    These are environment-agnostic intuitions.
    """

    def __init__(
        self,
        obs_dim: int,
        feature_dim: int = 128,
        n_source_envs: int = 4,
        meta_lr: float = 1e-3,
        adapt_lr: float = 1e-4,
        device: str = "cpu",
    ):
        self.device = device
        self.obs_dim = obs_dim
        self.n_source_envs = n_source_envs
        self.adapt_lr = adapt_lr

        # Universal encoder
        self.encoder = UniversalCuriosityEncoder(
            obs_dim, feature_dim, n_adapters=n_source_envs
        ).to(device)

        # RND target (frozen, for ground truth novelty)
        self.rnd_target = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        ).to(device)
        for p in self.rnd_target.parameters():
            p.requires_grad = False

        # RND predictor per source environment
        self.rnd_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim, 256),
                nn.ReLU(),
                nn.Linear(256, feature_dim),
            ).to(device)
            for _ in range(n_source_envs)
        ])

        # Meta optimizer (MAML-style)
        self.meta_optimizer = optim.Adam(
            self.encoder.parameters(),
            lr=meta_lr
        )

        # Per-environment optimizers
        self.env_optimizers = [
            optim.Adam(pred.parameters(), lr=adapt_lr)
            for pred in self.rnd_predictors
        ]

        # Training state
        self.meta_trained = False

    def compute_true_novelty(self, obs: torch.Tensor, env_idx: int) -> torch.Tensor:
        """Ground truth novelty from RND."""
        with torch.no_grad():
            target = self.rnd_target(obs)
        pred = self.rnd_predictors[env_idx](obs)
        return torch.mean((pred - target) ** 2, dim=-1)

    def meta_train_step(
        self,
        env_batches: Dict[int, torch.Tensor],  # env_idx -> observation batch
    ) -> float:
        """
        One step of meta-training across environments.

        Goal: Learn universal curiosity features that work across all envs.
        """
        total_loss = 0.0

        for env_idx, obs in env_batches.items():
            obs = obs.to(self.device)

            # Get predicted curiosity
            _, predicted_curiosity = self.encoder(obs)

            # Get ground truth novelty
            true_novelty = self.compute_true_novelty(obs, env_idx)

            # Loss: predict novelty
            loss = F.mse_loss(predicted_curiosity, true_novelty.detach())
            total_loss += loss

            # Update per-env RND predictor
            self.env_optimizers[env_idx].zero_grad()
            rnd_loss = true_novelty.mean()
            rnd_loss.backward(retain_graph=True)
            self.env_optimizers[env_idx].step()

        # Meta update
        self.meta_optimizer.zero_grad()
        total_loss.backward()
        self.meta_optimizer.step()

        return total_loss.item() / len(env_batches)

    def transfer_to_new_env(
        self,
        target_env_obs: torch.Tensor,
        n_adapt_steps: int = 100,
    ) -> int:
        """
        Transfer to a new target environment.

        1. Freeze backbone
        2. Add new adapter
        3. Fine-tune adapter on target env
        """
        self.encoder.freeze_backbone()

        # Add new adapter
        adapter_idx = self.encoder.add_new_adapter()

        # New RND predictor for target
        new_predictor = nn.Sequential(
            nn.Linear(self.obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.encoder.feature_dim),
        ).to(self.device)
        self.rnd_predictors.append(new_predictor)

        adapt_optimizer = optim.Adam(
            list(self.encoder.adapters[adapter_idx].parameters()) +
            list(new_predictor.parameters()),
            lr=self.adapt_lr
        )

        # Fine-tune on target
        for step in range(n_adapt_steps):
            adapt_optimizer.zero_grad()

            _, predicted = self.encoder(target_env_obs)

            with torch.no_grad():
                target = self.rnd_target(target_env_obs)
            pred = new_predictor(target_env_obs)
            true_novelty = torch.mean((pred - target) ** 2, dim=-1)

            loss = F.mse_loss(predicted, true_novelty.detach()) + true_novelty.mean()
            loss.backward()
            adapt_optimizer.step()

        self.meta_trained = True
        return adapter_idx

    def compute_intrinsic_reward(
        self,
        obs: torch.Tensor,
        use_transfer: bool = True,
    ) -> torch.Tensor:
        """
        Compute intrinsic reward using transferred curiosity.
        """
        _, curiosity = self.encoder(obs, use_adapter=use_transfer)
        return curiosity


@dataclass
class LearningProgress:
    """Tracks learning progress for adaptive curiosity."""
    step: int
    extrinsic_reward: float
    intrinsic_reward: float
    td_error: float
    policy_entropy: float
    exploration_rate: float  # new states / total


class AdaptiveCuriosity:
    """
    Adaptive Curiosity: Learning WHEN to be curious.

    Key insight: Curiosity should adapt to learning phase.
    - Early: Explore everything (high curiosity weight)
    - Mid: Focus on task-relevant novelty
    - Late: Minimal curiosity, exploit learned policy

    This module learns to schedule curiosity automatically.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 128,
        history_length: int = 100,
        device: str = "cpu",
    ):
        self.device = device
        self.history_length = history_length

        # Curiosity coefficient predictor
        # Input: learning progress features
        # Output: optimal curiosity coefficient
        self.coefficient_net = nn.Sequential(
            nn.Linear(6, hidden_dim),  # 6 progress features
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        ).to(device)

        self.optimizer = optim.Adam(self.coefficient_net.parameters(), lr=1e-4)

        # Progress history
        self.progress_history: deque = deque(maxlen=history_length)

        # Baseline curiosity module
        self.base_curiosity: Optional[Any] = None

        # Coefficient bounds
        self.min_coef = 0.001
        self.max_coef = 1.0

    def set_base_curiosity(self, curiosity_module):
        """Set the base curiosity module to be scheduled."""
        self.base_curiosity = curiosity_module

    def record_progress(self, progress: LearningProgress):
        """Record learning progress for coefficient prediction."""
        self.progress_history.append(progress)

    def get_progress_features(self) -> torch.Tensor:
        """Extract features from progress history."""
        if len(self.progress_history) < 10:
            return torch.zeros(6).to(self.device)

        recent = list(self.progress_history)[-50:]

        features = torch.tensor([
            # Reward trends
            np.mean([p.extrinsic_reward for p in recent]),
            np.std([p.extrinsic_reward for p in recent]),
            # Learning progress
            np.mean([p.td_error for p in recent]),
            # Exploration state
            np.mean([p.exploration_rate for p in recent]),
            np.mean([p.policy_entropy for p in recent]),
            # Intrinsic reward trend
            np.mean([p.intrinsic_reward for p in recent]),
        ], dtype=torch.float32).to(self.device)

        return features

    def get_curiosity_coefficient(self) -> float:
        """
        Predict optimal curiosity coefficient based on learning progress.
        """
        features = self.get_progress_features().unsqueeze(0)

        with torch.no_grad():
            coef = self.coefficient_net(features).item()

        # Scale to bounds
        return self.min_coef + coef * (self.max_coef - self.min_coef)

    def compute_intrinsic_reward(
        self,
        obs: torch.Tensor,
        update: bool = True,
    ) -> torch.Tensor:
        """
        Compute scaled intrinsic reward.
        """
        if self.base_curiosity is None:
            raise ValueError("Must set base_curiosity first")

        # Get base intrinsic reward
        base_reward = self.base_curiosity.compute_intrinsic_reward(obs, update=update)

        # Scale by learned coefficient
        coef = self.get_curiosity_coefficient()

        return coef * base_reward

    def train_coefficient_predictor(
        self,
        task_success: float,  # 0-1, how well did the episode go?
    ):
        """
        Train the coefficient predictor based on task success.

        If task succeeded with current coefficient, reinforce.
        If task failed, adjust.
        """
        if len(self.progress_history) < 10:
            return

        features = self.get_progress_features().unsqueeze(0)

        # Current prediction
        predicted_coef = self.coefficient_net(features)

        # Target: if successful, current coef was good
        # If not, we should have explored more (higher coef)
        if task_success > 0.5:
            target_coef = predicted_coef.detach()  # Reinforce current
        else:
            # Should have explored more
            target_coef = torch.clamp(predicted_coef.detach() + 0.1, 0, 1)

        loss = F.mse_loss(predicted_coef, target_coef)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class CuriosityEnsemble:
    """
    Ensemble of curiosity modules with learned mixing.

    Instead of choosing ONE curiosity type, learn to combine:
    - RND (prediction error)
    - Count-based (state visitation)
    - Empowerment (controllability)
    - Episodic (within-episode novelty)

    The mixer learns which curiosity signal is most useful
    at each point in training.
    """

    def __init__(
        self,
        curiosity_modules: List[Any],
        obs_dim: int,
        hidden_dim: int = 128,
        device: str = "cpu",
    ):
        self.device = device
        self.modules = curiosity_modules
        self.n_modules = len(curiosity_modules)

        # Mixer network: learns to weight different curiosity signals
        self.mixer = nn.Sequential(
            nn.Linear(obs_dim + self.n_modules, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_modules),
            nn.Softmax(dim=-1),
        ).to(device)

        self.optimizer = optim.Adam(self.mixer.parameters(), lr=1e-4)

        # Track which module was most useful
        self.module_credits: List[float] = [0.0] * self.n_modules

    def compute_intrinsic_reward(
        self,
        obs: torch.Tensor,
        update: bool = True,
    ) -> torch.Tensor:
        """
        Compute mixed intrinsic reward from ensemble.
        """
        # Get individual rewards
        individual_rewards = []
        for module in self.modules:
            reward = module.compute_intrinsic_reward(obs, update=False)
            if isinstance(reward, torch.Tensor):
                individual_rewards.append(reward)
            else:
                individual_rewards.append(torch.tensor([reward]).to(self.device))

        rewards_tensor = torch.stack(individual_rewards, dim=-1)

        # Get mixing weights
        mixer_input = torch.cat([obs, rewards_tensor], dim=-1)
        weights = self.mixer(mixer_input)

        # Weighted combination
        mixed_reward = (rewards_tensor * weights).sum(dim=-1)

        return mixed_reward

    def update_credits(self, reward_received: float, obs: torch.Tensor):
        """
        Update credit assignment based on task reward.

        Which curiosity module led to this reward?
        """
        with torch.no_grad():
            individual_rewards = []
            for module in self.modules:
                r = module.compute_intrinsic_reward(obs, update=False)
                if isinstance(r, torch.Tensor):
                    individual_rewards.append(r.mean().item())
                else:
                    individual_rewards.append(r)

            # Credit proportional to curiosity * task_reward
            for i, ir in enumerate(individual_rewards):
                self.module_credits[i] += ir * reward_received

    def train_mixer(self, task_reward: float, obs: torch.Tensor):
        """
        Train mixer to weight modules that led to task success.
        """
        # Get individual rewards
        individual_rewards = []
        for module in self.modules:
            reward = module.compute_intrinsic_reward(obs, update=False)
            if isinstance(reward, torch.Tensor):
                individual_rewards.append(reward.detach())
            else:
                individual_rewards.append(torch.tensor([reward]).to(self.device))

        rewards_tensor = torch.stack(individual_rewards, dim=-1)

        # Get mixing weights
        mixer_input = torch.cat([obs, rewards_tensor], dim=-1)
        weights = self.mixer(mixer_input)

        # Target: weight modules proportional to their credit
        credits = torch.tensor(self.module_credits, device=self.device)
        credits = credits / (credits.sum() + 1e-10)
        target_weights = credits.unsqueeze(0).expand(weights.shape)

        loss = F.kl_div(weights.log(), target_weights, reduction='batchmean')

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_module_weights(self) -> Dict[str, float]:
        """Get current importance of each module."""
        total = sum(self.module_credits) + 1e-10
        return {
            f"module_{i}": credit / total
            for i, credit in enumerate(self.module_credits)
        }


class CuriosityScheduler:
    """
    Curriculum-based curiosity scheduling.

    Schedules curiosity through predefined phases:
    1. Pure Exploration: High curiosity, ignore task reward
    2. Balanced: Mix curiosity and task reward
    3. Exploitation: Low curiosity, focus on task

    Unlike AdaptiveCuriosity which learns the schedule,
    this uses a predefined curriculum.
    """

    def __init__(
        self,
        total_steps: int,
        schedule_type: str = "linear",  # linear, cosine, step
        initial_coef: float = 1.0,
        final_coef: float = 0.01,
        warmup_steps: int = 1000,
    ):
        self.total_steps = total_steps
        self.schedule_type = schedule_type
        self.initial_coef = initial_coef
        self.final_coef = final_coef
        self.warmup_steps = warmup_steps

        self.current_step = 0

    def step(self):
        """Advance schedule by one step."""
        self.current_step += 1

    def get_coefficient(self) -> float:
        """Get current curiosity coefficient."""
        if self.current_step < self.warmup_steps:
            # Warmup: linear increase to initial
            return self.initial_coef * (self.current_step / self.warmup_steps)

        # Progress after warmup
        progress = (self.current_step - self.warmup_steps) / (
            self.total_steps - self.warmup_steps + 1e-10
        )
        progress = min(1.0, progress)

        if self.schedule_type == "linear":
            coef = self.initial_coef + progress * (self.final_coef - self.initial_coef)

        elif self.schedule_type == "cosine":
            coef = self.final_coef + 0.5 * (self.initial_coef - self.final_coef) * (
                1 + np.cos(np.pi * progress)
            )

        elif self.schedule_type == "step":
            # Step decay at 33% and 66%
            if progress < 0.33:
                coef = self.initial_coef
            elif progress < 0.66:
                coef = self.initial_coef * 0.3
            else:
                coef = self.final_coef

        else:
            coef = self.initial_coef

        return max(self.final_coef, coef)

    def get_phase(self) -> str:
        """Get current training phase."""
        progress = self.current_step / self.total_steps

        if progress < 0.33:
            return "exploration"
        elif progress < 0.66:
            return "balanced"
        else:
            return "exploitation"
