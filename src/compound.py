"""
VOID_RUNNER - Compound Curiosity System
========================================
Multi-faceted intrinsic motivation with adaptive scheduling.

The core thesis: No single curiosity signal is optimal for all situations.

Different curiosity types excel at different things:
- RND: Discovers novel states quickly
- Empowerment: Finds positions of control/influence
- Episodic: Enables revisiting and environment change detection
- Causal: Understands cause-effect relationships

This module combines them with an adaptive meta-controller that learns
WHEN to use each type of curiosity based on the agent's current situation.

Inspired by the idea that biological curiosity is multi-faceted:
- Sometimes we're curious about novelty ("what's that?")
- Sometimes about control ("can I affect that?")
- Sometimes about understanding ("how does that work?")
- Sometimes about memory ("wasn't there something interesting here before?")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional, List
from enum import Enum
from dataclasses import dataclass

from .curiosity import CuriosityModule
from .empowerment import EmpowermentModule, CausalCuriosity
from .episodic import EpisodicCuriosityModule


class CuriosityType(Enum):
    """Types of curiosity signals."""
    NOVELTY = "novelty"          # RND - prediction error
    EMPOWERMENT = "empowerment"  # Control/influence
    EPISODIC = "episodic"        # Memory-based
    CAUSAL = "causal"            # Cause-effect understanding


@dataclass
class CuriositySignal:
    """A single curiosity signal with metadata."""
    value: torch.Tensor
    type: CuriosityType
    confidence: float = 1.0
    info: Dict = None


class CuriosityScheduler(nn.Module):
    """
    Meta-controller that learns to weight different curiosity signals.

    Learns a policy over curiosity types based on:
    - Current state features
    - Recent exploration progress
    - Historical success of each curiosity type
    """

    def __init__(
        self,
        obs_dim: int,
        n_curiosity_types: int = 4,
        hidden_dim: int = 128,
        context_dim: int = 32,
    ):
        super().__init__()
        self.n_types = n_curiosity_types

        # Context encoder: summarizes recent exploration history
        self.context_encoder = nn.GRU(
            input_size=obs_dim + n_curiosity_types + 1,  # obs + curiosities + reward
            hidden_size=context_dim,
            batch_first=True
        )

        # Weight predictor: outputs mixing weights for curiosity types
        self.weight_net = nn.Sequential(
            nn.Linear(obs_dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_curiosity_types)
        )

        # Track which curiosity types led to discoveries
        self.success_counts = torch.ones(n_curiosity_types)
        self.total_counts = torch.ones(n_curiosity_types)

    def forward(
        self,
        obs: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute mixing weights for curiosity types.

        Returns:
            weights: [batch_size, n_curiosity_types] softmax weights
        """
        if context is None:
            context = torch.zeros(obs.shape[0], 32, device=obs.device)

        x = torch.cat([obs, context], dim=-1)
        logits = self.weight_net(x)

        # Add success-rate prior
        success_rate = self.success_counts / self.total_counts
        logits = logits + torch.log(success_rate + 1e-8).to(logits.device)

        return F.softmax(logits, dim=-1)

    def update_success(self, curiosity_type: int, success: bool):
        """Update success tracking for a curiosity type."""
        self.total_counts[curiosity_type] += 1
        if success:
            self.success_counts[curiosity_type] += 1


class CompoundCuriosity:
    """
    Multi-faceted curiosity system with adaptive mixing.

    Combines:
    1. Novelty (RND) - "What's new?"
    2. Empowerment - "What can I control?"
    3. Episodic - "What do I remember?"
    4. Causal - "What causes what?"

    With a learned scheduler that adapts the mix based on context.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        learning_rate: float = 1e-4,
        device: str = "cpu",
        enable_novelty: bool = True,
        enable_empowerment: bool = True,
        enable_episodic: bool = True,
        enable_causal: bool = True,
    ):
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Initialize curiosity modules
        self.modules = {}

        if enable_novelty:
            self.modules[CuriosityType.NOVELTY] = CuriosityModule(
                obs_dim, feature_dim=256, learning_rate=learning_rate, device=device
            )

        if enable_empowerment:
            self.modules[CuriosityType.EMPOWERMENT] = EmpowermentModule(
                obs_dim, action_dim, hidden_dim=hidden_dim,
                learning_rate=learning_rate, device=device
            )

        if enable_episodic:
            self.modules[CuriosityType.EPISODIC] = EpisodicCuriosityModule(
                obs_dim, hidden_dim=hidden_dim,
                learning_rate=learning_rate, device=device
            )

        if enable_causal:
            self.modules[CuriosityType.CAUSAL] = CausalCuriosity(
                obs_dim, action_dim, hidden_dim=hidden_dim,
                learning_rate=learning_rate, device=device
            )

        # Curiosity scheduler
        self.scheduler = CuriosityScheduler(
            obs_dim, n_curiosity_types=len(self.modules)
        ).to(device)

        self.scheduler_optimizer = optim.Adam(
            self.scheduler.parameters(), lr=learning_rate
        )

        # Context buffer for scheduler
        self.context_buffer = []
        self.context_hidden = None

        # Statistics
        self.curiosity_stats = {t: [] for t in self.modules.keys()}

    def compute_intrinsic_reward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        update: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute compound intrinsic reward from all curiosity sources.

        Returns weighted combination based on scheduler.
        """
        signals = {}
        info = {}

        # Compute each curiosity signal
        for ctype, module in self.modules.items():
            if ctype == CuriosityType.NOVELTY:
                # RND needs to update outside of no_grad context
                signal = module.compute_intrinsic_reward(next_obs, update=False)
                signals[ctype] = signal
                info['novelty'] = signal.mean().item()

            elif ctype == CuriosityType.EMPOWERMENT:
                signal, emp_info = module.compute_intrinsic_reward(
                    obs, action, next_obs, update=False
                )
                signals[ctype] = signal
                info.update({f'emp_{k}': v for k, v in emp_info.items()})

            elif ctype == CuriosityType.EPISODIC:
                signal, epi_info = module.compute_intrinsic_reward(
                    obs, next_obs, update=False
                )
                signals[ctype] = signal
                info.update({f'epi_{k}': v for k, v in epi_info.items()})

            elif ctype == CuriosityType.CAUSAL:
                signal = module.compute_causal_curiosity(
                    obs, action, next_obs, update=False
                )
                signals[ctype] = signal
                info['causal'] = signal.mean().item()

        # Get mixing weights from scheduler
        with torch.no_grad():
            weights = self.scheduler(obs)

        # Combine signals
        combined = torch.zeros(obs.shape[0], device=self.device)
        for i, ctype in enumerate(self.modules.keys()):
            if ctype in signals:
                combined += weights[:, i] * signals[ctype]
                self.curiosity_stats[ctype].append(signals[ctype].mean().item())

        # Store weights in info
        for i, ctype in enumerate(self.modules.keys()):
            info[f'weight_{ctype.value}'] = weights[:, i].mean().item()

        return combined, info

    def get_dominant_curiosity(self, obs: torch.Tensor) -> CuriosityType:
        """Get which curiosity type is currently dominant."""
        with torch.no_grad():
            weights = self.scheduler(obs)
            dominant_idx = weights.mean(dim=0).argmax().item()
            return list(self.modules.keys())[dominant_idx]

    def update_scheduler(self, reward: float, curiosity_type: CuriosityType):
        """
        Update scheduler based on exploration outcome.

        If following a particular curiosity type led to external reward,
        increase its weight.
        """
        type_idx = list(self.modules.keys()).index(curiosity_type)
        success = reward > 0
        self.scheduler.update_success(type_idx, success)

    def get_statistics(self) -> Dict:
        """Get curiosity statistics."""
        stats = {}
        for ctype, values in self.curiosity_stats.items():
            if values:
                stats[f'avg_{ctype.value}'] = np.mean(values[-100:])
        return stats

    def reset_stats(self):
        """Reset curiosity statistics."""
        self.curiosity_stats = {t: [] for t in self.modules.keys()}
        for module in self.modules.values():
            if hasattr(module, 'reset_stats'):
                module.reset_stats()


class AdversarialCuriosity:
    """
    Self-play curiosity with goal generation.

    Two-player game:
    1. Generator: Proposes challenging but achievable goals
    2. Agent: Tries to reach proposed goals

    The generator learns to propose goals at the "frontier" of
    the agent's capabilities - hard enough to be interesting,
    easy enough to be achievable.

    This creates automatic curriculum learning for exploration.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 256,
        goal_dim: int = None,
        learning_rate: float = 1e-4,
        success_threshold: float = 0.5,  # Target success rate
        device: str = "cpu"
    ):
        self.device = device
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim or obs_dim
        self.success_threshold = success_threshold

        # Goal generator: proposes target states
        self.generator = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        ).to(device)

        self.goal_mean = nn.Linear(hidden_dim, self.goal_dim).to(device)
        self.goal_log_std = nn.Linear(hidden_dim, self.goal_dim).to(device)

        # Discriminator: predicts if agent can reach goal
        self.discriminator = nn.Sequential(
            nn.Linear(obs_dim + self.goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        ).to(device)

        self.gen_optimizer = optim.Adam(
            list(self.generator.parameters()) +
            list(self.goal_mean.parameters()) +
            list(self.goal_log_std.parameters()),
            lr=learning_rate
        )

        self.disc_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=learning_rate
        )

        # Track goal attempts
        self.goal_history = []
        self.current_goal = None

    def generate_goal(self, obs: torch.Tensor) -> torch.Tensor:
        """Generate a challenging goal for the agent."""
        features = self.generator(obs)
        mean = self.goal_mean(features)
        log_std = torch.clamp(self.goal_log_std(features), -5, 2)
        std = torch.exp(log_std)

        # Sample goal
        goal = mean + std * torch.randn_like(mean)
        self.current_goal = goal.detach()

        return goal

    def compute_goal_reward(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor,
        threshold: float = 1.0
    ) -> torch.Tensor:
        """Reward for getting close to goal."""
        distance = torch.norm(obs - goal, dim=-1)
        reward = torch.exp(-distance / threshold)
        return reward

    def update(
        self,
        start_obs: torch.Tensor,
        goal: torch.Tensor,
        reached: bool
    ):
        """Update generator and discriminator based on attempt."""
        # Update discriminator (predict reachability)
        disc_input = torch.cat([start_obs, goal], dim=-1)
        pred_reachable = self.discriminator(disc_input)
        disc_target = torch.tensor([[float(reached)]], device=self.device)

        disc_loss = F.binary_cross_entropy(pred_reachable, disc_target)

        self.disc_optimizer.zero_grad()
        disc_loss.backward()
        self.disc_optimizer.step()

        # Update generator (target success rate at threshold)
        pred_reachable = self.discriminator(disc_input.detach())

        # Generator wants goals at ~50% success rate (challenging but achievable)
        target_difficulty = torch.tensor([[self.success_threshold]], device=self.device)
        gen_loss = F.mse_loss(pred_reachable, target_difficulty)

        self.gen_optimizer.zero_grad()
        gen_loss.backward()
        self.gen_optimizer.step()

        self.goal_history.append({
            'goal': goal.detach().cpu().numpy(),
            'reached': reached
        })

    def get_success_rate(self, last_n: int = 100) -> float:
        """Get recent goal success rate."""
        if not self.goal_history:
            return 0.5
        recent = self.goal_history[-last_n:]
        return sum(g['reached'] for g in recent) / len(recent)
