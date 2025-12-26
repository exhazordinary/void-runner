"""
VOID_RUNNER - Multi-Agent Curiosity
====================================
Curiosity-driven exploration in multi-agent settings.

Key challenges:
1. Credit assignment: Which agent caused the novelty?
2. Coordination: Should agents explore together or apart?
3. Stochasticity: Other agents' actions create noise

Approaches implemented:
1. EMC: Episodic Multi-agent with Curiosity
2. CERMIC: Calibrated multi-agent curiosity
3. Joint vs Individual curiosity signals

References:
- Wang et al., "Episodic Multi-agent RL with Curiosity-Driven Exploration"
- NeurIPS 2025: "Wonder Wins Ways: Curiosity through Multi-Agent Contextual Calibration"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque


class IndividualRND(nn.Module):
    """RND module for a single agent."""

    def __init__(self, obs_dim: int, feature_dim: int = 128):
        super().__init__()

        self.target = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

        # Freeze target
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            target = self.target(obs)
        pred = self.predictor(obs)
        return torch.mean((pred - target) ** 2, dim=-1)


class MultiAgentCuriosity:
    """
    Multi-agent curiosity with individual and joint signals.

    Each agent has:
    1. Individual curiosity: Novelty of own observations
    2. Joint curiosity: Novelty of joint observation (coordination)

    Total reward = α * individual + (1-α) * joint

    This encourages both individual exploration AND
    finding novel joint configurations.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        feature_dim: int = 128,
        alpha: float = 0.5,  # Balance individual vs joint
        learning_rate: float = 1e-4,
        device: str = "cpu"
    ):
        self.device = device
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.alpha = alpha

        # Individual curiosity modules (one per agent)
        self.individual_modules = nn.ModuleList([
            IndividualRND(obs_dim, feature_dim).to(device)
            for _ in range(n_agents)
        ])

        # Joint curiosity module (takes concatenated observations)
        self.joint_module = IndividualRND(
            obs_dim * n_agents, feature_dim
        ).to(device)

        # Optimizers
        self.individual_optimizers = [
            optim.Adam(
                self.individual_modules[i].predictor.parameters(),
                lr=learning_rate
            )
            for i in range(n_agents)
        ]

        self.joint_optimizer = optim.Adam(
            self.joint_module.predictor.parameters(),
            lr=learning_rate
        )

    def compute_intrinsic_rewards(
        self,
        observations: Dict[int, torch.Tensor],  # agent_id -> obs
        update: bool = True
    ) -> Dict[int, torch.Tensor]:
        """
        Compute intrinsic rewards for all agents.

        Args:
            observations: Dict mapping agent_id to observations

        Returns:
            Dict mapping agent_id to intrinsic reward
        """
        rewards = {}

        # Individual curiosity
        individual_bonuses = {}
        for agent_id, obs in observations.items():
            bonus = self.individual_modules[agent_id](obs)
            individual_bonuses[agent_id] = bonus

            if update:
                loss = bonus.mean()
                self.individual_optimizers[agent_id].zero_grad()
                loss.backward()
                self.individual_optimizers[agent_id].step()

        # Joint curiosity
        joint_obs = torch.cat(
            [observations[i] for i in range(self.n_agents)],
            dim=-1
        )
        joint_bonus = self.joint_module(joint_obs)

        if update:
            loss = joint_bonus.mean()
            self.joint_optimizer.zero_grad()
            loss.backward()
            self.joint_optimizer.step()

        # Combine individual and joint
        for agent_id in observations:
            rewards[agent_id] = (
                self.alpha * individual_bonuses[agent_id].detach() +
                (1 - self.alpha) * joint_bonus.detach()
            )

        return rewards


class EMC:
    """
    Episodic Multi-agent reinforcement learning with Curiosity.

    Key innovations:
    1. Predict individual Q-values as curiosity signal
    2. Episodic memory to exploit explored experience
    3. Coordinated exploration through Q-value prediction error

    Reference: "Episodic Multi-agent RL with Curiosity-driven Exploration"
    https://arxiv.org/abs/2111.11032
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        memory_size: int = 10000,
        learning_rate: float = 1e-4,
        device: str = "cpu"
    ):
        self.device = device
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Individual Q-networks (target for curiosity)
        self.q_targets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ).to(device)
            for _ in range(n_agents)
        ])

        # Freeze targets
        for target in self.q_targets:
            for param in target.parameters():
                param.requires_grad = False

        # Q-predictors (trained to match targets)
        self.q_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ).to(device)
            for _ in range(n_agents)
        ])

        self.optimizers = [
            optim.Adam(self.q_predictors[i].parameters(), lr=learning_rate)
            for i in range(n_agents)
        ]

        # Episodic memory per agent
        self.memories = [
            deque(maxlen=memory_size)
            for _ in range(n_agents)
        ]

    def _get_input(
        self,
        obs: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        if action.dim() == 1:
            action = F.one_hot(action.long(), self.action_dim).float()
        return torch.cat([obs, action], dim=-1)

    def compute_intrinsic_rewards(
        self,
        observations: Dict[int, torch.Tensor],
        actions: Dict[int, torch.Tensor],
        update: bool = True
    ) -> Dict[int, torch.Tensor]:
        """
        Compute Q-prediction error as curiosity signal.

        High error = novel state-action = explore here!
        """
        rewards = {}

        for agent_id in range(self.n_agents):
            obs = observations[agent_id]
            action = actions[agent_id]
            x = self._get_input(obs, action)

            # Target Q-value (frozen)
            with torch.no_grad():
                target_q = self.q_targets[agent_id](x)

            # Predicted Q-value
            pred_q = self.q_predictors[agent_id](x)

            # Prediction error = curiosity
            error = (pred_q - target_q) ** 2
            rewards[agent_id] = error.squeeze().detach()

            if update:
                loss = error.mean()
                self.optimizers[agent_id].zero_grad()
                loss.backward()
                self.optimizers[agent_id].step()

            # Store in episodic memory
            for o, a, r in zip(obs, action, error):
                self.memories[agent_id].append({
                    'obs': o.detach().cpu().numpy(),
                    'action': a.detach().cpu().numpy(),
                    'curiosity': r.item()
                })

        return rewards

    def sample_from_memory(
        self,
        agent_id: int,
        batch_size: int = 32
    ) -> Optional[List[Dict]]:
        """Sample high-curiosity experiences from memory."""
        memory = self.memories[agent_id]
        if len(memory) < batch_size:
            return None

        # Prioritize high-curiosity samples
        curiosities = np.array([m['curiosity'] for m in memory])
        probs = curiosities / (curiosities.sum() + 1e-8)

        indices = np.random.choice(
            len(memory), size=batch_size, p=probs, replace=False
        )
        return [memory[i] for i in indices]


class CERMICCuriosity:
    """
    Calibrated Multi-Agent Curiosity (CERMIC style).

    Key insight: In multi-agent settings, other agents create
    "apparent" novelty that's just their stochastic behavior.

    Solution: Calibrate curiosity by observing peers.
    If peers also find something novel, it's truly novel.
    If only you find it novel, it's probably noise.

    Reference: "Wonder Wins Ways: Curiosity through Multi-Agent
               Contextual Calibration" (NeurIPS 2025)
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        feature_dim: int = 128,
        calibration_weight: float = 0.3,
        learning_rate: float = 1e-4,
        device: str = "cpu"
    ):
        self.device = device
        self.n_agents = n_agents
        self.calibration_weight = calibration_weight

        # Individual predictors
        self.predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim, 128),
                nn.ReLU(),
                nn.Linear(128, feature_dim)
            ).to(device)
            for _ in range(n_agents)
        ])

        # Shared target (all agents predict the same target)
        self.shared_target = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        ).to(device)

        for param in self.shared_target.parameters():
            param.requires_grad = False

        self.optimizers = [
            optim.Adam(self.predictors[i].parameters(), lr=learning_rate)
            for i in range(n_agents)
        ]

        # Running mean of peer curiosity for calibration
        self.peer_curiosity_mean = np.zeros(n_agents)
        self.peer_curiosity_var = np.ones(n_agents)
        self.update_count = 0

    def compute_intrinsic_rewards(
        self,
        observations: Dict[int, torch.Tensor],
        update: bool = True
    ) -> Dict[int, torch.Tensor]:
        """
        Compute calibrated curiosity rewards.

        1. Get raw curiosity for each agent
        2. Compare with peer curiosity (calibration)
        3. High relative curiosity = true novelty
        """
        raw_curiosities = {}
        rewards = {}

        # Compute raw curiosity for each agent
        for agent_id, obs in observations.items():
            with torch.no_grad():
                target = self.shared_target(obs)
            pred = self.predictors[agent_id](obs)
            error = torch.mean((pred - target) ** 2, dim=-1)
            raw_curiosities[agent_id] = error

            if update:
                loss = error.mean()
                self.optimizers[agent_id].zero_grad()
                loss.backward()
                self.optimizers[agent_id].step()

        # Calibrate using peer curiosity
        mean_curiosity = torch.stack(
            list(raw_curiosities.values())
        ).mean(dim=0)

        for agent_id in observations:
            raw = raw_curiosities[agent_id].detach()

            # Calibration: subtract peer average
            # If everyone finds it novel, it's truly novel
            # If only you find it novel, discount it
            calibrated = raw - self.calibration_weight * (
                raw - mean_curiosity.detach()
            )

            rewards[agent_id] = torch.clamp(calibrated, min=0.0)

        return rewards


class CompetitiveCuriosity:
    """
    Competitive multi-agent exploration.

    Agents compete to discover novel states first.
    Once an agent visits a state, it becomes less valuable
    for other agents (scarcity).

    This naturally leads to diverse exploration patterns.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        hash_dim: int = 32,
        decay_rate: float = 0.5,  # How fast novelty decays after first visit
        device: str = "cpu"
    ):
        self.device = device
        self.n_agents = n_agents
        self.decay_rate = decay_rate

        # Random projection for hashing
        self.projection = np.random.randn(obs_dim, hash_dim)

        # Global novelty map (shared across agents)
        self.global_visits: Dict[str, int] = {}

        # Per-agent discovery tracking
        self.first_discoverer: Dict[str, int] = {}  # hash -> agent_id

    def _hash(self, obs: np.ndarray) -> str:
        proj = obs @ self.projection
        binary = (proj > 0).astype(int)
        return ''.join(map(str, binary))

    def compute_intrinsic_rewards(
        self,
        observations: Dict[int, np.ndarray]
    ) -> Dict[int, float]:
        """
        Compute competitive exploration rewards.

        First discoverer gets full reward.
        Later visitors get decayed reward.
        """
        rewards = {}

        for agent_id, obs in observations.items():
            key = self._hash(obs)

            if key not in self.global_visits:
                # First discovery!
                self.global_visits[key] = 1
                self.first_discoverer[key] = agent_id
                rewards[agent_id] = 1.0  # Full novelty reward
            else:
                # Already visited
                self.global_visits[key] += 1
                visit_count = self.global_visits[key]

                # Decayed reward
                base_reward = 1.0 / np.sqrt(visit_count)

                # Extra decay if someone else discovered first
                if self.first_discoverer[key] != agent_id:
                    base_reward *= self.decay_rate

                rewards[agent_id] = base_reward

        return rewards

    def get_coverage_by_agent(self) -> Dict[int, int]:
        """How many states did each agent discover first?"""
        coverage = {i: 0 for i in range(self.n_agents)}
        for agent_id in self.first_discoverer.values():
            coverage[agent_id] += 1
        return coverage
