"""
VOID_RUNNER - Count-Based Exploration Methods
==============================================
Pseudo-count and hash-based visitation counting.

The fundamental insight: Exploration bonus ∝ 1/√(visit_count)

But counting is hard in continuous/high-dimensional spaces!
Solutions:
1. Hash states to discrete buckets (SimHash, Locality-Sensitive Hashing)
2. Learn a density model and derive pseudo-counts
3. Use neural network prediction error as implicit count

This module implements multiple counting strategies.

References:
- Bellemare et al., "Unifying Count-Based Exploration and Intrinsic Motivation"
- Tang et al., "#Exploration: A Study of Count-Based Exploration for Deep RL"
- VCSAP: "Online RL exploration based on visitation count of state-action pairs"
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional
from collections import defaultdict
import hashlib


class SimHashCounter:
    """
    Locality-Sensitive Hashing for state counting.

    SimHash maps high-dimensional states to binary codes.
    Similar states get similar codes, enabling efficient counting.

    This is the "#Exploration" approach from Tang et al.
    """

    def __init__(
        self,
        obs_dim: int,
        hash_dim: int = 32,  # Number of hash bits
        decay: float = 0.999,  # Count decay for non-stationary envs
    ):
        self.obs_dim = obs_dim
        self.hash_dim = hash_dim
        self.decay = decay

        # Random projection matrix for SimHash
        self.projection = np.random.randn(obs_dim, hash_dim)
        self.projection /= np.linalg.norm(self.projection, axis=0, keepdims=True)

        # Count table
        self.counts: Dict[str, float] = defaultdict(float)
        self.total_count = 0

    def _hash(self, obs: np.ndarray) -> str:
        """Compute SimHash of observation."""
        # Project and binarize
        projection = obs @ self.projection
        binary = (projection > 0).astype(int)

        # Convert to string key
        return ''.join(map(str, binary))

    def get_count(self, obs: np.ndarray) -> float:
        """Get (possibly decayed) visit count."""
        key = self._hash(obs)
        return self.counts[key]

    def increment(self, obs: np.ndarray) -> float:
        """Increment count and return bonus."""
        key = self._hash(obs)
        self.counts[key] += 1
        self.total_count += 1

        # Exploration bonus: 1/sqrt(count)
        count = self.counts[key]
        bonus = 1.0 / np.sqrt(count)

        return bonus

    def compute_bonus_batch(self, obs: np.ndarray) -> np.ndarray:
        """Compute exploration bonus for batch of observations."""
        bonuses = []
        for o in obs:
            bonus = self.increment(o)
            bonuses.append(bonus)
        return np.array(bonuses)

    def decay_counts(self):
        """Apply decay to all counts (for non-stationary environments)."""
        for key in self.counts:
            self.counts[key] *= self.decay


class StateActionCounter:
    """
    VCSAP-style counting of state-action pairs.

    Key insight: Counting only states leads to over-exploration
    of certain state-action pairs. We should also count which
    ACTIONS were taken in each state.

    Bonus = β₁/√N(s) + β₂/√N(s,a)

    Where N(s) is state count and N(s,a) is state-action count.

    Reference: "VCSAP: Online RL exploration based on visitation
               count of state-action pairs"
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hash_dim: int = 32,
        beta_state: float = 0.5,
        beta_action: float = 0.5,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hash_dim = hash_dim
        self.beta_state = beta_state
        self.beta_action = beta_action

        # Random projections
        self.state_projection = np.random.randn(obs_dim, hash_dim)
        self.state_projection /= np.linalg.norm(self.state_projection, axis=0, keepdims=True)

        # Counts
        self.state_counts: Dict[str, float] = defaultdict(float)
        self.state_action_counts: Dict[str, float] = defaultdict(float)

    def _hash_state(self, obs: np.ndarray) -> str:
        projection = obs @ self.state_projection
        binary = (projection > 0).astype(int)
        return ''.join(map(str, binary))

    def _hash_state_action(self, obs: np.ndarray, action: int) -> str:
        state_hash = self._hash_state(obs)
        return f"{state_hash}_{action}"

    def compute_bonus(self, obs: np.ndarray, action: int) -> float:
        """Compute exploration bonus for state-action pair."""
        state_key = self._hash_state(obs)
        sa_key = self._hash_state_action(obs, action)

        # Increment counts
        self.state_counts[state_key] += 1
        self.state_action_counts[sa_key] += 1

        # Compute bonuses
        state_bonus = self.beta_state / np.sqrt(self.state_counts[state_key])
        action_bonus = self.beta_action / np.sqrt(self.state_action_counts[sa_key])

        return state_bonus + action_bonus

    def compute_bonus_batch(
        self,
        obs: np.ndarray,
        actions: np.ndarray
    ) -> np.ndarray:
        """Compute bonuses for batch."""
        bonuses = []
        for o, a in zip(obs, actions):
            bonus = self.compute_bonus(o, int(a))
            bonuses.append(bonus)
        return np.array(bonuses)


class NeuralDensityModel(nn.Module):
    """
    Learn a density model for pseudo-count estimation.

    The idea: Train a model to predict P(s).
    Pseudo-count is derived from how much the model's
    prediction changes after seeing s.

    n̂(s) = ρ(s)(1 - ρ'(s)) / (ρ'(s) - ρ(s))

    Where ρ(s) is density before update, ρ'(s) is after.

    Reference: Bellemare et al., "Unifying Count-Based Exploration
               and Intrinsic Motivation"
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 256,
        n_components: int = 32,
    ):
        super().__init__()

        # Gaussian mixture density model
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Output mixture parameters
        self.mixture_weights = nn.Linear(hidden_dim, n_components)
        self.mixture_means = nn.Linear(hidden_dim, n_components * obs_dim)
        self.mixture_log_vars = nn.Linear(hidden_dim, n_components * obs_dim)

        self.obs_dim = obs_dim
        self.n_components = n_components

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log density."""
        features = self.encoder(x)

        # Get mixture parameters
        weights = torch.softmax(self.mixture_weights(features), dim=-1)
        means = self.mixture_means(features).view(-1, self.n_components, self.obs_dim)
        log_vars = self.mixture_log_vars(features).view(-1, self.n_components, self.obs_dim)
        log_vars = torch.clamp(log_vars, -10, 2)

        # Compute Gaussian log probabilities
        x_expanded = x.unsqueeze(1).expand_as(means)
        diff = x_expanded - means
        log_probs = -0.5 * (log_vars + diff ** 2 / torch.exp(log_vars)).sum(dim=-1)

        # Mixture log probability
        log_density = torch.logsumexp(
            torch.log(weights + 1e-10) + log_probs,
            dim=-1
        )

        return log_density


class PseudoCountModule:
    """
    Pseudo-count exploration with neural density model.

    Computes exploration bonus ∝ 1/√n̂(s) where n̂(s) is
    the pseudo-count derived from density changes.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 256,
        learning_rate: float = 1e-4,
        device: str = "cpu",
    ):
        self.device = device

        self.density_model = NeuralDensityModel(
            obs_dim, hidden_dim
        ).to(device)

        self.optimizer = optim.Adam(
            self.density_model.parameters(),
            lr=learning_rate
        )

        # Store previous density for pseudo-count computation
        self.prev_density = None

    def compute_pseudo_count(
        self,
        obs: torch.Tensor,
        update: bool = True
    ) -> torch.Tensor:
        """
        Compute pseudo-count based on density change.

        n̂(s) ≈ ρ(s) / (ρ'(s) - ρ(s))

        Where ρ is density before update, ρ' is after.
        """
        # Get current density
        log_density_before = self.density_model(obs)
        density_before = torch.exp(log_density_before)

        if update:
            # Update density model
            loss = -log_density_before.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Get density after update
            with torch.no_grad():
                log_density_after = self.density_model(obs)
                density_after = torch.exp(log_density_after)

            # Compute pseudo-count
            # n̂ = ρ(1-ρ') / (ρ' - ρ)
            density_change = density_after - density_before.detach()
            pseudo_count = density_before.detach() / (density_change + 1e-8)
            pseudo_count = torch.clamp(pseudo_count, min=0.0, max=1e6)
        else:
            pseudo_count = torch.ones_like(density_before)

        return pseudo_count

    def compute_bonus(
        self,
        obs: torch.Tensor,
        update: bool = True
    ) -> torch.Tensor:
        """Compute exploration bonus from pseudo-count."""
        pseudo_count = self.compute_pseudo_count(obs, update)
        bonus = 1.0 / torch.sqrt(pseudo_count + 0.01)
        return bonus


class GoExploreArchive:
    """
    Go-Explore style state archive.

    Key insight: Maintain an archive of interesting states.
    Periodically "teleport" to archive states and explore from there.

    This enables "remember and return" exploration.

    Reference: Ecoffet et al., "First return then explore"
    """

    def __init__(
        self,
        obs_dim: int,
        max_size: int = 10000,
        hash_dim: int = 32,
        novelty_threshold: float = 0.1,
    ):
        self.obs_dim = obs_dim
        self.max_size = max_size
        self.hash_dim = hash_dim
        self.novelty_threshold = novelty_threshold

        # Random projection for hashing
        self.projection = np.random.randn(obs_dim, hash_dim)

        # Archive: hash -> (state, trajectory_to_reach, score)
        self.archive: Dict[str, Dict] = {}

        # Visit counts per cell
        self.visit_counts: Dict[str, int] = defaultdict(int)

    def _hash(self, obs: np.ndarray) -> str:
        projection = obs @ self.projection
        binary = (projection > 0).astype(int)
        return ''.join(map(str, binary))

    def add(
        self,
        obs: np.ndarray,
        trajectory: Optional[list] = None,
        score: float = 0.0
    ) -> bool:
        """
        Add state to archive if novel enough.

        Returns True if state was added.
        """
        key = self._hash(obs)
        self.visit_counts[key] += 1

        # Check if we should add/update
        if key not in self.archive:
            # New cell - add it
            self.archive[key] = {
                'state': obs.copy(),
                'trajectory': trajectory,
                'score': score,
                'count': 1
            }
            return True
        else:
            # Existing cell - update if better score
            if score > self.archive[key]['score']:
                self.archive[key]['state'] = obs.copy()
                self.archive[key]['trajectory'] = trajectory
                self.archive[key]['score'] = score
            self.archive[key]['count'] += 1
            return False

    def sample_state(self, strategy: str = "count") -> Optional[np.ndarray]:
        """
        Sample a state from archive for exploration.

        Strategies:
        - "count": Prefer less-visited cells
        - "score": Prefer high-score cells
        - "random": Uniform random
        """
        if not self.archive:
            return None

        cells = list(self.archive.values())

        if strategy == "count":
            # Weight by 1/count (prefer less visited)
            weights = np.array([1.0 / c['count'] for c in cells])
        elif strategy == "score":
            # Weight by score
            scores = np.array([c['score'] for c in cells])
            weights = scores - scores.min() + 1e-8
        else:
            weights = np.ones(len(cells))

        weights /= weights.sum()
        idx = np.random.choice(len(cells), p=weights)

        return cells[idx]['state']

    def get_exploration_bonus(self, obs: np.ndarray) -> float:
        """Get bonus based on cell visit count."""
        key = self._hash(obs)
        count = self.visit_counts[key]
        return 1.0 / np.sqrt(count + 1)

    def get_coverage(self) -> int:
        """Return number of unique cells discovered."""
        return len(self.archive)
