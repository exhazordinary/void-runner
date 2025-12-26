"""
VOID_RUNNER - Information-Theoretic Exploration Metrics
========================================================
Rigorous measurement of exploration quality.

How do we know if an agent is exploring "well"?

Traditional metrics (coverage, reward) miss important aspects:
- Coverage doesn't capture the STRUCTURE of exploration
- Reward is sparse and often uninformative

Information-theoretic metrics capture:
- State space entropy (diversity of states visited)
- Action entropy (diversity of behaviors)
- Mutual information (how much do actions tell us about outcomes?)
- Compression (can exploration be summarized efficiently?)

These give us a richer picture of exploration quality.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter, deque
from scipy import stats
from scipy.special import entr
import warnings


class ExplorationMetrics:
    """
    Comprehensive exploration metrics using information theory.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_bins: int = 20,  # Discretization bins for continuous states
        history_size: int = 10000
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_bins = n_bins

        # History buffers
        self.state_history = deque(maxlen=history_size)
        self.action_history = deque(maxlen=history_size)
        self.transition_history = deque(maxlen=history_size)

        # Running statistics
        self.state_min = np.full(state_dim, np.inf)
        self.state_max = np.full(state_dim, -np.inf)

    def update(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray
    ):
        """Record a transition."""
        self.state_history.append(state.copy())
        self.action_history.append(action)
        self.transition_history.append((
            self._discretize(state),
            action,
            self._discretize(next_state)
        ))

        # Update bounds for discretization
        self.state_min = np.minimum(self.state_min, state)
        self.state_max = np.maximum(self.state_max, state)

    def _discretize(self, state: np.ndarray) -> tuple:
        """Discretize continuous state for entropy calculations."""
        # Normalize to [0, 1]
        range_vals = self.state_max - self.state_min + 1e-8
        normalized = (state - self.state_min) / range_vals

        # Discretize
        discrete = np.clip(
            (normalized * self.n_bins).astype(int),
            0, self.n_bins - 1
        )
        return tuple(discrete)

    def state_entropy(self) -> float:
        """
        Entropy of visited state distribution.

        H(S) = -sum_s p(s) log p(s)

        High entropy = diverse exploration
        Low entropy = concentrated in few states
        """
        if len(self.state_history) < 10:
            return 0.0

        discrete_states = [self._discretize(s) for s in self.state_history]
        counts = Counter(discrete_states)
        total = len(discrete_states)

        probs = np.array([c / total for c in counts.values()])
        return float(stats.entropy(probs))

    def action_entropy(self) -> float:
        """
        Entropy of action distribution.

        H(A) = -sum_a p(a) log p(a)

        High entropy = diverse actions
        Low entropy = repetitive behavior
        """
        if len(self.action_history) < 10:
            return 0.0

        counts = Counter(self.action_history)
        total = len(self.action_history)

        probs = np.array([c / total for c in counts.values()])
        return float(stats.entropy(probs))

    def state_action_mutual_information(self) -> float:
        """
        Mutual information between states and actions.

        I(S; A) = H(A) - H(A|S)

        High MI = actions depend strongly on state (policy is state-dependent)
        Low MI = actions are random/independent of state
        """
        if len(self.state_history) < 100:
            return 0.0

        # Joint and marginal distributions
        states = [self._discretize(s) for s in self.state_history]
        actions = list(self.action_history)

        state_counts = Counter(states)
        action_counts = Counter(actions)
        joint_counts = Counter(zip(states, actions))

        total = len(states)

        # Compute MI
        mi = 0.0
        for (s, a), count in joint_counts.items():
            p_sa = count / total
            p_s = state_counts[s] / total
            p_a = action_counts[a] / total

            if p_sa > 0 and p_s > 0 and p_a > 0:
                mi += p_sa * np.log(p_sa / (p_s * p_a) + 1e-10)

        return max(0.0, float(mi))

    def transition_entropy(self) -> float:
        """
        Entropy of transition distribution.

        H(S'|S, A) measures unpredictability of dynamics.

        High = stochastic/complex environment
        Low = deterministic/simple environment
        """
        if len(self.transition_history) < 100:
            return 0.0

        # Group by (state, action), compute entropy of next states
        sa_to_next = {}
        for s, a, ns in self.transition_history:
            key = (s, a)
            if key not in sa_to_next:
                sa_to_next[key] = []
            sa_to_next[key].append(ns)

        # Average conditional entropy
        total_entropy = 0.0
        total_weight = 0

        for key, next_states in sa_to_next.items():
            if len(next_states) >= 2:
                counts = Counter(next_states)
                probs = np.array([c / len(next_states) for c in counts.values()])
                total_entropy += stats.entropy(probs) * len(next_states)
                total_weight += len(next_states)

        if total_weight == 0:
            return 0.0

        return float(total_entropy / total_weight)

    def coverage_rate(self) -> float:
        """
        Fraction of state space covered.

        Estimated as unique_states / expected_states.
        """
        if len(self.state_history) < 10:
            return 0.0

        discrete_states = set(self._discretize(s) for s in self.state_history)
        n_unique = len(discrete_states)

        # Expected coverage under uniform exploration
        n_steps = len(self.state_history)
        expected_unique = min(n_steps, self.n_bins ** self.state_dim)

        return n_unique / max(1, expected_unique)

    def exploration_efficiency(self) -> float:
        """
        How efficiently is the agent covering new ground?

        Measures unique states found per step over time.
        Higher = finding new states faster.
        """
        if len(self.state_history) < 100:
            return 0.0

        states = list(self.state_history)
        unique_counts = []
        seen = set()

        window_size = 100
        for i in range(0, len(states), window_size):
            window = states[i:i + window_size]
            new_unique = sum(
                1 for s in window
                if self._discretize(s) not in seen
            )
            for s in window:
                seen.add(self._discretize(s))
            unique_counts.append(new_unique / window_size)

        if len(unique_counts) < 2:
            return unique_counts[0] if unique_counts else 0.0

        # Return recent efficiency
        return float(np.mean(unique_counts[-5:]))

    def revisit_rate(self) -> float:
        """
        How often does the agent revisit known states?

        High = exploiting known territory
        Low = constantly exploring new areas
        """
        if len(self.state_history) < 100:
            return 0.0

        states = [self._discretize(s) for s in self.state_history]
        seen = set()
        revisits = 0

        for s in states:
            if s in seen:
                revisits += 1
            seen.add(s)

        return revisits / len(states)

    def curiosity_decay_rate(
        self,
        curiosity_values: List[float]
    ) -> float:
        """
        Rate at which curiosity is decaying.

        Positive = curiosity decreasing (learning the environment)
        Negative = curiosity increasing (finding new things)
        Zero = stable curiosity
        """
        if len(curiosity_values) < 10:
            return 0.0

        values = np.array(curiosity_values)

        # Linear regression to find trend
        x = np.arange(len(values))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            slope, _, _, _, _ = stats.linregress(x, values)

        return float(slope)

    def get_all_metrics(self) -> Dict[str, float]:
        """Compute all exploration metrics."""
        return {
            'state_entropy': self.state_entropy(),
            'action_entropy': self.action_entropy(),
            'state_action_mi': self.state_action_mutual_information(),
            'transition_entropy': self.transition_entropy(),
            'coverage_rate': self.coverage_rate(),
            'exploration_efficiency': self.exploration_efficiency(),
            'revisit_rate': self.revisit_rate(),
        }


class CompressionMetrics:
    """
    Measures how compressible the agent's behavior is.

    The idea: Good exploration should be hard to compress
    (diverse, unpredictable), while exploitation should be
    easy to compress (repetitive, predictable).
    """

    def __init__(self, history_size: int = 1000):
        self.action_sequence = deque(maxlen=history_size)
        self.state_sequence = deque(maxlen=history_size)

    def update(self, state: np.ndarray, action: int):
        """Record observation."""
        self.action_sequence.append(action)
        self.state_sequence.append(tuple(np.round(state, 2)))

    def action_compressibility(self) -> float:
        """
        How compressible is the action sequence?

        Uses ratio of run-length encoding to measure repetition.
        0 = completely random (incompressible)
        1 = completely repetitive (maximally compressible)
        """
        if len(self.action_sequence) < 10:
            return 0.5

        actions = list(self.action_sequence)

        # Run-length encoding
        runs = 1
        for i in range(1, len(actions)):
            if actions[i] != actions[i - 1]:
                runs += 1

        # Compressibility ratio
        return 1.0 - (runs / len(actions))

    def sequence_complexity(self) -> float:
        """
        Lempel-Ziv complexity of action sequence.

        Measures the number of distinct patterns.
        Higher = more complex exploration.
        """
        if len(self.action_sequence) < 10:
            return 0.0

        sequence = ''.join(str(a) for a in self.action_sequence)
        return self._lz_complexity(sequence) / len(sequence)

    def _lz_complexity(self, s: str) -> int:
        """Compute Lempel-Ziv complexity."""
        n = len(s)
        if n == 0:
            return 0

        complexity = 1
        i = 0
        j = 1

        while j < n:
            # Check if s[i:j+1] is in s[0:j]
            if s[i:j + 1] in s[:j]:
                j += 1
            else:
                complexity += 1
                i = j
                j = i + 1

        return complexity


class InformationGain:
    """
    Measures information gained through exploration.

    The agent's goal is to maximize information about the environment.
    This tracks how much the agent learns over time.
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        self.obs_dim = obs_dim

        # Simple predictor for measuring learning progress
        self.predictor = None  # Will initialize on first update
        self.prediction_errors = []
        self.learning_curve = []

    def _init_predictor(self, device):
        """Lazy initialization of predictor."""
        import torch.nn as nn
        self.predictor = nn.Sequential(
            nn.Linear(self.obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.obs_dim)
        ).to(device)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=1e-3)

    def update(self, obs: torch.Tensor, next_obs: torch.Tensor):
        """Update predictor and track information gain."""
        if self.predictor is None:
            self._init_predictor(obs.device)

        # Prediction error before update
        with torch.no_grad():
            pred = self.predictor(obs)
            error_before = torch.mean((pred - next_obs) ** 2).item()

        # Update
        pred = self.predictor(obs)
        loss = torch.mean((pred - next_obs) ** 2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Prediction error after update
        with torch.no_grad():
            pred = self.predictor(obs)
            error_after = torch.mean((pred - next_obs) ** 2).item()

        # Information gain = reduction in prediction error
        info_gain = max(0, error_before - error_after)

        self.prediction_errors.append(error_after)
        self.learning_curve.append(info_gain)

        return info_gain

    def get_learning_progress(self) -> float:
        """Get recent learning progress."""
        if len(self.prediction_errors) < 20:
            return 0.0

        # Trend in prediction error
        recent = self.prediction_errors[-100:]
        x = np.arange(len(recent))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            slope, _, _, _, _ = stats.linregress(x, recent)

        # Negative slope = decreasing error = learning
        return -float(slope)

    def get_cumulative_information(self) -> float:
        """Get total information gained."""
        return sum(self.learning_curve)
