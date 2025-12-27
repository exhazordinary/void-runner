"""
VOID_RUNNER - Curiosity Collapse Diagnostics
==============================================
Understanding why intrinsic motivation fails.

The Curiosity Collapse Problem:
------------------------------
Observation: Curiosity-driven agents often stop exploring mid-training.
- Early training: High intrinsic rewards, broad exploration
- Mid training: Rewards drop, exploration narrows
- Late training: Near-zero intrinsic rewards, stuck in local optima

Why does this happen?
1. Predictor Overfitting: RND predictor becomes too good
2. State Space Saturation: All reachable states become "familiar"
3. Distribution Shift: Training distribution != exploration distribution
4. Gradient Collapse: Gradients become too small to learn
5. Representation Collapse: Encoder maps everything to same region

This module instruments curiosity to diagnose the cause.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class CuriositySnapshot:
    """Snapshot of curiosity state at a point in training."""
    step: int

    # Reward statistics
    mean_intrinsic_reward: float
    std_intrinsic_reward: float
    max_intrinsic_reward: float
    min_intrinsic_reward: float

    # Predictor statistics
    predictor_loss: float
    predictor_gradient_norm: float

    # State coverage
    unique_states_seen: int
    state_revisit_rate: float  # Fraction of states that are revisits

    # Representation health
    embedding_mean_norm: float
    embedding_std: float
    embedding_rank: float  # Effective dimensionality

    # Distribution statistics
    state_entropy: float
    reward_entropy: float

    # Collapse indicators
    reward_decay_rate: float  # How fast rewards are dropping
    exploration_stagnation: float  # Are we visiting new states?


class CuriosityCollapseDetector:
    """
    Detects and diagnoses curiosity collapse.

    Monitors key indicators:
    1. Intrinsic reward trajectory
    2. Predictor learning dynamics
    3. State space coverage
    4. Representation quality
    """

    def __init__(
        self,
        window_size: int = 1000,
        collapse_threshold: float = 0.1,
        save_path: Optional[str] = None,
    ):
        self.window_size = window_size
        self.collapse_threshold = collapse_threshold
        self.save_path = Path(save_path) if save_path else None

        # History tracking
        self.reward_history: deque = deque(maxlen=window_size)
        self.loss_history: deque = deque(maxlen=window_size)
        self.gradient_history: deque = deque(maxlen=window_size)
        self.coverage_history: deque = deque(maxlen=window_size)

        # State tracking
        self.state_hashes: set = set()
        self.recent_states: deque = deque(maxlen=window_size)
        self.embedding_buffer: deque = deque(maxlen=1000)

        # Snapshots
        self.snapshots: List[CuriositySnapshot] = []
        self.step = 0

        # Collapse detection
        self.collapse_detected = False
        self.collapse_step: Optional[int] = None
        self.collapse_diagnosis: Optional[str] = None

    def _hash_state(self, state: np.ndarray) -> str:
        return str(tuple(np.round(state, 2)))

    def record_step(
        self,
        state: np.ndarray,
        intrinsic_reward: float,
        predictor_loss: Optional[float] = None,
        gradient_norm: Optional[float] = None,
        embedding: Optional[np.ndarray] = None,
    ):
        """Record data from one environment step."""
        self.step += 1

        # Reward
        self.reward_history.append(intrinsic_reward)

        # Loss and gradients
        if predictor_loss is not None:
            self.loss_history.append(predictor_loss)
        if gradient_norm is not None:
            self.gradient_history.append(gradient_norm)

        # State coverage
        state_hash = self._hash_state(state)
        is_new = state_hash not in self.state_hashes
        self.state_hashes.add(state_hash)
        self.coverage_history.append(1.0 if is_new else 0.0)
        self.recent_states.append(state)

        # Embedding
        if embedding is not None:
            self.embedding_buffer.append(embedding)

        # Periodic snapshot
        if self.step % 1000 == 0:
            self._take_snapshot()

        # Check for collapse
        if self.step % 100 == 0:
            self._check_collapse()

    def _take_snapshot(self):
        """Take a diagnostic snapshot."""
        if len(self.reward_history) < 100:
            return

        rewards = np.array(list(self.reward_history))
        coverage = np.array(list(self.coverage_history))

        # Embedding analysis
        if len(self.embedding_buffer) > 10:
            embeddings = np.array(list(self.embedding_buffer))
            embedding_mean_norm = np.mean(np.linalg.norm(embeddings, axis=1))
            embedding_std = np.std(embeddings)

            # Effective rank (measure of dimensionality)
            try:
                _, s, _ = np.linalg.svd(embeddings - embeddings.mean(axis=0))
                s_normalized = s / s.sum()
                embedding_rank = np.exp(-np.sum(s_normalized * np.log(s_normalized + 1e-10)))
            except Exception:
                embedding_rank = 0.0
        else:
            embedding_mean_norm = 0.0
            embedding_std = 0.0
            embedding_rank = 0.0

        # State entropy
        unique_recent = len(set(self._hash_state(s) for s in self.recent_states))
        state_entropy = np.log(unique_recent + 1)

        # Reward entropy (binned)
        reward_bins = np.histogram(rewards, bins=20)[0]
        reward_probs = reward_bins / (reward_bins.sum() + 1e-10)
        reward_entropy = -np.sum(reward_probs * np.log(reward_probs + 1e-10))

        # Reward decay rate
        if len(self.snapshots) > 0:
            prev_reward = self.snapshots[-1].mean_intrinsic_reward
            curr_reward = np.mean(rewards)
            reward_decay = (prev_reward - curr_reward) / (prev_reward + 1e-10)
        else:
            reward_decay = 0.0

        snapshot = CuriositySnapshot(
            step=self.step,
            mean_intrinsic_reward=float(np.mean(rewards)),
            std_intrinsic_reward=float(np.std(rewards)),
            max_intrinsic_reward=float(np.max(rewards)),
            min_intrinsic_reward=float(np.min(rewards)),
            predictor_loss=float(np.mean(list(self.loss_history))) if self.loss_history else 0.0,
            predictor_gradient_norm=float(np.mean(list(self.gradient_history))) if self.gradient_history else 0.0,
            unique_states_seen=len(self.state_hashes),
            state_revisit_rate=1.0 - float(np.mean(coverage)),
            embedding_mean_norm=embedding_mean_norm,
            embedding_std=embedding_std,
            embedding_rank=embedding_rank,
            state_entropy=state_entropy,
            reward_entropy=reward_entropy,
            reward_decay_rate=reward_decay,
            exploration_stagnation=1.0 - float(np.mean(coverage[-100:])) if len(coverage) >= 100 else 0.0,
        )

        self.snapshots.append(snapshot)

    def _check_collapse(self):
        """Check if curiosity has collapsed."""
        if len(self.reward_history) < self.window_size // 2:
            return

        if self.collapse_detected:
            return

        rewards = np.array(list(self.reward_history))

        # Early vs recent rewards
        early_mean = np.mean(rewards[:len(rewards)//4])
        recent_mean = np.mean(rewards[-len(rewards)//4:])

        # Collapse conditions
        conditions = []

        # 1. Reward collapse
        if early_mean > 0 and recent_mean / (early_mean + 1e-10) < self.collapse_threshold:
            conditions.append("reward_collapse")

        # 2. Coverage stagnation
        if len(self.coverage_history) > 200:
            recent_coverage = np.mean(list(self.coverage_history)[-200:])
            if recent_coverage < 0.01:  # Less than 1% new states
                conditions.append("coverage_stagnation")

        # 3. Gradient vanishing
        if self.gradient_history:
            grads = np.array(list(self.gradient_history))
            if np.mean(grads[-100:]) < 1e-6:
                conditions.append("gradient_vanishing")

        # 4. Representation collapse
        if len(self.snapshots) > 2:
            recent_rank = self.snapshots[-1].embedding_rank
            if recent_rank > 0 and recent_rank < 2.0:  # Collapsed to < 2 effective dimensions
                conditions.append("representation_collapse")

        if conditions:
            self.collapse_detected = True
            self.collapse_step = self.step
            self.collapse_diagnosis = ", ".join(conditions)

    def diagnose(self) -> Dict[str, Any]:
        """
        Comprehensive diagnosis of curiosity state.

        Returns detailed analysis of what's happening.
        """
        if not self.snapshots:
            return {"status": "insufficient_data"}

        recent = self.snapshots[-1]
        early = self.snapshots[0] if len(self.snapshots) > 1 else recent

        diagnosis = {
            "step": self.step,
            "collapse_detected": self.collapse_detected,
            "collapse_step": self.collapse_step,
            "collapse_diagnosis": self.collapse_diagnosis,

            # Health metrics
            "current_intrinsic_reward": recent.mean_intrinsic_reward,
            "reward_change": recent.mean_intrinsic_reward / (early.mean_intrinsic_reward + 1e-10),
            "coverage": len(self.state_hashes),
            "exploration_rate": 1.0 - recent.state_revisit_rate,

            # Detailed analysis
            "predictor_loss_trend": self._compute_trend([s.predictor_loss for s in self.snapshots]),
            "reward_trend": self._compute_trend([s.mean_intrinsic_reward for s in self.snapshots]),
            "coverage_trend": self._compute_trend([s.unique_states_seen for s in self.snapshots]),

            # Recommendations
            "recommendations": self._get_recommendations(),
        }

        return diagnosis

    def _compute_trend(self, values: List[float]) -> str:
        """Compute trend direction."""
        if len(values) < 2:
            return "insufficient_data"

        # Linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]

        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"

    def _get_recommendations(self) -> List[str]:
        """Get recommendations based on diagnosis."""
        recommendations = []

        if not self.snapshots:
            return recommendations

        recent = self.snapshots[-1]

        # Reward collapse
        if recent.mean_intrinsic_reward < 0.01:
            recommendations.append(
                "Intrinsic rewards near zero. Consider: "
                "1) Adding noise to predictor, "
                "2) Using DRND (multiple networks), "
                "3) Periodic predictor reset"
            )

        # Exploration stagnation
        if recent.exploration_stagnation > 0.99:
            recommendations.append(
                "Exploration stagnated. Consider: "
                "1) Increasing epsilon, "
                "2) Go-Explore style returning, "
                "3) Hierarchical exploration"
            )

        # Representation collapse
        if recent.embedding_rank < 5:
            recommendations.append(
                "Representation collapse detected. Consider: "
                "1) Adding contrastive loss, "
                "2) Larger embedding dimension, "
                "3) Batch normalization"
            )

        # Gradient issues
        if recent.predictor_gradient_norm < 1e-5:
            recommendations.append(
                "Gradients vanishing. Consider: "
                "1) Lower learning rate, "
                "2) Gradient clipping, "
                "3) Skip connections"
            )

        if recent.predictor_gradient_norm > 100:
            recommendations.append(
                "Gradients exploding. Consider: "
                "1) Gradient clipping, "
                "2) Layer normalization, "
                "3) Lower learning rate"
            )

        return recommendations

    def plot_diagnostics(self) -> Optional[str]:
        """Generate ASCII visualization of diagnostics."""
        if len(self.snapshots) < 2:
            return None

        lines = ["CURIOSITY DIAGNOSTICS", "=" * 50]

        # Reward trajectory
        rewards = [s.mean_intrinsic_reward for s in self.snapshots]
        lines.append("\nIntrinsic Reward over Time:")
        lines.append(self._ascii_plot(rewards, height=10))

        # Coverage
        coverage = [s.unique_states_seen for s in self.snapshots]
        lines.append("\nUnique States Discovered:")
        lines.append(self._ascii_plot(coverage, height=10))

        # Exploration rate
        exploration = [1 - s.state_revisit_rate for s in self.snapshots]
        lines.append("\nExploration Rate (new states / total):")
        lines.append(self._ascii_plot(exploration, height=10))

        return "\n".join(lines)

    def _ascii_plot(self, values: List[float], height: int = 10, width: int = 50) -> str:
        """Generate ASCII plot."""
        if not values:
            return "  (no data)"

        min_val, max_val = min(values), max(values)
        range_val = max_val - min_val + 1e-10

        # Sample to width
        if len(values) > width:
            indices = np.linspace(0, len(values) - 1, width).astype(int)
            values = [values[i] for i in indices]

        lines = []
        for row in range(height):
            threshold = max_val - (row / height) * range_val
            line = "  |"
            for val in values:
                if val >= threshold:
                    line += "*"
                else:
                    line += " "
            lines.append(line)

        lines.append("  +" + "-" * len(values))
        lines.append(f"   {min_val:.3f}" + " " * (len(values) - 10) + f"{max_val:.3f}")

        return "\n".join(lines)

    def save(self, filepath: str):
        """Save diagnostics to JSON."""
        data = {
            "snapshots": [
                {
                    "step": s.step,
                    "mean_intrinsic_reward": s.mean_intrinsic_reward,
                    "std_intrinsic_reward": s.std_intrinsic_reward,
                    "unique_states_seen": s.unique_states_seen,
                    "state_revisit_rate": s.state_revisit_rate,
                    "predictor_loss": s.predictor_loss,
                    "predictor_gradient_norm": s.predictor_gradient_norm,
                    "embedding_rank": s.embedding_rank,
                    "state_entropy": s.state_entropy,
                    "reward_entropy": s.reward_entropy,
                    "exploration_stagnation": s.exploration_stagnation,
                }
                for s in self.snapshots
            ],
            "diagnosis": self.diagnose(),
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


class CuriosityHealthMonitor:
    """
    Real-time curiosity health monitoring with alerts.

    Watches for early signs of collapse and suggests interventions.
    """

    def __init__(self, alert_callback=None):
        self.detector = CuriosityCollapseDetector()
        self.alert_callback = alert_callback or print

        self.last_alert_step = 0
        self.alert_cooldown = 1000

    def step(
        self,
        state: np.ndarray,
        intrinsic_reward: float,
        predictor_loss: Optional[float] = None,
        gradient_norm: Optional[float] = None,
        embedding: Optional[np.ndarray] = None,
    ):
        """Monitor one step and alert if issues detected."""
        self.detector.record_step(
            state, intrinsic_reward, predictor_loss, gradient_norm, embedding
        )

        # Check for issues periodically
        if self.detector.step % 500 == 0:
            self._check_health()

    def _check_health(self):
        """Check health and raise alerts."""
        if self.detector.step - self.last_alert_step < self.alert_cooldown:
            return

        diagnosis = self.detector.diagnose()

        if diagnosis.get("collapse_detected"):
            self.alert_callback(
                f"[ALERT] Curiosity collapse detected at step {diagnosis['collapse_step']}!\n"
                f"Cause: {diagnosis['collapse_diagnosis']}\n"
                f"Recommendations: {diagnosis['recommendations']}"
            )
            self.last_alert_step = self.detector.step

        elif diagnosis.get("reward_trend") == "decreasing":
            # Early warning
            recent_reward = diagnosis.get("current_intrinsic_reward", 0)
            if recent_reward < 0.1:
                self.alert_callback(
                    f"[WARNING] Intrinsic rewards declining (current: {recent_reward:.4f})\n"
                    f"Consider interventions before full collapse."
                )
                self.last_alert_step = self.detector.step

    def get_report(self) -> str:
        """Get full diagnostic report."""
        diagnosis = self.detector.diagnose()
        plot = self.detector.plot_diagnostics()

        report = [
            "=" * 60,
            "CURIOSITY HEALTH REPORT",
            "=" * 60,
            f"Step: {diagnosis['step']}",
            f"Collapse Detected: {diagnosis['collapse_detected']}",
            f"Current Reward: {diagnosis['current_intrinsic_reward']:.4f}",
            f"States Discovered: {diagnosis['coverage']}",
            f"Exploration Rate: {diagnosis['exploration_rate']:.2%}",
            "",
            "Trends:",
            f"  Reward: {diagnosis['reward_trend']}",
            f"  Coverage: {diagnosis['coverage_trend']}",
            f"  Predictor Loss: {diagnosis['predictor_loss_trend']}",
            "",
            "Recommendations:",
        ]

        for rec in diagnosis.get('recommendations', []):
            report.append(f"  - {rec}")

        if plot:
            report.append("")
            report.append(plot)

        return "\n".join(report)


def instrument_curiosity_module(curiosity_module, monitor: CuriosityHealthMonitor):
    """
    Decorator to instrument any curiosity module with health monitoring.

    Usage:
        monitor = CuriosityHealthMonitor()
        curiosity = CuriosityModule(obs_dim, device)
        curiosity = instrument_curiosity_module(curiosity, monitor)

        # Now curiosity.compute_intrinsic_reward automatically tracks health
    """
    original_compute = curiosity_module.compute_intrinsic_reward

    def instrumented_compute(obs, *args, **kwargs):
        reward = original_compute(obs, *args, **kwargs)

        # Record for monitoring
        if isinstance(reward, torch.Tensor):
            reward_np = reward.detach().cpu().numpy()
            obs_np = obs.detach().cpu().numpy() if isinstance(obs, torch.Tensor) else obs

            for i in range(len(reward_np)):
                monitor.step(obs_np[i], reward_np[i])

        return reward

    curiosity_module.compute_intrinsic_reward = instrumented_compute
    curiosity_module._health_monitor = monitor

    return curiosity_module
