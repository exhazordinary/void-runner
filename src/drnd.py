"""
VOID_RUNNER - Distributional Random Network Distillation (DRND)
================================================================
Implementation based on ICML 2024 paper:
"Exploration and Anti-Exploration with Distributional Random Network Distillation"
by Yang et al.

Key insight: Standard RND has "bonus inconsistency" - uneven reward distribution
at initialization and poor discrimination after training.

DRND solves this by:
1. Using N random target networks instead of 1
2. Averaging predictions reduces initial variance
3. Implicit pseudo-count estimation improves final consistency

Mathematical formulation:
    b(x) = α * ‖fθ(x) - μ(x)‖² + (1-α) * √(variance_term)

Where the variance term implicitly estimates 1/visit_count.

Reference: https://arxiv.org/abs/2401.09750
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


class DRNDTargetEnsemble(nn.Module):
    """
    Ensemble of N frozen random networks.

    Unlike single-network RND, this ensemble:
    1. Reduces variance in initial bonuses (averaging effect)
    2. Enables implicit pseudo-count estimation
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 512,
        hidden_dim: int = 256,
        n_networks: int = 5,  # Paper uses 5 networks
    ):
        super().__init__()
        self.n_networks = n_networks
        self.output_dim = output_dim

        # Create N independent random networks
        self.networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
            for _ in range(n_networks)
        ])

        # Initialize with orthogonal weights for diversity
        for i, net in enumerate(self.networks):
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    # Different seeds for each network
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2) * (1 + 0.1 * i))
                    nn.init.zeros_(m.bias)

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute target statistics across ensemble.

        Returns:
            outputs: All network outputs [N, batch, output_dim]
            mean: Mean across networks [batch, output_dim]
            second_moment: E[f²] across networks [batch, output_dim]
        """
        outputs = torch.stack([net(x) for net in self.networks])
        mean = outputs.mean(dim=0)
        second_moment = (outputs ** 2).mean(dim=0)

        return outputs, mean, second_moment


class DRNDPredictor(nn.Module):
    """
    Predictor network that learns to match the ensemble mean.

    The prediction error provides exploration bonus.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 512,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DRND:
    """
    Distributional Random Network Distillation.

    Improvements over RND:
    1. Ensemble averaging reduces initial bonus variance
    2. Implicit pseudo-count via variance estimation
    3. Better discrimination in visited vs unvisited states

    Intrinsic reward:
        b(x) = α * ‖f(x) - μ(x)‖² + (1-α) * pseudo_count_term

    Where pseudo_count_term ≈ 1/n(x) for visit count n(x).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 512,
        hidden_dim: int = 256,
        n_networks: int = 5,
        learning_rate: float = 1e-4,
        alpha: float = 0.5,  # Balance between RND-like and pseudo-count bonus
        update_proportion: float = 0.25,
        device: str = "cpu",
    ):
        self.device = device
        self.alpha = alpha
        self.update_proportion = update_proportion
        self.output_dim = output_dim

        # Target ensemble (frozen)
        self.target = DRNDTargetEnsemble(
            input_dim, output_dim, hidden_dim, n_networks
        ).to(device)

        # Predictor (trainable)
        self.predictor = DRNDPredictor(
            input_dim, output_dim, hidden_dim
        ).to(device)

        self.optimizer = optim.Adam(
            self.predictor.parameters(),
            lr=learning_rate
        )

        # Running statistics for normalization
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = 1e-4

        # Statistics tracking
        self.total_bonus = 0.0
        self.bonus_count = 0

    def compute_intrinsic_reward(
        self,
        obs: torch.Tensor,
        update: bool = True
    ) -> torch.Tensor:
        """
        Compute DRND intrinsic reward.

        b(x) = α * MSE_bonus + (1-α) * pseudo_count_bonus

        Args:
            obs: Observations [batch, obs_dim]
            update: Whether to update predictor

        Returns:
            Intrinsic rewards [batch]
        """
        # Get target statistics
        with torch.no_grad():
            _, target_mean, target_second_moment = self.target(obs)

        # Get predictor output
        pred = self.predictor(obs)

        # Component 1: RND-style MSE bonus
        mse_bonus = torch.mean((pred - target_mean) ** 2, dim=-1)

        # Component 2: Pseudo-count bonus (variance-based)
        # This estimates 1/n(x) where n(x) is visit count
        pred_sq = pred ** 2
        target_mean_sq = target_mean ** 2
        variance = target_second_moment - target_mean_sq + 1e-8

        # y(x) ≈ 1/n(x) is an unbiased estimator
        pseudo_count_term = (pred_sq - target_mean_sq) / variance
        pseudo_count_bonus = torch.sqrt(
            torch.clamp(pseudo_count_term.mean(dim=-1), min=0.0) + 1e-8
        )

        # Combined bonus
        intrinsic_reward = (
            self.alpha * mse_bonus +
            (1 - self.alpha) * pseudo_count_bonus
        )

        # Update predictor
        if update:
            mask = torch.rand(obs.shape[0], device=self.device) < self.update_proportion
            if mask.sum() > 0:
                loss = mse_bonus[mask].mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
                self.optimizer.step()

        # Update statistics
        reward_np = intrinsic_reward.detach().cpu().numpy()
        self._update_reward_stats(reward_np)

        self.total_bonus += intrinsic_reward.mean().item()
        self.bonus_count += 1

        return intrinsic_reward.detach()

    def _update_reward_stats(self, rewards: np.ndarray):
        """Online update of reward mean and variance."""
        batch_mean = rewards.mean()
        batch_var = rewards.var()
        batch_count = len(rewards)

        delta = batch_mean - self.reward_mean
        total_count = self.reward_count + batch_count

        self.reward_mean += delta * batch_count / total_count
        m_a = self.reward_var * self.reward_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.reward_count * batch_count / total_count
        self.reward_var = M2 / total_count
        self.reward_count = total_count

    def normalize_reward(self, reward: torch.Tensor) -> torch.Tensor:
        """Normalize reward using running statistics."""
        return reward / np.sqrt(self.reward_var + 1e-8)

    def get_avg_bonus(self) -> float:
        if self.bonus_count == 0:
            return 0.0
        return self.total_bonus / self.bonus_count

    def reset_stats(self):
        self.total_bonus = 0.0
        self.bonus_count = 0


class NGUEpisodicMemory:
    """
    Never Give Up (NGU) style episodic memory.

    Uses k-nearest neighbors in embedding space for fast
    within-episode novelty detection.

    Key insight: RND captures lifelong novelty but is slow to adapt.
    K-NN memory captures within-episode novelty and adapts instantly.

    Reference: "Never Give Up: Learning Directed Exploration Strategies"
    https://arxiv.org/abs/2002.06038
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        memory_size: int = 1000,
        k: int = 10,  # k for k-nearest neighbors
        kernel_epsilon: float = 0.001,
        cluster_distance: float = 0.008,
        max_similarity: float = 8.0,
        device: str = "cpu"
    ):
        self.embedding_dim = embedding_dim
        self.memory_size = memory_size
        self.k = k
        self.kernel_epsilon = kernel_epsilon
        self.cluster_distance = cluster_distance
        self.max_similarity = max_similarity
        self.device = device

        # Embedding memory (circular buffer per episode)
        self.memory = torch.zeros(memory_size, embedding_dim, device=device)
        self.memory_index = 0
        self.memory_count = 0

    def reset_episode(self):
        """Clear memory at episode start."""
        self.memory.zero_()
        self.memory_index = 0
        self.memory_count = 0

    def compute_episodic_bonus(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute episodic novelty using k-NN.

        Bonus is high when embedding is far from all memories.
        Uses kernel-based similarity: K(x, y) = ε / (d(x,y)² + ε)

        Args:
            embedding: [batch, embedding_dim]

        Returns:
            Episodic bonus [batch]
        """
        batch_size = embedding.shape[0]

        if self.memory_count < self.k:
            # Not enough memories yet
            bonus = torch.ones(batch_size, device=self.device)
        else:
            # Compute distances to all memories
            # embedding: [batch, dim], memory: [memory_count, dim]
            memory = self.memory[:self.memory_count]

            # Euclidean distances: [batch, memory_count]
            dists = torch.cdist(embedding, memory)

            # Get k-nearest distances
            k_dists, _ = torch.topk(dists, k=min(self.k, self.memory_count),
                                     dim=-1, largest=False)

            # Kernel-based similarity
            # K(d) = ε / (d² + ε)
            similarities = self.kernel_epsilon / (k_dists ** 2 + self.kernel_epsilon)

            # Average similarity to k-nearest neighbors
            avg_similarity = similarities.mean(dim=-1)

            # Clamp similarity
            avg_similarity = torch.clamp(avg_similarity, max=self.max_similarity)

            # Bonus is inverse of similarity (novel = low similarity = high bonus)
            # Using the NGU formula: 1/sqrt(sum of similarities)
            bonus = 1.0 / torch.sqrt(avg_similarity + 1e-8)

        # Add embeddings to memory
        self._add_to_memory(embedding)

        return bonus

    def _add_to_memory(self, embeddings: torch.Tensor):
        """Add embeddings to circular buffer."""
        for emb in embeddings:
            # Check if too similar to existing memory (clustering)
            if self.memory_count > 0:
                memory = self.memory[:self.memory_count]
                min_dist = torch.min(torch.norm(memory - emb, dim=-1))
                if min_dist < self.cluster_distance:
                    continue  # Skip similar embeddings

            self.memory[self.memory_index] = emb.detach()
            self.memory_index = (self.memory_index + 1) % self.memory_size
            self.memory_count = min(self.memory_count + 1, self.memory_size)


class NGUCombinedCuriosity:
    """
    Never Give Up combined curiosity: Episodic + Lifelong.

    r_intrinsic = r_episodic * min(max(r_lifelong, 1), L)

    Where:
    - r_episodic: Fast, within-episode novelty (k-NN based)
    - r_lifelong: Slow, across-episode novelty (RND/DRND based)
    - L: Lifelong reward ceiling

    This combination ensures:
    1. Revisiting same state in episode gives low reward
    2. States novel across entire training get high reward
    3. Multiplication prevents reward hacking
    """

    def __init__(
        self,
        obs_dim: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        n_drnd_networks: int = 5,
        learning_rate: float = 1e-4,
        lifelong_ceiling: float = 5.0,
        device: str = "cpu"
    ):
        self.device = device
        self.lifelong_ceiling = lifelong_ceiling

        # Embedding network for episodic memory
        self.embedding_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        ).to(device)

        # Inverse dynamics model to train embeddings (self-supervised)
        # Predicts action from (s, s') to make embeddings control-relevant
        self.inverse_model = None  # Would need action dim

        self.embedding_optimizer = optim.Adam(
            self.embedding_net.parameters(), lr=learning_rate
        )

        # Episodic memory (k-NN)
        self.episodic_memory = NGUEpisodicMemory(
            embedding_dim=embedding_dim,
            device=device
        )

        # Lifelong curiosity (DRND)
        self.lifelong = DRND(
            input_dim=obs_dim,
            output_dim=256,
            hidden_dim=hidden_dim,
            n_networks=n_drnd_networks,
            learning_rate=learning_rate,
            device=device
        )

    def compute_intrinsic_reward(
        self,
        obs: torch.Tensor,
        update: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined episodic + lifelong intrinsic reward.

        Returns:
            reward: Combined intrinsic reward
            info: Dict with component rewards
        """
        # Compute embedding for episodic memory
        embedding = self.embedding_net(obs)

        # Episodic bonus (fast, within-episode)
        episodic_bonus = self.episodic_memory.compute_episodic_bonus(embedding)

        # Lifelong bonus (slow, across-episodes)
        lifelong_bonus = self.lifelong.compute_intrinsic_reward(obs, update=update)

        # Normalize lifelong bonus
        lifelong_bonus = self.lifelong.normalize_reward(lifelong_bonus)

        # Clamp lifelong bonus
        lifelong_clamped = torch.clamp(
            torch.clamp(lifelong_bonus, min=1.0),
            max=self.lifelong_ceiling
        )

        # Combined reward (NGU formula)
        combined = episodic_bonus * lifelong_clamped

        info = {
            'episodic_bonus': episodic_bonus.mean().item(),
            'lifelong_bonus': lifelong_bonus.mean().item(),
            'combined_bonus': combined.mean().item(),
        }

        return combined, info

    def reset_episode(self):
        """Reset episodic memory at episode boundary."""
        self.episodic_memory.reset_episode()
