"""
VOID_RUNNER - Go-Explore Implementation
=========================================
Neural Go-Explore for hard exploration problems.

Key innovations from Ecoffet et al. (2019/2021):
1. Archive of discovered cells (states)
2. Return-then-explore: Go back to promising states, then explore
3. Cell selection prioritizes novelty + potential

2024 Enhancement (Intelligent Go-Explore):
- Neural state embeddings instead of hand-crafted features
- Learned cell representation via contrastive learning
- Foundation model integration for state evaluation

References:
- Ecoffet et al., "Go-Explore: A New Approach for Hard-Exploration Problems"
- Lu et al., "Intelligent Go-Explore with Foundation Models" (2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field
import heapq
import random


@dataclass
class Cell:
    """
    A cell represents a discretized state in the exploration archive.

    In neural Go-Explore, cells are defined by neural embeddings
    rather than hand-crafted features.
    """
    embedding: np.ndarray  # Neural embedding of state
    trajectory: List[int]  # Actions to reach this cell from start
    state_snapshot: Optional[Any] = None  # Environment state for restoration
    visit_count: int = 0
    novelty_score: float = float('inf')  # Higher = more novel
    potential_score: float = 0.0  # Estimated future exploration potential
    creation_step: int = 0

    def __hash__(self):
        # Hash based on discretized embedding
        return hash(tuple(np.round(self.embedding, decimals=2)))

    def priority(self, exploration_bonus: float = 1.0) -> float:
        """
        Priority for cell selection.
        Balances novelty (unexplored) vs potential (promising).
        """
        novelty_weight = 1.0 / (np.sqrt(self.visit_count) + 1)
        return (
            novelty_weight * self.novelty_score +
            exploration_bonus * self.potential_score
        )


class StateEncoder(nn.Module):
    """
    Neural encoder for state representation.

    Maps observations to compact embeddings that capture
    exploration-relevant features.
    """

    def __init__(
        self,
        obs_dim: int,
        embedding_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        # Projection head for contrastive learning
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        self.embedding_dim = embedding_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def project(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.projector(embedding)


class LocalitySensitiveHash:
    """
    LSH for efficient cell lookup in high-dimensional space.

    Maps continuous embeddings to discrete hash buckets
    for O(1) cell retrieval.
    """

    def __init__(
        self,
        embedding_dim: int,
        n_hashes: int = 16,
        bucket_size: float = 0.5,
    ):
        self.embedding_dim = embedding_dim
        self.n_hashes = n_hashes
        self.bucket_size = bucket_size

        # Random projection vectors for LSH
        self.projections = np.random.randn(n_hashes, embedding_dim)
        self.projections /= np.linalg.norm(self.projections, axis=1, keepdims=True)

    def hash(self, embedding: np.ndarray) -> str:
        """Convert embedding to hash key."""
        projected = embedding @ self.projections.T
        buckets = np.floor(projected / self.bucket_size).astype(int)
        return ",".join(map(str, buckets))


class CellArchive:
    """
    Archive of discovered cells with efficient lookup.

    Core data structure for Go-Explore:
    - Stores all discovered cells
    - Supports efficient nearest-neighbor lookup
    - Tracks exploration frontier
    """

    def __init__(
        self,
        embedding_dim: int,
        max_cells: int = 100000,
        similarity_threshold: float = 0.1,
    ):
        self.embedding_dim = embedding_dim
        self.max_cells = max_cells
        self.similarity_threshold = similarity_threshold

        self.cells: Dict[str, Cell] = {}
        self.lsh = LocalitySensitiveHash(embedding_dim)

        # Priority queue for cell selection
        self.frontier: List[Tuple[float, str]] = []

        # Statistics
        self.total_insertions = 0
        self.total_updates = 0

    def add_or_update(
        self,
        embedding: np.ndarray,
        trajectory: List[int],
        state_snapshot: Optional[Any] = None,
        step: int = 0,
    ) -> Tuple[bool, str]:
        """
        Add new cell or update existing one.

        Returns (is_new, cell_key)
        """
        key = self.lsh.hash(embedding)

        if key in self.cells:
            # Existing cell - update if better trajectory
            cell = self.cells[key]
            if len(trajectory) < len(cell.trajectory):
                cell.trajectory = trajectory.copy()
                cell.state_snapshot = state_snapshot
                self.total_updates += 1
            cell.visit_count += 1
            return False, key
        else:
            # New cell
            if len(self.cells) >= self.max_cells:
                self._evict_low_priority()

            cell = Cell(
                embedding=embedding.copy(),
                trajectory=trajectory.copy(),
                state_snapshot=state_snapshot,
                visit_count=1,
                creation_step=step,
            )
            self.cells[key] = cell
            self.total_insertions += 1

            # Add to frontier
            priority = -cell.priority()  # Negative for min-heap
            heapq.heappush(self.frontier, (priority, key))

            return True, key

    def _evict_low_priority(self):
        """Remove lowest priority cells when archive is full."""
        if not self.cells:
            return

        # Sort by priority and remove bottom 10%
        sorted_cells = sorted(
            self.cells.items(),
            key=lambda x: x[1].priority()
        )
        n_remove = max(1, len(sorted_cells) // 10)

        for key, _ in sorted_cells[:n_remove]:
            del self.cells[key]

    def select_cell(
        self,
        strategy: str = "weighted",
        temperature: float = 1.0,
    ) -> Optional[Cell]:
        """
        Select a cell for exploration.

        Strategies:
        - "weighted": Sample by priority (higher = more likely)
        - "frontier": Always take highest priority
        - "uniform": Random selection
        """
        if not self.cells:
            return None

        if strategy == "frontier":
            # Pop highest priority from frontier
            while self.frontier:
                _, key = heapq.heappop(self.frontier)
                if key in self.cells:
                    cell = self.cells[key]
                    cell.visit_count += 1
                    return cell
            return None

        elif strategy == "uniform":
            key = random.choice(list(self.cells.keys()))
            cell = self.cells[key]
            cell.visit_count += 1
            return cell

        else:  # weighted
            keys = list(self.cells.keys())
            priorities = np.array([
                self.cells[k].priority() for k in keys
            ])

            # Softmax with temperature
            priorities = priorities / temperature
            probs = np.exp(priorities - priorities.max())
            probs = probs / probs.sum()

            key = np.random.choice(keys, p=probs)
            cell = self.cells[key]
            cell.visit_count += 1
            return cell

    def update_novelty(self, key: str, novelty: float):
        """Update novelty score for a cell."""
        if key in self.cells:
            self.cells[key].novelty_score = novelty

    def update_potential(self, key: str, potential: float):
        """Update potential score for a cell."""
        if key in self.cells:
            self.cells[key].potential_score = potential

    def get_statistics(self) -> Dict[str, Any]:
        """Get archive statistics."""
        if not self.cells:
            return {"n_cells": 0}

        visit_counts = [c.visit_count for c in self.cells.values()]
        novelties = [c.novelty_score for c in self.cells.values()]
        traj_lengths = [len(c.trajectory) for c in self.cells.values()]

        return {
            "n_cells": len(self.cells),
            "total_insertions": self.total_insertions,
            "mean_visit_count": np.mean(visit_counts),
            "mean_novelty": np.mean([n for n in novelties if n < float('inf')]),
            "mean_trajectory_length": np.mean(traj_lengths),
            "max_trajectory_length": max(traj_lengths),
        }


class GoExplore:
    """
    Neural Go-Explore for hard exploration problems.

    Algorithm:
    1. Select a cell from archive (prioritize novel + promising)
    2. Return to that cell by replaying trajectory
    3. Explore from that cell using curiosity-driven policy
    4. Add new cells to archive

    This implementation uses neural embeddings for cell representation,
    making it applicable to high-dimensional observation spaces.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        max_cells: int = 100000,
        explore_steps: int = 100,
        learning_rate: float = 3e-4,
        device: str = "cpu",
    ):
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.explore_steps = explore_steps

        # State encoder
        self.encoder = StateEncoder(
            obs_dim, embedding_dim, hidden_dim
        ).to(device)

        # RND for novelty estimation
        self.rnd_target = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        ).to(device)

        self.rnd_predictor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        ).to(device)

        # Freeze RND target
        for param in self.rnd_target.parameters():
            param.requires_grad = False

        # Exploration policy
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        ).to(device)

        # Cell archive
        self.archive = CellArchive(
            embedding_dim, max_cells
        )

        # Optimizers
        self.encoder_optimizer = optim.Adam(
            self.encoder.parameters(), lr=learning_rate
        )
        self.rnd_optimizer = optim.Adam(
            self.rnd_predictor.parameters(), lr=learning_rate
        )
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(), lr=learning_rate
        )

        # State
        self.current_trajectory: List[int] = []
        self.step_count = 0

    def get_embedding(self, obs: np.ndarray) -> np.ndarray:
        """Get neural embedding for observation."""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            embedding = self.encoder(obs_t)
        return embedding.cpu().numpy().flatten()

    def compute_novelty(self, obs: np.ndarray) -> float:
        """Compute novelty using RND prediction error."""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            embedding = self.encoder(obs_t)

            target = self.rnd_target(embedding)
            pred = self.rnd_predictor(embedding)

            novelty = ((target - pred) ** 2).mean().item()
        return novelty

    def get_action(
        self,
        obs: np.ndarray,
        explore: bool = True,
        epsilon: float = 0.1,
    ) -> int:
        """Get action from exploration policy."""
        if explore and random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            logits = self.policy(obs_t)

            if explore:
                probs = F.softmax(logits / 0.5, dim=-1)  # Temperature
                action = torch.multinomial(probs, 1).item()
            else:
                action = logits.argmax().item()

        return action

    def process_observation(
        self,
        obs: np.ndarray,
        action: int,
        state_snapshot: Optional[Any] = None,
    ) -> Tuple[bool, float]:
        """
        Process new observation during exploration.

        Returns (is_new_cell, novelty)
        """
        self.current_trajectory.append(action)
        self.step_count += 1

        embedding = self.get_embedding(obs)
        novelty = self.compute_novelty(obs)

        is_new, key = self.archive.add_or_update(
            embedding,
            self.current_trajectory.copy(),
            state_snapshot,
            self.step_count,
        )

        self.archive.update_novelty(key, novelty)

        return is_new, novelty

    def select_cell_for_exploration(
        self,
        strategy: str = "weighted",
    ) -> Optional[Cell]:
        """Select a cell to explore from."""
        return self.archive.select_cell(strategy)

    def reset_trajectory(self):
        """Reset current trajectory (for new episode)."""
        self.current_trajectory = []

    def update_rnd(self, obs_batch: torch.Tensor) -> float:
        """Train RND predictor on observation batch."""
        embedding = self.encoder(obs_batch)

        with torch.no_grad():
            target = self.rnd_target(embedding)
        pred = self.rnd_predictor(embedding)

        loss = ((target - pred) ** 2).mean()

        self.rnd_optimizer.zero_grad()
        loss.backward()
        self.rnd_optimizer.step()

        return loss.item()

    def update_encoder_contrastive(
        self,
        obs_batch: torch.Tensor,
        temperature: float = 0.5,
    ) -> float:
        """
        Train encoder using contrastive learning.

        Augmented views of same state should have similar embeddings.
        """
        batch_size = obs_batch.size(0)

        # Create augmented views (simple noise augmentation)
        noise = torch.randn_like(obs_batch) * 0.05
        obs_aug = obs_batch + noise

        # Get embeddings
        z1 = self.encoder.project(self.encoder(obs_batch))
        z2 = self.encoder.project(self.encoder(obs_aug))

        # Normalize
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # InfoNCE loss
        sim_matrix = torch.matmul(z1, z2.T) / temperature
        labels = torch.arange(batch_size).to(self.device)

        loss = F.cross_entropy(sim_matrix, labels)

        self.encoder_optimizer.zero_grad()
        loss.backward()
        self.encoder_optimizer.step()

        return loss.item()

    def update_policy(
        self,
        obs_batch: torch.Tensor,
        action_batch: torch.Tensor,
        reward_batch: torch.Tensor,
    ) -> float:
        """
        Update exploration policy using novelty as reward.

        Simple policy gradient with novelty reward.
        """
        logits = self.policy(obs_batch)
        log_probs = F.log_softmax(logits, dim=-1)

        action_log_probs = log_probs.gather(
            1, action_batch.unsqueeze(-1)
        ).squeeze(-1)

        # Normalize rewards
        reward_batch = (reward_batch - reward_batch.mean()) / (
            reward_batch.std() + 1e-8
        )

        loss = -(action_log_probs * reward_batch).mean()

        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

        return loss.item()

    def get_statistics(self) -> Dict[str, Any]:
        """Get Go-Explore statistics."""
        stats = self.archive.get_statistics()
        stats["step_count"] = self.step_count
        return stats


class GoExploreTrainer:
    """
    Training loop for Go-Explore.

    Implements the full Go-Explore algorithm:
    1. Phase 1: Exploration (find all reachable states)
    2. Phase 2: Robustification (optional, via imitation learning)
    """

    def __init__(
        self,
        go_explore: GoExplore,
        env_factory,  # Callable that creates env
        explore_steps_per_cell: int = 100,
        batch_size: int = 256,
    ):
        self.go_explore = go_explore
        self.env_factory = env_factory
        self.explore_steps_per_cell = explore_steps_per_cell
        self.batch_size = batch_size

        # Experience buffer
        self.obs_buffer: List[np.ndarray] = []
        self.action_buffer: List[int] = []
        self.novelty_buffer: List[float] = []

    def explore_from_cell(
        self,
        cell: Cell,
        env,
    ) -> Dict[str, Any]:
        """
        Explore from a selected cell.

        1. Reset environment
        2. Replay trajectory to reach cell
        3. Explore for N steps using curiosity policy
        """
        # Reset
        obs, _ = env.reset()

        # Replay trajectory to reach cell
        for action in cell.trajectory:
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                # Failed to reach cell, just explore from here
                break

        # Explore from cell
        self.go_explore.reset_trajectory()
        self.go_explore.current_trajectory = cell.trajectory.copy()

        new_cells = 0
        total_novelty = 0.0

        for _ in range(self.explore_steps_per_cell):
            action = self.go_explore.get_action(obs, explore=True)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Try to get state snapshot if available
            snapshot = getattr(env, 'get_state', lambda: None)()

            is_new, novelty = self.go_explore.process_observation(
                next_obs, action, snapshot
            )

            if is_new:
                new_cells += 1
            total_novelty += novelty

            # Store experience
            self.obs_buffer.append(obs)
            self.action_buffer.append(action)
            self.novelty_buffer.append(novelty)

            if terminated or truncated:
                break

            obs = next_obs

        return {
            "new_cells": new_cells,
            "mean_novelty": total_novelty / self.explore_steps_per_cell,
        }

    def train_step(self) -> Dict[str, float]:
        """Train networks on buffered experience."""
        if len(self.obs_buffer) < self.batch_size:
            return {}

        # Sample batch
        indices = np.random.choice(
            len(self.obs_buffer), self.batch_size, replace=False
        )

        obs_batch = torch.FloatTensor(
            np.array([self.obs_buffer[i] for i in indices])
        ).to(self.go_explore.device)

        action_batch = torch.LongTensor(
            [self.action_buffer[i] for i in indices]
        ).to(self.go_explore.device)

        novelty_batch = torch.FloatTensor(
            [self.novelty_buffer[i] for i in indices]
        ).to(self.go_explore.device)

        # Update networks
        rnd_loss = self.go_explore.update_rnd(obs_batch)
        encoder_loss = self.go_explore.update_encoder_contrastive(obs_batch)
        policy_loss = self.go_explore.update_policy(
            obs_batch, action_batch, novelty_batch
        )

        # Clear old experience
        if len(self.obs_buffer) > 10000:
            self.obs_buffer = self.obs_buffer[-5000:]
            self.action_buffer = self.action_buffer[-5000:]
            self.novelty_buffer = self.novelty_buffer[-5000:]

        return {
            "rnd_loss": rnd_loss,
            "encoder_loss": encoder_loss,
            "policy_loss": policy_loss,
        }

    def run_exploration(
        self,
        n_iterations: int = 1000,
        log_interval: int = 50,
    ) -> List[Dict]:
        """
        Run full Go-Explore exploration loop.
        """
        history = []
        env = self.env_factory()

        # Initial exploration from start
        obs, _ = env.reset()
        self.go_explore.reset_trajectory()

        # Add start cell
        embedding = self.go_explore.get_embedding(obs)
        self.go_explore.archive.add_or_update(embedding, [], None, 0)

        for iteration in range(n_iterations):
            # Select cell
            cell = self.go_explore.select_cell_for_exploration()

            if cell is None:
                # No cells, do random exploration
                obs, _ = env.reset()
                self.go_explore.reset_trajectory()
                for _ in range(self.explore_steps_per_cell):
                    action = self.go_explore.get_action(obs, explore=True, epsilon=0.5)
                    next_obs, _, terminated, truncated, _ = env.step(action)
                    self.go_explore.process_observation(next_obs, action, None)
                    if terminated or truncated:
                        break
                    obs = next_obs
                continue

            # Explore from cell
            explore_stats = self.explore_from_cell(cell, env)

            # Train
            train_stats = self.train_step()

            # Log
            if iteration % log_interval == 0:
                archive_stats = self.go_explore.get_statistics()
                log_entry = {
                    "iteration": iteration,
                    **archive_stats,
                    **explore_stats,
                    **train_stats,
                }
                history.append(log_entry)

                print(f"Iter {iteration}: "
                      f"cells={archive_stats['n_cells']}, "
                      f"new={explore_stats['new_cells']}, "
                      f"novelty={explore_stats['mean_novelty']:.4f}")

        env.close()
        return history
