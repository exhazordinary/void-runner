"""
VOID_RUNNER - Episodic Curiosity Memory
========================================
Memory-augmented exploration with temporal curiosity dynamics.

Key insight: Curiosity should have MEMORY and TIME dynamics.

Problems with memoryless curiosity (like RND):
1. Once you've seen something, curiosity goes to zero forever
2. No way to "re-explore" - what if the environment changed?
3. No distinction between "recently boring" vs "boring long ago"

Episodic curiosity addresses this:
- Maintain explicit memory of interesting experiences
- Curiosity DECAYS over time (half-life of novelty)
- Periodic "curiosity resets" for re-exploration
- Goal-conditioned exploration: "go back to that interesting state"

This is inspired by how humans explore:
- We remember interesting places
- We sometimes revisit them to see what's changed
- Memory of exploration guides future exploration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from collections import deque
import heapq


@dataclass
class EpisodicMemory:
    """A single memory of an interesting state."""
    state: np.ndarray
    curiosity: float  # How interesting it was when first seen
    visit_count: int
    last_visit: int  # Timestep of last visit
    first_seen: int  # Timestep of first discovery


class CuriosityMemoryBank:
    """
    Episodic memory bank for curiosity-driven exploration.

    Stores interesting states with their curiosity values,
    allowing the agent to revisit them or use them as goals.
    """

    def __init__(
        self,
        capacity: int = 10000,
        state_dim: int = 2,
        curiosity_threshold: float = 0.5,
        decay_rate: float = 0.001,  # Curiosity decay per timestep
        revisit_bonus: float = 0.1,  # Bonus for revisiting after long time
    ):
        self.capacity = capacity
        self.state_dim = state_dim
        self.curiosity_threshold = curiosity_threshold
        self.decay_rate = decay_rate
        self.revisit_bonus = revisit_bonus

        self.memories: List[EpisodicMemory] = []
        self.state_index: Dict[tuple, int] = {}  # For fast lookup
        self.current_step = 0

        # Statistics
        self.total_memories_added = 0
        self.total_revisits = 0

    def _state_key(self, state: np.ndarray) -> tuple:
        """Convert state to hashable key (discretized)."""
        return tuple(np.round(state, decimals=1))

    def add(self, state: np.ndarray, curiosity: float) -> bool:
        """
        Add a state to memory if it's interesting enough.

        Returns True if memory was added/updated.
        """
        key = self._state_key(state)

        if key in self.state_index:
            # Update existing memory
            idx = self.state_index[key]
            memory = self.memories[idx]
            memory.visit_count += 1
            time_since_last = self.current_step - memory.last_visit
            memory.last_visit = self.current_step

            # Revisit bonus: high if we haven't been here in a while
            if time_since_last > 100:
                self.total_revisits += 1
                return True
            return False

        if curiosity < self.curiosity_threshold:
            return False

        # Add new memory
        memory = EpisodicMemory(
            state=state.copy(),
            curiosity=curiosity,
            visit_count=1,
            last_visit=self.current_step,
            first_seen=self.current_step
        )

        if len(self.memories) >= self.capacity:
            # Remove least interesting memory
            self._evict_least_interesting()

        self.memories.append(memory)
        self.state_index[key] = len(self.memories) - 1
        self.total_memories_added += 1
        return True

    def _evict_least_interesting(self):
        """Remove the memory with lowest current curiosity value."""
        if not self.memories:
            return

        # Find memory with lowest decayed curiosity
        min_curiosity = float('inf')
        min_idx = 0

        for i, memory in enumerate(self.memories):
            decayed = self._get_decayed_curiosity(memory)
            if decayed < min_curiosity:
                min_curiosity = decayed
                min_idx = i

        # Remove from index
        key = self._state_key(self.memories[min_idx].state)
        if key in self.state_index:
            del self.state_index[key]

        # Remove memory
        self.memories.pop(min_idx)

        # Rebuild index (expensive but rare)
        self.state_index = {
            self._state_key(m.state): i
            for i, m in enumerate(self.memories)
        }

    def _get_decayed_curiosity(self, memory: EpisodicMemory) -> float:
        """Get curiosity value with temporal decay."""
        time_since_seen = self.current_step - memory.last_visit
        decay = np.exp(-self.decay_rate * time_since_seen)
        return memory.curiosity * decay

    def get_curiosity_bonus(self, state: np.ndarray) -> float:
        """
        Get curiosity bonus for visiting a state.

        Returns high value for:
        - States never seen before
        - States not visited in a long time
        - States that were previously very interesting
        """
        key = self._state_key(state)

        if key not in self.state_index:
            return 1.0  # Maximum curiosity for new states

        memory = self.memories[self.state_index[key]]
        time_since_last = self.current_step - memory.last_visit

        # Revisit bonus grows over time
        revisit_curiosity = 1.0 - np.exp(-self.revisit_bonus * time_since_last)

        # Combine with original curiosity (decayed)
        base_curiosity = self._get_decayed_curiosity(memory)

        return max(revisit_curiosity, 0.1 * base_curiosity)

    def sample_goal(self, current_state: np.ndarray, k: int = 5) -> Optional[np.ndarray]:
        """
        Sample an interesting goal state to explore toward.

        Prefers states that:
        1. Are far from current position
        2. Have high decayed curiosity
        3. Haven't been visited recently
        """
        if len(self.memories) < k:
            return None

        # Score each memory
        scores = []
        for memory in self.memories:
            distance = np.linalg.norm(memory.state - current_state)
            curiosity = self._get_decayed_curiosity(memory)
            time_since = self.current_step - memory.last_visit

            # Combined score
            score = curiosity * (1 + 0.1 * distance) * (1 + 0.01 * time_since)
            scores.append(score)

        # Softmax sampling from top-k
        scores = np.array(scores)
        top_k_idx = np.argsort(scores)[-k:]
        top_k_scores = scores[top_k_idx]

        probs = np.exp(top_k_scores - top_k_scores.max())
        probs /= probs.sum()

        chosen_idx = np.random.choice(top_k_idx, p=probs)
        return self.memories[chosen_idx].state.copy()

    def step(self):
        """Advance timestep."""
        self.current_step += 1

    def get_exploration_map(self, size: int = 20) -> np.ndarray:
        """
        Generate a 2D exploration heatmap from memories.

        Useful for visualization.
        """
        heatmap = np.zeros((size, size))

        for memory in self.memories:
            x = int(np.clip(memory.state[0], 0, size - 1))
            y = int(np.clip(memory.state[1], 0, size - 1))
            heatmap[y, x] += self._get_decayed_curiosity(memory)

        return heatmap / (heatmap.max() + 1e-8)


class EpisodicCuriosityModule:
    """
    Combines RND-style novelty with episodic memory.

    The reward signal includes:
    1. RND prediction error (instant novelty)
    2. Episodic bonus (memory-based novelty)
    3. Reachability bonus (can we get back to interesting states?)
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 256,
        embedding_dim: int = 128,
        memory_capacity: int = 5000,
        learning_rate: float = 1e-4,
        device: str = "cpu"
    ):
        self.device = device
        self.obs_dim = obs_dim
        self.embedding_dim = embedding_dim

        # Embedding network for state comparison
        self.embedding_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        ).to(device)

        # Comparator network: predicts if two states are "similar"
        self.comparator = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        ).to(device)

        # Reachability network: predicts if state B is reachable from state A
        self.reachability = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        ).to(device)

        self.optimizer = optim.Adam(
            list(self.embedding_net.parameters()) +
            list(self.comparator.parameters()) +
            list(self.reachability.parameters()),
            lr=learning_rate
        )

        # Episodic memory
        self.memory = CuriosityMemoryBank(
            capacity=memory_capacity,
            state_dim=obs_dim
        )

        # Embedding memory for fast similarity computation
        self.embedding_memory = deque(maxlen=memory_capacity)

    def embed(self, obs: torch.Tensor) -> torch.Tensor:
        """Get embedding of observation."""
        return F.normalize(self.embedding_net(obs), dim=-1)

    def compute_episodic_curiosity(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute curiosity based on how different next_obs is from memory.

        Uses k-nearest neighbors in embedding space.
        """
        next_embedding = self.embed(next_obs)

        if len(self.embedding_memory) < 10:
            # Not enough memories yet
            return torch.ones(next_obs.shape[0], device=self.device)

        # Compare to recent embeddings
        memory_embeddings = torch.stack(list(self.embedding_memory)[-500:])

        # Cosine similarity to all memories
        similarities = torch.mm(next_embedding, memory_embeddings.T)

        # Curiosity = inverse of max similarity (novel = dissimilar to all memories)
        max_similarity, _ = similarities.max(dim=-1)
        curiosity = 1.0 - max_similarity

        return curiosity

    def compute_intrinsic_reward(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        update: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined episodic curiosity reward.
        """
        with torch.no_grad():
            # Embedding-based novelty
            episodic_curiosity = self.compute_episodic_curiosity(obs, next_obs)

            # Memory-based bonus
            memory_bonus = torch.tensor([
                self.memory.get_curiosity_bonus(s.cpu().numpy())
                for s in next_obs
            ], device=self.device)

        # Combined reward
        intrinsic_reward = 0.5 * episodic_curiosity + 0.5 * memory_bonus

        if update:
            # Store embeddings
            next_embedding = self.embed(next_obs).detach()
            for emb in next_embedding:
                self.embedding_memory.append(emb)

            # Update memory bank
            for s, c in zip(next_obs, intrinsic_reward):
                self.memory.add(s.cpu().numpy(), c.item())

            self.memory.step()

        return intrinsic_reward, {
            'episodic_curiosity': episodic_curiosity.mean().item(),
            'memory_bonus': memory_bonus.mean().item(),
            'memory_size': len(self.memory.memories)
        }

    def get_exploration_goal(self, current_obs: np.ndarray) -> Optional[np.ndarray]:
        """Sample a goal from memory for directed exploration."""
        return self.memory.sample_goal(current_obs)


class TemporalDifferenceNovelty:
    """
    Temporal difference novelty - curiosity about CHANGE over time.

    Instead of asking "is this state novel?", asks:
    "Has my understanding of this state CHANGED recently?"

    This captures re-learning and environment non-stationarity.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 256,
        history_len: int = 100,
        learning_rate: float = 1e-4,
        device: str = "cpu"
    ):
        self.device = device
        self.history_len = history_len

        # Prediction network
        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        ).to(device)

        self.optimizer = optim.Adam(self.predictor.parameters(), lr=learning_rate)

        # Store prediction error history for each state
        self.error_history = deque(maxlen=10000)

    def compute_td_novelty(
        self,
        obs: torch.Tensor,
        update: bool = True
    ) -> torch.Tensor:
        """
        Compute temporal difference in prediction error.

        High TD novelty = my predictions for this state are changing rapidly
        = I'm still learning about it = interesting!
        """
        # Current prediction error (self-prediction task)
        prediction = self.predictor(obs)
        current_error = torch.mean((prediction - obs) ** 2, dim=-1)

        # This is a simplified version - in practice you'd track per-state history
        novelty = current_error  # For now, just use current error

        if update:
            loss = current_error.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return novelty.detach()
