"""
VOID_RUNNER - LLM-Based Intrinsic Motivation
==============================================
Using Large Language Models for exploration guidance.

Inspired by:
1. Eureka (ICLR 2024): LLM-generated reward functions
2. ELLM: LLM-suggested exploration goals
3. Motif: Language-conditioned intrinsic motivation
4. IGE (2024): Intelligent Go-Explore with foundation models

Key ideas:
- LLMs encode commonsense about "interesting" states
- State descriptions -> LLM -> exploration goals/rewards
- Evolutionary optimization of LLM-generated rewards
- Language as abstraction for state space

Note: This module provides the framework and interfaces.
Actual LLM calls require API keys (OpenAI, Anthropic, etc.)
or local models (Llama, Mistral, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import re


@dataclass
class ExplorationGoal:
    """
    An exploration goal suggested by an LLM.

    Goals are natural language descriptions that can be
    converted to reward signals via embedding similarity.
    """
    description: str
    priority: float = 1.0
    achieved: bool = False
    creation_step: int = 0
    achievement_step: Optional[int] = None

    def to_dict(self) -> Dict:
        return {
            "description": self.description,
            "priority": self.priority,
            "achieved": self.achieved,
        }


class StateDescriber(ABC):
    """
    Abstract interface for converting states to natural language.

    Different environments need different describers:
    - Grid worlds: Position, objects, inventory
    - Atari: Frame description or OCR
    - Robotics: Joint positions, object poses
    """

    @abstractmethod
    def describe(self, state: Any) -> str:
        """Convert state to natural language description."""
        pass


class GridWorldDescriber(StateDescriber):
    """State describer for grid-based environments."""

    def __init__(
        self,
        object_names: Optional[Dict[int, str]] = None,
    ):
        self.object_names = object_names or {}

    def describe(self, state: Dict[str, Any]) -> str:
        """
        Expected state format:
        {
            "agent_pos": (x, y),
            "objects": [(obj_type, x, y), ...],
            "inventory": [obj_type, ...],
            "visited": [(x, y), ...],
        }
        """
        parts = []

        # Agent position
        if "agent_pos" in state:
            x, y = state["agent_pos"]
            parts.append(f"Agent is at position ({x}, {y})")

        # Nearby objects
        if "objects" in state:
            for obj_type, ox, oy in state["objects"]:
                obj_name = self.object_names.get(obj_type, f"object_{obj_type}")
                parts.append(f"There is a {obj_name} at ({ox}, {oy})")

        # Inventory
        if "inventory" in state:
            inv_names = [
                self.object_names.get(i, f"item_{i}")
                for i in state["inventory"]
            ]
            if inv_names:
                parts.append(f"Agent has: {', '.join(inv_names)}")

        # Exploration coverage
        if "visited" in state:
            parts.append(f"Explored {len(state['visited'])} locations")

        return ". ".join(parts) + "."


class RoboticsDescriber(StateDescriber):
    """State describer for robotics environments."""

    def describe(self, state: Dict[str, Any]) -> str:
        """
        Expected state format:
        {
            "end_effector_pos": (x, y, z),
            "gripper_state": "open" | "closed",
            "object_positions": [(name, x, y, z), ...],
            "task_progress": float,
        }
        """
        parts = []

        if "end_effector_pos" in state:
            x, y, z = state["end_effector_pos"]
            parts.append(f"Robot arm at ({x:.2f}, {y:.2f}, {z:.2f})")

        if "gripper_state" in state:
            parts.append(f"Gripper is {state['gripper_state']}")

        if "object_positions" in state:
            for name, x, y, z in state["object_positions"]:
                parts.append(f"{name} at ({x:.2f}, {y:.2f}, {z:.2f})")

        if "task_progress" in state:
            parts.append(f"Task {state['task_progress']*100:.0f}% complete")

        return ". ".join(parts) + "."


class LLMInterface(ABC):
    """
    Abstract interface for LLM providers.

    Implementations connect to specific APIs:
    - OpenAI (GPT-4, GPT-3.5)
    - Anthropic (Claude)
    - Local (Llama, Mistral via vLLM/ollama)
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """Generate text completion."""
        pass

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Get embedding for text."""
        pass


class MockLLM(LLMInterface):
    """
    Mock LLM for testing without API calls.

    Uses heuristics to generate reasonable-looking outputs.
    """

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.goal_templates = [
            "Try to reach the corner at ({}, {})",
            "Find and collect the hidden item",
            "Explore the unexplored region in the {} direction",
            "Complete a full loop of the environment",
            "Find the shortest path to the goal",
            "Visit all four corners",
            "Collect all items in the environment",
            "Find the secret passage",
            "Maximize exploration coverage",
            "Discover new state transitions",
        ]

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """Generate mock response based on prompt analysis."""
        # Parse state from prompt if possible
        if "suggest exploration goals" in prompt.lower():
            goals = np.random.choice(self.goal_templates, size=3, replace=False)
            return "Here are some exploration goals:\n" + "\n".join(
                f"{i+1}. {g.format(np.random.randint(0, 10), np.random.randint(0, 10))}"
                for i, g in enumerate(goals)
            )

        elif "evaluate" in prompt.lower() or "reward" in prompt.lower():
            score = np.random.uniform(0, 1)
            return f"Novelty score: {score:.2f}. This state is {'interesting' if score > 0.5 else 'common'}."

        else:
            return "I understand the state. Continue exploring."

    def embed(self, text: str) -> np.ndarray:
        """Generate deterministic embedding based on text hash."""
        # Use text hash for reproducibility
        text_hash = hash(text) % (2**32)
        np.random.seed(text_hash)
        embedding = np.random.randn(self.embedding_dim)
        embedding = embedding / np.linalg.norm(embedding)
        np.random.seed()  # Reset
        return embedding


class SentenceEmbedder(nn.Module):
    """
    Simple neural sentence embedder for goal matching.

    In production, use SentenceBERT or similar.
    This is a learnable approximation.
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim // 2,
            batch_first=True, bidirectional=True
        )
        self.output = nn.Linear(hidden_dim, embedding_dim)

        # Simple tokenizer (word-based)
        self.vocab: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

    def tokenize(self, text: str) -> List[int]:
        """Simple word tokenization."""
        words = re.findall(r'\w+', text.lower())
        tokens = []
        for word in words:
            if word not in self.vocab and len(self.vocab) < self.vocab_size:
                self.vocab[word] = len(self.vocab)
            tokens.append(self.vocab.get(word, 1))  # 1 = <UNK>
        return tokens

    def forward(self, text: str) -> torch.Tensor:
        tokens = self.tokenize(text)
        if not tokens:
            tokens = [1]  # <UNK>

        x = torch.LongTensor([tokens])
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)

        # Concatenate forward and backward hidden states
        h = torch.cat([h[0], h[1]], dim=-1)
        embedding = self.output(h)
        embedding = F.normalize(embedding, dim=-1)

        return embedding.squeeze(0)


class ELLMExplorer:
    """
    ELLM-style exploration with LLM-suggested goals.

    Algorithm:
    1. Describe current state in natural language
    2. Ask LLM to suggest exploration goals
    3. Compute similarity between achieved states and goals
    4. Use similarity as intrinsic reward

    Reference: "Guiding Pretraining in RL with LLMs" (Du et al.)
    """

    def __init__(
        self,
        state_describer: StateDescriber,
        llm: LLMInterface,
        embedding_dim: int = 128,
        goal_refresh_interval: int = 100,
        max_goals: int = 10,
        device: str = "cpu",
    ):
        self.state_describer = state_describer
        self.llm = llm
        self.device = device
        self.goal_refresh_interval = goal_refresh_interval
        self.max_goals = max_goals

        # Goal tracking
        self.goals: List[ExplorationGoal] = []
        self.goal_embeddings: List[np.ndarray] = []
        self.steps_since_refresh = 0
        self.step_count = 0

        # Sentence embedder for goal matching
        self.embedder = SentenceEmbedder(embedding_dim=embedding_dim).to(device)

        # History for LLM context
        self.achieved_goals: List[str] = []
        self.recent_states: List[str] = []

    def _generate_goals(self, state_description: str) -> List[ExplorationGoal]:
        """Ask LLM to suggest exploration goals."""
        prompt = f"""Current environment state:
{state_description}

Previously achieved goals:
{chr(10).join(self.achieved_goals[-5:]) if self.achieved_goals else "None yet"}

Recently visited states:
{chr(10).join(self.recent_states[-3:]) if self.recent_states else "None"}

Based on this information, suggest {self.max_goals} diverse exploration goals.
Each goal should be:
1. Achievable from the current state
2. Novel (not recently achieved)
3. Potentially useful for learning

Format: One goal per line, numbered 1-{self.max_goals}."""

        response = self.llm.generate(prompt, max_tokens=512, temperature=0.8)

        # Parse goals from response
        goals = []
        lines = response.strip().split("\n")

        for line in lines:
            # Remove numbering
            line = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
            if len(line) > 5:  # Filter very short lines
                goals.append(ExplorationGoal(
                    description=line,
                    priority=1.0,
                    creation_step=self.step_count,
                ))

        return goals[:self.max_goals]

    def refresh_goals(self, state_description: str):
        """Refresh goal list based on current state."""
        new_goals = self._generate_goals(state_description)

        # Keep unachieved high-priority goals
        kept_goals = [
            g for g in self.goals
            if not g.achieved and g.priority > 0.5
        ][:self.max_goals // 2]

        self.goals = kept_goals + new_goals
        self.goals = self.goals[:self.max_goals]

        # Update embeddings
        self.goal_embeddings = []
        for goal in self.goals:
            emb = self.llm.embed(goal.description)
            self.goal_embeddings.append(emb)

        self.steps_since_refresh = 0

    def compute_intrinsic_reward(
        self,
        state: Any,
        update_goals: bool = True,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute intrinsic reward based on goal similarity.

        Returns (reward, info_dict)
        """
        self.step_count += 1
        self.steps_since_refresh += 1

        # Describe state
        description = self.state_describer.describe(state)
        self.recent_states.append(description)
        self.recent_states = self.recent_states[-10:]  # Keep last 10

        # Refresh goals if needed
        if update_goals and self.steps_since_refresh >= self.goal_refresh_interval:
            self.refresh_goals(description)

        if not self.goals:
            self.refresh_goals(description)
            if not self.goals:
                return 0.0, {"matched_goal": None}

        # Compute similarity to each goal
        state_embedding = self.llm.embed(description)

        similarities = []
        for i, goal_emb in enumerate(self.goal_embeddings):
            sim = np.dot(state_embedding, goal_emb)
            similarities.append((sim, i))

        # Best match
        best_sim, best_idx = max(similarities)
        matched_goal = self.goals[best_idx]

        # Check if goal achieved
        info = {"matched_goal": matched_goal.description, "similarity": best_sim}

        if best_sim > 0.8:  # Achievement threshold
            if not matched_goal.achieved:
                matched_goal.achieved = True
                matched_goal.achievement_step = self.step_count
                self.achieved_goals.append(matched_goal.description)
                info["goal_achieved"] = True

        # Reward is similarity to best matching goal
        reward = max(0.0, best_sim)

        # Boost reward for novel achievements
        if info.get("goal_achieved"):
            reward *= 2.0

        return reward, info


class EurekaRewardGenerator:
    """
    Eureka-style reward function generation.

    Instead of hard-coding rewards, LLM writes reward code
    that is then optimized via evolutionary search.

    Reference: "Eureka: Human-Level Reward Design via Coding LLMs" (ICLR 2024)
    """

    def __init__(
        self,
        llm: LLMInterface,
        env_description: str,
        task_description: str,
        population_size: int = 8,
        generations: int = 5,
    ):
        self.llm = llm
        self.env_description = env_description
        self.task_description = task_description
        self.population_size = population_size
        self.generations = generations

        # Population of reward functions
        self.population: List[Dict] = []
        self.best_reward_fn: Optional[Callable] = None
        self.best_fitness: float = float('-inf')

    def _generate_reward_code(self, feedback: str = "") -> str:
        """Ask LLM to generate reward function code."""
        prompt = f"""Environment Description:
{self.env_description}

Task:
{self.task_description}

{f"Previous attempt feedback: {feedback}" if feedback else ""}

Write a Python reward function with this signature:
def compute_reward(state: dict, action: int, next_state: dict) -> float:
    '''
    Compute reward for the given transition.

    Args:
        state: Current state dictionary
        action: Action taken (integer)
        next_state: Resulting state dictionary

    Returns:
        float: Reward value
    '''
    # Your implementation here

The reward function should encourage the agent to accomplish the task.
Be creative but ensure the reward is well-shaped (not too sparse).
Only output the Python code, no explanation."""

        return self.llm.generate(prompt, max_tokens=512, temperature=0.7)

    def _parse_reward_function(self, code: str) -> Optional[Callable]:
        """Parse LLM-generated code into callable function."""
        try:
            # Extract function definition
            match = re.search(
                r"def compute_reward\([^)]*\)[^:]*:.*?(?=\ndef|\Z)",
                code,
                re.DOTALL
            )
            if not match:
                return None

            fn_code = match.group()

            # Create function in isolated namespace
            namespace = {"np": np}
            exec(fn_code, namespace)

            return namespace.get("compute_reward")

        except Exception as e:
            print(f"Failed to parse reward function: {e}")
            return None

    def _evaluate_reward_function(
        self,
        reward_fn: Callable,
        rollouts: List[List[Tuple]],  # List of trajectories
    ) -> float:
        """
        Evaluate fitness of a reward function.

        Fitness based on:
        1. Does it produce non-zero rewards?
        2. Is reward correlated with task progress?
        3. Is reward well-shaped (not too sparse)?
        """
        total_rewards = []
        reward_variance = []

        for trajectory in rollouts:
            traj_rewards = []
            for state, action, next_state, info in trajectory:
                try:
                    r = reward_fn(state, action, next_state)
                    traj_rewards.append(r)
                except Exception:
                    traj_rewards.append(0.0)

            total_rewards.append(sum(traj_rewards))
            if len(traj_rewards) > 1:
                reward_variance.append(np.var(traj_rewards))

        # Fitness components
        mean_total = np.mean(total_rewards) if total_rewards else 0
        mean_variance = np.mean(reward_variance) if reward_variance else 0

        # Good reward functions have:
        # - High mean reward for successful trajectories
        # - Non-zero variance (well-shaped, not flat)
        # - Reasonable scale
        fitness = mean_total * (1 + np.sqrt(mean_variance))

        return fitness

    def evolve(
        self,
        rollouts: List[List[Tuple]],
        verbose: bool = True,
    ) -> Callable:
        """
        Evolve population of reward functions.

        Returns best reward function found.
        """
        # Initialize population
        if verbose:
            print("Generating initial population...")

        for _ in range(self.population_size):
            code = self._generate_reward_code()
            fn = self._parse_reward_function(code)
            if fn is not None:
                fitness = self._evaluate_reward_function(fn, rollouts)
                self.population.append({
                    "code": code,
                    "fn": fn,
                    "fitness": fitness,
                })

        # Evolution loop
        for gen in range(self.generations):
            if verbose:
                fitnesses = [p["fitness"] for p in self.population]
                print(f"Generation {gen}: "
                      f"best={max(fitnesses):.3f}, "
                      f"mean={np.mean(fitnesses):.3f}")

            # Sort by fitness
            self.population.sort(key=lambda x: x["fitness"], reverse=True)

            # Keep top half
            survivors = self.population[:len(self.population) // 2]

            # Generate new candidates with feedback
            new_population = survivors.copy()

            for parent in survivors:
                # Mutate with LLM
                feedback = f"Previous fitness: {parent['fitness']:.3f}. Try to improve."
                new_code = self._generate_reward_code(feedback)
                new_fn = self._parse_reward_function(new_code)

                if new_fn is not None:
                    fitness = self._evaluate_reward_function(new_fn, rollouts)
                    new_population.append({
                        "code": new_code,
                        "fn": new_fn,
                        "fitness": fitness,
                    })

            self.population = new_population

        # Return best
        self.population.sort(key=lambda x: x["fitness"], reverse=True)
        best = self.population[0]
        self.best_reward_fn = best["fn"]
        self.best_fitness = best["fitness"]

        if verbose:
            print(f"Best fitness: {best['fitness']:.3f}")
            print("Best reward code:")
            print(best["code"][:500] + "..." if len(best["code"]) > 500 else best["code"])

        return best["fn"]


class LanguageConditionedCuriosity:
    """
    Language-conditioned curiosity module.

    Uses language descriptions to:
    1. Define what's "interesting" (via LLM)
    2. Guide exploration toward described goals
    3. Provide hierarchical abstraction over states

    This combines ideas from:
    - ELLM (goal suggestion)
    - Eureka (reward shaping)
    - Motif (language abstraction)
    """

    def __init__(
        self,
        state_describer: StateDescriber,
        llm: LLMInterface,
        embedding_dim: int = 128,
        device: str = "cpu",
    ):
        self.state_describer = state_describer
        self.llm = llm
        self.device = device

        # ELLM component
        self.ellm = ELLMExplorer(
            state_describer, llm, embedding_dim, device=device
        )

        # State novelty via language
        self.state_descriptions_seen: Dict[str, int] = {}

        # Language abstraction levels
        self.abstract_states: Dict[str, List[str]] = {}  # abstract -> concrete

    def get_abstract_state(self, state: Any) -> str:
        """Get abstract language description of state."""
        description = self.state_describer.describe(state)

        prompt = f"""Given this detailed state description:
{description}

Provide a brief, abstract summary (max 10 words) capturing the essential state:"""

        abstract = self.llm.generate(prompt, max_tokens=32, temperature=0.3)
        abstract = abstract.strip()[:100]

        # Track abstraction
        if abstract not in self.abstract_states:
            self.abstract_states[abstract] = []
        self.abstract_states[abstract].append(description)

        return abstract

    def compute_intrinsic_reward(
        self,
        state: Any,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute language-conditioned intrinsic reward.

        Combines:
        1. Goal-matching reward (ELLM)
        2. Language novelty (new descriptions)
        3. Abstract state novelty
        """
        # Get descriptions
        description = self.state_describer.describe(state)
        abstract = self.get_abstract_state(state)

        # Language novelty
        if description not in self.state_descriptions_seen:
            self.state_descriptions_seen[description] = 0
            language_novelty = 1.0
        else:
            self.state_descriptions_seen[description] += 1
            language_novelty = 1.0 / np.sqrt(
                self.state_descriptions_seen[description] + 1
            )

        # Abstract novelty
        abstract_count = len(self.abstract_states.get(abstract, []))
        abstract_novelty = 1.0 / np.sqrt(abstract_count + 1)

        # Goal-matching reward
        goal_reward, goal_info = self.ellm.compute_intrinsic_reward(state)

        # Combine
        total_reward = (
            0.4 * goal_reward +
            0.3 * language_novelty +
            0.3 * abstract_novelty
        )

        info = {
            "goal_reward": goal_reward,
            "language_novelty": language_novelty,
            "abstract_novelty": abstract_novelty,
            "abstract_state": abstract,
            **goal_info,
        }

        return total_reward, info

    def get_exploration_summary(self) -> str:
        """Get LLM-generated summary of exploration."""
        achieved = self.ellm.achieved_goals[-10:]
        abstracts = list(self.abstract_states.keys())[:20]

        prompt = f"""Exploration Summary:

Goals achieved:
{chr(10).join(achieved) if achieved else "None"}

Abstract states discovered:
{chr(10).join(abstracts) if abstracts else "None"}

Total unique states: {len(self.state_descriptions_seen)}

Provide a brief analysis of the exploration progress and suggest next steps:"""

        return self.llm.generate(prompt, max_tokens=256)
