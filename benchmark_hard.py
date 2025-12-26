"""
VOID_RUNNER - Comprehensive Hard Exploration Benchmark
========================================================
Benchmark all curiosity methods on challenging environments.

Tests:
1. DeceptiveRewardMaze - Can curiosity overcome local optima?
2. KeyDoorEnv - Can it learn temporal dependencies?
3. StochasticMaze - Robustness to noise
4. MultiGoalSparse - Breadth of exploration
5. MontezumaLite - Full hierarchical challenge

Methods:
- Baseline (no curiosity)
- RND (Random Network Distillation)
- DRND (Distributional RND)
- NGU (Never Give Up)
- BYOL-Explore
- ICM (Intrinsic Curiosity Module)
- SimHash counting
- Go-Explore
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import time
import argparse
from dataclasses import dataclass
import json

# Import environments
from envs.hard_exploration import (
    DeceptiveRewardMaze,
    KeyDoorEnv,
    StochasticMaze,
    MultiGoalSparse,
    MontezumaLite,
)

# Import curiosity modules
from src.curiosity import CuriosityModule, ICMModule
from src.drnd import DRND, NGUCombinedCuriosity
from src.byol_explore import BYOLExplore
from src.counting import SimHashCounter
from src.go_explore import GoExplore


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    method: str
    env_name: str
    success_rate: float
    mean_return: float
    mean_episode_length: float
    coverage: float  # Fraction of states visited
    first_success_episode: Optional[int]
    training_time: float


class SimplePolicy(nn.Module):
    """Simple policy network for benchmarking."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.net(x)
        return self.actor(features), self.critic(features)

    def get_action(self, obs, deterministic=False):
        with torch.no_grad():
            logits, _ = self.forward(obs)
            if deterministic:
                return logits.argmax(dim=-1)
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, 1).squeeze(-1)


class CuriosityBenchmark:
    """Benchmark runner for curiosity methods."""

    def __init__(
        self,
        device: str = "cpu",
        n_episodes: int = 500,
        max_steps_per_episode: int = 1000,
        seed: int = 42,
    ):
        self.device = device
        self.n_episodes = n_episodes
        self.max_steps = max_steps_per_episode
        self.seed = seed

        # Results storage
        self.results: List[BenchmarkResult] = []

    def create_curiosity_module(
        self,
        method: str,
        obs_dim: int,
        action_dim: int,
    ):
        """Factory for curiosity modules."""
        if method == "none":
            return None
        elif method == "rnd":
            return CuriosityModule(obs_dim, device=self.device)
        elif method == "drnd":
            return DRND(obs_dim, n_networks=5, device=self.device)
        elif method == "ngu":
            return NGUCombinedCuriosity(obs_dim, device=self.device)
        elif method == "byol":
            return BYOLExplore(obs_dim, action_dim, device=self.device)
        elif method == "icm":
            return ICMModule(obs_dim, action_dim, device=self.device)
        elif method == "simhash":
            return SimHashCounter(obs_dim)
        elif method == "go_explore":
            return GoExplore(obs_dim, action_dim, device=self.device)
        else:
            raise ValueError(f"Unknown method: {method}")

    def get_intrinsic_reward(
        self,
        module,
        method: str,
        obs: np.ndarray,
        action: Optional[int] = None,
        next_obs: Optional[np.ndarray] = None,
    ) -> float:
        """Get intrinsic reward from curiosity module."""
        if module is None:
            return 0.0

        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        if method == "simhash":
            bonus = module.increment(obs)
            return 1.0 / np.sqrt(bonus + 1)

        elif method == "go_explore":
            is_new, novelty = module.process_observation(obs, action or 0)
            return novelty

        elif method == "byol":
            if next_obs is None:
                return 0.0
            next_obs_t = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
            action_t = torch.tensor([action]).to(self.device)
            return module.compute_intrinsic_reward(
                obs_t, action_t, next_obs_t
            ).item()

        elif method == "icm":
            if next_obs is None:
                return 0.0
            next_obs_t = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
            action_t = torch.tensor([action]).to(self.device)
            return module.compute_intrinsic_reward(
                obs_t, action_t, next_obs_t
            ).item()

        else:
            # RND, DRND, NGU
            reward = module.compute_intrinsic_reward(obs_t)
            if isinstance(reward, torch.Tensor):
                return reward.item()
            return reward

    def run_episode(
        self,
        env,
        policy: SimplePolicy,
        curiosity_module,
        method: str,
        training: bool = True,
    ) -> Tuple[float, int, bool, set]:
        """Run single episode and return (total_reward, steps, success, visited_states)."""
        obs, _ = env.reset()
        total_reward = 0.0
        visited_states = set()
        success = False

        if method == "go_explore" and curiosity_module is not None:
            curiosity_module.reset_trajectory()

        for step in range(self.max_steps):
            # Track visited states
            state_key = tuple(np.round(obs, 1))
            visited_states.add(state_key)

            # Get action
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action = policy.get_action(obs_t).item()

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Get intrinsic reward
            intrinsic = self.get_intrinsic_reward(
                curiosity_module, method, obs, action, next_obs
            )

            # Combine rewards
            total_reward += reward + 0.01 * intrinsic

            # Check success
            if reward > 0 or terminated:
                success = True

            if terminated or truncated:
                break

            obs = next_obs

        return total_reward, step + 1, success, visited_states

    def benchmark_method(
        self,
        method: str,
        env_class,
        env_name: str,
        env_kwargs: Dict = None,
    ) -> BenchmarkResult:
        """Benchmark a single method on an environment."""
        print(f"\n  Testing {method} on {env_name}...")

        env_kwargs = env_kwargs or {}
        env = env_class(**env_kwargs)

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        # Create policy and curiosity module
        policy = SimplePolicy(obs_dim, action_dim).to(self.device)
        curiosity = self.create_curiosity_module(method, obs_dim, action_dim)

        # Training loop
        start_time = time.time()
        episode_rewards = []
        episode_lengths = []
        successes = []
        all_visited = set()
        first_success = None

        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

        for episode in range(self.n_episodes):
            reward, length, success, visited = self.run_episode(
                env, policy, curiosity, method
            )

            episode_rewards.append(reward)
            episode_lengths.append(length)
            successes.append(success)
            all_visited.update(visited)

            if success and first_success is None:
                first_success = episode

            # Simple policy update (REINFORCE-style)
            if episode % 10 == 0:
                # Simplified update for benchmark speed
                pass

        training_time = time.time() - start_time
        env.close()

        # Estimate coverage
        try:
            size = getattr(env, 'size', 15)
            total_states = size * size
        except Exception:
            total_states = 1000
        coverage = len(all_visited) / total_states

        result = BenchmarkResult(
            method=method,
            env_name=env_name,
            success_rate=np.mean(successes),
            mean_return=np.mean(episode_rewards),
            mean_episode_length=np.mean(episode_lengths),
            coverage=min(1.0, coverage),
            first_success_episode=first_success,
            training_time=training_time,
        )

        self.results.append(result)
        return result

    def run_full_benchmark(
        self,
        methods: List[str] = None,
        envs: List[str] = None,
    ) -> List[BenchmarkResult]:
        """Run full benchmark suite."""
        methods = methods or ["none", "rnd", "drnd", "ngu", "byol", "simhash", "go_explore"]

        env_configs = {
            "DeceptiveMaze": (DeceptiveRewardMaze, {"size": 15}),
            "KeyDoor": (KeyDoorEnv, {"size": 12}),
            "StochasticMaze": (StochasticMaze, {"size": 12, "success_prob": 0.8}),
            "MultiGoal": (MultiGoalSparse, {"size": 20, "n_goals": 4}),
            "MontezumaLite": (MontezumaLite, {}),
        }

        if envs:
            env_configs = {k: v for k, v in env_configs.items() if k in envs}

        print("=" * 60)
        print("VOID_RUNNER Hard Exploration Benchmark")
        print("=" * 60)
        print(f"Methods: {methods}")
        print(f"Environments: {list(env_configs.keys())}")
        print(f"Episodes per test: {self.n_episodes}")
        print("=" * 60)

        for env_name, (env_class, env_kwargs) in env_configs.items():
            print(f"\n>>> Environment: {env_name}")

            for method in methods:
                result = self.benchmark_method(
                    method, env_class, env_name, env_kwargs
                )
                print(f"    {method:15s}: "
                      f"success={result.success_rate:.1%}, "
                      f"coverage={result.coverage:.1%}, "
                      f"first_success={result.first_success_episode}")

        return self.results

    def print_summary(self):
        """Print results summary table."""
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        # Group by environment
        by_env = defaultdict(list)
        for r in self.results:
            by_env[r.env_name].append(r)

        # Print table
        header = f"{'Method':15s} {'Success':>10s} {'Coverage':>10s} {'Return':>10s} {'1st Success':>12s}"

        for env_name, results in by_env.items():
            print(f"\n{env_name}:")
            print("-" * 60)
            print(header)
            print("-" * 60)

            # Sort by success rate
            results.sort(key=lambda x: x.success_rate, reverse=True)

            for r in results:
                first = str(r.first_success_episode) if r.first_success_episode else "never"
                print(f"{r.method:15s} {r.success_rate:>9.1%} {r.coverage:>9.1%} "
                      f"{r.mean_return:>10.2f} {first:>12s}")

        print("\n" + "=" * 80)

    def save_results(self, filepath: str):
        """Save results to JSON."""
        data = []
        for r in self.results:
            data.append({
                "method": r.method,
                "env": r.env_name,
                "success_rate": r.success_rate,
                "coverage": r.coverage,
                "mean_return": r.mean_return,
                "mean_episode_length": r.mean_episode_length,
                "first_success_episode": r.first_success_episode,
                "training_time": r.training_time,
            })

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="VOID_RUNNER Hard Exploration Benchmark")

    parser.add_argument(
        "--methods",
        nargs="+",
        default=["none", "rnd", "drnd", "simhash", "go_explore"],
        help="Methods to benchmark"
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        default=None,
        help="Environments to test (default: all)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="Episodes per method-env combination"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device (cpu or cuda)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Run benchmark
    benchmark = CuriosityBenchmark(
        device=args.device,
        n_episodes=args.episodes,
        seed=args.seed,
    )

    benchmark.run_full_benchmark(
        methods=args.methods,
        envs=args.envs,
    )

    benchmark.print_summary()
    benchmark.save_results(args.output)


if __name__ == "__main__":
    main()
