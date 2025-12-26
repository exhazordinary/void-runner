"""
VOID_RUNNER - Comprehensive Exploration Benchmark
==================================================
Compare different intrinsic motivation methods on sparse reward tasks.

Methods benchmarked:
1. No intrinsic motivation (baseline)
2. RND (Random Network Distillation)
3. DRND (Distributional RND - ICML 2024)
4. NGU (Never Give Up - episodic + lifelong)
5. SimHash counting
6. ICM (Intrinsic Curiosity Module)
7. Compound curiosity (our multi-objective system)

Metrics:
- State coverage (unique states visited)
- Goal discovery rate
- Sample efficiency
- Exploration entropy
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from datetime import datetime
import json
from typing import Dict, List, Tuple

from src.networks import PolicyNetwork
from src.curiosity import CuriosityModule
from src.drnd import DRND, NGUCombinedCuriosity
from src.counting import SimHashCounter, StateActionCounter, GoExploreArchive
from src.metrics import ExplorationMetrics
from envs.sparse_maze import SparseMazeEnv, SparseGridWorldEnv


class BenchmarkAgent:
    """Simple PPO agent with pluggable intrinsic motivation."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        curiosity_module,
        device: str = "cpu"
    ):
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.curiosity = curiosity_module

        # Simple actor-critic
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        ).to(device)

        self.actor = nn.Linear(128, action_dim).to(device)
        self.critic = nn.Linear(128, 1).to(device)

        params = (
            list(self.policy.parameters()) +
            list(self.actor.parameters()) +
            list(self.critic.parameters())
        )
        self.optimizer = optim.Adam(params, lr=3e-4)

        # Rollout storage
        self.buffer = {
            'obs': [], 'actions': [], 'log_probs': [],
            'rewards': [], 'values': [], 'dones': [], 'next_obs': []
        }

    def select_action(self, obs: np.ndarray) -> Tuple[int, float, float]:
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            features = self.policy(obs_t)
            logits = self.actor(features)
            value = self.critic(features)

            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            return (
                action.item(),
                dist.log_prob(action).item(),
                value.item()
            )

    def store(self, obs, action, log_prob, reward, value, done, next_obs):
        self.buffer['obs'].append(obs)
        self.buffer['actions'].append(action)
        self.buffer['log_probs'].append(log_prob)
        self.buffer['rewards'].append(reward)
        self.buffer['values'].append(value)
        self.buffer['dones'].append(done)
        self.buffer['next_obs'].append(next_obs)

    def update(self, next_obs: np.ndarray, gamma: float = 0.99) -> Dict:
        if len(self.buffer['obs']) == 0:
            return {}

        # Convert to tensors
        obs = torch.FloatTensor(np.array(self.buffer['obs'])).to(self.device)
        actions = torch.LongTensor(self.buffer['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer['log_probs']).to(self.device)
        next_obs_t = torch.FloatTensor(np.array(self.buffer['next_obs'])).to(self.device)

        # Compute intrinsic rewards
        if self.curiosity is not None:
            if hasattr(self.curiosity, 'compute_bonus_batch'):
                # Count-based
                int_rewards = self.curiosity.compute_bonus_batch(
                    np.array(self.buffer['next_obs'])
                )
                int_rewards = torch.FloatTensor(int_rewards).to(self.device)
            elif hasattr(self.curiosity, 'compute_intrinsic_reward'):
                # Neural-based
                result = self.curiosity.compute_intrinsic_reward(next_obs_t, update=True)
                if isinstance(result, tuple):
                    int_rewards = result[0]
                else:
                    int_rewards = result
            else:
                int_rewards = torch.zeros(len(obs), device=self.device)
        else:
            int_rewards = torch.zeros(len(obs), device=self.device)

        # Combine rewards
        ext_rewards = torch.FloatTensor(self.buffer['rewards']).to(self.device)
        total_rewards = ext_rewards + 0.1 * int_rewards

        # Compute returns
        values = torch.FloatTensor(self.buffer['values']).to(self.device)
        dones = torch.FloatTensor(self.buffer['dones']).to(self.device)

        with torch.no_grad():
            next_obs_final = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
            next_value = self.critic(self.policy(next_obs_final)).item()

        returns = []
        R = next_value
        for r, d in zip(reversed(total_rewards.cpu().numpy()),
                       reversed(dones.cpu().numpy())):
            R = r + gamma * R * (1 - d)
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(self.device)

        # PPO update
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(4):
            features = self.policy(obs)
            logits = self.actor(features)
            new_values = self.critic(features).squeeze()

            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(new_values, returns)
            entropy_loss = -entropy.mean()

            loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.policy.parameters()) +
                list(self.actor.parameters()) +
                list(self.critic.parameters()),
                0.5
            )
            self.optimizer.step()

        # Clear buffer
        for k in self.buffer:
            self.buffer[k] = []

        return {
            'loss': loss.item(),
            'intrinsic_reward': int_rewards.mean().item(),
            'extrinsic_reward': ext_rewards.mean().item()
        }


def run_benchmark(
    method: str,
    env_name: str = "SparseMaze-v0",
    total_steps: int = 50000,
    seed: int = 42,
    device: str = "cpu"
) -> Dict:
    """Run benchmark for a single method."""

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create environment
    if env_name == "SparseMaze-v0":
        env = SparseMazeEnv(size=15, walls=True)
    else:
        env = SparseGridWorldEnv(size=30)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Create curiosity module based on method
    if method == "none":
        curiosity = None
    elif method == "rnd":
        curiosity = CuriosityModule(obs_dim, device=device)
    elif method == "drnd":
        curiosity = DRND(obs_dim, device=device)
    elif method == "ngu":
        curiosity = NGUCombinedCuriosity(obs_dim, device=device)
    elif method == "simhash":
        curiosity = SimHashCounter(obs_dim)
    elif method == "vcsap":
        curiosity = StateActionCounter(obs_dim, action_dim)
    else:
        raise ValueError(f"Unknown method: {method}")

    agent = BenchmarkAgent(obs_dim, action_dim, curiosity, device)
    metrics = ExplorationMetrics(obs_dim, action_dim)

    # Training loop
    obs, info = env.reset(seed=seed)
    episode_reward = 0
    episode_length = 0
    goals_reached = 0
    episode_count = 0

    rewards_history = []
    coverage_history = []
    goals_history = []

    rollout_steps = 128
    log_freq = 5000

    for step in range(1, total_steps + 1):
        action, log_prob, value = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.store(obs, action, log_prob, reward, value, float(done), next_obs)
        metrics.update(obs, action, next_obs)

        episode_reward += reward
        episode_length += 1

        if terminated and reward > 0:
            goals_reached += 1

        if done:
            rewards_history.append(episode_reward)
            episode_reward = 0
            episode_length = 0
            episode_count += 1

            # Reset episodic memory if applicable
            if hasattr(curiosity, 'reset_episode'):
                curiosity.reset_episode()

            obs, info = env.reset()
        else:
            obs = next_obs

        if step % rollout_steps == 0:
            agent.update(next_obs)

        if step % log_freq == 0:
            exp_metrics = metrics.get_all_metrics()
            coverage_history.append(exp_metrics['coverage_rate'])
            goals_history.append(goals_reached)

    # Final metrics
    final_metrics = metrics.get_all_metrics()

    return {
        'method': method,
        'env': env_name,
        'seed': seed,
        'total_steps': total_steps,
        'goals_reached': goals_reached,
        'episodes': episode_count,
        'final_coverage': final_metrics['coverage_rate'],
        'state_entropy': final_metrics['state_entropy'],
        'action_entropy': final_metrics['action_entropy'],
        'avg_reward': np.mean(rewards_history[-100:]) if rewards_history else 0,
        'coverage_history': coverage_history,
        'goals_history': goals_history,
    }


def main():
    parser = argparse.ArgumentParser(description="VOID_RUNNER Benchmark")
    parser.add_argument("--env", default="SparseMaze-v0")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="benchmark_results")
    parser.add_argument("--methods", nargs="+",
                        default=["none", "rnd", "drnd", "simhash", "vcsap"])

    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║              VOID_RUNNER EXPLORATION BENCHMARK               ║
╠══════════════════════════════════════════════════════════════╣
║  Environment: {args.env:<44} ║
║  Steps: {args.steps:<49} ║
║  Methods: {', '.join(args.methods):<46} ║
╚══════════════════════════════════════════════════════════════╝
""")

    results = []

    for method in args.methods:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {method.upper()}")
        print('='*60)

        method_results = []
        for seed in range(args.seeds):
            print(f"  Seed {seed + 1}/{args.seeds}...", end=" ", flush=True)
            result = run_benchmark(
                method=method,
                env_name=args.env,
                total_steps=args.steps,
                seed=seed,
                device=args.device
            )
            method_results.append(result)
            print(f"Goals: {result['goals_reached']}, "
                  f"Coverage: {result['final_coverage']*100:.1f}%")

        # Aggregate results
        avg_goals = np.mean([r['goals_reached'] for r in method_results])
        avg_coverage = np.mean([r['final_coverage'] for r in method_results])
        avg_entropy = np.mean([r['state_entropy'] for r in method_results])

        print(f"\n  Summary for {method}:")
        print(f"    Avg Goals: {avg_goals:.1f}")
        print(f"    Avg Coverage: {avg_coverage*100:.1f}%")
        print(f"    Avg State Entropy: {avg_entropy:.2f}")

        results.extend(method_results)

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_dir / f"results_{timestamp}.json", "w") as f:
        # Convert numpy to python types for JSON
        clean_results = []
        for r in results:
            clean_r = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                      for k, v in r.items()}
            clean_results.append(clean_r)
        json.dump(clean_results, f, indent=2)

    # Print final comparison
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print('='*60)
    print(f"{'Method':<15} {'Goals':>10} {'Coverage':>12} {'Entropy':>10}")
    print("-"*60)

    for method in args.methods:
        method_data = [r for r in results if r['method'] == method]
        avg_goals = np.mean([r['goals_reached'] for r in method_data])
        std_goals = np.std([r['goals_reached'] for r in method_data])
        avg_coverage = np.mean([r['final_coverage'] for r in method_data])
        avg_entropy = np.mean([r['state_entropy'] for r in method_data])

        print(f"{method:<15} {avg_goals:>7.1f}±{std_goals:.1f} "
              f"{avg_coverage*100:>10.1f}% {avg_entropy:>10.2f}")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
