"""
VOID_RUNNER - Curiosity Research Experiments
==============================================
Testing novel hypotheses about intrinsic motivation.

Experiments:
1. Adversarial vs RND: Does adversarial pressure prevent collapse?
2. Collapse Analysis: When and why does curiosity die?
3. Transfer Learning: Can curiosity generalize across environments?
4. Meta-Curiosity: Can we learn when to be curious?

Each experiment tests a specific hypothesis about curiosity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from collections import defaultdict
import json
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.sparse_maze import SparseMazeEnv
from envs.hard_exploration import DeceptiveRewardMaze, KeyDoorEnv

from src.curiosity import CuriosityModule
from src.drnd import DRND
from src.adversarial_curiosity import AdversarialCuriosity, SelfPlayCuriosity
from src.curiosity_diagnostics import CuriosityHealthMonitor, CuriosityCollapseDetector
from src.meta_curiosity import AdaptiveCuriosity, CuriosityScheduler, CuriosityEnsemble
from src.counting import SimHashCounter


class SimplePolicy(nn.Module):
    """Simple policy for experiments."""

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)

    def get_action(self, obs, epsilon=0.1):
        if np.random.random() < epsilon:
            return np.random.randint(self.net[-1].out_features)
        with torch.no_grad():
            logits = self.forward(obs)
            return logits.argmax().item()


def experiment_adversarial_vs_rnd():
    """
    Experiment 1: Does adversarial curiosity resist collapse?

    Hypothesis: Adversarial pressure maintains exploration drive
    even when RND curiosity collapses.

    Setup:
    - Run RND and Adversarial on same environment
    - Track intrinsic rewards over time
    - Measure exploration coverage
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Adversarial vs RND Collapse Resistance")
    print("=" * 60)

    env = SparseMazeEnv(size=15)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    n_steps = 20000
    device = "cpu"

    # Methods to compare
    methods = {
        "rnd": CuriosityModule(obs_dim, device=device),
        "adversarial": AdversarialCuriosity(obs_dim, device=device),
    }

    results = {name: {"rewards": [], "coverage": []} for name in methods}

    for method_name, curiosity in methods.items():
        print(f"\nTesting {method_name}...")

        policy = SimplePolicy(obs_dim, action_dim)
        visited_states = set()

        obs, _ = env.reset()
        episode_rewards = []

        for step in range(n_steps):
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            action = policy.get_action(obs_t)

            next_obs, reward, done, truncated, info = env.step(action)

            # Get intrinsic reward
            next_obs_t = torch.FloatTensor(next_obs).unsqueeze(0)

            if method_name == "adversarial":
                curiosity.add_experience(obs)
                intrinsic = curiosity.compute_intrinsic_reward(next_obs_t).item()
            else:
                intrinsic = curiosity.compute_intrinsic_reward(next_obs_t).item()

            episode_rewards.append(intrinsic)

            # Track coverage
            state_key = tuple(np.round(obs, 1))
            visited_states.add(state_key)

            # Record periodically
            if step % 100 == 0:
                results[method_name]["rewards"].append(np.mean(episode_rewards[-100:]))
                results[method_name]["coverage"].append(len(visited_states))
                episode_rewards = []

            if done or truncated:
                obs, _ = env.reset()
            else:
                obs = next_obs

        print(f"  Final coverage: {len(visited_states)} states")
        print(f"  Final avg reward: {results[method_name]['rewards'][-1]:.4f}")

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for name, data in results.items():
        axes[0].plot(data["rewards"], label=name)
        axes[1].plot(data["coverage"], label=name)

    axes[0].set_xlabel("Steps (x100)")
    axes[0].set_ylabel("Mean Intrinsic Reward")
    axes[0].set_title("Curiosity Collapse Comparison")
    axes[0].legend()
    axes[0].set_yscale("log")

    axes[1].set_xlabel("Steps (x100)")
    axes[1].set_ylabel("Unique States Visited")
    axes[1].set_title("Exploration Coverage")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("experiment1_adversarial_vs_rnd.png", dpi=150)
    print("\nSaved: experiment1_adversarial_vs_rnd.png")

    return results


def experiment_collapse_analysis():
    """
    Experiment 2: When and why does curiosity collapse?

    Hypothesis: Collapse has identifiable signatures that
    can be detected before it happens.

    Setup:
    - Run RND with full diagnostics
    - Track all metrics over time
    - Identify collapse point and cause
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Curiosity Collapse Analysis")
    print("=" * 60)

    env = DeceptiveRewardMaze(size=15)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    n_steps = 30000
    device = "cpu"

    # RND with health monitoring
    curiosity = CuriosityModule(obs_dim, device=device)
    monitor = CuriosityHealthMonitor()

    policy = SimplePolicy(obs_dim, action_dim)

    obs, _ = env.reset()

    print("\nRunning with diagnostics...")

    for step in range(n_steps):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        action = policy.get_action(obs_t)

        next_obs, reward, done, truncated, info = env.step(action)
        next_obs_t = torch.FloatTensor(next_obs).unsqueeze(0)

        intrinsic = curiosity.compute_intrinsic_reward(next_obs_t)

        # Record for monitoring
        monitor.step(
            state=obs,
            intrinsic_reward=intrinsic.mean().item(),
            predictor_loss=None,  # Would need to extract from curiosity module
        )

        if done or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs

        if step % 5000 == 0:
            print(f"  Step {step}")

    # Get diagnostic report
    print("\n" + monitor.get_report())

    # Save diagnostics
    monitor.detector.save("experiment2_collapse_diagnostics.json")
    print("\nSaved: experiment2_collapse_diagnostics.json")

    return monitor.detector.diagnose()


def experiment_curiosity_scheduling():
    """
    Experiment 3: Does adaptive scheduling improve exploration?

    Hypothesis: Adjusting curiosity coefficient based on learning
    progress is better than fixed coefficient.

    Setup:
    - Compare fixed vs scheduled vs adaptive curiosity
    - Measure task success and exploration
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Curiosity Scheduling")
    print("=" * 60)

    env = KeyDoorEnv(size=12)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    n_episodes = 200
    max_steps = 500
    device = "cpu"

    schedules = {
        "fixed_high": lambda s, t: 1.0,
        "fixed_low": lambda s, t: 0.1,
        "linear_decay": lambda s, t: max(0.1, 1.0 - s / t),
        "cosine": lambda s, t: 0.1 + 0.45 * (1 + np.cos(np.pi * s / t)),
    }

    results = {name: {"success": [], "coverage": []} for name in schedules}

    for sched_name, schedule_fn in schedules.items():
        print(f"\nTesting {sched_name} schedule...")

        curiosity = CuriosityModule(obs_dim, device=device)
        policy = SimplePolicy(obs_dim, action_dim)

        total_steps = 0
        total_episodes = n_episodes * max_steps

        successes = []
        coverages = []

        for episode in range(n_episodes):
            obs, _ = env.reset()
            episode_visited = set()
            success = False

            for step in range(max_steps):
                total_steps += 1

                # Get curiosity coefficient from schedule
                coef = schedule_fn(total_steps, total_episodes)

                obs_t = torch.FloatTensor(obs).unsqueeze(0)
                action = policy.get_action(obs_t)

                next_obs, reward, done, truncated, info = env.step(action)

                # Scaled intrinsic reward
                next_obs_t = torch.FloatTensor(next_obs).unsqueeze(0)
                intrinsic = curiosity.compute_intrinsic_reward(next_obs_t).item()
                scaled_reward = reward + coef * intrinsic

                episode_visited.add(tuple(np.round(obs, 1)))

                if reward > 0:
                    success = True

                if done or truncated:
                    break

                obs = next_obs

            successes.append(1.0 if success else 0.0)
            coverages.append(len(episode_visited))

            if episode % 50 == 0:
                recent_success = np.mean(successes[-20:]) if len(successes) >= 20 else 0
                print(f"  Episode {episode}: success rate = {recent_success:.2%}")

        results[sched_name]["success"] = successes
        results[sched_name]["coverage"] = coverages

        print(f"  Final success rate: {np.mean(successes):.2%}")

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    window = 20
    for name, data in results.items():
        # Smooth success rate
        smoothed = np.convolve(data["success"], np.ones(window)/window, mode='valid')
        axes[0].plot(smoothed, label=name)

        # Mean coverage
        smoothed_cov = np.convolve(data["coverage"], np.ones(window)/window, mode='valid')
        axes[1].plot(smoothed_cov, label=name)

    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Success Rate (smoothed)")
    axes[0].set_title("Task Success by Schedule")
    axes[0].legend()

    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("States per Episode")
    axes[1].set_title("Exploration by Schedule")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("experiment3_scheduling.png", dpi=150)
    print("\nSaved: experiment3_scheduling.png")

    return results


def experiment_ensemble_curiosity():
    """
    Experiment 4: Is ensemble curiosity better than single method?

    Hypothesis: Combining multiple curiosity signals is more robust.

    Setup:
    - Compare RND alone vs SimHash alone vs Ensemble
    - Track exploration and task success
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Ensemble Curiosity")
    print("=" * 60)

    env = DeceptiveRewardMaze(size=15)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    n_steps = 20000
    device = "cpu"

    # Create curiosity modules
    rnd = CuriosityModule(obs_dim, device=device)
    simhash = SimHashCounter(obs_dim)

    # Wrapper to make SimHash compatible
    class SimHashWrapper:
        def __init__(self, counter):
            self.counter = counter

        def compute_intrinsic_reward(self, obs, update=True):
            obs_np = obs.cpu().numpy()
            rewards = []
            for o in obs_np:
                count = self.counter.increment(o)
                rewards.append(1.0 / np.sqrt(count + 1))
            return torch.tensor(rewards)

    simhash_wrapped = SimHashWrapper(simhash)

    # Create ensemble
    ensemble = CuriosityEnsemble(
        curiosity_modules=[rnd, simhash_wrapped],
        obs_dim=obs_dim,
        device=device,
    )

    methods = {
        "rnd": rnd,
        "simhash": simhash_wrapped,
        "ensemble": ensemble,
    }

    results = {name: {"rewards": [], "coverage": [], "trap_escapes": 0} for name in methods}

    for method_name, curiosity in methods.items():
        print(f"\nTesting {method_name}...")

        policy = SimplePolicy(obs_dim, action_dim)
        visited_states = set()
        trap_escapes = 0

        obs, _ = env.reset()
        episode_rewards = []

        for step in range(n_steps):
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            action = policy.get_action(obs_t)

            next_obs, reward, done, truncated, info = env.step(action)
            next_obs_t = torch.FloatTensor(next_obs).unsqueeze(0)

            intrinsic = curiosity.compute_intrinsic_reward(next_obs_t)
            if isinstance(intrinsic, torch.Tensor):
                intrinsic = intrinsic.mean().item()

            episode_rewards.append(intrinsic)

            # Track coverage
            state_key = tuple(np.round(obs, 1))
            visited_states.add(state_key)

            # Track trap escapes
            if reward > 1.0:  # Found true goal
                trap_escapes += 1

            # Record periodically
            if step % 100 == 0:
                results[method_name]["rewards"].append(np.mean(episode_rewards[-100:]) if episode_rewards else 0)
                results[method_name]["coverage"].append(len(visited_states))
                episode_rewards = []

            if done or truncated:
                obs, _ = env.reset()
            else:
                obs = next_obs

        results[method_name]["trap_escapes"] = trap_escapes
        print(f"  Final coverage: {len(visited_states)} states")
        print(f"  Trap escapes: {trap_escapes}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for name, data in results.items():
        axes[0].plot(data["rewards"], label=name)
        axes[1].plot(data["coverage"], label=name)

    axes[0].set_xlabel("Steps (x100)")
    axes[0].set_ylabel("Mean Intrinsic Reward")
    axes[0].set_title("Intrinsic Reward Comparison")
    axes[0].legend()

    axes[1].set_xlabel("Steps (x100)")
    axes[1].set_ylabel("Unique States")
    axes[1].set_title("Exploration Coverage")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("experiment4_ensemble.png", dpi=150)
    print("\nSaved: experiment4_ensemble.png")

    return results


def main():
    """Run all experiments."""
    print("=" * 60)
    print("VOID_RUNNER CURIOSITY RESEARCH")
    print("=" * 60)

    # Create experiments directory
    os.makedirs("experiments", exist_ok=True)
    os.chdir("experiments")

    all_results = {}

    # Run experiments
    all_results["adversarial_vs_rnd"] = experiment_adversarial_vs_rnd()
    all_results["collapse_analysis"] = experiment_collapse_analysis()
    all_results["scheduling"] = experiment_curiosity_scheduling()
    all_results["ensemble"] = experiment_ensemble_curiosity()

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    print("\n1. Adversarial vs RND:")
    for method, data in all_results["adversarial_vs_rnd"].items():
        final_reward = data["rewards"][-1] if data["rewards"] else 0
        final_coverage = data["coverage"][-1] if data["coverage"] else 0
        print(f"   {method}: reward={final_reward:.4f}, coverage={final_coverage}")

    print("\n2. Collapse Analysis:")
    diagnosis = all_results["collapse_analysis"]
    print(f"   Collapse detected: {diagnosis.get('collapse_detected', 'N/A')}")
    print(f"   Diagnosis: {diagnosis.get('collapse_diagnosis', 'N/A')}")

    print("\n3. Scheduling:")
    for sched, data in all_results["scheduling"].items():
        success_rate = np.mean(data["success"]) if data["success"] else 0
        print(f"   {sched}: success={success_rate:.2%}")

    print("\n4. Ensemble:")
    for method, data in all_results["ensemble"].items():
        final_coverage = data["coverage"][-1] if data["coverage"] else 0
        escapes = data["trap_escapes"]
        print(f"   {method}: coverage={final_coverage}, escapes={escapes}")

    print("\n" + "=" * 60)
    print("All experiments complete. Check PNG files for visualizations.")


if __name__ == "__main__":
    main()
