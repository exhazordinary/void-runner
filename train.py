"""
VOID_RUNNER - Training Script
=============================
Train a curiosity-driven agent on sparse reward environments.

Usage:
    python train.py --env SparseMaze-v0 --steps 100000
    python train.py --env MountainCar-v0 --steps 500000
"""

import argparse
import sys
import os
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import gymnasium as gym
from collections import deque
from datetime import datetime
import json

from src.agent import VoidRunnerAgent
from envs.sparse_maze import SparseMazeEnv, SparseGridWorldEnv


def create_env(env_name: str, render_mode: str = None):
    """Create environment by name."""
    if env_name == "SparseMaze-v0":
        return SparseMazeEnv(size=15, walls=True, render_mode=render_mode)
    elif env_name == "SparseGrid-v0":
        return SparseGridWorldEnv(size=30, render_mode=render_mode)
    else:
        return gym.make(env_name, render_mode=render_mode)


def train(
    env_name: str = "SparseMaze-v0",
    total_steps: int = 100000,
    rollout_steps: int = 128,
    save_freq: int = 10000,
    log_freq: int = 1000,
    seed: int = 42,
    device: str = "auto",
    output_dir: str = "outputs"
):
    """
    Train VOID_RUNNER agent.

    Args:
        env_name: Environment to train on
        total_steps: Total training steps
        rollout_steps: Steps per rollout before update
        save_freq: Save checkpoint every N steps
        log_freq: Log metrics every N steps
        seed: Random seed
        device: Device to use (auto, cpu, cuda, mps)
        output_dir: Directory for outputs
    """
    # Setup device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                     VOID_RUNNER v1.0                         ║
║           Curiosity-Driven Exploration Agent                 ║
╠══════════════════════════════════════════════════════════════╣
║  "The reward of curiosity is not just what you find,        ║
║   but what you become while searching."                      ║
╚══════════════════════════════════════════════════════════════╝

Environment: {env_name}
Device: {device}
Total Steps: {total_steps:,}
Rollout Steps: {rollout_steps}
""")

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"{env_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create environment
    env = create_env(env_name)

    # Get dimensions
    obs_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
        continuous = False
    else:
        action_dim = env.action_space.shape[0]
        continuous = True

    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {action_dim} ({'continuous' if continuous else 'discrete'})")
    print(f"Output dir: {run_dir}\n")

    # Create agent
    agent = VoidRunnerAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma_ext=0.999,
        gamma_int=0.99,
        int_coef=1.0,
        ext_coef=2.0,
        n_epochs=4,
        batch_size=256,
        device=device,
        continuous=continuous,
    )

    # Training metrics
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    episode_intrinsic = deque(maxlen=100)
    coverages = deque(maxlen=100)
    metrics_history = []

    # Training loop
    obs, info = env.reset(seed=seed)
    episode_reward = 0
    episode_reward_int = 0
    episode_length = 0

    best_reward = float('-inf')
    goals_reached = 0

    print("Training started...")
    print("-" * 60)

    for step in range(1, total_steps + 1):
        # Select action
        action, log_prob, value_ext, value_int = agent.select_action(obs)

        # Take step
        next_obs, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated

        # Store transition
        agent.store_transition(
            obs=obs,
            action=action,
            log_prob=log_prob,
            reward_ext=np.array([reward]),
            value_ext=value_ext,
            value_int=value_int,
            done=np.array([float(done)]),
            next_obs=next_obs
        )

        episode_reward += reward
        episode_reward_int += agent.buffer.rewards_int[-1].item() if agent.buffer.rewards_int else 0
        episode_length += 1

        # Check for goal
        if terminated and reward > 0:
            goals_reached += 1

        # Episode done
        if done:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_intrinsic.append(episode_reward_int)
            if "coverage" in info:
                coverages.append(info["coverage"])

            obs, info = env.reset()
            episode_reward = 0
            episode_reward_int = 0
            episode_length = 0
        else:
            obs = next_obs

        # Update agent
        if step % rollout_steps == 0:
            update_info = agent.update(next_obs)
            agent.curiosity.reset_stats()

        # Logging
        if step % log_freq == 0:
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            avg_length = np.mean(episode_lengths) if episode_lengths else 0
            avg_intrinsic = np.mean(episode_intrinsic) if episode_intrinsic else 0
            avg_coverage = np.mean(coverages) if coverages else 0

            print(f"Step {step:>7,} | "
                  f"Reward: {avg_reward:>7.2f} | "
                  f"Curiosity: {avg_intrinsic:>7.2f} | "
                  f"Length: {avg_length:>6.1f} | "
                  f"Coverage: {avg_coverage*100:>5.1f}% | "
                  f"Goals: {goals_reached}")

            metrics_history.append({
                "step": step,
                "avg_reward": avg_reward,
                "avg_intrinsic": avg_intrinsic,
                "avg_length": avg_length,
                "coverage": avg_coverage,
                "goals_reached": goals_reached
            })

            # Track best
            if avg_reward > best_reward:
                best_reward = avg_reward
                agent.save(str(run_dir / "best_model.pt"))

        # Save checkpoint
        if step % save_freq == 0:
            agent.save(str(run_dir / f"checkpoint_{step}.pt"))

    # Save final model and metrics
    agent.save(str(run_dir / "final_model.pt"))

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics_history, f, indent=2)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best average reward: {best_reward:.2f}")
    print(f"Total goals reached: {goals_reached}")
    print(f"Models saved to: {run_dir}")

    return agent, metrics_history


def evaluate(
    agent: VoidRunnerAgent,
    env_name: str,
    n_episodes: int = 10,
    render: bool = False
):
    """Evaluate trained agent."""
    env = create_env(env_name, render_mode="human" if render else None)

    total_rewards = []
    total_coverages = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _, _, _ = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            episode_reward += reward

        total_rewards.append(episode_reward)
        if "coverage" in info:
            total_coverages.append(info["coverage"])

        print(f"Episode {ep + 1}: Reward = {episode_reward:.2f}, Coverage = {info.get('coverage', 0)*100:.1f}%")

    print(f"\nMean reward: {np.mean(total_rewards):.2f} +/- {np.std(total_rewards):.2f}")
    if total_coverages:
        print(f"Mean coverage: {np.mean(total_coverages)*100:.1f}%")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VOID_RUNNER Training")
    parser.add_argument("--env", type=str, default="SparseMaze-v0",
                        help="Environment name")
    parser.add_argument("--steps", type=int, default=100000,
                        help="Total training steps")
    parser.add_argument("--rollout", type=int, default=128,
                        help="Steps per rollout")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto, cpu, cuda, mps)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output", type=str, default="outputs",
                        help="Output directory")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluation mode")
    parser.add_argument("--model", type=str, default=None,
                        help="Model path for evaluation")

    args = parser.parse_args()

    if args.eval:
        if args.model is None:
            print("Please specify --model for evaluation")
            sys.exit(1)

        env = create_env(args.env)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else env.action_space.shape[0]
        continuous = not isinstance(env.action_space, gym.spaces.Discrete)

        agent = VoidRunnerAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=args.device if args.device != "auto" else "cpu",
            continuous=continuous,
        )
        agent.load(args.model)
        evaluate(agent, args.env, render=True)
    else:
        train(
            env_name=args.env,
            total_steps=args.steps,
            rollout_steps=args.rollout,
            seed=args.seed,
            device=args.device,
            output_dir=args.output
        )
