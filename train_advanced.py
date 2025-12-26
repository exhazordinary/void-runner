"""
VOID_RUNNER - Advanced Training with Compound Curiosity
========================================================
Train agents with multi-faceted intrinsic motivation.

This training script uses the full compound curiosity system:
- RND novelty detection
- Empowerment-based control seeking
- Episodic memory for revisiting
- Causal understanding

With adaptive scheduling that learns which curiosity type works best.
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

from src.networks import PolicyNetwork, RunningMeanStd
from src.compound import CompoundCuriosity, AdversarialCuriosity, CuriosityType
from src.metrics import ExplorationMetrics, CompressionMetrics, InformationGain
from envs.sparse_maze import SparseMazeEnv, SparseGridWorldEnv


class AdvancedVoidRunner:
    """
    Advanced VOID_RUNNER with compound curiosity.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        device: str = "cpu",
        curiosity_types: list = None,
    ):
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        # Policy network
        self.policy = PolicyNetwork(
            obs_dim, action_dim, hidden_dim=256
        ).to(device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Compound curiosity system
        enabled = {
            'enable_novelty': 'novelty' in (curiosity_types or ['novelty']),
            'enable_empowerment': 'empowerment' in (curiosity_types or []),
            'enable_episodic': 'episodic' in (curiosity_types or []),
            'enable_causal': 'causal' in (curiosity_types or []),
        }

        self.curiosity = CompoundCuriosity(
            obs_dim, action_dim,
            device=device,
            **enabled
        )

        # Exploration metrics
        self.metrics = ExplorationMetrics(obs_dim, action_dim)
        self.compression = CompressionMetrics()
        self.info_gain = InformationGain(obs_dim)

        # Rollout storage
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.next_observations = []

        # Statistics
        self.episode_rewards = deque(maxlen=100)
        self.episode_curiosity = deque(maxlen=100)

    def select_action(self, obs: np.ndarray):
        """Select action from policy."""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action, log_prob, value_ext, value_int = self.policy.get_action(obs_t)

            return (
                action.cpu().numpy().item(),
                log_prob.cpu().numpy().item(),
                (value_ext + value_int).cpu().numpy().item() / 2
            )

    def store(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
        next_obs: np.ndarray
    ):
        """Store transition."""
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.next_observations.append(next_obs)

        # Update metrics
        self.metrics.update(obs, action, next_obs)
        self.compression.update(obs, action)

    def compute_curiosity_rewards(self) -> np.ndarray:
        """Compute intrinsic rewards for stored transitions."""
        obs_t = torch.FloatTensor(np.array(self.observations)).to(self.device)
        actions_t = torch.LongTensor(self.actions).to(self.device)
        next_obs_t = torch.FloatTensor(np.array(self.next_observations)).to(self.device)

        with torch.no_grad():
            intrinsic, info = self.curiosity.compute_intrinsic_reward(
                obs_t, actions_t, next_obs_t, update=True
            )

        # Update info gain
        self.info_gain.update(obs_t, next_obs_t)

        return intrinsic.cpu().numpy()

    def update(self, next_obs: np.ndarray) -> dict:
        """PPO update."""
        if len(self.observations) == 0:
            return {}

        # Get curiosity rewards
        curiosity_rewards = self.compute_curiosity_rewards()

        # Combine with extrinsic rewards
        total_rewards = np.array(self.rewards) + curiosity_rewards

        # Compute GAE
        with torch.no_grad():
            next_obs_t = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
            _, v_ext, v_int = self.policy(next_obs_t)
            next_value = (v_ext + v_int).cpu().numpy().item() / 2

        advantages, returns = self._compute_gae(
            total_rewards,
            np.array(self.values),
            np.array(self.dones),
            next_value
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        obs_t = torch.FloatTensor(np.array(self.observations)).to(self.device)
        actions_t = torch.LongTensor(self.actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(self.log_probs).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        # PPO epochs
        n_epochs = 4
        batch_size = min(64, len(self.observations))
        n_samples = len(self.observations)

        total_loss = 0
        for _ in range(n_epochs):
            indices = np.random.permutation(n_samples)

            for start in range(0, n_samples, batch_size):
                batch_idx = indices[start:start + batch_size]

                log_prob, entropy, v_ext, v_int = self.policy.evaluate_actions(
                    obs_t[batch_idx], actions_t[batch_idx]
                )
                value = (v_ext + v_int).squeeze() / 2

                # Policy loss
                ratio = torch.exp(log_prob - old_log_probs_t[batch_idx])
                surr1 = ratio * advantages_t[batch_idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_t[batch_idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(value, returns_t[batch_idx])

                # Entropy bonus
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                total_loss += loss.item()

        # Clear buffers
        info = {
            'loss': total_loss / (n_epochs * (n_samples // batch_size + 1)),
            'avg_curiosity': curiosity_rewards.mean(),
            'avg_reward': np.mean(self.rewards),
        }

        # Add curiosity stats
        info.update(self.curiosity.get_statistics())

        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.next_observations.clear()

        return info

    def _compute_gae(self, rewards, values, dones, next_value):
        """Compute GAE."""
        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae

        returns = advantages + values
        return advantages, returns


def train_advanced(
    env_name: str = "SparseMaze-v0",
    total_steps: int = 100000,
    curiosity_types: list = None,
    device: str = "auto",
    output_dir: str = "outputs_advanced"
):
    """Train with compound curiosity."""

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    curiosity_types = curiosity_types or ['novelty', 'empowerment']

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║              VOID_RUNNER v2.0 - ADVANCED                     ║
║            Compound Curiosity Exploration                    ║
╠══════════════════════════════════════════════════════════════╣
║  Curiosity Types: {', '.join(curiosity_types):<40} ║
║  Device: {device:<50} ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Environment
    if env_name == "SparseMaze-v0":
        env = SparseMazeEnv(size=15, walls=True)
    elif env_name == "SparseGrid-v0":
        env = SparseGridWorldEnv(size=30)
    else:
        import gymnasium as gym
        env = gym.make(env_name)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Agent
    agent = AdvancedVoidRunner(
        obs_dim, action_dim,
        device=device,
        curiosity_types=curiosity_types
    )

    # Training
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"{env_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    obs, info = env.reset()
    episode_reward = 0
    episode_curiosity = 0
    episode_length = 0
    goals_reached = 0

    metrics_history = []
    rollout_steps = 128
    log_freq = 2000

    print("Training with compound curiosity...")
    print("-" * 60)

    for step in range(1, total_steps + 1):
        action, log_prob, value = agent.select_action(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.store(obs, action, log_prob, reward, value, done, next_obs)

        episode_reward += reward
        episode_length += 1

        if terminated and reward > 0:
            goals_reached += 1

        if done:
            agent.episode_rewards.append(episode_reward)
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0
        else:
            obs = next_obs

        # Update
        if step % rollout_steps == 0:
            update_info = agent.update(next_obs)

        # Log
        if step % log_freq == 0:
            avg_reward = np.mean(agent.episode_rewards) if agent.episode_rewards else 0

            # Get exploration metrics
            exp_metrics = agent.metrics.get_all_metrics()

            print(f"Step {step:>7,} | "
                  f"Reward: {avg_reward:>6.2f} | "
                  f"Goals: {goals_reached:>3} | "
                  f"StateH: {exp_metrics['state_entropy']:>5.2f} | "
                  f"Coverage: {exp_metrics['coverage_rate']*100:>5.1f}% | "
                  f"Efficiency: {exp_metrics['exploration_efficiency']:>5.3f}")

            # Get dominant curiosity type
            dummy_obs = torch.FloatTensor(obs).unsqueeze(0).to(device)
            dominant = agent.curiosity.get_dominant_curiosity(dummy_obs)
            print(f"         Dominant curiosity: {dominant.value}")

            metrics_history.append({
                'step': step,
                'avg_reward': avg_reward,
                'goals_reached': goals_reached,
                **exp_metrics,
                'dominant_curiosity': dominant.value
            })

    # Save
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics_history, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Training complete! Results saved to {run_dir}")
    print(f"Final goals reached: {goals_reached}")

    # Print final exploration analysis
    print("\n=== Exploration Analysis ===")
    final_metrics = agent.metrics.get_all_metrics()
    for name, value in final_metrics.items():
        print(f"  {name}: {value:.4f}")

    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="SparseMaze-v0")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--curiosity", nargs="+",
                        default=["novelty", "empowerment"],
                        choices=["novelty", "empowerment", "episodic", "causal"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="outputs_advanced")

    args = parser.parse_args()

    train_advanced(
        env_name=args.env,
        total_steps=args.steps,
        curiosity_types=args.curiosity,
        device=args.device,
        output_dir=args.output
    )
