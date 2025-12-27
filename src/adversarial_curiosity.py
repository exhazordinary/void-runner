"""
VOID_RUNNER - Adversarial Curiosity
=====================================
GAN-style exploration: Generator seeks novelty, Discriminator predicts familiarity.

The Core Idea:
--------------
Standard curiosity (RND, ICM) uses prediction error as novelty signal.
Problem: As predictor improves, ALL states become "familiar" → curiosity collapse.

Adversarial solution:
1. Generator (policy) actively seeks states to fool the discriminator
2. Discriminator learns to recognize visited vs unvisited states
3. Adversarial pressure maintains exploration drive

This is fundamentally different:
- RND: Passive prediction error
- Adversarial: Active search for blind spots

The discriminator's job is NOT to predict random features.
It's to answer: "Have I seen this state before?"
The generator wins by finding states where the answer is "no."

Theoretical Connection:
----------------------
This relates to:
- GANs (Goodfellow 2014): Generator vs discriminator game
- Intrinsic motivation via prediction (Schmidhuber)
- Adversarial skill discovery
- Maximum entropy exploration

The equilibrium: Generator finds genuinely novel states
at the rate the discriminator can absorb them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
from dataclasses import dataclass


class StateDiscriminator(nn.Module):
    """
    Discriminator: "Have I seen this state before?"

    Trained on:
    - Positive examples: States from replay buffer (visited)
    - Negative examples: Generated/hypothetical states (unvisited)

    The generator (policy) is rewarded for finding states
    the discriminator classifies as "unvisited."
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
    ):
        super().__init__()

        layers = []
        in_dim = obs_dim

        for i in range(n_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

        # Spectral normalization for training stability
        self._apply_spectral_norm()

    def _apply_spectral_norm(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.utils.parametrizations.spectral_norm(module)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Returns logit: positive = "seen before", negative = "novel"
        """
        return self.net(obs)

    def novelty_score(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Higher score = more novel (discriminator thinks it's unseen)
        """
        with torch.no_grad():
            logits = self.forward(obs)
            # Probability of being "unseen"
            novelty = torch.sigmoid(-logits)
        return novelty.squeeze(-1)


class StateGenerator(nn.Module):
    """
    Generator: Proposes "what states would be novel?"

    This isn't generating states directly (that's the policy's job).
    Instead, it learns an embedding of "novelty direction" -
    which directions in state space lead to unexplored territory.

    Used to:
    1. Guide policy toward novel states
    2. Generate hypothetical states for discriminator training
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.latent_dim = latent_dim

        # Encoder: state -> latent
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Latent statistics
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)

        # Decoder: latent -> hypothetical novel state
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

        # Novelty direction predictor
        self.novelty_direction = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
            nn.Tanh(),  # Bounded direction
        )

    def encode(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(obs)
        return self.mu(h), self.log_var(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: (reconstructed_obs, mu, log_var)
        """
        mu, log_var = self.encode(obs)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var

    def generate_novel_state(self, obs: torch.Tensor, step_size: float = 0.1) -> torch.Tensor:
        """
        Generate a hypothetical novel state near current observation.

        Takes current state and moves in "novelty direction."
        """
        direction = self.novelty_direction(obs)
        novel_state = obs + step_size * direction
        return novel_state

    def sample_hypothetical(self, n_samples: int, device: str = "cpu") -> torch.Tensor:
        """
        Sample hypothetical states from learned latent distribution.
        """
        z = torch.randn(n_samples, self.latent_dim).to(device)
        return self.decode(z)


class ExperienceBuffer:
    """
    Buffer of visited states for discriminator training.
    """

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, obs: np.ndarray):
        self.buffer.append(obs.copy())

    def sample(self, batch_size: int) -> np.ndarray:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return np.array([self.buffer[i] for i in indices])

    def __len__(self):
        return len(self.buffer)


@dataclass
class AdversarialStats:
    """Statistics for monitoring adversarial training."""
    discriminator_accuracy: float
    generator_loss: float
    discriminator_loss: float
    mean_novelty: float
    exploration_rate: float  # Fraction of states classified as novel


class AdversarialCuriosity:
    """
    Adversarial Curiosity Module.

    Training loop:
    1. Collect states from environment
    2. Train discriminator: real (visited) vs fake (generated)
    3. Generator tries to produce states discriminator marks as "unseen"
    4. Policy is rewarded for reaching states generator predicts as novel

    The key insight: This creates a curriculum.
    - Early: Everything is novel (high exploration)
    - Middle: Discriminator catches up, generator must find truly new states
    - Late: Equilibrium between finding and recognizing novelty

    Unlike RND where curiosity collapses, adversarial pressure
    maintains the exploration drive.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        buffer_size: int = 100000,
        discriminator_lr: float = 1e-4,
        generator_lr: float = 1e-4,
        n_discriminator_steps: int = 1,
        n_generator_steps: int = 1,
        label_smoothing: float = 0.1,
        device: str = "cpu",
    ):
        self.device = device
        self.obs_dim = obs_dim
        self.n_discriminator_steps = n_discriminator_steps
        self.n_generator_steps = n_generator_steps
        self.label_smoothing = label_smoothing

        # Networks
        self.discriminator = StateDiscriminator(obs_dim, hidden_dim).to(device)
        self.generator = StateGenerator(obs_dim, latent_dim, hidden_dim).to(device)

        # Optimizers
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=discriminator_lr,
            betas=(0.5, 0.999)
        )
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=generator_lr,
            betas=(0.5, 0.999)
        )

        # Experience buffer
        self.buffer = ExperienceBuffer(buffer_size)

        # Statistics
        self.stats_history: List[AdversarialStats] = []
        self.total_steps = 0

    def add_experience(self, obs: np.ndarray):
        """Add visited state to buffer."""
        self.buffer.add(obs)
        self.total_steps += 1

    def compute_intrinsic_reward(
        self,
        obs: torch.Tensor,
        update: bool = True,
    ) -> torch.Tensor:
        """
        Compute intrinsic reward based on discriminator's novelty assessment.

        Higher reward for states discriminator thinks are "unseen."
        """
        novelty = self.discriminator.novelty_score(obs)

        if update and len(self.buffer) >= 64:
            self._train_step(obs)

        return novelty

    def _train_step(self, current_obs: torch.Tensor):
        """
        One step of adversarial training.
        """
        batch_size = min(64, len(self.buffer))

        # Sample real (visited) states
        real_states = torch.FloatTensor(
            self.buffer.sample(batch_size)
        ).to(self.device)

        # === Train Discriminator ===
        for _ in range(self.n_discriminator_steps):
            self.d_optimizer.zero_grad()

            # Real states should be classified as "seen" (label = 1)
            real_logits = self.discriminator(real_states)
            real_labels = torch.ones_like(real_logits) * (1 - self.label_smoothing)
            d_loss_real = F.binary_cross_entropy_with_logits(real_logits, real_labels)

            # Generated states should be classified as "unseen" (label = 0)
            with torch.no_grad():
                fake_states = self.generator.sample_hypothetical(batch_size, self.device)

            fake_logits = self.discriminator(fake_states)
            fake_labels = torch.zeros_like(fake_logits) + self.label_smoothing
            d_loss_fake = F.binary_cross_entropy_with_logits(fake_logits, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            self.d_optimizer.step()

        # === Train Generator ===
        for _ in range(self.n_generator_steps):
            self.g_optimizer.zero_grad()

            # VAE reconstruction loss
            recon, mu, log_var = self.generator(real_states)
            recon_loss = F.mse_loss(recon, real_states)
            kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

            # Adversarial loss: generated states should fool discriminator
            fake_states = self.generator.sample_hypothetical(batch_size, self.device)
            fake_logits = self.discriminator(fake_states)
            # Generator wants discriminator to say "unseen" (but we flip for generator's perspective)
            g_adversarial_loss = F.binary_cross_entropy_with_logits(
                fake_logits,
                torch.ones_like(fake_logits)  # Generator wants these classified as "real" (confuse D)
            )

            # Novelty direction loss: encourage moving toward novel states
            novel_states = self.generator.generate_novel_state(real_states)
            novel_logits = self.discriminator(novel_states)
            novelty_loss = torch.mean(torch.sigmoid(novel_logits))  # Want low (unseen)

            g_loss = recon_loss + 0.01 * kl_loss + g_adversarial_loss + novelty_loss
            g_loss.backward()
            self.g_optimizer.step()

        # Record statistics
        with torch.no_grad():
            real_correct = (torch.sigmoid(real_logits) > 0.5).float().mean()
            fake_correct = (torch.sigmoid(fake_logits) < 0.5).float().mean()
            accuracy = (real_correct + fake_correct) / 2

            current_novelty = self.discriminator.novelty_score(current_obs).mean()

            stats = AdversarialStats(
                discriminator_accuracy=accuracy.item(),
                generator_loss=g_loss.item(),
                discriminator_loss=d_loss.item(),
                mean_novelty=current_novelty.item(),
                exploration_rate=(current_novelty > 0.5).float().mean().item(),
            )
            self.stats_history.append(stats)

    def get_novelty_direction(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get the direction toward novelty from current state.

        Useful for guiding policy toward unexplored regions.
        """
        return self.generator.novelty_direction(obs)

    def get_statistics(self) -> Dict:
        """Get training statistics."""
        if not self.stats_history:
            return {}

        recent = self.stats_history[-100:]
        return {
            "discriminator_accuracy": np.mean([s.discriminator_accuracy for s in recent]),
            "generator_loss": np.mean([s.generator_loss for s in recent]),
            "discriminator_loss": np.mean([s.discriminator_loss for s in recent]),
            "mean_novelty": np.mean([s.mean_novelty for s in recent]),
            "exploration_rate": np.mean([s.exploration_rate for s in recent]),
            "buffer_size": len(self.buffer),
            "total_steps": self.total_steps,
        }


class SelfPlayCuriosity:
    """
    Self-Play Curiosity: Two policies compete.

    Alice: Tries to reach states Bob can't predict
    Bob: Tries to predict where Alice will go

    This creates asymmetric exploration:
    - Alice develops increasingly sophisticated exploration strategies
    - Bob develops increasingly sophisticated state prediction

    The "Alice" policy becomes a strong explorer.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        device: str = "cpu",
    ):
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Alice: Explorer policy
        self.alice_policy = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        ).to(device)

        # Bob: State predictor (predicts Alice's next state)
        self.bob_predictor = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        ).to(device)

        self.alice_optimizer = optim.Adam(self.alice_policy.parameters(), lr=learning_rate)
        self.bob_optimizer = optim.Adam(self.bob_predictor.parameters(), lr=learning_rate)

        # Trajectory buffer for training
        self.trajectories: List[List[Tuple]] = []
        self.current_trajectory: List[Tuple] = []

    def alice_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Alice chooses action."""
        logits = self.alice_policy(obs)
        if deterministic:
            return logits.argmax(dim=-1)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)

    def bob_predict(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Bob predicts next state."""
        action_onehot = F.one_hot(action.long(), self.action_dim).float()
        x = torch.cat([obs, action_onehot], dim=-1)
        return self.bob_predictor(x)

    def step(self, obs: np.ndarray, action: int, next_obs: np.ndarray):
        """Record transition."""
        self.current_trajectory.append((obs, action, next_obs))

    def end_episode(self):
        """End current episode and start new one."""
        if self.current_trajectory:
            self.trajectories.append(self.current_trajectory)
            if len(self.trajectories) > 100:
                self.trajectories.pop(0)
        self.current_trajectory = []

    def compute_intrinsic_reward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Alice's reward: Bob's prediction error.
        High error = Alice found something Bob couldn't predict.
        """
        predicted = self.bob_predict(obs, action)
        error = torch.mean((predicted - next_obs) ** 2, dim=-1)
        return error

    def train(self, batch_size: int = 32) -> Dict[str, float]:
        """Train both Alice and Bob."""
        if not self.trajectories or sum(len(t) for t in self.trajectories) < batch_size:
            return {}

        # Sample transitions
        all_transitions = [t for traj in self.trajectories for t in traj]
        indices = np.random.choice(len(all_transitions), batch_size, replace=False)

        obs = torch.FloatTensor([all_transitions[i][0] for i in indices]).to(self.device)
        actions = torch.LongTensor([all_transitions[i][1] for i in indices]).to(self.device)
        next_obs = torch.FloatTensor([all_transitions[i][2] for i in indices]).to(self.device)

        # Train Bob (minimize prediction error)
        self.bob_optimizer.zero_grad()
        predicted = self.bob_predict(obs, actions)
        bob_loss = F.mse_loss(predicted, next_obs)
        bob_loss.backward()
        self.bob_optimizer.step()

        # Train Alice (maximize prediction error = minimize negative error)
        self.alice_optimizer.zero_grad()

        # Get Alice's action probabilities
        logits = self.alice_policy(obs)
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Reward is Bob's prediction error
        with torch.no_grad():
            predicted = self.bob_predict(obs, actions)
            rewards = torch.mean((predicted - next_obs) ** 2, dim=-1)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        alice_loss = -(action_log_probs * rewards).mean()
        alice_loss.backward()
        self.alice_optimizer.step()

        return {
            "bob_loss": bob_loss.item(),
            "alice_loss": alice_loss.item(),
            "mean_prediction_error": rewards.mean().item(),
        }


class CuriosityBottleneck:
    """
    Information Bottleneck for Curiosity.

    Key insight: Not all state information is relevant for novelty.
    Learn a compressed representation that captures only
    exploration-relevant features.

    Architecture:
    state -> encoder -> bottleneck (low dim) -> decoder -> novelty prediction

    The bottleneck forces the model to learn what MATTERS for exploration.
    """

    def __init__(
        self,
        obs_dim: int,
        bottleneck_dim: int = 16,
        hidden_dim: int = 256,
        beta: float = 0.01,  # Information bottleneck coefficient
        device: str = "cpu",
    ):
        self.device = device
        self.beta = beta

        # Encoder with stochastic bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        ).to(device)

        self.bottleneck_mu = nn.Linear(hidden_dim, bottleneck_dim).to(device)
        self.bottleneck_log_var = nn.Linear(hidden_dim, bottleneck_dim).to(device)

        # Novelty predictor from bottleneck
        self.novelty_head = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(device)

        # Visit counter (for ground truth novelty)
        self.visit_counts: Dict[str, int] = {}

        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) +
            list(self.bottleneck_mu.parameters()) +
            list(self.bottleneck_log_var.parameters()) +
            list(self.novelty_head.parameters()),
            lr=1e-4
        )

    def _hash_state(self, obs: np.ndarray) -> str:
        return str(tuple(np.round(obs, 2)))

    def _get_true_novelty(self, obs: np.ndarray) -> float:
        """Ground truth novelty based on visit count."""
        key = self._hash_state(obs)
        count = self.visit_counts.get(key, 0)
        self.visit_counts[key] = count + 1
        return 1.0 / np.sqrt(count + 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: (novelty_prediction, mu, log_var)
        """
        h = self.encoder(obs)
        mu = self.bottleneck_mu(h)
        log_var = self.bottleneck_log_var(h)

        # Reparameterization
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std

        novelty = self.novelty_head(z)
        return novelty, mu, log_var

    def compute_intrinsic_reward(
        self,
        obs: torch.Tensor,
        update: bool = True,
    ) -> torch.Tensor:
        """Compute intrinsic reward through information bottleneck."""
        novelty_pred, mu, log_var = self.forward(obs)

        if update:
            # Get ground truth novelty
            obs_np = obs.detach().cpu().numpy()
            true_novelty = torch.FloatTensor([
                self._get_true_novelty(o) for o in obs_np
            ]).unsqueeze(-1).to(self.device)

            # Prediction loss
            pred_loss = F.mse_loss(novelty_pred, true_novelty)

            # Information bottleneck (KL divergence)
            kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

            loss = pred_loss + self.beta * kl_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return novelty_pred.squeeze(-1).detach()

    def get_bottleneck_representation(self, obs: torch.Tensor) -> torch.Tensor:
        """Get the compressed exploration-relevant representation."""
        h = self.encoder(obs)
        return self.bottleneck_mu(h)
