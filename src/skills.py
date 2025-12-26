"""
VOID_RUNNER - Skill Discovery: DIAYN, DADS, and Hierarchical Exploration
=========================================================================
Unsupervised skill discovery for hierarchical reinforcement learning.

Key insight: Learn diverse, reusable skills WITHOUT external reward.
Then compose these skills for downstream tasks.

DIAYN (Diversity is All You Need):
- Maximize mutual information I(S; Z) between states and skills
- Each skill Z should visit distinguishable states
- Intrinsic reward: log q(z|s) - log p(z)

DADS (Dynamics-Aware Discovery of Skills):
- Skills should have PREDICTABLE dynamics
- Enables model-based planning with skills
- Better for hierarchical composition

DUSDi (Disentangled Skills - NeurIPS 2024):
- Learn skills that control INDIVIDUAL state dimensions
- Reduces skill entanglement for better composition

References:
- Eysenbach et al., "Diversity is All You Need" (DIAYN)
- Sharma et al., "Dynamics-Aware Unsupervised Discovery of Skills" (DADS)
- Hu et al., "Disentangled Unsupervised Skill Discovery" (DUSDi, NeurIPS 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional, List
from collections import deque


class SkillDiscriminator(nn.Module):
    """
    Skill discriminator q(z|s): predicts skill from state.

    This is trained to maximize I(S; Z).
    When it can perfectly predict skill from state,
    each skill visits distinguishable states.
    """

    def __init__(
        self,
        obs_dim: int,
        n_skills: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_skills),
        )

        self.n_skills = n_skills

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns logits for each skill."""
        return self.net(obs)

    def get_log_prob(self, obs: torch.Tensor, skill: torch.Tensor) -> torch.Tensor:
        """Get log q(z|s) for intrinsic reward."""
        logits = self.forward(obs)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(1, skill.unsqueeze(-1).long()).squeeze(-1)


class SkillConditionedPolicy(nn.Module):
    """
    Policy conditioned on skill: π(a|s, z)

    The policy takes both state and skill as input.
    Different skills should lead to different behaviors.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_skills: int,
        hidden_dim: int = 256,
        continuous: bool = False,
    ):
        super().__init__()
        self.continuous = continuous
        self.n_skills = n_skills

        # Skill embedding
        self.skill_embed = nn.Embedding(n_skills, hidden_dim // 4)

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + hidden_dim // 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        if continuous:
            self.mean = nn.Linear(hidden_dim, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.action_head = nn.Linear(hidden_dim, action_dim)

        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        obs: torch.Tensor,
        skill: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        skill_emb = self.skill_embed(skill.long())
        x = torch.cat([obs, skill_emb], dim=-1)
        features = self.encoder(x)

        if self.continuous:
            mean = self.mean(features)
            std = torch.exp(self.log_std)
            action_dist = (mean, std)
        else:
            action_dist = self.action_head(features)

        value = self.value_head(features)
        return action_dist, value

    def get_action(
        self,
        obs: torch.Tensor,
        skill: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        action_dist, value = self.forward(obs, skill)

        if self.continuous:
            mean, std = action_dist
            if deterministic:
                action = mean
            else:
                action = mean + std * torch.randn_like(mean)
            log_prob = -0.5 * ((action - mean) / std).pow(2).sum(-1)
        else:
            probs = F.softmax(action_dist, dim=-1)
            dist = torch.distributions.Categorical(probs)
            if deterministic:
                action = probs.argmax(dim=-1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)

        return action, log_prob


class DIAYN:
    """
    Diversity is All You Need: Unsupervised Skill Discovery.

    Objective: Maximize I(S; Z) - H(A|S, Z)
             = E_z[E_s[log q(z|s)]] + H(π(a|s,z))

    Intrinsic reward: r(s, z) = log q(z|s) - log p(z)
                    = "how well does state s identify skill z"

    Skills that visit unique states get high reward.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_skills: int = 10,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        entropy_coef: float = 0.1,
        device: str = "cpu",
        continuous: bool = False,
    ):
        self.device = device
        self.n_skills = n_skills
        self.entropy_coef = entropy_coef

        # Uniform prior over skills
        self.log_p_z = np.log(1.0 / n_skills)

        # Skill discriminator q(z|s)
        self.discriminator = SkillDiscriminator(
            obs_dim, n_skills, hidden_dim
        ).to(device)

        # Skill-conditioned policy π(a|s, z)
        self.policy = SkillConditionedPolicy(
            obs_dim, action_dim, n_skills, hidden_dim, continuous
        ).to(device)

        self.disc_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=learning_rate
        )
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(), lr=learning_rate
        )

        # Current skill
        self.current_skill = None

        # Statistics
        self.skill_usage = np.zeros(n_skills)

    def sample_skill(self) -> int:
        """Sample a skill from uniform prior."""
        skill = np.random.randint(0, self.n_skills)
        self.current_skill = skill
        self.skill_usage[skill] += 1
        return skill

    def get_action(
        self,
        obs: np.ndarray,
        skill: Optional[int] = None,
        deterministic: bool = False
    ) -> Tuple[int, float]:
        """Get action for current state and skill."""
        if skill is None:
            skill = self.current_skill

        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            skill_t = torch.tensor([skill]).to(self.device)

            action, log_prob = self.policy.get_action(
                obs_t, skill_t, deterministic
            )

        return action.item(), log_prob.item()

    def compute_intrinsic_reward(
        self,
        obs: torch.Tensor,
        skill: torch.Tensor
    ) -> torch.Tensor:
        """
        DIAYN intrinsic reward: log q(z|s) - log p(z)

        High reward when discriminator can identify skill from state.
        """
        log_q_z_s = self.discriminator.get_log_prob(obs, skill)
        intrinsic_reward = log_q_z_s - self.log_p_z
        return intrinsic_reward

    def update_discriminator(
        self,
        obs: torch.Tensor,
        skill: torch.Tensor
    ) -> float:
        """Train discriminator to predict skill from state."""
        logits = self.discriminator(obs)
        loss = F.cross_entropy(logits, skill.long())

        self.disc_optimizer.zero_grad()
        loss.backward()
        self.disc_optimizer.step()

        return loss.item()

    def get_skill_coverage(self) -> np.ndarray:
        """Get how evenly skills are being used."""
        return self.skill_usage / (self.skill_usage.sum() + 1e-8)


class DADS:
    """
    Dynamics-Aware Discovery of Skills.

    Key difference from DIAYN: Skills should be PREDICTABLE.

    Objective: Maximize I(S_{t+1}; Z | S_t)
             = "How much does skill tell us about next state?"

    This leads to skills with consistent, low-variance dynamics
    that are easier to plan with.

    Intrinsic reward: log p(s'|s, z) - log p(s'|s)
                    = "How much better can we predict s' knowing z?"
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_skills: int = 10,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        device: str = "cpu",
    ):
        self.device = device
        self.n_skills = n_skills
        self.obs_dim = obs_dim

        # Skill-conditioned dynamics model p(s'|s, z)
        self.dynamics = nn.Sequential(
            nn.Linear(obs_dim + n_skills, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim * 2),  # Mean and log_var
        ).to(device)

        # Marginal dynamics p(s'|s) - skill-agnostic
        self.marginal_dynamics = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim * 2),
        ).to(device)

        # Skill-conditioned policy
        self.policy = SkillConditionedPolicy(
            obs_dim, action_dim, n_skills, hidden_dim
        ).to(device)

        self.dynamics_optimizer = optim.Adam(
            list(self.dynamics.parameters()) +
            list(self.marginal_dynamics.parameters()),
            lr=learning_rate
        )
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(), lr=learning_rate
        )

        self.current_skill = None

    def sample_skill(self) -> int:
        skill = np.random.randint(0, self.n_skills)
        self.current_skill = skill
        return skill

    def _get_dynamics_log_prob(
        self,
        obs: torch.Tensor,
        skill: torch.Tensor,
        next_obs: torch.Tensor,
        use_marginal: bool = False
    ) -> torch.Tensor:
        """Get log probability of next state under dynamics model."""
        if use_marginal:
            output = self.marginal_dynamics(obs)
        else:
            skill_onehot = F.one_hot(skill.long(), self.n_skills).float()
            x = torch.cat([obs, skill_onehot], dim=-1)
            output = self.dynamics(x)

        mean, log_var = output.chunk(2, dim=-1)
        log_var = torch.clamp(log_var, -10, 2)

        # Gaussian log probability
        diff = next_obs - mean
        log_prob = -0.5 * (log_var + diff ** 2 / torch.exp(log_var))
        return log_prob.sum(dim=-1)

    def compute_intrinsic_reward(
        self,
        obs: torch.Tensor,
        skill: torch.Tensor,
        next_obs: torch.Tensor
    ) -> torch.Tensor:
        """
        DADS intrinsic reward: log p(s'|s,z) - log p(s'|s)

        How much better can we predict next state knowing skill?
        """
        log_p_conditional = self._get_dynamics_log_prob(obs, skill, next_obs)
        log_p_marginal = self._get_dynamics_log_prob(
            obs, skill, next_obs, use_marginal=True
        )

        intrinsic_reward = log_p_conditional - log_p_marginal
        return intrinsic_reward

    def update_dynamics(
        self,
        obs: torch.Tensor,
        skill: torch.Tensor,
        next_obs: torch.Tensor
    ) -> Tuple[float, float]:
        """Update both conditional and marginal dynamics."""
        # Conditional dynamics loss
        log_p_cond = self._get_dynamics_log_prob(obs, skill, next_obs)
        cond_loss = -log_p_cond.mean()

        # Marginal dynamics loss
        log_p_marg = self._get_dynamics_log_prob(
            obs, skill, next_obs, use_marginal=True
        )
        marg_loss = -log_p_marg.mean()

        total_loss = cond_loss + marg_loss

        self.dynamics_optimizer.zero_grad()
        total_loss.backward()
        self.dynamics_optimizer.step()

        return cond_loss.item(), marg_loss.item()


class HierarchicalExplorer:
    """
    Hierarchical exploration with skill-based abstraction.

    Two-level hierarchy:
    1. High-level: Chooses which skill to execute
    2. Low-level: Skill-conditioned policy executes skill

    The high-level policy can use any curiosity signal
    to decide which skill to try next.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_skills: int = 10,
        skill_duration: int = 50,  # Steps per skill
        hidden_dim: int = 256,
        device: str = "cpu",
    ):
        self.device = device
        self.n_skills = n_skills
        self.skill_duration = skill_duration

        # Low-level: DIAYN for skill learning
        self.skill_learner = DIAYN(
            obs_dim, action_dim, n_skills, hidden_dim, device=device
        )

        # High-level: Policy over skills
        self.meta_policy = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_skills),
        ).to(device)

        self.meta_optimizer = optim.Adam(
            self.meta_policy.parameters(), lr=3e-4
        )

        # Skill novelty tracking
        self.skill_novelty = np.ones(n_skills)  # UCB-style exploration
        self.skill_successes = np.zeros(n_skills)
        self.skill_attempts = np.ones(n_skills)

        self.current_skill = None
        self.steps_in_skill = 0

    def select_skill(self, obs: np.ndarray, use_ucb: bool = True) -> int:
        """Select skill using high-level policy + exploration bonus."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.meta_policy(obs_t)
            probs = F.softmax(logits, dim=-1).cpu().numpy().flatten()

        if use_ucb:
            # UCB exploration bonus
            ucb_bonus = np.sqrt(2 * np.log(self.skill_attempts.sum()) /
                               self.skill_attempts)
            probs = probs + 0.5 * ucb_bonus
            probs = probs / probs.sum()

        skill = np.random.choice(self.n_skills, p=probs)
        self.current_skill = skill
        self.steps_in_skill = 0
        self.skill_attempts[skill] += 1

        return skill

    def get_action(self, obs: np.ndarray) -> Tuple[int, float]:
        """Get action from current skill."""
        return self.skill_learner.get_action(obs, self.current_skill)

    def step(self) -> bool:
        """
        Advance step counter. Returns True if skill should change.
        """
        self.steps_in_skill += 1
        return self.steps_in_skill >= self.skill_duration

    def update_skill_success(self, success: bool):
        """Update skill success rate for UCB."""
        if success:
            self.skill_successes[self.current_skill] += 1
