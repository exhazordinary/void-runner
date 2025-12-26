"""
VOID_RUNNER - Empowerment Module
=================================
Intrinsic motivation based on CONTROL, not just novelty.

Empowerment = "How much can I affect my future?"

Mathematically: The channel capacity between actions and future states.
High empowerment states are those where the agent has maximum control
over what happens next. This drives agents toward "positions of power"
in the environment.

This is philosophically different from pure novelty-seeking:
- RND: "This is new to me" → Explore
- Empowerment: "I can affect things here" → Stay and master

The combination creates agents that seek novel states WHERE they have agency.

Reference: Klyubin et al., "Empowerment: A Universal Agent-Centric Measure of Control"
           Mohamed & Rezende, "Variational Information Maximisation for Intrinsically Motivated RL"
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from collections import deque


class SourceNetwork(nn.Module):
    """
    Source distribution q(a|s) - proposes actions that lead to diverse outcomes.

    This learns to propose actions that maximize information about future states.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Output action distribution parameters
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        features = self.net(obs)
        logits = self.action_head(features)
        return torch.distributions.Categorical(logits=logits)

    def sample(self, obs: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        dist = self.forward(obs)
        return dist.sample((n_samples,))


class ForwardDynamics(nn.Module):
    """
    Forward dynamics model p(s'|s, a) - predicts next state distribution.

    Used to estimate mutual information I(A; S'|S) for empowerment.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        ensemble_size: int = 5  # Ensemble for uncertainty estimation
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.ensemble_size = ensemble_size

        # Ensemble of forward models for uncertainty
        self.models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, obs_dim * 2)  # Mean and log_var
            )
            for _ in range(ensemble_size)
        ])

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict next state distribution.

        Returns:
            mean: Predicted next state mean
            var: Predicted next state variance
            epistemic_uncertainty: Disagreement between ensemble members
        """
        # One-hot encode discrete actions
        if len(action.shape) == 1:
            action_onehot = F.one_hot(action.long(), self.action_dim).float()
        else:
            action_onehot = action

        x = torch.cat([obs, action_onehot], dim=-1)

        means = []
        log_vars = []

        for model in self.models:
            output = model(x)
            mean, log_var = output.chunk(2, dim=-1)
            log_var = torch.clamp(log_var, -10, 2)  # Stability
            means.append(mean)
            log_vars.append(log_var)

        means = torch.stack(means)  # [ensemble, batch, obs_dim]
        log_vars = torch.stack(log_vars)

        # Aggregate predictions
        mean = means.mean(dim=0)
        var = (torch.exp(log_vars) + means ** 2).mean(dim=0) - mean ** 2

        # Epistemic uncertainty = disagreement between ensemble members
        epistemic = means.var(dim=0).mean(dim=-1)

        return mean, var, epistemic

    def get_loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor
    ) -> torch.Tensor:
        """Compute negative log-likelihood loss for all ensemble members."""
        if len(action.shape) == 1:
            action_onehot = F.one_hot(action.long(), self.action_dim).float()
        else:
            action_onehot = action

        x = torch.cat([obs, action_onehot], dim=-1)

        total_loss = 0
        for model in self.models:
            output = model(x)
            mean, log_var = output.chunk(2, dim=-1)
            log_var = torch.clamp(log_var, -10, 2)

            # Gaussian NLL
            diff = next_obs - mean
            loss = 0.5 * (log_var + diff ** 2 / torch.exp(log_var)).mean()
            total_loss += loss

        return total_loss / self.ensemble_size


class EmpowermentModule:
    """
    Computes empowerment as intrinsic motivation.

    Empowerment = I(A; S'|S) = H(S'|S) - H(S'|S, A)

    High empowerment means: Given my current state, my actions have
    significant, distinguishable effects on my future.

    This is approximated via variational methods using:
    1. A source network q(a|s) that proposes "empowering" actions
    2. A forward model p(s'|s,a) that predicts outcomes
    3. A planning network that reconstructs actions from state transitions
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_steps: int = 1,  # Planning horizon for empowerment
        hidden_dim: int = 256,
        learning_rate: float = 1e-4,
        n_samples: int = 10,  # Action samples for empowerment estimation
        device: str = "cpu"
    ):
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_steps = n_steps
        self.n_samples = n_samples

        # Source distribution - proposes empowering actions
        self.source = SourceNetwork(obs_dim, action_dim, hidden_dim).to(device)

        # Forward dynamics model
        self.dynamics = ForwardDynamics(obs_dim, action_dim, hidden_dim).to(device)

        # Planning network: reconstructs action from (s, s') transition
        # This estimates I(A; S'|S) via variational lower bound
        self.planning = nn.Sequential(
            nn.Linear(obs_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(device)

        self.optimizer = optim.Adam(
            list(self.source.parameters()) +
            list(self.dynamics.parameters()) +
            list(self.planning.parameters()),
            lr=learning_rate
        )

        # Statistics
        self.total_empowerment = 0
        self.count = 0

    def compute_empowerment(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Estimate empowerment at given state(s).

        Uses variational lower bound:
        E ≥ E_q(a|s)[log p(a|s,s')] + H(q(a|s))

        Where p(a|s,s') is approximated by the planning network.
        """
        batch_size = obs.shape[0]

        # Sample actions from source
        source_dist = self.source(obs)
        actions = source_dist.sample((self.n_samples,))  # [n_samples, batch]

        empowerment = torch.zeros(batch_size, device=self.device)

        for i in range(self.n_samples):
            action = actions[i]

            # Predict next state
            with torch.no_grad():
                next_obs_mean, next_obs_var, _ = self.dynamics(obs, action)
                # Sample from predicted distribution
                next_obs = next_obs_mean + torch.sqrt(next_obs_var) * torch.randn_like(next_obs_mean)

            # Reconstruct action probability using planning network
            planning_input = torch.cat([obs, next_obs], dim=-1)
            action_logits = self.planning(planning_input)
            planning_dist = torch.distributions.Categorical(logits=action_logits)

            # log p(a|s, s')
            log_prob = planning_dist.log_prob(action)
            empowerment += log_prob

        # Average over samples and add entropy of source
        empowerment = empowerment / self.n_samples + source_dist.entropy()

        self.total_empowerment += empowerment.mean().item()
        self.count += 1

        return empowerment

    def compute_intrinsic_reward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        update: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute empowerment-based intrinsic reward.

        Reward = Empowerment at next state + transition surprise

        This encourages reaching states with high future control
        AND experiencing unpredictable (but controllable) transitions.
        """
        # Empowerment at next state
        with torch.no_grad():
            emp = self.compute_empowerment(next_obs)

        # Transition surprise (epistemic uncertainty)
        _, _, epistemic = self.dynamics(obs, action)

        # Combined reward
        intrinsic_reward = emp + 0.1 * epistemic

        if update:
            self._update(obs, action, next_obs)

        return intrinsic_reward.detach(), {
            'empowerment': emp.mean().item(),
            'epistemic': epistemic.mean().item()
        }

    def _update(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor
    ):
        """Update all networks."""
        # Forward dynamics loss
        dynamics_loss = self.dynamics.get_loss(obs, action, next_obs)

        # Planning network loss (predict action from transition)
        planning_input = torch.cat([obs, next_obs], dim=-1)
        action_logits = self.planning(planning_input)
        planning_loss = F.cross_entropy(action_logits, action.long())

        # Source network loss (maximize empowerment)
        emp = self.compute_empowerment(obs)
        source_loss = -emp.mean()  # Maximize empowerment

        # Total loss
        loss = dynamics_loss + planning_loss + 0.01 * source_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.source.parameters()) +
            list(self.dynamics.parameters()) +
            list(self.planning.parameters()),
            1.0
        )
        self.optimizer.step()

    def get_avg_empowerment(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total_empowerment / self.count

    def reset_stats(self):
        self.total_empowerment = 0
        self.count = 0


class CausalCuriosity:
    """
    Curiosity about CAUSALITY, not just prediction.

    Key insight: Some things are hard to predict because they're random,
    others because we don't understand the causal mechanism.

    We only want curiosity about the latter!

    Uses interventional curiosity: Compare observational predictions
    with what happens when we actively intervene.

    Causal Curiosity = |P(s'|do(a), s) - P(s'|s)| for actions we can take

    High causal curiosity = "My actions have surprising effects here"
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        learning_rate: float = 1e-4,
        device: str = "cpu"
    ):
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Observational model: P(s'|s) - what happens without intervention
        self.observational = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        ).to(device)

        # Interventional model: P(s'|do(a), s) - what happens WITH intervention
        self.interventional = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        ).to(device)

        self.optimizer = optim.Adam(
            list(self.observational.parameters()) +
            list(self.interventional.parameters()),
            lr=learning_rate
        )

    def compute_causal_curiosity(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        update: bool = True
    ) -> torch.Tensor:
        """
        Measure how much our action changes the outcome vs passive observation.

        High value = our action had a surprising causal effect.
        """
        # What we'd predict without knowing the action (observational)
        obs_pred = self.observational(obs)

        # What we predict given the action (interventional)
        if len(action.shape) == 1:
            action_onehot = F.one_hot(action.long(), self.action_dim).float()
        else:
            action_onehot = action

        int_input = torch.cat([obs, action_onehot], dim=-1)
        int_pred = self.interventional(int_input)

        # Causal effect = how different is the interventional prediction
        causal_effect = torch.mean((int_pred - obs_pred) ** 2, dim=-1)

        # Also measure prediction error for interventional model
        int_error = torch.mean((int_pred - next_obs) ** 2, dim=-1)

        # Curiosity = causal effect + interventional surprise
        curiosity = causal_effect + 0.5 * int_error

        if update:
            # Update models
            obs_loss = F.mse_loss(obs_pred, next_obs)
            int_loss = F.mse_loss(int_pred, next_obs)

            loss = obs_loss + int_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return curiosity.detach()
