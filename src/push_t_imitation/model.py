"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.hidden_dims1, self.hidden_dims2, self.hidden_dims3 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dims1),
            nn.ReLU(),
            nn.Linear(self.hidden_dims1, self.hidden_dims2),
            nn.ReLU(),
            nn.Linear(self.hidden_dims2, self.hidden_dims3),
            nn.ReLU(),
            nn.Linear(self.hidden_dims3, self.chunk_size * self.action_dim)
        )

        
    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor
    ) -> torch.Tensor:
        action_pred = self.sample_actions(state)
        loss = nn.MSELoss()(action_pred, action_chunk)
        return loss


    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        return self.net(state).reshape(-1, self.chunk_size, self.action_dim)
        


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.state_dim = state_dim
        self.hidden_dims1, self.hidden_dims2, self.hidden_dims3 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(self.state_dim + self.chunk_size * self.action_dim + 1, self.hidden_dims1),# +1 for time embedding
            nn.ReLU(),
            nn.Linear(self.hidden_dims1, self.hidden_dims2),
            nn.ReLU(),
            nn.Linear(self.hidden_dims2, self.hidden_dims3),
            nn.ReLU(),
            nn.Linear(self.hidden_dims3, self.chunk_size * self.action_dim)
        )
    

    def forward(
        self,
        tx : torch.Tensor,
    ) -> torch.Tensor:
        v_pred = self.net(tx)
        return v_pred

    def get_input(self, state: torch.Tensor, action: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        action = action.reshape(-1, self.chunk_size * self.action_dim)
        return torch.cat([state, action, t], dim=-1)

    def interpolate_linear(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_0 = x_0.reshape(-1, self.chunk_size * self.action_dim)
        x_1 = x_1.reshape(-1, self.chunk_size * self.action_dim)
        t = t.reshape(-1, 1)
        return (1 - t) * x_0 + t * x_1

    def get_target_v(self, action: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        return (action_chunk - action).reshape(-1, self.chunk_size * self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        action = torch.randn_like(action_chunk).to(self.device)
        t = torch.randn(action_chunk.shape[0]).reshape(-1,1).to(self.device)
        x_t = self.interpolate_linear(action, action_chunk, t)
        input = self.get_input(state, x_t, t)
        v_target = self.get_target_v(action, action_chunk)
        v_pred = self.forward(input)
        loss = nn.MSELoss()(v_pred, v_target)
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        ts = torch.linspace(0, 1, num_steps + 1, device=self.device)
        actions = torch.randn(state.shape[0], self.chunk_size * self.action_dim, device=self.device)
        for i in range(num_steps):
            t = ts[i]
            dt = ts[i + 1] - t
            t = t.reshape(state.shape[0], 1)
            dt = dt.reshape(state.shape[0], 1)
            input = self.get_input(state, actions, t)

            v_pred = self.forward(input)
            actions = actions + dt * v_pred
        return actions.reshape(-1, self.chunk_size, self.action_dim)

PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
