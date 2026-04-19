"""Residual MLP Q-network — deeper alternative to the baseline."""
from __future__ import annotations

from typing import Any, Dict

import torch
from torch import Tensor, nn

from domino.models.base import Model
from domino.models.registry import register_model


class _ResidualBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        h = self.act(self.fc1(x))
        h = self.fc2(h)
        return self.norm(self.act(x + h))


@register_model("resnet")
class ResidualMLP(Model):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        num_blocks: int = 3,
    ) -> None:
        super().__init__(state_dim, action_dim)
        self.input = nn.Linear(state_dim, hidden_size)
        self.blocks = nn.ModuleList(
            [_ResidualBlock(hidden_size) for _ in range(num_blocks)]
        )
        self.head = nn.Linear(hidden_size, action_dim)

    def forward(self, x: Tensor) -> Tensor:
        h = torch.relu(self.input(x))
        for blk in self.blocks:
            h = blk(h)
        return self.head(h)

    @classmethod
    def from_config(
        cls, cfg: Dict[str, Any], state_dim: int, action_dim: int
    ) -> "ResidualMLP":
        return cls(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=cfg.get("hidden_size", 256),
            num_blocks=cfg.get("num_blocks", 3),
        )
