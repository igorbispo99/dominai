"""Plain MLP Q-network — default baseline."""
from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch import Tensor, nn

from domino.models.base import Model
from domino.models.registry import register_model


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
}


@register_model("mlp")
class MLP(Model):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: List[int],
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__(state_dim, action_dim)
        act_cls = _ACTIVATIONS[activation]
        layers: List[nn.Module] = []
        prev = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    @classmethod
    def from_config(
        cls, cfg: Dict[str, Any], state_dim: int, action_dim: int
    ) -> "MLP":
        return cls(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_sizes=cfg.get("hidden_sizes", [256, 256]),
            activation=cfg.get("activation", "relu"),
            dropout=cfg.get("dropout", 0.0),
        )
