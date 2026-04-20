"""MLP Q-network with optional LayerNorm + dueling head."""
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
        layer_norm: bool = True,
        dueling: bool = True,
    ) -> None:
        super().__init__(state_dim, action_dim)
        act_cls = _ACTIVATIONS[activation]
        trunk: List[nn.Module] = []
        prev = state_dim
        for h in hidden_sizes:
            trunk.append(nn.Linear(prev, h))
            if layer_norm:
                trunk.append(nn.LayerNorm(h))
            trunk.append(act_cls())
            if dropout > 0:
                trunk.append(nn.Dropout(dropout))
            prev = h
        self.trunk = nn.Sequential(*trunk)
        self.dueling = dueling
        if dueling:
            self.value_head = nn.Sequential(
                nn.Linear(prev, prev),
                act_cls(),
                nn.Linear(prev, 1),
            )
            self.adv_head = nn.Sequential(
                nn.Linear(prev, prev),
                act_cls(),
                nn.Linear(prev, action_dim),
            )
        else:
            self.head = nn.Linear(prev, action_dim)

    def forward(self, x: Tensor) -> Tensor:
        h = self.trunk(x)
        if self.dueling:
            v = self.value_head(h)
            a = self.adv_head(h)
            return v + (a - a.mean(dim=-1, keepdim=True))
        return self.head(h)

    @classmethod
    def from_config(
        cls, cfg: Dict[str, Any], state_dim: int, action_dim: int
    ) -> "MLP":
        return cls(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_sizes=cfg.get("hidden_sizes", [512, 512, 256]),
            activation=cfg.get("activation", "relu"),
            dropout=cfg.get("dropout", 0.0),
            layer_norm=cfg.get("layer_norm", True),
            dueling=cfg.get("dueling", True),
        )
