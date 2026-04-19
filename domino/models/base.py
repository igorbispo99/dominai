"""Abstract plugin interface for Q-function models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
from torch import Tensor, nn


class Model(nn.Module, ABC):
    """Maps an encoded state vector to Q-values over the discrete action space.

    Subclasses must implement ``forward(x) -> (B, action_dim)`` and provide
    a classmethod ``from_config(cfg, state_dim, action_dim)`` used by the
    registry to construct instances from YAML.
    """

    name: str = "abstract"

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor: ...

    @classmethod
    @abstractmethod
    def from_config(
        cls, cfg: Dict[str, Any], state_dim: int, action_dim: int
    ) -> "Model": ...

    def save(self, path: str, extra: Dict[str, Any] | None = None) -> None:
        payload: Dict[str, Any] = {
            "model_name": self.name,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "state_dict": self.state_dict(),
            "extra": extra or {},
        }
        torch.save(payload, path)

    @classmethod
    def load_state(cls, path: str, map_location: str = "cpu") -> Dict[str, Any]:
        return torch.load(path, map_location=map_location, weights_only=False)
