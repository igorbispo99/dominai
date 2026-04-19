"""Decorator-based registry for swapping Q-function architectures from YAML."""
from __future__ import annotations

from typing import Any, Dict, List, Type

from domino.models.base import Model

_REGISTRY: Dict[str, Type[Model]] = {}


def register_model(name: str):
    def _decorate(cls: Type[Model]) -> Type[Model]:
        if name in _REGISTRY:
            raise ValueError(f"model {name!r} already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return _decorate


def build_model(cfg: Dict[str, Any], state_dim: int, action_dim: int) -> Model:
    name = cfg.get("name")
    if name is None:
        raise ValueError("model config must have a 'name' field")
    if name not in _REGISTRY:
        raise KeyError(
            f"model {name!r} not registered. Available: {sorted(_REGISTRY)}"
        )
    cls = _REGISTRY[name]
    return cls.from_config(cfg, state_dim=state_dim, action_dim=action_dim)


def available_models() -> List[str]:
    return sorted(_REGISTRY)
