"""Save / load model checkpoints with their config so they can be reconstructed."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from domino.core.encoding import STATE_DIM
from domino.core.game import ACTION_DIM
from domino.models import build_model
from domino.models.base import Model


def save_checkpoint(
    path: str | os.PathLike,
    model: Model,
    model_cfg: Dict[str, Any],
    extra: Dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_cfg": model_cfg,
        "state_dict": model.state_dict(),
        "state_dim": model.state_dim,
        "action_dim": model.action_dim,
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | os.PathLike, map_location: str = "cpu"
) -> Tuple[Model, Dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model_cfg = payload["model_cfg"]
    model = build_model(
        model_cfg,
        state_dim=payload.get("state_dim", STATE_DIM),
        action_dim=payload.get("action_dim", ACTION_DIM),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload.get("extra", {})
