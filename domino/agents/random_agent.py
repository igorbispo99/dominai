"""Uniform-random legal-action baseline. Useful for evaluation."""
from __future__ import annotations

import random

import numpy as np

from domino.agents.base import Agent
from domino.core.game import GameState


class RandomAgent(Agent):
    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def select_action(self, state: GameState, legal_mask: np.ndarray) -> int:
        legal = np.flatnonzero(legal_mask)
        if len(legal) == 0:
            raise RuntimeError("no legal actions available")
        return int(self._rng.choice(legal))
