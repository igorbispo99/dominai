"""Abstract agent interface. Agents consume ``(state, legal_mask)`` and return an action id."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from domino.core.game import GameState


class Agent(ABC):
    @abstractmethod
    def select_action(self, state: GameState, legal_mask: np.ndarray) -> int: ...

    def reset(self) -> None:
        """Called at the start of every new game. Override if the agent holds per-episode state."""
