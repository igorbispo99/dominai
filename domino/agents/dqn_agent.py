"""ε-greedy DQN agent backed by a plugable ``Model``."""
from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch

from domino.agents.base import Agent
from domino.core.encoding import encode_state
from domino.core.game import GameState
from domino.models.base import Model


class DQNAgent(Agent):
    def __init__(
        self,
        model: Model,
        epsilon: float = 0.0,
        device: str = "cpu",
        seed: Optional[int] = None,
        player_id: int = 0,
    ) -> None:
        self.model = model
        self.epsilon = epsilon
        self.device = device
        self._rng = random.Random(seed)
        self.player_id = player_id

    def select_action(self, state: GameState, legal_mask: np.ndarray) -> int:
        legal = np.flatnonzero(legal_mask)
        if len(legal) == 0:
            raise RuntimeError("no legal actions available")
        if self._rng.random() < self.epsilon:
            return int(self._rng.choice(legal))
        return self._greedy(state, legal_mask)

    def _greedy(self, state: GameState, legal_mask: np.ndarray) -> int:
        obs = encode_state(state, player=self.player_id)
        with torch.no_grad():
            q = self.model(torch.from_numpy(obs).to(self.device).unsqueeze(0))
            q = q.squeeze(0).cpu().numpy()
        q_masked = np.where(legal_mask, q, -np.inf)
        return int(np.argmax(q_masked))
