"""Rule-based opponent that plays the legal piece with the highest pip count.

Simple but non-trivial baseline: prefers dumping heavy pieces, falling back to
PASS if no legal move. Useful to break self-play symmetry during training.
"""
from __future__ import annotations

import random
from typing import Optional

import numpy as np

from domino.agents.base import Agent
from domino.core.game import GameState, PASS_ACTION, action_to_move
from domino.core.piece import DECK


class HeuristicAgent(Agent):
    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def select_action(self, state: GameState, legal_mask: np.ndarray) -> int:
        legal = np.flatnonzero(legal_mask)
        if len(legal) == 0:
            raise RuntimeError("no legal actions available")

        play_actions = [int(a) for a in legal if a != PASS_ACTION]
        if not play_actions:
            return int(legal[0])  # only PASS available

        # pick legal action with highest pip count; ties broken randomly
        best_score = -1
        best: list[int] = []
        for a in play_actions:
            piece_idx, _ = action_to_move(a)
            score = DECK[piece_idx].pip_count()
            if score > best_score:
                best_score = score
                best = [a]
            elif score == best_score:
                best.append(a)
        return int(self._rng.choice(best))
