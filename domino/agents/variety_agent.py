"""Rule-based opponent that preserves hand variety by spending abundant numbers first."""
from __future__ import annotations

import random
from typing import Optional

import numpy as np

from domino.agents.base import Agent
from domino.core.game import GameState, PASS_ACTION, action_to_move
from domino.core.piece import DECK


class VarietyAgent(Agent):
    """Play legal tiles that consume the most frequent numbers in hand.

    The score prioritizes pieces containing high-frequency numbers and avoids
    spending rare values when possible, which helps preserve hand variety.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def select_action(self, state: GameState, legal_mask: np.ndarray) -> int:
        legal = np.flatnonzero(legal_mask)
        if len(legal) == 0:
            raise RuntimeError("no legal actions available")

        play_actions = [int(a) for a in legal if a != PASS_ACTION]
        if not play_actions:
            return int(legal[0])

        hand = state.hands[state.current_player]
        count_by_value = [0] * 7
        for piece_idx in hand:
            piece = DECK[piece_idx]
            count_by_value[piece.low] += 1
            if piece.high != piece.low:
                count_by_value[piece.high] += 1

        best_score: tuple[int, int, int, int] | None = None
        best: list[int] = []
        for a in play_actions:
            piece_idx, _ = action_to_move(a)
            piece = DECK[piece_idx]
            c_low = count_by_value[piece.low]
            c_high = count_by_value[piece.high]
            # 1) spend abundant values first, 2) avoid spending rare values,
            # 3) prefer doubles (consume one value), 4) break remaining ties by pip count.
            score = (
                max(c_low, c_high),
                min(c_low, c_high),
                int(piece.is_double()),
                piece.pip_count(),
            )
            if best_score is None or score > best_score:
                best_score = score
                best = [a]
            elif score == best_score:
                best.append(a)

        return int(self._rng.choice(best))
