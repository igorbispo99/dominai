"""State encoding as specified: 28 (hand) + 28 (played) + 2 (ends) + 4 (opp counts) = 62."""
from __future__ import annotations

import numpy as np

from domino.core.game import GameState
from domino.core.piece import NUM_PIECES

HAND_DIM = NUM_PIECES
PLAYED_DIM = NUM_PIECES
ENDS_DIM = 2
OPP_DIM = 4
STATE_DIM = HAND_DIM + PLAYED_DIM + ENDS_DIM + OPP_DIM  # 62

_OFF_PLAYED = HAND_DIM
_OFF_ENDS = HAND_DIM + PLAYED_DIM
_OFF_OPP = _OFF_ENDS + ENDS_DIM

_MAX_HAND_NORM = 7.0  # typical starting hand size for double-six


def encode_state(state: GameState, player: int) -> np.ndarray:
    """Encode state from ``player``'s perspective into a (STATE_DIM,) float32 vector."""
    vec = np.zeros(STATE_DIM, dtype=np.float32)

    for pi in state.hands[player]:
        vec[pi] = 1.0

    for pi in state.played:
        vec[_OFF_PLAYED + pi] = 1.0

    vec[_OFF_ENDS] = -1.0 if state.left_end is None else state.left_end / 6.0
    vec[_OFF_ENDS + 1] = -1.0 if state.right_end is None else state.right_end / 6.0

    # Opponent piece counts in circular order after `player`; slot 3 is padding (0)
    # to match the user's 4-position specification. With 4 players there are only
    # 3 opponents; the extra slot stays zero.
    for k in range(min(3, state.num_players - 1)):
        opp = (player + 1 + k) % state.num_players
        vec[_OFF_OPP + k] = len(state.hands[opp]) / _MAX_HAND_NORM

    return vec
