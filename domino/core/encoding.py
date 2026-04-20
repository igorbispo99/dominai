"""State encoding — fixed-size float32 vector for any ML model.

Layout (202 dims total):
  [  0: 28] hand          — which pieces the observing player holds
  [ 28: 56] played        — which pieces have been played (any player)
  [ 56: 70] ends          — left_end one-hot (7) + right_end one-hot (7)
  [ 70: 72] empty_flags   — 1 if left/right end is None (empty board)
  [ 72: 76] opp_counts    — normalised hand sizes, circular from observer
  [ 76:188] per_player    — 4 × 28 bits: which pieces each player placed,
                            circular from observer (slot 0 = self)
  [188:192] passes        — normalised pass count per player, same circular order
  [192:202] remaining_per_value — for each pip value 0..6, count of unseen
                                  pieces with that value (normalised by 8),
                                  padded to 10 dims for alignment. Helps the
                                  net reason about which ends are "safe".
"""
from __future__ import annotations

import numpy as np

from domino.core.game import GameState
from domino.core.piece import DECK, NUM_PIECES

HAND_DIM       = NUM_PIECES          # 28
PLAYED_DIM     = NUM_PIECES          # 28
ENDS_DIM       = 14                  # 7 one-hot left + 7 one-hot right
EMPTY_DIM      = 2                   # is-empty flag per end
OPP_DIM        = 4
PER_PLAYER_DIM = 4 * NUM_PIECES      # 112  (4 players × 28 pieces)
PASSES_DIM     = 4
REMAIN_DIM     = 10                  # 7 used + 3 pad

STATE_DIM = (
    HAND_DIM + PLAYED_DIM + ENDS_DIM + EMPTY_DIM + OPP_DIM
    + PER_PLAYER_DIM + PASSES_DIM + REMAIN_DIM
)  # 202

_OFF_PLAYED     = HAND_DIM
_OFF_ENDS       = HAND_DIM + PLAYED_DIM
_OFF_EMPTY      = _OFF_ENDS + ENDS_DIM
_OFF_OPP        = _OFF_EMPTY + EMPTY_DIM
_OFF_PER_PLAYER = _OFF_OPP + OPP_DIM
_OFF_PASSES     = _OFF_PER_PLAYER + PER_PLAYER_DIM
_OFF_REMAIN     = _OFF_PASSES + PASSES_DIM

_MAX_HAND_NORM  = 7.0
_MAX_PASS_NORM  = 7.0
_MAX_VALUE_COUNT = 8.0  # each pip value appears on up to 8 pieces (double + 7 mixed)


def encode_state(state: GameState, player: int) -> np.ndarray:
    """Encode state from ``player``'s perspective into a (STATE_DIM,) float32 vector."""
    vec = np.zeros(STATE_DIM, dtype=np.float32)

    # --- hand ----------------------------------------------------------------
    for pi in state.hands[player]:
        vec[pi] = 1.0

    # --- globally played pieces ----------------------------------------------
    for pi in state.played:
        vec[_OFF_PLAYED + pi] = 1.0

    # --- board ends (one-hot) ------------------------------------------------
    if state.left_end is not None:
        vec[_OFF_ENDS + state.left_end] = 1.0
    else:
        vec[_OFF_EMPTY] = 1.0
    if state.right_end is not None:
        vec[_OFF_ENDS + 7 + state.right_end] = 1.0
    else:
        vec[_OFF_EMPTY + 1] = 1.0

    # --- opponent hand sizes (circular from observer) ------------------------
    for k in range(min(3, state.num_players - 1)):
        opp = (player + 1 + k) % state.num_players
        vec[_OFF_OPP + k] = len(state.hands[opp]) / _MAX_HAND_NORM

    # --- per-player placed pieces (circular from observer) -------------------
    if state.player_played:
        for k in range(min(4, state.num_players)):
            p = (player + k) % state.num_players
            for pi in state.player_played[p]:
                vec[_OFF_PER_PLAYER + k * NUM_PIECES + pi] = 1.0

    # --- per-player pass counts (circular from observer) --------------------
    if state.player_passes:
        for k in range(min(4, state.num_players)):
            p = (player + k) % state.num_players
            vec[_OFF_PASSES + k] = min(state.player_passes[p] / _MAX_PASS_NORM, 1.0)

    # --- remaining (unseen) pieces per pip value ----------------------------
    # counts how many pieces with each pip value have NOT yet been played.
    seen = np.zeros(7, dtype=np.float32)
    for pi in state.played:
        p = DECK[pi]
        seen[p.low] += 1.0
        if p.low != p.high:
            seen[p.high] += 1.0
    # total occurrences of each value across the deck: value v appears on
    # 8 pieces (one per pair + the double counted once).
    total = np.array([8.0] * 7, dtype=np.float32)
    remaining = np.clip(total - seen, 0.0, None) / _MAX_VALUE_COUNT
    vec[_OFF_REMAIN:_OFF_REMAIN + 7] = remaining

    return vec
