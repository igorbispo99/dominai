import random

import numpy as np

from domino.core import new_game
from domino.core.encoding import (
    STATE_DIM,
    _OFF_ENDS,
    _OFF_OPP,
    _OFF_PLAYED,
    encode_state,
)


def test_state_dim():
    assert STATE_DIM == 62


def test_empty_table_encoding():
    rng = random.Random(123)
    state = new_game(num_players=4, mode="block", rng=rng)
    vec = encode_state(state, player=state.current_player)
    assert vec.shape == (STATE_DIM,)
    # ends should be -1 on empty table
    assert vec[_OFF_ENDS] == -1.0
    assert vec[_OFF_ENDS + 1] == -1.0
    # played block all zeros
    assert vec[_OFF_PLAYED : _OFF_PLAYED + 28].sum() == 0
    # hand block: exactly 7 ones
    assert vec[:28].sum() == 7
    # opponent counts: 3 opponents with 7 each (normalized to 1.0), slot 3 stays 0
    assert vec[_OFF_OPP] == 1.0
    assert vec[_OFF_OPP + 1] == 1.0
    assert vec[_OFF_OPP + 2] == 1.0
    assert vec[_OFF_OPP + 3] == 0.0


def test_hand_encoding_bijection():
    rng = random.Random(1)
    state = new_game(num_players=2, mode="block", rng=rng)
    vec = encode_state(state, player=0)
    hand_idxs = set(np.flatnonzero(vec[:28]).tolist())
    assert hand_idxs == set(state.hands[0])
