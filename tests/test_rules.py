"""End-to-end sanity tests for the core game engine."""
import random

import numpy as np
import pytest

from domino.core import (
    ACTION_DIM,
    DECK,
    PASS_ACTION,
    new_game,
)
from domino.core.game import legal_actions, step, move_to_action


def _play_out(state, agent_action):
    """Run a game to completion using ``agent_action(state, mask) -> int``."""
    done = False
    info = {}
    steps = 0
    while not done:
        mask = legal_actions(state)
        assert mask.any(), "no legal actions but game not terminal"
        action = agent_action(state, mask)
        assert mask[action], "agent chose an illegal action"
        state, done, info = step(state, action)
        steps += 1
        assert steps < 10_000, "runaway game"
    return state, info


def _random_agent(rng):
    def act(state, mask):
        legal = np.flatnonzero(mask)
        return int(rng.choice(legal))

    return act


def test_block_mode_terminates():
    rng = random.Random(42)
    state = new_game(num_players=4, mode="block", rng=rng)
    state, info = _play_out(state, _random_agent(rng))
    assert "winner" in info
    total_played_plus_hands = len(state.played) + sum(len(h) for h in state.hands)
    # No boneyard in a typical 4-player block game: all 28 pieces accounted for.
    assert total_played_plus_hands == 28


def test_draw_mode_terminates_with_boneyard():
    rng = random.Random(7)
    state = new_game(num_players=4, mode="draw", rng=rng)
    # Draw mode with 4 players: 5 per hand, 8 in boneyard.
    assert sum(len(h) for h in state.hands) == 20
    assert len(state.boneyard) == 8
    state, info = _play_out(state, _random_agent(rng))
    assert "winner" in info


def test_legal_mask_on_empty_table():
    rng = random.Random(0)
    state = new_game(num_players=2, mode="block", rng=rng)
    mask = legal_actions(state)
    hand = state.hands[state.current_player]
    # Every piece in hand should produce exactly one legal action (side=0).
    expected = {move_to_action(pi, 0) for pi in hand}
    assert set(np.flatnonzero(mask).tolist()) == expected


def test_pass_when_blocked():
    # Force a pathological starting state: player has only one piece, doesn't match ends.
    from domino.core.game import GameState

    # Build a state manually: current player holds [6|6], table ends are 0/0.
    p66 = 27  # (6,6) is last in canonical order
    p00 = 0
    state = GameState(
        hands=((p66,), ()),  # only p1 matters; p2 already won
        boneyard=(),
        played=(p00,),
        left_end=0,
        right_end=0,
        current_player=0,
        num_players=2,
        mode="block",
        passes_in_a_row=0,
    )
    mask = legal_actions(state)
    assert mask[PASS_ACTION]
    assert mask.sum() == 1


def test_action_dim_matches():
    assert ACTION_DIM == 57
    assert len(DECK) == 28
