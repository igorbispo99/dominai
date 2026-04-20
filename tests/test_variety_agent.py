import numpy as np

from domino.agents.variety_agent import VarietyAgent
from domino.core.game import ACTION_DIM, GameState, PASS_ACTION, move_to_action
from domino.core.piece import Piece, piece_index


def _state_with_hand(hand: tuple[int, ...]) -> GameState:
    return GameState(
        hands=(hand, ()),
        boneyard=(),
        played=(),
        left_end=2,
        right_end=3,
        current_player=0,
        num_players=2,
        mode="block",
        passes_in_a_row=0,
    )


def test_variety_agent_plays_abundant_number_first() -> None:
    # Hand has three pieces with 2 and only one with 3.
    p20 = piece_index(Piece(0, 2))
    p24 = piece_index(Piece(2, 4))
    p26 = piece_index(Piece(2, 6))
    p13 = piece_index(Piece(1, 3))
    state = _state_with_hand((p20, p24, p26, p13))

    mask = np.zeros(ACTION_DIM, dtype=bool)
    mask[move_to_action(p20, 0)] = True
    mask[move_to_action(p13, 0)] = True

    agent = VarietyAgent(seed=7)
    action = agent.select_action(state, mask)
    assert action == move_to_action(p20, 0)


def test_variety_agent_avoids_spending_rare_value_when_possible() -> None:
    # Both actions use value 2, but [2|2] avoids consuming rare value 3.
    p22 = piece_index(Piece(2, 2))
    p23 = piece_index(Piece(2, 3))
    p35 = piece_index(Piece(3, 5))
    state = _state_with_hand((p22, p23, p35))

    mask = np.zeros(ACTION_DIM, dtype=bool)
    mask[move_to_action(p22, 0)] = True
    mask[move_to_action(p23, 0)] = True

    agent = VarietyAgent(seed=7)
    action = agent.select_action(state, mask)
    assert action == move_to_action(p22, 0)


def test_variety_agent_passes_when_only_pass_is_legal() -> None:
    state = _state_with_hand(())
    mask = np.zeros(ACTION_DIM, dtype=bool)
    mask[PASS_ACTION] = True

    agent = VarietyAgent(seed=7)
    action = agent.select_action(state, mask)
    assert action == PASS_ACTION
