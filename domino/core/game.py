"""Game state, transitions, and legal-action computation for double-six dominó."""
from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

import numpy as np

from domino.core.piece import DECK, NUM_PIECES
from domino.core.rules import build_rule

# --- Action space -----------------------------------------------------------
# Actions 0..55 : piece_idx * 2 + side  (side 0 = left end, 1 = right end)
# Action 56     : PASS (used in block mode; in draw mode it's only legal after
#                 mandatory drawing has exhausted the boneyard).
PASS_ACTION: int = 56
ACTION_DIM: int = 57

Action = int


def action_to_move(action: int) -> Tuple[int, int]:
    assert 0 <= action < PASS_ACTION
    return action // 2, action % 2


def move_to_action(piece_idx: int, side: int) -> int:
    assert 0 <= piece_idx < NUM_PIECES and side in (0, 1)
    return piece_idx * 2 + side


# --- State ------------------------------------------------------------------

@dataclass(frozen=True)
class GameState:
    hands: Tuple[Tuple[int, ...], ...]
    boneyard: Tuple[int, ...]
    played: Tuple[int, ...]
    left_end: Optional[int]
    right_end: Optional[int]
    current_player: int
    num_players: int
    mode: str
    passes_in_a_row: int
    last_winner: Optional[int] = None
    # per-player history (indexed by player id)
    player_played: Tuple[Tuple[int, ...], ...] = ()  # pieces each player placed
    player_passes: Tuple[int, ...] = ()              # pass count per player

    @property
    def is_empty_table(self) -> bool:
        return self.left_end is None


# --- Helpers ----------------------------------------------------------------

def _has_playable_piece(
    hand: Tuple[int, ...], left_end: Optional[int], right_end: Optional[int]
) -> bool:
    if left_end is None:
        return len(hand) > 0
    for pi in hand:
        p = DECK[pi]
        if p.matches(left_end) or p.matches(right_end):  # type: ignore[arg-type]
            return True
    return False


def _pip_count(hand: Tuple[int, ...]) -> int:
    return sum(DECK[i].pip_count() for i in hand)


def _choose_starter(hands: Tuple[Tuple[int, ...], ...]) -> int:
    best_player, best_key = 0, (-1, -1)
    for p, h in enumerate(hands):
        for pi in h:
            piece = DECK[pi]
            key = (1 if piece.is_double() else 0, piece.pip_count())
            if key > best_key:
                best_key = key
                best_player = p
    return best_player


def _finalize_blocked(state: GameState) -> GameState:
    pips = [_pip_count(h) for h in state.hands]
    min_pip = min(pips)
    winners = [i for i, p in enumerate(pips) if p == min_pip]
    winner = winners[0] if len(winners) == 1 else None
    return replace(state, last_winner=winner)


def _prepare_turn(state: GameState) -> GameState:
    """Apply mandatory draws in draw mode until current player can play or boneyard empty."""
    if state.mode != "draw":
        return state
    player = state.current_player
    hand = state.hands[player]
    boneyard = state.boneyard
    while not _has_playable_piece(hand, state.left_end, state.right_end) and boneyard:
        drawn = boneyard[0]
        boneyard = boneyard[1:]
        hand = tuple(sorted(hand + (drawn,)))
    if hand is state.hands[player] and boneyard is state.boneyard:
        return state
    new_hands = tuple(
        hand if i == player else state.hands[i] for i in range(state.num_players)
    )
    return replace(state, hands=new_hands, boneyard=boneyard)


# --- Public API -------------------------------------------------------------

def new_game(
    num_players: int,
    mode: str = "block",
    rng: Optional[random.Random] = None,
    hand_size: Optional[int] = None,
) -> GameState:
    if rng is None:
        rng = random.Random()
    rule = build_rule(mode)
    hs = hand_size if hand_size is not None else rule.initial_hand_size(num_players)

    deck_indices = list(range(NUM_PIECES))
    rng.shuffle(deck_indices)

    hands = []
    pos = 0
    for _ in range(num_players):
        hands.append(tuple(sorted(deck_indices[pos : pos + hs])))
        pos += hs
    hands_t = tuple(hands)
    boneyard = tuple(deck_indices[pos:])

    state = GameState(
        hands=hands_t,
        boneyard=boneyard,
        played=(),
        left_end=None,
        right_end=None,
        current_player=_choose_starter(hands_t),
        num_players=num_players,
        mode=mode,
        passes_in_a_row=0,
        last_winner=None,
        player_played=tuple(() for _ in range(num_players)),
        player_passes=tuple(0 for _ in range(num_players)),
    )
    return _prepare_turn(state)


def legal_actions(state: GameState) -> np.ndarray:
    """Return a boolean mask of shape (ACTION_DIM,)."""
    mask = np.zeros(ACTION_DIM, dtype=bool)
    if state.last_winner is not None and any(len(h) == 0 for h in state.hands):
        return mask  # terminal — nothing legal
    hand = state.hands[state.current_player]
    if state.is_empty_table:
        for pi in hand:
            mask[move_to_action(pi, 0)] = True
        return mask
    for pi in hand:
        piece = DECK[pi]
        if piece.matches(state.left_end):  # type: ignore[arg-type]
            mask[move_to_action(pi, 0)] = True
        if piece.matches(state.right_end):  # type: ignore[arg-type]
            mask[move_to_action(pi, 1)] = True
    if not mask.any():
        mask[PASS_ACTION] = True
    return mask


def step(state: GameState, action: int) -> Tuple[GameState, bool, Dict]:
    """Apply action. Returns (next_state_ready_for_next_decision, done, info).

    info carries ``winner`` on terminal steps (``None`` on tie).
    """
    info: Dict = {}
    if action == PASS_ACTION:
        new_passes = state.passes_in_a_row + 1
        next_player = (state.current_player + 1) % state.num_players
        new_player_passes = tuple(
            state.player_passes[i] + 1 if i == state.current_player else state.player_passes[i]
            for i in range(state.num_players)
        ) if state.player_passes else state.player_passes
        ns = replace(state, current_player=next_player, passes_in_a_row=new_passes,
                     player_passes=new_player_passes)
        if new_passes >= state.num_players:
            ns = _finalize_blocked(ns)
            info["winner"] = ns.last_winner
            return ns, True, info
        return _prepare_turn(ns), False, info

    piece_idx, side = action_to_move(action)
    piece = DECK[piece_idx]
    player = state.current_player

    new_hand = tuple(pi for pi in state.hands[player] if pi != piece_idx)
    new_hands = tuple(
        new_hand if i == player else state.hands[i] for i in range(state.num_players)
    )
    new_played = state.played + (piece_idx,)
    new_player_played = tuple(
        state.player_played[i] + (piece_idx,) if i == player else state.player_played[i]
        for i in range(state.num_players)
    ) if state.player_played else state.player_played

    if state.is_empty_table:
        new_left, new_right = piece.low, piece.high
    elif side == 0:
        new_left = piece.other_end(state.left_end)  # type: ignore[arg-type]
        new_right = state.right_end
    else:
        new_left = state.left_end
        new_right = piece.other_end(state.right_end)  # type: ignore[arg-type]

    if len(new_hand) == 0:
        ns = replace(
            state,
            hands=new_hands,
            played=new_played,
            left_end=new_left,
            right_end=new_right,
            passes_in_a_row=0,
            last_winner=player,
            player_played=new_player_played,
        )
        info["winner"] = player
        return ns, True, info

    next_player = (player + 1) % state.num_players
    ns = replace(
        state,
        hands=new_hands,
        played=new_played,
        left_end=new_left,
        right_end=new_right,
        current_player=next_player,
        passes_in_a_row=0,
        player_played=new_player_played,
    )
    return _prepare_turn(ns), False, info
