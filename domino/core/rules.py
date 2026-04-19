"""Rule variants: block vs draw. Selected via config."""
from __future__ import annotations

from abc import ABC, abstractmethod


class Rule(ABC):
    mode: str = ""

    @abstractmethod
    def initial_hand_size(self, num_players: int) -> int: ...


class BlockRule(Rule):
    mode = "block"

    def initial_hand_size(self, num_players: int) -> int:
        if num_players < 2 or num_players > 4:
            raise ValueError("block mode supports 2–4 players")
        return 7 if num_players <= 4 else 28 // num_players


class DrawRule(Rule):
    mode = "draw"

    def initial_hand_size(self, num_players: int) -> int:
        if num_players < 2 or num_players > 4:
            raise ValueError("draw mode supports 2–4 players")
        return {2: 7, 3: 6, 4: 5}[num_players]


def build_rule(mode: str) -> Rule:
    if mode == "block":
        return BlockRule()
    if mode == "draw":
        return DrawRule()
    raise ValueError(f"unknown rule mode: {mode!r}")
