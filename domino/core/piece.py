"""Canonical domino pieces and indexing (double-six)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True, order=True)
class Piece:
    low: int
    high: int

    def __post_init__(self) -> None:
        if not (0 <= self.low <= self.high <= 6):
            raise ValueError(f"invalid piece values: ({self.low},{self.high})")

    def is_double(self) -> bool:
        return self.low == self.high

    def matches(self, value: int) -> bool:
        return self.low == value or self.high == value

    def other_end(self, value: int) -> int:
        if self.low == value:
            return self.high
        if self.high == value:
            return self.low
        raise ValueError(f"piece {self} does not contain value {value}")

    def pip_count(self) -> int:
        return self.low + self.high

    def __repr__(self) -> str:
        return f"[{self.low}|{self.high}]"


def _build_deck() -> Tuple[Piece, ...]:
    pieces = []
    for low in range(7):
        for high in range(low, 7):
            pieces.append(Piece(low, high))
    return tuple(pieces)


DECK: Tuple[Piece, ...] = _build_deck()
NUM_PIECES: int = len(DECK)  # 28

_PIECE_TO_IDX: Dict[Piece, int] = {p: i for i, p in enumerate(DECK)}


def piece_index(piece: Piece) -> int:
    return _PIECE_TO_IDX[piece]
