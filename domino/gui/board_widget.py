"""The table — horizontal strip of placed pieces with a scroll area."""
from __future__ import annotations

from typing import List, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from domino.core.piece import DECK
from domino.gui.piece_widget import PieceWidget


class BoardWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self._ends_label = QLabel("Mesa vazia")
        self._ends_label.setStyleSheet("font-weight: bold; padding: 4px;")
        outer.addWidget(self._ends_label)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self._inner = QWidget()
        self._row = QHBoxLayout(self._inner)
        self._row.setContentsMargins(8, 16, 8, 16)
        self._row.setSpacing(2)
        self._row.addStretch(1)
        self._row.addStretch(1)
        self._scroll.setWidget(self._inner)
        outer.addWidget(self._scroll, 1)

        # list of (piece_idx, "low"|"high" end touching LEFT side of drawn rect)
        self._placed: List[Tuple[int, int, int]] = []  # (piece_idx, left_val, right_val)

    def clear(self) -> None:
        while self._row.count() > 2:
            item = self._row.takeAt(1)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._placed = []
        self._ends_label.setText("Mesa vazia")

    def place(self, piece_idx: int, side: int, left_end: int, right_end: int) -> None:
        """Append the piece to the board following the linear layout.

        ``side`` is the side of the current table that the new piece is attached to:
        0 = left, 1 = right. ``left_end`` / ``right_end`` are the updated ends.
        """
        piece = DECK[piece_idx]
        horizontal = not piece.is_double()
        widget = PieceWidget(piece_idx, horizontal=horizontal)
        widget.set_disabled(True)
        if side == 0:
            # insert at the left (position 1 — right after the leading stretch)
            self._row.insertWidget(1, widget)
        else:
            # insert at the right (before the trailing stretch)
            self._row.insertWidget(self._row.count() - 1, widget)
        self._placed.append((piece_idx, left_end, right_end))
        self._ends_label.setText(f"Pontas: {left_end}  …  {right_end}")

    def set_ends(self, left_end, right_end) -> None:
        if left_end is None:
            self._ends_label.setText("Mesa vazia")
        else:
            self._ends_label.setText(f"Pontas: {left_end}  …  {right_end}")
