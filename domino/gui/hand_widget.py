"""The player's hand — a horizontal strip of clickable PieceWidgets."""
from __future__ import annotations

from typing import Iterable, Set

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QWidget

from domino.gui.piece_widget import PieceWidget


class HandWidget(QWidget):
    pieceClicked = pyqtSignal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setSpacing(8)
        self._layout.addStretch(1)
        self._widgets: dict[int, PieceWidget] = {}

    def set_hand(self, piece_idxs: Iterable[int], legal_idxs: Set[int]) -> None:
        # remove old widgets
        while self._layout.count() > 1:
            item = self._layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._widgets.clear()
        for pi in piece_idxs:
            w = PieceWidget(pi, horizontal=True)
            w.clicked.connect(self.pieceClicked.emit)
            is_legal = pi in legal_idxs
            w.set_highlighted(is_legal)
            w.set_disabled(not is_legal)
            self._widgets[pi] = w
            self._layout.insertWidget(self._layout.count() - 1, w)

    def freeze(self) -> None:
        """Disable all clicks (bot's turn / game over)."""
        for w in self._widgets.values():
            w.set_disabled(True)
            w.set_highlighted(False)
