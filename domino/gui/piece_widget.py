"""QWidget that paints a single domino piece (orientations: horizontal/vertical)."""
from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from domino.core.piece import DECK, Piece


# Pip positions inside a unit square (x,y in [0,1])
_PIP_LAYOUT = {
    0: [],
    1: [(0.5, 0.5)],
    2: [(0.25, 0.25), (0.75, 0.75)],
    3: [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)],
    4: [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)],
    5: [(0.25, 0.25), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.75, 0.75)],
    6: [(0.25, 0.25), (0.75, 0.25), (0.25, 0.5), (0.75, 0.5), (0.25, 0.75), (0.75, 0.75)],
}


HALF_SIZE = 46  # pixels per half of a piece


class PieceWidget(QWidget):
    """Displays one piece. Emits ``clicked(piece_idx)`` on left-click."""

    clicked = pyqtSignal(int)

    def __init__(
        self,
        piece_idx: int,
        horizontal: bool = True,
        face_down: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.piece_idx = piece_idx
        self.horizontal = horizontal
        self.face_down = face_down
        self._highlighted = False
        self._disabled = False
        self.setFixedSize(self._ideal_size())

    def _ideal_size(self) -> QSize:
        if self.horizontal:
            return QSize(HALF_SIZE * 2, HALF_SIZE)
        return QSize(HALF_SIZE, HALF_SIZE * 2)

    def set_orientation(self, horizontal: bool) -> None:
        if horizontal != self.horizontal:
            self.horizontal = horizontal
            self.setFixedSize(self._ideal_size())
            self.update()

    def set_highlighted(self, hl: bool) -> None:
        self._highlighted = hl
        self.update()

    def set_disabled(self, disabled: bool) -> None:
        self._disabled = disabled
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and not self._disabled:
            self.clicked.emit(self.piece_idx)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)

        # body
        body_color = QColor("#f8f6ef")
        if self.face_down:
            body_color = QColor("#3a4a5a")
        if self._disabled and not self.face_down:
            body_color = QColor("#d9d4c6")
        border = QColor("#d4af37") if self._highlighted else QColor("#222")
        pen = QPen(border, 3 if self._highlighted else 2)
        painter.setPen(pen)
        painter.setBrush(QBrush(body_color))
        painter.drawRoundedRect(rect, 8, 8)

        if self.face_down:
            painter.end()
            return

        # divider
        mid_pen = QPen(QColor("#222"), 2)
        painter.setPen(mid_pen)
        if self.horizontal:
            painter.drawLine(
                QPointF(self.width() / 2, 4), QPointF(self.width() / 2, self.height() - 4)
            )
            half_a = QRectF(0, 0, self.width() / 2, self.height())
            half_b = QRectF(self.width() / 2, 0, self.width() / 2, self.height())
        else:
            painter.drawLine(
                QPointF(4, self.height() / 2), QPointF(self.width() - 4, self.height() / 2)
            )
            half_a = QRectF(0, 0, self.width(), self.height() / 2)
            half_b = QRectF(0, self.height() / 2, self.width(), self.height() / 2)

        piece: Piece = DECK[self.piece_idx]
        # convention: low pip half appears first (left for horizontal, top for vertical)
        self._draw_half(painter, half_a, piece.low)
        self._draw_half(painter, half_b, piece.high)
        painter.end()

    @staticmethod
    def _draw_half(painter: QPainter, rect: QRectF, pips: int) -> None:
        pip_color = QColor("#111")
        painter.setBrush(QBrush(pip_color))
        painter.setPen(Qt.PenStyle.NoPen)
        r = min(rect.width(), rect.height()) * 0.09
        for fx, fy in _PIP_LAYOUT[pips]:
            cx = rect.x() + fx * rect.width()
            cy = rect.y() + fy * rect.height()
            painter.drawEllipse(QPointF(cx, cy), r, r)
