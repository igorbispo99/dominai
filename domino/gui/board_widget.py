"""The table with a connected snake layout and drop targets attached to the true ends."""
from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget

from domino.core.piece import DECK, Piece
from domino.gui.piece_widget import HALF_SIZE, PieceWidget


@dataclass(frozen=True)
class _PlacedPiece:
    piece_idx: int
    direction: str
    flipped: bool


class _EndDropZone(QLabel):
    dropped = pyqtSignal(int, int)  # piece_idx, side

    def __init__(self, side: int, parent=None) -> None:
        super().__init__(parent)
        self.side = side
        self._active = True
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedSize(30, 30)
        self._apply_idle_style()

    def set_active(self, active: bool) -> None:
        self._active = active
        self.setAcceptDrops(active)
        if active:
            self._apply_idle_style()
            self.show()
        else:
            self.hide()

    def dragEnterEvent(self, event) -> None:
        if not self._active:
            event.ignore()
            return
        piece_idx = BoardWidget.parse_piece_idx(event.mimeData().text())
        if piece_idx is None:
            event.ignore()
            return
        self.setStyleSheet(
            "background: #d7f1dc; border: 2px solid #2f7d32; border-radius: 15px;"
        )
        event.acceptProposedAction()

    def dragMoveEvent(self, event) -> None:
        if not self._active:
            event.ignore()
            return
        piece_idx = BoardWidget.parse_piece_idx(event.mimeData().text())
        if piece_idx is None:
            event.ignore()
            return
        event.acceptProposedAction()

    def dragLeaveEvent(self, event) -> None:
        self._apply_idle_style()
        event.accept()

    def dropEvent(self, event) -> None:
        if not self._active:
            event.ignore()
            return
        piece_idx = BoardWidget.parse_piece_idx(event.mimeData().text())
        if piece_idx is None:
            event.ignore()
            return
        self._apply_idle_style()
        self.dropped.emit(piece_idx, self.side)
        event.acceptProposedAction()

    def _apply_idle_style(self) -> None:
        self.setStyleSheet(
            "background: #f3faf4; border: 2px dashed #4f7f52; border-radius: 15px;"
        )


class BoardWidget(QWidget):
    pieceDropped = pyqtSignal(int, int)  # piece_idx, side(0=left, 1=right)

    H_PIECE_W = HALF_SIZE * 2
    H_PIECE_H = HALF_SIZE
    V_PIECE_W = HALF_SIZE
    V_PIECE_H = HALF_SIZE * 2
    DROP_SIZE = 30
    BRANCH_RUN = 4
    PIECE_GAP = 2
    ROW_GAP = 18
    MARGIN = 80

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self._drop_enabled = True
        self._center_piece: tuple[int, bool] | None = None
        self._left_entries: list[_PlacedPiece] = []
        self._right_entries: list[_PlacedPiece] = []
        self._piece_widgets: list[PieceWidget] = []

        self._ends_label = QLabel("Mesa vazia")
        self._ends_label.setStyleSheet("font-weight: bold; padding: 4px;")
        outer.addWidget(self._ends_label)

        self._hint_label = QLabel("Arraste uma peça até a ponta destacada na mesa.")
        self._hint_label.setStyleSheet("color: #555; padding-left: 4px;")
        outer.addWidget(self._hint_label)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self._canvas = QWidget()
        self._canvas.setMinimumSize(900, 360)
        self._canvas.setStyleSheet("background: #2f5d31; border-radius: 10px;")

        self._left_drop = _EndDropZone(side=0, parent=self._canvas)
        self._right_drop = _EndDropZone(side=1, parent=self._canvas)
        self._left_drop.dropped.connect(self.pieceDropped.emit)
        self._right_drop.dropped.connect(self.pieceDropped.emit)

        self._scroll.setWidget(self._canvas)
        outer.addWidget(self._scroll, 1)

        self._render_board()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render_board()

    def clear(self) -> None:
        self._center_piece = None
        self._left_entries = []
        self._right_entries = []
        self._ends_label.setText("Mesa vazia")
        self._hint_label.setText("Arraste uma peça até a ponta destacada na mesa.")
        self._render_board()

    def set_drop_enabled(self, enabled: bool) -> None:
        self._drop_enabled = enabled
        self._left_drop.set_active(enabled)
        self._right_drop.set_active(enabled)
        if enabled:
            self._hint_label.setText("Arraste uma peça até a ponta destacada na mesa.")
        else:
            self._hint_label.setText("")

    def place(
        self,
        piece_idx: int,
        side: int,
        left_end: int,
        right_end: int,
        prev_left_end: int | None = None,
        prev_right_end: int | None = None,
    ) -> None:
        if self._center_piece is None:
            self._center_piece = (piece_idx, False)
        else:
            direction = self._next_direction(side)
            match_value = prev_left_end if side == 0 else prev_right_end
            flipped = self._compute_flipped(DECK[piece_idx], direction, match_value)
            entry = _PlacedPiece(piece_idx=piece_idx, direction=direction, flipped=flipped)
            if side == 0:
                self._left_entries.append(entry)
            else:
                self._right_entries.append(entry)

        self._ends_label.setText(f"Pontas: {left_end}  …  {right_end}")
        if self._drop_enabled:
            self._hint_label.setText("Arraste uma peça até a ponta destacada na mesa.")
        self._render_board()

    def set_ends(self, left_end, right_end) -> None:
        if left_end is None:
            self._ends_label.setText("Mesa vazia")
        else:
            self._ends_label.setText(f"Pontas: {left_end}  …  {right_end}")

    def _render_board(self) -> None:
        for widget in self._piece_widgets:
            widget.deleteLater()
        self._piece_widgets = []

        placements, left_end_point, right_end_point, bounds = self._build_layout()
        viewport = self._scroll.viewport()
        viewport_width = viewport.width() if viewport is not None else 0
        canvas_width = max(viewport_width, bounds.width() + (self.MARGIN * 2))
        canvas_height = max(320, bounds.height() + (self.MARGIN * 2))
        self._canvas.resize(canvas_width, canvas_height)

        center_pad = max(0, (canvas_width - bounds.width() - (self.MARGIN * 2)) // 2)
        shift_x = self.MARGIN - bounds.left() + center_pad
        shift_y = self.MARGIN - bounds.top()

        for piece_idx, horizontal, flipped, rect in placements:
            widget = PieceWidget(piece_idx, horizontal=horizontal, flipped=flipped, parent=self._canvas)
            widget.set_disabled(True)
            widget.move(rect.x() + shift_x, rect.y() + shift_y)
            widget.show()
            self._piece_widgets.append(widget)

        self._position_drop_zone(self._left_drop, left_end_point, shift_x, shift_y)
        self._position_drop_zone(self._right_drop, right_end_point, shift_x, shift_y)
        self._left_drop.raise_()
        self._right_drop.raise_()
        self._left_drop.set_active(self._drop_enabled)
        self._right_drop.set_active(self._drop_enabled)

    def _build_layout(self) -> tuple[list[tuple[int, bool, bool, QRect]], QPoint, QPoint, QRect]:
        placements: list[tuple[int, bool, bool, QRect]] = []

        center_rect = QRect(0, 0, self.H_PIECE_W, self.H_PIECE_H)
        left_end = QPoint(center_rect.left(), center_rect.center().y())
        right_end = QPoint(center_rect.right() + 1, center_rect.center().y())
        bounds = QRect(center_rect)

        if self._center_piece is not None:
            center_idx, center_flipped = self._center_piece
            placements.append((center_idx, True, center_flipped, QRect(center_rect)))

        current_left_end = QPoint(left_end)
        prev_left_direction: str | None = None
        for entry in self._left_entries:
            rect, current_left_end, horizontal = self._place_from_endpoint(
                entry.piece_idx,
                current_left_end,
                entry.direction,
                prev_left_direction,
            )
            placements.append((entry.piece_idx, horizontal, entry.flipped, rect))
            bounds = bounds.united(rect)
            prev_left_direction = entry.direction

        current_right_end = QPoint(right_end)
        prev_right_direction: str | None = None
        for entry in self._right_entries:
            rect, current_right_end, horizontal = self._place_from_endpoint(
                entry.piece_idx,
                current_right_end,
                entry.direction,
                prev_right_direction,
            )
            placements.append((entry.piece_idx, horizontal, entry.flipped, rect))
            bounds = bounds.united(rect)
            prev_right_direction = entry.direction

        return placements, current_left_end, current_right_end, bounds

    def _next_direction(self, side: int) -> str:
        branch = self._left_entries if side == 0 else self._right_entries
        idx = len(branch)
        segment = 0
        consumed = idx
        while True:
            run_len = self.BRANCH_RUN if segment % 2 == 0 else 1
            if consumed < run_len:
                break
            consumed -= run_len
            segment += 1

        if segment % 2 == 1:
            return "S"

        if side == 1:
            return "E" if (segment // 2) % 2 == 0 else "W"
        return "W" if (segment // 2) % 2 == 0 else "E"

    def _place_from_endpoint(
        self,
        piece_idx: int,
        endpoint: QPoint,
        direction: str,
        prev_direction: str | None,
    ) -> tuple[QRect, QPoint, bool]:
        piece = DECK[piece_idx]
        if piece.is_double():
            # At turn points (S -> E/W), keep the first horizontal tile horizontal
            # and shift it down to avoid overlapping the vertical corner tile.
            if direction in {"E", "W"} and prev_direction == "S":
                row_y = endpoint.y() + (self.H_PIECE_H // 2) + self.ROW_GAP
                if direction == "E":
                    rect = QRect(
                        endpoint.x() + self.PIECE_GAP,
                        row_y,
                        self.H_PIECE_W,
                        self.H_PIECE_H,
                    )
                    next_end = QPoint(rect.right() + 1, rect.center().y())
                    return rect, next_end, True
                rect = QRect(
                    endpoint.x() - self.H_PIECE_W - self.PIECE_GAP,
                    row_y,
                    self.H_PIECE_W,
                    self.H_PIECE_H,
                )
                next_end = QPoint(rect.left(), rect.center().y())
                return rect, next_end, True

            if direction == "E":
                rect = QRect(
                    endpoint.x() + self.PIECE_GAP,
                    endpoint.y() - (self.V_PIECE_H // 2),
                    self.V_PIECE_W,
                    self.V_PIECE_H,
                )
                next_end = QPoint(rect.right() + 1, endpoint.y())
                return rect, next_end, False
            if direction == "W":
                rect = QRect(
                    endpoint.x() - self.V_PIECE_W - self.PIECE_GAP,
                    endpoint.y() - (self.V_PIECE_H // 2),
                    self.V_PIECE_W,
                    self.V_PIECE_H,
                )
                next_end = QPoint(rect.left(), endpoint.y())
                return rect, next_end, False
            top_y = endpoint.y() + (self.H_PIECE_H // 2) + self.ROW_GAP
            if prev_direction == "E":
                x = endpoint.x() + self.PIECE_GAP
            elif prev_direction == "W":
                x = endpoint.x() - self.H_PIECE_W - self.PIECE_GAP
            else:
                x = endpoint.x() - (self.H_PIECE_W // 2)
            rect = QRect(
                x,
                top_y,
                self.H_PIECE_W,
                self.H_PIECE_H,
            )
            next_end = QPoint(endpoint.x(), rect.bottom() + 1)
            return rect, next_end, True

        if direction == "E":
            if prev_direction == "S":
                row_y = endpoint.y() + (self.H_PIECE_H // 2) + self.ROW_GAP
                rect = QRect(endpoint.x() + self.PIECE_GAP, row_y, self.H_PIECE_W, self.H_PIECE_H)
                next_end = QPoint(rect.right() + 1, rect.center().y())
                return rect, next_end, True
            rect = QRect(
                endpoint.x() + self.PIECE_GAP,
                endpoint.y() - (self.H_PIECE_H // 2),
                self.H_PIECE_W,
                self.H_PIECE_H,
            )
            next_end = QPoint(rect.right() + 1, endpoint.y())
            return rect, next_end, True
        if direction == "W":
            if prev_direction == "S":
                row_y = endpoint.y() + (self.H_PIECE_H // 2) + self.ROW_GAP
                rect = QRect(
                    endpoint.x() - self.H_PIECE_W - self.PIECE_GAP,
                    row_y,
                    self.H_PIECE_W,
                    self.H_PIECE_H,
                )
                next_end = QPoint(rect.left(), rect.center().y())
                return rect, next_end, True
            rect = QRect(
                endpoint.x() - self.H_PIECE_W - self.PIECE_GAP,
                endpoint.y() - (self.H_PIECE_H // 2),
                self.H_PIECE_W,
                self.H_PIECE_H,
            )
            next_end = QPoint(rect.left(), endpoint.y())
            return rect, next_end, True

        top_y = endpoint.y() + (self.H_PIECE_H // 2) + self.ROW_GAP
        if prev_direction == "E":
            x = endpoint.x() + self.PIECE_GAP
        elif prev_direction == "W":
            x = endpoint.x() - self.V_PIECE_W - self.PIECE_GAP
        else:
            x = endpoint.x() - (self.V_PIECE_W // 2)
        rect = QRect(x, top_y, self.V_PIECE_W, self.V_PIECE_H)
        next_end = QPoint(endpoint.x(), rect.bottom() + 1)
        return rect, next_end, False

    def _position_drop_zone(self, zone: _EndDropZone, point: QPoint, shift_x: int, shift_y: int) -> None:
        zone.move(
            point.x() + shift_x - (self.DROP_SIZE // 2),
            point.y() + shift_y - (self.DROP_SIZE // 2),
        )
        if self._drop_enabled:
            zone.show()

    @staticmethod
    def _compute_flipped(piece: Piece, direction: str, match_value: int | None) -> bool:
        if match_value is None:
            return False
        if direction == "E":
            return piece.high == match_value
        if direction == "W":
            return piece.low == match_value
        return piece.high == match_value

    @staticmethod
    def parse_piece_idx(text: str) -> int | None:
        prefix = "domino-piece:"
        if not text.startswith(prefix):
            return None
        raw = text[len(prefix):]
        if not raw.isdigit():
            return None
        return int(raw)
