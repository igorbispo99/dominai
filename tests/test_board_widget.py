import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from domino.core.piece import piece_index, Piece
from domino.gui.board_widget import BoardWidget

APP = QApplication.instance() or QApplication([])


def _has_positive_overlap(rect_a, rect_b) -> bool:
    inter = rect_a.intersected(rect_b)
    return inter.width() > 0 and inter.height() > 0


def test_double_on_horizontal_run_renders_vertical() -> None:
    board = BoardWidget()

    board.place(piece_index(Piece(1, 2)), side=0, left_end=1, right_end=2)
    board.place(
        piece_index(Piece(2, 2)),
        side=1,
        left_end=1,
        right_end=2,
        prev_left_end=1,
        prev_right_end=2,
    )

    placements, _, _, _ = board._build_layout()

    assert len(placements) == 2
    _, horizontal, _, rect = placements[1]
    assert not horizontal
    assert rect.height() > rect.width()


def test_double_at_turn_does_not_overlap_previous_piece() -> None:
    board = BoardWidget()

    # Sequence crafted so the 7th placed piece is a double played on a W segment
    # right after an S segment (the corner that previously overlapped).
    seq = [
        Piece(1, 2),
        Piece(2, 2),
        Piece(2, 5),
        Piece(5, 5),
        Piece(5, 6),
        Piece(4, 6),
        Piece(4, 4),
    ]

    first = seq[0]
    board.place(piece_index(first), side=0, left_end=first.low, right_end=first.high)
    left_end, right_end = first.low, first.high

    for piece in seq[1:]:
        if piece.low != right_end and piece.high != right_end:
            continue
        new_right = piece.high if piece.low == right_end else piece.low
        board.place(
            piece_index(piece),
            side=1,
            left_end=left_end,
            right_end=new_right,
            prev_left_end=left_end,
            prev_right_end=right_end,
        )
        right_end = new_right

    placements, _, _, _ = board._build_layout()

    assert len(placements) == 7
    _, double_horizontal, _, _ = placements[6]
    assert double_horizontal

    for i in range(len(placements)):
        for j in range(i + 1, len(placements)):
            assert not _has_positive_overlap(placements[i][3], placements[j][3])


def test_turn_creates_clear_row_spacing() -> None:
    board = BoardWidget()

    seq = [
        Piece(1, 2),
        Piece(2, 3),
        Piece(3, 4),
        Piece(4, 5),
        Piece(5, 6),
        Piece(0, 6),
    ]

    first = seq[0]
    board.place(piece_index(first), side=0, left_end=first.low, right_end=first.high)
    left_end, right_end = first.low, first.high

    for piece in seq[1:]:
        new_right = piece.high if piece.low == right_end else piece.low
        board.place(
            piece_index(piece),
            side=1,
            left_end=left_end,
            right_end=new_right,
            prev_left_end=left_end,
            prev_right_end=right_end,
        )
        right_end = new_right

    placements, _, _, _ = board._build_layout()
    assert len(placements) == 6

    # Piece 5 is the first vertical corner (direction S) after a horizontal run.
    previous_row = placements[4][3]
    corner = placements[5][3]
    assert corner.top() - previous_row.bottom() >= BoardWidget.ROW_GAP