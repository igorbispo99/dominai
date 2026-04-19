"""Main window, game state machine, and start screen."""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from domino.agents import DQNAgent, RandomAgent
from domino.core.encoding import encode_state
from domino.core.game import (
    ACTION_DIM,
    PASS_ACTION,
    action_to_move,
    legal_actions,
    move_to_action,
    new_game,
    step,
)
from domino.core.piece import DECK
from domino.gui.board_widget import BoardWidget
from domino.gui.hand_widget import HandWidget
from domino.gui.piece_widget import PieceWidget
from domino.training.checkpoint import load_checkpoint


# ---------------------------------------------------------------------------
#  Start screen
# ---------------------------------------------------------------------------

class StartPage(QWidget):
    def __init__(self, on_start) -> None:
        super().__init__()
        self._on_start = on_start
        lay = QVBoxLayout(self)
        lay.addStretch(1)

        title = QLabel("<h1>Dominó AI</h1>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(title)

        form = QVBoxLayout()

        row = QHBoxLayout()
        row.addWidget(QLabel("Jogadores:"))
        self._players = QSpinBox()
        self._players.setRange(2, 4)
        self._players.setValue(4)
        row.addWidget(self._players)
        row.addStretch(1)
        form.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Modo:"))
        self._mode = QComboBox()
        self._mode.addItems(["block", "draw"])
        row.addWidget(self._mode)
        row.addStretch(1)
        form.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Seu assento:"))
        self._seat = QSpinBox()
        self._seat.setRange(0, 3)
        self._seat.setValue(0)
        row.addWidget(self._seat)
        row.addStretch(1)
        form.addLayout(row)

        row = QHBoxLayout()
        self._team = QCheckBox("Modo dupla (2v2, pares vs ímpares)")
        row.addWidget(self._team)
        form.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Checkpoint:"))
        self._ckpt = QLineEdit()
        self._ckpt.setPlaceholderText("opcional — bots aleatórios se vazio")
        row.addWidget(self._ckpt, 1)
        btn_browse = QPushButton("…")
        btn_browse.clicked.connect(self._pick_ckpt)
        row.addWidget(btn_browse)
        form.addLayout(row)

        lay.addLayout(form)
        lay.addSpacing(16)

        start_btn = QPushButton("Começar partida")
        start_btn.setMinimumHeight(40)
        start_btn.clicked.connect(self._start)
        lay.addWidget(start_btn)
        lay.addStretch(2)

    def _pick_ckpt(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Selecionar checkpoint", str(Path.cwd()), "Checkpoints (*.pt)"
        )
        if path:
            self._ckpt.setText(path)

    def _start(self) -> None:
        cfg = {
            "num_players": self._players.value(),
            "mode": self._mode.currentText(),
            "human_seat": self._seat.value(),
            "team_mode": self._team.isChecked(),
            "checkpoint": self._ckpt.text().strip() or None,
        }
        if cfg["human_seat"] >= cfg["num_players"]:
            QMessageBox.warning(self, "Aviso", "Assento do humano >= nº de jogadores.")
            return
        self._on_start(cfg)


# ---------------------------------------------------------------------------
#  Game page
# ---------------------------------------------------------------------------

BOT_DELAY_MS = 600


class GamePage(QWidget):
    def __init__(self, on_new_game) -> None:
        super().__init__()
        self._on_new_game = on_new_game
        self._state = None
        self._human_seat = 0
        self._num_players = 4
        self._bot_agents: dict[int, object] = {}
        self._pending_piece: Optional[int] = None  # awaiting side choice

        root = QHBoxLayout(self)

        # Central column: board + hand + status
        center = QVBoxLayout()
        self._status = QLabel("")
        self._status.setStyleSheet("font-size: 14px; padding: 6px;")
        center.addWidget(self._status)
        self._board = BoardWidget()
        center.addWidget(self._board, 1)

        # side-choice row
        side_row = QHBoxLayout()
        self._btn_left = QPushButton("← Jogar na esquerda")
        self._btn_right = QPushButton("Jogar na direita →")
        self._btn_left.clicked.connect(lambda: self._commit_human_side(0))
        self._btn_right.clicked.connect(lambda: self._commit_human_side(1))
        self._btn_left.setVisible(False)
        self._btn_right.setVisible(False)
        side_row.addStretch(1)
        side_row.addWidget(self._btn_left)
        side_row.addWidget(self._btn_right)
        side_row.addStretch(1)
        center.addLayout(side_row)

        self._hand = HandWidget()
        self._hand.pieceClicked.connect(self._on_hand_piece_clicked)
        center.addWidget(self._hand)

        root.addLayout(center, 3)

        # Right column: opponents + New Game button
        side = QVBoxLayout()
        self._opp_labels: list[QLabel] = []
        self._opp_box = QVBoxLayout()
        side.addLayout(self._opp_box)
        side.addStretch(1)
        btn_new = QPushButton("Novo jogo")
        btn_new.clicked.connect(lambda: self._on_new_game())
        side.addWidget(btn_new)
        root.addLayout(side, 1)

    # ---- lifecycle --------------------------------------------------------

    def start_game(self, cfg: dict, model=None) -> None:
        self._num_players = cfg["num_players"]
        self._human_seat = cfg["human_seat"]
        self._team_mode = cfg.get("team_mode", False)

        # set up bots
        self._bot_agents = {}
        for p in range(self._num_players):
            if p == self._human_seat:
                continue
            if model is not None:
                self._bot_agents[p] = DQNAgent(model, epsilon=0.0, player_id=p)
            else:
                self._bot_agents[p] = RandomAgent()

        # fresh state
        self._state = new_game(self._num_players, cfg["mode"])
        self._board.clear()
        self._refresh_opponents()
        self._refresh_status()
        self._advance_turn()

    # ---- UI refresh -------------------------------------------------------

    def _refresh_opponents(self) -> None:
        # clear box
        while self._opp_box.count():
            item = self._opp_box.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._opp_labels = []
        for p in range(self._num_players):
            if p == self._human_seat:
                continue
            count = len(self._state.hands[p])
            indicator = "▶ " if p == self._state.current_player else "   "
            lbl = QLabel(f"{indicator}Jogador {p}: {count} peça(s)")
            lbl.setStyleSheet("font-size: 13px; padding: 2px;")
            self._opp_box.addWidget(lbl)
            self._opp_labels.append(lbl)

    def _refresh_status(self) -> None:
        if self._state is None:
            return
        if self._state.last_winner is not None or any(
            len(h) == 0 for h in self._state.hands
        ):
            w = self._state.last_winner
            if w == self._human_seat:
                msg = "🎉 Você venceu!"
            elif w is None:
                msg = "Jogo trancado — empate."
            else:
                msg = f"Jogador {w} venceu."
            self._status.setText(msg)
            self._hand.freeze()
            self._btn_left.setVisible(False)
            self._btn_right.setVisible(False)
            return
        cp = self._state.current_player
        if cp == self._human_seat:
            self._status.setText("Sua vez.")
        else:
            self._status.setText(f"Vez do jogador {cp}…")

    # ---- Turn engine ------------------------------------------------------

    def _advance_turn(self) -> None:
        if self._state is None:
            return
        mask = legal_actions(self._state)
        self._refresh_opponents()
        self._refresh_status()

        if not mask.any():
            # terminal — nothing more to do
            self._hand.freeze()
            return

        cp = self._state.current_player
        if cp == self._human_seat:
            legal_pieces = self._extract_legal_pieces(mask)
            self._hand.set_hand(self._state.hands[cp], set(legal_pieces))
            # handle PASS action: no pieces are playable, single PASS option
            if mask[PASS_ACTION] and not legal_pieces:
                self._status.setText("Você não tem jogadas — passando a vez.")
                QTimer.singleShot(BOT_DELAY_MS, lambda: self._apply_action(PASS_ACTION))
        else:
            self._hand.set_hand(self._state.hands[self._human_seat],
                                self._compute_human_legals())
            QTimer.singleShot(BOT_DELAY_MS, self._bot_step)

    def _compute_human_legals(self) -> set:
        """Return which of the human's pieces would be legal IF it were their turn.
        (Used to paint the hand whilst bots play; purely informational.)"""
        st = self._state
        if st is None:
            return set()
        human_hand = st.hands[self._human_seat]
        if st.is_empty_table:
            return set(human_hand)
        legal = set()
        for pi in human_hand:
            p = DECK[pi]
            if st.left_end is not None and (p.matches(st.left_end) or p.matches(st.right_end)):
                legal.add(pi)
        return legal

    def _extract_legal_pieces(self, mask: np.ndarray) -> list[int]:
        return sorted({a // 2 for a in np.flatnonzero(mask) if a != PASS_ACTION})

    def _bot_step(self) -> None:
        if self._state is None:
            return
        mask = legal_actions(self._state)
        if not mask.any():
            return
        cp = self._state.current_player
        if cp == self._human_seat:
            self._advance_turn()
            return
        agent = self._bot_agents[cp]
        if isinstance(agent, DQNAgent):
            agent.player_id = cp
        action = agent.select_action(self._state, mask)
        self._apply_action(action)

    # ---- Human click handlers --------------------------------------------

    def _on_hand_piece_clicked(self, piece_idx: int) -> None:
        if self._state is None:
            return
        if self._state.current_player != self._human_seat:
            return
        mask = legal_actions(self._state)
        left_ok = bool(mask[move_to_action(piece_idx, 0)])
        right_ok = bool(mask[move_to_action(piece_idx, 1)])
        if not (left_ok or right_ok):
            return
        if left_ok and right_ok:
            self._pending_piece = piece_idx
            self._btn_left.setVisible(True)
            self._btn_right.setVisible(True)
            self._status.setText("Escolha em qual ponta jogar:")
        else:
            side = 0 if left_ok else 1
            self._apply_action(move_to_action(piece_idx, side))

    def _commit_human_side(self, side: int) -> None:
        if self._pending_piece is None:
            return
        action = move_to_action(self._pending_piece, side)
        self._pending_piece = None
        self._btn_left.setVisible(False)
        self._btn_right.setVisible(False)
        self._apply_action(action)

    # ---- Apply action -----------------------------------------------------

    def _apply_action(self, action: int) -> None:
        if self._state is None:
            return
        prev_state = self._state
        if action != PASS_ACTION:
            piece_idx, side = action_to_move(action)
        else:
            piece_idx, side = -1, -1
        next_state, done, _info = step(self._state, action)
        self._state = next_state

        if action != PASS_ACTION:
            self._board.place(
                piece_idx,
                side,
                next_state.left_end if next_state.left_end is not None else -1,
                next_state.right_end if next_state.right_end is not None else -1,
            )
        if done:
            self._refresh_opponents()
            self._refresh_status()
            return
        self._advance_turn()


# ---------------------------------------------------------------------------
#  Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self, checkpoint_path: Optional[str] = None) -> None:
        super().__init__()
        self.setWindowTitle("Dominó AI")
        self.resize(1000, 620)
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)
        self._start_page = StartPage(self._handle_start)
        self._game_page = GamePage(self._go_home)
        self._stack.addWidget(self._start_page)
        self._stack.addWidget(self._game_page)
        self._model = None

        if checkpoint_path:
            try:
                self._model, _ = load_checkpoint(checkpoint_path)
                self._start_page._ckpt.setText(checkpoint_path)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(
                    self, "Checkpoint", f"Falha ao carregar: {exc}"
                )

    def _handle_start(self, cfg: dict) -> None:
        model = None
        ckpt = cfg.get("checkpoint")
        if ckpt:
            try:
                model, _ = load_checkpoint(ckpt)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(self, "Checkpoint", f"Falha: {exc}")
                return
        elif self._model is not None:
            model = self._model
        self._game_page.start_game(cfg, model=model)
        self._stack.setCurrentIndex(1)

    def _go_home(self) -> None:
        self._stack.setCurrentIndex(0)
