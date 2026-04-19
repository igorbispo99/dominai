"""python -m domino.cli.play [--checkpoint path.pt]"""
from __future__ import annotations

import argparse
import sys

from PyQt6.QtWidgets import QApplication

from domino.gui.main_window import MainWindow


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Launch the Domino AI GUI")
    ap.add_argument("--checkpoint", default=None, help="optional DQN checkpoint .pt")
    args = ap.parse_args(argv)

    app = QApplication(sys.argv)
    win = MainWindow(checkpoint_path=args.checkpoint)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
