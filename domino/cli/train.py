"""python -m domino.cli.train --config config/training.yaml [--resume <ckpt>]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

from domino.training.checkpoint import load_checkpoint
from domino.training.trainer import Trainer, TrainerConfig


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _log(metrics: dict) -> None:
    parts = [f"{k}={v}" for k, v in metrics.items()]
    print("[train] " + "  ".join(parts), flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DQN self-play training for Domino AI")
    ap.add_argument("--config", required=True, help="path to training YAML")
    ap.add_argument("--resume", default=None, help="path to checkpoint .pt to resume from")
    args = ap.parse_args(argv)

    cfg_dict = _load_yaml(args.config)
    cfg = TrainerConfig.from_yaml(cfg_dict)
    trainer = Trainer(cfg)

    if args.resume:
        print(f"[train] resuming weights from {args.resume}")
        resumed_model, _extra = load_checkpoint(args.resume, map_location=cfg.device)
        trainer.online.load_state_dict(resumed_model.state_dict())
        trainer.target.load_state_dict(resumed_model.state_dict())

    print(
        f"[train] episodes={cfg.episodes} num_players={cfg.num_players} "
        f"mode={cfg.mode} model={cfg.model_cfg.get('name')} device={cfg.device}"
    )
    trainer.train(log_fn=_log)
    print("[train] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
