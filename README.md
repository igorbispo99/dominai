# Domino AI

Double-six domino engine with:

- **PyQt6 GUI** for human vs. 1-3 bots
- **DQN agent** (a neural network used as the Q-function) trained through **self-play**
- **Pluggable ML model** (MLP or ResNet — new models can be added via `@register_model`)
- Support for **2-4 players**, **block** and **draw** modes configured via YAML

## Installation

```bash
python -m pip install -e .
# or, to run tests:
python -m pip install -e ".[dev]"
```

Main dependencies: `torch`, `numpy`, `PyQt6`, `PyYAML`.

## Self-Play Training

```bash
python -m domino.cli.train --config config/training.yaml
# resume from a checkpoint:
python -m domino.cli.train --config config/training.yaml --resume checkpoints/run01/latest.pt
```

Every `eval_every` episodes, the seat 0 agent is evaluated against random opponents; the win rate is printed to stdout.

## Play Against the AI

```bash
python -m domino.cli.play --checkpoint checkpoints/run01/latest.pt
```

If `--checkpoint` is omitted, the bots play randomly, which is useful for testing the GUI.

## State Encoding

62-dimensional vector:

| Block | Dim | Description |
|---|---|---|
| Player hand | 28 | binary flag per tile |
| Table - played tiles | 28 | binary flag per tile |
| Table - open ends | 2 | normalized value from 0-6 (`-1` if empty) |
| Opponent counts | 4 | normalized by 7 |

Action space: 57 (28 tiles x 2 ends + `PASS`). Illegal actions are masked both during selection (epsilon-greedy) and during DQN target computation.

## Adding a New Model

```python
from domino.models.registry import register_model
from domino.models.base import Model

@register_model("my_net")
class MyNet(Model):
    def __init__(self, state_dim, action_dim, **kwargs):
        super().__init__(state_dim, action_dim)
        # ...

    def forward(self, x):
        ...

    @classmethod
    def from_config(cls, cfg, state_dim, action_dim):
        return cls(state_dim, action_dim, **cfg.get("kwargs", {}))
```

Then use `name: my_net` in the YAML `model:` block.

## Tests

```bash
python -m pytest
```

## Structure

```text
domino/
  core/       # tiles, rules, state, encoding
  models/     # Model interface + MLP/ResNet + registry
  agents/     # Agent, RandomAgent, DQNAgent
  training/   # ReplayBuffer, Trainer, checkpoint
  gui/        # PyQt6 - window, board, hand, tiles
  cli/        # train / play
config/       # example YAML files
tests/        # pytest
```
