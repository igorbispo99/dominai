# DominAI

Double-six domino project with a PyQt6 graphical interface, support for 2-4 players, and reinforcement learning training via DQN self-play.

## What the project does

- Lets a human play against 1 to 3 bots through the GUI.
- Trains a DQN agent via self-play using YAML configuration.
- Supports interchangeable model architectures through the `domino.models.registry` registry.
- Supports both `block` and `draw` modes.
- Includes auxiliary bots for training diversity: random, heuristic, and variety-based.

## Requirements

- Python 3.10+
- Main dependencies: `torch`, `numpy`, `PyYAML`, `PyQt6`

## Installation

```bash
python -m pip install -e .
```

To install development dependencies:

```bash
python -m pip install -e ".[dev]"
```

## Running the GUI

```bash
python -m domino.cli.play
```

To load a trained checkpoint:

```bash
python -m domino.cli.play --checkpoint checkpoints/run03/latest.pt
```

If `--checkpoint` is omitted, the bots use a random policy. The GUI start screen lets you configure:

- number of players
- match mode (`block` or `draw`)
- human seat
- team mode (`even vs odd seats`)
- optional checkpoint

## Training

Default training run via YAML:

```bash
python -m domino.cli.train --config config/training.yaml
```

Resume from an existing checkpoint:

```bash
python -m domino.cli.train --config config/training.yaml --resume checkpoints/run03/latest.pt
```

The `config/training.yaml` file controls three main sections:

- `game`: number of players, game mode, and `team_mode`
- `model`: Q-function approximator architecture and hyperparameters
- `training`: episodes, buffer, epsilon, checkpoints, evaluation, and opponent mix

The current example configuration uses:

- `num_players: 3`
- `mode: draw`
- `model.name: mlp`
- `checkpoint_dir: ./checkpoints/run03`
- `device: cuda`

If CUDA is not available on the machine, change it to `device: cpu`.

### Opponent mix during training

During training, opponent seats can use different policies according to the probabilities defined in the YAML:

- `opp_prob_random`
- `opp_prob_heuristic`
- `opp_prob_variety`
- `opp_prob_pool`

The remaining probability mass falls back to self-play with the online model.

### Checkpoints and evaluation

- checkpoints are saved according to `checkpoint_every`
- evaluation runs according to `eval_every`
- progress is printed to stdout by the training CLI

## State encoding and action space

The state uses a 62-dimensional vector:

| Block | Dimension | Description |
|---|---:|---|
| Player hand | 28 | binary flag per tile |
| Table - played tiles | 28 | binary flag per tile |
| Open ends | 2 | normalized values from 0 to 6, or `-1` when the table is empty |
| Opponent counts | 4 | counts normalized by 7 |

The action space has 57 positions:

- `28 x 2` possible piece/side plays
- `PASS_ACTION` to pass the turn

Illegal actions are masked both during action selection and during DQN target computation.

## Available models

The project registers models by name and instantiates them from the YAML `model` section.

Currently available models:

- `mlp`
- `resnet`

Custom model example:

```python
from domino.models.base import Model
from domino.models.registry import register_model


@register_model("my_net")
class MyNet(Model):
    def __init__(self, state_dim, action_dim, width=256):
        super().__init__(state_dim, action_dim)
        ...

    def forward(self, x):
        ...

    @classmethod
    def from_config(cls, cfg, state_dim, action_dim):
        return cls(
            state_dim=state_dim,
            action_dim=action_dim,
            width=cfg.get("width", 256),
        )
```

After that, use `name: my_net` in the YAML.

## Project structure

```text
domino/
  agents/     # DQN, random, heuristic, and variety agents
  cli/        # entry points for training and GUI
  core/       # rules, state, pieces, and encoding
  gui/        # main window and PyQt6 widgets
  models/     # base interface, registry, MLP, and residual MLP
  training/   # trainer, replay buffer, and checkpoints
config/       # example YAML configurations
checkpoints/  # saved training weights
tests/        # pytest suite
```

## Tests

```bash
python -m pytest
```

For a quick targeted run, the smoke and rules tests are a good starting point:

```bash
python -m pytest tests/test_trainer_smoke.py tests/test_rules.py
```

## Notes

- The play CLI only accepts `--checkpoint`; the other match options are configured on the GUI start screen.
- Team mode exists in the code and in the GUI, but it makes the most sense in 4-player matches.
- The files in `config/` are intended as starting points and can be adjusted for experiments.
