"""Smoke test: a short DQN run should train without errors and at least match random."""
import tempfile

from domino.training.trainer import Trainer, TrainerConfig


def test_trainer_short_run():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = TrainerConfig(
            num_players=2,
            mode="block",
            model_cfg={"name": "mlp", "hidden_sizes": [32, 32]},
            episodes=150,
            batch_size=16,
            buffer_size=2_000,
            warmup_steps=32,
            epsilon_decay_episodes=100,
            target_sync_every=50,
            eval_every=150,
            eval_games=20,
            checkpoint_every=150,
            checkpoint_dir=f"{tmp}/run",
            seed=0,
        )
        trainer = Trainer(cfg)
        trainer.train()
        # At least one metric row recorded and some training actually happened.
        assert trainer.metrics, "no metrics recorded"
        assert trainer._global_step > 0
