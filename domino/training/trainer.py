"""DQN self-play trainer. All seats share a single online Q-network."""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from domino.core.encoding import STATE_DIM, encode_state
from domino.core.game import ACTION_DIM, legal_actions, new_game, step
from domino.models import build_model
from domino.training.checkpoint import save_checkpoint
from domino.training.replay_buffer import ReplayBuffer, Transition


@dataclass
class TrainerConfig:
    # game
    num_players: int = 4
    mode: str = "block"
    team_mode: bool = False
    hand_size: Optional[int] = None
    # model
    model_cfg: Dict[str, Any] = field(
        default_factory=lambda: {"name": "mlp", "hidden_sizes": [256, 256]}
    )
    # training
    episodes: int = 10_000
    batch_size: int = 64
    buffer_size: int = 100_000
    warmup_steps: int = 1_000
    gamma: float = 0.99
    lr: float = 1e-3
    grad_clip: float = 10.0
    target_sync_every: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 5_000
    # logging / checkpointing
    eval_every: int = 500
    eval_games: int = 100
    checkpoint_every: int = 1_000
    checkpoint_dir: str = "./checkpoints/run"
    seed: int = 42
    device: str = "cpu"

    @classmethod
    def from_yaml(cls, yaml_cfg: Dict[str, Any]) -> "TrainerConfig":
        game = yaml_cfg.get("game", {})
        training = yaml_cfg.get("training", {})
        model_cfg = yaml_cfg.get("model", {"name": "mlp", "hidden_sizes": [256, 256]})
        eps = training.get("epsilon", {})
        return cls(
            num_players=game.get("num_players", 4),
            mode=game.get("mode", "block"),
            team_mode=game.get("team_mode", False),
            hand_size=game.get("hand_size"),
            model_cfg=model_cfg,
            episodes=training.get("episodes", 10_000),
            batch_size=training.get("batch_size", 64),
            buffer_size=training.get("buffer_size", 100_000),
            warmup_steps=training.get("warmup_steps", 1_000),
            gamma=training.get("gamma", 0.99),
            lr=training.get("lr", 1e-3),
            grad_clip=training.get("grad_clip", 10.0),
            target_sync_every=training.get("target_sync_every", 1_000),
            epsilon_start=eps.get("start", 1.0),
            epsilon_end=eps.get("end", 0.05),
            epsilon_decay_episodes=eps.get("decay_episodes", 5_000),
            eval_every=training.get("eval_every", 500),
            eval_games=training.get("eval_games", 100),
            checkpoint_every=training.get("checkpoint_every", 1_000),
            checkpoint_dir=training.get("checkpoint_dir", "./checkpoints/run"),
            seed=training.get("seed", 42),
            device=training.get("device", "cpu"),
        )


class Trainer:
    def __init__(self, cfg: TrainerConfig) -> None:
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self._rng = random.Random(cfg.seed)

        self.online = build_model(cfg.model_cfg, STATE_DIM, ACTION_DIM).to(cfg.device)
        self.target = build_model(cfg.model_cfg, STATE_DIM, ACTION_DIM).to(cfg.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optim = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_size, seed=cfg.seed)

        self._global_step = 0
        self.metrics: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ core

    def _epsilon(self, episode: int) -> float:
        frac = min(1.0, episode / max(1, self.cfg.epsilon_decay_episodes))
        return self.cfg.epsilon_start + (self.cfg.epsilon_end - self.cfg.epsilon_start) * frac

    def _select(self, obs: np.ndarray, mask: np.ndarray, epsilon: float) -> int:
        if self._rng.random() < epsilon:
            legal = np.flatnonzero(mask)
            return int(self._rng.choice(legal))
        with torch.no_grad():
            q = self.online(torch.from_numpy(obs).to(self.cfg.device).unsqueeze(0))
            q = q.squeeze(0).cpu().numpy()
        q = np.where(mask, q, -np.inf)
        return int(np.argmax(q))

    def _terminal_reward(self, player: int, winner: Optional[int]) -> float:
        if winner is None:
            return 0.0
        if self.cfg.team_mode:
            return 1.0 if (player % 2) == (winner % 2) else -1.0
        return 1.0 if player == winner else -1.0

    def _play_episode(self, epsilon: float) -> Optional[int]:
        state = new_game(
            self.cfg.num_players,
            self.cfg.mode,
            rng=self._rng,
            hand_size=self.cfg.hand_size,
        )
        pending: Dict[int, Optional[tuple]] = {p: None for p in range(self.cfg.num_players)}
        winner: Optional[int] = None
        safety = 0

        while True:
            mask = legal_actions(state)
            if not mask.any():
                break
            player = state.current_player
            obs = encode_state(state, player=player)
            action = self._select(obs, mask, epsilon)

            # close previous pending transition for this player
            prev = pending[player]
            if prev is not None:
                ps, pa = prev
                self.buffer.add(
                    Transition(
                        state=ps,
                        action=pa,
                        reward=0.0,
                        next_state=obs,
                        next_mask=mask.copy(),
                        done=False,
                    )
                )
            pending[player] = (obs, action)

            state, done, info = step(state, action)
            safety += 1
            if safety > 2000:
                break  # should not happen
            if done:
                winner = info.get("winner")
                for p in range(self.cfg.num_players):
                    if pending[p] is not None:
                        ps, pa = pending[p]
                        r = self._terminal_reward(p, winner)
                        self.buffer.add(
                            Transition(
                                state=ps,
                                action=pa,
                                reward=r,
                                next_state=np.zeros_like(ps),
                                next_mask=np.zeros(ACTION_DIM, dtype=bool),
                                done=True,
                            )
                        )
                break
        return winner

    def _train_step(self) -> float:
        batch = self.buffer.sample(self.cfg.batch_size)
        device = self.cfg.device
        s = torch.from_numpy(batch.states).to(device)
        a = torch.from_numpy(batch.actions).to(device)
        r = torch.from_numpy(batch.rewards).to(device)
        sp = torch.from_numpy(batch.next_states).to(device)
        done = torch.from_numpy(batch.dones).to(device)
        nmask = torch.from_numpy(batch.next_masks).to(device)

        with torch.no_grad():
            q_next = self.target(sp)
            q_next = q_next.masked_fill(~nmask, -1e9)
            max_q, _ = q_next.max(dim=1)
            # when done, the next state is all zeros and mask is all False → ignore bootstrap
            max_q = torch.where(done, torch.zeros_like(max_q), max_q)
            target = r + self.cfg.gamma * max_q

        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q, target)
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.grad_clip)
        self.optim.step()
        return float(loss.item())

    # --------------------------------------------------------------- public

    def train(self, log_fn=None) -> None:
        ckpt_dir = Path(self.cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        wins = 0
        losses: List[float] = []

        for ep in range(1, self.cfg.episodes + 1):
            eps = self._epsilon(ep)
            winner = self._play_episode(eps)
            if winner == 0:
                wins += 1

            if len(self.buffer) >= max(self.cfg.batch_size, self.cfg.warmup_steps):
                loss = self._train_step()
                losses.append(loss)
                self._global_step += 1
                if self._global_step % self.cfg.target_sync_every == 0:
                    self.target.load_state_dict(self.online.state_dict())

            if ep % self.cfg.eval_every == 0:
                avg_loss = float(np.mean(losses[-200:])) if losses else float("nan")
                win_rate_random = self.evaluate_vs_random(self.cfg.eval_games)
                m = {
                    "episode": ep,
                    "epsilon": eps,
                    "buffer": len(self.buffer),
                    "avg_loss": avg_loss,
                    "wr_vs_random": win_rate_random,
                    "elapsed_s": round(time.time() - t0, 1),
                }
                self.metrics.append(m)
                if log_fn is not None:
                    log_fn(m)

            if ep % self.cfg.checkpoint_every == 0:
                save_checkpoint(
                    ckpt_dir / f"ep_{ep:07d}.pt",
                    self.online,
                    self.cfg.model_cfg,
                    extra={"episode": ep, "metrics_tail": self.metrics[-10:]},
                )
                save_checkpoint(
                    ckpt_dir / "latest.pt",
                    self.online,
                    self.cfg.model_cfg,
                    extra={"episode": ep},
                )

    # --------------------------------------------------------------- eval

    def evaluate_vs_random(self, num_games: int = 100) -> float:
        """Play `num_games` as seat 0 (greedy) vs random opponents; returns win rate."""
        from domino.agents import DQNAgent, RandomAgent

        self.online.eval()
        hero = DQNAgent(self.online, epsilon=0.0, device=self.cfg.device, player_id=0)
        opps = [RandomAgent(seed=None) for _ in range(self.cfg.num_players - 1)]

        wins = 0
        for _ in range(num_games):
            state = new_game(self.cfg.num_players, self.cfg.mode, hand_size=self.cfg.hand_size)
            done = False
            info: Dict[str, Any] = {}
            safety = 0
            while not done:
                mask = legal_actions(state)
                if not mask.any():
                    break
                p = state.current_player
                if p == 0:
                    hero.player_id = 0
                    action = hero.select_action(state, mask)
                else:
                    action = opps[p - 1].select_action(state, mask)
                state, done, info = step(state, action)
                safety += 1
                if safety > 2000:
                    break
            winner = info.get("winner")
            if self.cfg.team_mode:
                if winner is not None and winner % 2 == 0:
                    wins += 1
            else:
                if winner == 0:
                    wins += 1
        self.online.train()
        return wins / num_games
