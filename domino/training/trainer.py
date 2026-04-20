"""DQN trainer — full self-play, Double DQN, shaped rewards."""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from domino.core.encoding import STATE_DIM, encode_state
from domino.core.game import (
    ACTION_DIM,
    PASS_ACTION,
    action_to_move,
    legal_actions,
    new_game,
    step,
)
from domino.core.piece import DECK
from domino.models import build_model
from domino.training.checkpoint import save_checkpoint
from domino.training.replay_buffer import ReplayBuffer, Transition
from domino.agents import HeuristicAgent, RandomAgent


# ---------------------------------------------------------------- reward shaping

_PLAY_REWARD_SCALE = 1.0 / 168.0   # max total per player ≈ 0.5 (7 pieces, mean pip 12)
_PASS_PENALTY      = -0.05
_TERMINAL_WIN      = 1.0
_TERMINAL_LOSS     = -1.0


# ---------------------------------------------------------------- opponent mix

_OPP_SELF     = "self"
_OPP_RANDOM   = "random"
_OPP_HEURISTIC = "heuristic"
_OPP_POOL     = "pool"


def _shaped_reward_for_action(prev_state, action: int) -> float:
    """Immediate reward for the actor that just took ``action`` in ``prev_state``."""
    if action == PASS_ACTION:
        return _PASS_PENALTY
    piece_idx, _ = action_to_move(action)
    return DECK[piece_idx].pip_count() * _PLAY_REWARD_SCALE


# ---------------------------------------------------------------- config

@dataclass
class TrainerConfig:
    # game
    num_players: int = 4
    mode: str = "block"
    team_mode: bool = False
    hand_size: Optional[int] = None
    # model
    model_cfg: Dict[str, Any] = field(
        default_factory=lambda: {"name": "mlp", "hidden_sizes": [512, 512, 256]}
    )
    # training
    episodes: int = 10_000
    batch_size: int = 128
    buffer_size: int = 200_000
    warmup_steps: int = 5_000
    gamma: float = 0.99
    lr: float = 3e-4
    grad_clip: float = 5.0
    target_sync_every: int = 2_000
    train_steps_per_episode: int = 2
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 30_000
    reward_shaping: bool = True
    # opponent pool (kept for eval / diversity, optional during play)
    pool_add_every: int = 5_000
    pool_max_size: int = 8
    # Per-episode probability of each opponent type for seats 1..n-1.
    # Values should sum ≤ 1.0; the remainder goes to self-play.
    opp_prob_random: float = 0.2
    opp_prob_heuristic: float = 0.2
    opp_prob_pool: float = 0.0
    # logging / checkpointing
    eval_every: int = 1_000
    eval_games: int = 100
    checkpoint_every: int = 5_000
    checkpoint_dir: str = "./checkpoints/run"
    seed: int = 42
    device: str = "cpu"

    @classmethod
    def from_yaml(cls, yaml_cfg: Dict[str, Any]) -> "TrainerConfig":
        game = yaml_cfg.get("game", {})
        training = yaml_cfg.get("training", {})
        model_cfg = yaml_cfg.get("model", {"name": "mlp", "hidden_sizes": [512, 512, 256]})
        eps = training.get("epsilon", {})
        pool = training.get("opponent_pool", {})
        return cls(
            num_players=game.get("num_players", 4),
            mode=game.get("mode", "block"),
            team_mode=game.get("team_mode", False),
            hand_size=game.get("hand_size"),
            model_cfg=model_cfg,
            episodes=training.get("episodes", 10_000),
            batch_size=training.get("batch_size", 128),
            buffer_size=training.get("buffer_size", 200_000),
            warmup_steps=training.get("warmup_steps", 5_000),
            gamma=training.get("gamma", 0.99),
            lr=training.get("lr", 3e-4),
            grad_clip=training.get("grad_clip", 5.0),
            target_sync_every=training.get("target_sync_every", 2_000),
            train_steps_per_episode=training.get("train_steps_per_episode", 2),
            epsilon_start=eps.get("start", 1.0),
            epsilon_end=eps.get("end", 0.05),
            epsilon_decay_episodes=eps.get("decay_episodes", 30_000),
            reward_shaping=training.get("reward_shaping", True),
            pool_add_every=pool.get("add_every", 5_000),
            pool_max_size=pool.get("max_size", 8),
            opp_prob_random=training.get("opp_prob_random", 0.2),
            opp_prob_heuristic=training.get("opp_prob_heuristic", 0.2),
            opp_prob_pool=training.get("opp_prob_pool", 0.0),
            eval_every=training.get("eval_every", 1_000),
            eval_games=training.get("eval_games", 100),
            checkpoint_every=training.get("checkpoint_every", 5_000),
            checkpoint_dir=training.get("checkpoint_dir", "./checkpoints/run"),
            seed=training.get("seed", 42),
            device=training.get("device", "cpu"),
        )


# ---------------------------------------------------------------- trainer

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

        # opponent pool — only consulted when opp_prob_pool > 0
        self._opponent_pool: List[Dict[str, Any]] = []
        self._opp_model = build_model(cfg.model_cfg, STATE_DIM, ACTION_DIM).to(cfg.device)
        self._opp_model.eval()

        # stateless helper agents (re-seeded per episode implicitly via their own rng)
        self._random_agent = RandomAgent(seed=None)
        self._heuristic_agent = HeuristicAgent(seed=None)

        # per-episode opponent selection: one of _OPP_*
        self._episode_opponent: str = _OPP_SELF

        self._global_step = 0
        self.metrics: List[Dict[str, Any]] = []

    # ---------------------------------------------------------------- schedule

    def _epsilon(self, episode: int) -> float:
        frac = min(1.0, episode / max(1, self.cfg.epsilon_decay_episodes))
        return self.cfg.epsilon_start + (self.cfg.epsilon_end - self.cfg.epsilon_start) * frac

    # ---------------------------------------------------------------- policy

    def _select_online(self, obs: np.ndarray, mask: np.ndarray, epsilon: float) -> int:
        if self._rng.random() < epsilon:
            legal = np.flatnonzero(mask)
            return int(self._rng.choice(legal))
        with torch.no_grad():
            q = self.online(torch.from_numpy(obs).to(self.cfg.device).unsqueeze(0))
            q = q.squeeze(0).cpu().numpy()
        q = np.where(mask, q, -np.inf)
        return int(np.argmax(q))

    def _select_pool(self, obs: np.ndarray, mask: np.ndarray) -> int:
        with torch.no_grad():
            q = self._opp_model(torch.from_numpy(obs).to(self.cfg.device).unsqueeze(0))
            q = q.squeeze(0).cpu().numpy()
        q = np.where(mask, q, -np.inf)
        return int(np.argmax(q))

    def _opponent_action(self, state, obs: np.ndarray, mask: np.ndarray) -> int:
        kind = self._episode_opponent
        if kind == _OPP_RANDOM:
            return self._random_agent.select_action(state, mask)
        if kind == _OPP_HEURISTIC:
            return self._heuristic_agent.select_action(state, mask)
        if kind == _OPP_POOL:
            return self._select_pool(obs, mask)
        # self-play fallback: sampled elsewhere
        return self._select_online(obs, mask, epsilon=0.0)

    def _roll_opponent(self) -> str:
        r = self._rng.random()
        cum = 0.0
        if self._opponent_pool and self.cfg.opp_prob_pool > 0:
            cum += self.cfg.opp_prob_pool
            if r < cum:
                return _OPP_POOL
        cum += self.cfg.opp_prob_random
        if r < cum:
            return _OPP_RANDOM
        cum += self.cfg.opp_prob_heuristic
        if r < cum:
            return _OPP_HEURISTIC
        return _OPP_SELF

    # ---------------------------------------------------------------- pool

    def _pool_snapshot(self) -> None:
        sd = {k: v.detach().cpu().clone() for k, v in self.online.state_dict().items()}
        self._opponent_pool.append(sd)
        if len(self._opponent_pool) > self.cfg.pool_max_size:
            self._opponent_pool.pop(0)

    def _refresh_opponent(self) -> None:
        if self._opponent_pool:
            sd = self._rng.choice(self._opponent_pool)
            self._opp_model.load_state_dict(
                {k: v.to(self.cfg.device) for k, v in sd.items()}
            )

    # ---------------------------------------------------------------- reward

    def _terminal_reward(self, player: int, winner: Optional[int]) -> float:
        if winner is None:
            return 0.0
        if self.cfg.team_mode:
            return _TERMINAL_WIN if (player % 2) == (winner % 2) else _TERMINAL_LOSS
        return _TERMINAL_WIN if player == winner else _TERMINAL_LOSS

    # ---------------------------------------------------------------- episode

    def _play_episode(self, epsilon: float) -> Optional[int]:
        """Play one episode with full self-play.

        Every seat selects actions from the online model (ε-greedy) and stores
        its own transitions. Between a player's turns the "reward" is the
        shaped reward they earned from their own action, plus any terminal
        reward when the episode ends before their next turn.
        """
        cfg = self.cfg
        num_players = cfg.num_players
        state = new_game(num_players, cfg.mode, rng=self._rng, hand_size=cfg.hand_size)

        # only seat 0 is trained when opponents aren't self-play
        is_self_play = self._episode_opponent == _OPP_SELF

        # pending transition per seat: (obs, action, accumulated_reward)
        pending: List[Optional[Tuple[np.ndarray, int, float]]] = [None] * num_players

        winner: Optional[int] = None
        zero_s = np.zeros(STATE_DIM, dtype=np.float32)
        zero_m = np.zeros(ACTION_DIM, dtype=bool)
        safety = 0

        while True:
            mask = legal_actions(state)
            if not mask.any():
                break
            player = state.current_player
            obs = encode_state(state, player=player)

            # close pending transition for this seat (bootstrap on current obs)
            if pending[player] is not None:
                p_obs, p_action, p_reward = pending[player]
                self.buffer.add(
                    Transition(
                        state=p_obs,
                        action=p_action,
                        reward=p_reward,
                        next_state=obs,
                        next_mask=mask.copy(),
                        done=False,
                    )
                )
                pending[player] = None

            # choose action — seat 0 always uses online (ε-greedy);
            # other seats use the configured opponent type this episode
            if player == 0 or is_self_play:
                action = self._select_online(obs, mask, epsilon)
            else:
                action = self._opponent_action(state, obs, mask)

            # reward from this action (actor only)
            shaped = _shaped_reward_for_action(state, action) if cfg.reward_shaping else 0.0

            next_state, done, info = step(state, action)

            # only record pending transitions for seats we are training
            trainable = is_self_play or (player == 0)
            if trainable:
                pending[player] = (obs, action, shaped)

            state = next_state
            safety += 1
            if safety > 2000:
                break
            if done:
                winner = info.get("winner")
                # close every pending transition with the terminal reward
                for p in range(num_players):
                    if pending[p] is None:
                        continue
                    p_obs, p_action, p_reward = pending[p]
                    total_r = p_reward + self._terminal_reward(p, winner)
                    self.buffer.add(
                        Transition(
                            state=p_obs,
                            action=p_action,
                            reward=total_r,
                            next_state=zero_s,
                            next_mask=zero_m,
                            done=True,
                        )
                    )
                    pending[p] = None
                break

        return winner

    # ---------------------------------------------------------------- train

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
            # Double DQN: online selects, target evaluates
            q_online_next = self.online(sp)
            q_online_next = q_online_next.masked_fill(~nmask, -1e9)
            a_star = q_online_next.argmax(dim=1, keepdim=True)
            q_tgt = self.target(sp).gather(1, a_star).squeeze(1)
            q_tgt = torch.where(done, torch.zeros_like(q_tgt), q_tgt)
            target = r + self.cfg.gamma * q_tgt

        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q, target)
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.grad_clip)
        self.optim.step()
        return float(loss.item())

    # ---------------------------------------------------------------- loop

    def train(self, log_fn=None) -> None:
        ckpt_dir = Path(self.cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        wins = 0
        losses: List[float] = []

        for ep in range(1, self.cfg.episodes + 1):
            eps = self._epsilon(ep)

            # sample opponent type for this episode (seats 1..n-1)
            self._episode_opponent = self._roll_opponent()
            if self._episode_opponent == _OPP_POOL:
                self._refresh_opponent()

            winner = self._play_episode(eps)
            if winner == 0:
                wins += 1

            # snapshot pool on schedule
            if ep % self.cfg.pool_add_every == 0:
                self._pool_snapshot()

            if len(self.buffer) >= max(self.cfg.batch_size, self.cfg.warmup_steps):
                for _ in range(self.cfg.train_steps_per_episode):
                    loss = self._train_step()
                    losses.append(loss)
                    self._global_step += 1
                    if self._global_step % self.cfg.target_sync_every == 0:
                        self.target.load_state_dict(self.online.state_dict())

            if ep % self.cfg.eval_every == 0:
                avg_loss = float(np.mean(losses[-500:])) if losses else float("nan")
                win_rate_random = self.evaluate_vs_random(self.cfg.eval_games)
                win_rate_heur = self.evaluate_vs_heuristic(self.cfg.eval_games)
                m = {
                    "episode": ep,
                    "epsilon": round(eps, 4),
                    "buffer": len(self.buffer),
                    "avg_loss": avg_loss,
                    "wr_vs_random": win_rate_random,
                    "wr_vs_heuristic": win_rate_heur,
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

    # ---------------------------------------------------------------- eval

    def evaluate_vs_random(self, num_games: int = 100) -> float:
        return self._evaluate_vs(num_games, opponent_factory=lambda: RandomAgent(seed=None))

    def evaluate_vs_heuristic(self, num_games: int = 100) -> float:
        return self._evaluate_vs(num_games, opponent_factory=lambda: HeuristicAgent(seed=None))

    def _evaluate_vs(self, num_games: int, opponent_factory) -> float:
        """Play `num_games` as seat 0 (greedy) vs given opponent type; returns win rate."""
        from domino.agents import DQNAgent

        self.online.eval()
        hero = DQNAgent(self.online, epsilon=0.0, device=self.cfg.device, player_id=0)
        opps = [opponent_factory() for _ in range(self.cfg.num_players - 1)]

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
