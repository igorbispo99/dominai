"""Uniform-sampling circular replay buffer for DQN."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Transition:
    state: np.ndarray       # (S,) float32
    action: int
    reward: float
    next_state: np.ndarray  # (S,) float32
    next_mask: np.ndarray   # (A,) bool
    done: bool


@dataclass
class Batch:
    states: np.ndarray      # (B, S) float32
    actions: np.ndarray     # (B,) int64
    rewards: np.ndarray     # (B,) float32
    next_states: np.ndarray # (B, S) float32
    next_masks: np.ndarray  # (B, A) bool
    dones: np.ndarray       # (B,) bool


class ReplayBuffer:
    def __init__(self, capacity: int, seed: Optional[int] = None) -> None:
        self.capacity = capacity
        self._buf: list[Transition] = []
        self._pos = 0
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._buf)

    def add(self, t: Transition) -> None:
        if len(self._buf) < self.capacity:
            self._buf.append(t)
        else:
            self._buf[self._pos] = t
            self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Batch:
        items = self._rng.sample(self._buf, batch_size)
        states = np.stack([t.state for t in items]).astype(np.float32)
        actions = np.array([t.action for t in items], dtype=np.int64)
        rewards = np.array([t.reward for t in items], dtype=np.float32)
        next_states = np.stack([t.next_state for t in items]).astype(np.float32)
        next_masks = np.stack([t.next_mask for t in items]).astype(bool)
        dones = np.array([t.done for t in items], dtype=bool)
        return Batch(states, actions, rewards, next_states, next_masks, dones)
