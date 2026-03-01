from __future__ import annotations
import numpy as np


class TabularQLearningAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.997,
        seed: int | None = 42,
    ):
        self.n_states = n_states
        self.n_actions = n_actions

        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.rng = np.random.default_rng(seed)
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float32)

    def choose_action(self, state: int, training: bool = True) -> int:
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.greedy_action(state)

    def greedy_action(self, state: int) -> int:
        q = self.q_table[state]
        max_q = np.max(q)
        best = np.flatnonzero(np.isclose(q, max_q))
        return int(self.rng.choice(best))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        current_q = self.q_table[state, action]
        target = reward if done else reward + self.gamma * float(np.max(self.q_table[next_state]))
        self.q_table[state, action] = current_q + self.alpha * (target - current_q)

    def decay_epsilon_step(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        np.save(path, self.q_table)

    def load(self, path: str) -> None:
        q = np.load(path)
        if q.shape != (self.n_states, self.n_actions):
            raise ValueError(f"Q-table shape mismatch: expected {(self.n_states, self.n_actions)}, got {q.shape}")
        self.q_table = q.astype(np.float32)