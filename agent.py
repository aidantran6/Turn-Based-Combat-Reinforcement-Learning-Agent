#!/usr/bin/env python3
"""
Q-Learning Agent for Pokemon Battle
"""

import random
import numpy as np
from typing import Tuple, Dict


class QLearningAgent:
    """
    Tabular Q-Learning agent.
    Q-table: state_index x n_actions -> Q-value
    """

    def __init__(self, state_space_size: int, n_actions: int,
                 alpha: float = 0.1, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.995, seed: int = 0):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # Q-table initialized to zero
        self.q_table = np.zeros((state_space_size, n_actions))

        # Stats tracking
        self.n_steps = 0
        self.n_episodes = 0

    def choose_action(self, state_idx: int) -> int:
        """Epsilon-greedy action selection."""
        if self.rng.random() < self.epsilon:
            return self.rng.randint(0, self.n_actions - 1)
        return int(np.argmax(self.q_table[state_idx]))

    def update(self, state_idx: int, action: int, reward: float,
               next_state_idx: int, done: bool) -> float:
        """
        Q-learning update rule:
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
        """
        current_q = self.q_table[state_idx, action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state_idx])
        td_error = target - current_q
        self.q_table[state_idx, action] += self.alpha * td_error
        self.n_steps += 1
        return abs(td_error)

    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.n_episodes += 1

    def best_action(self, state_idx: int) -> int:
        return int(np.argmax(self.q_table[state_idx]))

    def get_policy(self) -> np.ndarray:
        """Returns greedy policy: best action per state."""
        return np.argmax(self.q_table, axis=1)


class RandomAgent:
    """Baseline: picks random moves every turn."""
    def __init__(self, n_actions: int, seed: int = 1):
        self.n_actions = n_actions
        self.rng = random.Random(seed)

    def choose_action(self, state_idx: int) -> int:
        return self.rng.randint(0, self.n_actions - 1)
