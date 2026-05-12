#!/usr/bin/env python3
"""
Pokemon Battle Environment for Reinforcement Learning
Simple 1v1 battle: HP, moves, damage only.
"""

import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional


@dataclass
class Move:
    name: str
    power: int
    accuracy: float  # 0.0 - 1.0


@dataclass
class Pokemon:
    name: str
    max_hp: int
    moves: List[Move]
    current_hp: int = field(init=False)

    def __post_init__(self):
        self.current_hp = self.max_hp

    @property
    def is_fainted(self) -> bool:
        return self.current_hp <= 0

    @property
    def hp_ratio(self) -> float:
        return self.current_hp / self.max_hp

    def reset(self):
        self.current_hp = self.max_hp


# --- Predefined Pokemon roster ---
ROSTER = {
    "Charizard": Pokemon("Charizard", 100, [
        Move("Flamethrower", 90, 1.0),
        Move("Fire Blast", 110, 0.85),
        Move("Slash", 70, 1.0),
        Move("Ember", 40, 1.0),
    ]),
    "Blastoise": Pokemon("Blastoise", 100, [
        Move("Hydro Pump", 110, 0.80),
        Move("Water Gun", 40, 1.0),
        Move("Bite", 60, 1.0),
        Move("Surf", 90, 1.0),
    ]),
    "Venusaur": Pokemon("Venusaur", 100, [
        Move("Solar Beam", 120, 1.0),
        Move("Razor Leaf", 55, 0.95),
        Move("Vine Whip", 45, 1.0),
        Move("Body Slam", 85, 1.0),
    ]),
    "Pikachu": Pokemon("Pikachu", 70, [
        Move("Thunderbolt", 90, 1.0),
        Move("Thunder", 110, 0.70),
        Move("Quick Attack", 40, 1.0),
        Move("Thunder Shock", 40, 1.0),
    ]),
    "Mewtwo": Pokemon("Mewtwo", 110, [
        Move("Psychic", 90, 1.0),
        Move("Shadow Ball", 80, 1.0),
        Move("Hyper Beam", 150, 0.90),
        Move("Recover", 0, 1.0),   # Heals 30 HP
    ]),
}


def calculate_damage(move: Move, rng: random.Random) -> int:
    """Simple damage calculation with accuracy check."""
    if rng.random() > move.accuracy:
        return 0  # Miss
    # Random damage variance ±15%
    variance = rng.uniform(0.85, 1.15)
    return max(1, int(move.power * variance / 10))  # Scale down for HP pool


class BattleEnv:
    """
    Pokemon 1v1 Battle Environment.

    State: (agent_hp_bucket, opponent_hp_bucket, opponent_last_move_power)
    Action: index into agent's move list (0-3)
    """

    HP_BUCKETS = 5   # 0-19%, 20-39%, 40-59%, 60-79%, 80-100%
    MOVE_POWER_BUCKETS = 3  # low/med/high

    def __init__(self, agent_name: str = "Charizard", opponent_name: str = "Blastoise",
                 seed: int = 42):
        self.agent_template = ROSTER[agent_name]
        self.opponent_template = ROSTER[opponent_name]
        self.rng = random.Random(seed)
        self.agent: Optional[Pokemon] = None
        self.opponent: Optional[Pokemon] = None
        self.n_actions = len(self.agent_template.moves)
        self.done = False
        self.turn = 0

    def reset(self) -> Tuple:
        """Reset battle. Returns initial state."""
        import copy
        self.agent = copy.deepcopy(self.agent_template)
        self.opponent = copy.deepcopy(self.opponent_template)
        self.done = False
        self.turn = 0
        return self._get_state()

    def _hp_bucket(self, hp_ratio: float) -> int:
        return min(int(hp_ratio * self.HP_BUCKETS), self.HP_BUCKETS - 1)

    def _power_bucket(self, power: int) -> int:
        if power < 50:
            return 0
        elif power < 90:
            return 1
        return 2

    def _get_state(self, last_opp_power: int = 0) -> Tuple:
        agent_bucket = self._hp_bucket(self.agent.hp_ratio)
        opp_bucket = self._hp_bucket(self.opponent.hp_ratio)
        power_bucket = self._power_bucket(last_opp_power)
        return (agent_bucket, opp_bucket, power_bucket)

    def _opponent_act(self) -> Move:
        """Simple opponent AI: picks random move."""
        return self.rng.choice(self.opponent.moves)

    def step(self, action: int) -> Tuple[Tuple, float, bool, Dict]:
        """
        Agent takes action (move index).
        Returns: (next_state, reward, done, info)
        """
        assert not self.done, "Episode is done. Call reset()."
        self.turn += 1

        agent_move = self.agent.moves[action]
        opp_move = self._opponent_act()

        # --- Agent attacks opponent ---
        agent_dmg = calculate_damage(agent_move, self.rng)

        # Special: Mewtwo Recover heals agent instead
        if agent_move.name == "Recover":
            heal = 30
            self.agent.current_hp = min(self.agent.max_hp, self.agent.current_hp + heal)
            agent_dmg = 0
        else:
            self.opponent.current_hp -= agent_dmg

        # --- Opponent attacks agent ---
        opp_dmg = calculate_damage(opp_move, self.rng)
        self.agent.current_hp -= opp_dmg

        # --- Determine outcome ---
        agent_fainted = self.agent.is_fainted
        opp_fainted = self.opponent.is_fainted

        reward = 0.0
        if opp_fainted and not agent_fainted:
            reward = 1.0   # Win
            self.done = True
        elif agent_fainted and not opp_fainted:
            reward = -1.0  # Loss
            self.done = True
        elif agent_fainted and opp_fainted:
            reward = -0.5  # Tie (both faint)
            self.done = True
        else:
            # Small reward for dealing damage relative to remaining HP
            reward = (agent_dmg - opp_dmg) / 100.0

        next_state = self._get_state(opp_move.power)

        info = {
            "turn": self.turn,
            "agent_move": agent_move.name,
            "agent_dmg": agent_dmg,
            "opp_move": opp_move.name,
            "opp_dmg": opp_dmg,
            "agent_hp": max(0, self.agent.current_hp),
            "opp_hp": max(0, self.opponent.current_hp),
        }

        return next_state, reward, self.done, info

    @property
    def state_space_size(self) -> int:
        return self.HP_BUCKETS * self.HP_BUCKETS * self.MOVE_POWER_BUCKETS

    def state_to_index(self, state: Tuple) -> int:
        a, o, p = state
        return a * (self.HP_BUCKETS * self.MOVE_POWER_BUCKETS) + o * self.MOVE_POWER_BUCKETS + p
