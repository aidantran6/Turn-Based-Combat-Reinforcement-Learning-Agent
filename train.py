#!/usr/bin/env python3
"""
Training loop + battle statistics dashboard (CLI)
"""

import time
from collections import defaultdict, deque
from typing import List, Dict
import numpy as np

from pokemon_env import BattleEnv
from agent import QLearningAgent, RandomAgent


# ── ANSI colors ──────────────────────────────────────────────────────────────
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

BAR_WIDTH = 30


def hp_bar(ratio: float, width: int = BAR_WIDTH) -> str:
    filled = int(ratio * width)
    color = GREEN if ratio > 0.5 else (YELLOW if ratio > 0.25 else RED)
    return color + "█" * filled + DIM + "░" * (width - filled) + RESET


def print_header():
    print(f"\n{BOLD}{CYAN}{'═'*60}")
    print(f"  🎮  POKEMON Q-LEARNING BATTLE AGENT")
    print(f"{'═'*60}{RESET}\n")


def print_training_stats(episode: int, total: int, win_rate: float,
                         avg_turns: float, epsilon: float, recent_wins: deque):
    bar = int((episode / total) * 40)
    prog = f"[{'█'*bar}{'░'*(40-bar)}]"
    recent = sum(recent_wins) / len(recent_wins) * 100 if recent_wins else 0
    print(f"\r{CYAN}{prog}{RESET} Ep {episode:>5}/{total} | "
          f"WinRate: {GREEN}{win_rate*100:5.1f}%{RESET} | "
          f"Recent: {YELLOW}{recent:5.1f}%{RESET} | "
          f"AvgTurns: {avg_turns:4.1f} | "
          f"ε: {epsilon:.3f}", end="", flush=True)


def train(n_episodes: int = 5000, eval_every: int = 500,
          agent_name: str = "Charizard", opponent_name: str = "Blastoise",
          verbose_battles: int = 3) -> QLearningAgent:

    print_header()
    print(f"  Agent:    {BOLD}{agent_name}{RESET}")
    print(f"  Opponent: {BOLD}{opponent_name}{RESET}")
    print(f"  Episodes: {n_episodes}\n")

    env = BattleEnv(agent_name, opponent_name, seed=42)
    agent = QLearningAgent(
        state_space_size=env.state_space_size,
        n_actions=env.n_actions,
        alpha=0.1,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.998,
    )

    wins = losses = ties = 0
    total_turns = []
    move_usage: Dict[str, int] = defaultdict(int)
    recent_wins: deque = deque(maxlen=200)
    win_history: List[float] = []

    print(f"{BOLD}  Training...{RESET}\n")
    time.sleep(0.3)

    for ep in range(1, n_episodes + 1):
        state = env.reset()
        state_idx = env.state_to_index(state)
        ep_done = False

        while not ep_done:
            action = agent.choose_action(state_idx)
            next_state, reward, ep_done, info = env.step(action)
            next_idx = env.state_to_index(next_state)
            agent.update(state_idx, action, reward, next_idx, ep_done)
            move_usage[info["agent_move"]] += 1
            state_idx = next_idx

        agent.decay_epsilon()

        won = reward == 1.0
        lost = reward == -1.0
        tied = reward == -0.5

        if won:   wins += 1
        if lost:  losses += 1
        if tied:  ties += 1

        recent_wins.append(1 if won else 0)
        total_turns.append(info["turn"])

        win_rate = wins / ep
        avg_turns = sum(total_turns[-200:]) / min(len(total_turns), 200)

        if ep % 100 == 0:
            print_training_stats(ep, n_episodes, win_rate, avg_turns,
                                 agent.epsilon, recent_wins)
        if ep % eval_every == 0:
            win_history.append(wins / ep)

    print(f"\n\n{BOLD}{GREEN}  ✅ Training complete!{RESET}\n")

    # ── Final stats dashboard ─────────────────────────────────────────────────
    print(f"{BOLD}{CYAN}{'═'*60}")
    print(f"  📊  BATTLE STATISTICS DASHBOARD")
    print(f"{'═'*60}{RESET}\n")

    total = wins + losses + ties
    print(f"  {'Total Battles:':<20} {total}")
    print(f"  {'Wins:':<20} {GREEN}{wins} ({wins/total*100:.1f}%){RESET}")
    print(f"  {'Losses:':<20} {RED}{losses} ({losses/total*100:.1f}%){RESET}")
    print(f"  {'Ties:':<20} {YELLOW}{ties} ({ties/total*100:.1f}%){RESET}")
    print(f"  {'Avg Turns/Battle:':<20} {sum(total_turns)/len(total_turns):.1f}\n")

    # Win rate over time
    print(f"  {BOLD}Win Rate Progression:{RESET}")
    checkpoints = np.linspace(0, n_episodes, len(win_history) + 1, dtype=int)[1:]
    for ep_num, wr in zip(checkpoints, win_history):
        bar = int(wr * 20)
        color = GREEN if wr > 0.55 else (YELLOW if wr > 0.4 else RED)
        print(f"    Ep {ep_num:>5}: {color}{'█'*bar}{'░'*(20-bar)}{RESET} {wr*100:5.1f}%")

    # Move usage
    print(f"\n  {BOLD}Move Usage:{RESET}")
    total_moves = sum(move_usage.values())
    for move, count in sorted(move_usage.items(), key=lambda x: -x[1]):
        pct = count / total_moves
        bar = int(pct * 25)
        print(f"    {move:<18} {'█'*bar}{'░'*(25-bar)} {pct*100:5.1f}%")

    # Q-table insight
    print(f"\n  {BOLD}Q-Table Insight (greedy policy per HP bucket):{RESET}")
    move_names = [m.name for m in env.agent_template.moves]
    print(f"    {'Agent HP':<12} {'Opp HP':<12} Best Move")
    print(f"    {'─'*40}")
    for a_bucket in range(env.HP_BUCKETS):
        for o_bucket in range(0, env.HP_BUCKETS, 2):
            state = (a_bucket, o_bucket, 1)
            idx = env.state_to_index(state)
            best = agent.best_action(idx)
            a_label = f"{a_bucket*20}-{(a_bucket+1)*20}%"
            o_label = f"{o_bucket*20}-{(o_bucket+1)*20}%"
            print(f"    {a_label:<12} {o_label:<12} {CYAN}{move_names[best]}{RESET}")

    return agent


def evaluate(agent: QLearningAgent, env: BattleEnv, n_battles: int = 10,
             verbose: bool = True):
    """Run greedy battles and print turn-by-turn logs."""
    print(f"\n{BOLD}{CYAN}{'═'*60}")
    print(f"  ⚔️   LIVE BATTLE DEMO (greedy policy, {n_battles} battles)")
    print(f"{'═'*60}{RESET}\n")

    wins = 0
    for battle in range(1, n_battles + 1):
        state = env.reset()
        state_idx = env.state_to_index(state)
        done = False
        battle_log = []

        while not done:
            action = agent.best_action(state_idx)
            next_state, reward, done, info = env.step(action)
            state_idx = env.state_to_index(next_state)
            battle_log.append(info)

        won = reward == 1.0
        if won:
            wins += 1

        if verbose:
            result_str = f"{GREEN}WIN ✓{RESET}" if won else f"{RED}LOSS ✗{RESET}"
            print(f"  Battle {battle}: {result_str} in {info['turn']} turns")
            # Print last few turns
            for t in battle_log[-3:]:
                print(f"    Turn {t['turn']:>2}: {CYAN}{t['agent_move']}{RESET} "
                      f"({t['agent_dmg']} dmg) vs {RED}{t['opp_move']}{RESET} "
                      f"({t['opp_dmg']} dmg) | "
                      f"HP {GREEN}{t['agent_hp']}{RESET} vs {RED}{t['opp_hp']}{RESET}")
            print()

    print(f"  {BOLD}Greedy Win Rate: {GREEN}{wins}/{n_battles} "
          f"({wins/n_battles*100:.0f}%){RESET}\n")


def compare_vs_random(agent: QLearningAgent, agent_name: str, opponent_name: str,
                      n_battles: int = 500):
    """Compare trained agent vs random agent."""
    print(f"{BOLD}{CYAN}{'═'*60}")
    print(f"  🆚  AGENT VS RANDOM BASELINE ({n_battles} battles each)")
    print(f"{'═'*60}{RESET}\n")

    from agent import RandomAgent

    for label, act_fn in [("Trained Q-Agent", None), ("Random Baseline", "random")]:
        env = BattleEnv(agent_name, opponent_name, seed=99)
        rand = RandomAgent(env.n_actions, seed=7) if act_fn == "random" else None
        wins = 0

        for _ in range(n_battles):
            state = env.reset()
            state_idx = env.state_to_index(state)
            done = False
            while not done:
                action = (rand.choose_action(state_idx) if rand
                          else agent.best_action(state_idx))
                next_state, reward, done, info = env.step(action)
                state_idx = env.state_to_index(next_state)
            if reward == 1.0:
                wins += 1

        wr = wins / n_battles
        color = GREEN if wr > 0.5 else RED
        bar = int(wr * 30)
        print(f"  {label:<22} {color}{'█'*bar}{'░'*(30-bar)}{RESET} {wr*100:.1f}%")

    print()


if __name__ == "__main__":
    AGENT = "Charizard"
    OPPONENT = "Blastoise"

    trained_agent = train(n_episodes=5000, eval_every=1000,
                          agent_name=AGENT, opponent_name=OPPONENT)

    demo_env = BattleEnv(AGENT, OPPONENT, seed=777)
    evaluate(trained_agent, demo_env, n_battles=5)
    compare_vs_random(trained_agent, AGENT, OPPONENT, n_battles=500)
