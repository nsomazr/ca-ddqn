"""
Metrics utilities for CA-DRL / DDQN evaluation.

Provides:
- Episode-level running statistics (delay, reward, throughput)
- Aggregation helpers for algorithm comparison
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class EpisodeMetricsTracker:
    window_size: int = 100
    rewards: List[float] = field(default_factory=list)
    delays: List[float] = field(default_factory=list)
    throughputs: List[float] = field(default_factory=list)

    def update(self, episode_reward: float, average_delay: float, throughput: float) -> None:
        self.rewards.append(episode_reward)
        self.delays.append(average_delay)
        self.throughputs.append(throughput)

    def moving_average(self, xs: List[float]) -> np.ndarray:
        if not xs:
            return np.array([], dtype=np.float32)
        window = min(self.window_size, len(xs))
        cumsum = np.cumsum(np.insert(xs, 0, 0.0))
        ma = (cumsum[window:] - cumsum[:-window]) / window
        # Prepend first value to keep length aligned
        pad = np.full(len(xs) - len(ma), ma[0])
        return np.concatenate([pad, ma])

    def as_dict(self) -> Dict[str, List[float]]:
        return {
            "episode_rewards": self.rewards,
            "episode_delays": self.delays,
            "episode_throughputs": self.throughputs,
        }


def summarise_algorithm(name: str, episodes: Dict[str, List[float]]) -> Dict[str, float]:
    rewards = np.array(episodes["episode_rewards"], dtype=np.float32)
    delays = np.array(episodes["episode_delays"], dtype=np.float32)
    thr = np.array(episodes["episode_throughputs"], dtype=np.float32)
    return {
        "algorithm": name,
        "average_reward": float(rewards.mean()) if rewards.size else 0.0,
        "average_delay": float(delays.mean()) if delays.size else 0.0,
        "average_throughput": float(thr.mean()) if thr.size else 0.0,
    }


def attach_baseline_comparison(
    ca_drl: Dict[str, float],
    ddqn: Dict[str, float],
    fcfs_delay: float,
    fcfs_reward: float,
    random_delay: float,
    random_reward: float,
    episode_length: int,
) -> Dict[str, Dict[str, float]]:
    """
    Build a comparison table using CA-DRL/DDQN measurements and fixed Random Access / FCFS
    baselines from the paper (no baseline simulations are run).
    The insertion order is:
      Random Access -> FCFS -> CA-DRL -> DDQN
    so that plots always show Random before FCFS.
    """
    def _delay_improvement(alg_delay: float, base_delay: float) -> float:
        return (base_delay - alg_delay) / base_delay * 100.0 if base_delay > 0 else 0.0

    def _reward_improvement(alg_reward: float, base_reward: float) -> float:
        return (1.0 - alg_reward / base_reward) * 100.0 if base_reward != 0 else 0.0

    table: Dict[str, Dict[str, float]] = {}

    table["Random Access"] = {
        "average_delay": random_delay,
        "average_reward": random_reward,
        "average_reward_per_step": random_reward / float(episode_length),
        "average_throughput": None,  # N/A: throughput not reported in paper
    }
    table["FCFS"] = {
        "average_delay": fcfs_delay,
        "average_reward": fcfs_reward,
        "average_reward_per_step": fcfs_reward / float(episode_length),
        "average_throughput": None,  # N/A: throughput not reported in paper
    }
    table["CA-DRL"] = dict(ca_drl)
    table["CA-DRL"]["average_reward_per_step"] = ca_drl["average_reward"] / float(episode_length)
    table["DDQN"] = dict(ddqn)
    table["DDQN"]["average_reward_per_step"] = ddqn["average_reward"] / float(episode_length)

    for name in ["CA-DRL", "DDQN"]:
        d = table[name]
        d["delay_improvement_over_fcfs"] = _delay_improvement(d["average_delay"], fcfs_delay)
        d["delay_improvement_over_random"] = _delay_improvement(d["average_delay"], random_delay)
        d["reward_improvement_over_fcfs"] = _reward_improvement(d["average_reward"], fcfs_reward)
        d["reward_improvement_over_random"] = _reward_improvement(d["average_reward"], random_reward)

    return table


def cadrl_baseline_comparison(
    ca_drl: Dict[str, float],
    fcfs_delay: float,
    fcfs_reward: float,
    random_delay: float,
    random_reward: float,
    episode_length: int,
) -> Dict[str, Dict[str, float]]:
    """
    Build comparison table for reproduction phase (CA-DRL vs Random Access vs FCFS).
    The insertion order is:
      Random Access -> FCFS -> CA-DRL
    so that plots always show Random before FCFS.
    """
    table: Dict[str, Dict[str, float]] = {}

    table["Random Access"] = {
        "average_delay": random_delay,
        "average_reward": random_reward,
        "average_reward_per_step": random_reward / float(episode_length),
        "average_throughput": None,  # N/A
    }
    table["FCFS"] = {
        "average_delay": fcfs_delay,
        "average_reward": fcfs_reward,
        "average_reward_per_step": fcfs_reward / float(episode_length),
        "average_throughput": None,  # N/A
    }
    table["CA-DRL"] = dict(ca_drl)
    table["CA-DRL"]["average_reward_per_step"] = ca_drl["average_reward"] / float(episode_length)
    return table



