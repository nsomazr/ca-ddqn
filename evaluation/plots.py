"""
Plotting helpers for CA-DRL / DDQN experiments.

All plots are generated with Matplotlib and saved under `results/`.
"""

from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import os


def plot_training_curves(
    episodes: List[int],
    rewards: List[float],
    delays: List[float],
    save_path_prefix: str,
) -> None:
    """Plot reward and delay vs episode for a single algorithm."""
    x = np.array(episodes)

    # Ensure output directory exists (robust even if caller didn't create it)
    out_dir = os.path.dirname(save_path_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(x, rewards, label="Episode reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_rewards.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(x, delays, label="Average delay", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Average delay")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_delays.png")
    plt.close()


def plot_algorithm_comparison(
    summary: Dict[str, Dict[str, float]],
    save_path: str,
) -> None:
    """Bar plots comparing average delay, reward, and throughput across algorithms."""
    algos = list(summary.keys())
    delays = [summary[a]["average_delay"] for a in algos]
    # Prefer per-step reward if available to keep scale comparable
    rewards = [
        summary[a].get("average_reward_per_step", summary[a]["average_reward"])
        for a in algos
    ]
    throughputs = [
        0.0 if summary[a].get("average_throughput") is None else summary[a].get("average_throughput", 0.0)
        for a in algos
    ]

    x = np.arange(len(algos))
    width = 0.35

    plt.figure(figsize=(7, 4))
    plt.bar(x - width / 2, delays, width, label="Average delay")
    plt.xticks(x, algos, rotation=20)
    plt.ylabel("Delay ratio")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path.replace(".png", "_delay.png"))
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar(x - width / 2, rewards, width, label="Average reward", color="green")
    plt.xticks(x, algos, rotation=20)
    plt.ylabel("Reward (per step)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path.replace(".png", "_reward.png"))
    plt.close()

    # Throughput comparison
    plt.figure(figsize=(7, 4))
    plt.bar(x - width / 2, throughputs, width, label="Average throughput", color="purple")
    plt.xticks(x, algos, rotation=20)
    plt.ylabel("Throughput (bits / time unit)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path.replace(".png", "_throughput.png"))
    plt.close()



