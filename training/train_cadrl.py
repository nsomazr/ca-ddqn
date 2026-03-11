"""
Training script for reproducing CA-DRL with a DQN agent.

This script:
- Loads experiment configuration from `configs/experiment_config.yaml`
- Builds the CA-DRL satellite environment
- Trains the DQN-based CA-DRL agent
- Saves training curves and JSON metrics into `results/reproduce/`
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from algorithms.cadrl_agent import CADRLAgent, CADRLTrainer
from environment.wireless_env import SatelliteEnvironment
from evaluation.metrics import EpisodeMetricsTracker, summarise_algorithm, cadrl_baseline_comparison
from evaluation.plots import plot_training_curves, plot_algorithm_comparison
from dataclasses import asdict
from utils.config_loader import ExperimentConfig
from utils.reproducibility import ensure_reproducible_experiment


def main(device: str = "cpu") -> None:
    cfg = ExperimentConfig.from_yaml()
    ensure_reproducible_experiment(cfg.random_seed)

    # Directories (Phase 1: CA-DRL goes under results/ca-drl)
    base_dir = os.path.join("results", "ca-drl")
    os.makedirs(base_dir, exist_ok=True)
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Directory for saved models
    models_dir = os.path.join("models")
    os.makedirs(models_dir, exist_ok=True)

    # Environment and agent
    env = SatelliteEnvironment(cfg)
    agent = CADRLAgent(env, cfg.training, device=device)
    trainer = CADRLTrainer(env, agent)

    tracker = EpisodeMetricsTracker(window_size=cfg.evaluation.metrics_window)
    num_episodes = cfg.training.num_episodes
    rewards: List[float] = []
    delays: List[float] = []

    for ep in tqdm(range(num_episodes), desc="Training CA-DRL"):
        metrics = trainer.train_episode()
        episode_reward = metrics["episode_reward"]
        avg_delay = metrics["average_delay"]
        thr = metrics["throughput"]

        tracker.update(episode_reward, avg_delay, thr)
        rewards.append(episode_reward)
        delays.append(avg_delay)

    # Save trained CA-DRL model
    cadrl_model_path = os.path.join(models_dir, "ca_drl_model.pt")
    agent.save(cadrl_model_path)

    # Save curves
    episodes = list(range(num_episodes))
    plot_prefix = os.path.join(plots_dir, "cadrl_training")
    plot_training_curves(episodes, rewards, delays, plot_prefix)

    # Save summary stats (convert config dataclasses to plain dicts for JSON)
    summary = summarise_algorithm("CA-DRL", tracker.as_dict())

    # Build comparison table vs FCFS and Random Access (paper baselines)
    comparison = cadrl_baseline_comparison(
        ca_drl=summary,
        fcfs_delay=cfg.baselines.fcfs_delay,
        fcfs_reward=cfg.baselines.fcfs_reward,
        random_delay=cfg.baselines.random_delay,
        random_reward=cfg.baselines.random_reward,
        episode_length=cfg.environment.simulation_steps,
    )

    out = {
        "config": asdict(cfg),
        "summary": summary,
        "comparison": comparison,
    }
    with open(os.path.join(base_dir, "ca_drl_training_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    # Plot CA-DRL vs FCFS vs Random Access bar charts for all metrics
    comparison_plots_dir = plots_dir
    os.makedirs(comparison_plots_dir, exist_ok=True)
    comparison_plot_path = os.path.join(comparison_plots_dir, "algorithm_comparison_cadrl.png")
    plot_algorithm_comparison(comparison, save_path=comparison_plot_path)


if __name__ == "__main__":
    main()


