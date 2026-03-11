"""
Training script for the DDQN-based enhanced channel allocation algorithm.

Environment, state, action, and reward definitions are identical to CA-DRL; only
the learning algorithm is replaced with Double DQN.
"""

from __future__ import annotations

import json
import os
from typing import List

from tqdm import tqdm

from algorithms.ddqn_agent import DDQNAgent, DDQNTrainer
from environment.wireless_env import SatelliteEnvironment
from evaluation.metrics import EpisodeMetricsTracker, summarise_algorithm, attach_baseline_comparison
from evaluation.plots import plot_training_curves, plot_algorithm_comparison
from utils.config_loader import ExperimentConfig
from utils.reproducibility import ensure_reproducible_experiment


def main(device: str = "cpu") -> None:
    cfg = ExperimentConfig.from_yaml()
    ensure_reproducible_experiment(cfg.random_seed)

    # Phase 2: DDQN results under results/ddqn
    base_dir = os.path.join("results", "ddqn")
    os.makedirs(base_dir, exist_ok=True)
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Directory for saved models
    models_dir = os.path.join("models")
    os.makedirs(models_dir, exist_ok=True)

    env = SatelliteEnvironment(cfg)
    agent = DDQNAgent(env, cfg.training, device=device)
    trainer = DDQNTrainer(env, agent)

    tracker = EpisodeMetricsTracker(window_size=cfg.evaluation.metrics_window)
    num_episodes = cfg.training.num_episodes
    rewards: List[float] = []
    delays: List[float] = []

    for ep in tqdm(range(num_episodes), desc="Training DDQN"):
        metrics = trainer.train_episode()
        episode_reward = metrics["episode_reward"]
        avg_delay = metrics["average_delay"]
        thr = metrics["throughput"]

        tracker.update(episode_reward, avg_delay, thr)
        rewards.append(episode_reward)
        delays.append(avg_delay)

    # Save trained DDQN model
    ddqn_model_path = os.path.join(models_dir, "ddqn_model.pt")
    agent.save(ddqn_model_path)

    # Save curves
    episodes = list(range(num_episodes))
    plot_prefix = os.path.join(plots_dir, "ddqn_training")
    plot_training_curves(episodes, rewards, delays, plot_prefix)

    # Save DDQN summary and comparison to paper baselines and CA-DRL (if available)
    ddqn_summary = summarise_algorithm("DDQN", tracker.as_dict())

    # If CA-DRL reproduction has been run, include it; otherwise just DDQN + baselines
    ca_path = os.path.join("results", "ca-drl", "ca_drl_training_results.json")
    if os.path.exists(ca_path):
        with open(ca_path, "r") as f:
            ca_data = json.load(f)
        ca_summary = ca_data["summary"]
    else:
        ca_summary = {"average_delay": 0.0, "average_reward": 0.0, "average_throughput": 0.0}

    comparison = attach_baseline_comparison(
        ca_drl=ca_summary,
        ddqn=ddqn_summary,
        fcfs_delay=cfg.baselines.fcfs_delay,
        fcfs_reward=cfg.baselines.fcfs_reward,
        random_delay=cfg.baselines.random_delay,
        random_reward=cfg.baselines.random_reward,
        episode_length=cfg.environment.simulation_steps,
    )

    with open(os.path.join(base_dir, "ddqn_training_results.json"), "w") as f:
        json.dump(
            {
                "ddqn_summary": ddqn_summary,
                "comparison": comparison,
            },
            f,
            indent=2,
        )

    plot_algorithm_comparison(
        summary=comparison,
        save_path=os.path.join(plots_dir, "algorithm_comparison.png"),
    )


if __name__ == "__main__":
    main()


