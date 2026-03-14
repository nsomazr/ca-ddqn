from __future__ import annotations

import json
import os

from tqdm import tqdm

from algorithms.cadrl_agent import CADRLAgent, CADRLTrainer
from environment.wireless_env import SatelliteEnvironment
from evaluation.metrics import EpisodeMetricsTracker, summarise_algorithm, cadrl_baseline_comparison
from evaluation.plots import plot_algorithm_comparison
from utils.config_loader import ExperimentConfig
from utils.reproducibility import ensure_reproducible_experiment


def main(device: str = "cpu") -> None:
    cfg = ExperimentConfig.from_yaml()
    ensure_reproducible_experiment(cfg.random_seed)

    env = SatelliteEnvironment(cfg)

    models_dir = "models"
    ca_model_path = os.path.join(models_dir, "ca_drl_model.pt")
    if not os.path.exists(ca_model_path):
        raise FileNotFoundError(f"CA-DRL model not found at {ca_model_path}. Train CA-DRL first.")

    agent = CADRLAgent(env, cfg.training, device=device)
    agent.load(ca_model_path)
    trainer = CADRLTrainer(env, agent)

    tracker = EpisodeMetricsTracker(window_size=cfg.evaluation.metrics_window)
    for _ in tqdm(range(cfg.evaluation.num_eval_episodes), desc="Evaluating CA-DRL model"):
        metrics = trainer.evaluate_episode()
        tracker.update(
            metrics["episode_reward"],
            metrics["average_delay"],
            metrics["throughput"],
        )

    summary = summarise_algorithm("CA-DRL", tracker.as_dict())
    comparison = cadrl_baseline_comparison(
        ca_drl=summary,
        fcfs_delay=cfg.baselines.fcfs_delay,
        fcfs_reward=cfg.baselines.fcfs_reward,
        random_delay=cfg.baselines.random_delay,
        random_reward=cfg.baselines.random_reward,
        episode_length=cfg.environment.simulation_steps,
    )

    results_dir = os.path.join("results", "model_eval", "ca_drl")
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, "ca_drl_model_evaluation.json"), "w") as f:
        json.dump(
            {
                "summary": summary,
                "comparison": comparison,
            },
            f,
            indent=2,
        )

    plot_path = os.path.join(results_dir, "algorithm_comparison_cadrl_model.png")
    plot_algorithm_comparison(summary=comparison, save_path=plot_path)


if __name__ == "__main__":
    main()

