from __future__ import annotations

import json
import os

from tqdm import tqdm

from algorithms.ddqn_agent import DDQNAgent, DDQNTrainer
from environment.wireless_env import SatelliteEnvironment
from evaluation.metrics import EpisodeMetricsTracker, summarise_algorithm, attach_baseline_comparison
from evaluation.plots import plot_algorithm_comparison
from utils.config_loader import ExperimentConfig
from utils.reproducibility import ensure_reproducible_experiment


def main(device: str = "cpu") -> None:
    cfg = ExperimentConfig.from_yaml()
    ensure_reproducible_experiment(cfg.random_seed)

    env = SatelliteEnvironment(cfg)

    models_dir = "models"
    ddqn_model_path = os.path.join(models_dir, "ddqn_model.pt")
    if not os.path.exists(ddqn_model_path):
        raise FileNotFoundError(f"DDQN model not found at {ddqn_model_path}. Train DDQN first.")

    agent = DDQNAgent(env, cfg.training, device=device)
    agent.load(ddqn_model_path)
    trainer = DDQNTrainer(env, agent)

    tracker = EpisodeMetricsTracker(window_size=cfg.evaluation.metrics_window)
    for _ in tqdm(range(cfg.evaluation.num_eval_episodes), desc="Evaluating DDQN model"):
        metrics = trainer.evaluate_episode()
        tracker.update(
            metrics["episode_reward"],
            metrics["average_delay"],
            metrics["throughput"],
        )

    summary = summarise_algorithm("DDQN", tracker.as_dict())

    comparison = attach_baseline_comparison(
        ca_drl=summary,  # treated as algorithm under test; CA-DRL entries remain unused here
        ddqn=summary,
        fcfs_delay=cfg.baselines.fcfs_delay,
        fcfs_reward=cfg.baselines.fcfs_reward,
        random_delay=cfg.baselines.random_delay,
        random_reward=cfg.baselines.random_reward,
        episode_length=cfg.environment.simulation_steps,
    )

    results_dir = os.path.join("results", "model_eval", "ddqn")
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, "ddqn_model_evaluation.json"), "w") as f:
        json.dump(
            {
                "summary": summary,
                "comparison": comparison,
            },
            f,
            indent=2,
        )

    plot_path = os.path.join(results_dir, "algorithm_comparison_ddqn_model.png")
    plot_algorithm_comparison(summary=comparison, save_path=plot_path)


if __name__ == "__main__":
    main()

