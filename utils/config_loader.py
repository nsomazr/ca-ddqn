"""
YAML experiment configuration loader.

This module centralises access to `configs/experiment_config.yaml` so that
environment, algorithms, and training scripts all use a single source of truth
for hyperparameters and system parameters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict

import yaml


DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "configs",
    "experiment_config.yaml",
)


def _load_raw_config(path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Experiment config YAML not found at: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


@dataclass
class SatelliteConfig:
    num_channels: int
    platforms_to_process: int


@dataclass
class EnvironmentConfig:
    max_traffic_duration: int
    simulation_steps: int
    sensor_arrival_rate: float
    max_waiting_queue: int
    state_window_size: int


@dataclass
class TrainingConfig:
    discount_factor: float
    learning_rate: float
    num_episodes: int
    minibatch_size: int
    replay_buffer_size: int
    epsilon_start: float
    epsilon_min: float
    epsilon_decay: float
    target_update_frequency: int


@dataclass
class EvaluationConfig:
    num_eval_episodes: int
    eval_interval: int
    metrics_window: int


@dataclass
class BaselineConfig:
    fcfs_delay: float
    fcfs_reward: float
    random_delay: float
    random_reward: float


@dataclass
class ExperimentConfig:
    satellite: SatelliteConfig
    environment: EnvironmentConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    baselines: BaselineConfig
    random_seed: int

    @classmethod
    def from_yaml(cls, path: str = DEFAULT_CONFIG_PATH) -> "ExperimentConfig":
        raw = _load_raw_config(path)

        sat_raw = raw["satellite"]
        env_raw = raw["environment"]
        tr_raw = raw["training"]
        ev_raw = raw["evaluation"]
        base_raw = raw["baselines_from_paper"]

        satellite = SatelliteConfig(
            num_channels=int(sat_raw["num_channels"]),
            platforms_to_process=int(sat_raw["platforms_to_process"]),
        )
        environment = EnvironmentConfig(
            max_traffic_duration=int(env_raw["max_traffic_duration"]),
            simulation_steps=int(env_raw["simulation_steps"]),
            sensor_arrival_rate=float(env_raw["sensor_arrival_rate"]),
            max_waiting_queue=int(env_raw["max_waiting_queue"]),
            state_window_size=int(env_raw["state_window_size"]),
        )
        training = TrainingConfig(
            discount_factor=float(tr_raw["discount_factor"]),
            learning_rate=float(tr_raw["learning_rate"]),
            num_episodes=int(tr_raw["num_episodes"]),
            minibatch_size=int(tr_raw["minibatch_size"]),
            replay_buffer_size=int(tr_raw["replay_buffer_size"]),
            epsilon_start=float(tr_raw["epsilon_start"]),
            epsilon_min=float(tr_raw["epsilon_min"]),
            epsilon_decay=float(tr_raw["epsilon_decay"]),
            target_update_frequency=int(tr_raw["target_update_frequency"]),
        )
        evaluation = EvaluationConfig(
            num_eval_episodes=int(ev_raw["num_eval_episodes"]),
            eval_interval=int(ev_raw["eval_interval"]),
            metrics_window=int(ev_raw["metrics_window"]),
        )
        baselines = BaselineConfig(
            fcfs_delay=float(base_raw["fcfs"]["average_delay"]),
            fcfs_reward=float(base_raw["fcfs"]["average_reward"]),
            random_delay=float(base_raw["random_access"]["average_delay"]),
            random_reward=float(base_raw["random_access"]["average_reward"]),
        )

        seed = int(raw.get("reproducibility", {}).get("random_seed", 42))

        return cls(
            satellite=satellite,
            environment=environment,
            training=training,
            evaluation=evaluation,
            baselines=baselines,
            random_seed=seed,
        )


