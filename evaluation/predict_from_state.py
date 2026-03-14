from __future__ import annotations

import argparse
import json
import os
from typing import Literal

import numpy as np

from algorithms.cadrl_agent import CADRLAgent
from algorithms.ddqn_agent import DDQNAgent
from environment.wireless_env import SatelliteEnvironment
from utils.config_loader import ExperimentConfig


def load_state_from_file(path: str, expected_dim: int) -> np.ndarray:
    """
    Load a state vector from JSON or .npy file and validate its size.

    - JSON: expects a list of numbers, e.g. [0.0, 1.0, ...]
    - NPY:   saved numpy array with shape (state_dim,)
    """
    if path.endswith(".npy"):
        arr = np.load(path).astype(np.float32)
    else:
        with open(path, "r") as f:
            data = json.load(f)
        arr = np.array(data, dtype=np.float32)

    if arr.ndim != 1 or arr.shape[0] != expected_dim:
        raise ValueError(f"State vector must have shape ({expected_dim},), got {arr.shape}")
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict action from a saved CA-DRL or DDQN model given a state vector.")
    parser.add_argument(
        "--agent",
        type=str,
        choices=["cadrl", "ddqn"],
        required=True,
        help="Which agent to use: 'cadrl' (CA-DRL DQN) or 'ddqn' (Double DQN).",
    )
    parser.add_argument(
        "--state-file",
        type=str,
        required=False,
        help="Path to JSON or .npy file containing a single state vector. "
        "If omitted, the script will use the current environment state after reset().",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Computation device for the model (e.g. 'cpu' or 'cuda').",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml()
    env = SatelliteEnvironment(cfg)

    models_dir = "models"
    if args.agent == "cadrl":
        model_path = os.path.join(models_dir, "ca_drl_model.pt")
        AgentCls = CADRLAgent
    else:
        model_path = os.path.join(models_dir, "ddqn_model.pt")
        AgentCls = DDQNAgent

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Train and save the model first.")

    agent = AgentCls(env, cfg.training, device=args.device)
    agent.load(model_path)

    if args.state_file:
        state = load_state_from_file(args.state_file, env.state_dim)
        print(f"Loaded state from {args.state_file} with shape {state.shape}")
    else:
        # Use an environment-generated state for convenience
        state = env.reset()
        print("No --state-file provided; using environment.reset() state.")

    action, q_values = agent.act(state)

    print("\n=== Prediction ===")
    print(f"Agent type       : {args.agent}")
    print(f"State dimension  : {env.state_dim}")
    print(f"Chosen action    : {action} (0..{env.action_dim - 2} = queued sensor index, {env.action_dim - 1} = no-op)")
    print(f"Q-values (len={len(q_values)}):")
    print(q_values)


if __name__ == "__main__":
    main()

