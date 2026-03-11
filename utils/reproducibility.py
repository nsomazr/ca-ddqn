"""
Reproducibility utilities for CA-DRL / DDQN experiments.

All entry points (training scripts, notebooks) should call
`ensure_reproducible_experiment` before constructing the environment or agents.
"""

import os
import random
from typing import Optional

import numpy as np
import torch


DEFAULT_SEED = 42


def set_all_seeds(seed: int = DEFAULT_SEED) -> None:
    """Seed Python, NumPy and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_reproducible_experiment(seed: Optional[int] = None) -> int:
    """
    Configure the full stack for deterministic behaviour.

    Returns the seed actually used so it can be logged with results.
    """
    if seed is None:
        seed = DEFAULT_SEED

    set_all_seeds(seed)

    # Extra safety for CUDA 11+
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        # Older PyTorch versions may not support this; ignore gracefully.
        pass

    return seed


