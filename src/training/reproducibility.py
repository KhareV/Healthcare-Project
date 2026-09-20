"""Centralized deterministic seed and device helpers."""

import random

import numpy as np
import torch


def set_deterministic_seed(seed: int, *, deterministic_algorithms: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch CPU/CUDA generators."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic_algorithms


def resolve_device(requested: str) -> torch.device:
    """Resolve cpu/cuda/auto without hard-coding a GPU index."""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    raise ValueError("device must be auto, cpu, or cuda")

