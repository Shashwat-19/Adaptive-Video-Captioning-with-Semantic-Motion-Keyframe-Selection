"""Deterministic-seeding support."""

from __future__ import annotations

import random


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch when those optional dependencies exist."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
