"""Deterministic seeding for reproducibility (Rule 6).

Seeds Python's random, NumPy, and PyTorch (if available) with a single call.
"""

from __future__ import annotations

import os
import random

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and optionally PyTorch.

    Also sets PYTHONHASHSEED and, if PyTorch is available, configures
    CuDNN deterministic behavior.

    Args:
        seed: The integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Seed PyTorch if it's installed (it's a commented-out dep in Phase 0)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info("Seeded Python, NumPy, and PyTorch with seed=%d", seed)
    except ImportError:
        logger.info("Seeded Python and NumPy with seed=%d (PyTorch not installed)", seed)
