"""Global seeding for reproducible experiments."""

from __future__ import annotations

import os
import random

import numpy as np

from tulip.utils.optional import is_available


def set_global_seed(seed: int) -> None:
    """Seed every random source tulip may touch (stdlib, numpy, torch if present).

    Note:
        ``PYTHONHASHSEED`` is exported for *child processes* only -- hash
        randomisation of the current interpreter is fixed at startup and
        cannot be changed here. tulip's own determinism never relies on
        builtin ``hash()`` ordering (stable hashes use blake2b).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if is_available("torch"):
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
