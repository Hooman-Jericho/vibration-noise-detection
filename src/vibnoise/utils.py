"""Small shared helpers: logging, seeding, environment capture, JSON sanitising."""

from __future__ import annotations

import logging
import math
import platform
import random
from typing import Any

import numpy as np


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def env_info() -> dict[str, str]:
    import pandas
    import scipy
    import sklearn

    from . import __version__

    info = {
        "vibnoise": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "scikit-learn": sklearn.__version__,
    }
    try:
        import xgboost

        info["xgboost"] = xgboost.__version__
    except ImportError:  # optional dependency
        pass
    return info


def clean_json(obj: Any) -> Any:
    """Recursively convert numpy / pandas scalars to builtins and NaN/inf to None."""
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean_json(obj.tolist())
    if isinstance(obj, np.generic):
        return clean_json(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj
