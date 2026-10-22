"""Final-model fitting and (de)serialisation of self-describing model bundles."""

from __future__ import annotations

import datetime as dt
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from .config import Config
from .dataset import FeatureTable
from .evaluation import make_group_splits
from .models import make_pipeline, param_grid
from .utils import env_info

log = logging.getLogger(__name__)

BUNDLE_VERSION = 1


@dataclass
class Bundle:
    estimator: Pipeline
    config: Config
    feature_names: list[str]
    meta: dict[str, Any]


def fit_final(
    table: FeatureTable, model_name: str, cfg: Config
) -> tuple[Pipeline, dict[str, Any], float]:
    """Tune with *grouped* CV on all engines, then refit on everything."""
    if table.y is None:
        raise ValueError("training needs labeled data")
    pipe = make_pipeline(model_name, cfg.model.seed, cfg.model.pca_variance, cfg.model.n_jobs)
    splits, k = make_group_splits(
        table.X, table.y, table.groups, cfg.eval.inner_splits, cfg.model.seed
    )
    gs = GridSearchCV(
        pipe,
        param_grid(model_name),
        cv=splits,
        scoring=cfg.eval.scoring,
        n_jobs=cfg.model.n_jobs,
        refit=True,
        error_score="raise",
    )
    gs.fit(table.X, table.y)
    log.info(
        "best params %s (grouped %d-fold %s = %.3f)",
        gs.best_params_,
        k,
        cfg.eval.scoring,
        gs.best_score_,
    )
    return gs.best_estimator_, gs.best_params_, float(gs.best_score_)


def save_bundle(
    path: str | Path,
    estimator: Pipeline,
    cfg: Config,
    table: FeatureTable,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = table.engine_labels() if table.y is not None else {}
    meta = {
        "bundle_version": BUNDLE_VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "env": env_info(),
        "n_windows": len(table),
        "n_engines": len(table.engines),
        "engines_per_class": {
            "OK": sum(v == 0 for v in labels.values()),
            "NG": sum(v == 1 for v in labels.values()),
        },
        **(extra_meta or {}),
    }
    joblib.dump(
        {
            "estimator": estimator,
            "config": cfg.to_dict(),
            "feature_names": table.feature_names,
            "meta": meta,
        },
        path,
    )
    return path


def load_bundle(path: str | Path) -> Bundle:
    """Load a bundle. NOTE: joblib uses pickle - only load files you trust."""
    raw = joblib.load(path)
    missing = {"estimator", "config", "feature_names", "meta"} - set(raw)
    if missing:
        raise ValueError(f"{path} is not a vibnoise model bundle (missing {sorted(missing)})")
    saved = raw["meta"].get("env", {}).get("scikit-learn")
    if saved and saved != sklearn.__version__:
        warnings.warn(
            f"model was trained with scikit-learn {saved}, running {sklearn.__version__}; "
            "predictions may differ or fail",
            stacklevel=2,
        )
    return Bundle(
        raw["estimator"], Config.from_dict(raw["config"]), list(raw["feature_names"]), raw["meta"]
    )


def check_features(bundle: Bundle, table: FeatureTable) -> None:
    if list(table.feature_names) != list(bundle.feature_names) or table.X.shape[1] != len(
        bundle.feature_names
    ):
        raise ValueError("Feature layout of the new data does not match the trained model")
    if not np.isfinite(table.X).all():
        raise ValueError("non-finite features in input")
