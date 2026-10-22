"""Model registry: one leakage-safe ``Pipeline`` + one *valid* hyper-parameter grid per model.

Fixes over the original: scaling happens inside the pipeline (so it is fitted on training folds
only), models are constructed per call (no shared global estimator instances), the grids contain
no invalid combinations or options removed from scikit-learn (``max_features="auto"``,
``penalty="l1"`` with ``lbfgs``...), every stochastic model is seeded, and every model exposes
``predict_proba`` so probabilities and decisions are consistent.
"""

from __future__ import annotations

import copy
import importlib.util
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

_GRIDS: dict[str, list[dict[str, list[Any]]]] = {
    "logreg": [{"clf__C": list(np.logspace(-3, 3, 7))}],
    "svm": [
        {
            "clf__estimator__kernel": ["rbf"],
            "clf__estimator__C": [0.1, 1, 10, 100],
            "clf__estimator__gamma": ["scale", 1e-3, 1e-2, 1e-1],
        },
        {"clf__estimator__kernel": ["linear"], "clf__estimator__C": [0.01, 0.1, 1, 10]},
    ],
    "knn": [
        {"clf__n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21], "clf__weights": ["uniform", "distance"]}
    ],
    "dt": [{"clf__max_depth": [2, 3, 4, 6, 8, None], "clf__min_samples_leaf": [1, 3, 5, 10]}],
    "rf": [
        {
            "clf__max_depth": [3, 5, 8, None],
            "clf__max_features": ["sqrt", "log2"],
            "clf__min_samples_leaf": [1, 3],
        }
    ],
    "xgb": [
        {
            "clf__max_depth": [2, 3, 4, 6],
            "clf__n_estimators": [60, 100, 200],
            "clf__learning_rate": [0.05, 0.1, 0.2],
        }
    ],
}


def available_models() -> list[str]:
    names = ["logreg", "svm", "knn", "dt", "rf"]
    if importlib.util.find_spec("xgboost") is not None:
        names.append("xgb")
    return names


def _classifier(name: str, seed: int, n_jobs: int) -> BaseEstimator:
    if name == "logreg":
        return LogisticRegression(max_iter=5000, class_weight="balanced")
    if name == "svm":
        # Platt scaling via CalibratedClassifierCV (SVC(probability=True) is deprecated in 1.9)
        return CalibratedClassifierCV(
            SVC(class_weight="balanced", random_state=seed), method="sigmoid", cv=3, ensemble=False
        )
    if name == "knn":
        return KNeighborsClassifier()
    if name == "dt":
        return DecisionTreeClassifier(class_weight="balanced", random_state=seed)
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample", random_state=seed, n_jobs=n_jobs
        )
    if name == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "model 'xgb' needs the optional dependency: pip install xgboost"
            ) from exc
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=n_jobs,
            random_state=seed,
            verbosity=0,
        )
    raise ValueError(f"Unknown model {name!r}; choose from {sorted(_GRIDS)}")


def make_pipeline(
    name: str, seed: int = 42, pca_variance: float | None = None, n_jobs: int = 1
) -> Pipeline:
    steps: list[tuple[str, Any]] = [("scale", StandardScaler())]
    if pca_variance is not None:
        steps.append(("pca", PCA(n_components=pca_variance, svd_solver="full", random_state=seed)))
    steps.append(("clf", _classifier(name, seed, n_jobs)))
    return Pipeline(steps)


def param_grid(name: str) -> list[dict[str, list[Any]]]:
    if name not in _GRIDS:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(_GRIDS)}")
    return copy.deepcopy(_GRIDS[name])
