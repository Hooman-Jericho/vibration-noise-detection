"""Engine-level, leakage-free evaluation.

* The statistical unit is the **physical engine**, never the window: every split is grouped by
  engine id.
* Hyper-parameters are tuned **inside** each outer fold (nested CV), so the held-out engine never
  influences model selection.
* Window probabilities are aggregated per engine (soft mean-probability vote and hard majority
  vote). Ties go to NG because a missed fault costs more than a false alarm.
* Uncertainty is reported (Wilson interval) and an optional label-permutation test answers
  "is this better than chance with only N engines?".
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_score,
)

from .config import Config
from .models import make_pipeline, param_grid

log = logging.getLogger(__name__)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def engine_decision(p_ng: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    """Aggregate window-level P(NG) of one engine. Ties are resolved towards NG."""
    p = np.asarray(p_ng, dtype=float)
    frac = float(np.mean(p >= threshold))
    p_mean = float(p.mean())
    return {
        "n_windows": int(p.size),
        "p_ng_mean": p_mean,
        "frac_windows_ng": frac,
        "pred_soft": int(p_mean >= threshold),
        "pred_hard": int(frac >= 0.5),
    }


def make_group_splits(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    """Stratified, group-aware splits; ``n_splits`` is capped by the rarest class' engine count."""
    per_engine = pd.Series(y).groupby(np.asarray(groups)).first()
    k = min(n_splits, int(per_engine.value_counts().min()))
    if per_engine.nunique() < 2 or k < 2:
        raise ValueError(
            "Need at least 2 engines in each class for grouped cross-validation; "
            f"engines per class: {per_engine.value_counts().to_dict()}"
        )
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    return list(sgkf.split(X, y, groups)), k


@dataclass
class EvalResult:
    model: str
    engines: pd.DataFrame
    windows: pd.DataFrame
    metrics: dict[str, Any]
    best_params: list[dict[str, Any]]


def summarize(engines: pd.DataFrame, windows: pd.DataFrame, threshold: float) -> dict[str, Any]:
    y = engines["y_true"].to_numpy()
    pred = engines["pred_soft"].to_numpy()
    n = len(y)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else math.nan  # recall on faulty engines
    spec = tn / (tn + fp) if (tn + fp) else math.nan
    lo, hi = wilson_interval(int(tp + tn), n)
    wy = windows["y_true"].to_numpy()
    wp = windows["p_ng"].to_numpy()
    wpred = (wp >= threshold).astype(int)
    w_tn, w_fp, w_fn, w_tp = confusion_matrix(wy, wpred, labels=[0, 1]).ravel()
    w_sens = w_tp / (w_tp + w_fn) if (w_tp + w_fn) else math.nan
    w_spec = w_tn / (w_tn + w_fp) if (w_tn + w_fp) else math.nan
    two_classes = len(set(y.tolist())) == 2
    return {
        "n_engines": int(n),
        "n_windows": int(len(windows)),
        "engine": {
            "accuracy": (tp + tn) / n,
            "accuracy_ci95": [lo, hi],
            "balanced_accuracy": float(np.nanmean([sens, spec])),
            "sensitivity_ng": sens,
            "specificity_ok": spec,
            "auc": roc_auc_score(y, engines["p_ng_mean"]) if two_classes else math.nan,
            "accuracy_hard_vote": float(np.mean(engines["pred_hard"].to_numpy() == y)),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        },
        "window": {
            "accuracy": float(np.mean(wpred == wy)),
            "balanced_accuracy": float(np.nanmean([w_sens, w_spec])),
            "auc": roc_auc_score(wy, wp) if len(set(wy.tolist())) == 2 else math.nan,
        },
    }


def nested_loeo(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, model_name: str, cfg: Config
) -> EvalResult:
    """Nested Leave-One-Engine-Out: inner grouped CV tunes, the held-out engine only evaluates."""
    pipe = make_pipeline(model_name, cfg.model.seed, cfg.model.pca_variance, cfg.model.n_jobs)
    grid = param_grid(model_name)
    engines = sorted(set(groups.tolist()))
    w_rows: list[dict[str, Any]] = []
    e_rows: list[dict[str, Any]] = []
    best: list[dict[str, Any]] = []

    for i, g in enumerate(engines, 1):
        test = groups == g
        train = ~test
        splits, k = make_group_splits(
            X[train], y[train], groups[train], cfg.eval.inner_splits, cfg.model.seed
        )
        gs = GridSearchCV(
            pipe,
            grid,
            cv=splits,
            scoring=cfg.eval.scoring,
            n_jobs=cfg.model.n_jobs,
            refit=True,
            error_score="raise",
        )
        gs.fit(X[train], y[train])
        p = gs.predict_proba(X[test])[:, 1]
        y_true = int(y[test][0])
        dec = engine_decision(p, cfg.eval.threshold)
        e_rows.append({"engine_id": g, "y_true": y_true, **dec})
        w_rows += [{"engine_id": g, "y_true": y_true, "p_ng": float(v)} for v in p]
        best.append(
            {
                "engine_id": g,
                "params": gs.best_params_,
                "inner_score": float(gs.best_score_),
                "inner_splits": k,
            }
        )
        log.info(
            "[%d/%d] %s %-8s true=%d p_ng=%.3f -> %s",
            i,
            len(engines),
            model_name,
            g,
            y_true,
            dec["p_ng_mean"],
            "NG" if dec["pred_soft"] else "OK",
        )

    engines_df = pd.DataFrame(e_rows)
    windows_df = pd.DataFrame(w_rows)
    return EvalResult(
        model_name,
        engines_df,
        windows_df,
        summarize(engines_df, windows_df, cfg.eval.threshold),
        best,
    )


def permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_name: str,
    cfg: Config,
    n_perm: int,
    observed: float,
) -> dict[str, Any]:
    """Shuffle labels *between engines* and redo the whole nested evaluation.

    Returns a one-sided p-value for engine-level balanced accuracy: the probability of doing
    at least as well by chance given this exact number of engines.
    """
    rng = np.random.default_rng(cfg.model.seed)
    engines = np.array(sorted(set(groups.tolist())))
    labels = np.array([int(y[groups == g][0]) for g in engines])
    null: list[float] = []
    for i in range(n_perm):
        mapping = dict(zip(engines, rng.permutation(labels), strict=True))
        y_perm = np.array([mapping[g] for g in groups])
        res = nested_loeo(X, y_perm, groups, model_name, cfg)
        null.append(res.metrics["engine"]["balanced_accuracy"])
        log.info("permutation %d/%d -> balanced accuracy %.3f", i + 1, n_perm, null[-1])
    p_value = (1 + sum(v >= observed for v in null)) / (n_perm + 1)
    return {
        "n_permutations": n_perm,
        "observed": observed,
        "null_mean": float(np.mean(null)),
        "null_max": float(np.max(null)),
        "p_value": float(p_value),
    }


def leakage_audit(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, model_name: str, cfg: Config
) -> dict[str, float]:
    """Quantify how much a naive window-level K-fold CV would over-estimate performance.

    Both numbers use the same untuned model; the only difference is whether windows of one
    engine may be split across train and validation folds.
    """
    pipe = make_pipeline(model_name, cfg.model.seed, cfg.model.pca_variance, cfg.model.n_jobs)
    naive_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.model.seed)
    naive = cross_val_score(pipe, X, y, cv=naive_cv, scoring="balanced_accuracy").mean()
    splits, _ = make_group_splits(X, y, groups, 5, cfg.model.seed)
    grouped = cross_val_score(pipe, X, y, cv=splits, scoring="balanced_accuracy").mean()
    return {
        "naive_window_cv": float(naive),
        "grouped_cv": float(grouped),
        "inflation": float(naive - grouped),
    }
