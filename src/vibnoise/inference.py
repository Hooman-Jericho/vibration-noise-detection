"""Inference on unseen recordings with per-engine aggregation (fixes the original predict loop)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .dataset import build_feature_table
from .evaluation import engine_decision
from .io import discover_recordings
from .training import Bundle, check_features


def predict_recordings(
    bundle: Bundle, path: str | Path, threshold: float | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(engines, windows)`` data frames for every engine found under ``path``.

    Each engine is judged **only from its own windows** (the original loop re-predicted the whole
    test set for every engine). The decision is NG if mean P(NG) >= threshold.
    """
    cfg = bundle.config
    thr = cfg.eval.threshold if threshold is None else threshold
    recs = discover_recordings(path, cfg.data, labeled=False)
    table = build_feature_table(recs, cfg)
    check_features(bundle, table)
    p = bundle.estimator.predict_proba(table.X)[:, 1]

    windows = pd.DataFrame(
        {
            "engine_id": table.groups,
            "recording": table.recording,
            "interval": table.interval,
            "window": table.window,
            "p_ng": p,
        }
    )
    rows = []
    for eid, sub in windows.groupby("engine_id", sort=True):
        dec = engine_decision(sub["p_ng"].to_numpy(), thr)
        rows.append(
            {
                "engine_id": eid,
                "recordings": ";".join(sorted(sub["recording"].unique())),
                "n_windows": dec["n_windows"],
                "p_ng_mean": dec["p_ng_mean"],
                "frac_windows_ng": dec["frac_windows_ng"],
                "decision": "NG" if dec["pred_soft"] else "OK",
                "decision_hard_vote": "NG" if dec["pred_hard"] else "OK",
            }
        )
    return pd.DataFrame(rows), windows
