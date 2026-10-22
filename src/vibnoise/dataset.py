"""Recordings -> windows -> feature table with explicit group (engine) bookkeeping."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .config import Config
from .features import extract_window, feature_names
from .io import Recording, read_recording
from .signal import highpass, iter_windows, resolve_intervals

log = logging.getLogger(__name__)


@dataclass
class FeatureTable:
    X: np.ndarray  # (n_windows, n_features)
    y: np.ndarray | None  # (n_windows,) 0 = OK, 1 = NG; None for unlabeled data
    groups: np.ndarray  # engine id of every window - the unit of independence
    recording: np.ndarray  # recording id of every window
    interval: np.ndarray  # operating-interval index of every window
    window: np.ndarray  # window index inside the recording
    feature_names: list[str]

    def __len__(self) -> int:
        return len(self.X)

    @property
    def engines(self) -> list[str]:
        return sorted(set(self.groups.tolist()))

    def engine_labels(self) -> dict[str, int]:
        if self.y is None:
            raise ValueError("table has no labels")
        out: dict[str, int] = {}
        for g, label in zip(self.groups, self.y, strict=True):
            if out.setdefault(str(g), int(label)) != int(label):
                raise ValueError(f"engine {g} has inconsistent labels")
        return out


def build_feature_table(recordings: list[Recording], cfg: Config) -> FeatureTable:
    fs = cfg.data.fs
    trim_n = int(round(cfg.filter.trim_s * fs))
    win_n = None if cfg.segment.window_s is None else int(round(cfg.segment.window_s * fs))
    hop_n = None if cfg.segment.hop_s is None else int(round(cfg.segment.hop_s * fs))
    expected = 1 if cfg.segment.intervals is None else len(cfg.segment.intervals)

    rows: list[np.ndarray] = []
    labels: list[int | None] = []
    groups: list[str] = []
    rec_ids: list[str] = []
    interval_ids: list[int] = []
    window_ids: list[int] = []

    for rec in recordings:
        x = read_recording(rec.path, cfg.data)
        xf = highpass(x, fs, cfg.filter.cutoff_hz, cfg.filter.order, cfg.filter.zero_phase)
        ivs = resolve_intervals(len(xf), cfg.segment.intervals, trim_n)
        if len(ivs) < expected:
            log.warning(
                "%s: only %d of %d configured intervals fit in the recording (%d samples)",
                rec.recording_id,
                len(ivs),
                expected,
                len(xf),
            )
        w = 0
        for iv_idx, a, b in ivs:
            spans = list(iter_windows(a, b, win_n, hop_n))
            if not spans:
                log.warning(
                    "%s: interval %d (%d samples) is shorter than one window - skipped",
                    rec.recording_id,
                    iv_idx,
                    b - a,
                )
            for s, e in spans:
                rows.append(extract_window(xf[s:e], fs, cfg.features))
                labels.append(rec.label)
                groups.append(rec.engine_id)
                rec_ids.append(rec.recording_id)
                interval_ids.append(iv_idx)
                window_ids.append(w)
                w += 1
        log.info("%s -> %d windows", rec.recording_id, w)

    if not rows:
        raise ValueError("No windows were produced; check segment/filter/feature settings")

    y = None if any(v is None for v in labels) else np.asarray(labels, dtype=int)
    return FeatureTable(
        X=np.vstack(rows),
        y=y,
        groups=np.asarray(groups),
        recording=np.asarray(rec_ids),
        interval=np.asarray(interval_ids),
        window=np.asarray(window_ids),
        feature_names=feature_names(fs, cfg.data.sensors, cfg.features),
    )
