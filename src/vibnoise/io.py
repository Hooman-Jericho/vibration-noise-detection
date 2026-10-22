"""Recording discovery and validated CSV reading.

Fixes over the original project: labels come from a *strict* folder->class map (no silent
"everything that is not OK is NG"), and the physical **engine id** is parsed from the file name
so that several recordings of the same engine always stay in the same cross-validation group.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DataConfig

log = logging.getLogger(__name__)

STATE_TO_LABEL = {"OK": 0, "NG": 1}


@dataclass(frozen=True)
class Recording:
    path: Path
    label: int | None
    engine_id: str
    recording_id: str


def parse_engine_id(filename: str, regex: str | None) -> tuple[str, str | None]:
    """Return ``(engine_id, state)``; ``state`` is "OK"/"NG" if the regex has a ``state`` group."""
    stem = Path(filename).stem
    if not regex:
        return stem, None
    m = re.search(regex, stem, flags=re.IGNORECASE)
    if m is None:
        raise ValueError(f"engine_regex {regex!r} does not match file name {filename!r}")
    groups = {k: v for k, v in m.groupdict().items() if v is not None}
    state = groups["state"].upper() if "state" in groups else None
    if groups:
        engine_id = "_".join(v.upper() if k == "state" else v for k, v in groups.items())
    else:
        engine_id = m.group(0)
    return engine_id, state


def _iter_csv(root: Path) -> Iterator[Path]:
    if root.is_file():
        yield root
        return
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() == ".csv":
            yield p


def discover_recordings(root: str | Path, cfg: DataConfig, labeled: bool = True) -> list[Recording]:
    """Find CSV recordings under ``root`` (a directory or a single file)."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Data path does not exist: {root}")
    base = root if root.is_dir() else root.parent
    recs: list[Recording] = []
    for p in _iter_csv(root):
        label: int | None = None
        if labeled:
            folder = p.parent.name
            if folder not in cfg.label_map:
                raise ValueError(
                    f"{p}: parent folder {folder!r} is not in data.label_map "
                    f"{sorted(cfg.label_map)}; refusing to guess a label."
                )
            label = cfg.label_map[folder]
            engine_id, state = parse_engine_id(p.name, cfg.engine_regex)
            if state is not None and STATE_TO_LABEL[state] != label:
                raise ValueError(
                    f"{p}: file name says {state} but folder {folder!r} maps to class {label}"
                )
        else:
            try:
                engine_id, _ = parse_engine_id(p.name, cfg.engine_regex)
            except ValueError:
                engine_id = p.stem  # unseen data: fall back to one group per file
        rid = p.relative_to(base).as_posix() if root.is_dir() else p.name
        recs.append(Recording(path=p, label=label, engine_id=engine_id, recording_id=rid))
    if not recs:
        raise FileNotFoundError(f"No .csv recordings found under {root}")
    if labeled:
        by_engine: dict[str, set[int | None]] = defaultdict(set)
        for r in recs:
            by_engine[r.engine_id].add(r.label)
        clash = [e for e, labs in by_engine.items() if len(labs) > 1]
        if clash:
            raise ValueError(
                f"Engine id(s) {clash} appear in both classes; check data.engine_regex"
            )
    return recs


def read_recording(path: str | Path, cfg: DataConfig) -> np.ndarray:
    """Read one CSV into an ``(n_samples, n_sensors)`` float array, validating its content."""
    df = pd.read_csv(path)
    missing = [c for c in cfg.sensors if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing column(s) {missing}; found {list(df.columns)}")
    x = df[list(cfg.sensors)].to_numpy(dtype=float)
    bad = ~np.isfinite(x)
    if bad.any():
        raise ValueError(f"{path}: {int(bad.sum())} non-finite sample(s) in sensor columns")
    if "time" in df.columns and len(df) > 2:
        dt = float(np.median(np.diff(df["time"].to_numpy(dtype=float))))
        if dt > 0 and abs(1.0 / dt - cfg.fs) / cfg.fs > 0.05:
            log.warning(
                "%s: time column implies %.1f Hz but data.fs=%.1f Hz", path, 1.0 / dt, cfg.fs
            )
    return x


def fingerprint(recs: list[Recording]) -> str:
    """SHA-256 over file names and contents: ties a trained model to the exact data it saw."""
    h = hashlib.sha256()
    for r in sorted(recs, key=lambda r: r.recording_id):
        h.update(r.recording_id.encode())
        with open(r.path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()
