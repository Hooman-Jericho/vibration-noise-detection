"""Typed, validated configuration (YAML in, dataclasses out).

Every tunable that used to be a magic number scattered over the code base
(cut-off, filter order, interval boundaries, PSD resolution, ...) lives here,
is documented, validated and stored next to every trained model.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

KNOWN_FEATURES = ("psd", "bands", "stats")


@dataclass
class DataConfig:
    fs: float = 1000.0  # sampling rate [Hz]
    sensors: list[str] = field(default_factory=lambda: ["sensor_1", "sensor_2"])
    # sub-folder name -> class (0 = healthy / OK, 1 = faulty / NG). Anything else is an error.
    label_map: dict[str, int] = field(default_factory=lambda: {"OK_clean": 0, "NG_clean": 1})
    # Regex applied to the file stem. Named groups are joined to form the physical engine id,
    # e.g. "DAS64100_OK_2_1st" -> "OK_2". Set to null to use the file stem as engine id.
    engine_regex: str | None = r"(?P<state>OK|NG)_(?P<unit>\d+)"


@dataclass
class FilterConfig:
    cutoff_hz: float = 10.0
    order: int = 8
    zero_phase: bool = True  # sosfiltfilt (no phase distortion, no start-up transient)
    trim_s: float = 1.0  # discard this much at both ends of the recording (edge effects)


@dataclass
class SegmentConfig:
    # [[start, end], ...] in samples (end=null -> end of recording; negative -> from the end).
    # null -> the whole recording is a single operating interval.
    intervals: list[list[int | None]] | None = None
    window_s: float | None = (
        10.0  # fixed-length windows inside each interval (null -> whole interval)
    )
    hop_s: float | None = None  # null -> non-overlapping


@dataclass
class FeatureConfig:
    names: list[str] = field(default_factory=lambda: ["psd"])  # psd | bands | stats
    nperseg: int = 512
    fmin_hz: float = 10.0
    fmax_hz: float | None = 450.0
    eps: float = 1e-12
    bands_hz: list[list[float]] = field(
        default_factory=lambda: [[10, 25], [25, 50], [50, 100], [100, 200], [200, 400]]
    )


@dataclass
class ModelConfig:
    name: str = "svm"
    pca_variance: float | None = (
        None  # optional PCA (fraction of variance kept) inside the pipeline
    )
    seed: int = 42
    n_jobs: int = 1


@dataclass
class EvalConfig:
    inner_splits: int = 3
    scoring: str = "balanced_accuracy"
    threshold: float = 0.5  # p(NG) >= threshold -> NG (ties go to the *unsafe* side on purpose)
    permutations: int = 0


_SECTIONS: dict[str, type] = {
    "data": DataConfig,
    "filter": FilterConfig,
    "segment": SegmentConfig,
    "features": FeatureConfig,
    "model": ModelConfig,
    "eval": EvalConfig,
}


def _section(cls: type, mapping: dict[str, Any] | None) -> Any:
    mapping = dict(mapping or {})
    known = {f.name for f in fields(cls)}
    unknown = set(mapping) - known
    if unknown:
        raise ValueError(
            f"Unknown key(s) {sorted(unknown)} in section for {cls.__name__}; "
            f"valid keys: {sorted(known)}"
        )
    return cls(**mapping)


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> Config:
        d = dict(d or {})
        unknown = set(d) - set(_SECTIONS)
        if unknown:
            raise ValueError(
                f"Unknown config section(s) {sorted(unknown)}; valid sections: {sorted(_SECTIONS)}"
            )
        cfg = cls(**{name: _section(_SECTIONS[name], d.get(name)) for name in _SECTIONS})
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        d, f, s, ft, m, e = (
            self.data,
            self.filter,
            self.segment,
            self.features,
            self.model,
            self.eval,
        )
        nyq = d.fs / 2
        if d.fs <= 0:
            raise ValueError("data.fs must be > 0")
        if not d.sensors:
            raise ValueError("data.sensors must not be empty")
        if not d.label_map or not set(d.label_map.values()) <= {0, 1}:
            raise ValueError("data.label_map values must be 0 (OK) or 1 (NG)")
        if not 0 < f.cutoff_hz < nyq:
            raise ValueError(f"filter.cutoff_hz must be in (0, {nyq})")
        if f.order < 1 or f.trim_s < 0:
            raise ValueError("filter.order must be >= 1 and filter.trim_s >= 0")
        if s.window_s is not None and s.window_s <= 0:
            raise ValueError("segment.window_s must be > 0 or null")
        if s.hop_s is not None and s.hop_s <= 0:
            raise ValueError("segment.hop_s must be > 0 or null")
        bad = [n for n in ft.names if n not in KNOWN_FEATURES]
        if bad or not ft.names:
            raise ValueError(
                f"features.names must be a non-empty subset of {KNOWN_FEATURES}; got {ft.names}"
            )
        if ft.nperseg < 16:
            raise ValueError("features.nperseg must be >= 16")
        if ft.fmax_hz is not None and not ft.fmin_hz < ft.fmax_hz <= nyq:
            raise ValueError(f"need features.fmin_hz < features.fmax_hz <= {nyq}")
        if not 0 < e.threshold < 1:
            raise ValueError("eval.threshold must be in (0, 1)")
        if e.inner_splits < 2:
            raise ValueError("eval.inner_splits must be >= 2")
        if m.pca_variance is not None and not 0 < m.pca_variance <= 1:
            raise ValueError("model.pca_variance must be in (0, 1]")


def apply_overrides(d: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``section.key=value`` overrides (value parsed as YAML) to a config dict."""
    out = copy.deepcopy(d)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like section.key=value, got {item!r}")
        key, raw = item.split("=", 1)
        parts = key.strip().split(".")
        if len(parts) != 2:
            raise ValueError(f"Override key must be section.key, got {key!r}")
        out.setdefault(parts[0], {})
        out[parts[0]][parts[1]] = yaml.safe_load(raw)
    return out


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    base: dict[str, Any] = {}
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            base = yaml.safe_load(fh) or {}
    return Config.from_dict(apply_overrides(base, overrides))
