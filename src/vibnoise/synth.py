"""Synthetic compressor recordings so the whole pipeline is runnable and testable without the
(private) real data set.

Healthy machines: harmonics of a rotation frequency + broadband noise.
Faulty machines: additional narrow-band resonance (~180 Hz) and amplitude modulation, scaled by
``severity``. Every *engine* also gets its own random gain / DC offset / drift - a realistic
nuisance "fingerprint" that makes naive window-level cross-validation look better than it is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.signal import butter, sosfiltfilt

ORDINALS = {1: "1st", 2: "2nd", 3: "3rd"}


def _ordinal(n: int) -> str:
    return ORDINALS.get(n, f"{n}th")


def _recording(
    rng: np.random.Generator,
    fs: float,
    interval_samples: list[int],
    faulty: bool,
    severity: float,
    gain: float,
    f0_shift: float,
    offset: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bp = butter(4, [170, 190], btype="bandpass", fs=fs, output="sos")
    s1_parts, s2_parts = [], []
    for k, n in enumerate(interval_samples):
        t = np.arange(n) / fs
        f0 = 25.0 + 6.0 * k + f0_shift
        phase = rng.uniform(0, 2 * np.pi, size=3)
        base = sum(
            a * np.sin(2 * np.pi * (h + 1) * f0 * t + phase[h])
            for h, a in enumerate((1.0, 0.5, 0.25))
        )
        mod = 1.0 + (0.3 * severity * np.sin(2 * np.pi * 7.0 * t) if faulty else 0.0)
        noise = rng.normal(0, 0.5, size=n)
        res = sosfiltfilt(bp, rng.normal(0, 1.0, size=n)) * (0.9 * severity) if faulty else 0.0
        s1_parts.append(gain * (base * mod + noise + res))
        s2_parts.append(
            0.8 * gain * (np.roll(base, 37) * mod + rng.normal(0, 0.6, size=n) + 0.7 * res)
        )
    total = sum(interval_samples)
    t_all = np.arange(total) / fs
    drift = 2.0 * np.sin(2 * np.pi * 0.2 * t_all)  # slow drift + DC to exercise the high-pass
    return (
        t_all,
        np.concatenate(s1_parts) + offset + drift,
        np.concatenate(s2_parts) - offset + drift,
    )


def generate_dataset(
    out_dir: str | Path,
    engines_per_class: int = 6,
    recordings_per_engine: int = 2,
    fs: float = 1000.0,
    interval_seconds: tuple[float, ...] = (12, 10, 8, 8, 6, 6, 6),
    severity: float = 1.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Write ``OK_clean/`` and ``NG_clean/`` CSV folders; returns the suggested config dict."""
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    sizes = [int(round(s * fs)) for s in interval_seconds]
    for state, folder, faulty in (("OK", "OK_clean", False), ("NG", "NG_clean", True)):
        (out / folder).mkdir(parents=True, exist_ok=True)
        for unit in range(1, engines_per_class + 1):
            gain = float(rng.lognormal(0.0, 0.15))
            f0_shift = float(rng.normal(0.0, 1.5))
            offset = float(rng.normal(0.0, 1.0))
            for rep in range(1, recordings_per_engine + 1):
                t, s1, s2 = _recording(rng, fs, sizes, faulty, severity, gain, f0_shift, offset)
                serial = int(rng.integers(10000, 99999))
                name = f"DAS{serial}_{state}_{unit}_{_ordinal(rep)}.CSV"
                pd.DataFrame({"time": t, "sensor_1": s1, "sensor_2": s2}).to_csv(
                    out / folder / name, index=False
                )
    bounds = np.concatenate([[0], np.cumsum(sizes)]).tolist()
    suggested = {
        "data": {"fs": fs},
        "segment": {"intervals": [[bounds[i], bounds[i + 1]] for i in range(len(sizes))]},
    }
    with open(out / "synth_config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(suggested, fh, sort_keys=False)
    return suggested
