"""Feature extraction. Everything is explicit and sensor-aligned *by construction*.

Differences to the original: PSD is band-limited and uses ``log10(P + eps)`` (never ``-inf``),
statistical features are bounded and NaN-free (no ``log`` of a possibly negative kurtosis), the
mis-named "wavelet_packet_energy" (it was an RMS) is called what it is, and there is a low
dimensional ``bands`` feature to fight the p >> n problem of raw PSD bins.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, skew

from .config import FeatureConfig


def _welch(x: np.ndarray, fs: float, cfg: FeatureConfig) -> tuple[np.ndarray, np.ndarray]:
    if len(x) < cfg.nperseg:
        raise ValueError(
            f"window has {len(x)} samples but features.nperseg={cfg.nperseg}; "
            "use longer windows or a smaller nperseg"
        )
    return welch(x, fs=fs, nperseg=cfg.nperseg, detrend="constant", scaling="density")


def _band_mask(f: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    fmax = f[-1] if cfg.fmax_hz is None else cfg.fmax_hz
    return (f >= cfg.fmin_hz) & (f <= fmax)


def psd_vector(x: np.ndarray, fs: float, cfg: FeatureConfig) -> np.ndarray:
    f, p = _welch(x, fs, cfg)
    return np.log10(p[_band_mask(f, cfg)] + cfg.eps)


def bands_vector(x: np.ndarray, fs: float, cfg: FeatureConfig) -> np.ndarray:
    f, p = _welch(x, fs, cfg)
    df = f[1] - f[0]
    return np.array(
        [np.log10(p[(f >= lo) & (f < hi)].sum() * df + cfg.eps) for lo, hi in cfg.bands_hz]
    )


def stats_vector(x: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(np.max(np.abs(x)))
    sk = float(skew(x)) if np.std(x) > 0 else 0.0
    ku = float(kurtosis(x)) if np.std(x) > 0 else 0.0  # excess kurtosis (can be negative)
    return np.array(
        [np.log10(rms + cfg.eps), np.log10(peak + cfg.eps), peak / (rms + cfg.eps), sk, ku]
    )


def _psd_freqs(fs: float, cfg: FeatureConfig) -> np.ndarray:
    f = np.fft.rfftfreq(cfg.nperseg, 1.0 / fs)
    return f[_band_mask(f, cfg)]


def feature_names(fs: float, sensors: list[str], cfg: FeatureConfig) -> list[str]:
    names: list[str] = []
    for s in sensors:
        for kind in cfg.names:
            if kind == "psd":
                names += [f"{s}:psd:{f:.1f}Hz" for f in _psd_freqs(fs, cfg)]
            elif kind == "bands":
                names += [f"{s}:band:{lo:g}-{hi:g}Hz" for lo, hi in cfg.bands_hz]
            elif kind == "stats":
                names += [f"{s}:stat:{n}" for n in ("log_rms", "log_peak", "crest", "skew", "kurt")]
    return names


def extract_window(window: np.ndarray, fs: float, cfg: FeatureConfig) -> np.ndarray:
    """``window`` has shape ``(n_samples, n_sensors)``; returns a 1-D feature vector."""
    parts: list[np.ndarray] = []
    for j in range(window.shape[1]):
        x = window[:, j]
        for kind in cfg.names:
            if kind == "psd":
                parts.append(psd_vector(x, fs, cfg))
            elif kind == "bands":
                parts.append(bands_vector(x, fs, cfg))
            elif kind == "stats":
                parts.append(stats_vector(x, cfg))
            else:  # guarded by Config.validate, kept for direct callers
                raise ValueError(f"unknown feature {kind!r}")
    vec = np.concatenate(parts)
    if not np.isfinite(vec).all():
        raise ValueError("non-finite feature value produced")
    return vec
