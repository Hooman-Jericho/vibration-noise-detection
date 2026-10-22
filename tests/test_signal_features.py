import numpy as np
import pytest

from vibnoise.config import FeatureConfig
from vibnoise.features import extract_window, feature_names
from vibnoise.signal import highpass, iter_windows, resolve_intervals

FS = 1000.0


def _t(n):
    return np.arange(n) / FS


def test_highpass_removes_dc_and_keeps_passband():
    t = _t(20000)
    x = 5.0 + np.sin(2 * np.pi * 50 * t)
    y = highpass(x, FS, 10, order=8)
    mid = slice(2000, -2000)
    assert abs(y[mid].mean()) < 1e-3
    assert np.std(y[mid]) == pytest.approx(np.std(np.sin(2 * np.pi * 50 * t)), rel=0.02)


def test_zero_phase_has_no_start_transient_or_lag():
    t = _t(20000)
    x = np.sin(2 * np.pi * 40 * t)
    y = highpass(x, FS, 10, order=8, zero_phase=True)
    assert np.corrcoef(x[1000:-1000], y[1000:-1000])[0, 1] > 0.9999
    assert np.max(np.abs(y[:200])) < 1.2  # no ringing at the start


def test_highpass_rejects_bad_cutoff():
    with pytest.raises(ValueError):
        highpass(np.zeros(100), FS, 600)


def test_resolve_intervals_semantics():
    # original-style spec: last interval ends at -1 (python style: last sample)
    ivs = resolve_intervals(1000, [[0, 400], [400, 700], [700, -1]], trim_n=0)
    assert ivs == [(0, 0, 400), (1, 400, 700), (2, 700, 999)]
    # trimming clips the edges, out-of-range intervals disappear
    ivs = resolve_intervals(1000, [[0, 400], [5000, 6000]], trim_n=100)
    assert ivs == [(0, 100, 400)]
    # None = whole recording
    assert resolve_intervals(1000, None, trim_n=50) == [(0, 50, 950)]
    with pytest.raises(ValueError):
        resolve_intervals(10, [[1, 2, 3]])


def test_iter_windows():
    assert list(iter_windows(0, 10, 4, None)) == [(0, 4), (4, 8)]
    assert list(iter_windows(0, 10, 4, 2)) == [(0, 4), (2, 6), (4, 8), (6, 10)]
    assert list(iter_windows(0, 3, 4, None)) == []
    assert list(iter_windows(5, 9, None, None)) == [(5, 9)]


def test_features_match_names_and_are_finite_on_silence():
    cfg = FeatureConfig(names=["psd", "bands", "stats"], nperseg=256)
    window = np.zeros((1000, 2))  # the original code produced -inf here
    vec = extract_window(window, FS, cfg)
    assert np.isfinite(vec).all()
    assert len(vec) == len(feature_names(FS, ["a", "b"], cfg))


def test_psd_peak_is_at_the_right_frequency():
    cfg = FeatureConfig(names=["psd"], nperseg=500, fmin_hz=10, fmax_hz=450)
    x = np.sin(2 * np.pi * 100 * _t(5000))
    vec = extract_window(x[:, None], FS, cfg)
    names = feature_names(FS, ["s"], cfg)
    assert names[int(np.argmax(vec))] == "s:psd:100.0Hz"


def test_window_shorter_than_nperseg_is_an_error():
    with pytest.raises(ValueError, match="nperseg"):
        extract_window(np.zeros((100, 1)), FS, FeatureConfig(nperseg=256))


def test_constant_signal_stats_not_nan():
    cfg = FeatureConfig(names=["stats"])
    assert np.isfinite(extract_window(np.ones((500, 1)), FS, cfg)).all()
