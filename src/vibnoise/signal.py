"""Signal conditioning and segmentation."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt


def highpass(
    x: np.ndarray, fs: float, cutoff_hz: float, order: int = 8, zero_phase: bool = True
) -> np.ndarray:
    """Butterworth high-pass (second-order sections, numerically stable).

    ``zero_phase=True`` uses forward-backward filtering: no phase distortion and no
    start-up transient (the original causal order-20 filter left a transient in the first
    interval). Note that forward-backward filtering squares the magnitude response, i.e.
    the effective order is ``2 * order``.
    """
    x = np.asarray(x, dtype=float)
    if not 0 < cutoff_hz < fs / 2:
        raise ValueError(f"cutoff_hz must be in (0, {fs / 2}); got {cutoff_hz}")
    sos = butter(order, cutoff_hz, btype="highpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=0) if zero_phase else sosfilt(sos, x, axis=0)


def resolve_intervals(
    n: int, intervals: Sequence[Sequence[int | None]] | None, trim_n: int = 0
) -> list[tuple[int, int, int]]:
    """Turn user intervals into ``(index, start, end)`` clipped to ``[trim_n, n - trim_n]``.

    ``None`` end means "to the end", negative values count from the end (Python style, so the
    original ``-1`` keeps meaning "last sample"). Intervals that fall outside the recording
    are dropped (the caller decides whether that is an error).
    """
    lo, hi = trim_n, n - trim_n
    spec: Sequence[Sequence[int | None]] = intervals if intervals is not None else [[0, None]]
    out: list[tuple[int, int, int]] = []
    for idx, pair in enumerate(spec):
        if len(pair) != 2:
            raise ValueError(f"interval {idx} must be [start, end]; got {pair!r}")
        a = 0 if pair[0] is None else int(pair[0])
        b = n if pair[1] is None else int(pair[1])
        a = a + n if a < 0 else a
        b = b + n if b < 0 else b
        a, b = max(a, lo), min(b, hi)
        if b > a:
            out.append((idx, a, b))
    return out


def iter_windows(
    start: int, end: int, win: int | None, hop: int | None
) -> Iterator[tuple[int, int]]:
    """Yield full-length windows inside ``[start, end)``; ``win=None`` yields the whole span."""
    if win is None:
        yield start, end
        return
    hop = win if hop is None else hop
    s = start
    while s + win <= end:
        yield s, s + win
        s += hop
