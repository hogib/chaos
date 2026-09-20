"""Wrapper functions that compute per-window features of a signal segment."""

import numpy as np

from chaos.chaos_algorithms import (corrdim_core, rosenstein_lye_core,
                                    samp_ent_core, wolf_lye_core)


def compute_raw_stats(segment: np.ndarray) -> dict:
    """Computes basic statistical metrics of the raw signal.

    Args:
        segment: Raw signal window (1-D ndarray).

    Returns:
        Dictionary with keys ``ham_mean``, ``ham_std``, ``ham_min``,
        ``ham_max`` and ``aktivite_std``. Values are ``nan`` if the input
        cannot be processed.
    """
    result = {
        "ham_mean": np.nan,
        "ham_std": np.nan,
        "ham_min": np.nan,
        "ham_max": np.nan,
        "aktivite_std": np.nan,
    }
    try:
        if np.any(np.isnan(segment)):
            result["ham_mean"] = float(np.nanmean(segment))
            result["ham_std"] = float(np.nanstd(segment, ddof=1))
            result["ham_min"] = float(np.nanmin(segment))
            result["ham_max"] = float(np.nanmax(segment))
        else:
            result["ham_mean"] = float(np.mean(segment))
            result["ham_std"] = float(np.std(segment, ddof=1))
            result["ham_min"] = float(np.min(segment))
            result["ham_max"] = float(np.max(segment))
        result["aktivite_std"] = result["ham_std"]
    except Exception:
        pass
    return result


def compute_norm_stats(segment: np.ndarray) -> dict:
    """Computes min/max of the z-score-normalized signal.

    Args:
        segment: Raw signal window (1-D ndarray).

    Returns:
        Dictionary with keys ``norm_min`` and ``norm_max``. Values are ``nan``
        if the segment has zero standard deviation or cannot be processed.
    """
    result = {"norm_min": np.nan, "norm_max": np.nan}
    try:
        seg_std = float(np.std(segment, ddof=1))
        if seg_std == 0:
            return result
        seg_norm = (segment - np.mean(segment)) / seg_std
        result["norm_min"] = float(np.min(seg_norm))
        result["norm_max"] = float(np.max(seg_norm))
    except Exception:
        pass
    return result


def compute_lyapunov_wolf(
    segment: np.ndarray,
    fs: float,
    tau: int = 5,
    m: int = 5,
    evolve: int = 5,
    min_samples: int = 500,
) -> dict:
    """Computes the maximum Lyapunov exponent via the Wolf (1985) method.

    Args:
        segment: Normalized signal (1-D ndarray).
        fs: Sampling frequency in Hz.
        tau: Time delay.
        m: Embedding dimension.
        evolve: Number of evolution steps.
        min_samples: Minimum length below which ``nan`` is returned.

    Returns:
        Dictionary with key ``wolf_lye``.
    """
    result = {"wolf_lye": np.nan}
    if len(segment) < min_samples:
        return result
    try:
        _, val = wolf_lye_core(segment, fs, tau, m, evolve)
        result["wolf_lye"] = round(float(val), 5)
    except Exception:
        pass
    return result


def compute_lyapunov_rosenstein(
    segment: np.ndarray,
    fs: float,
    tau: int = 5,
    m: int = 5,
    slope_ros: list | None = None,
    mean_period: float = 1.0,
) -> dict:
    """Computes the short-term Lyapunov exponent via Rosenstein (1993).

    Args:
        segment: Normalized signal (1-D ndarray).
        fs: Sampling frequency in Hz.
        tau: Time delay.
        m: Embedding dimension.
        slope_ros: Slope window ``[short_start, short_end]`` in periods.
        mean_period: Reciprocal of the dominant frequency in seconds.

    Returns:
        Dictionary with key ``ros_short``.
    """
    if slope_ros is None:
        slope_ros = [0, 0.2]
    result = {"ros_short": np.nan}
    try:
        out = rosenstein_lye_core(segment, fs, tau, m, slope_ros, mean_period)
        result["ros_short"] = round(float(out[0]), 5)
    except Exception:
        pass
    return result


def compute_sample_entropy(
    segment: np.ndarray,
    m: int = 2,
    r: float = 0.2,
) -> dict:
    """Computes Sample Entropy of the signal.

    Args:
        segment: Normalized signal (1-D ndarray).
        m: Template length.
        r: Tolerance as a multiple of the signal's standard deviation.

    Returns:
        Dictionary with key ``samp_ent``.
    """
    result = {"samp_ent": np.nan}
    try:
        val = samp_ent_core(segment, m, r)
        if not np.isnan(val):
            result["samp_ent"] = round(val, 5)
    except Exception:
        pass
    return result


def compute_corr_dim(
    segment: np.ndarray,
    tau: int = 5,
    m: int = 5,
) -> dict:
    """Computes the correlation dimension of the signal.

    Args:
        segment: Normalized signal (1-D ndarray).
        tau: Time delay.
        m: Embedding dimension.

    Returns:
        Dictionary with key ``corr_dim``.
    """
    result = {"corr_dim": np.nan}
    try:
        result["corr_dim"] = round(corrdim_core(segment, tau, m), 5)
    except Exception:
        pass
    return result
