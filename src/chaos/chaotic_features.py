"""Wrapper functions that compute per-window features of a signal segment."""

import numpy as np

from chaos.chaos_algorithms import (corrdim_core, rosenstein_lye_core,
                                    samp_ent_core, wolf_lye_core)

R2_WARNING_THRESHOLD = 0.8

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

def _fit_slope_with_r2(time_sec, ave_ln_div, start_s, end_s, fs):
    """Same lo/hi rule as rosenstein_lye_core's internal fit, plus R^2 and point count."""
    nz = int(np.count_nonzero(ave_ln_div))
    lo = 0 if start_s == 0 else round(start_s * fs)
    hi = round(end_s * fs)
    if hi > nz or hi <= lo:
        return np.nan, np.nan, 0

    x = time_sec[lo:hi + 1]
    y = ave_ln_div[lo:hi + 1]
    if len(x) < 2:
        return np.nan, np.nan, len(x)

    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(slope), r2, len(x)

def compute_lyapunov_rosenstein(
    segment: np.ndarray,
    fs: float,
    mean_period: float,
    tau: int = 5,
    m: int = 4,
    slope_ros: list | None = None,
) -> dict:
    """Short-term Lyapunov exponent via Rosenstein, with fit quality.

    The primary output key is ``ros_short`` so it flows directly into the
    feature CSV (which filters rows through ``FEATURE_KEYS``). Extra keys
    ``ros_r2``, ``ros_n_points`` and ``ros_low_fit_quality`` are also returned
    — add them to ``FEATURE_KEYS`` in ``chaos/extraction.py`` if you want them
    written to disk.

    Args:
        segment: Normalized signal (1-D ndarray).
        fs: Sampling frequency in Hz.
        mean_period: Reciprocal of the power-spectrum-weighted mean frequency,
            in seconds.
        tau: Time delay.
        m: Embedding dimension.
        slope_ros: ``[short_start_s, short_end_s]`` fit window in seconds.

    Returns:
        Dictionary with keys ``ros_short``, ``ros_r2``, ``ros_n_points`` and
        ``ros_low_fit_quality``.
    """
    if slope_ros is None:
        slope_ros = [0.2, 4.0]

    result = {
        "ros_short": np.nan,
        "ros_r2": np.nan,
        "ros_n_points": 0,
        "ros_low_fit_quality": False,
    }
    try:
        _, out_matrix = rosenstein_lye_core(
            segment, fs, tau, m, slope_ros, mean_period
        )
        ave_ln_div = out_matrix[2]
        time_sec = np.arange(len(ave_ln_div)) / fs
        lye, r2, n_points = _fit_slope_with_r2(
            time_sec, ave_ln_div, slope_ros[0], slope_ros[1], fs
        )
        result["ros_short"] = round(lye, 5) if np.isfinite(lye) else np.nan
        result["ros_r2"] = round(r2, 5) if np.isfinite(r2) else np.nan
        result["ros_n_points"] = n_points
        result["ros_low_fit_quality"] = bool(
            np.isfinite(r2) and r2 < R2_WARNING_THRESHOLD
        )
    except Exception as e:
        print(f"[rosenstein] {type(e).__name__}: {e}")
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
