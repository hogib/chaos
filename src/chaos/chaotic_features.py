"""Wrapper functions that compute per-window features of a signal segment."""

import numpy as np

from chaos.chaos_algorithms import (corrdim_core, fit_log_divergence,
                                    rosenstein_lye_core, samp_ent_core,
                                    wolf_lye_core)

R2_WARNING_THRESHOLD = 0.8
MIN_FIT_POINTS = 3

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

def split_slope_windows(slope_ros):
    """Splits a configured Rosenstein slope spec into its fit windows.

    Args:
        slope_ros: Either ``[short_start, short_end]`` or
            ``[short_start, short_end, long_start, long_end]``, in mean periods.

    Returns:
        Tuple ``(short, long)`` of ``(start, end)`` pairs; ``long`` is ``None``
        when only a short window was supplied.

    Raises:
        ValueError: If the spec does not have exactly two or four entries.
    """
    values = [float(v) for v in slope_ros]
    if len(values) == 2:
        return (values[0], values[1]), None
    if len(values) == 4:
        return (values[0], values[1]), (values[2], values[3])
    raise ValueError(
        "rosenstein slope must have 2 entries [short_start, short_end] or 4 "
        f"entries [short_start, short_end, long_start, long_end]; got {values}"
    )


def compute_lyapunov_rosenstein(
    segment: np.ndarray,
    fs: float,
    mean_period: float,
    tau: int = 5,
    m: int = 4,
    slope_ros: list | None = None,
) -> dict:
    """Short- and long-term Lyapunov exponents via Rosenstein, with fit quality.

    The divergence curve is computed once and then fitted over each configured
    window. Both windows are expressed in *mean periods*, matching
    :func:`chaos.chaos_algorithms.fit_log_divergence`, so the exponents are in
    units of 1/period.

    ``ros_low_fit_quality`` is set when the short-window fit rests on fewer
    than :data:`MIN_FIT_POINTS` samples or when its R-squared falls below
    :data:`R2_WARNING_THRESHOLD`. A two-point window always reports
    R-squared 1.0 — that is an artefact of fitting a line through two points,
    not a good fit, hence the separate point-count check.

    Args:
        segment: Normalized signal (1-D ndarray).
        fs: Sampling frequency in Hz.
        mean_period: Reciprocal of the power-spectrum-weighted mean frequency,
            in seconds.
        tau: Time delay.
        m: Embedding dimension.
        slope_ros: Fit windows in mean periods; see
            :func:`split_slope_windows`.

    Returns:
        Dictionary with keys ``ros_short``, ``ros_r2``, ``ros_n_points``,
        ``ros_low_fit_quality``, ``ros_long``, ``ros_long_r2`` and
        ``ros_long_n_points``.
    """
    if slope_ros is None:
        slope_ros = [0.2, 4.0]
    short_window, long_window = split_slope_windows(slope_ros)

    result = {
        "ros_short": np.nan,
        "ros_r2": np.nan,
        "ros_n_points": 0,
        "ros_low_fit_quality": False,
        "ros_long": np.nan,
        "ros_long_r2": np.nan,
        "ros_long_n_points": 0,
    }
    try:
        _, out_matrix = rosenstein_lye_core(
            segment, fs, tau, m, slope_ros, mean_period
        )
        ave_ln_div = out_matrix[2]

        lye, r2, n_points = fit_log_divergence(
            ave_ln_div, fs, mean_period, *short_window
        )
        result["ros_short"] = round(lye, 5) if np.isfinite(lye) else np.nan
        result["ros_r2"] = round(r2, 5) if np.isfinite(r2) else np.nan
        result["ros_n_points"] = n_points
        result["ros_low_fit_quality"] = bool(
            n_points < MIN_FIT_POINTS
            or (np.isfinite(r2) and r2 < R2_WARNING_THRESHOLD)
        )

        if long_window is not None:
            lye_l, r2_l, n_l = fit_log_divergence(
                ave_ln_div, fs, mean_period, *long_window
            )
            result["ros_long"] = round(lye_l, 5) if np.isfinite(lye_l) else np.nan
            result["ros_long_r2"] = round(r2_l, 5) if np.isfinite(r2_l) else np.nan
            result["ros_long_n_points"] = n_l
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
