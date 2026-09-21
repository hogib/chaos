"""Unit tests for chaos/chaotic_features.py — per-window wrappers."""

import inspect

import numpy as np
import pytest

from chaos.chaos_algorithms import fit_log_divergence
from chaos.chaotic_features import (MIN_FIT_POINTS, R2_WARNING_THRESHOLD,
                                    compute_corr_dim,
                                    compute_lyapunov_rosenstein,
                                    compute_lyapunov_wolf, compute_norm_stats,
                                    compute_raw_stats, compute_sample_entropy,
                                    split_slope_windows)

# ---------------------------------------------------------------------------
# Signature / contract regression tests
# ---------------------------------------------------------------------------

def test_rosenstein_signature_has_slope_ros():
    """Regression: extraction.py uses slope_ros=..., not slope=..."""
    sig = inspect.signature(compute_lyapunov_rosenstein)
    assert "slope_ros" in sig.parameters, (
        "compute_lyapunov_rosenstein must accept slope_ros"
    )


def test_rosenstein_signature_has_mean_period_positional():
    """mean_period is required and positional — the caller must supply it."""
    sig = inspect.signature(compute_lyapunov_rosenstein)
    mp = sig.parameters["mean_period"]
    assert mp.default is inspect.Parameter.empty, "mean_period must be required"


def test_r2_warning_threshold_defined():
    """Regression: the wrapper references R2_WARNING_THRESHOLD; it must exist."""
    assert isinstance(R2_WARNING_THRESHOLD, float)


# ---------------------------------------------------------------------------
# compute_raw_stats / compute_norm_stats
# ---------------------------------------------------------------------------

def test_raw_stats_keys(synthetic_segment):
    out = compute_raw_stats(synthetic_segment)
    assert set(out) == {
        "ham_mean", "ham_std", "ham_min", "ham_max", "aktivite_std",
    }


def test_raw_stats_values_correct():
    seg = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = compute_raw_stats(seg)
    assert out["ham_mean"] == pytest.approx(3.0)
    assert out["ham_min"] == 1.0
    assert out["ham_max"] == 5.0
    assert out["aktivite_std"] == out["ham_std"]


def test_raw_stats_handles_nan():
    seg = np.array([1.0, 2.0, np.nan, 4.0])
    out = compute_raw_stats(seg)
    assert np.isfinite(out["ham_mean"])


def test_norm_stats_min_max(synthetic_segment):
    out = compute_norm_stats(synthetic_segment)
    assert set(out) == {"norm_min", "norm_max"}
    assert out["norm_min"] < 0 < out["norm_max"]


def test_norm_stats_zero_std_returns_nan():
    seg = np.ones(100)
    out = compute_norm_stats(seg)
    assert np.isnan(out["norm_min"]) and np.isnan(out["norm_max"])


# ---------------------------------------------------------------------------
# compute_lyapunov_rosenstein — full contract
# ---------------------------------------------------------------------------

def test_rosenstein_returns_short_window_keys(synthetic_segment, sample_rate):
    out = compute_lyapunov_rosenstein(
        synthetic_segment, fs=sample_rate, mean_period=1.0,
        tau=5, m=4, slope_ros=[0.2, 4.0],
    )
    assert set(out) == {
        "ros_short", "ros_r2", "ros_n_points", "ros_low_fit_quality",
        "ros_long", "ros_long_r2", "ros_long_n_points",
    }


def test_rosenstein_two_element_slope_leaves_long_empty(synthetic_segment,
                                                        sample_rate):
    out = compute_lyapunov_rosenstein(
        synthetic_segment, fs=sample_rate, mean_period=1.0,
        tau=5, m=4, slope_ros=[0.2, 4.0],
    )
    assert np.isnan(out["ros_long"])
    assert out["ros_long_n_points"] == 0


def test_rosenstein_produces_finite_values(synthetic_segment, sample_rate):
    out = compute_lyapunov_rosenstein(
        synthetic_segment, fs=sample_rate, mean_period=1.0,
        tau=5, m=4, slope_ros=[0.2, 4.0],
    )
    assert np.isfinite(out["ros_short"]), f"got {out}"
    assert np.isfinite(out["ros_r2"]), f"got {out}"
    assert out["ros_n_points"] > 0


def test_rosenstein_uses_defaults_when_slope_none(synthetic_segment, sample_rate):
    out = compute_lyapunov_rosenstein(
        synthetic_segment, fs=sample_rate, mean_period=1.0,
    )
    assert np.isfinite(out["ros_short"])


def test_rosenstein_low_fit_quality_flag(synthetic_segment, sample_rate):
    out = compute_lyapunov_rosenstein(
        synthetic_segment, fs=sample_rate, mean_period=1.0,
        tau=5, m=4, slope_ros=[0.2, 4.0],
    )
    assert isinstance(out["ros_low_fit_quality"], bool)


# ---------------------------------------------------------------------------
# split_slope_windows
# ---------------------------------------------------------------------------

def test_split_slope_two_elements():
    short, long = split_slope_windows([0.2, 4.0])
    assert short == (0.2, 4.0)
    assert long is None


def test_split_slope_four_elements():
    short, long = split_slope_windows([0, 2, 4, 10])
    assert short == (0.0, 2.0)
    assert long == (4.0, 10.0)


@pytest.mark.parametrize("bad", [[], [1.0], [1, 2, 3], [1, 2, 3, 4, 5]])
def test_split_slope_rejects_other_lengths(bad):
    with pytest.raises(ValueError, match="2 entries|4 entries"):
        split_slope_windows(bad)


# ---------------------------------------------------------------------------
# Other wrappers — smoke tests
# ---------------------------------------------------------------------------

def test_wolf_returns_nan_below_min_samples(sample_rate):
    tiny = np.random.RandomState(0).randn(100)
    out = compute_lyapunov_wolf(tiny, fs=sample_rate, min_samples=500)
    assert np.isnan(out["wolf_lye"])


def test_wolf_finite_on_valid(synthetic_segment, sample_rate):
    out = compute_lyapunov_wolf(synthetic_segment, fs=sample_rate)
    assert np.isfinite(out["wolf_lye"])


def test_sample_entropy_key(synthetic_segment):
    out = compute_sample_entropy(synthetic_segment)
    assert set(out) == {"samp_ent"}
    assert np.isfinite(out["samp_ent"])


def test_corr_dim_key(synthetic_segment):
    out = compute_corr_dim(synthetic_segment)
    assert set(out) == {"corr_dim"}
    assert np.isfinite(out["corr_dim"])


# ---------------------------------------------------------------------------
# Rosenstein fit-quality flag — the bug that produced a whole results CSV of
# two-point "regressions" reporting R² = 1.0 on every single row.
# ---------------------------------------------------------------------------

def test_rosenstein_flags_degenerate_two_point_window(synthetic_segment,
                                                      sample_rate):
    """slope=[0, 0.2] at fs=5 spans samples 0..1 — a line through two points.
    R² is 1.0 by construction, so the point count must drive the warning."""
    out = compute_lyapunov_rosenstein(
        synthetic_segment, fs=sample_rate, mean_period=1.0,
        tau=5, m=4, slope_ros=[0, 0.2],
    )
    assert out["ros_n_points"] == 2
    assert out["ros_r2"] == 1.0
    assert out["ros_low_fit_quality"] is True, (
        "a two-point fit reporting R2=1.0 must not pass as a good fit"
    )


def test_rosenstein_four_element_slope_fills_long_window(synthetic_segment,
                                                         sample_rate):
    out = compute_lyapunov_rosenstein(
        synthetic_segment, fs=sample_rate, mean_period=1.0,
        tau=5, m=4, slope_ros=[0, 2, 4, 10],
    )
    assert out["ros_n_points"] >= MIN_FIT_POINTS
    assert out["ros_long_n_points"] >= MIN_FIT_POINTS
    assert np.isfinite(out["ros_short"]) and np.isfinite(out["ros_long"])


def test_rosenstein_short_and_long_windows_differ(synthetic_segment, sample_rate):
    """Distinct windows must fit distinct parts of the divergence curve."""
    out = compute_lyapunov_rosenstein(
        synthetic_segment, fs=sample_rate, mean_period=1.0,
        tau=5, m=4, slope_ros=[0, 2, 4, 10],
    )
    assert out["ros_short"] != out["ros_long"]


def test_rosenstein_mean_period_scales_the_exponent(synthetic_segment, sample_rate):
    """The fit window is in mean periods, so halving mean_period halves the
    sample span *and* doubles the time axis — the wrapper and the core must
    agree on that convention (they used to disagree)."""
    a = compute_lyapunov_rosenstein(
        synthetic_segment, fs=sample_rate, mean_period=1.0,
        tau=5, m=4, slope_ros=[0, 2],
    )
    b = compute_lyapunov_rosenstein(
        synthetic_segment, fs=sample_rate, mean_period=2.0,
        tau=5, m=4, slope_ros=[0, 2],
    )
    assert b["ros_n_points"] > a["ros_n_points"]


def test_rosenstein_raises_nothing_on_bad_slope(synthetic_segment, sample_rate):
    """A malformed slope is a config error and must surface, not be swallowed
    into a silent column of NaN."""
    with pytest.raises(ValueError):
        compute_lyapunov_rosenstein(
            synthetic_segment, fs=sample_rate, mean_period=1.0,
            slope_ros=[1, 2, 3],
        )


# ---------------------------------------------------------------------------
# fit_log_divergence — the shared fit used by both the core and the wrapper
# ---------------------------------------------------------------------------

def test_fit_log_divergence_on_linear_ramp():
    """Perfect line of slope 3 per period → slope 3.0, R² 1.0."""
    fs, mean_period = 10.0, 1.0
    t = np.arange(101) / fs / mean_period
    y = 3.0 * t + 1.5
    slope, r2, n = fit_log_divergence(y, fs, mean_period, 0.0, 10.0)
    assert slope == pytest.approx(3.0, abs=1e-9)
    assert r2 == pytest.approx(1.0, abs=1e-9)
    assert n == 101


def test_fit_log_divergence_rejects_empty_window():
    y = np.arange(1, 11, dtype=float)
    slope, r2, n = fit_log_divergence(y, 1.0, 1.0, 0.5, 0.5)
    assert np.isnan(slope) and np.isnan(r2) and n == 0


def test_fit_log_divergence_rejects_window_past_data():
    y = np.arange(1, 11, dtype=float)
    slope, r2, n = fit_log_divergence(y, 1.0, 1.0, 0.0, 100.0)
    assert np.isnan(slope) and n == 0


def test_fit_log_divergence_rejects_negative_start():
    y = np.arange(1, 11, dtype=float)
    slope, r2, n = fit_log_divergence(y, 1.0, 1.0, -1.0, 5.0)
    assert np.isnan(slope) and n == 0


def test_fit_log_divergence_r2_nan_on_flat_curve():
    """Zero variance in y leaves R² undefined rather than 0 or 1."""
    y = np.full(20, 2.0)
    slope, r2, n = fit_log_divergence(y, 1.0, 1.0, 0.0, 10.0)
    assert slope == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(r2)
    assert n == 11


def test_fit_log_divergence_units_are_per_period():
    """Same samples, different mean_period → slope scales by mean_period."""
    y = np.arange(1, 22, dtype=float)
    s1, _, n1 = fit_log_divergence(y, 1.0, 1.0, 0.0, 10.0)
    s2, _, n2 = fit_log_divergence(y, 1.0, 2.0, 0.0, 5.0)
    assert n1 == n2 == 11
    assert s2 == pytest.approx(s1 * 2.0)
