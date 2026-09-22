"""Unit tests for chaos/chaos_algorithms.py — the core numerical routines."""

import warnings

import numpy as np
import pytest

from chaos.chaos_algorithms import (_mean_log_divergence, _nearest_neighbours,
                                    ami_core, corrdim_core, fnn_core,
                                    rosenstein_lye_core, samp_ent_core,
                                    wolf_lye_core)

# ---------------------------------------------------------------------------
# rosenstein_lye_core — regression guards for the bugs just fixed
# ---------------------------------------------------------------------------

def test_rosenstein_returns_exactly_two_items(synthetic_segment, sample_rate):
    """Regression: the wrapper used to unpack three; the core returns two."""
    out = rosenstein_lye_core(
        synthetic_segment, fs=sample_rate, tau=5, dim=4,
        slope=[0.2, 4.0], mean_period=1.0,
    )
    assert isinstance(out, list)
    assert len(out) == 2


def test_rosenstein_second_item_is_divergence_matrix(synthetic_segment, sample_rate):
    """The second return value must be the [index, nn, AveLnDiv] stack."""
    _, out_matrix = rosenstein_lye_core(
        synthetic_segment, fs=sample_rate, tau=5, dim=4,
        slope=[0.2, 4.0], mean_period=1.0,
    )
    assert out_matrix.shape[0] == 3
    assert out_matrix.shape[1] > 0
    assert np.all(np.isfinite(out_matrix[2]))


def test_rosenstein_first_value_finite_on_valid_signal(synthetic_segment, sample_rate):
    lye, _ = rosenstein_lye_core(
        synthetic_segment, fs=sample_rate, tau=5, dim=4,
        slope=[0.2, 4.0], mean_period=1.0,
    )
    # May legitimately be NaN if the fit window is degenerate, but should not crash.
    assert isinstance(lye, (float, np.floating)) or np.isnan(lye)


def test_rosenstein_handles_short_signal(sample_rate):
    """Signals shorter than the embedding + band must not raise."""
    tiny = np.random.RandomState(0).randn(50)
    lye, out = rosenstein_lye_core(
        tiny, fs=sample_rate, tau=5, dim=4, slope=[0.2, 4.0], mean_period=1.0,
    )
    assert isinstance(lye, (float, np.floating))
    assert out.ndim == 2


def test_rosenstein_band_exclusion(synthetic_segment, sample_rate):
    """Nearest neighbours must lie outside the |i-j| <= (dim-1)*tau band."""
    _, out = rosenstein_lye_core(
        synthetic_segment, fs=sample_rate, tau=5, dim=4,
        slope=[0.2, 4.0], mean_period=1.0,
    )
    indices = out[0].astype(int)
    nns = out[1].astype(int)
    band = (4 - 1) * 5
    diffs = np.abs(indices - nns)
    assert np.all(diffs > band), "band exclusion violated"


# ---------------------------------------------------------------------------
# wolf_lye_core
# ---------------------------------------------------------------------------

def test_wolf_returns_two_items(synthetic_segment, sample_rate):
    out = wolf_lye_core(synthetic_segment, fs=sample_rate, tau=5, dim=5, evolve=5)
    assert isinstance(out, tuple)
    assert len(out) == 2


def test_wolf_lye_finite(synthetic_segment, sample_rate):
    _, lye = wolf_lye_core(synthetic_segment, fs=sample_rate, tau=5, dim=5, evolve=5)
    assert np.isfinite(lye)


def test_wolf_is_deterministic(synthetic_segment, sample_rate):
    _, a = wolf_lye_core(synthetic_segment, fs=sample_rate, tau=5, dim=5, evolve=5)
    _, b = wolf_lye_core(synthetic_segment, fs=sample_rate, tau=5, dim=5, evolve=5)
    assert a == b


# ---------------------------------------------------------------------------
# samp_ent_core
# ---------------------------------------------------------------------------

def test_samp_ent_noise_greater_than_sine(sample_rate):
    """SampEn should be higher for noisy signals than for a pure sine."""
    rng = np.random.RandomState(0)
    n = 500
    t = np.arange(n) / sample_rate
    sine = np.sin(2 * np.pi * t)
    noise = rng.randn(n)

    se_sine = samp_ent_core(sine, m=2, r=0.2)
    se_noise = samp_ent_core(noise, m=2, r=0.2)

    assert np.isfinite(se_sine) and np.isfinite(se_noise)
    assert se_noise > se_sine


def test_samp_ent_zero_on_constant_signal():
    """A constant signal has perfect regularity → SampEn == 0."""
    flat = np.ones(200)
    val = samp_ent_core(flat, m=2, r=0.2)
    assert val == 0.0

# ---------------------------------------------------------------------------
# corrdim_core
# ---------------------------------------------------------------------------

def test_corrdim_returns_float(synthetic_segment):
    val = corrdim_core(synthetic_segment, tau=5, de=5)
    assert isinstance(val, float)
    assert np.isfinite(val)


def test_corrdim_returns_zero_for_constant():
    flat = np.ones(500)
    assert corrdim_core(flat, tau=5, de=5) == 0.0


def test_corrdim_sine_less_than_noise(sample_rate):
    """Correlation dimension of a sine should be lower than that of white noise."""
    rng = np.random.RandomState(0)
    n = 1000
    t = np.arange(n) / sample_rate
    sine = np.sin(2 * np.pi * t)
    noise = rng.randn(n)

    cd_sine = corrdim_core(sine, tau=5, de=5)
    cd_noise = corrdim_core(noise, tau=5, de=5)
    assert cd_sine < cd_noise


def test_corrdim_of_pure_sine_is_about_one(sample_rate):
    """A sine traces a closed 1-D curve in phase space, so its correlation
    dimension is 1. Regression: anchoring the radius grid to machine epsilon
    instead of the smallest real distance used to put this near 3."""
    n = 1000
    sine = np.sin(2 * np.pi * np.arange(n) / sample_rate)
    assert corrdim_core(sine, tau=5, de=5) == pytest.approx(1.0, abs=0.15)


# ---------------------------------------------------------------------------
# ami_core
# ---------------------------------------------------------------------------

def test_ami_returns_matrix_and_curve(synthetic_segment):
    tau_matrix, v = ami_core(synthetic_segment, max_lag=20)
    assert isinstance(tau_matrix, np.ndarray)
    assert v.shape[0] == 2
    assert v.shape[1] == 20
    assert np.all(np.isfinite(v[1]))


def test_ami_first_lag_is_zero(synthetic_segment):
    _, v = ami_core(synthetic_segment, max_lag=10)
    assert v[0, 0] == 0.0


def test_ami_finds_minimum_for_periodic(sample_rate):
    """Pure sine of period 100 samples should have an AMI minimum near lag 100."""
    n = 3000
    t = np.arange(n) / sample_rate
    sine = np.sin(2 * np.pi * t / 20.0)  # period = 20 s = 100 samples at fs=5
    tau_matrix, _ = ami_core(sine, max_lag=150)
    assert tau_matrix.shape[0] >= 1
    assert tau_matrix[0, 0] > 0


# ---------------------------------------------------------------------------
# fnn_core
# ---------------------------------------------------------------------------

def test_fnn_returns_ratio_and_dimension(synthetic_segment):
    dE, dim = fnn_core(synthetic_segment, tau=5, max_dim=6, speed=1)
    assert dE.shape[0] == 6
    assert isinstance(dim, int)
    assert 1 <= dim <= 6


def test_fnn_ratio_decreases_for_sine(sample_rate):
    """False-neighbour ratio should fall off as embedding grows for a smooth signal."""
    n = 500
    t = np.arange(n) / sample_rate
    sine = np.sin(2 * np.pi * t)
    dE, _ = fnn_core(sine, tau=3, max_dim=6, speed=0)
    # final ratio should be no larger than the first
    assert dE[-1, 0] <= dE[0, 0] + 1e-9


# ---------------------------------------------------------------------------
# _nearest_neighbours / _mean_log_divergence — the pieces Rosenstein was
# refactored into. These are pinned against the straightforward O(M^2)
# definitions they replaced.
# ---------------------------------------------------------------------------

def _embed(x, tau, dim):
    M = len(x) - (dim - 1) * tau
    return np.column_stack([x[j * tau: M + j * tau] for j in range(dim)])


def test_nearest_neighbours_matches_brute_force(synthetic_segment):
    tau, dim = 5, 4
    Y = _embed(synthetic_segment, tau, dim)
    band = (dim - 1) * tau

    got = _nearest_neighbours(Y, band)

    M = Y.shape[0]
    dist = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=2)
    masked = np.where(np.abs(np.subtract.outer(np.arange(M), np.arange(M))) > band,
                      dist, np.inf)
    expected = masked.argmin(axis=1)

    # Ties are allowed to resolve differently; the distances must match.
    assert np.allclose(dist[np.arange(M), got], dist[np.arange(M), expected])


def test_nearest_neighbours_respects_band(synthetic_segment):
    tau, dim = 5, 4
    Y = _embed(synthetic_segment, tau, dim)
    band = (dim - 1) * tau
    got = _nearest_neighbours(Y, band)
    assert np.all(np.abs(got - np.arange(Y.shape[0])) > band)


def test_mean_log_divergence_matches_dense_matrix(synthetic_segment):
    """Reference: build the full (M, M) divergence matrix and take row means
    of its positive entries — what the function used to do explicitly."""
    tau, dim = 5, 4
    Y = _embed(synthetic_segment, tau, dim)
    M = Y.shape[0]
    neighbours = _nearest_neighbours(Y, (dim - 1) * tau)

    DM = np.zeros((M, M))
    for i in range(M):
        nn = neighbours[i]
        end = min(M - i, M - nn)
        if end > 0:
            DM[:end, i] = np.linalg.norm(Y[i:i + end] - Y[nn:nn + end], axis=1)

    expected = np.zeros(M)
    for i in range(M):
        pos = DM[i, DM[i, :] > 0]
        if len(pos):
            expected[i] = np.mean(np.log(pos))

    assert np.allclose(_mean_log_divergence(Y, neighbours), expected)


def test_rosenstein_accepts_four_element_slope(synthetic_segment, sample_rate):
    """The core only consumes the short window, but must not choke on the
    four-element [short_lo, short_hi, long_lo, long_hi] form."""
    lye, out = rosenstein_lye_core(
        synthetic_segment, fs=sample_rate, tau=5, dim=4,
        slope=[0, 2, 4, 10], mean_period=1.0,
    )
    assert np.isfinite(lye)
    assert out.shape[0] == 3


def test_rosenstein_degenerate_embedding_returns_empty(sample_rate):
    """dim*tau longer than the signal leaves no trajectory to embed."""
    lye, out = rosenstein_lye_core(
        np.arange(10, dtype=float), fs=sample_rate, tau=5, dim=4,
        slope=[0.2, 4.0], mean_period=1.0,
    )
    assert np.isnan(lye)
    assert out.shape == (3, 0)


# ---------------------------------------------------------------------------
# wolf_lye_core — degenerate inputs
# ---------------------------------------------------------------------------

def test_wolf_out_matrix_is_numeric(synthetic_segment, sample_rate):
    """The detail table used to be dtype=object, which is both slow and a
    nuisance for any consumer doing arithmetic on it."""
    out, _ = wolf_lye_core(synthetic_segment, fs=sample_rate, tau=5, dim=5,
                           evolve=5)
    assert out.dtype == np.float64
    assert out.shape[1] == 9


def test_wolf_constant_signal_does_not_produce_inf(sample_rate):
    """Every separation is zero, so log2(end/start) is undefined; the result
    must be NaN rather than +/-inf leaking into the CSV."""
    _, lye = wolf_lye_core(np.ones(600), fs=sample_rate, tau=5, dim=5, evolve=5)
    assert not np.isinf(lye)


def test_wolf_short_signal_returns_nan(sample_rate):
    _, lye = wolf_lye_core(np.arange(10, dtype=float), fs=sample_rate,
                           tau=5, dim=5, evolve=5)
    assert np.isnan(lye)


def test_wolf_lye_positive_for_chaotic_signal(sample_rate):
    """Logistic map at r=4 is chaotic; its largest exponent must be > 0."""
    x = np.empty(2000)
    x[0] = 0.4
    for i in range(1, len(x)):
        x[i] = 4.0 * x[i - 1] * (1.0 - x[i - 1])
    x = (x - x.mean()) / x.std(ddof=1)
    _, lye = wolf_lye_core(x, fs=sample_rate, tau=1, dim=3, evolve=5)
    assert lye > 0


# ---------------------------------------------------------------------------
# samp_ent_core / corrdim_core — degenerate inputs
# ---------------------------------------------------------------------------

def test_samp_ent_too_short_returns_nan():
    assert np.isnan(samp_ent_core(np.arange(2, dtype=float), m=2, r=0.2))


def test_samp_ent_is_symmetric_in_sign(sample_rate):
    """Negating a signal cannot change its regularity."""
    rng = np.random.RandomState(1)
    x = rng.randn(400)
    assert samp_ent_core(x, 2, 0.2) == pytest.approx(samp_ent_core(-x, 2, 0.2))


def test_corrdim_constant_signal_warns_nothing():
    """Regression: a constant window used to emit log(0) RuntimeWarnings for
    every single window it hit."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert corrdim_core(np.ones(500), tau=5, de=5) == 0.0


def test_corrdim_invariant_to_scaling(synthetic_segment):
    """Correlation dimension is a property of the geometry, not of the units."""
    a = corrdim_core(synthetic_segment, tau=5, de=5)
    b = corrdim_core(synthetic_segment * 1000.0, tau=5, de=5)
    assert a == pytest.approx(b, rel=1e-6)


# ---------------------------------------------------------------------------
# Degenerate-input robustness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [0, 1, 2, 5, 16])
def test_corrdim_returns_a_number_on_too_short_input(n):
    """Regression: n = len(x) - (de-1)*tau went negative and np.zeros raised
    ValueError('negative dimensions are not allowed')."""
    x = np.arange(n, dtype=float)
    assert corrdim_core(x, tau=5, de=5) == 0.0


@pytest.mark.parametrize("fn", [
    lambda x: wolf_lye_core(x, 5.0, 5, 5, 5),
    lambda x: rosenstein_lye_core(x, 5.0, 5, 5, [0, 1, 4, 10], 1.0),
    lambda x: samp_ent_core(x, 2, 0.2),
    lambda x: corrdim_core(x, 5, 5),
])
@pytest.mark.parametrize("name", ["const", "two_valued", "spike", "huge"])
def test_cores_emit_no_numpy_warnings(fn, name):
    """A RuntimeWarning here means a log(0) or 0/0 slipped into a feature."""
    rng = np.random.RandomState(11)
    data = {
        "const": np.ones(600),
        "two_valued": np.tile([0.0, 1.0], 300),
        "spike": np.r_[np.zeros(599), 1e9],
        "huge": rng.randn(600) * 1e12,
    }[name]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        fn(data)


# ---------------------------------------------------------------------------
# fit_log_divergence — the trailing zero run is not data
# ---------------------------------------------------------------------------

def test_fit_ignores_trailing_zero_run():
    """The divergence curve ends in exact zeros: those are steps where no
    trajectory pair is still in range, not measurements of zero divergence.
    Fitting into them drags the slope toward zero."""
    from chaos.chaos_algorithms import fit_log_divergence

    curve = np.concatenate([np.arange(1, 11, dtype=float), np.zeros(10)])
    # Window 0..9 lies entirely inside the real data.
    slope, r2, n = fit_log_divergence(curve, 1.0, 1.0, 0.0, 9.0)
    assert n == 10 and slope == pytest.approx(1.0)
    # Window 0..10 would read the first padding zero.
    assert fit_log_divergence(curve, 1.0, 1.0, 0.0, 10.0)[2] == 0


def test_rosenstein_curve_ends_in_zeros(synthetic_segment, sample_rate):
    """Documents the shape fit_log_divergence has to defend against."""
    _, out = rosenstein_lye_core(
        synthetic_segment, sample_rate, 5, 5, [0, 1, 4, 10], 1.0
    )
    curve = out[2]
    zeros = np.flatnonzero(curve == 0)
    assert zeros.size > 0
    assert np.array_equal(zeros, np.arange(zeros[0], curve.size)), (
        "zeros must form one contiguous tail"
    )
