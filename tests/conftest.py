"""Shared fixtures for the CHAOS test suite."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

RAW_FS = 100.0
START = "2020-01-24T00:00:00"


@pytest.fixture
def sample_rate():
    return 5.0


@pytest.fixture
def synthetic_segment(sample_rate):
    rng = np.random.RandomState(42)
    n = 1000
    t = np.arange(n) / sample_rate
    signal = np.sin(2 * np.pi * 1.0 * t) + 0.3 * rng.randn(n)
    return (signal - signal.mean()) / signal.std(ddof=1)


@pytest.fixture
def rng():
    return np.random.RandomState(0)


@pytest.fixture
def fake_features_config():
    return {
        "wolf": {"tau": 5, "m": 5, "evolve": 5, "min_samples": 500},
        "rosenstein": {
            "tau": 5, "m": 4,
            "slope": [0, 2, 4, 10], "mean_period": 1.0,
        },
        "sample_entropy": {"m": 2, "r": 0.2},
        "corr_dim": {"tau": 5, "m": 5},
    }


@pytest.fixture
def fake_cfg(tmp_path, sample_rate, fake_features_config):
    """A settings stand-in for the streaming pipeline."""
    out = tmp_path / "results"
    out.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        Fs=sample_rate,
        FEATURES=fake_features_config,
        CHANNELS=["N"],
        PREPROCESS_CHANNELS=["E", "N", "Z"],
        N_JOBS=1,
        WARMUP_COUNT=3,
        WIN_SEC=200.0,
        STEP_SEC=50.0,
        WinSize=1000,
        StepSize=250,
        MAX_GAP_SEC=200.0,
        FREQMIN=0.1,
        FREQMAX=2.0,
        GAP_THRESHOLD=2.0,
        FILTER_PAD_SEC=None,
        STATION="TEST",
        EARTHQUAKE_NAME="EVENT",
        MSEED_INPUT_DIR=tmp_path / "raw",
        OUTPUT_ROOT=out,
        CACHE_ENABLED=False,
        CACHE_ROOT=tmp_path / "cache",
    )


@pytest.fixture
def preprocess_cfg(fake_cfg):
    """Alias kept so preprocessing tests read naturally."""
    return fake_cfg


# ---------------------------------------------------------------------------
# Synthetic waveform builders
# ---------------------------------------------------------------------------

def waveform(n, seed=0, fs=RAW_FS, t_offset=0.0):
    """A broadband-looking signal with energy inside the 0.1-2 Hz passband."""
    rng = np.random.RandomState(seed)
    t = np.arange(n) / fs + t_offset
    return (
        1000.0 * np.sin(2 * np.pi * 0.5 * t)
        + 300.0 * np.sin(2 * np.pi * 1.3 * t)
        + 50.0 * rng.randn(n)
    )


def trace(component, data, starttime, fs=RAW_FS):
    """Wraps samples in a broadband Trace."""
    from obspy import Trace

    tr = Trace(data=np.asarray(data, dtype=np.float64))
    tr.stats.network = "XX"
    tr.stats.station = "TEST"
    tr.stats.location = ""
    tr.stats.channel = f"BH{component}"
    tr.stats.sampling_rate = fs
    tr.stats.starttime = starttime
    return tr


def write_mseed(path, start, seconds, channels=("E", "N", "Z"), seed=0,
                gap=None, offsets=None, fs=RAW_FS):
    """Writes one multi-component MSEED file.

    Args:
        path: Destination path (parent directories are created).
        start: :class:`obspy.UTCDateTime` of the first sample.
        seconds: Span covered by the file.
        channels: Components to write.
        seed: Waveform seed; the same seed gives the same ground motion.
        gap: ``(offset_sec, duration_sec)`` to leave out of every channel.
        offsets: Per-channel start delay in seconds.
        fs: Raw sampling rate.

    Returns:
        The path written.
    """
    from obspy import Stream

    offsets = offsets or {}
    path.parent.mkdir(parents=True, exist_ok=True)
    traces = []
    for component in channels:
        shift = offsets.get(component, 0.0)
        npts = int(round((seconds - shift) * fs))
        data = waveform(npts, seed=seed, fs=fs,
                        t_offset=float(start) + shift)
        origin = start + shift
        if gap is None:
            traces.append(trace(component, data, origin, fs))
            continue
        cut = int(round(gap[0] * fs))
        resume = cut + int(round(gap[1] * fs))
        traces.append(trace(component, data[:cut], origin, fs))
        traces.append(
            trace(component, data[resume:], origin + resume / fs, fs)
        )
    Stream(traces).write(str(path), format="MSEED")
    return path


@pytest.fixture
def mseed_factory(tmp_path):
    """Returns a helper that writes MSEED files into the raw input dir."""
    def _make(name, start_offset_sec, seconds, **kwargs):
        from obspy import UTCDateTime

        return write_mseed(
            tmp_path / "raw" / name,
            UTCDateTime(START) + start_offset_sec,
            seconds,
            **kwargs,
        )
    return _make


@pytest.fixture
def two_hour_file(mseed_factory):
    """A single continuous 2-hour, 3-component file."""
    return mseed_factory("XX_TEST__part00.mseed", 0, 7200.0)
