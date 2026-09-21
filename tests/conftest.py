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


# Gap length in seconds. 60.07 s is 6007 raw samples at 100 Hz, which is not
# a multiple of the decimation factor, so the resumed segment starts off the
# decimated grid -- the case the phase handling has to get right.
GAP_SEC = 60.07


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
def fake_cfg(sample_rate, fake_features_config):
    return SimpleNamespace(
        Fs=sample_rate,
        FEATURES=fake_features_config,
        CHANNELS=["N"],
        N_JOBS=1,
        WARMUP_COUNT=3,
        WIN_SEC=200,
        STEP_SEC=50,
        PREV_SEC=150,
        WinSize=1000,
        StepSize=250,
        PREV_LEN=750,
        STATION="TEST",
        EARTHQUAKE_NAME="EVENT",
    )


@pytest.fixture
def preprocess_cfg(tmp_path, sample_rate):
    """Config for the preprocessing stage, writing into a temp tree."""
    return SimpleNamespace(
        Fs=sample_rate,
        PREPROCESS_WINDOW_SEC=3600.0,
        FREQMIN=0.1,
        FREQMAX=2.0,
        GAP_THRESHOLD=2.0,
        PREPROCESS_CHANNELS=["E", "N", "Z"],
        MSEED_INPUT_DIR=tmp_path / "raw",
        DATA_ROOT=tmp_path / "proceeded",
        N_JOBS=1,
        STATION="TEST",
        EARTHQUAKE_NAME="EVENT",
    )


TOTAL_SEC = 7200.0
RAW_FS = 100.0
GAP_START_SEC = 1500.0


def _waveform(seed, npts=int(TOTAL_SEC * RAW_FS), fs=RAW_FS):
    """One component's full 2-hour waveform, as a function of absolute time.

    Built once per component so that the gapped and gap-free fixtures share
    byte-identical samples everywhere outside the gap; that is what makes the
    two directly comparable.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(npts) / fs
    return (
        1000.0 * np.sin(2 * np.pi * 0.5 * t)
        + 300.0 * np.sin(2 * np.pi * 1.3 * t)
        + 50.0 * rng.randn(npts)
    )


def _trace(component, data, starttime, fs=RAW_FS):
    """Wraps a sample array in a broadband Trace."""
    from obspy import Trace

    tr = Trace(data=np.asarray(data, dtype=np.float64))
    tr.stats.network = "XX"
    tr.stats.station = "TEST"
    tr.stats.location = ""
    tr.stats.channel = f"BH{component}"
    tr.stats.sampling_rate = fs
    tr.stats.starttime = starttime
    return tr


def _write(stream, tmp_path, name):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"XX_TEST__24012020_000000_{name}.mseed"
    stream.write(str(path), format="MSEED")
    return path


def _build(tmp_path, name, gapped):
    """Writes a 2-hour, 3-component file where ``gapped`` components have a gap.

    Every component's samples come from the same :func:`_waveform` call, so a
    gapped file and a gap-free one agree exactly outside the gap.
    """
    from obspy import Stream, UTCDateTime

    start = UTCDateTime("2020-01-24T00:00:00")
    cut = int(GAP_START_SEC * RAW_FS)
    resume = cut + int(GAP_SEC * RAW_FS)

    traces = []
    for seed, c in enumerate(("E", "N", "Z")):
        full = _waveform(seed)
        if c in gapped:
            traces.append(_trace(c, full[:cut], start))
            traces.append(
                _trace(c, full[resume:], start + resume / RAW_FS)
            )
        else:
            traces.append(_trace(c, full, start))
    return _write(Stream(traces), tmp_path, name)


@pytest.fixture
def synthetic_mseed(tmp_path):
    """A 2-hour, 3-component 100 Hz MSEED file with no gaps.

    Naming follows the ``_YYYYMMDD_`` convention the date parser looks for.
    Each fixture uses a distinct file name so several can coexist in one test.
    """
    return _build(tmp_path, "clean", gapped=())


@pytest.fixture
def synthetic_mseed_with_gap(tmp_path):
    """Same span and samples as :func:`synthetic_mseed`, gapped in every channel.

    The gap falls inside the first hour; the second hour is continuous.
    """
    return _build(tmp_path, "gap", gapped=("E", "N", "Z"))


@pytest.fixture
def synthetic_mseed_gap_one_channel(tmp_path):
    """Two-hour file where only channel N has a gap.

    Exercises the case where, inside one window, one channel goes down the
    gap-handling branch while the others take the gap-free branch.
    """
    return _build(tmp_path, "gapn", gapped=("N",))


@pytest.fixture
def extraction_tree(tmp_path, fake_cfg, sample_rate):
    """Builds a ``proceeded/`` tree of per-channel CSVs and a matching cfg.

    Two date folders, two hourly files each, for channel ``N`` only — enough
    to exercise warm-up skipping and the previous-window carry-over.
    """
    rng = np.random.RandomState(7)
    data_root = tmp_path / "proceeded"
    n_per_file = 3000

    for date_name, stamps in (
        ("2020_01_24", ("20200124_000000", "20200124_010000")),
        ("2020_01_25", ("20200125_000000", "20200125_010000")),
    ):
        ch_dir = data_root / date_name / "N"
        ch_dir.mkdir(parents=True)
        for stamp in stamps:
            t = np.arange(n_per_file) / sample_rate
            sig = np.sin(2 * np.pi * 0.3 * t) + 0.4 * rng.randn(n_per_file)
            np.savetxt(ch_dir / f"{stamp}_N.csv", sig, delimiter=",", fmt="%.8f")

    fake_cfg.DATA_ROOT = data_root
    fake_cfg.OUTPUT_ROOT = tmp_path / "results"
    fake_cfg.OUTPUT_ROOT.mkdir(parents=True)
    return fake_cfg
