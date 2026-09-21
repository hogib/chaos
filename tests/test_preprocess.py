"""Unit tests for chaos/preprocess.py — helpers and worker pooling."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime

from chaos.preprocess import (_decimation_grid, _gap_duration_sec,
                              _process_single_mseed, _resolve_workers,
                              _save_csv_with_retry, parse_file_date,
                              run_mseed_preprocessing)


@pytest.fixture
def simple_stream():
    """A one-trace stream useful for helper tests."""
    tr = Trace(data=np.zeros(1000))
    tr.stats.station = "TEST"
    tr.stats.channel = "BHN"
    tr.stats.network = "XX"
    tr.stats.location = ""
    tr.stats.sampling_rate = 100.0
    tr.stats.starttime = UTCDateTime("2020-01-01T00:00:00")
    return Stream([tr])


# ---------------------------------------------------------------------------
# _gap_duration_sec
# ---------------------------------------------------------------------------

def test_gap_duration_uses_matching_channel_fs(simple_stream):
    # gap tuple layout: (net, sta, loc, cha, t1, t2, dur, npts)
    gap = ("XX", "TEST", "", "BHN", None, None, 0.0, 500)
    assert _gap_duration_sec(gap, simple_stream) == pytest.approx(5.0)


def test_gap_duration_falls_back_when_no_channel_match(simple_stream):
    gap = ("XX", "TEST", "", "BHZ", None, None, 0.0, 1000)
    assert _gap_duration_sec(gap, simple_stream) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# parse_file_date
# ---------------------------------------------------------------------------

def test_parse_file_date_yyyymmdd(simple_stream, tmp_path):
    f = tmp_path / "XX_TEST__20200124_000000.mseed"
    f.write_bytes(b"")
    assert parse_file_date(f, simple_stream).strftime("%Y-%m-%d") == "2020-01-24"


def test_parse_file_date_ddmmyyyy(simple_stream, tmp_path):
    f = tmp_path / "XX_TEST__24012020_000000.mseed"
    f.write_bytes(b"")
    # DDMMYYYY is tried first; YYYYMMDD is the fallback
    assert parse_file_date(f, simple_stream).strftime("%Y-%m-%d") == "2020-01-24"


def test_parse_file_date_fallback_to_stream_start(simple_stream, tmp_path):
    f = tmp_path / "no_date_here.mseed"
    f.write_bytes(b"")
    assert parse_file_date(f, simple_stream).strftime("%Y-%m-%d") == "2020-01-01"


# ---------------------------------------------------------------------------
# _save_csv_with_retry
# ---------------------------------------------------------------------------

def test_save_csv_writes_file(tmp_path):
    out = tmp_path / "out.csv"
    data = np.array([1.0, 2.0, 3.0])
    _save_csv_with_retry(out, data)
    loaded = np.loadtxt(out, delimiter=",")
    assert np.allclose(loaded, data)


def test_save_csv_retries_on_permission_error(tmp_path):
    out = tmp_path / "out.csv"
    data = np.array([1.0, 2.0])

    call_count = {"n": 0}
    real_savetxt = np.savetxt

    def flaky(path, *args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise PermissionError("locked")
        return real_savetxt(path, *args, **kwargs)

    with patch("chaos.preprocess.np.savetxt", side_effect=flaky), \
         patch("chaos.preprocess.time.sleep"):
        _save_csv_with_retry(out, data, retries=5, delay=0.0)

    assert call_count["n"] == 3
    assert out.exists()


# ---------------------------------------------------------------------------
# _resolve_workers
# ---------------------------------------------------------------------------

def test_resolve_workers_caps_at_n_items():
    cfg = SimpleNamespace(N_JOBS=64)
    assert _resolve_workers(cfg, n_items=4) == 4


def test_resolve_workers_uses_config():
    cfg = SimpleNamespace(N_JOBS=3)
    assert _resolve_workers(cfg, n_items=100) == 3


def test_resolve_workers_defaults_to_cpu_count():
    cfg = SimpleNamespace(N_JOBS=-1)
    import os
    expected = min(100, os.cpu_count() or 1)
    assert _resolve_workers(cfg, n_items=100) == expected


def test_resolve_workers_never_below_one():
    cfg = SimpleNamespace(N_JOBS=0)
    assert _resolve_workers(cfg, n_items=0) == 1


# ---------------------------------------------------------------------------
# _process_single_mseed — end-to-end, gap-free input
# ---------------------------------------------------------------------------

def _csvs(base):
    return sorted(base.rglob("*.csv"))


def test_process_single_mseed_writes_hourly_csvs(preprocess_cfg, synthetic_mseed):
    count, _ = _process_single_mseed(
        preprocess_cfg, synthetic_mseed, preprocess_cfg.DATA_ROOT
    )
    # 2 hours of data x 3 components
    assert count == 6
    assert len(_csvs(preprocess_cfg.DATA_ROOT)) == 6


def test_process_single_mseed_folder_layout(preprocess_cfg, synthetic_mseed):
    _process_single_mseed(preprocess_cfg, synthetic_mseed,
                          preprocess_cfg.DATA_ROOT)
    day = preprocess_cfg.DATA_ROOT / "2020_01_24"
    assert day.is_dir()
    assert {p.name for p in day.iterdir()} == {"E", "N", "Z"}
    assert (day / "N" / "20200124_000000_N.csv").is_file()
    assert (day / "N" / "20200124_010000_N.csv").is_file()


def test_process_single_mseed_decimates_to_target_fs(preprocess_cfg,
                                                     synthetic_mseed):
    """100 Hz in, Fs=5 Hz out: one hour must land on 3600*5 samples (+1 for the
    inclusive slice endpoint)."""
    _process_single_mseed(preprocess_cfg, synthetic_mseed,
                          preprocess_cfg.DATA_ROOT)
    data = np.loadtxt(
        preprocess_cfg.DATA_ROOT / "2020_01_24" / "N" / "20200124_000000_N.csv",
        delimiter=",",
    )
    assert abs(len(data) - 3600 * preprocess_cfg.Fs) <= 1


def test_process_single_mseed_channels_have_equal_length(preprocess_cfg,
                                                         synthetic_mseed):
    """Feature extraction pairs channels by index, so a length mismatch would
    silently misalign them."""
    _process_single_mseed(preprocess_cfg, synthetic_mseed,
                          preprocess_cfg.DATA_ROOT)
    day = preprocess_cfg.DATA_ROOT / "2020_01_24"
    lengths = {
        c: len(np.loadtxt(day / c / f"20200124_000000_{c}.csv", delimiter=","))
        for c in ("E", "N", "Z")
    }
    assert len(set(lengths.values())) == 1, lengths


def test_process_single_mseed_applies_bandpass(preprocess_cfg, synthetic_mseed):
    """The 1.3 Hz component sits in the passband, so the output must retain
    energy; a pure DC/constant output would mean the filter ate everything."""
    _process_single_mseed(preprocess_cfg, synthetic_mseed,
                          preprocess_cfg.DATA_ROOT)
    data = np.loadtxt(
        preprocess_cfg.DATA_ROOT / "2020_01_24" / "N" / "20200124_000000_N.csv",
        delimiter=",",
    )
    assert np.std(data) > 1.0
    assert abs(np.mean(data)) < np.std(data)


def test_process_single_mseed_reports_no_gaps(preprocess_cfg, synthetic_mseed):
    _, report = _process_single_mseed(
        preprocess_cfg, synthetic_mseed, preprocess_cfg.DATA_ROOT
    )
    assert "0 small" in report[0] and "0 large" in report[0]


# ---------------------------------------------------------------------------
# _process_single_mseed — gapped input
# ---------------------------------------------------------------------------

def test_process_gap_is_reported(preprocess_cfg, synthetic_mseed_with_gap):
    _, report = _process_single_mseed(
        preprocess_cfg, synthetic_mseed_with_gap, preprocess_cfg.DATA_ROOT
    )
    # one gap per component
    assert "3 large" in report[0]
    assert sum("LARGE" in line for line in report[1:]) == 3


def test_process_gap_preserved_as_nan(preprocess_cfg, synthetic_mseed_with_gap):
    """A 60 s gap is far above the 2 s threshold, so it must survive as NaN
    rather than being interpolated across."""
    _process_single_mseed(preprocess_cfg, synthetic_mseed_with_gap,
                          preprocess_cfg.DATA_ROOT)
    data = np.loadtxt(
        preprocess_cfg.DATA_ROOT / "2020_01_24" / "N" / "20200124_000000_N.csv",
        delimiter=",",
    )
    n_nan = int(np.isnan(data).sum())
    assert n_nan > 0
    # 60 s at 5 Hz = 300 samples, plus filter edge effects on the segments.
    assert 200 <= n_nan <= 900, n_nan


def test_process_gap_channels_still_equal_length(preprocess_cfg,
                                                 synthetic_mseed_with_gap):
    """Regression: the gapped branch used floor() for its output length while
    the gap-free branch used obspy's ceil(), so a window where one channel had
    a gap produced channel CSVs of different lengths."""
    _process_single_mseed(preprocess_cfg, synthetic_mseed_with_gap,
                          preprocess_cfg.DATA_ROOT)
    day = preprocess_cfg.DATA_ROOT / "2020_01_24"
    for stamp in ("20200124_000000", "20200124_010000"):
        lengths = {
            c: len(np.loadtxt(day / c / f"{stamp}_{c}.csv", delimiter=","))
            for c in ("E", "N", "Z")
        }
        assert len(set(lengths.values())) == 1, (stamp, lengths)


def test_process_gap_output_length_matches_gap_free(preprocess_cfg,
                                                    synthetic_mseed,
                                                    synthetic_mseed_with_gap,
                                                    tmp_path):
    """The same wall-clock hour must yield the same sample count whether or not
    it contained a gap — otherwise the two are not on a common time base."""
    clean_out = tmp_path / "clean"
    gappy_out = tmp_path / "gappy"
    _process_single_mseed(preprocess_cfg, synthetic_mseed, clean_out)
    _process_single_mseed(preprocess_cfg, synthetic_mseed_with_gap, gappy_out)

    rel = "2020_01_24/N/20200124_000000_N.csv"
    a = np.loadtxt(clean_out / rel, delimiter=",")
    b = np.loadtxt(gappy_out / rel, delimiter=",")
    assert len(a) == len(b)


def test_process_gap_keeps_samples_on_the_decimation_grid(
        preprocess_cfg, synthetic_mseed, synthetic_mseed_with_gap, tmp_path):
    """Regression: each clean segment used to restart the decimation phase at
    its own start index, so every sample *after* a gap sat on a different grid
    than the same hour processed without one.

    Both fixtures carry the same waveform, so outside the gap the two outputs
    must agree almost exactly. Measured separation is ~0.99999999 when the
    phase is handled and ~0.955 when it is not.
    """
    clean_out = tmp_path / "clean2"
    gappy_out = tmp_path / "gappy2"
    _process_single_mseed(preprocess_cfg, synthetic_mseed, clean_out)
    _process_single_mseed(preprocess_cfg, synthetic_mseed_with_gap, gappy_out)

    rel = "2020_01_24/N/20200124_000000_N.csv"
    a = np.loadtxt(clean_out / rel, delimiter=",")
    b = np.loadtxt(gappy_out / rel, delimiter=",")

    # Well clear of the gap (which ends around index 7800) and of the filter
    # edge effects on either side of it.
    post_gap = slice(8200, 17500)
    seg_a, seg_b = a[post_gap], b[post_gap]
    assert not np.isnan(seg_b).any()

    corr = np.corrcoef(seg_a, seg_b)[0, 1]
    assert corr > 0.999, f"decimation phase mismatch after gap (corr={corr:.4f})"


def test_process_gap_in_one_channel_keeps_lengths_aligned(
        preprocess_cfg, synthetic_mseed_gap_one_channel):
    """Only N is gapped, so within one window N takes the gap-handling branch
    while E and Z take the gap-free one. The two branches must agree on the
    output length, or the channels silently desynchronise."""
    _process_single_mseed(preprocess_cfg, synthetic_mseed_gap_one_channel,
                          preprocess_cfg.DATA_ROOT)
    day = preprocess_cfg.DATA_ROOT / "2020_01_24"
    lengths = {
        c: len(np.loadtxt(day / c / f"20200124_000000_{c}.csv", delimiter=","))
        for c in ("E", "N", "Z")
    }
    assert len(set(lengths.values())) == 1, lengths

    n = np.loadtxt(day / "N" / "20200124_000000_N.csv", delimiter=",")
    e = np.loadtxt(day / "E" / "20200124_000000_E.csv", delimiter=",")
    assert np.isnan(n).any(), "the gapped channel should retain its NaN block"
    assert not np.isnan(e).any(), "the clean channel should have no NaN"


# ---------------------------------------------------------------------------
# _decimation_grid
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seg_start,factor,expected", [
    (0, 20, (0, 0)),
    (20, 20, (0, 1)),
    (156007, 20, (13, 7801)),
    (1, 20, (19, 1)),
    (7, 1, (0, 7)),
])
def test_decimation_grid_values(seg_start, factor, expected):
    assert _decimation_grid(seg_start, factor) == expected


@pytest.mark.parametrize("seg_start", [0, 1, 7, 19, 20, 999, 156007, 360000])
def test_decimation_grid_lands_on_absolute_multiples(seg_start):
    """The first kept sample must sit at an absolute index divisible by the
    decimation factor — that is what makes the grid global rather than
    per-segment."""
    factor = 20
    phase, out_start = _decimation_grid(seg_start, factor)
    assert 0 <= phase < factor
    assert (seg_start + phase) % factor == 0
    assert out_start * factor == seg_start + phase


# ---------------------------------------------------------------------------
# run_mseed_preprocessing — the gap log must actually contain the gaps
# ---------------------------------------------------------------------------

def test_run_writes_gap_log_with_content(preprocess_cfg,
                                         synthetic_mseed_with_gap):
    """Regression: the log file was created with only a header, and every gap
    line went to the worker's stdout where it was lost."""
    assert run_mseed_preprocessing(preprocess_cfg) is True

    logs = list((preprocess_cfg.DATA_ROOT / "logs").glob("gap_report_*.txt"))
    assert len(logs) == 1
    text = logs[0].read_text(encoding="utf-8")

    assert "GAP REPORT" in text
    assert synthetic_mseed_with_gap.name in text
    assert "3 large" in text
    assert "LARGE" in text


def test_run_returns_false_without_input(preprocess_cfg):
    preprocess_cfg.MSEED_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    assert run_mseed_preprocessing(preprocess_cfg) is False


def test_run_produces_the_expected_csv_tree(preprocess_cfg, synthetic_mseed):
    run_mseed_preprocessing(preprocess_cfg)
    assert len(_csvs(preprocess_cfg.DATA_ROOT)) == 6
