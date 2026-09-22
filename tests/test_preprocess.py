"""Unit tests for chaos/preprocess.py — the global grid and per-file decimation."""

from types import SimpleNamespace

import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime

from chaos.preprocess import (_as_float_array, _contiguous_runs,
                              _gap_duration_sec, _resolve_workers,
                              filter_pad_sec, grid_index, grid_index_floor,
                              grid_time, preprocess_file)
from conftest import RAW_FS, START, trace, waveform


@pytest.fixture
def simple_stream():
    tr = Trace(data=np.zeros(1000))
    tr.stats.station = "TEST"
    tr.stats.channel = "BHN"
    tr.stats.network = "XX"
    tr.stats.location = ""
    tr.stats.sampling_rate = 100.0
    tr.stats.starttime = UTCDateTime("2020-01-01T00:00:00")
    return Stream([tr])


# ---------------------------------------------------------------------------
# The global sample grid
# ---------------------------------------------------------------------------

def test_grid_index_round_trips():
    fs = 5.0
    t = UTCDateTime("2020-01-24T00:00:00")
    assert grid_time(grid_index(t, fs), fs) == t


def test_grid_index_is_absolute_not_per_file():
    """Two files an hour apart must land on the same grid, which is what lets
    them be decimated independently and still line up to the sample."""
    fs = 5.0
    a = UTCDateTime("2020-01-24T00:00:00")
    assert grid_index(a + 3600, fs) - grid_index(a, fs) == int(3600 * fs)


def test_grid_index_rounds_up_and_floor_rounds_down():
    fs = 5.0
    between = UTCDateTime(0) + 0.1      # between grid samples 0 and 1
    assert grid_index(between, fs) == 1
    assert grid_index_floor(between, fs) == 0


def test_grid_index_exact_sample_is_itself():
    fs = 5.0
    exact = UTCDateTime(0) + 0.4        # exactly grid sample 2
    assert grid_index(exact, fs) == 2
    assert grid_index_floor(exact, fs) == 2


# ---------------------------------------------------------------------------
# filter_pad_sec
# ---------------------------------------------------------------------------

def test_filter_pad_defaults_scale_with_freqmin():
    assert filter_pad_sec(SimpleNamespace(FREQMIN=0.1)) == 60.0
    assert filter_pad_sec(SimpleNamespace(FREQMIN=0.01)) == 600.0


def test_filter_pad_honours_explicit_setting():
    cfg = SimpleNamespace(FREQMIN=0.1, FILTER_PAD_SEC=250.0)
    assert filter_pad_sec(cfg) == 250.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_gap_duration_uses_matching_channel_fs(simple_stream):
    gap = ("XX", "TEST", "", "BHN", None, None, 0.0, 500)
    assert _gap_duration_sec(gap, simple_stream) == pytest.approx(5.0)


def test_gap_duration_falls_back_when_no_channel_match(simple_stream):
    gap = ("XX", "TEST", "", "BHZ", None, None, 0.0, 1000)
    assert _gap_duration_sec(gap, simple_stream) == pytest.approx(10.0)


def test_as_float_array_turns_mask_into_nan():
    masked = np.ma.array([1.0, 2.0, 3.0], mask=[False, True, False])
    out = _as_float_array(masked)
    assert np.isnan(out[1]) and out[0] == 1.0


def test_contiguous_runs_finds_each_stretch():
    valid = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0], dtype=bool)
    assert list(_contiguous_runs(valid)) == [(1, 3), (5, 8)]


def test_contiguous_runs_empty_when_nothing_valid():
    assert list(_contiguous_runs(np.zeros(5, dtype=bool))) == []


def test_resolve_workers_caps_at_n_items():
    assert _resolve_workers(SimpleNamespace(N_JOBS=64), n_items=4) == 4


def test_resolve_workers_uses_config():
    assert _resolve_workers(SimpleNamespace(N_JOBS=3), n_items=100) == 3


def test_resolve_workers_never_below_one():
    assert _resolve_workers(SimpleNamespace(N_JOBS=0), n_items=0) == 1


# ---------------------------------------------------------------------------
# preprocess_file
# ---------------------------------------------------------------------------

def test_preprocess_file_covers_exactly_its_own_span(preprocess_cfg,
                                                     two_hour_file):
    """The block must cover the file's span and not one sample more: padding
    is context for the filter, never data."""
    start_index, channels, _ = preprocess_file(preprocess_cfg, two_hour_file)

    assert grid_time(start_index, preprocess_cfg.Fs) == UTCDateTime(START)
    for name, arr in channels.items():
        assert len(arr) == int(7200 * preprocess_cfg.Fs), name


def test_preprocess_file_channels_have_equal_length(preprocess_cfg,
                                                    two_hour_file):
    _, channels, _ = preprocess_file(preprocess_cfg, two_hour_file)
    assert len({len(a) for a in channels.values()}) == 1


def test_preprocess_file_honours_configured_channels(preprocess_cfg,
                                                     two_hour_file):
    preprocess_cfg.PREPROCESS_CHANNELS = ["N"]
    _, channels, _ = preprocess_file(preprocess_cfg, two_hour_file)
    assert set(channels) == {"N"}


def test_preprocess_file_applies_the_bandpass(preprocess_cfg, two_hour_file):
    """The 1.3 Hz component sits in the passband, so energy must survive."""
    _, channels, _ = preprocess_file(preprocess_cfg, two_hour_file)
    data = channels["N"]
    assert np.nanstd(data) > 1.0
    assert abs(np.nanmean(data)) < np.nanstd(data)


def test_preprocess_file_reports_no_gaps(preprocess_cfg, two_hour_file):
    _, _, report = preprocess_file(preprocess_cfg, two_hour_file)
    assert "0 small" in report[0] and "0 large" in report[0]


def test_preprocess_file_keeps_large_gap_as_nan(preprocess_cfg, mseed_factory):
    path = mseed_factory("gapped.mseed", 0, 7200.0, gap=(1500.0, 60.0))
    _, channels, report = preprocess_file(preprocess_cfg, path)

    # One gap per channel: obspy counts them per trace pair, not per instant.
    assert "3 large" in report[0]
    assert sum("LARGE" in line for line in report) == 3
    nan = np.flatnonzero(np.isnan(channels["N"]))
    assert nan.size > 0, "a 60 s gap must survive as NaN, not be invented"
    assert nan.min() / preprocess_cfg.Fs == pytest.approx(1500, abs=40)


def test_preprocess_file_interpolates_small_gap(preprocess_cfg, mseed_factory):
    """Below the threshold a gap is bridged, so nothing reaches the output."""
    path = mseed_factory("small.mseed", 0, 7200.0, gap=(1500.0, 0.5))
    _, channels, report = preprocess_file(preprocess_cfg, path)
    assert "3 small" in report[0]
    assert not np.isnan(channels["N"]).any()


def test_preprocess_file_places_late_channel_at_its_true_offset(
        preprocess_cfg, mseed_factory):
    """Channels rarely start on the same sample. A late one must sit where it
    belongs on the grid, not be shunted to the top of the block."""
    path = mseed_factory("skew.mseed", 0, 5400.0,
                         offsets={"E": 0.0, "N": 120.0, "Z": 240.0})
    _, channels, _ = preprocess_file(preprocess_cfg, path)

    assert len({len(a) for a in channels.values()}) == 1
    for name, late_sec in (("E", 0.0), ("N", 120.0), ("Z", 240.0)):
        arr = channels[name]
        leading = int(np.argmax(~np.isnan(arr))) if np.isnan(arr[0]) else 0
        assert leading == pytest.approx(late_sec * preprocess_cfg.Fs, abs=2), name


def test_preprocess_file_warns_when_fs_does_not_divide(preprocess_cfg,
                                                       two_hour_file):
    """Decimation divides by an integer, so a target that does not divide the
    instrument rate silently yields a different rate; it must be said."""
    preprocess_cfg.Fs = 3.0
    _, _, report = preprocess_file(preprocess_cfg, two_hour_file)
    assert any("not the configured fs" in line for line in report)


# ---------------------------------------------------------------------------
# Padding — the property that makes parallel preprocessing exact
# ---------------------------------------------------------------------------

def _filter_continuous(data, cfg, fs=RAW_FS):
    """Filters a whole array in one pass: the reference the pads must match."""
    tr = Trace(data=data.copy())
    tr.stats.sampling_rate = fs
    tr.stats.starttime = UTCDateTime(START)
    tr.detrend("demean")
    tr.detrend("linear")
    tr.filter("bandpass", freqmin=cfg.FREQMIN, freqmax=cfg.FREQMAX,
              corners=4, zerophase=True)
    return tr.data


def _split_files(full, tmp_path, n_chunks, prefix):
    """Writes ``full`` as ``n_chunks`` consecutive single-channel files."""
    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    chunk = len(full) // n_chunks
    files = []
    for i in range(n_chunks):
        path = raw / f"{prefix}{i}.mseed"
        Stream([
            trace("N", full[i * chunk:(i + 1) * chunk],
                  UTCDateTime(START) + i * chunk / RAW_FS)
        ]).write(str(path), format="MSEED")
        files.append(path)
    return files


def test_padded_per_file_filtering_matches_continuous(preprocess_cfg, tmp_path):
    """Regression for the whole reason the pipeline was restructured.

    Filtering each chunk in isolation is wrong by ~98% of the signal's standard
    deviation at its edges. Borrowing context from the neighbouring files
    brings it back to a rounding error — which is what lets files be processed
    in parallel and still give the continuous answer.
    """
    full = waveform(int(3 * 3600 * RAW_FS), seed=3,
                    t_offset=float(UTCDateTime(START)))
    reference = _filter_continuous(full, preprocess_cfg)
    files = _split_files(full, tmp_path, 3, "p")

    preprocess_cfg.PREPROCESS_CHANNELS = ["N"]
    stitched = np.concatenate([
        preprocess_file(
            preprocess_cfg, path,
            files[i - 1] if i else None,
            files[i + 1] if i + 1 < len(files) else None,
        )[1]["N"]
        for i, path in enumerate(files)
    ])

    factor = int(RAW_FS / preprocess_cfg.Fs)
    expected = reference[::factor][:len(stitched)]
    rel = np.max(np.abs(stitched[:len(expected)] - expected)) / np.std(expected)
    assert rel < 0.01, f"padded filtering drifted from continuous by {rel:.3%}"


def test_without_padding_the_seams_are_wrong(preprocess_cfg, tmp_path):
    """The complement of the test above: with no neighbours to borrow from the
    seams really are badly wrong. This is what the old per-hour code shipped."""
    full = waveform(int(2 * 3600 * RAW_FS), seed=4,
                    t_offset=float(UTCDateTime(START)))
    reference = _filter_continuous(full, preprocess_cfg)
    files = _split_files(full, tmp_path, 2, "u")

    preprocess_cfg.PREPROCESS_CHANNELS = ["N"]
    unpadded = [preprocess_file(preprocess_cfg, p)[1]["N"] for p in files]
    got = np.concatenate(unpadded)

    factor = int(RAW_FS / preprocess_cfg.Fs)
    expected = reference[::factor][:len(got)]
    seam = len(unpadded[0])
    near_seam = np.max(
        np.abs(got[seam - 50:seam + 50] - expected[seam - 50:seam + 50])
    ) / np.std(expected)
    assert near_seam > 0.05, (
        "expected an unpadded seam to be visibly wrong; if this fails, the "
        "padding test above is no longer proving anything"
    )


def test_gap_report_ignores_the_padding_seams(preprocess_cfg, mseed_factory):
    """Reading a file together with its neighbours makes the joins between
    them look like gaps. They belong to no file and must not be reported as
    this one's, or every clean file grows phantom gaps."""
    files = [
        mseed_factory(f"seam{i}.mseed", i * 3600, 3600.0, seed=i)
        for i in range(3)
    ]
    _, _, report = preprocess_file(preprocess_cfg, files[1], files[0], files[2])
    assert "0 small" in report[0] and "0 large" in report[0]


def test_gap_report_still_sees_the_file_s_own_gaps(preprocess_cfg,
                                                   mseed_factory):
    """The complement: filtering out seams must not filter out real gaps."""
    files = [
        mseed_factory("own0.mseed", 0, 3600.0, seed=0),
        mseed_factory("own1.mseed", 3600, 3600.0, seed=1, gap=(1800.0, 30.0)),
        mseed_factory("own2.mseed", 7200, 3600.0, seed=2),
    ]
    _, _, report = preprocess_file(preprocess_cfg, files[1], files[0], files[2])
    assert "3 large" in report[0]
