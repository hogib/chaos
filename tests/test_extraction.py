"""Unit tests for chaos/extraction.py — window naming and the streamed run."""

import io
import contextlib

import numpy as np
import pandas as pd
import pytest
from obspy import UTCDateTime

from chaos.extraction import (FEATURE_KEYS, _iso, _WindowNamer, compute_window,
                              run_feature_extraction)


def _run(cfg):
    """Runs extraction quietly and returns the results frame."""
    with contextlib.redirect_stdout(io.StringIO()):
        ok = run_feature_extraction(cfg)
    if not ok:
        return None
    return pd.read_csv(next(cfg.OUTPUT_ROOT.glob("*_features.csv")))


# ---------------------------------------------------------------------------
# compute_window — unchanged by the restructure, guarded so it stays that way
# ---------------------------------------------------------------------------

def test_compute_window_keys_match_feature_keys(fake_cfg, synthetic_segment):
    """Every key in FEATURE_KEYS must exist in compute_window's output, and no
    extra feature keys may be returned that the CSV writer would drop."""
    result = compute_window(synthetic_segment, 0, fake_cfg)
    assert set(result.keys()) - {"w"} == set(FEATURE_KEYS)


def test_compute_window_rejects_nan_segment(fake_cfg, synthetic_segment):
    seg = synthetic_segment.copy()
    seg[10] = np.nan
    result = compute_window(seg, 0, fake_cfg)
    assert all(np.isnan(result[k]) for k in ("ham_mean", "wolf_lye", "samp_ent"))


def test_compute_window_rejects_zero_std_segment(fake_cfg):
    result = compute_window(np.ones(1000), 0, fake_cfg)
    assert np.isnan(result["ham_mean"])


def test_compute_window_returns_no_infinities(fake_cfg, synthetic_segment):
    """An inf reaching the CSV is worse than a NaN: it survives dropna()."""
    result = compute_window(synthetic_segment, 0, fake_cfg)
    numeric = [v for v in result.values() if isinstance(v, (int, float))]
    assert not any(np.isinf(v) for v in numeric)


def test_compute_window_echoes_window_index(fake_cfg, synthetic_segment):
    assert compute_window(synthetic_segment, 17, fake_cfg)["w"] == 17


# ---------------------------------------------------------------------------
# Window naming
# ---------------------------------------------------------------------------

def test_iso_formats_utc_with_milliseconds():
    assert _iso(UTCDateTime("2020-01-24T01:02:03.456789")) == \
        "2020-01-24T01:02:03.456Z"


def test_window_ids_number_within_each_hour():
    namer = _WindowNamer()
    base = UTCDateTime("2020-01-24T00:00:00")
    ids = [namer.name(base + i * 1800, base + i * 1800 + 200)[0]
           for i in range(5)]
    assert ids == [
        "2020_01_24_00_w01", "2020_01_24_00_w02",
        "2020_01_24_01_w01", "2020_01_24_01_w02",
        "2020_01_24_02_w01",
    ]


def test_time_min_counts_from_the_start_of_the_hour():
    namer = _WindowNamer()
    start = UTCDateTime("2020-01-24T03:10:00")
    _, time_min = namer.name(start, start + 200)
    assert time_min == pytest.approx((10 * 60 + 200) / 60.0, abs=1e-3)


def test_window_id_hour_always_two_digits():
    namer = _WindowNamer()
    t = UTCDateTime("2020-01-24T07:00:00")
    assert namer.name(t, t + 200)[0].split("_")[3] == "07"


# ---------------------------------------------------------------------------
# End-to-end streamed runs
# ---------------------------------------------------------------------------

def test_run_writes_csv_with_expected_columns(fake_cfg, mseed_factory):
    mseed_factory("a0.mseed", 0, 3600.0)
    df = _run(fake_cfg)

    expected = ["window_start", "window_end", "Window_ID", "Time_min"] + [
        f"N_{k}" for k in FEATURE_KEYS
    ]
    assert list(df.columns) == expected


def test_run_window_count_matches_the_recording(fake_cfg, mseed_factory):
    mseed_factory("b0.mseed", 0, 3600.0)
    df = _run(fake_cfg)
    n = int(3600 * fake_cfg.Fs)
    assert len(df) == (n - fake_cfg.WinSize) // fake_cfg.StepSize + 1


def test_run_blanks_the_warmup_windows(fake_cfg, mseed_factory):
    mseed_factory("c0.mseed", 0, 3600.0)
    df = _run(fake_cfg)
    assert df["N_ham_std"].head(fake_cfg.WARMUP_COUNT).isna().all()
    # The warm-up windows are reported but blank; the next one is real.
    assert df["N_ham_std"].notna().iloc[fake_cfg.WARMUP_COUNT]
    assert len(df) > fake_cfg.WARMUP_COUNT


def test_run_window_ids_are_unique(fake_cfg, mseed_factory):
    for i in range(2):
        mseed_factory(f"d{i}.mseed", i * 3600, 3600.0, seed=i)
    df = _run(fake_cfg)
    assert not df["Window_ID"].duplicated().any()


def test_run_timestamps_advance_by_one_step(fake_cfg, mseed_factory):
    mseed_factory("e0.mseed", 0, 3600.0)
    df = _run(fake_cfg)
    times = pd.to_datetime(df["window_start"], format="ISO8601")
    steps = times.diff().dropna().dt.total_seconds().unique()
    assert steps == pytest.approx([fake_cfg.STEP_SEC])


def test_run_window_span_is_one_window(fake_cfg, mseed_factory):
    mseed_factory("f0.mseed", 0, 3600.0)
    df = _run(fake_cfg)
    start = pd.to_datetime(df["window_start"], format="ISO8601")
    end = pd.to_datetime(df["window_end"], format="ISO8601")
    assert (end - start).dt.total_seconds().unique() == pytest.approx(
        [fake_cfg.WIN_SEC]
    )


def test_run_reports_nothing_without_input(fake_cfg):
    fake_cfg.MSEED_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    assert _run(fake_cfg) is None


def test_run_writes_the_gap_log(fake_cfg, mseed_factory):
    mseed_factory("g0.mseed", 0, 3600.0, gap=(1500.0, 60.0))
    _run(fake_cfg)
    logs = list((fake_cfg.OUTPUT_ROOT / "logs").glob("gap_report_*.txt"))
    assert len(logs) == 1
    assert "LARGE" in logs[0].read_text()


def test_run_creates_no_intermediate_files(fake_cfg, mseed_factory, tmp_path):
    """The whole point: nothing but the results and its log hits the disk."""
    mseed_factory("h0.mseed", 0, 3600.0)
    _run(fake_cfg)

    written = {
        p.relative_to(tmp_path).parts[0]
        for p in tmp_path.rglob("*") if p.is_file()
    }
    assert written == {"raw", "results"}
    assert not (tmp_path / "proceeded").exists()


def test_run_marks_gapped_windows_nan_but_keeps_going(fake_cfg, mseed_factory):
    """A gap must blank the windows that touch it, not the rest of the hour."""
    mseed_factory("i0.mseed", 0, 7200.0, gap=(3600.0, 120.0))
    df = _run(fake_cfg)
    assert df["N_ham_std"].isna().any()
    assert df["N_ham_std"].notna().sum() > len(df) // 2


# ---------------------------------------------------------------------------
# The equivalence property that makes streaming credible
# ---------------------------------------------------------------------------

def _write_split(cfg, tmp_path, full, splits, fs):
    """Writes one waveform as consecutive files split at ``splits``."""
    from obspy import Stream

    from conftest import START, trace

    raw = cfg.MSEED_INPUT_DIR
    if raw.exists():
        for old in raw.glob("*.mseed"):
            old.unlink()
    raw.mkdir(parents=True, exist_ok=True)
    bounds = list(zip([0] + splits, splits + [len(next(iter(full.values())))]))
    for k, (a, b) in enumerate(bounds):
        Stream([
            trace(c, full[c][a:b], UTCDateTime(START) + a / fs)
            for c in full
        ]).write(str(raw / f"part{k:02d}.mseed"), format="MSEED")


def test_split_files_give_the_same_features_as_one(fake_cfg, tmp_path):
    """Where the recording happens to be cut into files is an accident of how
    it was archived. It must not change a single feature value."""
    from conftest import RAW_FS, START, waveform

    total = int(2 * 3600 * RAW_FS)
    full = {
        c: waveform(total, seed=i, t_offset=float(UTCDateTime(START)))
        for i, c in enumerate("ENZ")
    }

    _write_split(fake_cfg, tmp_path, full, [], RAW_FS)
    one = _run(fake_cfg)

    _write_split(fake_cfg, tmp_path, full,
                 [int(0.4 * 3600 * RAW_FS), int(1.3 * 3600 * RAW_FS)], RAW_FS)
    many = _run(fake_cfg)

    assert list(one["window_start"]) == list(many["window_start"])
    assert list(one["Window_ID"]) == list(many["Window_ID"])

    # Waveform-level features must agree to floating-point noise. corr_dim is
    # excluded: it picks its fit range by thresholding the correlation
    # integral, so roughly one window in 200 sits on a bin edge and moves by
    # ~1% for an input change of 1e-9. That is a property of the estimator,
    # not of where the files were cut.
    for column in ("N_ham_std", "N_ham_min", "N_ham_max", "N_norm_max",
                   "N_wolf_lye", "N_samp_ent", "N_ros_short", "N_ros_long"):
        a = one[column].to_numpy(float)
        b = many[column].to_numpy(float)
        assert np.array_equal(np.isnan(a), np.isnan(b)), column
        scale = np.nanmax(np.abs(a)) or 1.0
        worst = np.nanmax(np.abs(a - b)) / scale
        assert worst < 1e-3, f"{column} moved by {worst:.2e} when files split"
