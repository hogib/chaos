"""Unit tests for chaos/extraction.py — key-consistency and integration."""

import numpy as np
import pytest

import pandas as pd

from chaos.extraction import (FEATURE_KEYS, _extract_hour, compute_window,
                              run_feature_extraction)

# ---------------------------------------------------------------------------
# THE key-consistency regression test — would have caught Bug 2
# ---------------------------------------------------------------------------

def test_compute_window_keys_match_feature_keys(fake_cfg, synthetic_segment):
    """Every key in FEATURE_KEYS must exist in compute_window's output, and
    no extra feature keys may be returned that the CSV writer would drop."""
    result = compute_window(synthetic_segment, 0, fake_cfg)
    keys = set(result.keys()) - {"w"}
    expected = set(FEATURE_KEYS)

    missing = expected - keys
    extra = keys - expected

    assert not missing, (
        f"FEATURE_KEYS contains {sorted(missing)} but compute_window does not "
        f"produce them — those columns will be blank in the CSV."
    )
    assert not extra, (
        f"compute_window produces {sorted(extra)} which are not in FEATURE_KEYS "
        f"— those values will be silently dropped by the row builder."
    )


# ---------------------------------------------------------------------------
# End-to-end compute_window contract — would have caught Bug 1
# ---------------------------------------------------------------------------

def test_compute_window_populates_rosenstein_columns(fake_cfg, synthetic_segment):
    """Regression: if extraction calls the wrapper with the wrong kwarg, every
    Rosenstein key is NaN (exception swallowed). All must be finite here."""
    result = compute_window(synthetic_segment, 0, fake_cfg)

    assert np.isfinite(result["ros_short"]), (
        "ros_short is NaN — most likely a swallowed TypeError in the "
        "compute_lyapunov_rosenstein call."
    )
    assert np.isfinite(result["ros_r2"])
    assert result["ros_n_points"] > 0


def test_compute_window_populates_wolf_and_sampen(fake_cfg, synthetic_segment):
    result = compute_window(synthetic_segment, 0, fake_cfg)
    assert np.isfinite(result["wolf_lye"])
    assert np.isfinite(result["samp_ent"])
    assert np.isfinite(result["corr_dim"])
    assert np.isfinite(result["ham_mean"])


def test_compute_window_rejects_nan_segment(fake_cfg, synthetic_segment):
    seg = synthetic_segment.copy()
    seg[10] = np.nan
    result = compute_window(seg, 0, fake_cfg)
    assert all(np.isnan(result[k]) for k in FEATURE_KEYS)


def test_compute_window_rejects_zero_std_segment(fake_cfg):
    seg = np.ones(1000)
    result = compute_window(seg, 0, fake_cfg)
    assert all(np.isnan(result[k]) for k in FEATURE_KEYS)


def test_compute_window_populates_rosenstein_long_window(fake_cfg,
                                                         synthetic_segment):
    """The configured slope is 4-element, so the long-window keys must be
    filled rather than left at their NaN defaults."""
    result = compute_window(synthetic_segment, 0, fake_cfg)
    assert np.isfinite(result["ros_long"])
    assert result["ros_long_n_points"] > 0


def test_compute_window_returns_no_infinities(fake_cfg, synthetic_segment):
    """An inf reaching the CSV is worse than a NaN: it survives dropna()."""
    result = compute_window(synthetic_segment, 0, fake_cfg)
    numeric = [v for k, v in result.items() if isinstance(v, (int, float))]
    assert not any(np.isinf(v) for v in numeric)


def test_compute_window_echoes_window_index(fake_cfg, synthetic_segment):
    for w in (0, 1, 7):
        assert compute_window(synthetic_segment, w, fake_cfg)["w"] == w


# ---------------------------------------------------------------------------
# _extract_hour
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "stem,expected",
    [
        ("20250423_143000", 14),
        ("20250423_143000_N", 14),
        ("20250423_000000_Z", 0),
        ("20250423_235959_E", 23),
        ("14", 14),
        ("garbage", -1),
        ("", -1),
    ],
)
def test_extract_hour(stem, expected):
    assert _extract_hour(stem) == expected


# ---------------------------------------------------------------------------
# run_feature_extraction — end-to-end over a real folder tree
# ---------------------------------------------------------------------------

def _only_csv(cfg):
    files = list(cfg.OUTPUT_ROOT.glob("*.csv"))
    assert len(files) == 1, f"expected one output CSV, got {files}"
    return pd.read_csv(files[0])


def test_run_feature_extraction_writes_csv(extraction_tree):
    run_feature_extraction(extraction_tree)
    df = _only_csv(extraction_tree)
    assert len(df) > 0


def test_run_feature_extraction_column_layout(extraction_tree):
    run_feature_extraction(extraction_tree)
    df = _only_csv(extraction_tree)
    expected = ["Window_ID", "Time_min"] + [
        f"{ch}_{key}"
        for ch in extraction_tree.CHANNELS
        for key in FEATURE_KEYS
    ]
    assert list(df.columns) == expected


def test_run_feature_extraction_skips_warmup_windows(extraction_tree):
    """The first WARMUP_COUNT windows of the run are placeholders, because the
    previous-window buffer has not filled yet."""
    run_feature_extraction(extraction_tree)
    df = _only_csv(extraction_tree)
    head = df.head(extraction_tree.WARMUP_COUNT)
    assert head["N_wolf_lye"].isna().all()
    assert df["N_wolf_lye"].iloc[extraction_tree.WARMUP_COUNT:].notna().any()


def test_run_feature_extraction_window_ids_are_unique(extraction_tree):
    run_feature_extraction(extraction_tree)
    df = _only_csv(extraction_tree)
    assert df["Window_ID"].is_unique


def test_run_feature_extraction_covers_every_date(extraction_tree):
    run_feature_extraction(extraction_tree)
    df = _only_csv(extraction_tree)
    dates = {wid.rsplit("_", 2)[0] for wid in df["Window_ID"]}
    assert dates == {"2020_01_24", "2020_01_25"}


def test_run_feature_extraction_produces_real_rosenstein_fits(extraction_tree):
    """Regression for the results CSV that shipped with ros_n_points == 2 and
    ros_r2 == 1.0 on all 53,535 rows."""
    run_feature_extraction(extraction_tree)
    df = _only_csv(extraction_tree)
    fitted = df["N_ros_n_points"].dropna()
    fitted = fitted[fitted > 0]
    assert len(fitted) > 0
    assert (fitted >= 3).all(), "short fit window collapsed to <3 points"
    assert df["N_ros_r2"].dropna().nunique() > 1, (
        "every R2 identical — the fit window is almost certainly degenerate"
    )


def test_run_feature_extraction_handles_missing_data_root(fake_cfg, tmp_path,
                                                          capsys):
    fake_cfg.DATA_ROOT = tmp_path / "nope"
    fake_cfg.OUTPUT_ROOT = tmp_path / "out"
    fake_cfg.OUTPUT_ROOT.mkdir()
    run_feature_extraction(fake_cfg)
    assert "not found" in capsys.readouterr().out
    assert not list(fake_cfg.OUTPUT_ROOT.glob("*.csv"))


def test_run_feature_extraction_handles_empty_data_root(fake_cfg, tmp_path,
                                                        capsys):
    fake_cfg.DATA_ROOT = tmp_path / "proceeded"
    fake_cfg.DATA_ROOT.mkdir()
    fake_cfg.OUTPUT_ROOT = tmp_path / "out"
    fake_cfg.OUTPUT_ROOT.mkdir()
    run_feature_extraction(fake_cfg)
    assert "No date folders" in capsys.readouterr().out


def test_run_feature_extraction_ignores_logs_folder(extraction_tree):
    (extraction_tree.DATA_ROOT / "logs").mkdir()
    (extraction_tree.DATA_ROOT / "logs" / "gap.txt").write_text("noise")
    run_feature_extraction(extraction_tree)
    df = _only_csv(extraction_tree)
    assert not any(wid.startswith("logs") for wid in df["Window_ID"])


# ---------------------------------------------------------------------------
# Ragged channels and the Time_min clock
# ---------------------------------------------------------------------------

def _write_channel(root, date_name, stamp, channel, data):
    ch_dir = root / date_name / channel
    ch_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(ch_dir / f"{stamp}_{channel}.csv", data, delimiter=",",
               fmt="%.8f")


def test_run_feature_extraction_pads_ragged_channels(fake_cfg, tmp_path,
                                                     recwarn):
    """Regression: window counts came from channels[0] alone, so a shorter
    channel was sliced past its end and features were computed on a silently
    truncated window -- a plausible number for a window that does not exist.
    The short channel must read as NaN instead."""
    root = tmp_path / "proceeded"
    rng = np.random.RandomState(3)
    _write_channel(root, "2020_01_24", "20200124_000000", "E", rng.randn(3000))
    _write_channel(root, "2020_01_24", "20200124_000000", "N", rng.randn(2000))

    fake_cfg.CHANNELS = ["E", "N"]
    fake_cfg.WARMUP_COUNT = 0
    fake_cfg.DATA_ROOT = root
    fake_cfg.OUTPUT_ROOT = tmp_path / "out"
    fake_cfg.OUTPUT_ROOT.mkdir()

    run_feature_extraction(fake_cfg)
    df = _only_csv(fake_cfg)

    # E covers all 9 windows; N runs out after 2000 samples.
    assert df["E_ham_std"].notna().all()
    assert df["N_ham_std"].isna().any()
    # Nothing truncated: every window N does report is a full one.
    assert df["N_ham_std"].notna().sum() < len(df)
    assert not any("empty slice" in str(w.message) for w in recwarn)


def test_run_feature_extraction_time_min_is_minutes_into_the_hour(
        extraction_tree):
    """Regression: Time_min counted from the start of the carry-over buffer,
    so every row sat PREV_SEC too late -- and the first file of the run, which
    has no carry-over, used a different offset again."""
    run_feature_extraction(extraction_tree)
    df = _only_csv(extraction_tree)

    cfg = extraction_tree
    win_min = cfg.WinSize / cfg.Fs / 60.0
    step_min = cfg.StepSize / cfg.Fs / 60.0
    carry_min = cfg.PREV_LEN / cfg.Fs / 60.0

    # Hour 00 of the first date has no carry-over: window 1 ends one window
    # length into the hour.
    first = df[df["Window_ID"].str.startswith("2020_01_24_00_")]
    assert first["Time_min"].iloc[0] == pytest.approx(win_min, abs=1e-3)

    # Every later hour does have one, so window 1 ends earlier in the hour.
    later = df[df["Window_ID"].str.startswith("2020_01_24_01_")]
    assert later["Time_min"].iloc[0] == pytest.approx(
        win_min - carry_min, abs=1e-3
    )
    # The clock restarts each hour and advances one step per window.
    steps = later["Time_min"].diff().dropna().to_numpy()
    assert steps == pytest.approx(np.full(len(steps), step_min), abs=1e-3)


def test_run_feature_extraction_window_id_hour_is_two_digits(extraction_tree):
    run_feature_extraction(extraction_tree)
    df = _only_csv(extraction_tree)
    hours = {wid.split("_")[3] for wid in df["Window_ID"]}
    assert hours == {"00", "01"}
