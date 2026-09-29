"""Tests for target periods: day pieces, quiet selection, period runs."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
from obspy import UTCDateTime

from chaos import periods as pr
from chaos import pipeline as main_module
from chaos.pipeline import Settings, plan_jobs
from chaos.recording import (file_spans, iter_blocks, iter_piece_blocks,
                             plan_pieces)
from tests.conftest import write_mseed
from tests.test_mean_period import BASE

T0 = UTCDateTime("2024-05-01T00:00:00")
DAY = 86400.0


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "SCRIPT_DIR", tmp_path)
    return tmp_path


def _config(**target):
    config = copy.deepcopy(BASE)
    config["feature_extraction"]["features"]["rosenstein"]["mean_period"] = 1.0
    config["target"] = {"start_date": None, "duration_days": 1,
                        "quiet_window": False, **target}
    config["quiet_window"] = {"samples": 15, "min_mag": 3.0,
                              "max_dist_km": None, "seed": 0}
    return config


# ---------------------------------------------------------------------------
# plan_pieces
# ---------------------------------------------------------------------------

def _spans(*ranges):
    return [(T0 + a, T0 + b, f"f{i}") for i, (a, b) in enumerate(ranges)]


def test_pieces_split_on_day_boundaries():
    spans = _spans((0, 3 * DAY))
    pieces = plan_pieces(spans, T0 + 0.5 * DAY, T0 + 2.5 * DAY, 60)
    assert [(p.t0 - T0, p.t1 - T0) for p in pieces] == [
        (0.5 * DAY, DAY), (DAY, 2 * DAY), (2 * DAY, 2.5 * DAY)]


def test_pieces_list_every_file_their_padding_reaches():
    spans = _spans((0, DAY), (DAY, 2 * DAY), (2 * DAY, 3 * DAY))
    (piece,) = plan_pieces(spans, T0 + DAY, T0 + 2 * DAY, 60)
    assert piece.files == ("f0", "f1", "f2")
    (piece,) = plan_pieces(spans, T0 + DAY + 3600, T0 + DAY + 7200, 60)
    assert piece.files == ("f1",)


def test_pieces_skip_days_without_data():
    spans = _spans((0, DAY), (2 * DAY, 3 * DAY))
    pieces = plan_pieces(spans, T0, T0 + 3 * DAY, 60)
    assert [p.t0 - T0 for p in pieces] == [0, 2 * DAY]


def test_pieces_reproduce_whole_file_preprocessing(fake_cfg, tmp_path):
    """Cutting a file into pieces must give the same samples as processing it
    whole: pieces borrow filter padding from their neighbours."""
    path = write_mseed(tmp_path / "raw" / "XX_TEST_a.mseed", T0, 4 * 3600.0)
    whole = list(iter_blocks(fake_cfg, [path]))
    spans, _ = file_spans([path])
    pieces = plan_pieces(spans, T0, T0 + 4 * 3600, 60, piece_sec=3600)
    assert len(pieces) == 4
    cut = list(iter_piece_blocks(fake_cfg, pieces))

    def joined(blocks):
        start = blocks[0].start_index
        data = np.concatenate([b.data["N"] for b in blocks])
        assert all(b.start_index == blocks[i - 1].end_index
                   for i, b in enumerate(blocks) if i)
        return start, data

    s1, a = joined(whole)
    s2, b = joined(cut)
    assert s1 == s2 and len(a) == len(b)
    valid = ~np.isnan(a) & ~np.isnan(b)
    assert valid.mean() > 0.99
    assert np.max(np.abs(a[valid] - b[valid])) < 1e-3 * np.nanstd(a)


# ---------------------------------------------------------------------------
# Quiet selection
# ---------------------------------------------------------------------------

def test_quiet_candidates_avoid_events_and_gaps():
    stretches = [[float(T0), float(T0) + 10 * DAY]]
    ev_s = np.array([float(T0) + 4.5 * DAY])
    ev_e = ev_s + 600
    cands = pr.quiet_candidates(stretches, ev_s, ev_e, 2, pad=60)
    days = [(c - float(T0)) / DAY for c in cands]
    # Day 0 lacks padding before it, day 8 lacks it after; days 3 and 4 hold
    # the event.
    assert days == [1, 2, 5, 6, 7]


def test_pick_quiet_is_non_overlapping_and_seeded():
    cands = [float(T0) + d * DAY for d in range(40)]
    a = pr.pick_quiet(cands, 3, 15, seed=1)
    assert a == pr.pick_quiet(cands, 3, 15, seed=1)
    assert a != pr.pick_quiet(cands, 3, 15, seed=2)
    assert all(b - x >= 3 * DAY for x, b in zip(a, a[1:]))
    assert len(pr.pick_quiet(cands, 3, 100, seed=1)) <= 14


def test_target_period_label_and_validation():
    p = pr.target_period("2025-04-16", 7)
    assert p.label == "20250416_7d" and p.days == 7
    assert pr.target_period("2025-04-16T12:00:00", 0.5).label == "20250416_0.5d"
    with pytest.raises(ValueError, match="duration_days"):
        pr.target_period("2025-04-16", 0)
    with pytest.raises(ValueError, match="not a date"):
        pr.target_period("sometime", 1)


def test_discover_stations(root):
    (root / "data").mkdir()
    for name in ("TU_ELBA_x.mseed", "KO_CAME_y.mseed", "junk.mseed"):
        (root / "data" / name).write_bytes(b"")
    assert pr.discover_stations(root, _config()) == ["CAME", "ELBA"]


def _catalog(path, events):
    lines = ["Date,Longitude,Latitude,Depth,Type,Magnitude"]
    lines += [f"{t.strftime('%d/%m/%Y %H:%M:%S')},29.0,41.0,7.0,ML,{m}"
              for t, m in events]
    path.write_text("\n".join(lines) + "\n")


def _day_files(root, days, station="TEST"):
    for d in days:
        write_mseed(root / "data" / f"XX_{station}_{d:02d}.mseed",
                    T0 + d * DAY, DAY, channels=("N",), fs=10.0, seed=d)


def test_quiet_periods_end_to_end(root):
    _day_files(root, range(6))
    # M4 on day 2 breaks quiet; M1 on day 4 is below the threshold.
    _catalog(root / "catalog.csv", [(T0 + 2.5 * DAY, 4.0),
                                    (T0 + 4.5 * DAY, 1.0)])
    config = _config()
    periods, info = pr.quiet_periods(root, config, "TEST", 1)
    starts = [(p.t0 - T0) / DAY for p in periods]
    # Day 0 has no padding before it, day 5 none after, day 2 is loud.
    assert starts == [1, 3, 4]
    assert info["samples_found"] == 3 and info["candidates"] == 3
    assert periods[0].label == "quiet_1d/20240502"


def test_quiet_periods_explains_when_nothing_is_quiet(root):
    _day_files(root, range(3))
    _catalog(root / "catalog.csv", [(T0 + 1.5 * DAY, 5.0)])
    with pytest.raises(ValueError, match="no quiet 1-day period"):
        pr.quiet_periods(root, _config(), "TEST", 1)


# ---------------------------------------------------------------------------
# Period runs through the pipeline
# ---------------------------------------------------------------------------

def test_plan_jobs_modes(root):
    _day_files(root, range(4))
    _catalog(root / "catalog.csv", [(T0 - 30 * DAY + 3723, 1.0)])
    jobs, _ = plan_jobs(_config())
    assert jobs == [("TEST", "EVENT", None)]

    jobs, _ = plan_jobs(_config(start_date="2024-05-02", duration_days=2))
    ((station, eq, period),) = jobs
    assert station == "TEST" and eq is None
    assert period.label == "20240502_2d"

    jobs, quiet = plan_jobs(_config(quiet_window=True, duration_days=1))
    assert [p.label for _, _, p in jobs] == ["quiet_1d/20240502",
                                             "quiet_1d/20240503"]
    assert quiet["TEST"][1]["samples_found"] == 2


def test_period_run_writes_features_and_settings(root):
    _day_files(root, range(3))
    _catalog(root / "catalog.csv", [(T0 - 30 * DAY + 3723, 1.0)])
    config = _config(start_date="2024-05-02T00:00:00", duration_days=0.25)
    config["feature_extraction"]["fs"] = 5.0
    main_module.run(config, command="chaos run --start 2024-05-02")

    out = root / "results" / "TEST" / "20240502_0.25d" / "ENZ"
    frame = pd.read_csv(out / "TEST_20240502_0.25d_features.csv")
    starts = pd.to_datetime(frame["window_start"], utc=True)
    ends = pd.to_datetime(frame["window_end"], utc=True)
    assert starts.min() >= pd.Timestamp("2024-05-02", tz="UTC")
    assert ends.max() <= pd.Timestamp("2024-05-02T06:00:00", tz="UTC")
    assert frame["N_ros_short"].notna().sum() > 0

    saved = json.loads((out / "settings.json").read_text())
    assert saved["run"]["mode"] == "period"
    assert saved["run"]["command"] == "chaos run --start 2024-05-02"
    assert saved["run"]["mean_period_source"] == "config"
    assert saved["config"]["target"]["duration_days"] == 0.25
    assert saved["config"]["feature_extraction"]["features"]["rosenstein"][
        "mean_period"] == 1.0


def test_settings_json_records_the_estimated_mean_period(root, monkeypatch):
    from chaos import mean_period as mp

    def fake(cfg, config, **_):
        cfg.MEAN_PERIOD_SOURCE = "cache/somewhere.json"
        return 1.5
    monkeypatch.setattr(mp, "resolve_mean_period", fake)
    config = _config()
    config["feature_extraction"]["features"]["rosenstein"]["mean_period"] = None
    cfg = Settings(config, "TEST", "EVENT")
    cfg.resolve_mean_period()
    saved = json.loads(cfg.export({"note": 1}).read_text())
    assert saved["run"]["mean_period_s"] == 1.5
    assert saved["run"]["mean_period_source"] == "cache/somewhere.json"
    assert saved["run"]["note"] == 1
    assert saved["config"]["feature_extraction"]["features"]["rosenstein"][
        "mean_period"] == 1.5
    # The shared config is untouched.
    assert config["feature_extraction"]["features"]["rosenstein"][
        "mean_period"] is None


def test_quiet_run_writes_periods_and_summary(root):
    _day_files(root, range(4))
    _catalog(root / "catalog.csv", [(T0 - 30 * DAY + 3723, 1.0)])
    config = _config(quiet_window=True, duration_days=0.25)
    config["quiet_window"]["samples"] = 2
    config["feature_extraction"]["n_jobs"] = 1
    main_module.run(config)

    group = root / "results" / "TEST" / "quiet_0.25d"
    periods = pd.read_csv(group / "periods.csv", dtype={"label": str})
    summary = pd.read_csv(group / "summary.csv")
    assert len(periods) == 2 and list(summary["status"]) == ["ok", "ok"]
    info = json.loads((group / "periods.json").read_text())
    assert info["samples_found"] == 2
    for label in periods["label"]:
        run_dir = group / label / "ENZ"
        assert (run_dir / f"TEST_quiet_0.25d_{label}_features.csv").is_file()
        saved = json.loads((run_dir / "settings.json").read_text())
        assert saved["run"]["quiet_selection"]["samples_found"] == 2


def test_period_shorter_than_a_window_is_rejected(root):
    config = _config()
    period = pr.target_period("2024-05-01", 100 / DAY)
    with pytest.raises(ValueError, match="shorter than one window"):
        Settings(config, "TEST", period=period)
