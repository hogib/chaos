"""Tests for chaos/mean_period.py and how the pipeline picks the value."""

import copy

import pytest
from obspy import UTCDateTime

from chaos import mean_period as mp
from chaos import pipeline as main_module
from chaos.pipeline import Settings
from tests.conftest import write_mseed

BASE = {
    "process_all": False,
    "single_run": {"station": "TEST", "earthquake_name": "EVENT"},
    "paths": {"raw_dir": "raw", "results_dir": "results",
              "results_subdir": "ENZ", "cache_dir": "cache",
              "catalog": "catalog.csv", "data_dir": "data",
              "data_pattern": "*_{station}_*.mseed"},
    "cache": {"enabled": False},
    "mean_period_estimation": {
        "options": {"dayfirst": True, "chunk_sec": 1200.0, "chunks": 10,
                    "jobs": 1},
    },
    "preprocessing": {"freq_min": 0.1, "freq_max": 2.0,
                      "gap_threshold_sec": 2.0, "filter_pad_sec": None,
                      "channels": ["E", "N", "Z"]},
    "feature_extraction": {
        "fs": 5.0, "win_sec": 200, "step_sec": 50, "max_gap_sec": None,
        "n_jobs": 1, "warmup_count": 3, "channels": ["N"],
        "features": {
            "wolf": {"tau": 5, "m": 5, "evolve": 5, "min_samples": 500},
            "rosenstein": {"tau": 5, "m": 4, "slope": [0, 1, 4, 10],
                           "mean_period": None},
            "sample_entropy": {"m": 2, "r": 0.2},
            "corr_dim": {"tau": 5, "m": 5},
        },
    },
}

START = UTCDateTime("2024-05-01T00:00:00")


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "SCRIPT_DIR", tmp_path)
    return tmp_path


def _config(**rosenstein):
    config = copy.deepcopy(BASE)
    config["feature_extraction"]["features"]["rosenstein"].update(rosenstein)
    return config


def _write_catalog(path, events):
    lines = ["Date,Longitude,Latitude,Depth,Type,Magnitude"]
    for t, mag in events:
        lines.append(f"{t.strftime('%d/%m/%Y %H:%M:%S')},29.0,41.0,7.0,ML,{mag}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_configured_value_is_used_without_estimating(root, monkeypatch):
    def boom(*_):
        raise AssertionError("must not estimate when a value is given")
    monkeypatch.setattr(mp, "resolve_mean_period", boom)

    cfg = Settings(_config(mean_period=0.8), "TEST", "EVENT")
    assert not cfg.needs_mean_period
    cfg.resolve_mean_period()
    assert cfg.FEATURES["rosenstein"]["mean_period"] == 0.8


@pytest.mark.parametrize("missing", ["null", "absent"])
def test_unset_value_is_estimated(root, monkeypatch, missing):
    config = _config()
    if missing == "absent":
        del config["feature_extraction"]["features"]["rosenstein"][
            "mean_period"]
    monkeypatch.setattr(mp, "resolve_mean_period", lambda cfg, c: 1.7)

    cfg = Settings(config, "TEST", "EVENT")
    assert cfg.needs_mean_period
    cfg.resolve_mean_period()
    assert cfg.FEATURES["rosenstein"]["mean_period"] == 1.7
    # The shared config must stay unset for the next run.
    assert "mean_period" not in config["feature_extraction"]["features"][
        "rosenstein"] or config["feature_extraction"]["features"][
        "rosenstein"]["mean_period"] is None


def test_unset_value_without_catalog_is_rejected(root):
    config = _config()
    del config["paths"]["catalog"]
    with pytest.raises(ValueError, match="paths.catalog"):
        Settings(config, "TEST", "EVENT")


def test_estimated_value_is_validated(root, monkeypatch):
    # 0.1 s at 5 Hz makes the short window [0, 1] a single sample.
    monkeypatch.setattr(mp, "resolve_mean_period", lambda cfg, c: 0.1)
    cfg = Settings(_config(), "TEST", "EVENT")
    with pytest.raises(ValueError, match="short window"):
        cfg.resolve_mean_period()


def test_options_reject_unknown_keys(tmp_path):
    with pytest.raises(ValueError, match="bogus"):
        mp.options(tmp_path / "c.csv", [tmp_path], bogus=1)
    args = mp.options(tmp_path / "c.csv", [tmp_path], chunks=5)
    assert args.chunks == 5 and args.chunk_sec == 3600.0


def test_old_config_layout_still_works(root, monkeypatch):
    # Before the data keys moved to paths they lived here.
    config = _config()
    for key in ("catalog", "data_dir", "data_pattern"):
        del config["paths"][key]
    config["mean_period_estimation"].update(
        catalog="catalog.csv", data_dir="data", pattern="*_{station}_*.mseed")
    assert mp.setting(config, "catalog") == "catalog.csv"
    assert mp.data_pattern(config, "ELBA") == "*_ELBA_*.mseed"
    Settings(config, "TEST", "EVENT")   # validates without complaint


def test_find_recordings_split_files(tmp_path):
    for name in ("TU_TEST_a.mseed", "TU_TEST_b.mseed", "TU_OTHER_a.mseed"):
        (tmp_path / name).write_bytes(b"")
    pooled = mp.find_recordings([tmp_path], "*_TEST_*.mseed")
    split = mp.find_recordings([tmp_path], "*_TEST_*.mseed", split_files=True)
    assert len(pooled) == 1 and len(pooled[0][1]) == 2
    assert [len(f) for _, f in split] == [1, 1]


def test_resolve_from_quiet_data_and_cache(root):
    # Two 3-hour files; an M4 event in the first hour of each must keep the
    # chunks it touches out of the estimate.
    write_mseed(root / "data" / "TU_TEST_a.mseed", START, 3 * 3600.0)
    write_mseed(root / "data" / "TU_TEST_b.mseed", START + 86400,
                3 * 3600.0, seed=1)
    write_mseed(root / "data" / "TU_OTHER_a.mseed", START, 600.0)
    _write_catalog(root / "catalog.csv",
                   [(START + 1800, 4.0), (START + 86400 + 1800, 4.0)])

    cfg = Settings(_config(), "TEST", "EVENT")
    cfg.resolve_mean_period()
    value = cfg.FEATURES["rosenstein"]["mean_period"]
    # The synthetic signal is 0.5 Hz and 1.3 Hz sines plus noise.
    assert 0.5 < value < 2.0

    windows = list((root / "cache" / "mean_period" / "TEST").rglob(
        "mean_period_windows.csv"))
    assert len(windows) == 1
    import pandas as pd
    frame = pd.read_csv(windows[0])
    starts = pd.to_datetime(frame["window_start"], utc=True)
    first_hour = starts < pd.Timestamp((START + 3600).datetime, tz="UTC")
    assert not first_hour.any()
    assert set(frame["files"]) == {"TU_TEST_a.mseed", "TU_TEST_b.mseed"}

    # Second run reads the cached summary and gives the same value.
    again = Settings(_config(), "TEST", "EVENT")
    again.resolve_mean_period()
    assert again.FEATURES["rosenstein"]["mean_period"] == value


COORDS = """network,station,latitude,longitude,elevation
6G,ELBA,41.147,28.43,331.0
KO,CAME,36.9435,29.3023,100.0
6G,CAME,38.748,27.312,100.0
XX,TWIN,40.0,30.0,1.0
YY,TWIN,40.0,30.0,1.0
"""


@pytest.fixture
def coords_csv(tmp_path):
    path = tmp_path / "station_coords.csv"
    path.write_text(COORDS)
    return path


def test_station_coords_lookup(coords_csv):
    assert mp.station_coords(coords_csv, "ELBA") == (41.147, 28.43)
    # Case-insensitive, and the network need not match when there is one row.
    assert mp.station_coords(coords_csv, "elba", "TU") == (41.147, 28.43)
    assert mp.station_coords(coords_csv, "NOPE") is None
    # Two networks, same place: no ambiguity.
    assert mp.station_coords(coords_csv, "TWIN") == (40.0, 30.0)


def test_station_coords_network_breaks_ties(coords_csv):
    assert mp.station_coords(coords_csv, "CAME", "KO") == (36.9435, 29.3023)
    assert mp.station_coords(coords_csv, "CAME", "6G") == (38.748, 27.312)
    with pytest.raises(ValueError, match="several places"):
        mp.station_coords(coords_csv, "CAME", "TU")
    with pytest.raises(ValueError, match="several places"):
        mp.station_coords(coords_csv, "CAME")


def test_resolve_uses_station_coords_for_distance(root):
    # A far event over the first hour must no longer exclude it once the
    # station's coordinates are known; a near one still does.
    write_mseed(root / "data" / "XX_TEST_a.mseed", START, 3 * 3600.0)
    (root / "station_coords.csv").write_text(
        "network,station,latitude,longitude\nXX,TEST,41.0,29.0\n")
    lines = ["Date,Longitude,Latitude,Depth,Type,Magnitude",
             f"{(START + 1800).strftime('%d/%m/%Y %H:%M:%S')},"
             "40.0,38.0,7.0,ML,4.0",
             f"{(START + 2 * 3600 + 1800).strftime('%d/%m/%Y %H:%M:%S')},"
             "29.1,41.0,7.0,ML,4.0"]
    (root / "catalog.csv").write_text("\n".join(lines) + "\n")

    config = _config()
    config["paths"]["station_coords"] = "station_coords.csv"
    config["mean_period_estimation"]["options"]["max_dist_km"] = 200
    Settings(config, "TEST", "EVENT").resolve_mean_period()

    import pandas as pd
    (windows,) = (root / "cache" / "mean_period" / "TEST").rglob(
        "mean_period_windows.csv")
    starts = pd.to_datetime(pd.read_csv(windows)["window_start"], utc=True)
    offset = (starts - pd.Timestamp(START.datetime, tz="UTC")) \
        .dt.total_seconds()
    win_end = offset + 200
    far, near = 1800, 2 * 3600 + 1800
    # Far event ignored: windows cover its origin time.
    assert ((offset <= far) & (win_end >= far)).any()
    # Near event still excluded: nothing within its exclusion interval.
    assert not ((win_end > near - 120) & (offset < near + 300)).any()


def test_resolve_rejects_missing_coords_file(root):
    write_mseed(root / "data" / "XX_TEST_a.mseed", START, 3600.0)
    _write_catalog(root / "catalog.csv", [(START, 3.0)])
    config = _config()
    config["paths"]["station_coords"] = "missing.csv"
    with pytest.raises(ValueError, match="station_coords not found"):
        Settings(config, "TEST", "EVENT").resolve_mean_period()
