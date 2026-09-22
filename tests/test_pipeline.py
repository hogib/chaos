"""Unit tests for chaos/pipeline.py — config loading, paths, job collection."""

import json

import pytest

from chaos import pipeline as main_module
from chaos.pipeline import (Settings, _collect_jobs, _find_project_root,
                            _run_single, load_config)

# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_config():
    return {
        "process_all": False,
        "single_run": {"station": "ELBA", "earthquake_name": "EVENT"},
        "paths": {
            "raw_dir": "raw",
            "processed_dir": "proceeded",
            "results_dir": "results",
            "results_subdir": "ENZ",
        },
        "preprocessing": {
            "window_sec": 3600.0,
            "freq_min": 0.1,
            "freq_max": 2.0,
            "gap_threshold_sec": 2.0,
            "channels": ["E", "N", "Z"],
        },
        "feature_extraction": {
            "fs": 5.0,
            "win_sec": 200,
            "step_sec": 50,
            "prev_sec": 150,
            "n_jobs": 2,
            "warmup_count": 3,
            "channels": ["N"],
            "features": {
                "wolf": {"tau": 5, "m": 5, "evolve": 5, "min_samples": 500},
                "rosenstein": {
                    "tau": 5, "m": 4, "slope": [0.2, 4.0], "mean_period": 1.0,
                },
                "sample_entropy": {"m": 2, "r": 0.2},
                "corr_dim": {"tau": 5, "m": 5},
            },
        },
    }


def test_load_config_reads_json(tmp_path, minimal_config):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(minimal_config))
    assert load_config(path) == minimal_config


def test_load_config_raises_on_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.json")


def test_load_config_raises_on_invalid_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    with pytest.raises(json.JSONDecodeError):
        load_config(path)


# ---------------------------------------------------------------------------
# Settings — path and unit derivation
# ---------------------------------------------------------------------------

def test_settings_derives_paths(tmp_path, minimal_config, monkeypatch):
    monkeypatch.setattr(main_module, "SCRIPT_DIR", tmp_path)
    cfg = Settings(minimal_config, "ELBA", "EVENT")

    assert cfg.MSEED_INPUT_DIR == tmp_path / "raw" / "ELBA" / "EVENT"
    assert cfg.DATA_ROOT == tmp_path / "proceeded" / "ELBA" / "EVENT"
    assert cfg.OUTPUT_ROOT == tmp_path / "results" / "ELBA" / "EVENT" / "ENZ"


def test_settings_creates_output_root(tmp_path, minimal_config, monkeypatch):
    monkeypatch.setattr(main_module, "SCRIPT_DIR", tmp_path)
    cfg = Settings(minimal_config, "ELBA", "EVENT")
    assert cfg.OUTPUT_ROOT.is_dir()


def test_settings_derives_sample_counts(tmp_path, minimal_config, monkeypatch):
    monkeypatch.setattr(main_module, "SCRIPT_DIR", tmp_path)
    cfg = Settings(minimal_config, "ELBA", "EVENT")

    assert cfg.WinSize == int(200 * 5.0) == 1000
    assert cfg.StepSize == int(50 * 5.0) == 250
    assert cfg.PREV_LEN == int(150 * 5.0) == 750


def test_settings_exposes_feature_block(tmp_path, minimal_config, monkeypatch):
    monkeypatch.setattr(main_module, "SCRIPT_DIR", tmp_path)
    cfg = Settings(minimal_config, "ELBA", "EVENT")
    assert "rosenstein" in cfg.FEATURES
    assert cfg.FEATURES["rosenstein"]["mean_period"] == 1.0


# ---------------------------------------------------------------------------
# _collect_jobs
# ---------------------------------------------------------------------------

def test_collect_jobs_finds_all(tmp_path):
    raw = tmp_path / "raw"
    (raw / "ELBA" / "EV1").mkdir(parents=True)
    (raw / "ELBA" / "EV2").mkdir(parents=True)
    (raw / "ELZG" / "EV1").mkdir(parents=True)
    (raw / "ignore_me.txt").write_text("")

    jobs = _collect_jobs(raw)
    assert set(jobs) == {("ELBA", "EV1"), ("ELBA", "EV2"), ("ELZG", "EV1")}


def test_collect_jobs_returns_empty_when_missing(tmp_path):
    assert _collect_jobs(tmp_path / "does_not_exist") == []


def test_collect_jobs_is_sorted(tmp_path):
    raw = tmp_path / "raw"
    for s in ("ZZZ", "AAA", "MMM"):
        for e in ("E2", "E1"):
            (raw / s / e).mkdir(parents=True)
    jobs = _collect_jobs(raw)
    assert jobs == sorted(jobs)


# ---------------------------------------------------------------------------
# Settings — worker count must come from the config
# ---------------------------------------------------------------------------

def test_settings_reads_n_jobs_from_config(tmp_path, minimal_config, monkeypatch):
    monkeypatch.setattr(main_module, "SCRIPT_DIR", tmp_path)
    cfg = Settings(minimal_config, "ELBA", "EVENT")
    assert cfg.N_JOBS == 2


def test_run_single_honours_config_n_jobs(tmp_path, minimal_config, monkeypatch):
    """Regression: _run_single used to hard-code n_jobs=-1, so the configured
    value was silently discarded for every single-station run."""
    monkeypatch.setattr(main_module, "SCRIPT_DIR", tmp_path)
    seen = {}
    monkeypatch.setattr(main_module, "run_mseed_preprocessing",
                        lambda cfg: seen.setdefault("pre", cfg.N_JOBS))
    monkeypatch.setattr(main_module, "run_feature_extraction",
                        lambda cfg: seen.setdefault("fe", cfg.N_JOBS))

    _run_single(minimal_config, "ELBA", "EVENT")
    assert seen == {"pre": 2, "fe": 2}


def test_run_single_override_wins(tmp_path, minimal_config, monkeypatch):
    """Batch mode passes n_jobs=1 to avoid oversubscribing the outer pool."""
    monkeypatch.setattr(main_module, "SCRIPT_DIR", tmp_path)
    seen = {}
    monkeypatch.setattr(main_module, "run_mseed_preprocessing",
                        lambda cfg: seen.setdefault("pre", cfg.N_JOBS))
    monkeypatch.setattr(main_module, "run_feature_extraction", lambda cfg: None)

    _run_single(minimal_config, "ELBA", "EVENT", n_jobs=1)
    assert seen["pre"] == 1


# ---------------------------------------------------------------------------
# _find_project_root
# ---------------------------------------------------------------------------

def test_find_project_root_honours_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAOS_ROOT", str(tmp_path))
    assert _find_project_root() == tmp_path.resolve()


def test_find_project_root_finds_repo_config(monkeypatch):
    """Without CHAOS_ROOT, the repo root (the one holding config.json) wins."""
    monkeypatch.delenv("CHAOS_ROOT", raising=False)
    root = _find_project_root()
    assert (root / "config.json").is_file()


# ---------------------------------------------------------------------------
# The shipped config.json must actually be loadable by Settings
# ---------------------------------------------------------------------------

def test_shipped_config_is_valid(tmp_path, monkeypatch):
    real = load_config()
    monkeypatch.setattr(main_module, "SCRIPT_DIR", tmp_path)
    cfg = Settings(real, "ELBA", "EVENT")
    assert cfg.WinSize > 0 and cfg.StepSize > 0
    assert set(cfg.FEATURES) == {"wolf", "rosenstein", "sample_entropy", "corr_dim"}


def test_shipped_config_rosenstein_window_is_usable():
    """Regression: config has twice shipped a two-sample short window
    (slope=[0, 0.2, 5, 2], then slope=[0, 0.2, 4, 10]), which collapses the
    short fit to a line through two points and makes every R-squared a
    meaningless 1.0. The whole results CSV inherits the artefact."""
    from chaos.chaotic_features import MIN_FIT_POINTS, split_slope_windows

    real = load_config()
    fe = real["feature_extraction"]
    ros = fe["features"]["rosenstein"]

    for label, window in zip(("short", "long"),
                             split_slope_windows(ros["slope"])):
        if window is None:
            continue
        lo = round(window[0] * ros["mean_period"] * fe["fs"])
        hi = round(window[1] * ros["mean_period"] * fe["fs"])
        assert hi - lo + 1 >= MIN_FIT_POINTS, (
            f"{label} fit window {window} spans {hi - lo + 1} samples at "
            f"fs={fe['fs']}; a line through <{MIN_FIT_POINTS} points always "
            "reports R2=1"
        )


def test_shipped_config_passes_validation(tmp_path, monkeypatch):
    """The config that ships must survive the same checks a user's would."""
    monkeypatch.setattr(main_module, "SCRIPT_DIR", tmp_path)
    Settings(load_config(), "ELBA", "EVENT").validate()


# ---------------------------------------------------------------------------
# Settings.validate — reject configs that produce plausible-looking garbage
# ---------------------------------------------------------------------------

def _settings(config, tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "SCRIPT_DIR", tmp_path)
    return Settings(config, "ELBA", "EVENT")


def test_validate_rejects_degenerate_rosenstein_window(minimal_config, tmp_path,
                                                       monkeypatch):
    """The bug that produced 53,535 rows of R2=1.0 must not load at all."""
    minimal_config["feature_extraction"]["features"]["rosenstein"]["slope"] = \
        [0, 0.2, 4, 10]
    with pytest.raises(ValueError, match="rosenstein short window"):
        _settings(minimal_config, tmp_path, monkeypatch)


def test_validate_rejects_degenerate_long_window(minimal_config, tmp_path,
                                                 monkeypatch):
    minimal_config["feature_extraction"]["features"]["rosenstein"]["slope"] = \
        [0, 1, 4, 4.1]
    with pytest.raises(ValueError, match="rosenstein long window"):
        _settings(minimal_config, tmp_path, monkeypatch)


def test_validate_rejects_freqmax_above_decimated_nyquist(minimal_config,
                                                          tmp_path,
                                                          monkeypatch):
    """Decimation runs with no_filter=True, so a passband reaching fs/2 folds
    straight back into the band the features are measured on."""
    minimal_config["preprocessing"]["freq_max"] = 3.0  # fs is 5.0
    with pytest.raises(ValueError, match="Nyquist"):
        _settings(minimal_config, tmp_path, monkeypatch)


def test_validate_rejects_non_advancing_window(minimal_config, tmp_path,
                                               monkeypatch):
    minimal_config["feature_extraction"]["step_sec"] = 0
    with pytest.raises(ValueError, match="never advance"):
        _settings(minimal_config, tmp_path, monkeypatch)


def test_validate_rejects_empty_channels(minimal_config, tmp_path, monkeypatch):
    minimal_config["feature_extraction"]["channels"] = []
    with pytest.raises(ValueError, match="channels is empty"):
        _settings(minimal_config, tmp_path, monkeypatch)


def test_validate_rejects_inverted_band(minimal_config, tmp_path, monkeypatch):
    minimal_config["preprocessing"]["freq_min"] = 2.0
    minimal_config["preprocessing"]["freq_max"] = 0.1
    with pytest.raises(ValueError, match="freq_min < freq_max"):
        _settings(minimal_config, tmp_path, monkeypatch)


def test_validate_reports_every_problem_at_once(minimal_config, tmp_path,
                                                monkeypatch):
    """One run, one list -- not one error per re-run."""
    minimal_config["feature_extraction"]["channels"] = []
    minimal_config["feature_extraction"]["step_sec"] = 0
    with pytest.raises(ValueError) as exc:
        _settings(minimal_config, tmp_path, monkeypatch)
    assert "channels is empty" in str(exc.value)
    assert "never advance" in str(exc.value)
