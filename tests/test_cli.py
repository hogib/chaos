"""Tests for the chaos command line."""

import copy
import json

import pytest

from chaos import cli
from chaos import pipeline as main_module
from tests.test_mean_period import BASE


def _args(*argv):
    argv = list(argv)
    if not argv or argv[0] not in ("run", "quiet", "mean-period", "config"):
        argv = ["run", *argv]
    return cli.build_parser().parse_args(argv)


def _config():
    config = copy.deepcopy(BASE)
    config["target"] = {"start_date": None, "duration_days": 7,
                        "quiet_window": False}
    config["quiet_window"] = {"samples": 15, "min_mag": 3.0,
                              "max_dist_km": 300, "seed": 0}
    return config


def test_no_flags_leaves_config_alone():
    config = _config()
    assert cli.apply_overrides(config, _args()) == config


def test_flags_override_config():
    out = cli.apply_overrides(_config(), _args(
        "--station", "ELBA", "--start", "2025-04-16", "--days", "3",
        "--samples", "5", "--min-mag", "3.5", "--max-dist", "200",
        "--seed", "7", "--jobs", "4", "--all", "--mean-period", "2.5"))
    assert out["single_run"]["station"] == "ELBA"
    assert out["target"] == {"start_date": "2025-04-16", "duration_days": 3.0,
                             "quiet_window": False}
    assert out["quiet_window"] == {"samples": 5, "min_mag": 3.5,
                                   "max_dist_km": 200.0, "seed": 7}
    assert out["feature_extraction"]["n_jobs"] == 4
    assert out["process_all"] is True
    assert out["feature_extraction"]["features"]["rosenstein"][
        "mean_period"] == 2.5


def test_mean_period_auto_clears_the_value():
    config = _config()
    config["feature_extraction"]["features"]["rosenstein"]["mean_period"] = 1.0
    out = cli.apply_overrides(config, _args("--mean-period", "auto"))
    assert out["feature_extraction"]["features"]["rosenstein"][
        "mean_period"] is None
    with pytest.raises(SystemExit):
        _args("--mean-period", "-1")


def test_event_switches_back_to_event_mode():
    config = _config()
    config["target"].update(start_date="2025-01-01", quiet_window=True)
    out = cli.apply_overrides(config, _args("--event", "EQ1"))
    assert out["single_run"]["earthquake_name"] == "EQ1"
    assert out["target"]["start_date"] is None
    assert out["target"]["quiet_window"] is False


def test_quiet_flags():
    out = cli.apply_overrides(_config(), _args("--quiet"))
    assert out["target"]["quiet_window"] is True
    config = _config()
    config["target"]["quiet_window"] = True
    out = cli.apply_overrides(config, _args("--no-quiet"))
    assert out["target"]["quiet_window"] is False


def test_set_parses_json_and_flags_win():
    out = cli.apply_overrides(_config(), _args(
        "--set", "feature_extraction.win_sec=300",
        "--set", "feature_extraction.channels=[\"N\",\"E\"]",
        "--set", "single_run.station=CAME",
        "--station", "ELBA"))
    assert out["feature_extraction"]["win_sec"] == 300
    assert out["feature_extraction"]["channels"] == ["N", "E"]
    assert out["single_run"]["station"] == "ELBA"


def test_set_rejects_unknown_keys_but_allows_catalog_options():
    with pytest.raises(ValueError, match="unknown key"):
        cli.apply_overrides(_config(), _args("--set", "paths.nope=1"))
    with pytest.raises(ValueError, match="no section"):
        cli.apply_overrides(_config(), _args("--set", "nope.x=1"))
    with pytest.raises(ValueError, match="KEY=VALUE"):
        cli.apply_overrides(_config(), _args("--set", "paths"))
    out = cli.apply_overrides(_config(), _args(
        "--set", "mean_period_estimation.options.min_mag=2"))
    assert out["mean_period_estimation"]["options"]["min_mag"] == 2


def test_format_config_round_trips():
    config = _config()
    assert json.loads(cli.format_config(config)) == config
    assert '"slope": [0, 1, 4, 10]' in cli.format_config(config)


@pytest.fixture
def project(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_config()))
    monkeypatch.setattr(main_module, "SCRIPT_DIR", tmp_path)
    monkeypatch.setattr(main_module, "CONFIG_PATH", path)
    seen = {}

    def fake_run(config, command=None, dry_run=False,
                 refresh_mean_period=False):
        seen.update(config=config, command=command, dry_run=dry_run)
    monkeypatch.setattr(main_module, "run", fake_run)
    return path, seen


def test_main_defaults_to_run(project):
    path, seen = project
    cli.main([])
    assert seen["config"] == _config()
    cli.main(["--days", "3", "--dry-run"])
    assert seen["config"]["target"]["duration_days"] == 3.0
    assert seen["dry_run"] is True
    assert seen["command"] == "chaos run --days 3 --dry-run"
    # Nothing was written back without --save.
    assert json.loads(path.read_text()) == _config()


def test_save_writes_the_overrides(project):
    path, _ = project
    cli.main(["config", "--days", "3", "--station", "ELBA", "--save"])
    saved = json.loads(path.read_text())
    assert saved["target"]["duration_days"] == 3.0
    assert saved["single_run"]["station"] == "ELBA"


def test_config_prints_effective_settings(project, capsys):
    cli.main(["config", "--set", "feature_extraction.fs=10"])
    printed = json.loads(capsys.readouterr().out)
    assert printed["feature_extraction"]["fs"] == 10


def test_bad_override_exits_cleanly(project):
    with pytest.raises(SystemExit, match="unknown key"):
        cli.main(["--set", "paths.nope=1"])
