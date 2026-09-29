"""Entry point for the CHAOS feature-extraction pipeline.

All non-derived settings (event identifiers, folder names, filter bands,
window sizes, per-metric parameters) are read from ``config.json`` at the
project root. The file is re-read on every execution.
"""

import copy
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from chaos.extraction import run_feature_extraction


def _find_project_root() -> Path:
    """Locates the directory holding ``config.json`` and the data folders.

    Checked in order: the ``CHAOS_ROOT`` environment variable, the first
    ancestor of this file that contains a ``config.json`` (the layout when
    running from a source checkout), then the current working directory (the
    layout when running the installed ``chaos`` command).

    Returns:
        The resolved project root.
    """
    env_root = os.environ.get("CHAOS_ROOT")
    if env_root:
        return Path(env_root).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "config.json").is_file():
            return parent
    return Path.cwd()


SCRIPT_DIR = _find_project_root()
CONFIG_PATH = SCRIPT_DIR / "config.json"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Loads the JSON configuration file.

    Args:
        path: Path to the JSON file. Defaults to ``config.json`` at the
            project root.

    Returns:
        Parsed configuration as a nested dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class Settings:
    """Per-run configuration object.

    Combines the project-wide config dictionary with one run: either a
    ``(station, earthquake_name)`` folder under ``raw``, or a station and a
    :class:`chaos.periods.Period` cut out of the continuous data folder.
    Derived quantities are computed once here so downstream code never
    recomputes them.

    Args:
        config: Parsed project configuration (see :func:`load_config`).
        station: Station code (e.g. ``"ELBA"``).
        earthquake_name: Folder name identifying the earthquake event. Unused
            when ``period`` is given, where the period's label names the run.
        period: Date range to run on instead of an event folder.
        make_dirs: Create the output folder (off for commands that only
            inspect settings).
    """

    def __init__(self, config: dict, station: str,
                 earthquake_name: str | None = None, period=None,
                 make_dirs: bool = True):
        from chaos.mean_period import data_pattern, project_path, setting

        self._config = config
        self.SCRIPT_DIR = SCRIPT_DIR

        self.STATION = station
        self.PERIOD = period
        self.EARTHQUAKE_NAME = period.label if period else earthquake_name
        # Quiet runs are labelled "quiet_7d/20240612"; the CSV name is flat.
        self.RUN_NAME = self.EARTHQUAKE_NAME.replace("/", "_")
        self.MEAN_PERIOD_SOURCE = None

        paths = config["paths"]
        if period is None:
            self.MSEED_INPUT_DIR = (
                self.SCRIPT_DIR / paths["raw_dir"] / station / earthquake_name
            )
            self.DATA_PATTERN = "*.mseed"
        else:
            self.MSEED_INPUT_DIR = project_path(
                self.SCRIPT_DIR, setting(config, "data_dir", "data"))
            self.DATA_PATTERN = data_pattern(config, station)
        self.OUTPUT_ROOT = (
            self.SCRIPT_DIR / paths["results_dir"] / station
            / self.EARTHQUAKE_NAME / paths["results_subdir"]
        )
        if make_dirs:
            self.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        # Period runs share one cache per station: day pieces are the same
        # whichever period they are read for.
        self.CACHE_ROOT = (
            self.SCRIPT_DIR / paths.get("cache_dir", "cache") / station
            / (self.EARTHQUAKE_NAME if period is None else "_pieces")
        )
        self.CACHE_ENABLED = bool(config.get("cache", {}).get("enabled", False))

        pre = config["preprocessing"]
        self.FREQMIN = float(pre["freq_min"])
        self.FREQMAX = float(pre["freq_max"])
        self.GAP_THRESHOLD = float(pre["gap_threshold_sec"])
        self.PREPROCESS_CHANNELS = list(pre["channels"])
        self.FILTER_PAD_SEC = (
            float(pre["filter_pad_sec"]) if pre.get("filter_pad_sec") else None
        )

        fe = config["feature_extraction"]
        self.Fs = float(fe["fs"])
        self.WIN_SEC = float(fe["win_sec"])
        self.STEP_SEC = float(fe["step_sec"])
        self.N_JOBS = int(fe["n_jobs"])
        self.WARMUP_COUNT = int(fe["warmup_count"])
        self.CHANNELS = list(fe["channels"])
        # Copied so filling in an estimated mean period never leaks into the
        # shared config dict, which other runs in the same process reuse.
        self.FEATURES = copy.deepcopy(fe["features"])
        # A seam longer than one window cannot be bridged by any window
        # anyway, so that is the natural point to stop representing it.
        self.MAX_GAP_SEC = (
            float(fe["max_gap_sec"]) if fe.get("max_gap_sec") else self.WIN_SEC
        )

        self.WinSize = int(self.WIN_SEC * self.Fs)
        self.StepSize = int(self.STEP_SEC * self.Fs)

        self.validate()

    @property
    def needs_mean_period(self) -> bool:
        """True while ``rosenstein.mean_period`` is unset and must be estimated."""
        ros = self.FEATURES.get("rosenstein") or {}
        return ros.get("mean_period") is None

    def resolve_mean_period(self, refresh: bool = False) -> None:
        """Fills in an unset Rosenstein mean period from catalog-quiet data.

        A value given in config.json is always used as is; only a missing or
        ``null`` one is estimated (see :mod:`chaos.mean_period`). The settings
        are validated again afterwards, since the fit windows are measured in
        mean periods and only now have a length in samples.
        """
        if not self.needs_mean_period:
            self.MEAN_PERIOD_SOURCE = "config"
            return
        from chaos.mean_period import resolve_mean_period

        if refresh:
            value = resolve_mean_period(self, self._config, refresh=True)
        else:
            value = resolve_mean_period(self, self._config)
        self.FEATURES["rosenstein"]["mean_period"] = value
        print(f"[mean_period] {self.STATION}/{self.EARTHQUAKE_NAME}: using "
              f"{value:.3f} s")
        self.validate()

    def export(self, extra: dict | None = None) -> Path:
        """Writes ``settings.json`` beside the results.

        It holds the configuration exactly as this run used it — command-line
        overrides applied, the mean period filled in, the period resolved — so
        a results folder can always be traced back to, and re-run from, its
        own settings.

        Args:
            extra: Additional run details to record (e.g. quiet selection).

        Returns:
            The path written.
        """
        from datetime import datetime, timezone

        config = copy.deepcopy(self._config)
        config["feature_extraction"]["features"] = copy.deepcopy(self.FEATURES)
        config["single_run"] = {"station": self.STATION,
                                "earthquake_name": (None if self.PERIOD
                                                    else self.EARTHQUAKE_NAME)}
        config["process_all"] = False
        target = config.setdefault("target", {})
        if self.PERIOD is not None:
            target["start_date"] = str(self.PERIOD.t0)
            target["duration_days"] = self.PERIOD.days
        else:
            target["start_date"] = None
        target["quiet_window"] = False
        run = {
            "station": self.STATION,
            "label": self.EARTHQUAKE_NAME,
            "mode": "period" if self.PERIOD else "event",
            "period_start": str(self.PERIOD.t0) if self.PERIOD else None,
            "period_end": str(self.PERIOD.t1) if self.PERIOD else None,
            "mean_period_s": self.FEATURES["rosenstein"].get("mean_period"),
            "mean_period_source": self.MEAN_PERIOD_SOURCE,
            "written_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds"),
            "command": getattr(self, "COMMAND", None),
            **(extra or {}),
        }
        path = self.OUTPUT_ROOT / "settings.json"
        path.write_text(json.dumps({"run": run, "config": config}, indent=2,
                                   default=str), encoding="utf-8")
        return path

    def validate(self) -> None:
        """Rejects settings that would yield silently meaningless output.

        These are all mistakes that produce a full, plausible-looking results
        CSV rather than an error: a fit window too short to be a fit, a
        passband above the Nyquist frequency of the decimated signal, a
        non-advancing window. Catching them here costs one check per run and
        saves re-deriving a whole dataset.

        Raises:
            ValueError: If any setting cannot produce usable features.
        """
        from chaos.chaotic_features import MIN_FIT_POINTS, split_slope_windows

        problems: list[str] = []

        # Keys from the two-stage layout. Left in place they would be read as
        # settings that no longer do anything, so they are named explicitly
        # rather than ignored.
        for section, key, replacement in (
            ("paths", "processed_dir",
             "there is no intermediate tree any more; remove it"),
            ("feature_extraction", "prev_sec",
             "windows now slide over one continuous recording, so there is no "
             "carry-over to size; remove it"),
            ("preprocessing", "window_sec",
             "the recording is no longer cut into fixed windows before "
             "extraction; remove it"),
        ):
            if key in self._config.get(section, {}):
                problems.append(f"{section}.{key} is obsolete — {replacement}")

        if self.Fs <= 0:
            problems.append(f"feature_extraction.fs must be > 0, got {self.Fs}")
        if self.StepSize > self.WinSize:
            problems.append(
                f"step_sec={self.STEP_SEC} exceeds win_sec={self.WIN_SEC}; "
                "windows would skip samples entirely"
            )
        if self.MAX_GAP_SEC < 0:
            problems.append(
                f"feature_extraction.max_gap_sec must be >= 0, got "
                f"{self.MAX_GAP_SEC}"
            )
        if self.WinSize < 2:
            problems.append(
                f"win_sec={self.WIN_SEC} at fs={self.Fs} is {self.WinSize} "
                "samples; a window needs at least 2"
            )
        if self.StepSize < 1:
            problems.append(
                f"step_sec={self.STEP_SEC} at fs={self.Fs} is {self.StepSize} "
                "samples; the window would never advance"
            )
        if not self.CHANNELS:
            problems.append("feature_extraction.channels is empty")
        if not self.PREPROCESS_CHANNELS:
            problems.append("preprocessing.channels is empty")

        if not 0 < self.FREQMIN < self.FREQMAX:
            problems.append(
                f"need 0 < freq_min < freq_max, got {self.FREQMIN} and "
                f"{self.FREQMAX}"
            )
        elif self.Fs > 0 and self.FREQMAX >= self.Fs / 2:
            # Decimation runs with no_filter=True, so the bandpass is the only
            # anti-alias filter in the chain. A passband reaching the decimated
            # Nyquist folds energy back into the band being measured.
            problems.append(
                f"freq_max={self.FREQMAX} is at or above the Nyquist frequency "
                f"of the decimated signal (fs/2 = {self.Fs / 2}); decimation "
                "would alias it back into the passband"
            )

        ros = self.FEATURES.get("rosenstein")
        if ros is None:
            problems.append("feature_extraction.features.rosenstein is missing")
        elif ros.get("mean_period") is None:
            # Estimated later by resolve_mean_period, which validates again.
            try:
                split_slope_windows(ros["slope"])
            except ValueError as exc:
                problems.append(str(exc))
            from chaos.mean_period import setting
            if not setting(self._config, "catalog"):
                problems.append(
                    "rosenstein.mean_period is not set, so it must be "
                    "estimated from quiet data; set paths.catalog or give "
                    "a value"
                )
        else:
            mean_period = float(ros["mean_period"])
            if mean_period <= 0:
                problems.append(
                    f"rosenstein.mean_period must be > 0, got {mean_period}"
                )
            else:
                try:
                    windows = split_slope_windows(ros["slope"])
                except ValueError as exc:
                    problems.append(str(exc))
                    windows = ()
                for label, window in zip(("short", "long"), windows):
                    if window is None:
                        continue
                    lo = int(round(window[0] * mean_period * self.Fs))
                    hi = int(round(window[1] * mean_period * self.Fs))
                    n_points = hi - lo + 1
                    if hi <= lo or n_points < MIN_FIT_POINTS:
                        problems.append(
                            f"rosenstein {label} window {tuple(window)} spans "
                            f"{max(0, n_points)} sample(s) at fs={self.Fs} and "
                            f"mean_period={mean_period}; a straight line "
                            f"through fewer than {MIN_FIT_POINTS} points always "
                            "reports R2=1. Widen it or raise fs."
                        )

        if self.PERIOD is not None and \
                self.PERIOD.t1 - self.PERIOD.t0 < self.WIN_SEC:
            problems.append(
                f"the target period ({self.PERIOD.days:g} days) is shorter "
                f"than one window (win_sec={self.WIN_SEC})"
            )

        if problems:
            raise ValueError(
                "Invalid configuration for "
                f"{self.STATION}/{self.EARTHQUAKE_NAME}:\n  - "
                + "\n  - ".join(problems)
            )


def _collect_jobs(raw_root: Path) -> list[tuple[str, str]]:
    """Returns every ``(station, earthquake_name)`` folder pair under ``raw``.

    Args:
        raw_root: Directory containing one sub-folder per station.

    Returns:
        A list of ``(station, earthquake_name)`` tuples.
    """
    jobs: list[tuple[str, str]] = []
    if not raw_root.exists():
        return jobs

    for station_dir in sorted(raw_root.iterdir()):
        if not station_dir.is_dir():
            continue
        for eq_dir in sorted(station_dir.iterdir()):
            if not eq_dir.is_dir():
                continue
            jobs.append((station_dir.name, eq_dir.name))
    return jobs


def _run_single(config: dict, station: str, earthquake_name: str | None,
                n_jobs: int | None = None, period=None,
                extra: dict | None = None, command: str | None = None) -> int:
    """Runs the pipeline for one event folder or one period.

    Args:
        config: Parsed project configuration.
        station: Station code.
        earthquake_name: Earthquake folder name (event mode).
        n_jobs: Worker count, overriding the configured ``n_jobs``. Pass ``1``
            when jobs are already parallelised externally to avoid
            oversubscription; leave as ``None`` to honour the config.
        period: :class:`chaos.periods.Period` to run on instead of a folder.
        extra: Additional details recorded in ``settings.json``.
        command: The command line, recorded in ``settings.json``.

    Returns:
        Number of feature rows written.
    """
    cfg = Settings(config, station, earthquake_name, period=period)
    cfg.COMMAND = command
    cfg.resolve_mean_period()
    if n_jobs is not None:
        cfg.N_JOBS = n_jobs
    path = cfg.export(extra)
    print(f"[settings] {path}")
    if not run_feature_extraction(cfg):
        return 0
    out = cfg.OUTPUT_ROOT / f"{cfg.STATION}_{cfg.RUN_NAME}_features.csv"
    if not out.is_file():
        return 0
    with open(out, encoding="utf-8") as fh:
        return max(0, sum(1 for _ in fh) - 1)


def _job_name(job) -> str:
    station, eq, period = job
    return f"{station} / {period.label if period else eq}"


def plan_jobs(config: dict) -> tuple[list, dict]:
    """Works out every run the configuration asks for.

    Returns:
        Tuple ``(jobs, quiet)``: ``jobs`` is a list of ``(station,
        earthquake_name, period)``; ``quiet`` maps station to ``(periods,
        info)`` for quiet-window runs.

    Raises:
        ValueError: If the target settings are unusable.
    """
    from chaos.periods import discover_stations, quiet_periods, target_period

    target = config.get("target") or {}
    process_all = bool(config.get("process_all", False))
    single = config.get("single_run") or {}

    if target.get("quiet_window"):
        days = target.get("duration_days")
        if not days:
            raise ValueError("target.quiet_window needs target.duration_days")
        stations = (discover_stations(SCRIPT_DIR, config) if process_all
                    else [single["station"]])
        jobs, quiet = [], {}
        for station in stations:
            try:
                periods, info = quiet_periods(SCRIPT_DIR, config, station, days)
            except ValueError as exc:
                print(f"[SKIP] {station}: {exc}")
                continue
            quiet[station] = (periods, info)
            jobs.extend((station, None, p) for p in periods)
        return jobs, quiet

    if target.get("start_date"):
        if not target.get("duration_days"):
            raise ValueError("target.start_date needs target.duration_days")
        period = target_period(target["start_date"], target["duration_days"])
        stations = (discover_stations(SCRIPT_DIR, config) if process_all
                    else [single["station"]])
        return [(s, None, period) for s in stations], {}

    if not process_all:
        return [(single["station"], single["earthquake_name"], None)], {}
    raw_root = SCRIPT_DIR / config["paths"]["raw_dir"]
    return [(s, e, None) for s, e in _collect_jobs(raw_root)], {}


def run(config: dict, command: str | None = None, dry_run: bool = False,
        refresh_mean_period: bool = False) -> None:
    """Plans and runs every job the configuration asks for."""
    jobs, quiet = plan_jobs(config)
    if not jobs:
        print("[WARNING] Nothing to run: no event folders, stations or quiet "
              "periods were found")
        return

    if dry_run:
        print(f"[DRY RUN] {len(jobs)} run(s):")
        for job in jobs:
            station, eq, period = job
            where = (f"{period.t0} -> {period.t1}" if period
                     else str(SCRIPT_DIR / config["paths"]["raw_dir"]
                              / station / eq))
            print(f"  {_job_name(job):40s} {where}")
        return

    for station, (periods, info) in quiet.items():
        from chaos.periods import write_periods
        group = periods[0].label.split("/")[0]
        write_periods(SCRIPT_DIR / config["paths"]["results_dir"] / station
                      / group / "periods.csv", periods, info)

    # Estimating the mean period here, one station at a time, fills its cache
    # before the pool starts; otherwise every run of a station would race to
    # estimate the same station-wide value at once.
    failed, resolved = [], {}
    for job in jobs:
        station, eq, period = job
        try:
            if station not in resolved:
                Settings(config, station, eq, period=period,
                         make_dirs=False) \
                    .resolve_mean_period(refresh=refresh_mean_period)
                resolved[station] = True
        except Exception as exc:
            print(f"[SKIP] {station}: {exc}")
            resolved[station] = False
        if not resolved[station]:
            failed.append(job)
    jobs = [j for j in jobs if j not in failed]
    if not jobs:
        return

    def extra_for(job):
        station, _, period = job
        if station in quiet and period is not None:
            return {"quiet_selection": quiet[station][1]}
        return None

    results = {}
    if len(jobs) == 1:
        job = jobs[0]
        results[job] = _run_single(config, job[0], job[1], period=job[2],
                                   extra=extra_for(job), command=command)
    else:
        print(f"[BATCH] {len(jobs)} job(s). Processing in parallel...\n")
        max_workers = min(len(jobs), os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(_run_single, config, s, e, 1, p, extra_for((s, e, p)),
                          command): (s, e, p)
                for s, e, p in jobs
            }
            for i, fut in enumerate(as_completed(futures), 1):
                job = futures[fut]
                try:
                    results[job] = fut.result()
                    print(f"[{i}/{len(jobs)}] OK   {_job_name(job)}")
                except Exception as exc:
                    results[job] = exc
                    print(f"[{i}/{len(jobs)}] FAIL {_job_name(job)}: {exc}")
        print(f"\n[BATCH] All {len(jobs)} job(s) completed.")

    for station, (periods, _) in quiet.items():
        rows = []
        for period in periods:
            outcome = results.get((station, None, period))
            rows.append({
                "label": period.label.split("/")[-1],
                "start": str(period.t0), "end": str(period.t1),
                "status": ("fail" if isinstance(outcome, Exception)
                           else "skipped" if outcome is None
                           else "ok" if outcome else "empty"),
                "rows": outcome if isinstance(outcome, int) else 0,
                "error": str(outcome) if isinstance(outcome, Exception) else "",
            })
        import pandas as pd
        group = periods[0].label.split("/")[0]
        path = (SCRIPT_DIR / config["paths"]["results_dir"] / station / group
                / "summary.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"[quiet] {station}: summary in {path}")


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point (``chaos``); see :mod:`chaos.cli`."""
    from chaos.cli import main as cli_main
    cli_main(argv)


if __name__ == "__main__":
    main()
