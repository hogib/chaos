"""Entry point for the CHAOS feature-extraction pipeline.

All non-derived settings (event identifiers, folder names, filter bands,
window sizes, per-metric parameters) are read from ``config.json`` at the
project root. The file is re-read on every execution.
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from chaos.extraction import run_feature_extraction
from chaos.preprocess import run_mseed_preprocessing


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

    Combines the project-wide config dictionary with a specific
    ``(station, earthquake_name)`` pair. Derived quantities are computed once
    here so downstream code never recomputes them.

    Args:
        config: Parsed project configuration (see :func:`load_config`).
        station: Station code (e.g. ``"ELBA"``).
        earthquake_name: Folder name identifying the earthquake event.
    """

    def __init__(self, config: dict, station: str, earthquake_name: str):
        self._config = config
        self.SCRIPT_DIR = SCRIPT_DIR

        self.STATION = station
        self.EARTHQUAKE_NAME = earthquake_name

        paths = config["paths"]
        self.MSEED_INPUT_DIR = (
            self.SCRIPT_DIR / paths["raw_dir"] / station / earthquake_name
        )
        self.DATA_ROOT = (
            self.SCRIPT_DIR / paths["processed_dir"] / station / earthquake_name
        )
        self.OUTPUT_ROOT = (
            self.SCRIPT_DIR / paths["results_dir"] / station / earthquake_name
            / paths["results_subdir"]
        )
        self.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

        pre = config["preprocessing"]
        self.PREPROCESS_WINDOW_SEC = float(pre["window_sec"])
        self.FREQMIN = float(pre["freq_min"])
        self.FREQMAX = float(pre["freq_max"])
        self.GAP_THRESHOLD = float(pre["gap_threshold_sec"])
        self.PREPROCESS_CHANNELS = list(pre["channels"])

        fe = config["feature_extraction"]
        self.Fs = float(fe["fs"])
        self.WIN_SEC = float(fe["win_sec"])
        self.STEP_SEC = float(fe["step_sec"])
        self.PREV_SEC = float(fe["prev_sec"])
        self.N_JOBS = int(fe["n_jobs"])
        self.WARMUP_COUNT = int(fe["warmup_count"])
        self.CHANNELS = list(fe["channels"])
        self.FEATURES = fe["features"]

        self.WinSize = int(self.WIN_SEC * self.Fs)
        self.StepSize = int(self.STEP_SEC * self.Fs)
        self.PREV_LEN = int(self.PREV_SEC * self.Fs)

        self.validate()

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

        if self.Fs <= 0:
            problems.append(f"feature_extraction.fs must be > 0, got {self.Fs}")
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
        if self.PREV_LEN < 0:
            problems.append(f"prev_sec must be >= 0, got {self.PREV_SEC}")
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


def _run_single(config: dict, station: str, earthquake_name: str,
                n_jobs: int | None = None) -> None:
    """Runs both pipeline stages for a single station/earthquake pair.

    Args:
        config: Parsed project configuration.
        station: Station code.
        earthquake_name: Earthquake folder name.
        n_jobs: Worker count used inside feature extraction, overriding the
            configured ``n_jobs``. Pass ``1`` when jobs are already
            parallelised externally to avoid oversubscription; leave as
            ``None`` to honour the config.
    """
    cfg = Settings(config, station, earthquake_name)
    if n_jobs is not None:
        cfg.N_JOBS = n_jobs
    run_mseed_preprocessing(cfg)
    run_feature_extraction(cfg)


def main() -> None:
    """Reads ``config.json`` and dispatches the pipeline."""
    config = load_config()

    if not config.get("process_all", False):
        single = config["single_run"]
        _run_single(config, single["station"], single["earthquake_name"])
        return

    raw_root = SCRIPT_DIR / config["paths"]["raw_dir"]
    jobs = _collect_jobs(raw_root)
    if not jobs:
        print(f"[WARNING] No station/earthquake folders found under {raw_root}")
        return

    print(f"[BATCH] {len(jobs)} job(s) found. Processing in parallel...\n")

    cpu_count = os.cpu_count() or 1
    max_workers = min(len(jobs), cpu_count)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_run_single, config, s, e, 1): (s, e) for s, e in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            station, eq = futures[fut]
            try:
                fut.result()
                print(f"[{i}/{len(jobs)}] OK   {station} / {eq}")
            except Exception as exc:
                print(f"[{i}/{len(jobs)}] FAIL {station} / {eq}: {exc}")

    print(f"\n[BATCH] All {len(jobs)} job(s) completed.")


if __name__ == "__main__":
    main()
