"""Sliding-window feature extraction over the streamed recording.

Windows slide across one continuous timeline rather than across hour-sized
files, so there is no carry-over buffer to splice and no window whose samples
straddle two independently filtered chunks.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from chaos.preprocess import grid_time
from chaos.recording import RollingBuffer, iter_blocks

from chaos.chaotic_features import (compute_corr_dim,
                                    compute_lyapunov_rosenstein,
                                    compute_lyapunov_wolf, compute_norm_stats,
                                    compute_raw_stats, compute_sample_entropy)

FEATURE_KEYS = [
    "ham_mean", "ham_std", "ham_min", "ham_max", "aktivite_std",
    "norm_min", "norm_max",
    "wolf_lye",
    "ros_short", "ros_r2", "ros_n_points", "ros_low_fit_quality",
    "ros_long", "ros_long_r2", "ros_long_n_points",
    "samp_ent", "corr_dim",
]


def compute_window(segment: np.ndarray, w_idx: int, cfg) -> dict:
    """Computes all features for a single sliding window.

    Args:
        segment: Raw signal window (1-D ndarray).
        w_idx: Window index (echoed unchanged into the result).
        cfg: Configuration object exposing ``Fs`` and ``FEATURES``.

    Returns:
        Dictionary mapping each feature name to its value.
    """
    result = {"w": w_idx}
    for key in FEATURE_KEYS:
        result[key] = np.nan

    if np.any(np.isnan(segment)):
        return result

    seg_std = float(np.std(segment, ddof=1))
    if seg_std == 0:
        return result

    result.update(compute_raw_stats(segment))

    try:
        result.update(compute_norm_stats(segment))
    except Exception:
        pass

    seg_norm = (segment - np.mean(segment)) / seg_std

    wolf_cfg = cfg.FEATURES["wolf"]
    try:
        result.update(compute_lyapunov_wolf(
            seg_norm, cfg.Fs,
            tau=wolf_cfg["tau"], m=wolf_cfg["m"],
            evolve=wolf_cfg["evolve"], min_samples=wolf_cfg["min_samples"],
        ))
    except Exception:
        pass

    ros_cfg = cfg.FEATURES["rosenstein"]
    try:
        result.update(compute_lyapunov_rosenstein(
            seg_norm,
            cfg.Fs,
            mean_period=ros_cfg["mean_period"],
            tau=ros_cfg["tau"],
            m=ros_cfg["m"],
            slope_ros=ros_cfg["slope"],
        ))
    except Exception as e:
        print(f"[rosenstein] {type(e).__name__}: {e}")

    se_cfg = cfg.FEATURES["sample_entropy"]
    try:
        result.update(compute_sample_entropy(seg_norm, m=se_cfg["m"], r=se_cfg["r"]))
    except Exception:
        pass

    cd_cfg = cfg.FEATURES["corr_dim"]
    try:
        result.update(compute_corr_dim(seg_norm, tau=cd_cfg["tau"], m=cd_cfg["m"]))
    except Exception:
        pass

    return result


BATCH_WINDOWS = 512


class _WindowNamer:
    """Assigns ``Window_ID`` and ``Time_min`` from a window's absolute time.

    Both columns describe the window's position inside the calendar hour its
    start falls in, which is what they meant before the pipeline became
    continuous — so anything already reading them keeps working.
    """

    def __init__(self):
        self._hour_key = None
        self._counter = 0

    def name(self, start_time, end_time):
        """Returns ``(window_id, time_min)`` for one window.

        Args:
            start_time: Window start as :class:`obspy.UTCDateTime`.
            end_time: Window end as :class:`obspy.UTCDateTime`.

        Returns:
            Tuple of the window's id and the minutes from the start of its
            hour to its end.
        """
        dt = start_time.datetime
        hour_key = (dt.year, dt.month, dt.day, dt.hour)
        if hour_key != self._hour_key:
            self._hour_key = hour_key
            self._counter = 0
        self._counter += 1

        hour_start = start_time - (dt.minute * 60 + dt.second
                                   + dt.microsecond / 1e6)
        window_id = (
            f"{dt.year:04d}_{dt.month:02d}_{dt.day:02d}_"
            f"{dt.hour:02d}_w{self._counter:02d}"
        )
        return window_id, round(float(end_time - hour_start) / 60.0, 3)


def _iso(utc_time) -> str:
    """Formats a :class:`obspy.UTCDateTime` as a compact ISO UTC string."""
    return utc_time.datetime.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _blank_row(channels):
    """Returns the per-channel feature cells of a window that was not computed."""
    return {
        f"{ch}_{key}": np.nan for ch in channels for key in FEATURE_KEYS
    }


def _flush(rows, out_path, first_write) -> bool:
    """Appends a batch of rows to the results CSV.

    Rows are written as they are produced rather than accumulated, so a
    year-long run does not have to hold its own results in memory.

    Args:
        rows: List of row dictionaries.
        out_path: Destination CSV.
        first_write: Whether the header still needs writing.

    Returns:
        ``False``, to be assigned back to the caller's ``first_write``.
    """
    if not rows:
        return first_write
    frame = pd.DataFrame(rows)
    frame.to_csv(
        out_path,
        index=False,
        mode="w" if first_write else "a",
        header=first_write,
    )
    return False


def run_feature_extraction(cfg) -> bool:
    """Streams the recording and writes one feature row per window.

    Args:
        cfg: Configuration object (see :class:`chaos.pipeline.Settings`).

    Returns:
        ``True`` if at least one row was written.
    """
    print("\n" + "=" * 50)
    print("FEATURE EXTRACTION (MSEED -> FEATURES, NO INTERMEDIATE FILES)")
    print("=" * 50)

    channels = list(cfg.CHANNELS)
    mseed_files = sorted(cfg.MSEED_INPUT_DIR.glob("*.mseed"))
    if not mseed_files:
        print(f"[SKIPPED] No .mseed files found in: {cfg.MSEED_INPUT_DIR}")
        return False

    print(f"Station   : {cfg.STATION} | Channels: {', '.join(channels)}")
    print(f"Window    : {cfg.WIN_SEC}s  | Step: {cfg.STEP_SEC}s  | Fs: {cfg.Fs} Hz")
    print(f"Files     : {len(mseed_files)}")

    out_path = cfg.OUTPUT_ROOT / f"{cfg.STATION}_{cfg.EARTHQUAKE_NAME}_features.csv"
    namer = _WindowNamer()
    buffer = RollingBuffer(channels, cfg.WinSize, cfg.StepSize)

    gap_lines: list[str] = []
    pending: list[tuple] = []
    rows: list[dict] = []
    first_write = True
    total_rows = 0
    emitted = 0

    with Parallel(n_jobs=cfg.N_JOBS, prefer="processes") as parallel:

        def run_batch():
            """Computes the queued windows and turns them into CSV rows."""
            nonlocal pending, rows, total_rows, first_write
            if not pending:
                return
            tasks = [
                delayed(compute_window)(segments[ch], idx, cfg)
                for idx, (_, segments) in enumerate(pending)
                for ch in channels
            ]
            flat = parallel(tasks)
            for w, (meta, _) in enumerate(pending):
                row = dict(meta)
                for c, ch in enumerate(channels):
                    res = flat[w * len(channels) + c]
                    for key in FEATURE_KEYS:
                        row[f"{ch}_{key}"] = res.get(key, np.nan)
                rows.append(row)
            total_rows += len(pending)
            pending = []
            if len(rows) >= BATCH_WINDOWS:
                first_write = _flush(rows, out_path, first_write)
                rows = []

        for block in iter_blocks(cfg, mseed_files):
            gap_lines.extend(block.gap_report)
            # A block that does not continue the previous one restarts the
            # window grid at its own first sample; the buffer detects that and
            # drops its tail, so no window ever spans an outage the stitcher
            # judged too long to represent.
            for start_index, segments in buffer.push(block):
                start_time = grid_time(start_index, cfg.Fs)
                end_time = grid_time(start_index + cfg.WinSize, cfg.Fs)
                window_id, time_min = namer.name(start_time, end_time)
                meta = {
                    "window_start": _iso(start_time),
                    "window_end": _iso(end_time),
                    "Window_ID": window_id,
                    "Time_min": time_min,
                }
                if emitted < cfg.WARMUP_COUNT:
                    # The first windows of a run are reported but left blank:
                    # they are what the warm-up count exists to discard.
                    rows.append({**meta, **_blank_row(channels)})
                    total_rows += 1
                else:
                    pending.append((meta, {ch: seg.copy()
                                           for ch, seg in segments.items()}))
                emitted += 1
                if len(pending) >= BATCH_WINDOWS:
                    run_batch()

            run_batch()
            print(f"  [WINDOWS] {total_rows} rows so far")

        run_batch()

    first_write = _flush(rows, out_path, first_write)

    _write_gap_log(cfg, mseed_files, gap_lines)

    if total_rows == 0:
        print("\n[WARNING] No windows could be formed from the input.")
        return False

    print(f"\nCSV created: {out_path}")
    print(f"   {total_rows} rows")
    preview = pd.read_csv(out_path, nrows=5)
    print("\n" + "=" * 50)
    print("FIRST 5 ROWS OF CSV")
    print("=" * 50)
    print(preview.to_string(index=False))
    print("=" * 50 + "\n")
    return True


def _write_gap_log(cfg, mseed_files, gap_lines) -> None:
    """Writes the gap report beside the results.

    Args:
        cfg: Configuration object.
        mseed_files: The input files, for the header.
        gap_lines: Report lines gathered from every block.
    """
    from datetime import datetime

    log_dir = cfg.OUTPUT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"gap_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(path, "w", encoding="utf-8") as log:
        log.write(
            f"GAP REPORT — {cfg.STATION} / {cfg.EARTHQUAKE_NAME} "
            f"({len(mseed_files)} file(s))\n"
            f"Target FS: {cfg.Fs} Hz | Bandpass: {cfg.FREQMIN}–{cfg.FREQMAX} Hz | "
            f"Gap threshold: {cfg.GAP_THRESHOLD} s\n\n"
        )
        for line in gap_lines:
            log.write(line + "\n")
