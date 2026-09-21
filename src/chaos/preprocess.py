"""MSEED preprocessing: gap handling, bandpass filtering, decimation, CSV export."""

import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import numpy.ma as _nma
import scipy.interpolate as _sci_interp
from obspy import UTCDateTime, read


def _gap_duration_sec(gap, st_full) -> float:
    """Returns the duration of an ObsPy gap tuple in seconds.

    Args:
        gap: Gap tuple as returned by :meth:`obspy.Stream.get_gaps`.
            Field 7 is the number of missing samples.
        st_full: Stream the gap belongs to (used to look up the sampling rate).

    Returns:
        Gap duration in seconds.
    """
    tr_match = st_full.select(channel=gap[3])
    fs_val = (
        tr_match[0].stats.sampling_rate
        if tr_match
        else (st_full[0].stats.sampling_rate if len(st_full) > 0 else 100.0)
    )
    return gap[7] / fs_val


def parse_file_date(mseed_file, st_full) -> datetime:
    """Parses a ``YYYYMMDD`` date from the MSEED file name.

    Falls back to the stream start time if no valid date is found.

    Args:
        mseed_file: Path to the MSEED file.
        st_full: Stream read from that file.

    Returns:
        The parsed :class:`datetime.datetime`.
    """
    match = re.search(r"_(\d{8})_", mseed_file.name)
    if match:
        date_str = match.group(1)
        try:
            return datetime(int(date_str[4:8]), int(date_str[2:4]), int(date_str[0:2]))
        except ValueError:
            pass
        try:
            return datetime(int(date_str[0:4]), int(date_str[4:6]), int(date_str[6:8]))
        except ValueError:
            pass
    return st_full[0].stats.starttime.datetime


def _save_csv_with_retry(csv_path, data, retries: int = 5, delay: float = 1.0) -> None:
    """Writes an array to CSV, retrying on transient file locks.

    On Windows, antivirus / indexer / cloud-sync can briefly lock newly
    created files; a short wait usually clears the lock.

    Args:
        csv_path: Destination path.
        data: 1-D array to write.
        retries: Maximum number of attempts.
        delay: Base delay between retries (scaled by the attempt number).
    """
    for attempt in range(1, retries + 1):
        try:
            np.savetxt(csv_path, data, delimiter=",", fmt="%.8f")
            return
        except PermissionError:
            if attempt == retries:
                raise
            print(f"  [WARN] {csv_path.name} locked, retry {attempt}/{retries - 1}...")
            time.sleep(delay * attempt)


def _decimation_grid(seg_start, factor):
    """Places a clean segment onto the window's global decimation grid.

    Decimating each segment from its own first sample would put the samples
    after a gap on a different grid than the samples before it (and than the
    same window processed without a gap). Instead the segment is decimated
    from its first sample whose absolute index is a multiple of ``factor``.

    Args:
        seg_start: Index of the segment's first sample within the window.
        factor: Decimation factor.

    Returns:
        Tuple ``(phase, out_start)`` — the offset into the segment at which to
        start taking every ``factor``-th sample, and the index in the decimated
        output where those samples begin.
    """
    phase = (-seg_start) % factor
    return phase, (seg_start + phase) // factor


def _process_single_mseed(cfg, mseed_file, output_base) -> int:
    """Processes one MSEED file into hourly per-channel CSV files.

    The pipeline is: read → gap handling → bandpass filter → decimation →
    hourly windowing → CSV export. Small gaps (< ``cfg.GAP_THRESHOLD`` s) are
    filled by cubic interpolation. Large gaps are preserved as NaN, and only
    the clean segments around them are filtered to avoid smearing data.

    Args:
        cfg: Configuration object (see :class:`main.Settings`).
        mseed_file: Path to the MSEED file to process.
        output_base: Root directory where hourly CSVs are written.

    Returns:
        Tuple ``(csv_count, gap_report_lines)``.
    """
    st_full = read(str(mseed_file))

    for tr in st_full:
        if tr.data.dtype != np.float64:
            tr.data = tr.data.astype(np.float64)

    real_fs = st_full[0].stats.sampling_rate
    decimation_factor = max(1, int(real_fs / cfg.Fs))

    gaps = st_full.get_gaps(min_gap=-1)
    actual_gaps = [g for g in gaps if g[7] > 0] if gaps else []
    large_gaps = [g for g in actual_gaps if _gap_duration_sec(g, st_full) >= cfg.GAP_THRESHOLD]
    small_gaps = [g for g in actual_gaps if _gap_duration_sec(g, st_full) < cfg.GAP_THRESHOLD]

    gap_report = [
        f"{mseed_file.name}: {len(small_gaps)} small (<{cfg.GAP_THRESHOLD}s), "
        f"{len(large_gaps)} large (>={cfg.GAP_THRESHOLD}s)"
    ]
    for gap in large_gaps:
        gap_report.append(
            f"    LARGE {gap[3]} {gap[4]} -> {gap[5]} "
            f"({_gap_duration_sec(gap, st_full):.2f}s, {gap[7]} samples)"
        )
    print(f"  [GAP] {gap_report[0]}")

    if not actual_gaps:
        st_full.merge()
    elif not large_gaps:
        st_full.merge(fill_value=np.nan)
        for tr in st_full:
            data = (
                np.ma.filled(tr.data.astype(float), np.nan)
                if np.ma.is_masked(tr.data)
                else np.array(tr.data, dtype=float)
            )
            nan_mask = np.isnan(data)
            if nan_mask.any():
                valid = np.where(~nan_mask)[0]
                if len(valid) > 3:
                    interp = _sci_interp.interp1d(
                        valid, data[valid], kind="cubic",
                        bounds_error=False, fill_value="extrapolate",
                    )
                    data[nan_mask] = interp(np.where(nan_mask)[0])
            tr.data = data
    else:
        st_full.merge(fill_value=np.nan)
        for tr in st_full:
            data = (
                np.ma.filled(tr.data.astype(float), np.nan)
                if np.ma.is_masked(tr.data)
                else np.array(tr.data, dtype=float)
            )
            big_gap_mask = np.zeros(len(data), dtype=bool)
            tr_start, tr_fs = tr.stats.starttime, tr.stats.sampling_rate

            for lg in large_gaps:
                if lg[1] != tr.stats.station or lg[3] != tr.stats.channel:
                    continue
                gi_start = max(0, int((lg[4] - tr_start) * tr_fs))
                gi_end = min(len(data), int((lg[5] - tr_start) * tr_fs) + 1)
                if gi_start < gi_end:
                    big_gap_mask[gi_start:gi_end] = True

            small_nan = np.isnan(data) & ~big_gap_mask
            if small_nan.any():
                valid = np.where(~np.isnan(data))[0]
                if len(valid) > 3:
                    interp = _sci_interp.interp1d(
                        valid, data[valid], kind="cubic",
                        bounds_error=False, fill_value="extrapolate",
                    )
                    data[np.where(small_nan)[0]] = interp(np.where(small_nan)[0])

            tr.data = np.ma.array(data, mask=big_gap_mask)

    file_date = parse_file_date(mseed_file, st_full)
    day_start = UTCDateTime(file_date.year, file_date.month, file_date.day, 0, 0, 0)
    day_end = day_start + 86400
    num_windows = int((day_end - day_start) / cfg.PREPROCESS_WINDOW_SEC)
    min_seg_len = max(20, int(3.0 / cfg.FREQMIN * real_fs))

    total_csv_created = 0
    for window_idx in range(num_windows):
        window_start = day_start + (window_idx * cfg.PREPROCESS_WINDOW_SEC)
        window_end = window_start + cfg.PREPROCESS_WINDOW_SEC
        # Stream.slice() returns views into st_full; the copy below is the
        # only one needed, and it is window-sized rather than day-sized.
        st_window = st_full.slice(starttime=window_start, endtime=window_end)

        if len(st_window) == 0 or len(st_window[0].data) < 100:
            continue

        st_proc = st_window.copy()
        win_nan_map = {}
        for tr in st_proc:
            data = (
                np.ma.filled(tr.data.astype(float), np.nan)
                if np.ma.is_masked(tr.data)
                else np.array(tr.data, dtype=float)
            )
            nan_mask = np.isnan(data)
            if nan_mask.any():
                win_nan_map[tr.id] = nan_mask
            tr.data = data

        st_decimated = st_proc.copy()
        for tr in st_proc:
            res_tr = st_decimated.select(id=tr.id)[0]
            if tr.id not in win_nan_map:
                tmp = tr.copy()
                tmp.detrend("demean")
                tmp.detrend("linear")
                tmp.filter(
                    "bandpass",
                    freqmin=cfg.FREQMIN, freqmax=cfg.FREQMAX,
                    corners=4, zerophase=True,
                )
                if decimation_factor > 1:
                    tmp.decimate(factor=decimation_factor, no_filter=True)
                    tmp.detrend("linear")
                res_tr.data = tmp.data
                res_tr.stats.sampling_rate = tmp.stats.sampling_rate
            else:
                nan_mask = win_nan_map[tr.id]
                raw = tr.data.copy().astype(float)
                # obspy's decimate keeps samples 0, f, 2f, ... i.e. ceil(n / f)
                # of them. The gapped branch must produce the same count, or
                # channels of one window end up with different CSV lengths.
                out_n = -(-len(raw) // decimation_factor)
                out = np.full(out_n, np.nan, dtype=float)

                pad = np.concatenate([[False], ~nan_mask, [False]])
                diff_arr = np.diff(pad.astype(np.int8))
                seg_starts = np.where(diff_arr == 1)[0]
                seg_ends = np.where(diff_arr == -1)[0]
                for seg_start, seg_end in zip(seg_starts, seg_ends):
                    if (seg_end - seg_start) < min_seg_len:
                        continue
                    tmp = tr.copy()
                    tmp.data = raw[seg_start:seg_end].copy()
                    tmp.stats.starttime = (
                        tr.stats.starttime + seg_start / tr.stats.sampling_rate
                    )
                    try:
                        tmp.detrend("demean")
                        tmp.detrend("linear")
                        tmp.filter(
                            "bandpass",
                            freqmin=cfg.FREQMIN, freqmax=cfg.FREQMAX,
                            corners=4, zerophase=True,
                        )
                    except Exception:
                        continue
                    phase, out_start = _decimation_grid(
                        seg_start, decimation_factor
                    )
                    dec = (
                        tmp.data[phase::decimation_factor]
                        if decimation_factor > 1
                        else tmp.data
                    )
                    out_end = min(out_start + len(dec), out_n)
                    if out_end - out_start > 0:
                        out[out_start:out_end] = dec[: out_end - out_start]

                res_tr.data = out
                res_tr.stats.sampling_rate = cfg.Fs

        date_folder = window_start.datetime.strftime("%Y_%m_%d")
        for component in ("E", "N", "Z"):
            try:
                tr = st_decimated.select(component=component)[0]
            except IndexError:
                continue
            comp_dir = output_base / date_folder / component
            comp_dir.mkdir(parents=True, exist_ok=True)
            csv_path = (
                comp_dir
                / f"{window_start.datetime.strftime('%Y%m%d_%H%M%S')}_{component}.csv"
            )
            data_out = (
                _nma.filled(tr.data.astype(float), np.nan)
                if _nma.is_masked(tr.data)
                else tr.data.astype(float)
            )
            _save_csv_with_retry(csv_path, data_out)
            total_csv_created += 1

    return total_csv_created, gap_report


def _process_one(args):
    """Pickle-friendly wrapper used by :class:`ProcessPoolExecutor`.

    Returns:
        Tuple ``(file_name, csv_count, gap_report_lines)``.
    """
    cfg, mseed_file, output_base = args
    count, gap_report = _process_single_mseed(cfg, mseed_file, output_base)
    return mseed_file.name, count, gap_report


def _resolve_workers(cfg, n_items: int) -> int:
    """Returns the worker count to use for the given config and workload.

    Args:
        cfg: Configuration object with an ``N_JOBS`` attribute.
        n_items: Number of independent tasks to run.

    Returns:
        A positive integer that never exceeds ``n_items``.
    """
    requested = cfg.N_JOBS if cfg.N_JOBS and cfg.N_JOBS > 0 else (os.cpu_count() or 1)
    return max(1, min(n_items, requested))


def run_mseed_preprocessing(cfg) -> bool:
    """Runs the preprocessing stage over every MSEED file in the input dir.

    Each MSEED file is independent; files are distributed across a process
    pool. This assumes that no two MSEED files produce CSVs with the same
    timestamped name (usually true for one-file-per-day inputs).

    Args:
        cfg: Configuration object (see :class:`main.Settings`).

    Returns:
        ``True`` if at least one file was processed, ``False`` otherwise.
    """
    print("\n" + "=" * 50)
    print("STAGE 1: MSEED PREPROCESSING (GAP, FILTER, DOWNSAMPLE)")
    print("=" * 50)

    mseed_files = sorted(cfg.MSEED_INPUT_DIR.glob("*.mseed"))
    if not mseed_files:
        print(f"[SKIPPED] No .mseed files found in: {cfg.MSEED_INPUT_DIR}")
        return False

    start_time = datetime.now()
    output_base = cfg.DATA_ROOT
    output_base.mkdir(parents=True, exist_ok=True)

    log_dir = output_base / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    gap_log_path = log_dir / f"gap_report_{start_time.strftime('%Y%m%d_%H%M%S')}.txt"
    header = (
        f"GAP REPORT — {cfg.STATION} / {cfg.EARTHQUAKE_NAME} "
        f"({len(mseed_files)} file(s))\n"
        f"Target FS: {cfg.Fs} Hz | Bandpass: {cfg.FREQMIN}–{cfg.FREQMAX} Hz | "
        f"Gap threshold: {cfg.GAP_THRESHOLD} s\n\n"
    )

    max_workers = _resolve_workers(cfg, len(mseed_files))
    total_csv_created = 0
    # Keyed by file name so the log is ordered by input file rather than by
    # whichever worker happened to finish first.
    gap_lines: dict[str, list[str]] = {}

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_process_one, (cfg, f, output_base)): f for f in mseed_files
        }
        for i, fut in enumerate(as_completed(futures), 1):
            f = futures[fut]
            try:
                _, count, report = fut.result()
                total_csv_created += count
                gap_lines[f.name] = report
                print(f"[{i}/{len(mseed_files)}] {f.name}: {count} CSV")
            except Exception as exc:
                gap_lines[f.name] = [f"{f.name}: FAILED — {exc}"]
                print(f"[{i}/{len(mseed_files)}] {f.name}: FAILED — {exc}")

    with open(gap_log_path, "w", encoding="utf-8") as glog:
        glog.write(header)
        for f in mseed_files:
            for line in gap_lines.get(f.name, [f"{f.name}: no report"]):
                glog.write(line + "\n")

    print(
        f"[INFO] Preprocessing complete. "
        f"Total {total_csv_created} CSV → {output_base}"
    )
    return True
