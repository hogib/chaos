"""Sliding-window feature extraction from preprocessed per-channel CSVs."""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from chaos.chaotic_features import (compute_corr_dim,
                                    compute_lyapunov_rosenstein,
                                    compute_lyapunov_wolf, compute_norm_stats,
                                    compute_raw_stats, compute_sample_entropy)

FEATURE_KEYS = [
    "ham_mean", "ham_std", "ham_min", "ham_max", "aktivite_std",
    "norm_min", "norm_max",
    "wolf_lye", "ros_short", "samp_ent", "corr_dim",
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
            seg_norm, cfg.Fs,
            tau=ros_cfg["tau"], m=ros_cfg["m"],
            slope=ros_cfg["slope"], mean_period=ros_cfg["mean_period"],
        ))
    except Exception:
        pass

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


def _extract_hour(stem: str) -> int:
    """Extracts the hour from a CSV file stem.

    Args:
        stem: Filename stem of the form ``YYYYMMDD_HHMMSS[_CHANNEL]``.

    Returns:
        Hour of day, or ``-1`` if it cannot be determined.
    """
    digits = [d for d in stem.split("_") if d.isdigit()]
    if len(digits) >= 2 and len(digits[1]) >= 2:
        return int(digits[1][:2])
    if len(digits) == 1 and int(digits[0]) < 24:
        return int(digits[0])
    return -1


def run_feature_extraction(cfg) -> None:
    """Runs the sliding-window feature extraction stage over all dates.

    Args:
        cfg: Configuration object (see :class:`main.Settings`).
    """
    print("\n" + "=" * 50)
    print("STAGE 2: FEATURE EXTRACTION (3 CHANNELS)")
    print("=" * 50)

    channels = cfg.CHANNELS

    if not cfg.DATA_ROOT.exists():
        print(f"[ERROR] Data folder not found: {cfg.DATA_ROOT}")
        return

    date_folders = sorted(
        d for d in cfg.DATA_ROOT.iterdir() if d.is_dir() and d.name != "logs"
    )
    if not date_folders:
        print(f"[ERROR] No date folders found under {cfg.DATA_ROOT}.")
        return

    print(f"Station   : {cfg.STATION} | Channels: {', '.join(channels)}")
    print(f"Window    : {cfg.WIN_SEC}s  | Step: {cfg.STEP_SEC}s  | Fs: {cfg.Fs} Hz")

    csv_rows: list[dict] = []
    prev_data = {ch: np.array([]) for ch in channels}
    is_first_file = True

    for d_idx, date_dir in enumerate(date_folders):
        ref_dir = None
        for ch in channels:
            ch_dir = date_dir / ch
            if ch_dir.exists() and any(ch_dir.glob("*.csv")):
                ref_dir = ch_dir
                break

        if ref_dir is None:
            continue

        csv_files = sorted(ref_dir.glob("*.csv"))
        date_name = date_dir.name
        print(
            f"[{d_idx + 1}/{len(date_folders)}] Date: {date_name}  "
            f"({len(csv_files)} files)"
        )

        for f_idx, ref_csv in enumerate(csv_files):
            timestamp = ref_csv.stem.rsplit("_", 1)[0]
            print(
                f"  [{f_idx + 1}/{len(csv_files)}] {ref_csv.name}... ",
                end="", flush=True,
            )

            raw = {}
            ref_len = None
            for ch in channels:
                ch_path = date_dir / ch / f"{timestamp}_{ch}.csv"
                try:
                    df = pd.read_csv(str(ch_path), header=None)
                    arr = df.iloc[:, 0].to_numpy(dtype=float)
                    raw[ch] = arr
                    ref_len = len(arr)
                except Exception:
                    raw[ch] = None

            if ref_len is None:
                print("ERROR (all channels failed to load)")
                continue

            for ch in channels:
                if raw[ch] is None:
                    raw[ch] = np.full(ref_len, np.nan)

            x_total = {}
            for ch in channels:
                x_total[ch] = (
                    np.concatenate([prev_data[ch], raw[ch]])
                    if len(prev_data[ch]) > 0
                    else raw[ch].copy()
                )

            n_total = len(x_total[channels[0]])
            num_windows = max(0, (n_total - cfg.WinSize) // cfg.StepSize + 1)

            if num_windows == 0:
                print(f"WARNING: insufficient data ({n_total} samples)")
                for ch in channels:
                    n_t = len(x_total[ch])
                    prev_data[ch] = (
                        x_total[ch][-cfg.PREV_LEN:]
                        if n_t >= cfg.PREV_LEN
                        else x_total[ch]
                    )
                continue

            starts = np.arange(num_windows) * cfg.StepSize
            ends = starts + cfg.WinSize
            time_stamps = ends / cfg.Fs / 60.0
            hour_num = _extract_hour(ref_csv.stem)

            skip_count = (
                min(cfg.WARMUP_COUNT, num_windows)
                if is_first_file and cfg.WARMUP_COUNT > 0
                else 0
            )

            ch_results: dict[str, list] = {}
            for ch in channels:
                segs = [x_total[ch][s:e] for s, e in zip(starts[skip_count:], ends[skip_count:])]
                ch_results[ch] = list(
                    Parallel(n_jobs=cfg.N_JOBS, prefer="processes")(
                        delayed(compute_window)(seg, skip_count + w, cfg)
                        for w, seg in enumerate(segs)
                    )
                )

            for w in range(num_windows):
                row = {
                    "Window_ID": f"{date_name}_{hour_num:02d}_w{w + 1:02d}",
                    "Time_min": round(float(time_stamps[w]), 3),
                }
                if w < skip_count:
                    for ch in channels:
                        for key in FEATURE_KEYS:
                            row[f"{ch}_{key}"] = np.nan
                else:
                    res_idx = w - skip_count
                    for ch in channels:
                        res = ch_results[ch][res_idx]
                        for key in FEATURE_KEYS:
                            row[f"{ch}_{key}"] = res.get(key, np.nan)
                csv_rows.append(row)

            for ch in channels:
                n_t = len(x_total[ch])
                prev_data[ch] = (
                    x_total[ch][-cfg.PREV_LEN:]
                    if n_t >= cfg.PREV_LEN
                    else x_total[ch].copy()
                )

            print(f"OK ({num_windows} windows)")
            is_first_file = False

    if not csv_rows:
        print("\n[WARNING] No data to save.")
        return

    result_df = pd.DataFrame(csv_rows)
    csv_name = (
        f"{cfg.STATION}_{date_folders[0].name}-{date_folders[-1].name}"
        f"_ENZ_features.csv"
    )
    out_path = cfg.OUTPUT_ROOT / csv_name
    result_df.to_csv(out_path, index=False)

    print(f"\nCSV created: {out_path}")
    print(f"   {len(result_df)} rows  |  {len(result_df.columns)} columns")
    print("\n" + "=" * 50)
    print("FIRST 5 ROWS OF CSV")
    print("=" * 50)
    print(result_df.head(5).to_string(index=False))
    print("=" * 50 + "\n")
