"""Estimates Rosenstein's mean period from seismically quiet waveform data.

Rosenstein et al. (1993) define the mean period as the reciprocal of the
power-weighted mean frequency of the signal. It sets both the Theiler band
(neighbours closer in time than about one mean period are excluded) and the
units of the divergence-curve fit window, so it should be measured on the same
filtered, decimated signal the Lyapunov windows see, and on background noise
rather than on earthquakes.

What this script does:

1. Reads an earthquake catalog CSV and turns every event into an exclusion
   interval around its origin time. The interval after the event grows with
   magnitude, following the inverse of a coda-duration magnitude relation.
2. Finds one or more recordings: a directory holding ``*.mseed`` files, or a
   directory whose subdirectories each hold them (e.g. 20 events of 21 days).
3. Cuts each recording into fixed-length chunks, keeps the ones that overlap no
   exclusion interval, and randomly samples some of them.
4. Reads only those chunks (plus filter padding), filters and decimates them
   with the pipeline's own ``_decimate_segment`` so the result matches what
   feature extraction computes, and splits them into ``win_sec`` windows.
5. Computes the mean frequency of each window via an FFT, drops windows that
   are much louder than the recording's typical level (uncatalogued events,
   glitches), and reports the median mean period plus its spread.

Settings for filtering, decimation, window length and channel come from the
pipeline's ``config.json`` so the estimate cannot drift from the pipeline.

Example:
    python -m chaos.mean_period catalog.csv raw/ELBA --chunks 30 --jobs 4

The pipeline calls :func:`resolve_mean_period` itself whenever
``rosenstein.mean_period`` is missing or ``null`` in config.json.
"""

import argparse
import hashlib
import json
import math
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from obspy import UTCDateTime, read

from chaos.preprocess import (_decimate_segment, filter_pad_sec, grid_index,
                              grid_time)

# --------------------------------------------------------------------------- #
# Catalog                                                                      #
# --------------------------------------------------------------------------- #

_TURKISH = str.maketrans("ıİşŞğĞüÜöÖçÇ", "iIsSgGuUoOcC")

# Normalised column names tried in order when --time-col is not given.
_COMBINED_TIME = ["origin_time", "origintime", "event_time", "datetime",
                  "date_time", "olus_zamani", "time", "date", "tarih",
                  "olus_tarihi"]
_DATE_ONLY = ["date", "tarih", "olus_tarihi"]
_CLOCK_ONLY = ["time", "saat", "olus_zamani"]
_MAGNITUDE = ["magnitude", "mag", "mw", "ml", "md", "ms", "mb", "xm",
              "buyukluk"]
_LATITUDE = ["latitude", "lat", "enlem"]
_LONGITUDE = ["longitude", "lon", "lng", "long", "boylam"]


def _read_table(path, **kwargs):
    """Reads a CSV whose separator is guessed from its header line.

    pandas' own sniffing (``sep=None``) needs the slow Python engine, which
    takes half a minute on a nationwide catalog; guessing from the header
    keeps the C engine.
    """
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        header = fh.readline()
    sep = max((",", ";", "\t", "|"), key=header.count)
    return pd.read_csv(path, sep=sep, encoding="utf-8-sig",
                       encoding_errors="replace", **kwargs)


def _norm(name) -> str:
    """Lower-cases a column name and strips accents and punctuation."""
    text = str(name).strip().translate(_TURKISH).lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _column(df, cols, name):
    """Resolves a user-supplied column name, forgiving case and accents."""
    if name in df.columns:
        return name
    if _norm(name) in cols:
        return cols[_norm(name)]
    raise SystemExit(
        f"catalog has no column {name!r}; its columns are {list(df.columns)}"
    )


def _to_datetime(values, dayfirst):
    """Parses a column of date strings, tolerating mixed formats."""
    text = values.astype(str).str.strip()
    # One format inferred from the first row parses vectorised; "mixed" runs
    # dateutil row by row, which costs most of a minute on a big catalog, so
    # it is only the fallback for files that really do mix formats.
    try:
        fast = pd.to_datetime(text, errors="coerce", dayfirst=dayfirst)
        if len(fast) and fast.notna().mean() >= 0.99:
            return fast
    except (TypeError, ValueError):
        pass
    try:
        return pd.to_datetime(text, errors="coerce", dayfirst=dayfirst,
                              format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(text, errors="coerce", dayfirst=dayfirst)


def _has_clock(times) -> bool:
    """True if at least one parsed time is not exactly midnight."""
    valid = times.dropna()
    return bool(len(valid)) and bool((valid.dt.normalize() != valid).any())


def _numeric(series):
    """Parses numbers that may use a decimal comma."""
    return pd.to_numeric(
        series.astype(str).str.strip().str.replace(",", ".", regex=False),
        errors="coerce",
    )


def _event_times(df, cols, args):
    """Returns event origin times as UTC epoch seconds and a description."""
    if args.time_col:
        tcol = _column(df, cols, args.time_col)
        if args.date_col:
            dcol = _column(df, cols, args.date_col)
            raw = (df[dcol].astype(str).str.strip() + " "
                   + df[tcol].astype(str).str.strip())
            source = f"{dcol} + {tcol}"
        else:
            raw, source = df[tcol], tcol
        times = _to_datetime(raw, args.dayfirst)
    else:
        times, source = None, None
        for key in _COMBINED_TIME:
            if key not in cols:
                continue
            candidate = _to_datetime(df[cols[key]], args.dayfirst)
            if candidate.notna().mean() > 0.5 and _has_clock(candidate):
                times, source = candidate, cols[key]
                break
        if times is None:
            for dkey in _DATE_ONLY:
                for ckey in _CLOCK_ONLY:
                    if dkey == ckey or dkey not in cols or ckey not in cols:
                        continue
                    raw = (df[cols[dkey]].astype(str).str.strip() + " "
                           + df[cols[ckey]].astype(str).str.strip())
                    candidate = _to_datetime(raw, args.dayfirst)
                    if candidate.notna().mean() > 0.5:
                        times = candidate
                        source = f"{cols[dkey]} + {cols[ckey]}"
                        break
                if times is not None:
                    break
        if times is None:
            raise SystemExit(
                "could not find the event time in the catalog; pass "
                "--time-col (and --date-col if date and clock are separate). "
                f"Columns: {list(df.columns)}"
            )

    if getattr(times.dt, "tz", None) is not None:
        times = times.dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        times = times - pd.Timedelta(hours=args.catalog_utc_offset)

    epoch = (times - pd.Timestamp("1970-01-01")).dt.total_seconds()
    return epoch.to_numpy(dtype=float), source


def _magnitudes(df, cols, args):
    """Returns one magnitude per event, taking the largest reported type."""
    if args.mag_col:
        names = [_column(df, cols, args.mag_col)]
    else:
        names = list(dict.fromkeys(cols[k] for k in _MAGNITUDE if k in cols))
    if not names:
        print(f"  [WARN] no magnitude column found; every event is treated "
              f"as M{args.unknown_mag}")
        return np.full(len(df), args.unknown_mag), "none"
    values = pd.DataFrame({n: _numeric(df[n]) for n in names})
    # Several magnitude types (ML, Mw, Md...) per row: the largest is the
    # conservative choice, since it gives the longest exclusion.
    mags = values.max(axis=1).fillna(args.unknown_mag)
    return mags.to_numpy(dtype=float), ", ".join(names)


def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def post_event_sec(mag, args):
    """Seconds after an origin time during which the data are not quiet.

    Inverts the duration magnitude relation of Lee et al. (1972),
    ``Md = 2 log10(T) - 0.87``, with its distance term dropped, to get a coda
    duration, then multiplies it by a safety factor. M2 gives ~27 s of coda,
    M4 ~4.5 min, M6.2 ~57 min before the factor is applied.
    """
    coda = 10.0 ** ((np.asarray(mag, dtype=float) + 0.87) / 2.0)
    return np.maximum(args.min_post_sec, args.coda_factor * coda)


def load_exclusions(args):
    """Reads the catalog and returns merged, sorted exclusion intervals."""
    df = _read_table(args.catalog)
    if df.empty:
        raise SystemExit(f"{args.catalog} has no rows")
    cols = {_norm(c): c for c in df.columns}

    times, time_source = _event_times(df, cols, args)
    mags, mag_source = _magnitudes(df, cols, args)

    keep = np.isfinite(times) & (mags >= args.min_mag)
    n_bad_time = int((~np.isfinite(times)).sum())

    if args.station_lat is not None and args.station_lon is not None \
            and args.max_dist_km is not None:
        lat_key = next((k for k in _LATITUDE if k in cols), None)
        lon_key = next((k for k in _LONGITUDE if k in cols), None)
        lat_col = _column(df, cols, args.lat_col) if args.lat_col else (
            cols[lat_key] if lat_key else None)
        lon_col = _column(df, cols, args.lon_col) if args.lon_col else (
            cols[lon_key] if lon_key else None)
        if lat_col is None or lon_col is None:
            raise SystemExit("--max-dist-km needs latitude/longitude columns; "
                             "pass --lat-col and --lon-col")
        dist = _haversine_km(args.station_lat, args.station_lon,
                             _numeric(df[lat_col]).to_numpy(),
                             _numeric(df[lon_col]).to_numpy())
        # An event with no usable location is kept: excluding too much only
        # costs data, excluding too little contaminates the estimate.
        keep &= ~(np.isfinite(dist) & (dist > args.max_dist_km))

    t, m = times[keep], mags[keep]
    starts = t - args.pre_sec
    ends = t + post_event_sec(m, args)

    order = np.argsort(starts)
    merged = []
    for s, e in zip(starts[order], ends[order]):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    merged = np.array(merged, dtype=float).reshape(-1, 2)

    print(f"Catalog   : {args.catalog} ({len(df)} rows)")
    print(f"  time    : {time_source} (read as UTC{args.catalog_utc_offset:+g} "
          "unless the strings carry their own zone)")
    print(f"  mag     : {mag_source}")
    print(f"  events  : {int(keep.sum())} used, {n_bad_time} unparseable times")
    if keep.any():
        first = pd.Timestamp(t.min(), unit="s")
        last = pd.Timestamp(t.max(), unit="s")
        print(f"  span    : {first} -> {last} UTC")
    return merged[:, 0].copy(), merged[:, 1].copy(), t


def station_coords(path, station, network=None):
    """Looks a station up in a coordinates CSV.

    The file needs station, latitude and longitude columns (names are matched
    like the catalog's); a network column, if present, settles station codes
    that more than one network uses.

    Args:
        path: The CSV path.
        station: Station code, matched case-insensitively.
        network: Network code of the data, used only to break ties.

    Returns:
        ``(latitude, longitude)``, or ``None`` if the station is not listed.

    Raises:
        ValueError: If the columns are missing or the code stays ambiguous.
    """
    df = _read_table(path, dtype=str)
    cols = {_norm(c): c for c in df.columns}
    sta_col = next((cols[k] for k in ("station", "sta", "code", "istasyon")
                    if k in cols), None)
    lat_col = next((cols[k] for k in _LATITUDE if k in cols), None)
    lon_col = next((cols[k] for k in _LONGITUDE if k in cols), None)
    if sta_col is None or lat_col is None or lon_col is None:
        raise ValueError(f"{path} needs station, latitude and longitude "
                         f"columns; its columns are {list(df.columns)}")
    net_col = next((cols[k] for k in ("network", "net") if k in cols), None)

    rows = df[df[sta_col].str.strip().str.upper() == station.upper()].copy()
    rows["_lat"] = _numeric(rows[lat_col])
    rows["_lon"] = _numeric(rows[lon_col])
    rows = rows.dropna(subset=["_lat", "_lon"])
    if rows.empty:
        return None
    if len(rows) > 1 and net_col is not None and network:
        same = rows[rows[net_col].str.strip().str.upper() == network.upper()]
        if not same.empty:
            rows = same
    distinct = rows[["_lat", "_lon"]].round(3).drop_duplicates()
    if len(distinct) > 1:
        listed = ", ".join(
            f"{r[net_col] if net_col else '?'} ({r['_lat']}, {r['_lon']})"
            for _, r in rows.iterrows())
        raise ValueError(
            f"station {station} appears at several places in {path}: {listed}"
            f"; the data's network ({network or 'unknown'}) does not pick one."
            " Set station_lat/station_lon in mean_period_estimation.options")
    return float(rows["_lat"].iloc[0]), float(rows["_lon"].iloc[0])


def _data_network(recordings):
    """Network code from the first readable MSEED header, or ``None``."""
    for _, files in recordings:
        for path in files:
            try:
                return read(str(path), headonly=True)[0].stats.network or None
            except Exception:
                continue
    return None


def overlaps(starts, ends, a, b) -> bool:
    """True if ``[a, b)`` touches any merged exclusion interval.

    Merged intervals are disjoint and sorted, so their ends are sorted too and
    only the last interval starting before ``b`` can reach past ``a``.
    """
    i = int(np.searchsorted(starts, b, side="left")) - 1
    return i >= 0 and ends[i] > a


# --------------------------------------------------------------------------- #
# Waveforms                                                                    #
# --------------------------------------------------------------------------- #

def find_recordings(paths, pattern, split_files=False):
    """Returns ``(label, files)`` for each directory of MSEED files.

    With ``split_files`` every file is its own recording, which suits a folder
    of long files (e.g. 21-day ones): each is sampled separately, measured in
    parallel, and gets its own loudness baseline.
    """
    recordings = []
    for raw in paths:
        root = Path(raw)
        if not root.is_dir():
            raise SystemExit(f"not a directory: {root}")
        groups = []
        files = sorted(root.glob(pattern))
        if files:
            groups.append((f"{root.parent.name}/{root.name}", files))
        else:
            for sub in sorted(d for d in root.iterdir() if d.is_dir()):
                files = sorted(sub.glob(pattern))
                if files:
                    groups.append((f"{root.name}/{sub.name}", files))
        for label, files in groups:
            if split_files:
                recordings.extend((f"{label}/{f.stem}", [f]) for f in files)
            else:
                recordings.append((label, files))
    return recordings


def spectral_stats(segment, fs):
    """Mean and peak frequency of one window.

    The mean frequency is Rosenstein's: the power-weighted average over the
    periodogram, DC excluded. A Hann taper keeps the window edges from leaking
    power into the high frequencies and biasing the mean upwards.
    """
    x = segment - segment.mean()
    power = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    power[0] = 0.0
    total = power.sum()
    if not np.isfinite(total) or total <= 0:
        return np.nan, np.nan
    mean_freq = float((freqs * power).sum() / total)
    peak_freq = float(freqs[1:][np.argmax(power[1:])])
    return mean_freq, peak_freq


def process_recording(job):
    """Samples quiet chunks from one recording and measures every window."""
    label, files, cfg, channel, starts, ends, opts, rec_index = job
    pad = filter_pad_sec(cfg)
    chunk = float(opts.chunk_sec)
    win = int(round(cfg.WIN_SEC * cfg.Fs))
    n_chunk = int(round(chunk * cfg.Fs))
    counts = dict(files=len(files), unreadable=0, candidates=0, quiet=0,
                  sampled=0, used=0, gap=0, incomplete=0, bad_fs=0)

    spans = []
    for path in files:
        try:
            stream = read(str(path), headonly=True)
        except Exception:
            counts["unreadable"] += 1
            continue
        sel = stream.select(component=channel)
        if sel:
            spans.append((float(min(tr.stats.starttime for tr in sel)),
                          float(max(tr.stats.endtime for tr in sel)),
                          float(sel[0].stats.sampling_rate), path))
    spans.sort(key=lambda s: s[0])

    # Files that touch or overlap form one stretch of continuous coverage, so
    # a chunk may straddle a file boundary; with hour-long files no chunk
    # would fit inside a single file once the filter padding is added.
    stretches = []
    for s, e, fs_raw, _ in spans:
        if stretches and s <= stretches[-1][1] + 1.5 / fs_raw:
            stretches[-1][1] = max(stretches[-1][1], e)
        else:
            stretches.append([s, e])

    # Chunks sit on a lattice of whole chunk lengths from the epoch and must
    # fit inside one stretch with filter padding on both sides.
    candidates = []
    for s, e in stretches:
        a = math.ceil((s + pad) / chunk) * chunk
        while a + chunk + pad <= e:
            candidates.append(a)
            a += chunk
    counts["candidates"] = len(candidates)

    quiet = [a for a in candidates
             if not overlaps(starts, ends, a, a + chunk)]
    counts["quiet"] = len(quiet)

    rng = np.random.default_rng([opts.seed, rec_index])
    size = min(opts.chunks, len(quiet))
    picks = sorted(rng.choice(len(quiet), size=size, replace=False)) \
        if size else []
    counts["sampled"] = len(picks)

    rows, warned_fs = [], False
    for k in picks:
        a = quiet[k]
        t0 = grid_time(grid_index(UTCDateTime(a), cfg.Fs), cfg.Fs)
        lo, hi = float(t0) - pad, float(t0) + chunk + pad
        stream, names = None, []
        try:
            for s, e, _, path in spans:
                if e < lo or s > hi:
                    continue
                part = read(str(path), starttime=t0 - pad,
                            endtime=t0 + chunk + pad)
                stream = part if stream is None else stream + part
                names.append(path.name)
        except Exception:
            stream = None
        if stream is None:
            counts["incomplete"] += 1
            continue
        sel = stream.select(component=channel)
        # Day files often overhang midnight by a sample or two; method=1 keeps
        # the shared samples once, and any real gap comes back masked.
        sel.merge(method=1, fill_value=None)
        if len(sel) != 1:
            counts["gap"] += 1
            continue
        tr = sel[0]
        data = tr.data
        if np.ma.isMaskedArray(data):
            if np.ma.getmaskarray(data).any():
                counts["gap"] += 1
                continue
            data = np.ma.getdata(data)
        real_fs = tr.stats.sampling_rate
        if tr.stats.starttime > t0 - pad + 1.0 / real_fs or \
                tr.stats.endtime < t0 + chunk + pad - 1.0 / real_fs:
            counts["incomplete"] += 1
            continue

        factor = max(1, int(real_fs / cfg.Fs))
        if abs(real_fs / factor - cfg.Fs) > 1e-9:
            if not warned_fs:
                print(f"  [WARN] {label}: {real_fs:g} Hz does not decimate to "
                      f"{cfg.Fs:g} Hz; skipping those chunks")
                warned_fs = True
            counts["bad_fs"] += 1
            continue

        origin = grid_index(t0, cfg.Fs)
        out = np.full(n_chunk, np.nan)
        _decimate_segment(tr, np.asarray(data, dtype=np.float64),
                          tr.stats.starttime, cfg, factor, out, origin)
        if np.isnan(out).any():
            counts["gap"] += 1
            continue
        counts["used"] += 1

        for w in range(n_chunk // win):
            seg = out[w * win:(w + 1) * win]
            mean_f, peak_f = spectral_stats(seg, cfg.Fs)
            rows.append({
                "recording": label,
                "files": "+".join(names),
                "window_start": str(grid_time(origin + w * win, cfg.Fs)),
                "mean_freq_hz": mean_f,
                "mean_period_s": 1.0 / mean_f if mean_f > 0 else np.nan,
                "peak_freq_hz": peak_f,
                "std": float(np.std(seg, ddof=1)),
            })

    # Catalog screening misses teleseisms, uncatalogued micro-events and
    # instrument glitches. Anything far above the recording's typical level
    # is one of those, not background.
    if rows:
        std = np.array([r["std"] for r in rows])
        limit = opts.loud_factor * np.median(std)
        for r, s in zip(rows, std):
            r["kept"] = bool(s <= limit)
    return label, counts, rows


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #

def summarise(frame, cfg, config, out_dir):
    """Prints and writes the summary, and returns it as a dict."""
    kept = frame[frame["kept"]]
    periods = kept["mean_period_s"].dropna()
    q25, q50, q75 = np.percentile(periods, [25, 50, 75])
    per_rec = kept.groupby("recording")["mean_period_s"].median()

    theiler = int(math.ceil(q50 * cfg.Fs))
    summary = {
        "mean_period_s": round(float(q50), 3),
        "mean_period_iqr_s": [round(float(q25), 3), round(float(q75), 3)],
        "windows_used": int(len(periods)),
        "windows_rejected_loud": int((~frame["kept"]).sum()),
        "recordings": int(frame["recording"].nunique()),
        "per_recording_median_s": {k: round(float(v), 3)
                                   for k, v in per_rec.items()},
        "theiler_band_samples": theiler,
        "fs_hz": cfg.Fs,
        "band_hz": [cfg.FREQMIN, cfg.FREQMAX],
        "win_sec": cfg.WIN_SEC,
    }

    print("\n" + "=" * 60)
    print("MEAN PERIOD")
    print("=" * 60)
    print(f"Median     : {q50:.3f} s   (mean frequency {1 / q50:.3f} Hz)")
    print(f"IQR        : {q25:.3f} – {q75:.3f} s")
    print(f"Windows    : {len(periods)} used, "
          f"{summary['windows_rejected_loud']} dropped as too loud")
    print(f"Recordings : median ranges {per_rec.min():.3f} – "
          f"{per_rec.max():.3f} s across {len(per_rec)}")
    spread = (per_rec.max() - per_rec.min()) / q50
    if spread > 0.25:
        print(f"  [NOTE] recordings differ by {spread:.0%} of the median; a "
              "single fixed value is a compromise (sea state and season move "
              "the microseism peak)")
    print(f"Theiler    : >= {theiler} samples (one mean period at "
          f"{cfg.Fs:g} Hz); the pipeline currently uses (m - 1) * tau")

    ros = config.get("feature_extraction", {}).get("features", {}) \
        .get("rosenstein", {})
    if "slope" in ros:
        try:
            from chaos.chaotic_features import split_slope_windows
            windows = split_slope_windows(ros["slope"])
        except Exception:
            windows = ()
        for name, w in zip(("short", "long"), windows):
            if w is None:
                continue
            lo, hi = w[0] * q50, w[1] * q50
            print(f"Fit window : {name} {list(w)} mean periods = "
                  f"{lo:.1f}–{hi:.1f} s = samples "
                  f"{round(lo * cfg.Fs)}–{round(hi * cfg.Fs)} with this value")
    if "mean_period" in ros:
        print(f"Config     : rosenstein.mean_period is currently "
              f"{ros['mean_period']}")
    print("=" * 60)

    (out_dir / "mean_period_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def plot(frame, summary, out_dir):
    """Saves a histogram and a per-recording box plot, if matplotlib exists."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping the plot")
        return
    kept = frame[frame["kept"]]
    labels = sorted(kept["recording"].unique())
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7.5), gridspec_kw={"height_ratios": [1, 1.3]})

    ax1.hist(kept["mean_period_s"].dropna(), bins=60, color="#4c72b0")
    ax1.axvline(summary["mean_period_s"], color="#c44e52", lw=2,
                label=f"median {summary['mean_period_s']:.2f} s")
    ax1.set_xlabel("Mean period (s)")
    ax1.set_ylabel("Windows")
    ax1.legend()

    ax2.boxplot([kept.loc[kept["recording"] == r, "mean_period_s"].dropna()
                 for r in labels], showfliers=False)
    ax2.set_xticks(range(1, len(labels) + 1))
    ax2.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax2.axhline(summary["mean_period_s"], color="#c44e52", lw=1, ls="--")
    ax2.set_ylabel("Mean period (s)")

    fig.suptitle("Rosenstein mean period on catalog-quiet windows")
    fig.tight_layout()
    fig.savefig(out_dir / "mean_period.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def build_cfg(config, channel):
    """Pulls the preprocessing settings the pipeline would use."""
    pre, fe = config["preprocessing"], config["feature_extraction"]
    cfg = SimpleNamespace(
        Fs=float(fe["fs"]),
        FREQMIN=float(pre["freq_min"]),
        FREQMAX=float(pre["freq_max"]),
        FILTER_PAD_SEC=(float(pre["filter_pad_sec"])
                        if pre.get("filter_pad_sec") else None),
        WIN_SEC=float(fe["win_sec"]),
    )
    if cfg.FREQMAX >= cfg.Fs / 2:
        raise SystemExit(f"freq_max {cfg.FREQMAX} is at or above Nyquist "
                         f"({cfg.Fs / 2}); fix config.json first")
    return cfg, channel or list(fe["channels"])[0]


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("catalog", help="earthquake catalog CSV")
    p.add_argument("dirs", nargs="+",
                   help="directories of MSEED files, or parents of them")
    p.add_argument("--config", default=None,
                   help="pipeline config.json (default: the pipeline's own)")
    p.add_argument("--channel", default=None,
                   help="component letter (default: first of "
                        "feature_extraction.channels)")
    p.add_argument("--pattern", default="*.mseed", help="MSEED file glob")
    p.add_argument("--split-files", action="store_true",
                   help="treat every MSEED file as its own recording")
    p.add_argument("--out", default="mean_period_out", help="output folder")

    g = p.add_argument_group("sampling")
    g.add_argument("--chunk-sec", type=float, default=3600.0,
                   help="length of each quiet chunk")
    g.add_argument("--chunks", type=int, default=30,
                   help="quiet chunks sampled per recording")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--loud-factor", type=float, default=3.0,
                   help="drop windows whose std exceeds this times the "
                        "recording median")
    g.add_argument("--jobs", type=int, default=4,
                   help="recordings processed in parallel")

    g = p.add_argument_group("event exclusion")
    g.add_argument("--min-mag", type=float, default=0.0,
                   help="ignore catalog events below this magnitude")
    g.add_argument("--unknown-mag", type=float, default=3.0,
                   help="magnitude assumed when a row has none")
    g.add_argument("--pre-sec", type=float, default=120.0,
                   help="seconds excluded before each origin time")
    g.add_argument("--min-post-sec", type=float, default=300.0,
                   help="minimum seconds excluded after each origin time")
    g.add_argument("--coda-factor", type=float, default=3.0,
                   help="multiplier on the magnitude-based coda duration")
    g.add_argument("--station-lat", type=float, default=None)
    g.add_argument("--station-lon", type=float, default=None)
    g.add_argument("--max-dist-km", type=float, default=None,
                   help="ignore events farther than this from the station")

    g = p.add_argument_group("catalog columns (auto-detected if omitted)")
    g.add_argument("--time-col", default=None,
                   help="origin time, or clock time if --date-col is given")
    g.add_argument("--date-col", default=None)
    g.add_argument("--mag-col", default=None)
    g.add_argument("--lat-col", default=None)
    g.add_argument("--lon-col", default=None)
    g.add_argument("--catalog-utc-offset", type=float, default=0.0,
                   help="hours the catalog times are ahead of UTC (TRT = 3)")
    g.add_argument("--dayfirst", action="store_true",
                   help="read ambiguous dates as day/month")
    return p


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def estimate(args, cfg, channel, config):
    """Runs the whole estimate and returns the summary dict.

    Args:
        args: Options as produced by :func:`parse_args`.
        cfg: Object exposing ``Fs``, ``FREQMIN``, ``FREQMAX``,
            ``FILTER_PAD_SEC`` and ``WIN_SEC`` (a
            :class:`chaos.pipeline.Settings` works).
        channel: Component letter to measure.
        config: Parsed pipeline config, used only for reporting.

    Raises:
        SystemExit: If the catalog or waveforms yield nothing measurable.
    """
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    starts, ends, event_times = load_exclusions(args)
    recordings = find_recordings(args.dirs, args.pattern, args.split_files)
    if not recordings:
        raise SystemExit(f"no {args.pattern} files found under {args.dirs}")

    print(f"Waveforms : {len(recordings)} recording(s), channel {channel}, "
          f"{cfg.FREQMIN}–{cfg.FREQMAX} Hz, {cfg.Fs:g} Hz, "
          f"{cfg.WIN_SEC:g} s windows")
    print(f"Sampling  : {args.chunks} x {args.chunk_sec:g} s chunks per "
          "recording\n")

    opts = SimpleNamespace(chunk_sec=args.chunk_sec, chunks=args.chunks,
                           seed=args.seed, loud_factor=args.loud_factor)
    jobs = [(label, files, cfg, channel, starts, ends, opts, i)
            for i, (label, files) in enumerate(recordings)]

    results = []
    workers = max(1, min(args.jobs, len(jobs)))
    if workers == 1:
        results = [process_recording(j) for j in jobs]
        for label, counts, _ in results:
            _report(label, counts)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(process_recording, j) for j in jobs]
            for fut in as_completed(futures):
                label, counts, rows = fut.result()
                _report(label, counts)
                results.append((label, counts, rows))

    rows = [r for _, _, rs in results for r in rs]
    if not rows:
        raise SystemExit("\nno quiet windows could be measured; see the "
                         "per-recording counts above")

    frame = pd.DataFrame(rows).sort_values(["recording", "window_start"])
    if not frame["kept"].any() or frame.loc[frame["kept"],
                                            "mean_period_s"].isna().all():
        raise SystemExit("\nevery quiet window was rejected or unmeasurable")
    frame.to_csv(out_dir / "mean_period_windows.csv", index=False)

    # A catalog in the wrong time zone or format still parses, but then
    # excludes the wrong hours. Events landing inside the waveforms is the
    # cheapest evidence that the two line up.
    lo = min(pd.Timestamp(r["window_start"]).timestamp() for r in rows)
    hi = max(pd.Timestamp(r["window_start"]).timestamp() for r in rows)
    inside = int(((event_times >= lo - 86400) & (event_times <= hi + 86400))
                 .sum())
    if inside == 0:
        print("\n[WARN] no catalog event falls within the waveform period; "
              "check the catalog covers these dates and its time zone")

    summary = summarise(frame, cfg, config, out_dir)
    plot(frame, summary, out_dir)
    print(f"\nWrote {out_dir}/mean_period_windows.csv, "
          "mean_period_summary.json, mean_period.png")
    return summary


def run(argv=None):
    args = parse_args(argv)

    from chaos.pipeline import CONFIG_PATH, load_config
    config = load_config(Path(args.config) if args.config else CONFIG_PATH)
    cfg, channel = build_cfg(config, args.channel)
    estimate(args, cfg, channel, config)


def _report(label, counts):
    print(f"  [{label}] files {counts['files']} | chunks {counts['candidates']}"
          f" -> quiet {counts['quiet']} -> sampled {counts['sampled']} -> used "
          f"{counts['used']} (gaps {counts['gap']}, incomplete "
          f"{counts['incomplete']}, bad fs {counts['bad_fs']}, unreadable "
          f"{counts['unreadable']})")


# --------------------------------------------------------------------------- #
# Pipeline integration                                                         #
# --------------------------------------------------------------------------- #

def options(catalog, dirs, **overrides):
    """Builds estimator options with the CLI defaults plus ``overrides``.

    Keys are the CLI option names with underscores (``chunk_sec``,
    ``catalog_utc_offset``...), so config.json and the command line share one
    set of names and one set of defaults.

    Raises:
        ValueError: If an override names no known option.
    """
    args = parse_args([str(catalog), *(str(d) for d in dirs)])
    for key, value in overrides.items():
        if key in ("catalog", "dirs") or not hasattr(args, key):
            raise ValueError(f"mean_period_estimation: unknown option {key!r}")
        setattr(args, key, value)
    return args


def _cache_key(cfg, channel, args, recordings) -> str:
    """Digest of everything that can change the estimate."""
    catalog = Path(args.catalog).stat()
    payload = {
        "fs": float(cfg.Fs),
        "band": [float(cfg.FREQMIN), float(cfg.FREQMAX)],
        "pad": float(filter_pad_sec(cfg)),
        "win_sec": float(cfg.WIN_SEC),
        "channel": channel,
        "options": {k: v for k, v in sorted(vars(args).items())
                    if k not in ("out", "dirs", "catalog", "jobs", "config")},
        "catalog": [str(Path(args.catalog).resolve()), catalog.st_size,
                    catalog.st_mtime],
        "files": [[label, f.name, f.stat().st_size, f.stat().st_mtime]
                  for label, files in recordings for f in files],
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def setting(config, key, default=None):
    """Reads a shared data setting: ``paths`` first, then the older
    ``mean_period_estimation`` location, so configs written before the keys
    moved keep working."""
    for section in ("paths", "mean_period_estimation"):
        value = (config.get(section) or {}).get(key)
        if value is not None:
            return value
    return default


def project_path(root, value):
    """Resolves a config path against the project root."""
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path


def data_pattern(config, station):
    """The MSEED glob for ``station`` inside the data folder."""
    return setting(config, "data_pattern",
                   setting(config, "pattern", "*_{station}_*.mseed")) \
        .format(station=station)


def lookup_coords(root, config, station, files):
    """Finds a station's coordinates in the configured coordinates CSV.

    Args:
        root: Project root.
        config: Parsed configuration.
        station: Station code.
        files: The station's MSEED paths, read for the network code that
            settles codes used by more than one network.

    Returns:
        ``(latitude, longitude)``, or ``None`` when no file is configured or
        the station is not listed (the latter with a warning).

    Raises:
        ValueError: If the file is missing or the station is ambiguous.
    """
    name = setting(config, "station_coords")
    if not name:
        return None
    path = project_path(root, name)
    if not path.is_file():
        raise ValueError(f"station_coords not found: {path}")
    coords = station_coords(path, station, _data_network([("", list(files))]))
    if coords is None:
        print(f"  [WARN] {station} is not in {path.name}; catalog events are "
              "used regardless of distance")
    return coords


def resolve_mean_period(cfg, config, refresh=False) -> float:
    """Estimates Rosenstein's mean period for a pipeline run.

    Quiet periods are found by screening the station's continuous data
    against the earthquake catalog. The data settings live in ``paths``:

    * ``catalog`` — catalog CSV (required);
    * ``data_dir`` — folder of continuous MSEED (default ``data``);
    * ``data_pattern`` — file glob inside it; ``{station}`` is replaced by
      the run's station (default ``*_{station}_*.mseed``);
    * ``station_coords`` — CSV of station coordinates. The run's station is
      looked up there to fill ``station_lat``/``station_lon``, so that
      ``max_dist_km`` ignores events too far away to disturb it.

    ``mean_period_estimation.options`` takes any CLI option by its underscore
    name (``chunks``, ``dayfirst``, ``max_dist_km``...); coordinates given
    there win over the lookup.

    Relative paths resolve against the project root. Every event of a station
    therefore shares one value, which keeps their features comparable. The
    result is cached under ``<cache_dir>/mean_period/<station>`` keyed on
    every input, so later runs reuse it for free.

    Args:
        cfg: A :class:`chaos.pipeline.Settings`.
        config: The parsed project configuration.
        refresh: Re-estimate even if a cached value exists.

    Returns:
        The median mean period in seconds.

    Raises:
        ValueError: If the settings are incomplete or nothing quiet could be
            measured.
    """
    est = config.get("mean_period_estimation") or {}
    if not setting(config, "catalog"):
        raise ValueError(
            "rosenstein.mean_period is not set, so it is estimated from quiet "
            "data, which needs paths.catalog"
        )

    catalog = project_path(cfg.SCRIPT_DIR, setting(config, "catalog"))
    if not catalog.is_file():
        raise ValueError(f"catalog not found: {catalog}")
    data_dir = project_path(cfg.SCRIPT_DIR, setting(config, "data_dir", "data"))
    if not data_dir.is_dir():
        raise ValueError(f"data_dir not found: {data_dir}")

    overrides = {"split_files": True, **(est.get("options") or {})}
    overrides["pattern"] = data_pattern(config, cfg.STATION)
    args = options(catalog, [data_dir], **overrides)
    channel = cfg.CHANNELS[0]
    recordings = find_recordings(args.dirs, args.pattern, args.split_files)
    if not recordings:
        raise ValueError(f"no {args.pattern} files under {data_dir} to "
                         "estimate the mean period from")

    if args.station_lat is None or args.station_lon is None:
        coords = lookup_coords(cfg.SCRIPT_DIR, config, cfg.STATION,
                               [f for _, fs in recordings for f in fs])
        if coords is not None:
            args.station_lat, args.station_lon = coords
            print(f"[mean_period] {cfg.STATION}: at {coords[0]:.4f}, "
                  f"{coords[1]:.4f}")
    if args.max_dist_km is not None and (args.station_lat is None
                                         or args.station_lon is None):
        print(f"  [WARN] max_dist_km is set but {cfg.STATION} has no "
              "coordinates; it is ignored")

    out_dir = (cfg.SCRIPT_DIR / config["paths"].get("cache_dir", "cache")
               / "mean_period" / cfg.STATION
               / _cache_key(cfg, channel, args, recordings))
    summary_path = out_dir / "mean_period_summary.json"
    if summary_path.is_file() and not refresh:
        value = float(json.loads(summary_path.read_text())["mean_period_s"])
        print(f"[mean_period] {cfg.STATION}: {value:.3f} s (cached, "
              f"{summary_path})")
        cfg.MEAN_PERIOD_SOURCE = str(summary_path)
        return value

    print(f"[mean_period] {cfg.STATION}: estimating from catalog-quiet "
          f"periods in {data_dir}")
    args.out = str(out_dir)
    try:
        summary = estimate(args, cfg, channel, config)
    except SystemExit as exc:
        raise ValueError(f"mean period estimation failed for {cfg.STATION}: "
                         f"{str(exc).strip()}") from None
    cfg.MEAN_PERIOD_SOURCE = str(summary_path)
    return float(summary["mean_period_s"])


if __name__ == "__main__":
    sys.exit(run())
