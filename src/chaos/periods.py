"""Target periods: date ranges cut out of the continuous data folder.

A run can cover a date range instead of a ``raw/<station>/<event>`` folder.
The range is read straight out of the long files in ``paths.data_dir``, and
with ``target.quiet_window`` the pipeline instead picks several seismically
quiet ranges of the same length and runs on each of them.

Quiet here means no catalog event of at least ``quiet_window.min_mag`` within
``quiet_window.max_dist_km`` of the station — a range with no event at all
does not exist near an active fault — and the range, with its filter padding,
lying entirely inside recorded data.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import UTCDateTime

from chaos.mean_period import (data_pattern, load_exclusions, lookup_coords,
                               options, overlaps, project_path, setting)
from chaos.preprocess import filter_pad_sec
from chaos.recording import file_spans

DAY = 86400.0

# Files ``NET_STA_...`` name their station second.
_STATION_RE = re.compile(r"^[A-Za-z0-9]+_([A-Za-z0-9]+)_")


@dataclass(frozen=True)
class Period:
    """A target date range.

    Attributes:
        t0: Start (:class:`obspy.UTCDateTime`).
        t1: End, exclusive.
        label: Folder name the run writes to, e.g. ``20250416_7d`` or
            ``quiet_7d/20240612``.
    """

    t0: UTCDateTime
    t1: UTCDateTime
    label: str

    @property
    def days(self) -> float:
        return (self.t1 - self.t0) / DAY

    def __hash__(self):
        # UTCDateTime does not hash to an int; runs are keyed by period.
        return hash((float(self.t0), float(self.t1), self.label))


def parse_date(value) -> UTCDateTime:
    """Parses a start date; strings without a zone are read as UTC.

    Raises:
        ValueError: If the value is not a date.
    """
    try:
        return UTCDateTime(str(value))
    except Exception:
        raise ValueError(f"target.start_date {value!r} is not a date "
                         "(use YYYY-MM-DD or an ISO timestamp)") from None


def _days_label(days) -> str:
    return f"{days:g}d"


def target_period(start_date, duration_days) -> Period:
    """The period ``[start_date, start_date + duration_days)``."""
    t0 = parse_date(start_date)
    days = float(duration_days)
    if days <= 0:
        raise ValueError(f"target.duration_days must be > 0, got {days}")
    return Period(t0, t0 + days * DAY,
                  f"{t0.strftime('%Y%m%d')}_{_days_label(days)}")


def data_dir(root, config) -> Path:
    """The continuous data folder."""
    return project_path(root, setting(config, "data_dir", "data"))


def station_files(root, config, station):
    """The station's MSEED files in the data folder, in name order."""
    return sorted(data_dir(root, config).glob(data_pattern(config, station)))


def discover_stations(root, config):
    """Station codes that have files in the data folder."""
    stations = set()
    for path in data_dir(root, config).glob("*.mseed"):
        match = _STATION_RE.match(path.name)
        if match:
            stations.add(match.group(1))
    return sorted(stations)


def coverage(spans, tolerance=1.0):
    """Merges file spans into stretches of continuous coverage.

    Args:
        spans: ``(start, end, path)`` tuples from
            :func:`chaos.recording.file_spans`.
        tolerance: Seams up to this many seconds still count as continuous.

    Returns:
        A list of ``[start, end]`` pairs of epoch seconds.
    """
    stretches = []
    for s, e, _ in spans:
        s, e = float(s), float(e)
        if stretches and s <= stretches[-1][1] + tolerance:
            stretches[-1][1] = max(stretches[-1][1], e)
        else:
            stretches.append([s, e])
    return stretches


def quiet_candidates(stretches, ex_starts, ex_ends, days, pad):
    """UTC-midnight starts of every quiet ``days``-long range.

    Args:
        stretches: Coverage from :func:`coverage`.
        ex_starts: Sorted starts of merged event exclusion intervals.
        ex_ends: Their ends.
        days: Range length in days.
        pad: Seconds of data needed either side for filtering.

    Returns:
        A list of epoch seconds.
    """
    length = days * DAY
    out = []
    for s, e in stretches:
        a = np.ceil((s + pad) / DAY) * DAY
        while a + length + pad <= e:
            if not overlaps(ex_starts, ex_ends, a, a + length):
                out.append(float(a))
            a += DAY
    return out


def pick_quiet(candidates, days, samples, seed):
    """Randomly picks up to ``samples`` non-overlapping ranges.

    Candidates are visited in a seeded random order and each is kept if it
    does not overlap one already kept, so the same inputs and seed always
    give the same periods.

    Returns:
        The chosen starts, in time order.
    """
    rng = np.random.default_rng(seed)
    length = days * DAY
    chosen = []
    for i in rng.permutation(len(candidates)):
        a = candidates[i]
        if all(abs(a - b) >= length for b in chosen):
            chosen.append(a)
            if len(chosen) >= samples:
                break
    return sorted(chosen)


def quiet_periods(root, config, station, days, verbose=True):
    """Picks the quiet periods for ``station`` from ``config``.

    Args:
        root: Project root.
        config: Parsed configuration (``paths``, ``quiet_window``,
            ``mean_period_estimation.options`` for catalog parsing).
        station: Station code.
        days: Length of each period in days.
        verbose: Print what was found.

    Returns:
        Tuple ``(periods, info)`` where ``info`` describes the selection
        (candidate count, parameters) for the run's records.

    Raises:
        ValueError: If the inputs are missing or nothing quiet exists.
    """
    qw = config.get("quiet_window") or {}
    samples = int(qw.get("samples", 15))
    min_mag = float(qw.get("min_mag", 3.0))
    max_dist = qw.get("max_dist_km", 300)
    seed = int(qw.get("seed", 0))
    days = float(days)
    if days <= 0:
        raise ValueError(f"target.duration_days must be > 0, got {days}")

    catalog = setting(config, "catalog")
    if not catalog:
        raise ValueError("quiet_window needs paths.catalog")
    catalog = project_path(root, catalog)
    if not catalog.is_file():
        raise ValueError(f"catalog not found: {catalog}")

    files = station_files(root, config, station)
    if not files:
        raise ValueError(f"no {data_pattern(config, station)} files in "
                         f"{data_dir(root, config)}")
    spans, unreadable = file_spans(files)
    for name in unreadable:
        print(f"  [WARN] skipping unreadable file: {name}")
    stretches = coverage(spans)

    catalog_opts = dict((config.get("mean_period_estimation") or {})
                        .get("options") or {})
    for key in ("chunks", "chunk_sec", "jobs", "loud_factor", "split_files",
                "seed"):
        catalog_opts.pop(key, None)
    args = options(catalog, [data_dir(root, config)], **catalog_opts)
    args.min_mag = min_mag
    args.max_dist_km = None if max_dist is None else float(max_dist)
    coords = lookup_coords(root, config, station, files)
    if coords is not None:
        args.station_lat, args.station_lon = coords
    elif args.max_dist_km is not None:
        print(f"  [WARN] {station} has no coordinates; every M>={min_mag:g} "
              "event counts regardless of distance")
    try:
        ex_starts, ex_ends, _ = load_exclusions(args)
    except SystemExit as exc:
        raise ValueError(f"reading the catalog failed: {exc}") from None

    pad = filter_pad_sec(_PadCfg(config))
    cands = quiet_candidates(stretches, ex_starts, ex_ends, days, pad)
    chosen = pick_quiet(cands, days, samples, seed)
    info = {
        "station": station,
        "days": days,
        "samples_requested": samples,
        "samples_found": len(chosen),
        "candidates": len(cands),
        "min_mag": min_mag,
        "max_dist_km": args.max_dist_km,
        "station_coords": list(coords) if coords else None,
        "seed": seed,
    }
    if not chosen:
        raise ValueError(
            f"no quiet {days:g}-day period for {station} (no M>={min_mag:g} "
            f"within {max_dist} km, inside recorded data); raise "
            "quiet_window.min_mag or shorten target.duration_days")
    if verbose:
        print(f"[quiet] {station}: {len(cands)} quiet {days:g}-day start "
              f"days, picked {len(chosen)} of {samples} requested")
    if len(chosen) < samples:
        print(f"  [WARN] only {len(chosen)} non-overlapping quiet periods "
              f"exist; raise quiet_window.min_mag or shorten "
              "target.duration_days for more")

    group = f"quiet_{_days_label(days)}"
    periods = [Period(UTCDateTime(a), UTCDateTime(a + days * DAY),
                      f"{group}/{UTCDateTime(a).strftime('%Y%m%d')}")
               for a in chosen]
    return periods, info


def write_periods(path, periods, info):
    """Writes the chosen periods as CSV (plus the selection parameters)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"label": p.label.split("/")[-1], "start": str(p.t0),
                   "end": str(p.t1), "days": p.days} for p in periods]) \
        .to_csv(path, index=False)
    import json
    path.with_suffix(".json").write_text(json.dumps(info, indent=2),
                                         encoding="utf-8")


class _PadCfg:
    """Just enough of a Settings for :func:`filter_pad_sec`."""

    def __init__(self, config):
        pre = config["preprocessing"]
        self.FREQMIN = float(pre["freq_min"])
        self.FILTER_PAD_SEC = (float(pre["filter_pad_sec"])
                               if pre.get("filter_pad_sec") else None)
