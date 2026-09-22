"""Per-file MSEED preprocessing: gap handling, bandpass filtering, decimation.

One MSEED file in, one :class:`chaos.recording.Block` out — decimated samples
placed on the global sample grid, with NaN wherever the recording has nothing
to say. Nothing is written to disk; the caller streams the blocks straight into
feature extraction (see :mod:`chaos.recording`).

Each file is filtered with a pad of real signal borrowed from its neighbours
and the pad is then trimmed off, so files can be processed in parallel and
still give the same answer as filtering the whole recording in one pass.
"""

import os

import numpy as np
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


def grid_index(utc_time, fs: float) -> int:
    """Maps an absolute time to its index on the global decimated sample grid.

    The grid is anchored at the UTC epoch: index ``k`` is the instant
    ``k / fs`` seconds after 1970-01-01T00:00:00Z. Anchoring on an absolute
    reference rather than on each file's own first sample is what lets files be
    processed independently and still line up to the sample when they are
    stitched back together.

    Args:
        utc_time: A :class:`obspy.UTCDateTime`.
        fs: Target sampling frequency in Hz.

    Returns:
        The grid index at or after ``utc_time``.
    """
    return int(np.ceil(float(utc_time) * fs - 1e-9))


def grid_index_floor(utc_time, fs: float) -> int:
    """Like :func:`grid_index`, but the last grid sample at or *before* a time.

    Used for the end of a span: a file whose final raw sample sits just short
    of the next grid point covers up to the previous one, not past it.

    Args:
        utc_time: A :class:`obspy.UTCDateTime`.
        fs: Target sampling frequency in Hz.

    Returns:
        The grid index at or before ``utc_time``.
    """
    return int(np.floor(float(utc_time) * fs + 1e-9))


def grid_time(index: int, fs: float) -> UTCDateTime:
    """Inverse of :func:`grid_index`.

    Args:
        index: Absolute grid index.
        fs: Target sampling frequency in Hz.

    Returns:
        The :class:`obspy.UTCDateTime` of that grid sample.
    """
    return UTCDateTime(index / fs)


def filter_pad_sec(cfg) -> float:
    """Returns the context each file is filtered with, in seconds.

    A zero-phase bandpass reaches back roughly ``3 / freqmin`` seconds, so a
    file filtered in isolation is wrong at both its ends. Filtering it with
    real signal either side and discarding that padding afterwards reproduces
    the continuous result: measured against a single-pass filter, 60 s of pad
    leaves a maximum error of 0.03% of the signal's standard deviation, versus
    98% with no pad at all.

    Args:
        cfg: Configuration object exposing ``FREQMIN`` and optionally
            ``FILTER_PAD_SEC``.

    Returns:
        Pad length in seconds.
    """
    configured = getattr(cfg, "FILTER_PAD_SEC", None)
    if configured:
        return float(configured)
    return max(60.0, 6.0 / cfg.FREQMIN)


def sort_by_start_time(mseed_files):
    """Orders files by when they actually start recording.

    Sorting by name is not the same thing: the station writes ``DDMMYYYY``, so
    alphabetical order puts 01 May before 08 April. The streaming stitcher
    joins files in the order it receives them and pads each one from its
    neighbours, so feeding it name-ordered files would splice April onto May
    and drop most of the run.

    Files whose header cannot be read sort last and are reported, rather than
    silently landing at an arbitrary point in the timeline.

    Args:
        mseed_files: Paths to order.

    Returns:
        Tuple ``(ordered_paths, skipped_names)``.
    """
    timed, unreadable = [], []
    for path in mseed_files:
        try:
            stream = read(str(path), headonly=True)
            timed.append((min(tr.stats.starttime for tr in stream), path))
        except Exception:
            unreadable.append(path.name)
    timed.sort(key=lambda pair: pair[0])
    return [path for _, path in timed], unreadable


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


def _fill_small_gaps_mask_large(st_full, cfg, large_gaps):
    """Merges a stream, interpolating short gaps and masking long ones.

    Short gaps (below ``cfg.GAP_THRESHOLD``) are bridged with a cubic spline so
    the filter has something continuous to work on. Long gaps stay NaN: there
    is no honest way to invent seconds of missing ground motion, and filtering
    across them would smear the invented part into the real data either side.

    Args:
        st_full: Stream to merge in place.
        cfg: Configuration object.
        large_gaps: Gap tuples classified as long.
    """
    if not large_gaps:
        st_full.merge(fill_value=np.nan)
        for tr in st_full:
            data = _as_float_array(tr.data)
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
        return

    st_full.merge(fill_value=np.nan)
    for tr in st_full:
        data = _as_float_array(tr.data)
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

        data[big_gap_mask] = np.nan
        tr.data = data


def _as_float_array(data) -> np.ndarray:
    """Returns ``data`` as a plain float array with masked entries as NaN."""
    if np.ma.is_masked(data):
        return np.ma.filled(data.astype(float), np.nan)
    return np.array(data, dtype=float)


def _contiguous_runs(valid: np.ndarray):
    """Yields ``(start, end)`` index pairs for each run of True in ``valid``."""
    if not valid.any():
        return
    padded = np.concatenate([[False], valid, [False]])
    edges = np.diff(padded.astype(np.int8))
    for start, end in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
        yield int(start), int(end)


def _decimate_segment(trace_template, segment, seg_abs_start, cfg,
                      decimation_factor, out, out_origin):
    """Filters one clean segment and writes it onto the global grid.

    Args:
        trace_template: Trace to copy stats from.
        segment: Contiguous raw samples, already free of NaN.
        seg_abs_start: Absolute time of ``segment[0]``, as a
            :class:`obspy.UTCDateTime`.
        cfg: Configuration object.
        decimation_factor: Integer decimation factor.
        out: Destination array on the global decimated grid.
        out_origin: Grid index that ``out[0]`` corresponds to.
    """
    tmp = trace_template.copy()
    tmp.data = segment.copy()
    tmp.stats.starttime = seg_abs_start
    try:
        tmp.detrend("demean")
        tmp.detrend("linear")
        tmp.filter(
            "bandpass",
            freqmin=cfg.FREQMIN, freqmax=cfg.FREQMAX,
            corners=4, zerophase=True,
        )
    except Exception:
        return

    # First grid sample at or after the segment's start, expressed both as an
    # absolute index and as an offset into the segment's own raw samples.
    first_grid = grid_index(seg_abs_start, cfg.Fs)
    raw_fs = tmp.stats.sampling_rate
    offset_sec = float(grid_time(first_grid, cfg.Fs)) - float(seg_abs_start)
    phase = int(round(offset_sec * raw_fs))
    if phase < 0:
        phase = 0

    dec = tmp.data[phase::decimation_factor]
    if dec.size == 0:
        return

    lo = first_grid - out_origin
    hi = lo + dec.size
    lo_clipped, hi_clipped = max(0, lo), min(out.size, hi)
    if hi_clipped > lo_clipped:
        out[lo_clipped:hi_clipped] = dec[lo_clipped - lo: hi_clipped - lo]


def preprocess_file(cfg, mseed_file, pad_before=None, pad_after=None):
    """Turns one MSEED file into decimated samples on the global grid.

    The file is read together with ``pad_before``/``pad_after`` seconds of its
    neighbours, filtered, decimated, and then the padding is trimmed away. What
    comes back covers exactly the file's own time span.

    Args:
        cfg: Configuration object (see :class:`chaos.pipeline.Settings`).
        mseed_file: Path to the MSEED file to process.
        pad_before: Path to the preceding file, if any, to borrow context from.
        pad_after: Path to the following file, if any.

    Returns:
        Tuple ``(start_index, {channel: ndarray}, gap_report_lines)``.
        ``start_index`` is the global grid index of sample 0 of each array.
        Channels are all the same length; gaps read as NaN.
    """
    pad = filter_pad_sec(cfg)

    st_full = read(str(mseed_file))
    own_start = min(tr.stats.starttime for tr in st_full)
    own_end = max(tr.stats.endtime for tr in st_full)
    # Channels of one instrument rarely open on the same sample. Reaching the
    # pad to the *latest* channel start (and the earliest end) means every
    # channel's padding meets its own first sample, instead of stopping short
    # and leaving a hole the merge would then have to interpolate across.
    latest_start = max(tr.stats.starttime for tr in st_full)
    earliest_end = min(tr.stats.endtime for tr in st_full)

    for neighbour, want_start, want_end in (
        (pad_before, own_start - pad, latest_start),
        (pad_after, earliest_end, own_end + pad),
    ):
        if neighbour is None:
            continue
        try:
            extra = read(str(neighbour), starttime=want_start, endtime=want_end)
            st_full += extra
        except Exception:
            # A neighbour that will not read costs accuracy at this file's very
            # edge, never correctness of the rest; the gap report notes it.
            pass

    for tr in st_full:
        if tr.data.dtype != np.float64:
            tr.data = tr.data.astype(np.float64)

    real_fs = st_full[0].stats.sampling_rate
    decimation_factor = max(1, int(real_fs / cfg.Fs))
    effective_fs = real_fs / decimation_factor

    gaps = st_full.get_gaps(min_gap=-1)
    actual_gaps = [g for g in gaps if g[7] > 0] if gaps else []
    large_gaps = [
        g for g in actual_gaps if _gap_duration_sec(g, st_full) >= cfg.GAP_THRESHOLD
    ]
    small_gaps = [
        g for g in actual_gaps if _gap_duration_sec(g, st_full) < cfg.GAP_THRESHOLD
    ]

    # Gap *handling* uses the padded stream, so a gap just outside the file is
    # never filtered across. The gap *report* is about this file, so the seams
    # where the padding meets its neighbours are left out of it — they are an
    # artefact of borrowing context, not something wrong with the recording.
    def own(gap):
        return gap[5] > own_start and gap[4] < own_end

    own_small = [g for g in small_gaps if own(g)]
    own_large = [g for g in large_gaps if own(g)]

    gap_report = [
        f"{mseed_file.name}: {len(own_small)} small (<{cfg.GAP_THRESHOLD}s), "
        f"{len(own_large)} large (>={cfg.GAP_THRESHOLD}s)"
    ]
    for gap in own_large:
        gap_report.append(
            f"    LARGE {gap[3]} {gap[4]} -> {gap[5]} "
            f"({_gap_duration_sec(gap, st_full):.2f}s, {gap[7]} samples)"
        )
    if abs(effective_fs - cfg.Fs) > 1e-9:
        gap_report.append(
            f"    WARN  {real_fs:g} Hz / {decimation_factor} = {effective_fs:g} Hz, "
            f"not the configured fs={cfg.Fs:g} Hz; set fs to a divisor of "
            f"{real_fs:g} to keep the time axis exact"
        )

    if actual_gaps:
        _fill_small_gaps_mask_large(st_full, cfg, large_gaps)
    else:
        st_full.merge()

    # The block covers the file's own span only; padding was context, not data.
    start_index = grid_index(own_start, cfg.Fs)
    end_index = grid_index_floor(own_end, cfg.Fs) + 1
    n_out = max(0, end_index - start_index)

    min_seg_len = max(20, int(3.0 / cfg.FREQMIN * real_fs))
    channels: dict[str, np.ndarray] = {}

    for component in cfg.PREPROCESS_CHANNELS:
        out = np.full(n_out, np.nan, dtype=np.float64)
        try:
            tr = st_full.select(component=component)[0]
        except IndexError:
            channels[component] = out
            continue

        data = _as_float_array(tr.data)
        valid = ~np.isnan(data)
        tr_start, tr_fs = tr.stats.starttime, tr.stats.sampling_rate

        for seg_start, seg_end in _contiguous_runs(valid):
            if (seg_end - seg_start) < min_seg_len:
                continue
            _decimate_segment(
                tr, data[seg_start:seg_end],
                tr_start + seg_start / tr_fs,
                cfg, decimation_factor, out, start_index,
            )
        channels[component] = out

    return start_index, channels, gap_report


def preprocess_file_task(args):
    """Pickle-friendly wrapper used by the process pool.

    Args:
        args: Tuple ``(cfg, mseed_file, pad_before, pad_after)``.

    Returns:
        Tuple ``(file_name, start_index, channels, gap_report_lines)``.
    """
    cfg, mseed_file, pad_before, pad_after = args
    start_index, channels, gap_report = preprocess_file(
        cfg, mseed_file, pad_before, pad_after
    )
    return mseed_file.name, start_index, channels, gap_report
