"""Streams a directory of MSEED files as one continuous decimated recording.

Files are preprocessed in parallel (see :func:`chaos.preprocess.preprocess_file`)
but consumed in time order, so what comes out the far end is a sequence of
contiguous blocks on one global sample grid — the recording as it actually
happened, never materialised in full.

Holding the whole recording would cost ~10 MB per day per three channels, so a
year would not fit. Yielding blocks keeps memory flat in the length of the run.
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from obspy import UTCDateTime, read

from chaos.cache import cache_key, load_block, store_block
from chaos.preprocess import (_resolve_workers, grid_time,
                              preprocess_file_task, preprocess_piece_task,
                              sort_by_start_time)

# A day of three 100 Hz channels is ~200 MB decoded, a comfortable amount for
# each worker to hold; the source files themselves may span weeks.
PIECE_SEC = 86400.0


@dataclass
class Block:
    """A contiguous run of decimated samples on the global grid.

    Attributes:
        start_index: Global grid index of sample 0 (see
            :func:`chaos.preprocess.grid_index`).
        data: Channel name to samples. Every channel has the same length;
            gaps inside the block read as NaN.
        gap_report: Log lines describing what was found in the source files.
    """

    start_index: int
    data: dict[str, np.ndarray]
    gap_report: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(next(iter(self.data.values()))) if self.data else 0

    @property
    def end_index(self) -> int:
        """One past the last grid index this block covers."""
        return self.start_index + len(self)

    def start_time(self, fs: float):
        """Returns the :class:`obspy.UTCDateTime` of sample 0."""
        return grid_time(self.start_index, fs)


@dataclass(frozen=True)
class Piece:
    """One span of a target period together with the files that cover it.

    Pieces stand in for files when the recording is cut out of long
    continuous files by date. They quack like a path where the block cache
    needs one (``name``, ``stem``, ``stat()``), so cached entries are tied to
    the span and to the files it was read from.

    Attributes:
        t0: Start of the span (:class:`obspy.UTCDateTime`).
        t1: End of the span, exclusive.
        files: MSEED paths overlapping the span or its filter padding.
    """

    t0: UTCDateTime
    t1: UTCDateTime
    files: tuple

    @property
    def name(self) -> str:
        return f"{self.t0.strftime('%Y%m%dT%H%M%S')}-{self.t1.strftime('%Y%m%dT%H%M%S')}"

    @property
    def stem(self) -> str:
        return self.name

    def __hash__(self):
        return hash((float(self.t0), float(self.t1), self.files))

    def stat(self):
        """Combined size and newest mtime of the source files."""
        stats = [Path(f).stat() for f in self.files]
        return SimpleNamespace(st_size=sum(st.st_size for st in stats),
                               st_mtime=max(st.st_mtime for st in stats))


def file_spans(mseed_files):
    """Returns ``(start, end, path)`` from each readable file header, by start.

    Args:
        mseed_files: Paths to inspect.

    Returns:
        Tuple ``(spans, unreadable_names)``.
    """
    spans, unreadable = [], []
    for path in mseed_files:
        try:
            st = read(str(path), headonly=True)
        except Exception:
            unreadable.append(Path(path).name)
            continue
        if len(st):
            spans.append((min(tr.stats.starttime for tr in st),
                          max(tr.stats.endtime for tr in st), Path(path)))
    spans.sort(key=lambda s: s[0])
    return spans, unreadable


def plan_pieces(spans, t0, t1, pad_sec, piece_sec=PIECE_SEC):
    """Cuts ``[t0, t1)`` into pieces, each listing the files it needs.

    Pieces start on multiples of ``piece_sec`` from the epoch (UTC midnights
    for day pieces), so a period's pieces line up with another period's and
    the block cache can share them. Pieces no file overlaps are left out.

    Args:
        spans: ``(start, end, path)`` tuples from :func:`file_spans`.
        t0: Period start.
        t1: Period end, exclusive.
        pad_sec: Filter padding each piece reads either side.
        piece_sec: Piece length in seconds.

    Returns:
        A list of :class:`Piece`, in time order.
    """
    t0, t1 = UTCDateTime(t0), UTCDateTime(t1)
    pieces = []
    a = UTCDateTime(np.floor(float(t0) / piece_sec) * piece_sec)
    while a < t1:
        lo, hi = max(a, t0), min(a + piece_sec, t1)
        files = tuple(path for s, e, path in spans
                      if e >= lo - pad_sec and s <= hi + pad_sec)
        own = any(e > lo and s < hi for s, e, _ in spans)
        if files and own:
            pieces.append(Piece(lo, hi, files))
        a += piece_sec
    return pieces


def _merge_into(target: Block, start_index: int, data: dict) -> None:
    """Writes ``data`` into ``target`` at ``start_index``, keeping real samples.

    Day-long files routinely overhang midnight by a second or two, so the same
    instant can arrive twice. Whichever copy is not NaN wins; if both are real
    the first one stays, because they are the same ground motion recorded once.
    """
    for channel, samples in data.items():
        if channel not in target.data:
            continue
        lo = start_index - target.start_index
        hi = lo + len(samples)
        lo_c, hi_c = max(0, lo), min(len(target), hi)
        if hi_c <= lo_c:
            continue
        dest = target.data[channel][lo_c:hi_c]
        src = samples[lo_c - lo: hi_c - lo]
        np.copyto(dest, src, where=np.isnan(dest) & ~np.isnan(src))


def _extend(block: Block, start_index: int, data: dict, fs: float) -> Block:
    """Grows ``block`` so it also covers ``data`` at ``start_index``."""
    new_start = min(block.start_index, start_index)
    new_end = max(block.end_index, start_index + len(next(iter(data.values()))))
    grown = Block(
        start_index=new_start,
        data={
            ch: np.full(new_end - new_start, np.nan, dtype=np.float64)
            for ch in block.data
        },
        gap_report=block.gap_report,
    )
    _merge_into(grown, block.start_index, block.data)
    _merge_into(grown, start_index, data)
    return grown


def iter_blocks(cfg, mseed_files):
    """Yields the recording as time-ordered contiguous blocks.

    Consecutive files are joined into one block when the seam between them is
    small enough to be a gap worth representing; a longer outage ends the block
    instead, so a week off air does not turn into a million NaN rows.

    Args:
        cfg: Configuration object (see :class:`chaos.pipeline.Settings`).
        mseed_files: MSEED paths in any order; they are put into time order
            here, which is not the same as name order.

    Yields:
        :class:`Block` instances in increasing time order.
    """
    if not mseed_files:
        return

    mseed_files, unreadable = sort_by_start_time(mseed_files)
    for name in unreadable:
        print(f"  [WARN] skipping unreadable file: {name}")
    if not mseed_files:
        return

    workers = _resolve_workers(cfg, len(mseed_files))
    key = cache_key(cfg) if cfg.CACHE_ENABLED else None
    results = _stream(cfg, mseed_files, workers, key,
                      preprocess_file_task,
                      lambda i: _task_args(cfg, mseed_files, i))
    yield from _stitch(cfg, results)


def iter_piece_blocks(cfg, pieces):
    """Like :func:`iter_blocks`, for pieces cut out of longer files.

    Args:
        cfg: Configuration object.
        pieces: :class:`Piece` list in time order (see :func:`plan_pieces`).

    Yields:
        :class:`Block` instances in increasing time order.
    """
    if not pieces:
        return
    workers = _resolve_workers(cfg, len(pieces))
    key = cache_key(cfg) if cfg.CACHE_ENABLED else None
    results = _stream(cfg, pieces, workers, key, preprocess_piece_task,
                      lambda i: (cfg, pieces[i]))
    yield from _stitch(cfg, (r for r in results if r is not None))


def _stitch(cfg, results):
    """Joins per-source results into contiguous blocks (see iter_blocks)."""
    max_gap_samples = int(round(cfg.MAX_GAP_SEC * cfg.Fs))
    pending: Block | None = None

    for name, start_index, data, report in results:
        print(f"  [READ] {name}: {report[0].split(': ', 1)[-1]}")

        if pending is None:
            pending = Block(start_index, data, list(report))
            continue

        seam = start_index - pending.end_index
        if seam > max_gap_samples:
            yield pending
            pending = Block(start_index, data, list(report))
            continue

        pending.gap_report.extend(report)
        pending = _extend(pending, start_index, data, cfg.Fs)

        # Sources arrive in order of start time, so nothing still to come can
        # begin before this one did: everything earlier than start_index is
        # settled and can go downstream now. Without this the whole recording
        # would accumulate in one block, which is exactly what streaming is
        # supposed to avoid.
        head, pending = _split(pending, start_index)
        if head is not None:
            yield head

    if pending is not None and len(pending):
        yield pending


def _split(block: Block, at_index: int):
    """Cuts ``block`` at ``at_index`` into a settled head and a live tail.

    Args:
        block: The block to cut.
        at_index: Global grid index to cut at.

    Returns:
        Tuple ``(head, tail)``. ``head`` is ``None`` when there is nothing
        settled yet; the two are contiguous, so the consumer sees no seam.
    """
    cut = at_index - block.start_index
    if cut <= 0:
        return None, block
    cut = min(cut, len(block))

    head = Block(
        start_index=block.start_index,
        data={ch: arr[:cut] for ch, arr in block.data.items()},
        gap_report=block.gap_report,
    )
    tail = Block(
        start_index=block.start_index + cut,
        data={ch: arr[cut:].copy() for ch, arr in block.data.items()},
        gap_report=[],
    )
    return head, tail


def _stream(cfg, sources, workers, key, task, task_args):
    """Preprocesses sources in parallel, yielding results in source order.

    At most ``workers + 2`` sources are in flight, which bounds how much
    decoded waveform is resident at once regardless of how many there are.

    Args:
        cfg: Configuration object.
        sources: Sorted MSEED paths or :class:`Piece` objects.
        workers: Process pool size.
        key: Cache key, or ``None`` when caching is off.
        task: Pickle-friendly function run on ``task_args(i)``; returns
            ``(name, start_index, channels, gap_report_lines)`` or ``None``.
        task_args: Builds the argument tuple for source ``i``.

    Yields:
        Whatever ``task`` returns, one per source, in order.
    """
    if workers == 1:
        for i in range(len(sources)):
            cached = _cached(cfg, sources, i, key)
            if cached is not None:
                yield cached
                continue
            result = task(task_args(i))
            if key is not None and result is not None:
                store_block(cfg, key, sources[i], result)
            yield result
        return

    in_flight: dict[int, object] = {}
    # Each finished source sits in memory until the consumer reaches it, and a
    # day of three channels is ~10 MB, so the lookahead is what really sets
    # peak memory. A couple more than the worker count keeps everyone busy
    # without hoarding a queue of decoded waveform nobody is ready for.
    ahead = workers + 2

    with ProcessPoolExecutor(max_workers=workers) as ex:
        next_to_submit = 0
        for i in range(len(sources)):
            while next_to_submit < len(sources) and \
                    next_to_submit < i + ahead:
                j = next_to_submit
                cached = _cached(cfg, sources, j, key)
                if cached is not None:
                    in_flight[j] = ("cached", cached)
                else:
                    in_flight[j] = ("future", ex.submit(task, task_args(j)))
                next_to_submit += 1

            kind, slot = in_flight.pop(i)
            result = slot if kind == "cached" else slot.result()
            if key is not None and kind == "future" and result is not None:
                store_block(cfg, key, sources[i], result)
            yield result


def _task_args(cfg, mseed_files, i):
    """Builds the argument tuple for one file, with its neighbours as pads."""
    return (
        cfg,
        mseed_files[i],
        mseed_files[i - 1] if i > 0 else None,
        mseed_files[i + 1] if i + 1 < len(mseed_files) else None,
    )


def _cached(cfg, sources, i, key):
    """Returns a cached result for source ``i``, or ``None`` on a miss."""
    if key is None:
        return None
    return load_block(cfg, key, sources[i])


class RollingBuffer:
    """Emits fixed-length windows from blocks arriving one after another.

    Only the samples a future window could still need are retained, so memory
    stays at roughly one window plus one block however long the recording runs.

    Args:
        channels: Channel names to track.
        win_size: Window length in samples.
        step_size: Distance between consecutive window starts, in samples.
    """

    def __init__(self, channels, win_size: int, step_size: int):
        self.channels = list(channels)
        self.win_size = int(win_size)
        self.step_size = int(step_size)
        self._data = {ch: np.empty(0, dtype=np.float64) for ch in self.channels}
        self._origin = 0           # grid index of _data[ch][0]
        self._end_index = None     # one past the last buffered grid index
        self._next_start = 0       # grid index of the next window to emit

    def push(self, block: Block):
        """Adds a block and yields every window it completes.

        A block that does not begin exactly where the buffered samples end is
        a fresh start, not a continuation: the tail is dropped and the window
        grid restarts, so no window is ever assembled from two sides of an
        outage.

        Args:
            block: The next block in time order.

        Yields:
            Tuple ``(window_start_index, {channel: segment})``.
        """
        if self._end_index is None or block.start_index != self._end_index:
            self.reset(block.start_index)

        for ch in self.channels:
            incoming = block.data.get(ch)
            if incoming is None:
                incoming = np.full(len(block), np.nan, dtype=np.float64)
            self._data[ch] = np.concatenate([self._data[ch], incoming])
        self._end_index += len(block)

        yield from self._drain()

    def reset(self, new_origin: int) -> None:
        """Drops everything buffered and restarts the grid at ``new_origin``.

        Windows begin on a fixed lattice of whole ``step_size`` multiples
        measured from the epoch, not at whichever sample the recording happens
        to open on. A station that came online three seconds before midnight
        would otherwise shift every window in the run, so re-running with one
        more day of data prepended would move every row and make two runs of
        the same event impossible to compare.
        """
        self._data = {ch: np.empty(0, dtype=np.float64) for ch in self.channels}
        self._origin = new_origin
        self._end_index = new_origin
        self._next_start = -(-new_origin // self.step_size) * self.step_size

    def _drain(self):
        while self._next_start + self.win_size <= self._end_index:
            lo = self._next_start - self._origin
            hi = lo + self.win_size
            yield self._next_start, {
                ch: self._data[ch][lo:hi] for ch in self.channels
            }
            self._next_start += self.step_size
            self._trim()

    def _trim(self) -> None:
        """Discards samples no future window can reach."""
        drop = min(self._next_start, self._end_index) - self._origin
        if drop <= 0:
            return
        for ch in self.channels:
            self._data[ch] = self._data[ch][drop:].copy()
        self._origin += drop

    @property
    def buffered_samples(self) -> int:
        """Samples currently retained per channel (used by the tests)."""
        return len(self._data[self.channels[0]]) if self.channels else 0
