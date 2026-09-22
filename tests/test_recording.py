"""Unit tests for chaos/recording.py — stitching and the rolling buffer."""

import numpy as np
import pytest
from obspy import UTCDateTime

from chaos.recording import Block, RollingBuffer, iter_blocks
from conftest import START


def _block(start_index, n, channels=("N",), fill=1.0):
    return Block(
        start_index=start_index,
        data={ch: np.full(n, fill, dtype=np.float64) for ch in channels},
    )



def _joined(blocks, channel="N"):
    """Asserts the blocks tile the timeline and returns the joined samples.

    Blocks are emitted as soon as they are settled rather than accumulated, so
    a contiguous recording arrives in several pieces. What must hold is that
    the pieces abut exactly -- no gap, no overlap, nothing dropped.
    """
    import numpy as _np

    for before, after in zip(blocks, blocks[1:]):
        assert before.end_index == after.start_index, (
            f"blocks do not abut: {before.end_index} then {after.start_index}"
        )
    return _np.concatenate([b.data[channel] for b in blocks])


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------

def test_block_length_and_end_index():
    b = _block(100, 50)
    assert len(b) == 50
    assert b.end_index == 150


def test_block_start_time_is_on_the_global_grid():
    fs = 5.0
    t = UTCDateTime(START)
    from chaos.preprocess import grid_index
    b = _block(grid_index(t, fs), 10)
    assert b.start_time(fs) == t


# ---------------------------------------------------------------------------
# iter_blocks — stitching whole files together
# ---------------------------------------------------------------------------

def test_contiguous_files_tile_one_unbroken_recording(fake_cfg, mseed_factory):
    """Three back-to-back hours are one recording: the blocks must abut and
    together cover every sample, with no seam left inside them."""
    files = [
        mseed_factory(f"c{i}.mseed", i * 3600, 3600.0, seed=i)
        for i in range(3)
    ]
    joined = _joined(list(iter_blocks(fake_cfg, files)))

    assert len(joined) == int(3 * 3600 * fake_cfg.Fs)
    assert not np.isnan(joined).any()


def test_short_seam_is_bridged_with_nan(fake_cfg, mseed_factory):
    """A seam shorter than max_gap_sec stays inside one block as NaN, so the
    windows crossing it are reported as empty rather than silently joined."""
    fake_cfg.MAX_GAP_SEC = 200.0
    files = [
        mseed_factory("s0.mseed", 0, 3600.0, seed=0),
        mseed_factory("s1.mseed", 3600 + 100, 3600.0, seed=1),
    ]
    joined = _joined(list(iter_blocks(fake_cfg, files)))
    assert np.isnan(joined).sum() == int(100 * fake_cfg.Fs)


def test_long_outage_ends_the_block(fake_cfg, mseed_factory):
    """A four-hour outage must not become a million NaN rows."""
    fake_cfg.MAX_GAP_SEC = 200.0
    files = [
        mseed_factory("o0.mseed", 0, 3600.0, seed=0),
        mseed_factory("o1.mseed", 3600 * 5, 3600.0, seed=1),
    ]
    blocks = list(iter_blocks(fake_cfg, files))

    assert len(blocks) == 2
    assert blocks[0].start_time(fake_cfg.Fs) == UTCDateTime(START)
    assert blocks[1].start_time(fake_cfg.Fs) == UTCDateTime(START) + 3600 * 5
    for b in blocks:
        assert not np.isnan(b.data["N"]).any()


def test_overlapping_files_do_not_duplicate_or_erase(fake_cfg, mseed_factory):
    """Day files overhang midnight, so the same instant arrives twice. The
    block must cover the union once, with no NaN punched into good data."""
    files = [
        mseed_factory("v0.mseed", 0, 3600.0, seed=0),
        mseed_factory("v1.mseed", 3600 - 30, 3600.0, seed=0),
    ]
    joined = _joined(list(iter_blocks(fake_cfg, files)))
    assert len(joined) == int((3600 + 3600 - 30) * fake_cfg.Fs)
    assert not np.isnan(joined).any()


def test_iter_blocks_handles_no_files(fake_cfg):
    assert list(iter_blocks(fake_cfg, [])) == []


def test_gap_report_survives_stitching(fake_cfg, mseed_factory):
    files = [
        mseed_factory("r0.mseed", 0, 3600.0, seed=0),
        mseed_factory("r1.mseed", 3600, 3600.0, seed=1),
    ]
    blocks = list(iter_blocks(fake_cfg, files))
    assert any("r0.mseed" in line for line in blocks[0].gap_report)
    assert any("r1.mseed" in line for line in blocks[0].gap_report)


# ---------------------------------------------------------------------------
# RollingBuffer
# ---------------------------------------------------------------------------

def test_buffer_emits_windows_at_the_configured_step():
    buf = RollingBuffer(["N"], win_size=10, step_size=5)
    starts = [s for s, _ in buf.push(_block(0, 30))]
    assert starts == [0, 5, 10, 15, 20]


def test_buffer_window_contents_are_the_right_slice():
    buf = RollingBuffer(["N"], win_size=4, step_size=2)
    block = Block(0, {"N": np.arange(10, dtype=np.float64)})
    got = [(s, seg["N"].tolist()) for s, seg in buf.push(block)]
    assert got[0] == (0, [0.0, 1.0, 2.0, 3.0])
    assert got[1] == (2, [2.0, 3.0, 4.0, 5.0])
    assert got[-1] == (6, [6.0, 7.0, 8.0, 9.0])


def test_buffer_joins_windows_across_block_boundaries():
    """A window that spans two arrivals is the whole point of the buffer."""
    buf = RollingBuffer(["N"], win_size=10, step_size=10)
    first = Block(0, {"N": np.arange(6, dtype=np.float64)})
    second = Block(6, {"N": np.arange(6, 12, dtype=np.float64)})

    assert list(buf.push(first)) == []
    out = list(buf.push(second))
    assert len(out) == 1
    assert out[0][1]["N"].tolist() == list(range(10))


def test_buffer_restarts_when_a_block_does_not_continue():
    """Blocks either side of an outage must never share a window."""
    buf = RollingBuffer(["N"], win_size=4, step_size=4)
    list(buf.push(Block(0, {"N": np.zeros(6)})))      # leaves 2 samples buffered
    starts = [s for s, _ in buf.push(Block(1000, {"N": np.ones(8)}))]
    assert starts == [1000, 1004], "window grid must restart at the new block"


def test_buffer_memory_stays_bounded():
    """The buffer must not grow with the length of the recording."""
    win, step = 1000, 250
    buf = RollingBuffer(["E", "N", "Z"], win_size=win, step_size=step)
    peak = 0
    index = 0
    for _ in range(200):                      # 200 blocks of 5000 samples
        block = _block(index, 5000, channels=("E", "N", "Z"))
        for _ in buf.push(block):
            peak = max(peak, buf.buffered_samples)
        peak = max(peak, buf.buffered_samples)
        index += 5000
    assert peak <= win + 5000, f"buffer grew to {peak} samples"


def test_buffer_fills_missing_channel_with_nan():
    buf = RollingBuffer(["N", "Z"], win_size=4, step_size=4)
    block = Block(0, {"N": np.ones(4)})       # Z absent entirely
    _, segments = next(iter(buf.push(block)))
    assert segments["N"].tolist() == [1.0] * 4
    assert np.isnan(segments["Z"]).all()


@pytest.mark.parametrize("win,step", [(10, 1), (10, 10), (7, 3)])
def test_buffer_window_count_matches_the_formula(win, step):
    n = 100
    buf = RollingBuffer(["N"], win_size=win, step_size=step)
    count = sum(1 for _ in buf.push(_block(0, n)))
    assert count == (n - win) // step + 1


def test_window_grid_is_reproducible_regardless_of_where_data_starts():
    """Regression: windows used to begin at whichever sample the recording
    opened on, so a station that came online a few seconds early shifted every
    window in the run and two runs of the same event could not be compared."""
    win, step = 10, 5
    a = RollingBuffer(["N"], win_size=win, step_size=step)
    b = RollingBuffer(["N"], win_size=win, step_size=step)

    # Same lattice, two recordings that open at awkward offsets.
    starts_a = [s for s, _ in a.push(_block(1003, 60))]
    starts_b = [s for s, _ in b.push(_block(1007, 60))]

    assert all(s % step == 0 for s in starts_a), starts_a
    assert all(s % step == 0 for s in starts_b), starts_b
    assert starts_a[0] == 1005 and starts_b[0] == 1010


def test_window_grid_survives_a_restart():
    """After an outage the grid must land back on the same lattice."""
    buf = RollingBuffer(["N"], win_size=10, step_size=5)
    list(buf.push(_block(0, 20)))
    starts = [s for s, _ in buf.push(_block(9999, 40))]
    assert all(s % 5 == 0 for s in starts)
    assert starts[0] == 10000


def test_files_are_ordered_by_time_not_by_name(fake_cfg, mseed_factory):
    """Regression: the station names files DDMMYYYY, so sorting by name puts
    01 May before 08 April. The stitcher joins files in the order it gets them,
    so name order would splice the wrong months together and drop most of the
    recording."""
    later = mseed_factory("01052025_000000.mseed", 3600, 3600.0, seed=1)
    earlier = mseed_factory("08042025_000000.mseed", 0, 3600.0, seed=0)

    # Name order puts "01052025" first even though it records an hour later.
    assert sorted([later, earlier], key=lambda p: p.name)[0] == later

    joined = _joined(list(iter_blocks(fake_cfg, [later, earlier])))
    assert len(joined) == int(2 * 3600 * fake_cfg.Fs)
    assert not np.isnan(joined).any()


def test_unreadable_file_is_skipped_not_mis_placed(fake_cfg, mseed_factory,
                                                   capsys):
    good = [mseed_factory(f"ok{i}.mseed", i * 3600, 3600.0, seed=i)
            for i in range(2)]
    bad = fake_cfg.MSEED_INPUT_DIR / "broken.mseed"
    bad.write_bytes(b"not miniseed at all")

    joined = _joined(list(iter_blocks(fake_cfg, good + [bad])))
    assert len(joined) == int(2 * 3600 * fake_cfg.Fs)
    assert "broken.mseed" in capsys.readouterr().out


def test_blocks_do_not_grow_with_the_recording(fake_cfg, mseed_factory):
    """Regression: contiguous files were merged into one ever-growing block,
    so a year of unbroken recording became a single array in memory -- exactly
    what streaming exists to avoid."""
    files = [
        mseed_factory(f"n{i:02d}.mseed", i * 3600, 3600.0, seed=i)
        for i in range(8)
    ]
    blocks = list(iter_blocks(fake_cfg, files))
    one_hour = int(3600 * fake_cfg.Fs)

    assert len(blocks) > 1, "the recording must arrive in pieces"
    assert max(len(b) for b in blocks) <= 2 * one_hour, (
        f"largest block was {max(len(b) for b in blocks)} samples"
    )
    assert len(_joined(blocks)) == 8 * one_hour
