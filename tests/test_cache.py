"""Unit tests for chaos/cache.py — the optional .npz block cache."""

import numpy as np
import pytest

from chaos.cache import cache_key, load_block, store_block
from chaos.preprocess import preprocess_file
from chaos.recording import iter_blocks


@pytest.fixture
def cached_cfg(fake_cfg):
    fake_cfg.CACHE_ENABLED = True
    return fake_cfg


# ---------------------------------------------------------------------------
# cache_key
# ---------------------------------------------------------------------------

def test_key_is_stable_for_the_same_settings(fake_cfg):
    assert cache_key(fake_cfg) == cache_key(fake_cfg)


@pytest.mark.parametrize("attr,value", [
    ("Fs", 10.0),
    ("FREQMIN", 0.2),
    ("FREQMAX", 1.5),
    ("GAP_THRESHOLD", 5.0),
    ("FILTER_PAD_SEC", 300.0),
])
def test_key_changes_when_preprocessing_changes(fake_cfg, attr, value):
    """Anything that alters a decimated sample must invalidate the cache."""
    before = cache_key(fake_cfg)
    setattr(fake_cfg, attr, value)
    assert cache_key(fake_cfg) != before


def test_key_changes_with_channels(fake_cfg):
    before = cache_key(fake_cfg)
    fake_cfg.PREPROCESS_CHANNELS = ["N"]
    assert cache_key(fake_cfg) != before


def test_key_ignores_feature_settings(fake_cfg):
    """Changing the Rosenstein window does not change a single sample, and
    re-deriving them is exactly what the cache exists to avoid."""
    before = cache_key(fake_cfg)
    fake_cfg.FEATURES["rosenstein"]["slope"] = [0, 3, 6, 12]
    fake_cfg.WIN_SEC = 400.0
    fake_cfg.WARMUP_COUNT = 99
    assert cache_key(fake_cfg) == before


# ---------------------------------------------------------------------------
# store / load
# ---------------------------------------------------------------------------

def test_round_trip_preserves_the_block(cached_cfg, two_hour_file):
    key = cache_key(cached_cfg)
    start_index, channels, report = preprocess_file(cached_cfg, two_hour_file)
    store_block(cached_cfg, key, two_hour_file,
                (two_hour_file.name, start_index, channels, report))

    loaded = load_block(cached_cfg, key, two_hour_file)
    assert loaded is not None
    name, idx, chans, rep = loaded
    assert name == two_hour_file.name
    assert idx == start_index
    assert rep == report
    for ch in channels:
        np.testing.assert_array_equal(chans[ch], channels[ch])


def test_miss_when_nothing_was_stored(cached_cfg, two_hour_file):
    assert load_block(cached_cfg, cache_key(cached_cfg), two_hour_file) is None


def test_miss_when_the_source_file_changed(cached_cfg, two_hour_file,
                                           mseed_factory):
    key = cache_key(cached_cfg)
    result = (two_hour_file.name, *preprocess_file(cached_cfg, two_hour_file))
    store_block(cached_cfg, key, two_hour_file, result)
    assert load_block(cached_cfg, key, two_hour_file) is not None

    # Rewrite the same path with different data.
    mseed_factory(two_hour_file.name, 0, 3600.0, seed=99)
    assert load_block(cached_cfg, key, two_hour_file) is None


def test_miss_when_settings_changed(cached_cfg, two_hour_file):
    result = (two_hour_file.name, *preprocess_file(cached_cfg, two_hour_file))
    store_block(cached_cfg, cache_key(cached_cfg), two_hour_file, result)

    cached_cfg.FREQMIN = 0.5
    assert load_block(cached_cfg, cache_key(cached_cfg), two_hour_file) is None


def test_corrupt_entry_is_a_miss_not_a_crash(cached_cfg, two_hour_file):
    key = cache_key(cached_cfg)
    result = (two_hour_file.name, *preprocess_file(cached_cfg, two_hour_file))
    store_block(cached_cfg, key, two_hour_file, result)

    entry = cached_cfg.CACHE_ROOT / key / f"{two_hour_file.stem}.npz"
    entry.write_bytes(b"not an npz")
    assert load_block(cached_cfg, key, two_hour_file) is None


def test_store_failure_never_breaks_the_run(cached_cfg, two_hour_file,
                                            monkeypatch, capsys):
    """The cache is an optimisation; losing it must not lose the run."""
    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("chaos.cache.np.savez_compressed", boom)
    result = (two_hour_file.name, *preprocess_file(cached_cfg, two_hour_file))
    store_block(cached_cfg, cache_key(cached_cfg), two_hour_file, result)
    assert "could not cache" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# End to end through iter_blocks
# ---------------------------------------------------------------------------

def test_cached_run_matches_uncached_run(fake_cfg, mseed_factory):
    """A cache hit must be indistinguishable from doing the work."""
    files = [
        mseed_factory(f"e{i}.mseed", i * 3600, 3600.0, seed=i)
        for i in range(2)
    ]

    fake_cfg.CACHE_ENABLED = False
    cold = list(iter_blocks(fake_cfg, files))

    fake_cfg.CACHE_ENABLED = True
    warming = list(iter_blocks(fake_cfg, files))   # populates
    warm = list(iter_blocks(fake_cfg, files))      # reads back

    def joined(blocks, channel):
        return np.concatenate([b.data[channel] for b in blocks])

    assert warm[0].start_index == cold[0].start_index
    assert len(warming) == len(warm) == len(cold)
    for ch in cold[0].data:
        np.testing.assert_allclose(joined(warm, ch), joined(cold, ch))


def test_cache_files_land_under_the_key(fake_cfg, mseed_factory):
    fake_cfg.CACHE_ENABLED = True
    files = [mseed_factory("k0.mseed", 0, 3600.0)]
    list(iter_blocks(fake_cfg, files))

    key = cache_key(fake_cfg)
    assert (fake_cfg.CACHE_ROOT / key / "k0.npz").is_file()
