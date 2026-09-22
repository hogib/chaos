"""Optional ``.npz`` cache of preprocessed waveform blocks.

Preprocessing a 31-day run costs about 11 s of wall time on a dozen cores,
roughly 2% of what feature extraction costs, so caching earns its keep only
when you are iterating on feature parameters and re-running repeatedly. It is
therefore off unless ``cache.enabled`` is set.

A cache entry is valid only if both the preprocessing parameters and the source
file are unchanged, so editing ``config.json`` or replacing a raw file
invalidates it without anyone having to remember to clear anything.
"""

import hashlib
import json

import numpy as np


def cache_key(cfg) -> str:
    """Returns a short digest of everything that affects preprocessing.

    Feature-side settings are deliberately excluded: changing the Rosenstein
    window or the window length does not change a single decimated sample, and
    re-deriving them is exactly what the cache exists to avoid.

    Args:
        cfg: Configuration object (see :class:`chaos.pipeline.Settings`).

    Returns:
        A 16-character hex digest.
    """
    from chaos.preprocess import filter_pad_sec

    payload = json.dumps(
        {
            "fs": float(cfg.Fs),
            "freq_min": float(cfg.FREQMIN),
            "freq_max": float(cfg.FREQMAX),
            "gap_threshold_sec": float(cfg.GAP_THRESHOLD),
            "channels": list(cfg.PREPROCESS_CHANNELS),
            "pad_sec": float(filter_pad_sec(cfg)),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _entry_path(cfg, key: str, mseed_file):
    """Returns the cache file path for one source file under one key."""
    return cfg.CACHE_ROOT / key / f"{mseed_file.stem}.npz"


def load_block(cfg, key: str, mseed_file):
    """Reads a cached block, or returns ``None`` if there is no valid one.

    Args:
        cfg: Configuration object.
        key: Digest from :func:`cache_key`.
        mseed_file: The source MSEED path.

    Returns:
        Tuple ``(file_name, start_index, channels, gap_report_lines)``, or
        ``None`` on a miss or a stale entry.
    """
    path = _entry_path(cfg, key, mseed_file)
    if not path.is_file():
        return None
    try:
        stat = mseed_file.stat()
        with np.load(path, allow_pickle=False) as npz:
            if int(npz["src_size"]) != stat.st_size:
                return None
            if abs(float(npz["src_mtime"]) - stat.st_mtime) > 1e-6:
                return None
            channels = {
                name: npz[f"ch_{name}"] for name in cfg.PREPROCESS_CHANNELS
                if f"ch_{name}" in npz
            }
            if set(channels) != set(cfg.PREPROCESS_CHANNELS):
                return None
            return (
                mseed_file.name,
                int(npz["start_index"]),
                channels,
                json.loads(str(npz["gap_report"])),
            )
    except Exception:
        # A truncated or unreadable entry is a miss, never a failure: the
        # caller just does the work again and overwrites it.
        return None


def store_block(cfg, key: str, mseed_file, result) -> None:
    """Writes one preprocessed block to the cache.

    Args:
        cfg: Configuration object.
        key: Digest from :func:`cache_key`.
        mseed_file: The source MSEED path.
        result: Tuple ``(file_name, start_index, channels, gap_report)``.
    """
    _, start_index, channels, gap_report = result
    path = _entry_path(cfg, key, mseed_file)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        stat = mseed_file.stat()
        arrays = {f"ch_{name}": arr for name, arr in channels.items()}
        # np.savez_compressed appends ".npz" unless the name already ends in
        # it, so the temp name has to end there too or the rename below looks
        # for a file numpy never wrote.
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(
            tmp,
            start_index=np.int64(start_index),
            src_size=np.int64(stat.st_size),
            src_mtime=np.float64(stat.st_mtime),
            gap_report=json.dumps(list(gap_report)),
            **arrays,
        )
        tmp.replace(path)
    except Exception as exc:
        # The cache is an optimisation. Losing it must never lose the run.
        print(f"  [WARN] could not cache {mseed_file.name}: {exc}")
