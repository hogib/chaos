"""CHAOS: chaotic-feature extraction for seismic waveforms."""

__all__ = ["main"]


def __getattr__(name):
    # Lazy so that importing a leaf module (chaos.chaos_algorithms, say) does
    # not drag in obspy and the rest of the pipeline via the package __init__.
    if name == "main":
        from chaos.pipeline import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
