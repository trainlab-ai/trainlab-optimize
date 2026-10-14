"""TrainLab Optimize: find and prove LLM cost savings in your codebase."""

__version__ = "0.2.2"
SCHEMA_VERSION = "1"

_INSTRUMENT_API = {"instrument", "uninstrument", "flush", "tag", "current_tags", "status"}


def __getattr__(name):
    """`trainlab_optimize.instrument()` etc. without importing the runtime unless used."""
    if name in _INSTRUMENT_API:
        import importlib
        return getattr(importlib.import_module(".runtime", __name__), name)
    raise AttributeError(f"module 'trainlab_optimize' has no attribute {name!r}")
