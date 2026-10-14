"""Profile 2: record every LLM call your app makes, with the line of code that made it.

    import trainlab_optimize as trainlab
    trainlab.instrument()                       # once, at startup

    with trainlab.tag(feature="invoice-extract", team="billing"):
        client.chat.completions.create(...)

Then `trainlab import .trainlab/traces` and `trainlab report` give exact cost per line,
per feature and per team. Nothing leaves the machine; no prompt text is stored.

Kill switch: TRAINLAB_INSTRUMENT=0 makes instrument() a no-op.
"""
from __future__ import annotations

import os

from .core import CallEvent, Recorder, current_tags, tag
from .sinks import CallbackSink, JsonlSink, OTelSink

__all__ = ["instrument", "uninstrument", "flush", "tag", "current_tags", "status", "CallEvent",
           "JsonlSink", "CallbackSink", "OTelSink"]

_recorder: Recorder | None = None


def _ref():
    return _recorder


def instrument(path: str | os.PathLike | None = None, *, service: str | None = None,
               project_root: str | os.PathLike | None = None, tags: dict | None = None, otel: bool = False,
               sinks: list | None = None, sample_rate: float = 1.0, include_packages: tuple[str, ...] | list = (),
               sdks: list[str] | None = None) -> Recorder | None:
    """Start recording. Safe to call more than once (later calls replace the settings).

    path              where JSONL traces go (default <project_root>/.trainlab/traces); False = no file
    service           service name stored on each call (default $TRAINLAB_SERVICE / $OTEL_SERVICE_NAME)
    project_root      paths are stored relative to this (default: current directory)
    tags              tags on every call, e.g. {"env": "prod"}
    otel              also emit one OpenTelemetry span per call
    sinks             extra sinks (objects with write(event) and flush())
    sample_rate       record this fraction of calls; reports scale them back up
    include_packages  your own packages if the app runs from site-packages
    sdks              limit to these: openai, anthropic, litellm, bedrock, google-genai
    """
    global _recorder
    if os.environ.get("TRAINLAB_INSTRUMENT", "").strip().lower() in ("0", "false", "off", "no"):
        return None
    root = str(project_root or os.environ.get("TRAINLAB_PROJECT_ROOT") or os.getcwd())
    out = []
    if path is not False:
        out.append(JsonlSink(path or os.path.join(root, ".trainlab", "traces")))
    if otel:
        try:
            out.append(OTelSink())
        except ImportError:
            import warnings
            warnings.warn("trainlab instrument: otel=True but opentelemetry-api is not installed; skipped",
                          RuntimeWarning, stacklevel=2)
    out.extend(sinks or [])
    service = service or os.environ.get("TRAINLAB_SERVICE") or os.environ.get("OTEL_SERVICE_NAME")
    rec = Recorder(root, service, tags or {}, out, sample_rate, include_packages=tuple(include_packages))
    if _recorder is not None:
        _recorder.flush()
    _recorder = rec
    from . import patches
    rec.patched = patches.apply(_ref, sdks)
    return rec


def uninstrument() -> None:
    """Stop recording and restore every patched SDK method."""
    global _recorder
    from . import patches
    patches.remove()
    if _recorder is not None:
        _recorder.flush()
        _recorder.enabled = False
    _recorder = None


def flush() -> None:
    """Write buffered calls now (they are also written every 2 s and at exit)."""
    if _recorder is not None:
        _recorder.flush()


def status() -> dict:
    r = _recorder
    if r is None:
        return {"enabled": False}
    from . import patches
    return {"enabled": r.enabled, "patched": sorted(patches.patched), "on_import": patches.pending(),
            "service": r.service,
            "sample_rate": r.sample_rate, "sinks": [type(s).__name__ for s in r.sinks], **r.counts}


