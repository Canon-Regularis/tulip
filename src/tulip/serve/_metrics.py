"""Dependency-free, in-process metrics for the serving app.

The HTTP service must be observable in production, but pulling in
``prometheus_client`` would add a hard dependency to an optional extra whose
only need is a handful of counters. This module implements just enough of the
`Prometheus text exposition format
<https://prometheus.io/docs/instrumenting/exposition_formats/>`_ to be scraped:
a request counter labelled by ``(method, path, status)`` and a per-path latency
summary (``_sum`` + ``_count``).

The registry is thread-safe (a single lock guards all mutation and snapshotting)
because a WSGI/ASGI server may service requests on multiple worker threads, and
a torn read of the counter maps would corrupt the exposition output.
"""

from __future__ import annotations

import threading

#: Metric names, kept as constants so the middleware and the renderer never drift.
_COUNTER_NAME = "tulip_requests_total"
_DURATION_NAME = "tulip_request_duration_ms"

#: Exposition ``Content-Type`` for the Prometheus text format (version 0.0.4).
CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

#: Path label every series collapses into once the cardinality ceiling is hit.
_CAPPED_PATH = "<capped>"

#: Defense-in-depth ceiling on distinct label tuples. The middleware already
#: labels by bounded route templates, so this is never reached in normal use; it
#: caps memory should a future label source (a new path parameter) ever leak an
#: unbounded value into the registry, turning a would-be OOM into a lost detail.
_DEFAULT_MAX_SERIES = 1024


def _escape_label_value(value: str) -> str:
    """Escape a Prometheus label value (backslash, double-quote, newline).

    Path templates and methods are effectively a closed vocabulary here, but
    escaping is cheap insurance against a future route whose template contains a
    character that would otherwise produce malformed exposition text.
    """
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class MetricsRegistry:
    """Thread-safe tally of request counts and per-path latency.

    One instance is created per :func:`~tulip.serve.app.create_app` call and
    closed over by the observability middleware and the ``/metrics`` endpoint,
    so metrics are naturally scoped to a single application rather than leaking
    across apps (which matters for isolated tests).
    """

    def __init__(self, *, max_series: int = _DEFAULT_MAX_SERIES) -> None:
        self._lock = threading.Lock()
        self._max_series = max_series
        self._request_counts: dict[tuple[str, str, int], int] = {}
        self._duration_sum_ms: dict[str, float] = {}
        self._duration_count: dict[str, int] = {}

    def observe(self, *, method: str, path: str, status: int, duration_ms: float) -> None:
        """Record one completed request.

        Args:
            method: HTTP method (``GET``, ``POST``, ...), normalised by the caller
                to a bounded set so an arbitrary extension method cannot explode
                cardinality.
            path: The matched route *template* (e.g. ``/predict/text``), never the
                concrete URL, so path parameters cannot explode cardinality.
            status: Final HTTP status code of the response.
            duration_ms: Wall-clock handler duration in milliseconds.
        """
        with self._lock:
            key = (method, path, status)
            # Once the ceiling is reached, a genuinely new label tuple collapses
            # into a single ``<capped>`` bucket rather than growing the maps
            # without bound (see _DEFAULT_MAX_SERIES).
            if key not in self._request_counts and len(self._request_counts) >= self._max_series:
                path = _CAPPED_PATH
                key = (method, path, status)
            self._request_counts[key] = self._request_counts.get(key, 0) + 1
            self._duration_sum_ms[path] = self._duration_sum_ms.get(path, 0.0) + duration_ms
            self._duration_count[path] = self._duration_count.get(path, 0) + 1

    def render(self) -> str:
        """Serialise the current tallies as Prometheus text exposition format.

        A consistent snapshot is taken under the lock and formatted outside it,
        so rendering never blocks concurrent request accounting for longer than
        a dict copy. ``# HELP``/``# TYPE`` headers are always emitted (even with
        no samples) so a scraper sees well-formed, self-describing output.
        """
        with self._lock:
            counts = dict(self._request_counts)
            duration_sum = dict(self._duration_sum_ms)
            duration_count = dict(self._duration_count)

        lines: list[str] = [
            f"# HELP {_COUNTER_NAME} Total HTTP requests handled, by method, path, and status.",
            f"# TYPE {_COUNTER_NAME} counter",
        ]
        for (method, path, status), value in sorted(counts.items()):
            labels = (
                f'method="{_escape_label_value(method)}",'
                f'path="{_escape_label_value(path)}",'
                f'status="{status}"'
            )
            lines.append(f"{_COUNTER_NAME}{{{labels}}} {value}")

        lines.append(
            f"# HELP {_DURATION_NAME} Request handler duration in milliseconds, summed per path."
        )
        lines.append(f"# TYPE {_DURATION_NAME} summary")
        for path in sorted(duration_sum):
            path_label = f'path="{_escape_label_value(path)}"'
            lines.append(f"{_DURATION_NAME}_sum{{{path_label}}} {duration_sum[path]:.6f}")
            lines.append(f"{_DURATION_NAME}_count{{{path_label}}} {duration_count[path]}")

        return "\n".join(lines) + "\n"


__all__ = ["CONTENT_TYPE", "MetricsRegistry"]
