"""Clock abstraction so memory writes use externally-injected timestamps.

The dataset under ``data/ChronicCompanion-set/`` carries a ``dialogue_timestamp``
per sample but has no real wall-clock relation to when the experiment runs.
Wherever the memory pipeline used to call :func:`datetime.now`, it now takes
a :class:`Clock` instead so tests can pin "now" to ``dialogue_timestamp`` and
get reproducible window cut-offs / ``updated_at`` fields.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...
    def now_iso(self) -> str: ...


def parse_external_timestamp(ts: str | datetime) -> datetime:
    """Parse a string timestamp into an aware UTC :class:`~datetime.datetime`.

    Accepts either ISO-8601 (``"2025-02-06T13:00:00+00:00"``, ``"...Z"``) or
    the dataset's compact form ``"YYYY-MM-DD HH:MM:SS"``. Naive inputs are
    treated as UTC so the rest of the pipeline can compare them consistently.
    """
    if isinstance(ts, datetime):
        dt = ts
    else:
        text = str(ts).strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            try:
                dt = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
            except ValueError as err:
                raise ValueError(f"Unrecognised timestamp: {ts!r}") from err
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class RealClock:
    """Wraps :func:`datetime.now` (UTC)."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def now_iso(self) -> str:
        return self.now().isoformat()


class FixedClock:
    """Always returns the same externally-injected timestamp.

    Used by the session ingest orchestrator so all "current time" reads under
    a sample (``recent_status`` window cutoff, ``profile.updated_at``,
    ``cluster.updated_at``) collapse to the sample's ``dialogue_timestamp``.
    """

    def __init__(self, ts: str | datetime) -> None:
        self._dt = parse_external_timestamp(ts)

    def now(self) -> datetime:
        return self._dt

    def now_iso(self) -> str:
        return self._dt.isoformat()
