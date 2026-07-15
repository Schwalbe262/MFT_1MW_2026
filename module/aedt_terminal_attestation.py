"""Project-scoped terminal evidence for nonblocking pooled AEDT solves.

AEDT 2025.2 exposes only Desktop-wide running-state APIs.  Those APIs are not
meaningful when several lease-owned projects solve in one Desktop, so pooled
clients use an exact project/design message cursor and a stage-specific result
artifact instead.  This module contains only the message half of that contract;
the solver modules validate their native convergence artifacts separately.
"""

from dataclasses import dataclass
from typing import Any


_NORMAL_COMPLETION = "normal completion of simulation on server"
_FATAL_MESSAGE_MARKERS = (
    "fatal",
    "simulation aborted",
    "simulation failed",
    "solver process terminated",
    "failed to solve",
)


@dataclass(frozen=True)
class ScopedMessageCursor:
    """One immutable cursor for an exact AEDT project and design."""

    project: str
    design: str
    messages: tuple[str, ...]
    errors: tuple[str, ...]


@dataclass(frozen=True)
class ScopedMessageUpdate:
    """New exact-scope messages observed after a cursor."""

    cursor: ScopedMessageCursor
    new_messages: tuple[str, ...]
    new_errors: tuple[str, ...]
    normal_completion: bool
    fatal_messages: tuple[str, ...]


def _scope(project: Any, design: Any) -> tuple[str, str]:
    project_name = str(project or "").strip()
    design_name = str(design or "").strip()
    if not project_name or not design_name:
        raise RuntimeError("AEDT terminal-message project/design scope is empty")
    return project_name, design_name


def _messages(desktop: Any, project: str, design: str, severity: int) -> tuple[str, ...]:
    getter = getattr(desktop, "GetMessages", None)
    if not callable(getter):
        raise RuntimeError("native AEDT Desktop has no GetMessages API")
    values = getter(project, design, int(severity))
    return tuple(str(value) for value in (values or []))


def capture_scoped_message_cursor(
    desktop: Any, project: Any, design: Any,
) -> ScopedMessageCursor:
    """Capture info/all and error histories without using global message scope."""

    project_name, design_name = _scope(project, design)
    return ScopedMessageCursor(
        project=project_name,
        design=design_name,
        messages=_messages(desktop, project_name, design_name, 0),
        errors=_messages(desktop, project_name, design_name, 2),
    )


def _new_suffix(
    previous: tuple[str, ...], current: tuple[str, ...], *, label: str,
) -> tuple[str, ...]:
    """Return only safely attributable appended entries.

    AEDT can trim the front of its message history.  A suffix/prefix overlap is
    safe in that case.  A reset with no overlap is ambiguous and must never turn
    an old Normal-completion message into evidence for the current dispatch.
    """

    if not previous:
        return current
    if len(current) >= len(previous) and current[:len(previous)] == previous:
        return current[len(previous):]
    for overlap in range(min(len(previous), len(current)), 0, -1):
        if previous[-overlap:] == current[:overlap]:
            return current[overlap:]
    if current == previous:
        return ()
    raise RuntimeError(
        f"exact AEDT {label} history changed without a safe cursor overlap"
    )


def advance_scoped_message_cursor(
    desktop: Any, cursor: ScopedMessageCursor,
) -> ScopedMessageUpdate:
    """Advance one exact-scope cursor and classify only newly appended entries."""

    project_name, design_name = _scope(cursor.project, cursor.design)
    current_messages = _messages(desktop, project_name, design_name, 0)
    current_errors = _messages(desktop, project_name, design_name, 2)
    new_messages = _new_suffix(
        cursor.messages, current_messages, label="message"
    )
    new_errors = _new_suffix(cursor.errors, current_errors, label="error-message")

    fatal = list(new_errors)
    error_set = set(new_errors)
    for message in new_messages:
        normalized = message.casefold()
        if message not in error_set and any(
            marker in normalized for marker in _FATAL_MESSAGE_MARKERS
        ):
            fatal.append(message)
    normal = any(
        _NORMAL_COMPLETION in message.casefold() for message in new_messages
    )
    next_cursor = ScopedMessageCursor(
        project=project_name,
        design=design_name,
        messages=current_messages,
        errors=current_errors,
    )
    return ScopedMessageUpdate(
        cursor=next_cursor,
        new_messages=new_messages,
        new_errors=new_errors,
        normal_completion=normal,
        fatal_messages=tuple(fatal),
    )
