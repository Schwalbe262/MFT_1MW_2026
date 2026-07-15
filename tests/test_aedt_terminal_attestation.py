import pytest

from module.aedt_terminal_attestation import (
    advance_scoped_message_cursor,
    capture_scoped_message_cursor,
)


class _Desktop:
    def __init__(self):
        self.phase = 0
        self.calls = []

    def GetMessages(self, project, design, severity):
        self.calls.append((project, design, severity))
        if (project, design) != ("own-project", "own-design"):
            # A Desktop-global/sibling Normal must never be queried or accepted.
            return ["Normal completion of simulation on server: sibling"]
        if severity == 2:
            return [] if self.phase < 2 else ["Fatal solver process terminated"]
        baseline = ["Normal completion of simulation on server: stale"]
        if self.phase >= 1:
            baseline.append("Normal completion of simulation on server: own-new")
        return baseline


def test_exact_scope_does_not_accept_stale_global_or_sibling_normal():
    desktop = _Desktop()
    cursor = capture_scoped_message_cursor(
        desktop, "own-project", "own-design"
    )

    unchanged = advance_scoped_message_cursor(desktop, cursor)
    assert unchanged.normal_completion is False
    assert unchanged.new_messages == ()

    desktop.phase = 1
    completed = advance_scoped_message_cursor(desktop, unchanged.cursor)
    assert completed.normal_completion is True
    assert completed.new_messages == (
        "Normal completion of simulation on server: own-new",
    )
    assert all(
        call[:2] == ("own-project", "own-design") for call in desktop.calls
    )


def test_new_exact_design_error_is_fatal_even_after_normal():
    desktop = _Desktop()
    cursor = capture_scoped_message_cursor(
        desktop, "own-project", "own-design"
    )
    desktop.phase = 2

    update = advance_scoped_message_cursor(desktop, cursor)

    assert update.normal_completion is True
    assert update.fatal_messages == ("Fatal solver process terminated",)


def test_error_severity_normal_is_not_downgraded_to_success():
    """An error-channel Normal can be fallout from a contaminated Desktop.

    AEDT emitted this exact severity oddity in q16 only after a sibling's
    asynchronous script macro lost its active project and was aborted.  The
    text still records Normal syntax, but its error-channel provenance must
    keep the whole exact scope fail-closed rather than manufacture a valid
    terminal result.
    """

    line = "[error] Normal completion of simulation on server: shared-node"

    class ErrorNormalDesktop:
        phase = 0

        def GetMessages(self, _project, _design, severity):
            if self.phase == 0:
                return []
            return [line] if severity in (0, 2) else []

    desktop = ErrorNormalDesktop()
    cursor = capture_scoped_message_cursor(desktop, "own-project", "own-design")
    desktop.phase = 1

    update = advance_scoped_message_cursor(desktop, cursor)

    assert update.normal_completion is True
    assert update.new_errors == (line,)
    assert update.fatal_messages == (line,)


def test_message_history_reset_without_cursor_overlap_fails_closed():
    class ResetDesktop:
        value = ["before"]

        def GetMessages(self, _project, _design, severity):
            return [] if severity == 2 else list(self.value)

    desktop = ResetDesktop()
    cursor = capture_scoped_message_cursor(desktop, "p", "d")
    desktop.value = ["Normal completion of simulation on server: ambiguous"]

    with pytest.raises(RuntimeError, match="without a safe cursor overlap"):
        advance_scoped_message_cursor(desktop, cursor)
