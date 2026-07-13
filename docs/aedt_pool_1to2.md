# Experimental shared AEDT 1:2 MFT runner gate

The default backend remains `standalone`. Exclusive pooled 1:1 still requires
`MFT_AEDT_EXCLUSIVE_1TO1=1`. The shared path is accepted only by the separate
disposable acknowledgement:

```bash
export MFT_AEDT_BACKEND=pooled
export MFT_AEDT_SHARED_1TO2_PILOT=1
```

Exactly one of the two acknowledgement variables may be set. The shared pilot
requests `exclusive_session=false`, attaches with `new_desktop=False` and
`close_on_exit=False`, and retains the same host-owned project close protocol.
This variable is deliberately named as a pilot and is not used by production
campaign submission.

For the isolated abort case only,
`MFT_AEDT_PILOT_PRE_SOLVE_READY_FILE` and
`MFT_AEDT_PILOT_PRE_SOLVE_HANG_SECONDS` create a marker and wait immediately
after project creation, before modeling or solve. The orchestrator may then
terminate the client and request a project-local pre-solve close. These hooks
are ignored outside the explicit shared pilot.

The previous solver-PID fault experiment was not a valid isolation proof:
surviving PID/gRPC evidence did not yield a terminal sibling row or field
solution. Mid-solve timeout remains a session quarantine/recycle event, not a
project-local cancel.
