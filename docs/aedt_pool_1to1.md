# Experimental AEDT attach backend (exclusive 1:1)

The production default in `run_simulation_260706.py` remains `standalone`:
each runner creates and owns its own AEDT Desktop.  The `pooled` backend in
this revision exists only for one disposable one-Desktop/one-project pilot.

It is enabled only when both variables are present:

```bash
export MFT_AEDT_BACKEND=pooled
export MFT_AEDT_EXCLUSIVE_1TO1=1
```

The runner additionally requires `MFT_AEDT_SCHEDULER_URL` and
`MFT_SLURM_SCHEDULER_ROOT`.  It loads the scheduler attach client from that
exact scheduler checkout, requests `exclusive_session=true`, attaches with
`new_desktop=False` and `close_on_exit=False`, and binds the generated MFT
project name to the lease.

In pooled mode the runner never owns Desktop shutdown.  It does not call
`release_desktop`, terminate AEDT descendants, or delete the project workspace
before the scheduler session host acknowledges project close.  Solver timeout
or uncertain solver state is reported as a session quarantine fault.

`--hold` is deliberately unsupported.  Output telemetry adds
`aedt_backend`, `aedt_lease_id`, and `aedt_exclusive_session` so the terminal
artifact proves which path ran.

Passing this 1:1 lifecycle test does not authorize two projects per Desktop or
production pool enablement.  The earlier 1:2 experiment did not produce a
valid terminal sibling artifact and remains a failed isolation gate.
