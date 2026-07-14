# Bounded shared AEDT canary

Standalone remains the default. A canary task must opt in explicitly:

```bash
export MFT_AEDT_BACKEND=pooled
export MFT_AEDT_SHARED_CANARY=1
```

Exactly one pooled acknowledgement may be set. The canary acknowledgement
requests a non-exclusive shared lease but does not enable the disposable
pre-solve marker/hang hooks. It is intended only for scheduler-controlled
tasks while the AEDT pool is in explicit canary mode with small session and
project limits. The current validated bound is two projects per AEDT. The
acknowledgement is intentionally not named after that bound so a later,
separately validated N does not require a new runner protocol.

The task must also receive `MFT_AEDT_SCHEDULER_URL` and
`MFT_SLURM_SCHEDULER_ROOT`. The runner attaches with `new_desktop=False` and
`close_on_exit=False`; it closes its project through the lease protocol and
never owns Desktop shutdown. A possible solve timeout quarantines and recycles
the whole pooled Desktop after sibling grace.

Attachment is routed through the central
`slurm_scheduler.aedt_attach_client` from the configured scheduler root. No
node-local discovery file, host lookup, or local port handoff is required.

Do not set this acknowledgement on the existing standalone campaign. Do not
ramp beyond the current canary limits until terminal data, license return,
timeout handling, and per-stage runtime are reviewed.
