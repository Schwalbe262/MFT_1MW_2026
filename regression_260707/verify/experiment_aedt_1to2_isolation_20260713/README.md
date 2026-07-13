# AEDT one-Desktop / two-project fault-isolation pilot

This is a standalone, fail-closed cluster experiment for PyAEDT 0.22.0 and
AEDT 2025 R2. It is not connected to the production scheduler or b171 campaign.

The pilot requires all of the following before activation can be considered:

- both projects and asynchronous solves belong to one AEDT PID;
- FlexLM shows exactly one local `electronics_desktop` checkout and at least
  two local `elec_solve_maxwell` checkouts during overlap;
- a project-local invalid-setup failure leaves the shared Desktop reachable;
- a unique project-A `3dedy` PID is mapped by phased launch evidence before a
  SIGTERM is sent;
- project B's solver survives, gRPC stays healthy, its identity is unchanged,
  and its convergence artifact is readable.

Any missing or ambiguous evidence produces `activation_allowed=false` and the
recommended policy is quarantine, drain, and requeue on a fresh Desktop.
