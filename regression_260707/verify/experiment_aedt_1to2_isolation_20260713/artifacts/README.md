# Cluster pilot evidence

Primary fault injection: Slurm job `732549`, account `harry261`, node `n040`,
PyAEDT 0.22.0, AEDT 2025 R2.

Fresh-Desktop healthy-sibling reopen: Slurm job `732554`, submitted with
`--priority=TOP`; Slurm read back priority `4294210729`. It completed with exit
code 0.

Final audit result: **REJECT**. The sibling solver PID survived the targeted
fault, but it did not produce a valid terminal solution. The fresh-Desktop
convergence export says `Completed: N/A`, has zero pass rows, and the saved
results contain no field-solution payload. A `LastAdaptive` name and geometry
quantities alone are not accepted as solve evidence.

Survival of the sibling solver PID is a necessary process-isolation condition,
not a sufficient solution-validity condition. The original automated reopen
probe treated file existence and `LastAdaptive` metadata as a pass; the
corrected audit explicitly rejects that false positive.

The only safe operational response remains: quarantine the shared Desktop as
soon as one child faults, stop admitting work, drain/recycle the Desktop and
allocation, and requeue both the failed task and any sibling without a fully
attested terminal result. Reusing the faulted Desktop is forbidden.

No production scheduler pool or b171 task was modified or enabled.
