# Final1000 actual canary fixtures

`final1000_actual_canaries_68216_68219.json.gz.b64` is a deterministic
gzip/base64 snapshot of the four completed, infeasible canary `result.json`
objects read through the scheduler's GET-only remote-file API on 2026-07-21.
The decoded JSON is a list of `{task_id, stage_id, seed, result}` objects.

| Task | Stage | Seed | Result payload seal |
|---:|---|---:|---|
| 68216 | entry-1200-t125 | 2207500000 | `cfb1a6838f8b8e2969d8767bea62e293faa406051cd703342740da64d7513c21` |
| 68217 | bridge-1150-t115 | 2307500000 | `b4667a2f76cb1ffc79dc8c11af31c7a6bcb3875f9221d1482f132863e04c1d7d` |
| 68218 | close-1075-t107p5 | 2407500000 | `9c0b7d440b1040822bb7a891cf086ceef8b063f5da6b5f95604a8e25ae998141` |
| 68219 | final-1000-t100 | 2457500000 | `9fe589e0ce238deae362a61158208238d6c164f5648e94821083053a8ce60e6c` |

All four report `physical_feasible_count=0` and
`feasible_pareto_count=0`. Their compact results omit the three legacy
physical-summary dictionaries while retaining sealed terminal replay
evidence, the complete 320-row terminal physical/optimizer matrices, and the
ordered 20-constraint infeasibility projection.

The decoded fixture SHA-256 is
`b18514ee8758e7d60517323a743ecf528ccc14e282ecabd40c39d0ca9a9cf557`.
