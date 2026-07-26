# Octree regression-attribution gate

The Phase 0/9 evidence path is a fail-closed JSON artifact plus an independent
baseline comparator. It covers four fixed lanes:

| Lane | Contract | Purpose |
| --- | --- | --- |
| `mini` | minimal power dam, exactly 500 steps and 2.0 simulated seconds | end-to-end numerical and performance acceptance |
| `ui` | UI dam, exactly 62 steps and 0.496 simulated seconds | interactive workload |
| `quiescent` | 60-step window after a 500-step settle prefix | exposes fixed recurring cost |
| `moving-interface` | minimal power dam, 62 steps and 0.248 simulated seconds | two-level topology/interface work |

Capture a lane with `npm run capture:octree-regression-<lane>`. Capture writes
to `artifacts/octree-regression/<lane>.json`. Each artifact contains exact
physics stage time, dispatch count by owning stage, active/scheduled lane
ratios, authoritative bytes by owner, residual, volume drift, projection
energy ratio, topology epoch, and content hashes for production host source and
WGSL. The projection-energy ratio remains sourced from the stability envelope;
compact reconstructed checkpoints independently publish the maximum mechanical
energy-loss ratio, preserving a dissipation gate after deletion of the old
energy-ledger module. Wall time and the lane completion contract are recorded
alongside them.

Missing evidence is represented as `null` and as a named `blockers` entry. This
includes an initialized projection-energy value with no paired pre/post sample,
or compact dissipation checkpoints with incomplete row/liquid reconstruction. A
capture still writes the artifact for diagnosis, then exits non-zero. In
particular, a runtime work-accounting snapshot with any `null` scheduled/active
counter does not become zero or a guessed utilization ratio.

Compare against an explicit baseline:

```sh
npm run compare:octree-regression -- \
  --baseline=/absolute/path/to/baseline.json \
  --candidate=/absolute/path/to/candidate.json
```

When `--baseline` is omitted, the comparator loads
`docs/baselines/octree-regression/<lane>.json`. Baselines should only be checked
in after all blockers are eliminated; no placeholder baseline is accepted.
The default comparator rejects a stage-time or wall regression above 5%, any
dispatch or authority-memory increase, an active/scheduled ratio decrease over
5%, and bounded residual/volume/energy degradation. It reports the first
regressed stage/metric in stable execution order, followed by every additional
failure and both content revisions. Missing metrics, mismatched stage keys, an
incomplete lane, or a mini artifact other than the exact 500-step/2-second run
fail closed.
