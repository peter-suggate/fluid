# VEX rebuild work reduction

Implemented the first candidate from [the work proof](cm12-small-fluid-work-reduction-proof-2026-09-11.md). Topology rebuilds now clear both retired validity-mask banks during the existing initialization pass, then use the existing accepted-packet list for all eight extension sweeps. The redundant sweep clear and its helper are deleted. Numerical depth, liquid-seed threshold, accepted cells and velocity recurrence are unchanged.

## Scheduling and memory

Initialization retains full direct coverage on rebuilds and the existing compact coverage on stable frames. Sweeps select compact or direct execution using the existing occupancy threshold, independently of rebuild status. Dense images and the guarded empty-list case remain supported.

The schedule grows from eight to eleven words: separate initialization and sweep argument triples plus the cached generation/slot and selection metadata. This adds **12 bytes before alignment**. The existing **24-byte indirect buffer** is reused: initialization at byte 0, sweeps at byte 12, then transport overwrites byte 0 after VEX finishes. The existing argument copy grows from 12 to 24 bytes. There are **no added dispatches, passes, barriers, packet lists or GPU buffers**. Reported total allocation is unchanged at **158,878,160 bytes** in both Figure 7 captures; the extra words fit existing allocation padding.

Changed production files:

- `lib/methods/adaptive-mass/sparse-cm12-velocity-extension.ts`
- `lib/methods/adaptive-mass/sparse-cm12-velocity-extension.wgsl.ts`
- `lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts`

## Verified work reduction

For the Figure 7 radius-0.1 reset, the proof predicts **33,696 → 6,048 VEX workgroups**, including initialization: **82.05% fewer**, removing **1,769,472 lane invocations**.

The actual terminal Dawn receipt at advance 32 verifies the new scheduling on evolved topology:

| Receipt | Control | Changed |
|---|---:|---:|
| Topology generation | 20 | 20 |
| Schedule rebuilt | yes | yes |
| Direct packet domain | 3,744 | 3,744 |
| Accepted packets | 232 | 232 |
| Initialization workgroups | 3,744 | 3,744 |
| Workgroups per sweep | 3,744 | 232 |
| Nine-pass total workgroups | 33,696 | 5,600 |
| Valid cells | 2,328 | 2,328 |
| Fault count | 0 | 0 |

That frame removes **28,096 workgroups / 1,798,144 lane invocations**. The control’s initialization count follows from its single shared indirect record; the changed receipt reports initialization and sweeps separately.

## Validation

The new Dawn fixture runs the production schedule and initializer through clipped bootstrap, stable compaction, retirement and rerung, recycled dense occupancy, empty rebuild/stable frames and slot-only changes. It starts mask banks with stale bits and checks both retired banks, partial-packet masks, initial velocity/depth values and separate indirect argument counts. This and the existing layout/large-dispatch schedule tests pass (four tests total).

The Figure 7 A/B uses an isolated pre-change source snapshot, radius 0.1 m, scene timestep, B8/P8, tolerance 0.194, eight warmup and 24 measured frames. Dawn runs serially with the browser simulation unloaded. Whole-world density and gamma hashes match in the first pair; no WebGPU errors occurred. The initial control was captured from the pre-change source snapshot. Separate frozen repeat snapshots were prepared with concurrent renderer edits synchronized across both arms, but the shared GPU lease prevented that repeat.

Initial pair: **13.6970 → 12.6484 ms** median advance. The sweep substage median is **0.6554 ms in both runs**, so the total difference cannot yet be attributed confidently to this change. Timestamp quantization is approximately 0.065536 ms. The reverse-order repeat could not run because another Dawn test, followed by a browser verification, acquired the shared GPU lease. No elapsed-time speedup is claimed from this single pair. The final canonical regression gate is recorded below.

Artifacts under `artifacts/cm12-figure-7-radius-01/`: `vex-rebuild-control.json`, `vex-rebuild-compact.json`, and `vex-rebuild.patch`. Existing unrelated repository-wide TypeScript errors remain; no errors were reported in the changed files.

The final canonical gate passed **all 17 lanes** in **271.5 seconds**, with unchanged thresholds. Mini32: **28.5082 ms** (40 ms ceiling); mini64: **85.2623 ms** (110 ms ceiling). Receipt: `artifacts/cm12-figure-7-radius-01/vex-rebuild-regression.json`.

Completion: production work reduction is implemented and regression-gated. The original browser tab was closed externally while another browser GPU verification was active; it was not recreated or overwritten.
