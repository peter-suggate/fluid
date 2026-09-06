# Adaptive-mass fallback and repair audit

Static source audit, 2026-09-07. The inventory below records the source **before D4 removal**; its line numbers are historical audit anchors. `R` below means `lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts`; `H` means its `.ts` host counterpart. Other paths are relative to `lib/methods/adaptive-mass/` unless stated otherwise.

## D4 removal following the audit

Removed all six resident D4 preserve/commit kernels and their dispatches: scalar averaging, velocity/pressure averaging (both after projection and after topology publication), and activity merging. Removed the CPU `preserveD4Symmetry` operation and its sticky authority cache/options. The former symmetry stage is now `scalar-publication`, which only publishes the output receipt. Stage descriptions, profiling consumers and package scripts have been updated. Old frame-control D4 metadata retains its layout for compatibility; the resident advertises no D4 capability and initializes both authorities false. There is no remaining D4 field-mutating kernel to enable.

Deleted three tests dedicated to the deleted averaging/merging kernels. Physical symmetry/property lanes and their thresholds remain unchanged. Added a CPU regression that conditions a symmetric frame, applies an asymmetric edit, then verifies disabled conditioning preserves that edit. It fails against the original conditioning implementation and passes after removal.

Validation: the four-file CPU run initially passed 35/36 tests; its large-CFL tile-resolution failure also reproduces with the original CPU dynamics and conditioning implementations. A subsequent focused run of B4 profile, conditioning controls, phase-1 receipts and trace segmentation passed. TypeScript checking reports errors outside the changed adaptive-mass code; the initially detected stage-catalog mismatch was fixed. `git diff --check` passes. The required full Dawn command was attempted but refused an existing repository-wide WebGPU lease; no GPU correctness/performance result is claimed. A separate Fluid Lab browser tab was also present, so no competing GPU run was launched.

Additional diagnostic concern: H's dense diagnostic pressure view masks values by density (`rho >= 0.5`). It does not expose every raw pressure slot. Use raw state/journal captures when diagnosing membership or pressure errors rather than relying only on this presentation view.

## Main findings

The strongest bug-hunting targets are corrupt incidence ranges becoming empty ranges, missing transport support becoming zero/self-retained transport, nonnegative clamps on newly computed conserved quantities, D4 averaging before diagnostics, and pressure breakdown/restart paths. The fault-reporting machinery itself also needs hardening.

“Repair” is overloaded here. Incremental rebuilding of dirty pressure coefficients is normal maintenance. Gamma/beta conditioning, diffusion, sharpening, and volume correction are explicitly part of the implemented CM12 method. Conversely, eight post-conditioning capacity relays and symmetry averaging can hide the magnitude and origin of an error. Record the original state before any of these operations changes it.

This is a source-level inventory of the identified mechanisms, not proof that every path executes in a particular scene or an exhaustive dynamic coverage claim. Generated variants and sparse absence require call-site contracts; a search for the word `fallback` alone misses the most consequential paths.

## Highest-priority invariant failures

| Location | Existing behavior | Failure to introduce / root-cause investigation |
| --- | --- | --- |
| `sparse-cm12-row-access.wgsl.ts:283`, `boundedIncidenceEnd` | Reversed or oversized incidence ranges become `[begin, begin)`. Callers see a cell with no rows. No fault is recorded here. | Fail on the raw range, preserving cell, begin/end, generation and arena offsets. Investigate stale incidence publication, wrong bank and corrupt addressing. Keep a bounded return after recording failure to prevent runaway GPU loops. |
| R:1941–2003, `compactOwnerCellAt`, `scheduledCompactOwnerCellAt`, `ownerCellAt` | Absent owners, out-of-range local offsets, inactive cells and closed cells converge on `INVALID`. | Distinguish intentional sparse air, physical solid, outside world and broken accepted topology. Fail when an expected represented/open owner is missing. |
| `sparse-cm12-transport-execution-image.wgsl.ts:123–157` | Invalid leaf flags, zero scale, inconsistent geometry/counts and packet overflow yield the same invalid owner/packet sentinels used for empty work. | Fail malformed represented leaves/packets at publication and validate required sample coverage at consumption. Do not classify padding as corruption. |
| R:2020–2035, 2075–2106, 3050–3065 | Missing velocity owners are zero-valued; interpolation skips unsupported corners without reporting lost support. | Capture support weight and query coordinates. Fail required wet characteristic support gaps; investigate frontier residency, extension depth and authored-domain bounds in signed-world execution. |
| R:3110–3199 | Stencil builders zero weights for `INVALID` cells. Later normalization can conceal partial support loss. | Preserve the raw eight-corner ownership/weight stencil and classify every missing positive-weight corner. |
| R:3840–3869 and fused equivalent near 4013 | Forward deficit with `visible<=1e-9` is deposited back into its donor, including gamma and momentum. | Fail empty arrival support for an open transport donor before the self-deposit. Investigate characteristic reach, world bounds, solid classification and ownership generation. |
| R:3915–3921 and fused equivalent 4078–4084 | Negative gathered rho/gamma become zero; dry rho retains prior gamma and zeros velocity. | Assert finite raw rho/gamma/momentum and materially nonnegative rho/gamma before clamps. Record dry-state gamma retention separately; it is an explicit state policy. |
| R:4440–4455, `traceSharpeningMass` | Missing current/candidate owner ends the trace; small gradient and distance/step exhaustion also stop it. | Split termination reasons. Fail missing required ownership; measure unresolved traces separately from valid solid/distance/flat-gradient stops. |
| R:4490, `scatterSharpeningCell` | No valid scatter weight returns the whole removed mass to its source. Partial support is renormalized; integer remainder also returns to source. | Fail unaccounted empty support. Keep exact fixed-point remainder handling distinct from returning an entire failed transfer. |
| R:4515–4538 | Sharpening finalization clamps density/gamma nonnegative and snaps density to capacity within `1/CM12_SPARSE_TRANSPORT_FIXED`. | Assert raw values and receipt arithmetic before mutation; count and sum endpoint snap mass. Test whether repeated snaps create cumulative mass drift. |
| R:4573–4627; H:6593–6647 | Eight capacity-repair rounds redistribute excess into all open neighbors; no-neighbor/zero-share cases retain it. Moved mass is capped at `1073741823` fixed units; final density is clamped. | In hunting mode fail at first excessive pre-repair density and snapshot transport/sharpening provenance. Separately fail integer saturation and residual excess after the last round. More rounds are not a root-cause fix. QA alternate/gather/early-exit variants at R:4636–4768 must obey the same contract. |
| R:4777, 6185, 6274 and their commit kernels | D4 authority averages density/gamma, transformed velocity and pressure; activity uses maxima/unions/minima over symmetry orbits. Missing members are skipped. | For scenes with valid D4 authority, assert the unmodified orbit before averaging/merging, including matching ownership/rungs. Make strict runs stop on the first mismatch. Otherwise post-repair symmetry tests cannot establish symmetry of the numerical method. |
| R:5496, 5748, 5811 | Nonpositive pressure diagonals substitute a zero preconditioned residual. | Assert finite positive diagonals for rows that mathematically require them; explicitly account for isolated/nullspace cases. Investigate membership and coefficient assembly. |
| R:5788–5790, 5849–5856 | Small/nonpositive curvature zeros alpha and marks breakdown; small previous gamma zeros beta; previous alpha is denominator-floored. | Fail nonfinite values immediately. For curvature, first distinguish true convergence from breakdown above tolerance, then capture pressure/operator/residual/direction and dot products. |
| R:5930–5958, 6040–6084; H:6788–6801 | True-residual drift (>16x recursive squared residual) or curvature loss triggers residual replacement and a fresh Jacobi direction. Successful recovery clears breakdown; unsuccessful recovery disables solving. | Fail on attempted recovery in strict hunting runs, after checking true convergence. `scalars[18]` preserves a count but does not itself reject the frame. Split drift and curvature reasons. |
| R:6026–6037; H:8081; `webgpu-adaptive-mass-solver.ts:1321–1339` | Final true residual, drift and recovery count are reported. Diagnostic assignment is not a throw. | Require a finite final true residual satisfying the chosen solve acceptance contract before projected output is accepted. Record iteration-budget exhaustion distinctly. |
| `sparse-cm12-interned-ref-lookup.wgsl.ts:43–50, 62, 74–80` | Bad canonical/template/range values return zero counts or invalid references. | Fail malformed compiled metadata. Keep valid absent adjacency distinct from invalid metadata. The alternate anchor at line 67 is a representation choice, not evidence of repair. |
| `sparse-cm12-effective-transport-velocity.wgsl.ts:39–51` | Out-of-capacity velocity publications silently do nothing. | Fail out-of-capacity IDs for real publication work; allow only documented sentinel/padding calls. |
| R:5113–5135 | Unknown hierarchy levels fall through generated accessors (edge returns zero). | Reject levels outside the compiled hierarchy at their producer; do not silently manufacture a zero coefficient. |

## Other repair, substitution and bounded-work paths

| Location | Mechanism | Treatment |
| --- | --- | --- |
| `lib/core/cm12-numerics.ts:171–207`; R:3784–3918 | Gamma clamp (interior minimum 0.5, maximum 2.5), donor beta column normalization, forward deficit scatter, diffusion, sharpening limiter, capped volume-correction divergence. | These are explicit method operations. Measure activation/magnitude and validate inputs. Making every activation fatal would reject the implemented method rather than identify implementation bugs. |
| R:2114, 2132, 3076 | Characteristic substep count capped at 16. | Fail or explicitly report requested substeps beyond the supported budget; investigate timestep and velocity bounds. |
| R:1878–1895, 1929; trace call sites | Solid clipping uses eight probes and eight bisections; world-coordinate clamps bound traces. | Normal physical boundary treatment needs to remain explicit. Assert coverage/reach assumptions, especially long chords and signed-world vs authored-domain limits. |
| R:3480–3525 | Unsupported native face corners use the collocated face-support interpolant. | Record reason/count/weight. Mixed-resolution seams can legitimately require this path; fail only when native support was promised. |
| R:3670–3774 | Hose injection creates excess then relays mass/momentum along nozzle direction. Missing receiver retains excess; final mass clamps nonnegative and small mass zeros velocity. | Explicit source algorithm, but unresolved required downstream support, negative raw mass and fixed-point overflow should fail. Source mass must have a separate ledger. |
| R:1858–1864, 3887–3893, 4516–4520 | Fully dynamically covered cells preserve mass; other nontransport cells can be reset to rho=0/gamma=1. | Preservation is deliberate. Fail unreceipted removal of wet mass when a cell becomes static-solid/inactive; distinguish transfer and retirement from ordinary dry clears. |
| R:4896–4950, 5420–5440 | Proven planar columns override density-derived ghost-fluid distance; otherwise general theta is used, including a bounded theta and default 1 when phase weights are insufficient. | Shape-dependent algorithm choice. Assert promised phase coverage and finite inputs; proof rejection alone is not an error. |
| R:4955–4980 | `pressureCellSubmerged` keeps a slightly underfilled (<0.5) cell in pressure when incidence neighbors are already pressure members. | A membership stabilization dependent on accepted neighbor membership. Count activations and capture preclassification density/neighbor epoch. Test whether it hides a transport hole or gives history-dependent membership. |
| R:8800–8818, 8896 | Refinement adds a mass-restoring correction, scales negative reconstruction slopes toward the mean, then clamps rho nonnegative. | Conservative prolongation/positivity limiting may be valid. Verify raw reconstructed mass, limited mass and final mass separately; don't let correction conceal wrong overlap volumes or indexing. |
| R:8947–8975, 9086, topology publication | Transfer conservation failures set activity fault flags and reject candidate state. Accepted state can remain visible. | Already a rejection mechanism; strict host handling must surface rejection as a failed frame rather than a visually frozen/retried update. Distinguish deliberate retirement residue. |
| R:7433–7558, 9196–9203 | Recovery rung/lock and refinement floors retain/promote resolution after activity. | Adaptivity policy, not numerical repair by itself. Record cause and transitions; prevent refinement from silently substituting for failed accuracy guarantees. |
| R:2178–3005, 6481–6537, 9903–10410 | Presentation normalizes/clamps occupancy, substitutes air for missing owners, uses general spatial contouring when height/column proofs fail, and global sampling when page caches do not fit. | Keep shape/cache choices explicit. Fail missing data where publication promised coverage. Diagnose physics using raw mass, not bounded visible occupancy. Macro cache fallback is not automatically a correctness failure. |
| `sparse-cm12-transport-execution-image.wgsl.ts:111–119` | Lookup outside staged 3x3x3 directory reads the global directory. | Equivalent slower lookup; count if a performance/support guarantee says it should never occur. |
| `sparse-cm12-transport-home-frame-halo.wgsl.ts:102–104, 159–167, 193–210, 243–257` | Radius/shift caps; global lookup or whole-stencil/global-velocity fallback on unsupported halo coverage/shape. | Separate helper not imported by the current resident generator. Audit if re-enabled; enforce any claimed halo-fit guarantee. |
| `sparse-cm12-velocity-extension.wgsl.ts:84, 118–158, 253–260, 305–360` | Bounded depth/schedule counts, invalid packet masks, skipped unsupported neighbors, extension through available open neighbors. | Extension is intentional. Assert packet counts/depth before caps and required final characteristic support, rather than demanding every air cell extend. |
| `sparse-cm12-height-reconstruction.wgsl.ts:180–185, 210–222` | Tiny curvature stops height CG; final heights clamp and receive a mean-restoring correction. | Currently disabled by `SPARSE_CM12_COMMON_HEIGHT_ENABLED=false` at line 8. If enabled, separate convergence from failure and validate correction/conservation before stamping publication. |

## Repairs that already have explicit fault protocols

`sparse-cm12-pressure-topology-repair.wgsl.ts` validates producer coverage, cell ranges, executed cells and coefficient generation. `sparse-cm12-canonical-membership.wgsl.ts` rebuilds dirty membership leaves. `sparse-cm12-persistent-pressure-cache-aggregate.wgsl.ts` repairs dirty worksets, aggregate/hierarchy edges and diagonals, checking repaired/executed counts. These are expected incremental updates; fail their violated contracts, not the fact that a dirty leaf was rebuilt.

Frame-control, presentation, topology-effects and pressure-cache modules also have explicit fault phases/codes. Confirm every such device-side fault reaches one mandatory host failure path. A zeroed indirect dispatch, rejected candidate, diagnostic field or first-fault record alone does not throw a JavaScript exception.

Several fault writers use a single `atomicCompareExchangeWeak` to claim the initial fault without retry: frame-control:142, frame-plan-presentation:117, persistent-pressure-cache:107, aggregate-cache:220, canonical-membership:214, pressure-topology-repair:57 and topology-effects-authority:46. Harden these before relying on the new assertions: a weak compare/exchange can fail spuriously, leaving the code/owner unclaimed even though another phase word records failure. Use an unconditional sticky fault indication and a correctly retried first-record claim; keep record publication coherent. This is a source-level risk, not a reproduced device failure.

The resident generator already rejects missing required pressure/transport layouts at R:358–376. Optional chaining with `?? 0` later in those required-layout expressions is therefore not by itself a reachable runtime fallback.

## Adjacent implementation

The separate LoSasso adaptive-mass shader is not the Sparse CM12 resident implementation. In `lib/methods/losasso/webgpu-octree-losasso-adaptive-mass.wgsl.ts`, `velocityAtGrid` (107) returns zero on missing owner/node support; `emitOutgoingTransfers` (139) sends an unresolved recipient back to its donor and increments `control[6]`; sharpening trace/plan (174–176) retains donor mass on failed trace or missing weights. Transfer crossing is clamped to one at lines 129/139. These deserve equivalent explicit support/Courant contracts if that method is in scope. Its finite/capacity/overflow error bits are separate from the unresolved-recipient counter.

## Recommended implementation sequence

1. Add a dedicated persistent assertion buffer, independent of reused scratch arenas: sticky fault bits, stage, frame, generation, cell/row/brick, reason, raw operands and attempted correction. Make the host stop on it. Preserve the first failure across later dispatches and prevent failed output from being accepted. GPU workgroup barriers must remain uniform; an assertion must not introduce divergent early returns around barriers.
2. Harden incidence/metadata/ID validity and finite/fixed-point-range checks first. Check both individual conversions and accumulated integer receipts. These should be fatal for all real work, not hidden behind optional diagnostics.
3. Add strict pre-repair gates for capacity redistribution, D4 mutation, empty-stencil self-return and pressure restart. Record ordinary algorithm choices separately. Fail before repair changes the evidence; retain a source snapshot or stage capture.
4. Run a minimal reproduction for each first failure. Existing phase-1 captures and `sharpeningPhaseLimitForQA` (`transport`/transform/finalize/capacity rounds as supported by the host) help isolate transport vs conditioning vs relay. Pressure journals already record seed and encoded iterations. Compare raw conservation, support coverage and symmetry before publication.
5. Fix the producer, add a focused regression for that failure, then run `npm run test:dawn:sparse-cm12` for substantive simulation/topology/publication/boundary/editing changes. Do not run Dawn concurrently with the browser or another Dawn process. Do not increase tolerances, timing limits, relay rounds or hidden refinement floors to make the failure disappear.

Suggested initial order: incidence corruption → missing required transport support → raw negative/nonfinite/fixed-point failures → D4 pre-repair differences → capacity excess provenance → pressure drift/curvature. This exposes structural defects before investigating floating-point behavior.


## Failure handling implemented after this audit

D4 repair removal is committed as `471016cb`. The next implementation adds a
sticky 16-word first-failure record at the tail of the resident topology arena.
Each stage copies its fault count into GPU-owned uniform parameters; uniform
entry guards then suppress later stages, publication, and already-queued later
frames. Host parameter updates cannot clear that flag. A failing stage can have
partial writes: this is a halt with evidence, not rollback. A fresh simulation
must be created to restart. Generation replacement checks the old arena before
transfer and the new arena before adoption, so replacement cannot erase a fault.

Mandatory readback checks run on frame completion, initial publication,
diagnostic reads, and paused/live edits. A failed readback itself halts. The
renderer retains the first failure and closes admission; the UI releases its
GPU, displays **SIMULATION HALTED**, and offers copy/download JSON containing
kernel, frame, generation, owner, named operands, exact raw words, scene inputs,
and method configuration. Nonfinite float operands may become JSON null; their
exact bits remain in `rawWords`. Unknown host-side owner/generation fields are
`-1`, never an invented GPU identity.

The first converted sites are:

- Invalid resident incidence ranges: record `INCIDENCE_RANGE` before returning
  a safe empty range for the remainder of the failing stage.
- Empty forward-deficit support: record `EMPTY_DEFICIT_STENCIL`; donor
  self-return is removed in both packet and direct implementations.
- Empty sharpening support: record `EMPTY_SHARPENING_STENCIL`; donor
  self-return is removed.
- Negative/nonfinite transported density or gamma: record
  `INVALID_CONSERVED_VALUE`; the subsequent nonnegative clamps are removed.
- Unexpected host generation exceptions: record `TOPOLOGY_GENERATION_FAILURE`
  and halt instead of logging and retaining accepted state. Explicit budget
  deferral, stale plans, and cancellation remain normal control flow.

The GPU regression lane `simulation-failure-halt` injects corrupt incidence into
the production validator, verifies healthy work first, and proves later stages
and frames cannot publish or overwrite the original evidence. Decoder/UI tests
cover raw-bit retention, report content, and refusal to restart after failure.

Validation on this working tree: the new Dawn halt lane and live-liquid-injection
lane pass, as do seven focused CPU/UI/manifest tests. The full canonical gate
remains failing: physical D4 symmetry, topology/hydrostatic/mini64 timeouts, and
mini32 at 41.6154 ms against the unchanged 40 ms ceiling. The suite exhausted its
180-second budget before all lanes. An isolated checkout of `471016cb` measured
42.5329 ms in that same mini32 lane, so that performance failure also exists
without this failure-handling implementation. This is not a clean regression
gate or a new performance baseline. Type checking still reports unrelated
existing errors.

Next work remains the density-capacity redistribution passes, pressure restart
and residual repair, and connecting the existing device fault protocols to this
mandatory report. The inventory above is not marked resolved by adding a report
mechanism; violated numerical contracts still need their root causes fixed.
