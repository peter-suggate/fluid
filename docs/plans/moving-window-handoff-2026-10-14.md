# Moving window hand-over — 14 October 2026

## Status: solver integration landed; fluid acceptance blocked

The running solver now decides and applies F at page commit. The open-ground
probe changed F to `(29,0,0)` and ran 120 steps with no fatal. This is real
solver integration, replacing the earlier primitive-only hand-over.

**This is not a passing moving-window feature yet.** The tank A/B shifted at
step 10 with unchanged liquid/particle totals at that boundary, then its
forced run failed at step 19 with pressure nonfinite. Its control completed
30 steps. The travel fixture also failed to establish liquid crossing the
old window; its initial velocity property was not consumed by this solver.
The fixture now uses the existing QA velocity upload, but that correction
has not been run. The supplied rules allow the full A/B and travel checks at
most once each; both were used. No further gate was run.

Shared interfaces were re-read, including the phase-3 page record, directory,
live frame binding, and phase-2 open-world terrain. Existing changes were
preserved through exact-match edits. All work was serial in this tree, with
no commits, worktrees, index mutations, unit tests or browser runs.

## What landed and exact ownership

Every existing-file edit is journalled as an **incremental `diff -U0`** in
`/private/tmp/claude-501/handoff/moving-window-owned.patch`. It records exact
before/after hunks as applied, including later corrections; it is not a
combined working-tree diff (which would include other sessions). The helper
used is `/private/tmp/claude-501/handoff/window-edit.py`. This document is
also entirely this handoff's. Entire new files owned here are:

- `lib/methods/uniform/uniform-moving-window.ts`: decision twin/WGSL and standalone primitive oracle wrapper.
- `lib/methods/uniform/uniform-window-address.ts`: shared runtime header and signed world classification/address WGSL.
- `tools/check-window-shift-twin.ts`: host arithmetic and real signed pool-record oracle.
- `tools/check-window-shift-decision.ts`: standalone device decision oracle.
- `tools/check-window-shift.ts`: actual solver forced-shift A/B, smoke and travel tools.

Existing files, grouped by their changes (paths under `lib/methods/uniform/`
unless explicitly prefixed):

| Files | Owned changes |
| --- | --- |
| `uniform-page-pool.ts` | Runtime header, gather bounds in requests and resident walk, signed world classification, one tiny decision launch before claims, mirrored F, GPU params updates, force command, lattice publication and directory header |
| `uniform-lattice-address.ts` | Header included in block-word/neighbour offsets, scene world policy |
| `uniform-mixed-frame.ts` | Pass live params to pool initialization |
| `webgpu-uniform-reference.ts` | GPU-owned frame/tuning lanes after seed; step-based force hook; world drop requests; live diagnostic frame; analytic terrain binding |
| `webgpu-uniform-reference.wgsl.ts` | Runtime address helper, world mask, translated cached terrain |
| `uniform-mixed-topology.wgsl.ts` | Runtime address helper, including delegated topology frame |
| `uniform-mixed-layout-builder.ts` | Runtime address helper; authored region comparisons in initial-world tiles |
| `uniform-mixed-dynamic.ts` | World requests and host tile planes |
| `uniform-mixed-residency.ts` | Requests past the old window in signed world pages; closure identity checked in world pages |
| `uniform-narrow-band-flip.ts` | Runtime refinement addressing, particle frame stamp and rebase folded into advect, live particle renderer frame |
| `uniform-narrow-band-order.ts` | Runtime address helper |
| `uniform-mixed-root-owners.ts` | Runtime address helper |
| `uniform-mixed-pressure-cycles.ts` | Runtime directory and block decoding |
| `uniform-block-neighbours.ts` | Runtime neighbour lookup and translated block pages |
| `webgpu-uniform-velocity-extrapolation.wgsl.ts` | Runtime block decoding |
| `uniform-detail-fields.ts` | Runtime addressing in seam kernels |
| `uniform-mixed-presentation.wgsl.ts` | Runtime addressing |
| `uniform-mixed-solid.wgsl.ts` | Runtime record addressing; translated static mask and cached terrain; analytic terrain beyond its initial cache; four height taps in a loop |
| `uniform-open-world-terrain.ts` | Shared analytic height function and exposed immutable sampling params; enabled marker |
| `uniform-solid-boundary.wgsl.ts` | Procedural hash queried in initial-world cells |
| `uniform-particle-contact.wgsl.ts` | Share the translated continuous terrain surface function |
| `lib/core/solid-occupancy-mask.ts` | Optional moving-frame lookup, preserving the default caller contract |
| `lib/core/domain-pages.wgsl.ts` | Signed-world directory lookup and classification with lattice publication |
| `lib/core/method-view-records.ts` | Document directory header and lattice page publication contract |
| `lib/core/webgpu-particle-overlay.ts` | Optional live GPU frame binding and world-space vertex conversion |
| `lib/core/model.ts`, `lib/core/scene-lattice-dimensions.ts` | Probe-only initial box placement override, off by default |

## Decisions implemented

- F is signed i32 pages relative to the immutable initial scene frame.
  World metres = initial origin + h * (32 F + lattice cells). The phase-2
  initial placement is part of that initial frame, not a second runtime offset.
- Directory slots remain world page modulo 64; blocks and texture data never
  move. Full signed xyz stay in allocator records. Packed block words store
  world residues; shared readers return `(residue-F)&63` lattice coordinates.
- Every directory mirror now includes 32 words before its block words:
  F 0..2, revision 3, delta 4..6, sticky edge fatal 7, boundary counter 8,
  world policy 12, initial box low 13..15, box extent 16..18, force step 20,
  force delta 21..23. `UNIFORM_PAGE_TABLE_WORDS` includes this header.
- Policies: 0 is a standalone explicit table, 1 a finite tank, 2 the existing
  open-top column, 3 an open ground plane. SOLID is classified in world
  space; old modulo SOLID sentinels do not wall off unrelated world pages.
- Gather every old resident, including idle ones, and every non-solid
  request/pin/host command. At most 60 pages per axis fit with the preferred
  two-page margin. The named edge fatal handles an over-wide extent, alias,
  invalid forced shift or signed overflow; it is copied into the pool receipt.
- Commit order: host commands/resident bounds -> one-invocation decision ->
  claims/retirement -> publication -> residency/neighbours/new blocks.
  No data-moving pass, blocking advance readback or runtime pipeline build.
- A finite tank needs no automatic movement. Open ground keeps y unchanged
  while the union rests on the floor and fits below the upper margin; a
  blocked low edge alone should not recenter it. Other axes recenter normally.
- The GPU alone updates params.frame.xyz and params.tuning.xyz after initial
  upload. Host step writes skip those lanes. Sources, bodies and renderers
  use that live frame. Immutable host world authoring/mask coordinates still
  use the initial frame; readStats copies its matching live origin.
- Particle state grew from 48 to 64 bytes. Words 12..14 stamp the frame in
  which current particles were written. Existing advect subtracts 32 times
  the difference; commitCount stamps the new frame, including bootstrap.
- Published block xyz and bounds are **current lattice pages**, matched to
  the live GPU frame; private allocator records remain signed world pages.
  The overlay's copied directory now includes the runtime header. Phase 3's
  existing frame bindings and extraction interfaces remain in place.
- Terrain columns remain cached in the initial frame. Mixed solid sampling
  uses shared analytic heights beyond that cache, with no extra bake/pass.
  Native legacy terrain sampling was translated but still clamps the cache.
- `probeWindowShift(step, delta)` is off by default, consumed exactly once
  even though the first numerical step has a kick and a normal commit. The
  A/B tool uses step 10 and delta `[-2,-1,-1]`, which fits a tank at page zero.
- Large-world float precision is out of scope. Rebased particle float low
  bits may differ, but the observed pressure failure is not accepted as noise.

## Integration audit: persistent coordinates and derived coordinates

This is the original coordinate audit used for the integration above. The
listed treatments have been implemented where noted in the ownership table;
this is not evidence of whole-solver invariance. The failed fluid gate means
the audit still needs a correctness follow-through, especially cached state
and zero-coordinate boundary assumptions.

| Storage / derivation | Files / symbols | Required treatment |
| --- | --- | --- |
| Scene frame, authoring box placement, world-to-cell | `lib/core/scene-lattice-dimensions.ts`: `sceneFluidFrame`, `sceneFluidBoxPage`, `fluidFrameCell` | Keep immutable initial origin and h; derive live origin from GPU F. Host mirror is diagnostic/off-advance only. |
| Full signed page identity, requests, pins, allocator tags | `uniform-page-pool.ts`: records, `upPageOf`, `upRequest`, `upEntry`, `upClaim`, `upSlots`, host command/pin arrays | Keep world coordinates. Include every kept/idle resident and non-solid request in bounds. Do not feed lattice q to a world-keyed allocator. Propagate edge fatal through `upFailWith`, retaining first-page detail. |
| Packed block words and directory slots | `uniform-lattice-address.ts`: `umBlockPage`, `umTileCoord`, `umCellCoord`, `umVertexCoord`; `uniform-page-pool.ts`: `upWindowPage` | Decode world residue minus F; lookup `(lattice page + F) & 63`. Data keys stay block-local and storage never moves. |
| Direct page readers and mirrors | `webgpu-uniform-reference.wgsl.ts`, `uniform-mixed-topology.wgsl.ts`, `uniform-mixed-layout-builder.ts`, `uniform-narrow-band-flip.ts` refine module, `uniform-narrow-band-order.ts`, `uniform-mixed-solid.wgsl.ts`, `uniform-mixed-presentation.wgsl.ts`, `uniform-mixed-root-owners.ts`, `uniform-detail-fields.ts` seam | Every independently declared `umPageWord`/`umBlockWord` must share the same GPU frame. Updating only the common address text is insufficient. |
| Mirror allocation/copies and neighbour rows | `uniform-mixed-ownership.ts`, `uniform-mixed-topology.wgsl.ts`, `uniform-block-neighbours.ts`, `uniform-mixed-solid.wgsl.ts`: `pagedTopologyWords`, page-pool tail | If appending F to mirror tails, update all length/offset calculations and copy/adopt paths together. Neighbour/wall rows already follow the packed words; do not overwrite them with a new header. Separate prebound F is an alternative. |
| Static AIR/SOLID classifications | `uniform-lattice-address.ts`: outside callback in `uniformLatticeAddress`; `uniform-page-pool.ts`: `upKind`, `upWord`, `upRequest` | Static toroidal SOLID entries are not world geometry. Open ground currently marks y=56..63 SOLID to represent the floor under y=0. Under y motion that repeats the floor/ceiling every 64 pages. New exposed pages need world classification without rebuilding the table or trusting old sentinels. |
| Residency seed boxes | `uniform-mixed-residency.ts`: `rsAsk`, `rsSeedBox`, `rsResident`; topology residency helpers | Local 3-bit seed boxes and page keys can stay unchanged; derive their page coordinates through F. Collect signed world request bounds before the old out-of-window rejection. The current `rsAsk` edge branch must be replaced, not merely suppressed. |
| Live and scratch particle positions | `uniform-narrow-band-flip.ts`: parity banks, refine, `advectParticles`, bootstrap, resample; `uniform-narrow-band-order.ts`: `pack` | Apply delta exactly once to each live position before a new-frame consumer reads it. Keep x=-1 dead marks intact. Velocity and `before.xyz` are motions, not positions; do not translate them. The inactive particle bank also serves as extension scratch. |
| Compact particle positions | `uniform-narrow-band-order.ts`: `links` writes in `pack`; `uniform-narrow-band-flip.ts`: `nbPosition`, `markParticleTiles` | These are cached lattice floats independent of the particle bank. Rebuild/rebase before first use in a new frame. Rebasing only `Particle.position` is insufficient. |
| Surface seeds and nearest banks | `uniform-narrow-band-seed.wgsl.ts`, `uniform-narrow-band-nearest.wgsl.ts`, `uniform-narrow-band-redistance.wgsl.ts`, refine's `NB_DEPTH_B` access | Cell/tile keys and page-local row codes are stable. Coordinates decoded from keys must see new F; cached absolute point/trace data need lifetime checks. |
| Authored refinement regions | `uniform-mixed-layout-builder.ts`: `UniformMixedRegion`, `uniformMixedRegionTable`; `uniform-detail-requests.ts`, `uniform-detail-planner.ts` | Region min/max currently are unsigned window tiles and host-clipped to 0..512. Store world boxes or translate at GPU comparisons; no stale host lattice bounds after shift. |
| Shared params and inflow | `webgpu-uniform-reference.ts`: `writeParams`; `webgpu-uniform-reference.wgsl.ts`; `lib/core/inflow-boundary.ts` | The host uploads the whole 72-float params scratch. GPU writing frame.xyz alone would be undone next step. Maintain distinct host input and GPU-derived live params, or have all consumers read F directly. `tuning.xyz` also changes by -32 delta. |
| Sources, scheduled drops, host edits | `uniform-source.wgsl.ts`, `uniform-mixed-source.wgsl.ts`, `webgpu-uniform-reference.ts`: `requestDropPages`, `liveFluidEditRefusal`, pins/source updates | Keep world-authored source commands. Convert through current F on GPU. Host page requests can remain signed world pages without an awaited mirror. UI diagnostics may use asynchronous receipts. |
| Bodies, procedural solids, voxel masks, terrain and bed slope | `uniform-mixed-solid.wgsl.ts`: `umSolidBoxCell`, world/trace helpers, mask directory; `uniform-mixed-bodies.ts`; `uniform-solid-boundary.wgsl.ts`; `uniform-bed-slope-bake.ts` | World body positions stay fixed. Translate GPU lattice sampling and box origin; preserve world masks/boundary sample coordinates. Audit any box-shaped caches separately. Keep the other solid session's hunks. |
| Low-face and wall rows | `uniform-lattice-address.ts`: `umFaceLow`, `umLowFaceAbove`; `uniform-block-neighbours.ts`; `uniform-page-initialise.ts` | Block/face-local rows stay put, but world SOLID classification must remain correct on both sides of a face. Recompute derived neighbours in the existing pass when needed. |
| Field atlases, ownership and pressure keys | `uniform-detail-fields.ts`, topology helpers, root owners, pressure pages, `uniform-texture-pages.ts` | Block-local field/texture addresses and owner/tile keys stay put. Fix shared coordinate decode/readers; no field copy or solve should be added for shifting. |
| Surface extraction and page overlay | `lib/core/webgpu-water-pipeline.ts`: `mixedFrame`; `lib/core/method-view-records.ts`: `GPUFluidVolumePageSource`; `uniform-page-pool.ts`: published block records | Use GPU-derived live origin and the same F as coordinate decoding. Published full xyz are world pages; do not multiply them by h and add an already-shifted origin a second time. Preserve phase-3 record layout. |
| Particle layer and front diagnostics | `uniform-narrow-band-flip.ts`: `particleSource`; `webgpu-uniform-reference.ts`: front receipt conversion | Particle presentation must use live GPU origin. Diagnostics can use a receipt carrying its matching F/revision; a newer or older host mirror is not the frame of that receipt. |

All `uniform-*.ts` paths in that table are under `lib/methods/uniform/`
unless explicitly prefixed otherwise.

## Remaining blocker and investigation

The concrete blocker is the tank's forced-shift pressure failure, not a
missing interface or permission. Current A/B control reaches step 30 at
liquid 3071.829102 / 3584 particles. The shifted run matches the control at
step 10 (3071.939941 / 3584) with F=(-2,-1,-1), then fails at step 19,
layout generation 4, candidate Infinity, accepted 0. No pool fatal precedes it.
An earlier version that automatically recentered the tank to approximately
(-31,-31,-31) failed in its initial pressure kick. Avoiding unnecessary
finite-tank recentering does not resolve the forced-shift failure.

The next investigation should isolate the first differing field/stage across
that boundary, including low-face and vertex sampling when a world wall no
longer lies at lattice coordinate zero, static cut/solid records, pressure
geometry, cached coordinate payloads, and particle rebasing. Do not change
timesteps, suppress the fatal, loosen A/B tolerances or add another solve.

Other unverified seams: analytic terrain beyond the old cache, procedural
solids/contact after a shift, sources and bodies, live edits/refinement away
from the initial box, renderer snapshots, y travel, repeated automatic shifts,
and simultaneous frames in flight. No browser validation was done.

## Oracle commands and observed output

Host-only (re-run after integration; the first two result lines below remain unchanged):

```sh
node --import tsx tools/check-window-shift-twin.ts
```

```text
PASS host twin: sets=10000 pages=320000 shifts=7182 forced=5000 signedPoolRecords=4090 tableMoves=0
PASS UNIFORM_PAGE_POOL_FATAL.edge=32: rejected=6 (65-page extent, margin, invalid force, signed overflow); signed endpoint cases=2
SCOPE: host address/decision arithmetic; see check-window-shift.ts for the fluid gate
```

Device primitive only (run in the earlier primitive-only turn, from bash with GPU lock; the historical final UNVERIFIED line below describes that earlier state):

```sh
/usr/bin/lockf /private/tmp/claude-501/fluid-gpu.lock env WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/check-window-shift-decision.ts
```

```text
PASS travelling bounds: decisions=128 F=3712,0,-2048 shifts=128 fatal=0 decisionCompile_ms=28.52
PASS forced boundary 10: decisions=12 F=-99997,0,89998 shifts=1 fatal=0 decisionCompile_ms=1.01
PASS empty: decisions=2 F=-99,-34,40000 shifts=0 fatal=0 decisionCompile_ms=0.86
PASS i32 low: decisions=2 F=-2147483648,-31,-31 shifts=1 fatal=0 decisionCompile_ms=0.81
PASS i32 high: decisions=2 F=2147483584,-31,-31 shifts=1 fatal=0 decisionCompile_ms=0.70
PASS extent edge sticky: decisions=2 F=0,0,0 shifts=0 fatal=32 decisionCompile_ms=0.61
PASS margin edge: decisions=1 F=0,0,0 shifts=0 fatal=32 decisionCompile_ms=0.77
PASS full signed extent: decisions=1 F=0,0,0 shifts=0 fatal=32 decisionCompile_ms=0.64
PASS force outside bounds: decisions=1 F=0,0,0 shifts=0 fatal=32 decisionCompile_ms=0.68
PASS force i32 overflow: decisions=1 F=2147483584,0,0 shifts=0 fatal=32 decisionCompile_ms=0.94
PASS force i32 underflow: decisions=1 F=-2147483648,0,0 shifts=0 fatal=32 decisionCompile_ms=0.72
PASS force INT_MIN delta: decisions=1 F=-2147483648,0,0 shifts=1 fatal=0 decisionCompile_ms=0.83
PASS decision primitive: cases=12 decisions=154 shifts=132 fatalCases=6 gatherCompile_ms=206.59 maxCompile_ms=206.59 errors=0
UNVERIFIED: fluid forced-shift A/B and travel-past-window gates; primitive is not attached to the solver
```

Solver commands (bash, from the repository, one full gate per scene only in
the main session after resolving the blocker):

```sh
/usr/bin/lockf /private/tmp/claude-501/fluid-gpu.lock env WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/check-window-shift.ts ab hydrostatic-power-two-level
/usr/bin/lockf /private/tmp/claude-501/fluid-gpu.lock env WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/check-window-shift.ts ab uniform-trough-dam-break
/usr/bin/lockf /private/tmp/claude-501/fluid-gpu.lock env WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/check-window-shift.ts travel
# Short single-run fixture, if a targeted smoke is needed:
/usr/bin/lockf /private/tmp/claude-501/fluid-gpu.lock env WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/check-window-shift.ts smoke
```

Runs performed in this integration turn:

1. `tools/probe-nbflip-peaks.ts hydrostatic-power-two-level 1 1`, under the
   same lock/env: shader compilation succeeded, then initial kick pressure
   nonfinite, candidate Infinity / accepted 0, layout generation 1; pool
   fatal none, particles 0. Fixed static-mask addressing afterwards.
2. `check-window-shift.ts smoke`: same initial-kick failure after that fix.
3. `check-window-shift.ts smoke`, after keeping a finite tank's initial F:
   step 1 liquid 3072, step 9 liquid 3071.958984 / particles 3584. The then
   positive force `[2,-1,1]` could not hold page zero and correctly raised
   edge at step 10. Corrected force to `[-2,-1,-1]` afterwards.
4. `check-window-shift.ts ab hydrostatic-power-two-level`, once:

```text
control step=1 F=0,0,0 shifts=0 liquid=3072.000000 particles=0 fatal=none
control step=9 F=0,0,0 shifts=0 liquid=3071.958984 particles=3584 fatal=none
control step=10 F=0,0,0 shifts=0 liquid=3071.939941 particles=3584 fatal=none
control step=30 F=0,0,0 shifts=0 liquid=3071.829102 particles=3584 fatal=none
forced step=1 F=0,0,0 shifts=0 liquid=3072.000000 particles=0 fatal=none
forced step=9 F=0,0,0 shifts=0 liquid=3071.958984 particles=3584 fatal=none
forced step=10 F=-2,-1,-1 shifts=1 liquid=3071.939941 particles=3584 fatal=none
FAIL frame 19: pressure went nonfinite (candidate Infinity, accepted 0), layout generation 4
```

5. `check-window-shift.ts travel`, once, after avoiding unnecessary ground-y
   recentering:

```text
step=1 F=29,0,0 shifts=1 liquid=36870.512207 particles=28646 fatal=none
step=9 F=29,0,0 shifts=1 liquid=38632.775879 particles=67656 fatal=none
step=10 F=29,0,0 shifts=1 liquid=38476.126953 particles=76983 fatal=none
step=20 F=29,0,0 shifts=1 liquid=37263.092773 particles=115231 fatal=none
step=40 F=29,0,0 shifts=1 liquid=37384.855957 particles=144527 fatal=none
step=60 F=29,0,0 shifts=1 liquid=38029.658203 particles=157192 fatal=none
step=80 F=29,0,0 shifts=1 liquid=37856.835449 particles=112288 fatal=none
step=100 F=29,0,0 shifts=1 liquid=38229.149902 particles=171757 fatal=none
step=120 F=29,0,0 shifts=1 liquid=39098.137695 particles=197107 fatal=none
FAIL liquid front crossed initial +x window
longest pipeline wall ms: seedSources 4683.66, seedInitial 4202.75, advectParticles 3435.69
```

The travel run was a fixture bug: setting `initialVelocity_m_s` alone did
not upload velocity in this solver. The current tool now explicitly calls
`initializeVelocityForQA` with 20 m/s and prints front_m versus the old edge.
It requires **liquid-front** crossing, not just a resident request beyond the
edge. That corrected version is unrun. The recorded liquid drift above is
also about +6.04% relative to step 1; it would fail the next 6% assertion.

Each running process was under two minutes; waiting for another session's
GPU lock sometimes took longer. Full A/B and travel were not repeated.
Trough A/B has not been run. Recorded compile timings are concurrent Dawn
pipeline wall times on Metal, all under 5 s in the travel run; no Metal dump
was taken. The last native terrain/contact helper cleanup has only type/host
verification, not another device run.

`npm run check:types`: passed after the integration and probe correction
(`tsc --noEmit`, exit 0). `git diff --check`: passed. Unit suite deliberately
not run per the supplied common.md.

## What is unverified / not accepted

- The tank forced-shift A/B is a known failure, not a pending optional gate.
- Corrected liquid travel beyond 2,048 cells is unverified; the recorded
  120-step run proves an applied GPU shift and no fatal, not edge crossing.
- Trough A/B, renderer, moving solids/bodies/sources, analytic terrain outside
  the cache and repeated/vertical shifts are unverified.
- Base edge failure is source reasoning only: old `rsAsk` rejects x=64 at
  y=1 when the wrapped x=0 entry is AIR or resident. The corrected travel
  fixture must first prove its liquid reaches that plane; no base run was made.
- Host arithmetic and the earlier device decision primitive pass. Neither
  overrides the failed physical gate. This hand-over is blocked on solver
  correctness and is not ready to be described as a completed feature.
