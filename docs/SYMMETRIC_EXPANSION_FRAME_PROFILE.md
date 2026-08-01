# Symmetric-expansion fine-band-4 frame profile

Date: 2026-08-02. Device: Apple M1 Max (32 GPU cores, 98,304 threads,
4 partitions x 768 scheduler slots). Backend: Dawn/Metal, headless.
Scene: `symmetric-expansion` at `32 x 16 x 32` cells, `0.05 m`, leaf 32,
interface band 3, **global fine level-set factor 4**, `dt = 0.004 s`.
Companion documents: `SYMMETRIC_EXPANSION_ORACLE.md` (what this scene proves),
`POWER_LIQUIDS_FINE_BAND_10X.md` (the non-pressure ladder),
`POWER_LIQUIDS_ULTIMATE_M1MAX.md` (protocol and refutation log — normative).

## The lane

The symmetry oracle's `fine-factor-4` lane is one step and carries the D4
evidence collectors, so it cannot be a wall clock. `symmetric-expansion` is now
also a **profiling lane** — the same solver configuration with the collectors
removed:

```sh
node --import tsx tools/benchmark-power-dam.ts --lane=symmetric-expansion --steps=62
node --import tsx tools/profile-mini-dam-xctrace.ts --lane=symmetric-expansion \
  --counters --isolate-pass-labels --out=artifacts/xctrace-symmetric-expansion
```

It is `lib/scene-webgpu-smoke-catalog.ts`'s `symmetric-expansion/performance`
lane plus a `POWER_DAM_LANE_ENVIRONMENT` entry, so a change scored here is
re-gated on symmetry by `test:webgpu:symmetric-expansion*` without moving the
scene.

**Warm-cache caution.** The first run in a fresh shell measured
405 ms/advance against 228–256 ms for every later run: pipeline construction is
not cached across the first process. Never compare a first run with a later
one; interleave every A/B.

## Capture

`report.html` / `summary.json` in `artifacts/xctrace-symmetric-expansion/`.
Measurement integrity, printed by the run itself:

- attribution **full** — 277 exact stages, **0 composite buckets**;
- counter exclusive coverage **98.7 %**;
- 14 complete advances retained, frame median **166.9 ms**, GPU busy
  **136.2 ms**; the window landed 25.5–28.5 s into stepping (advance ~124–138);
- untraced control printed by the same run: **233.7 ms/advance over 174
  advances**. Per-stage numbers below are a ranking under label isolation, not
  a wall.

## The one number that explains the frame

| | value |
|---|---:|
| mean compute occupancy | **8.7 %** |
| mean ALU utilisation | **16.4 %** |
| GPU read bandwidth | **2.65 GB/s** (of ~400) |
| GPU write bandwidth | **0.30 GB/s** |

The frame is neither ALU-bound nor bandwidth-bound. It is **starved**: 91 % of
the machine is dark for 136 ms. Every item below is a work-removal or a
launch-width question, never a dispatch-count question (refutations #9 and the
general lesson in `POWER_LIQUIDS_ULTIMATE_M1MAX.md`).

## By family

| family | ms/adv | share | calls |
|---|---:|---:|---:|
| SPGrid V-cycle (first-order M1) | 39.94 | 29.3 % | 146 |
| Section 5 air support | 34.88 | 25.6 % | 9 |
| pressure operator / §4.3 / CG | 19.92 | 14.6 % | 860 |
| fine transport | 10.71 | 7.9 % | 13 |
| octree topology | 7.78 | 5.7 % | 20 |
| fine JFA / redistance | 6.51 | 4.8 % | 34 |
| fine volume / summary / restriction | 6.38 | 4.7 % | 37 |
| structured velocity | 3.80 | 2.8 % | 29 |
| power coarse level set | 0.63 | 0.5 % | 8 |
| other | 5.66 | 4.2 % | 1269 |

## The stages that matter

| ms/adv | calls | occ | ALU | placement | stage |
|---:|---:|---:|---:|---:|---|
| 30.21 | 1 | 15.5 % | 34.4 % | 1.22x | March Section 5 sparse changed frontier to a fixed point — topology-commit |
| 29.42 | 11 | **0.1 %** | 0.1 % | **4.00x** | SPGrid V-cycle — coarse V-cycle tail levels 2-bottom |
| 11.47 | 165 | 1.8 % | 5.0 % | 1.04x | SPGrid §6.3 — ordered merged-band row fold |
| 7.02 | 1 | 14.3 % | 29.5 % | 1.09x | Advect fine phi rare |
| 3.66 | 1 | 1.2 % | 4.6 % | 2.29x | Octree resident grading closure |
| 3.52 | 1 | 16.7 % | 35.6 % | 1.10x | Advect fine phi common |
| 2.79 | 165 | 13.3 % | 9.4 % | 1.20x | SPGrid §6.3 — parallel merged-band adjoint children |
| 1.95 | 11 | 25.9 % | 28.0 % | 1.15x | SPGrid V-cycle — restrict level 0 |
| 1.90 | 1 | 0.5 % | 0.2 % | **4.00x** | SPGrid V-cycle — candidate build level sets |
| 1.63 | 1 | 0.2 % | 0.6 % | **4.00x** | Compare topology-tile refinement signatures |
| 1.54 | 2 | **53.7 %** | **72.6 %** | 1.01x | Fine JFA — cooperative flood B to A stride 1 |

Placement 4.00x means the stage ran on one of the GPU's four partitions with
the other three at literally 0.00 % — a dispatch too narrow to reach the
machine. The JFA flood is the one healthy family in the frame and is the shape
everything else should be measured against.

## Defect 1 — the coarse V-cycle tail was authored for a 16-cell domain (FIXED)

`encodeCorrectionBody` ran the two widest levels as globally synchronized
per-level dispatches and folded **every remaining level into one 64-lane
workgroup**, with the cut hardcoded as `Math.min(2, levelCount - 1)`. The
in-file comment states the assumption outright: *"the remaining <=63-cell tail
fits in one workgroup"*. That is true for a 16-cubed domain, where level two
holds 64 cells. It is false for every larger scene:

| domain | level cells (finest first) | old tail | new tail |
|---|---|---|---|
| 16 x 16 x 16 | 4096, 512, **64**, 8, 1 | from level 2 | from level 2 |
| 24 x 16 x 24 | 9216, 1152, 144, **18**, 4, 1 | from level 2 | from level 3 |
| 32 x 16 x 32 | 16384, 2048, 256, **32**, 4, 1 | from level 2 | from level 3 |
| 64 x 20 x 64 | 81920, 10240, 1280, 192, **32**, 4, 1 | from level 2 | from level 4 |

On this scene the tail therefore owned 293 cells, and its restriction walks
parent chains **one coarse cell at a time on lane zero** — the wide
`restrictAndGhostAccumulate` runs one workgroup *per* coarse cell with a
byte-identical body. Measured: **29.42 ms/advance, 0.1 % occupancy, 4.00x
placement** — 64 threads of 98,304 for 22 % of the frame.

`planOctreeVCycleParallelLevels` (`lib/octree-solve-tail-policy.ts`) now
derives the cut from immutable level geometry, with the former constant kept as
a **floor** so the selection can only ever add wide levels. Consequences:

- every domain at or below 16 cells — including the shipping mini lane and
  every `<=16` authored scene — keeps a **bit-identical command graph**;
- `FLUID_OCTREE_VCYCLE_TAIL_CELLS` reproduces any earlier cut (256 on this
  scene restores the former two) and is the rollback lever.

The arithmetic per cell is unchanged either way: the tail's `coarseSmooth` /
`coarseRestrict` / `coarseProlong` are transcriptions of the dispatched
kernels' bodies (`coarseApplied` is `applied` with the level passed in rather
than read from the dispatch uniform), and both fold through the same
`canonical8Sum` / `canonical18Sum`. Only the launch shape moves.

**Measured, interleaved A/B/A/B at fixed `--steps=62`, 0 validation errors on
every arm:**

| round | former cut (`=256`) | widened (`=64`) |
|---|---:|---:|
| 1 | 256.76 | 233.10 |
| 2 | 253.18 | 231.11 |
| earlier pair | 254.47 / 241.97 | 229.47 / 227.94 |

Mean over the interleaved pair: **254.97 -> 232.11 ms/advance, -22.9 ms
(-9.0 %)**, repeatable in every pairing.

### Paper compatibility (Aanjaneya et al. 2017, §4.3)

The preconditioner `M`'s action is: (1) smooth the second-order power-diagram
discretization near boundaries and level transitions, (2) update the residual
and run an **aggressively optimized first-order V-cycle over the entire
domain**, (3) a second matching smoothing sweep so `M` is symmetric.

This change touches step (2) and only its launch shape. Unchanged: the level
count and therefore the domain the V-cycle covers; the Chebyshev degree and
phase order; the restriction/prolongation transfer records (`E` and `E^T` still
consume the same parent chains, so they cannot diverge); the exact one-cell
bottom solve, which stays in the tail; and the `k` boundary-smoothing halves of
steps (1) and (3). The descend/ascend mirror that makes `M` symmetric is
preserved on both sides of the cut — the wide part is
`pre-smooth, restrict ... prolong, post-smooth reversed`, and the tail's own
loop has the same shape — so the property §4.3 requires is structural, not a
consequence of where the cut falls. Section 5 is untouched.

### Symmetry verification

The D4 oracle's one-step and three-step lanes pass **`maximumAbsoluteError` 0**
on every enforced field, including the seven §4.3 stages this change re-shapes:
`preconditionerPreSmoothed`, `preconditionerZeroSmoothed`,
`preconditionerFirstOperatorImage`, `preconditionerFirstSmoothed`,
`preconditionerInnerResidual`, `preconditionerInnerCorrection`,
`preconditionerPostCorrected`, plus `rhs`, `diagonal`, `section63Diagonal`,
`section63CaseId` and exact topology.

The 250-step factor-1 lane and the factor-4 frame-zero lane remain red, as
`SYMMETRIC_EXPANSION_ORACLE.md` documents them to be. Both were re-run on both
arms of the rollback lever so the red state is compared like for like rather
than assumed unchanged:

| oracle quantity | former cut (`=256`) | widened (`=64`) |
|---|---|---|
| factor 1, 250 steps — volume / velocity / pressure / rhs first loss | step 68, t=0.272 s | step 68, t=0.272 s |
| factor 1 — diagonal / exact topology first loss | step 69, t=0.276 s | step 69, t=0.276 s |
| factor 1 — wall contact (-x, +x, -z, +z) | 68, 68, 68, 68 — spread **0** | 68, 68, 68, 68 — spread **0** |
| factor 4, step 1 — volume max abs error | 8.58307e-06 (176 cells) | 8.58307e-06 (176 cells) |
| factor 4, step 1 — velocity max abs error | 1.86265e-08 (8,492) | 1.86265e-08 (9,130) |
| factor 4, step 1 — pressure max abs error | 4.27246e-04 (3,092) | 4.27246e-04 (3,414) |

Every first-divergence step is identical and every maximum error is identical
to the last bit. Only the factor-4 mismatch *counts* move, which is the
expected Gate-B signature: the change is a launch-shape change across a storage
round-trip, so it is not claimed bit-exact (refutation #12). Nothing in the
oracle gets redder.

Note for `SYMMETRIC_EXPANSION_ORACLE.md`: its recorded factor-1 divergences
(volume step 13, topology step 16, wall spread `67, 63, 67, 64`) are stale
against this working tree, which reaches step 68 with a wall spread of 0 on
both arms. That improvement predates this change.

## Defect 2 — Section 5 relaxes the seeded liquid interior (FIX AUTHORED, UNSCORED)

`March Section 5 sparse changed frontier to a fixed point` is 30.2 ms at
15.5 % occupancy and 34.4 % ALU — the highest-ALU large stage in the frame, so
it is doing real arithmetic, just far too much of it. Its settled-fine twin
costs 0.06 ms, so this is entirely the topology-commit publication.

The wide-wave prefix is **not** the lever. A sweep of
`FLUID_OCTREE_AIR_SUPPORT_FRONTIER_WAVES` (added as a measurement lever;
the persistent tail remains the exact unbounded authority) measured, clean wall
at 62 advances:

| waves | 0 | 12 (authored) | 24 | 48 |
|---|---:|---:|---:|---:|
| ms/advance | 233.90 | 227.27 | 227.47 | 227.05 |

Flat. The propagation work is a fixed quantity that redistributes between the
wide waves and the 3-workgroup persistent tail; adding waves cannot remove it.

### The first reading of this stage was wrong

This section previously concluded, from the mini-lane cost ratio alone, that
the march "extrapolates across the entire air domain" and that the fix was the
unbuilt *seed-to-demand corridor* of `POWER_LIQUIDS_5X_GPU_PLAN.md` §5. A
direct census refutes that. `FLUID_OCTREE_AIR_SUPPORT_CENSUS=1`
(`WebGPUOctreeProjection.readAirSupportCensus`) publishes the Section 5 control
header per advance; on this scene at the settled generation:

| quantity | value |
|---|---:|
| direct (liquid) rows | 1,152 |
| support (air corridor) rows | 1,716 |
| owned face patches | 34,416 |
| **seed patches** | **13,184** |
| domain cells | 16,384 |

The marched set is **already the exact corridor**: 1,716 air rows around 1,152
liquid rows, not the 14,000-cell air region. `reconstructAirSupportVectors`
consumes precisely that set (`publishedDirectDemandedRow`), so §5's demand
bound is in place and *no corridor work remains to be done*. The cost is not
the size of the marched set. It is **cost per visit multiplied by an initial
frontier that is 38 % of every face in the publication**.

### What the 30 ms actually is

Wave zero's frontier is every seed. The liquid interior is seeded solid —
13,184 of 13,824 liquid-row patches carry a seed — so nearly every reciprocal
destination in wave zero is *itself a seed*, and each one was being enqueued
(one dedup atomic) and then relaxed through the full 30x4 closest-face gather.
A seeded patch carries squared distance zero and every candidate distance is
the separation of two distinct owned face centres, so `betterFace` can never
displace it: **that entire wave-zero relaxation is provably a no-op.**

Three exact removals, all authored behind
`FLUID_OCTREE_AIR_SUPPORT_MARCH_FASTPATH` (`0` restores the authored source
for each, so the pair can be scored interleaved):

1. **Settled-seed frontier skip** — `appendFrontierDestination` drops
   destinations that are already seeds. An unseeded patch on a liquid row keeps
   its queue slot exactly as before, so no reachable face loses a relay.
2. **Loop-invariant patch centre** — `faceDistanceSquaredFrom` takes the
   marching patch's own quarter-cell centre as a value. It was being re-derived
   for every one of up to 120 candidates.
3. **Single-bank candidate read** — `select(faceB[x],faceA[x],readA)` is a
   function call, so both banks were loaded and one discarded. The bank is
   dispatch-uniform; branching halves the traffic of the hottest loop.

All three compute the same marched field. **None is scored yet**: the
`symmetric-expansion` and `mini` lanes both stopped constructing partway
through this work (`Octree pressure capacity exceeds the persistent production
executor limit`, and `power MGPCG authority is unavailable` on mini), which is
an unrelated in-flight pressure-executor change in the working tree. The
numbers belong here once either lane runs again.

## Levers measured and rejected on this scene

- **`FLUID_OCTREE_SECTION43_SHELL_DEPTH=4`**: 221.98 vs 229.95 ms at k=8, same
  4 executed CG iterations. ~3.5 %, but §4.3 states `k ≈ 8` is what achieves the
  6–10 iteration convergence, and the repo restricts k=4 to the separately
  validated small two-level profile. Not taken without a convergence study
  across the run.
- **`FLUID_OCTREE_PERSISTENT_MGPCG_LARGE=1`**: the discovery arm does not
  publish control buffers for this scene and fails the step-1 tripwire. Not a
  lever here. (Row count is only 2,124, so the one-workgroup executor would be
  8 workgroups' worth of rows either way — see refutation #2.)

## What 5x would take

The whole frame runs at 8.7 % occupancy, so the headroom is real, but it is
spread across the families above rather than concentrated. Sized from this
capture, against the 136.2 ms attributed busy:

| item | ms | status |
|---|---:|---|
| coarse V-cycle tail widening | ~26 | **landed**, -9.0 % clean wall |
| Section 5 seed-frontier and candidate-scan removals | ~? of 30 | authored, **unscored** — lanes down |
| §4.3 merged-band applies (k, or phase packing at 1.8 % occ) | ~8 | needs convergence study |
| fine transport substeps / staged addressing (E-4, E-5) | ~7 | ladder Wave 1/2 |
| grading closure from the delta cone (E-3) | ~3 | ladder Wave 1 |
| one-workgroup-per-level candidate rebuild (4.00x placement) | ~4 | refutation #10 applies; hard |
| capacity-shaped fine-lane launches (E-6) | ~4 | ladder Wave 1 |

That ledger is ~80 ms of the 136 ms, i.e. a credible path to roughly **2.5x**
inside the current architecture. The remaining factor needs the Wave 2
delta-repair regime and phase packing that `POWER_LIQUIDS_FINE_BAND_10X.md`
scopes; nothing measured here contradicts that document's conclusion that
work removal and delta-shaped launches are the only levers that have ever
moved this wall.
