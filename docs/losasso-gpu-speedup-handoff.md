# Losasso solver 20x GPU speedup — implementation handoff

**Date:** 2026-08-04
**Capture:** `artifacts/xctrace-losasso-d4-cutover-2026-08-04/report.html`
**Target:** 148.32 ms attributed GPU → ~7.4 ms (20x) for the first-advance interval.

## Framing: this is a latency problem, not a throughput problem

The capture's three worst labels (118.01 ms, 79.6% of attributed time) all run at
≤0.1% occupancy and ≤0.5% ALU. The live problem is **1,152 rows and 3,616 faces**.
One application of the pressure operator is ~12K useful flops; the entire MGPCG
solve is a sub-millisecond workload on this hardware. Do not evaluate changes by
occupancy — evaluate them by (a) serialized work done by 1–64 threads,
(b) capacity-shaped work (4,096 rows / 98,304 faces / 262,144 directory slots
touched when 1,152 / 3,616 / 3,616 are live), and (c) count of serialized
dispatch boundaries (952 in the V-cycle alone).

The second structural theme: the codebase buys run-to-run determinism with
**exact 36-limb radix-256 arithmetic under unordered atomics**. Determinism only
requires a **fixed reduction order**; where the order is (or can be made)
deterministic, plain f32 summed in that order is bitwise reproducible for free.
Several of the exact-accumulator sites already have a deterministic order and
are paying for insurance against a nondeterminism that no longer exists.

Measured budget and where it must land:

| Label | Now | After this plan |
|---|---|---|
| V-cycle — initialize levels (17 calls) | 61.27 ms | ~4–6 ms |
| Hierarchy — publish geometric levels (3 calls) | 38.21 ms | ~1–2 ms |
| Topology — publish compact axis faces (1 call) | 18.53 ms | <1 ms |
| Everything else | ~30 ms | needs the same audit (out of scope here) |

The remaining ~30 ms almost certainly carries the same diseases (capacity-shaped
dispatch, exact clears); re-capture after WP1–WP6 and repeat the audit there to
close the last gap to 7.4 ms.

## Verified anatomy (line refs current as of 2026-08-04 working tree)

- **V-cycle** `lib/webgpu-octree-losasso-vcycle-gpu.ts`
  - Row operator `image()` with two private 36-limb accumulators: lines 42–58.
  - `clearLevel` (line 59) and `clearRestriction` (line 88) do **not** check
    `stopped()`; every other kernel does (line 27).
  - `clearRestriction` is `dispatchWorkgroups(1)` (line 233) — 64 lanes clearing
    `rowCapacity(4096) × 36 = 147,456` atomic words per transfer
    (`clearFixedPartial` loop: `lib/webgpu-exact-f32-reduction.ts:147`),
    4 transfers per V-cycle, 17 calls → ~10M single-workgroup atomic stores.
  - Schedule: 56 dispatches per call (`encodeCorrection`, lines 195–262);
    52 indirect via per-level `rowDispatch`, which is hierarchy-owned and
    nonzero even after MGPCG convergence — retired iterations still run the
    whole pass graph (0.855 ms each × 5 retired iterations).
  - Capacity coupling to the exact reducer: `SIGNED_RADIX_256_F32_MAX_TERMS`
    checks at `webgpu-octree-losasso-vcycle-gpu.ts:155` and
    `webgpu-octree-losasso-hierarchy.ts:102`.
- **Hierarchy** `lib/webgpu-octree-losasso-hierarchy.ts` / `.wgsl.ts`
  - Per transition, three `@workgroup_size(1)` singletons dispatched with one
    workgroup (`encodeTransition`, `.ts:253`): `buildLosassoParentRows`
    (`.wgsl.ts:63` — clears the 8,192-slot directory and 4,096 scratch rows,
    then serially hashes every fine row), `buildLosassoCoarseFaces`
    (`.wgsl.ts:121` — serial stable filter over all fine faces),
    `buildLosassoCoarseCSR` (`.wgsl.ts:146` — serial count/prefix/scatter).
    Only `finishLosassoParentRows` is wide.
  - 4 transitions per publish, 3 publishes in the first advance (ready-commit
    path near `lib/webgpu-octree.ts:4383`; bootstrap `encodeHierarchyRefresh`
    at `lib/webgpu-octree.ts:4861`; post-volume-correction republish just
    below, ~`:4865`+). Publishes 2 and 3 exist because coarse-phi ghost
    conditioning changes **face fields**, not leaf topology.
- **Topology** `lib/webgpu-octree-losasso-backend.ts` / `.wgsl.ts`
  - Nine kernels in one pass (`PUBLISH_ENTRY_POINTS`, `.ts:152`); host
    dispatches by **capacity**: `rowGroups = ceil(4096/64)`,
    `faceGroups = ceil(98304/64)` (`.ts:434–436`) while 3,616 faces are live.
  - Singletons: `prefixLosassoFaces` (`.wgsl.ts:259`),
    `prefixLosassoIncidences` (`.wgsl.ts:419`), and the big one,
    `finishLosassoPublication` (`.wgsl.ts:481`): one thread stores 2 MiB of
    zeros into the 262,144-slot `faceDirectory`, then inserts 3,616 faces,
    then clears adjacency offsets. This kernel alone explains the 18.53 ms.
  - `sortLosassoIncidences` (`.wgsl.ts:444`) insertion-sorts each row's ≤24
    incidences — so **CSR incidence order is already deterministic** — and then
    assembles diagonal/RHS with two more private 36-limb accumulators
    (`.wgsl.ts:462–478`).

Provenance caveat: the topology WGSL has post-capture local edits (closed-boundary
face flags / multiplicity). They don't change the structure above, but re-baseline
timings with a fresh capture before/after each work package.

## Work packages

Ordered by leverage ÷ risk. WP1–WP4 change no numerical contracts and can land
independently, each behind a fresh capture. WP5 is a contract renegotiation and
gets its own gate run. WP6 depends on WP5 (or a ported exact deposit).

### WP1 — Stop doing dead and capacity-shaped work (no contract changes)

1. **Gate the clears on `stopped()`.** Add the check to `clearLevel`
   (`vcycle-gpu.ts:59`) and `clearRestriction` (`vcycle-gpu.ts:88`).
   `clearLevel` needs the `solve` binding added to its bind list (`:202`).
   Wins the ~4.3 ms retired-iteration tail almost entirely.
2. **Live-sized, wide restriction clear.** Replace the
   `dispatchWorkgroups(1)` at `vcycle-gpu.ts:232` with an indirect dispatch
   over the coarse level's `rowDispatch`, each invocation clearing its own
   scalar's 36 limbs (bind `coarseControl`, clear `coarseControl[1] × 36`
   words instead of `rowCapacity × 36`). This is a stopgap that WP5 deletes
   outright, but it's a few lines and removes the single largest V-cycle cost
   (~40+ ms across 17 calls) immediately.
3. **Live-count face dispatch in topology.** `prefixLosassoFaces` already
   computes the live face total into `authority[2]` (`.wgsl.ts:267`) before
   the face-wide kernels run. Have it also write a face dispatch record
   (workgroup count) into a scratch indirect buffer, and switch
   `emitLosassoFaces` / `conditionLosassoFaces` / `scatterLosassoIncidences`
   to `dispatchWorkgroupsIndirect` (`backend.ts:436–441`). 27x fewer lanes
   launched; modest but free, and it removes capacity from the scaling law.

### WP2 — Decompose `finishLosassoPublication` (topology: 18.5 ms → <1 ms)

The singleton does three jobs; only one is genuinely serial.

1. **Parallel directory clear.** New 64-wide kernel clearing the 262,144
   `faceDirectory` slots (or generation-tag the slots — store the publication
   epoch in the slot and treat stale epochs as empty — making the clear a
   uniform bump; the tag fits by widening the slot or folding epoch bits into
   the `.y` hash word).
2. **Parallel adjacency-offset clear** (`count+1` words — trivially wide).
3. **Keep the insertion serial.** Inserting 3,616 faces with ~1 probe each is
   a few-microsecond loop; keeping it singleton preserves the deterministic
   probe/layout order with zero design risk. Do **not** build a parallel
   open-addressing insert here — that's the determinism trap flagged in the
   analysis, and at 3,616 items it buys nothing.
4. **Cooperative prefix scans.** Replace `prefixLosassoFaces` and
   `prefixLosassoIncidences` with single-workgroup (256-lane) cooperative
   scans over `rows()` using workgroup memory + barriers. Fixed lane
   assignment ⇒ deterministic. A 4K-element workgroup scan is microseconds.

### WP3 — Publish the hierarchy once per topology epoch (38.2 ms → ~13 ms)

Publishes 2 and 3 rebuild identical topology because only L0 face **fields**
(open fractions, ghost inverse-distances, normal velocities) changed.

1. During `buildLosassoCoarseFaces`, record `coarseFaceSource[coarseFaceId] =
   fineFaceId` (the retained-patch mapping already exists implicitly — the
   kernel copies the fine face and rewrites rows, `.wgsl.ts:140`). One extra
   u32 buffer per level.
2. Add `encodeCoefficientRefresh`: per level, a wide kernel that gathers
   `openFraction` / `normalVelocity` / free-surface `inverseDistance` from the
   (already refreshed) finer level through `coarseFaceSource`. Run levels in
   order L1→L4. Pure gather with a build-time-fixed mapping ⇒ deterministic.
   Parent maps, offsets, `rowFaces`, dispatch records, volumes all reused.
3. Key the cached topology to the accepted owner/topology epoch
   (`candidatePowerGeneration` is the existing generation counter in
   `webgpu-octree.ts`); any ready-commit flip invalidates and rebuilds.

**Implementer must verify:** that the second and third publish sites read the
same finest face buffers as the first (candidate vs. authority aliasing across
the ready-commit flip), and that coarse-phi conditioning touches only the face
fields listed above. If conditioning can retire a face (open fraction → 0),
refresh must preserve it as a zero-coefficient face rather than drop it —
topology stays fixed within the epoch.

### WP4 — Cooperative rebuild of the hierarchy singletons (~13 ms → ~1–2 ms)

For the one remaining build per epoch, convert each transition's three
singletons into **one fused single-workgroup (256-lane) kernel** with
`workgroupBarrier()`/`storageBarrier()` between phases:

- Clears (directory, scratch, control) strided across lanes.
- Parent discovery: keep **encounter-order parent IDs** by processing fine rows
  in fixed chunks — lanes precompute parent keys and hash probes in parallel;
  a short serial commit loop (or lane-0 walk over precomputed keys) assigns
  IDs in fine-row order. The serial part shrinks to ~1,152 directory commits
  with all address math done in parallel. (Full sort-unique-scan machinery is
  over-engineering at 4K rows; a cooperative workgroup gets determinism from
  fixed lane assignment.)
- Face filter and CSR count/prefix/scatter: lane-parallel with workgroup scans;
  scatter order kept deterministic by having each lane own a contiguous face
  range and using scan-derived cursors instead of atomics.

This also collapses 17 dispatches per publish (1 extract + 4×4) to 5. Levels
must stay sequential (L+1 needs L), which the fused-per-transition shape
respects.

### WP5 — Replace exact limbs with fixed-order summation (contract renegotiation)

The high-risk, high-leverage package. **Land alone, behind its own D4 gate
run.** The exact radix-256 superaccumulators exist to make results independent
of scheduling/permutation under atomic deposits. Everywhere below, the
summation order is already deterministic or can be fixed cheaply, and plain
fixed-order f32 (optionally Neumaier-compensated) is equally
scheduling-independent:

1. **Restriction** (`depositRestriction`/`clearRestriction`/`finishRestriction`,
   `vcycle-gpu.ts:88–100`): build per-parent child lists once per topology
   epoch (WP3/WP4 already have `fineParents` and per-parent counts in
   `scratch`; an exclusive scan + scatter yields `childOffsets`/`childList`
   sorted by child row ID). Restriction becomes one lane per coarse row
   serially summing its children in list order — deterministic, atomic-free.
   Deletes `restrictionPartials` (the 147,456-word buffers), both clear paths,
   and the `SIGNED_RADIX_256_F32_MAX_TERMS` capacity couplings at
   `hierarchy.ts:102` and `vcycle-gpu.ts:155`.
2. **Row operator `image()`** (`vcycle-gpu.ts:42`): incidence order is fixed —
   backend CSR is insertion-sorted (`backend.wgsl.ts:453–461`) and hierarchy
   CSR is built in deterministic face order (`hierarchy.wgsl.ts:158`). Replace
   the two 36-limb private accumulators with fixed-order compensated f32.
   Cuts register pressure and per-face cost in the hottest kernel; unlocks
   splitting long coarse rows later (fixed chunk boundaries + fixed combine
   tree stays deterministic).
3. **Topology row assembly** (`sortLosassoIncidences`,
   `backend.wgsl.ts:462–478`): same replacement, same argument — it runs after
   its own sort.

**Contract notes.** The computed values change (correctly-rounded exact sum →
fixed-order rounded sum), so D4 baselines must be re-blessed. Two standing
warnings from prior work apply: the canonical-fold scratch is itself the
rounding barrier D4 exactness leans on — coordinate this change with that
contract rather than assuming limb-removal is transparent; and a disappointing
first A/B here is a lead, not a verdict. Keep `webgpu-exact-f32-reduction.ts`
for any site where order genuinely cannot be fixed; after this package there
should be none in the Losasso solve path.

### WP6 — Fuse the sub-L0 V-cycle into one dispatch (~5–8 ms → ~2–3 ms)

Depends on WP5's restriction rewrite (or on porting the exact deposit into the
fused kernel — don't; sequence after WP5).

Levels L1..L4 hold at most a few hundred live rows. One 256-lane workgroup,
lanes striding rows, can execute the entire sub-L0 schedule — restrict into
L1, descend (2 pre-smooths, residual, restrict per level), 8 bottom sweeps,
ascend (prolong, 2 post-smooths per level), prolong into L0 — using
`workgroupBarrier()` + `storageBarrier()` between phases, because a single
workgroup's barrier **is** a global barrier for its own writes. Correctness
does not depend on live count (worst case 4,096 rows / 256 lanes = 16 rows per
lane of bounded extra work), so no host-side live-count decision is needed;
fuse levels ≥1 unconditionally and keep L0's smooths/residual/copies as
ordinary wide dispatches. This takes the 56 dispatches per V-cycle to ~8 and
the 952 across the solve to ~140, and it makes a retired iteration genuinely
one empty dispatch. Ping-pong buffer choreography (`xA`/`xB`, `:242–246`)
moves inside the kernel as index flips between barriers.

## Sequencing, validation, and expected trajectory

| Step | Change | Risk | Expected label deltas |
|---|---|---|---|
| 1 | WP1 (gates, live clears, live dispatch) | Low | V-cycle 61→~18; tail ~free |
| 2 | WP2 (topology decomposition) | Low | Topology 18.5→<1 |
| 3 | WP3 (once-per-epoch publish) | Medium (aliasing verify) | Hierarchy 38→~13 |
| 4 | WP4 (cooperative build) | Medium | Hierarchy →~1–2 |
| 5 | WP5 (fixed-order sums) — own gate run, re-bless D4 | High | V-cycle →~6–8; topology assembly cheaper |
| 6 | WP6 (fused sub-L0 cycle) | Medium | V-cycle →~4–6 |

Validation per step:

- Fresh xctrace capture via the existing lane
  (`tools/profile-mini-dam-xctrace.ts` + `tools/xctrace-frame-report.ts`) into
  a new dated `artifacts/xctrace-losasso-*` directory; compare label tables,
  not the wall — and never compare walls across tripwire modes (`failfast`
  alone costs ~27%).
- D4 / dry-identity gate after every step; steps 1–4 must be bit-identical
  (they change scheduling and lane counts but no summation order — that
  invariant is the point of the deterministic designs above). Step 5 is the
  only intentional numeric change.
- The mini-dam lane is nondeterministic in known benign ways (see the
  dam-248 corridor-island finding); use the gate's counters, not eyeballed
  fields, to judge steps.

What I did **not** verify and the implementer should before WP3: the exact
buffer wiring of the three publish call sites across the ready-commit flip
(`webgpu-octree.ts` ~4383 / 4861 / 4865+), and the precise field set touched by
`losassoCoarsePhi` conditioning. Everything else cited above was read directly
from the current working tree.

## What NOT to do (the naive readings)

- Don't chase occupancy on the wide kernels, or the irregular-access story
  (CSR gathers, owner-page pointer chasing, `findRow` binary search,
  divergence in `conditionLosassoFaces`). Real, but not first-order until the
  singleton/capacity/dispatch structure above is gone. Re-profile after WP6
  before spending anything there.
- Don't build GPU radix-sort/unique pipelines for 4K-item problems; a
  cooperative workgroup with scans is simpler, deterministic, and sufficient.
- Don't parallelize the 3,616-face hash insertion; parallelize the clears
  around it.
- Don't treat exact summation as immovable — the contract is determinism, and
  fixed ordering satisfies it — but equally don't slip WP5 in with other work;
  it changes bits and must be blessed alone.
