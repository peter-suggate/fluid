# Losasso advance 10x — structural handoff

**Date:** 2026-08-05 (v2 — supersedes the same-day dispatch-centric draft)
**Capture:** `artifacts/xctrace-losasso-speedup-report-2026-08-05/report.html`
(symmetric-expansion lane, first advance, `npm run profile:losasso-d4-xctrace`)
**Baseline:** 98 ms untraced per advance → target ~10 ms.
**Predecessor:** `docs/losasso-gpu-speedup-handoff.md` (WP1–WP6; WP1–WP5
landed — V-cycle 61→19 ms, hierarchy 38→3.8, topology 18.5→4.5).

## Implementation closeout — 2026-08-05

This implementation pass is complete. It used 50 symmetric-expansion steps
as the steady-state benchmark instead of waiting for the full gate. The
working-tree baseline at the start of the pass was **56.38 ms/advance**; the
best retained run is **47.80 ms/advance** (a 15.2% wall-time reduction), with
a subsequent shipping-control run at 48.00 ms. The benchmark shape is now
506.4 dispatches, 306 indirect dispatches, 39.14 compute passes, and 243
MGPCG iterations per advance.

The symmetric-expansion D4 check remained clean after each retained change:
exact topology, volume, and pressure-diagonal agreement; velocity max error
`2.98e-8`, pressure max error `0.000610`, RHS max error `0.000101`, and zero
validation or tripwire failures.

### Retained structural changes

| Area | What landed | Result |
|---|---|---|
| Solve schedule | Fused sub-L0 is the default. Factor-one cooperative reduction drains are selected from the compact coarse-row bound, independent of the fine level-set factor. | Removed 32 dispatches and 64 pass boundaries from the affected advance shape; 50-step comparison improved from 50.88–53.54 ms to a best 47.80 ms. |
| Solve storage | V-cycle vector arenas are packed contiguously by hierarchy bounds rather than every level striding by `rowCapacity`. | Approximate arena footprint fell from 320 KiB to 100 KiB without changing D4 results. |
| Topology/quiescence | Structural grading can quiesce; the retained directory, exact GPU row topology, and epoch-local logical face-directory prefix are carried across unchanged epochs. Migration snapshots and migration launches are live-sized. | Full structural work is no longer unconditionally coupled to every steady advance. |
| Coarse phi/hierarchy | Stable coarse-phi publication and transported-field refresh reuse retained GPU state. Exact-topology hierarchy reuse now skips row/face/CSR reconstruction and interleaves a field-only coarse-face refresh between levels. | The full hierarchy rebuild becomes an epoch-change path; the moving symmetric scene often changes topology, so this was correctness-neutral rather than a claimed steady benchmark win. |
| Fine transport/derived state | Fine transport is live-sized; fine-summary parent work is fused; volume correction can predicate later rounds and use the single-reduction shortcut where valid. | Deletes capacity-shaped work and unnecessary follow-up work while preserving the numeric contract. |
| Host bind churn | Backend and hierarchy publishers cache bind groups by pipeline and exact buffer bindings; caches are cleared at destruction. | Removes repeated Dawn bind-group construction. Wall-time effect was inside run-to-run noise, so no standalone speedup is claimed. |

SP1 is complete. SP2 and SP4 have substantial retained pieces but are not the
fully generalized epoch scheduler and compact-u16 representation described
below. SP3 is partial, SP5 has bind caching but not a single-submission
advance, and SP6 was not attempted.

### Rejected experiment and remaining evidence

A topology-page-delta-driven summary-mip refresh was implemented and tested,
then fully reverted. The topology changed-key list is not a complete signal
for fine values dirtied by post-transport redistance: D4 remained within its
tolerance, but the solve graph changed (535.4 dispatches, 103 passes, 275
MGPCG iterations) and the 50-step wall regressed to 53.54 ms. A future delta
summary path needs an explicit post-redistance value-dirty signal; it must not
reuse the topology delta as a proxy.

The final clean first-advance capture is at
`artifacts/xctrace-losasso-d4-frame/report.html` (summary:
`artifacts/xctrace-losasso-d4-frame/summary.json`). Its largest labels were
hierarchy geometric publication (12.04 ms), full fine seeding (11.14 ms),
and fine transport (7.86 ms). The first two are cold/epoch costs in this
capture—fine seeding averaged only 0.02 occurrences per advance in the
50-step audit—so fine transport is the remaining recurring large label.
This is evidence for future work, not part of this closeout.

## Why this is a structural program, not a dispatch program

The v1 draft of this handoff prescribed making the existing frame's links
cheaper: fuse the label chains, live-size the dispatches, trim the CG loop.
That accepts a false premise — that every advance must re-derive the world.
It must not. A 4 ms step at these scales moves the interface less than one
fine cell; the topology epoch changes rarely; the live problem is 1,152
coarse rows / 3,616 faces and a band-limited 128×64×128 fine grid. Yet every
advance currently runs full topology publication, hierarchy publication,
2–3 coarse-phi exchanges, band reseeding, a summary-mip rebuild, and three
full volume reductions — ~38 of the 78 attributed GPU ms reconstructing state
that is overwhelmingly identical to last advance's, through capacity-sized
structures that are ~30–70× larger than the data they carry.

Three structural principles replace the package list:

1. **Maintenance, not reconstruction.** The steady-state advance is
   transport + solve + projection plus *delta* maintenance of derived state.
   Full publication happens only when the topology epoch actually changes,
   and the "did it change" decision is made on the GPU.
2. **A static, GPU-resident schedule.** The host encodes the same advance
   every time. All data-dependence — did topology change, did the volume
   correction converge, how many pages are live — lives in GPU-written
   indirect dispatch records and kernel predicates, never in host branches,
   readbacks, or mid-advance fences.
3. **Small is fast.** State is sized to the live problem and packed. The
   entire coarse problem (rows, faces, CSR, solve vectors) is ~100–200 KB
   when packed — it should be cache-resident across the whole advance.
   Capacity-shaped structures are not just wasted lanes; they are wasted
   bytes, wasted clears, and cache-hostile probes.

Dispatch-count and fence-count reductions then fall out as consequences.
The old rule stands: never evaluate by occupancy — the only high-ALU labels
in the capture are exact-limb reductions burning insurance, not physics.

## Measurement caveats (read before comparing numbers)

- **The capture was heavily contended:** Google Chrome Helper used 209 ms of
  GPU inside the window; only 30.7% of our GPU time was uncontended. Absolute
  per-label times are inflated; the 33 ms / 20 ms timeline gaps are partly
  preemption. Re-capture on a quiet machine before baselining anything.
- **The tree has post-capture edits.** The capture shows per-level
  `Recompute direct fine-summary parents level N` labels (3.70 ms for level 2
  alone); the current tree has already replaced them with a single indirect
  `recomputeFineSummaryAllParents` and a delta ABI
  (`lib/webgpu-octree-fine-levelset-summary-direct.ts:412–418`). Re-baseline
  before attributing time to the summary chain.
- The traced 292-encoders/295-blits shape is an isolation-mode artifact
  (`FLUID_GPU_ISOLATE_PASS_ENCODERS=1`). The untraced structure is the
  command audit's 14 encoders. Audit counters cover the whole run
  (construction + 1 advance), not the advance alone.
- Traced/untraced distortion is 2.13x. Compare label tables, never walls;
  never across tripwire modes.

## What the capture says (contention-inflated; shares are the signal)

78.15 ms attributed GPU; 87.9 ms GPU idle (traced). By subsystem:

| ms | share | subsystem | steady-state necessity |
|---|---|---|---|
| 19.10 | 24.4% | V-cycle preconditioner (17 calls, 0.06% occ) | needed — but fused form exists, disabled |
| 10.95 | 14.0% | Volume measure/correct ×3 full reductions | delta-carry + cadence should replace |
| 9.28 | 11.9% | Fine transport (capacity-shaped ×2) | needed — at live size |
| 6.27 | 8.0% | Coarse phi ×3 rounds | epoch-change only (steady: refresh) |
| 6.18 | 7.9% | Summary mips (22 labels at capture) | delta maintenance only |
| 5.51 | 7.0% | Topology publish | epoch-change only |
| 5.07 | 6.5% | CG iterations + projection | needed |
| 5.02 | 6.4% | Fine band seeding | delta maintenance only |
| 3.95 | 5.1% | Hierarchy publish + refreshes | epoch-change only |
| 3.07 | 3.9% | Extension band | epoch/delta |
| 3.67 | 4.7% | Grading + other | epoch only |

Untraced command audit (whole run): 1,221 dispatches (1,007 indirect),
110 passes, 1,029 bind-group creations, 14 submissions, 7 completion fences.

## The delta architecture is already half-built (verified anatomy)

This program mostly finishes machinery that exists. Line refs current as of
2026-08-05 working tree:

- **Change detection exists:** `Compare topology-tile refinement signatures`
  (`lib/webgpu-octree.ts:4108`) already computes whether refinement changed.
  Cost in capture: 0.26 ms. It gates nothing structural yet.
- **Transport already emits an interface delta:**
  `Structured fine transport interface delta`
  (`lib/webgpu-octree-fine-levelset-transport.ts:374–375`, layout at `:185`).
- **The summary module already consumes a changed-keys page delta with a
  warm/cold mode select** (`summary-direct.ts:553–556`:
  `let mode=select(2u,1u,cold)` — warm path sizes work by `pageDelta[0]`,
  cold by the full worklist). The structural question is only why cold/full
  still wins in practice (generation bump per advance? capacity-shaped
  recompute dispatch?) — answer it with the re-baseline, then make warm the
  steady case.
- **The volume ledger is documented as a delta carry**
  (`lib/webgpu-octree-fine-levelset-volume.ts:293` — "remain an exact delta
  carry") and already has a cadence hook (`fineLevelSetVolumeCadence`,
  `:313–320`) and live-lane indirect sizing (`:356–358`). What it does *not*
  yet do: skip the second and third full reductions (`:360–398` unrolls
  measure → apply → re-reduce → apply-residual → re-measure unconditionally).
- **The hierarchy consumer is already refresh-shaped:**
  `encodeHierarchyCoefficientRefresh` (`lib/webgpu-octree.ts:4863`, `:4874`)
  — WP3's field-refresh-not-republish pattern, live in-tree. The coarse-phi
  *producer* is not: all three encodes (`:4399` ready-commit, `:4860`
  first-advance bootstrap, `:4871` post-volume-correction) run the full
  restrict/ghost/finalize/volume ladder
  (`lib/webgpu-octree-losasso-coarse-phi.ts:194`).
- **Epoch counters exist** (`candidatePowerGeneration` in
  `webgpu-octree.ts`), used today to key hierarchy topology caching.
- **The fused solve exists, disabled:** fused sub-L0 V-cycle behind
  `FLUID_LOSASSO_FUSED_SUB_L0=1`
  (`lib/webgpu-octree-losasso-vcycle-gpu.ts:258–263`), 9 dispatches per
  application vs ~47, pending a D4 re-bless.

And the capacity-shaped state this program shrinks:

- `faceDirectory`: 262,144 slots (2 MiB) for 3,616 live faces
  (`finishLosassoPublication`, predecessor doc §Topology) — a 2× live-sized
  power-of-two hash is ~8K slots / 64 KB.
- V-cycle vector arenas stride by `rowCapacity` = 4,096 while 1,152 rows are
  live (`vcycle-gpu.ts:271`) — 3.5× footprint on every solve vector, ×4
  vectors × 5 levels.
- Coarse-phi directory probes 4-word entries with up to 32 linear probes
  (`coarse-phi.ts:76–79`).
- Fine transport dispatches `pageCapacity` workgroups twice
  (`fine-transport.ts:192–193`) though the live count is GPU-resident in
  `worklist[1]`.
- Volume `reductionScratch` is capacity-sized to preserve exact addition
  order (`volume.ts:352–355`).
- Row/face indices are u32 everywhere; live counts fit u16 after per-epoch
  compaction; CSR incidences are ≤24 per row.

In-repo precedent that packing pays: `c5061fc` "Pack fine level-set pages",
and the packet-BVH render lanes.

## Work packages

SP0 is hygiene. SP1 is a gate run. SP2–SP3 are the core structural change.
SP4 packs the state. SP5 removes the host from the loop. SP6 renegotiates
numeric contracts and lands each piece alone.

### SP0 — Clean re-baseline

Uncontended capture (browser closed) of the current tree (it already differs
from the 08-05 capture in the summary chain). Record per-label tables and the
per-advance (not per-run) sync/dispatch audit via `FLUID_AWAIT_EVERY_STEPS`
deltas. Everything below is judged against this table.

### SP1 — Flip the fused sub-L0 solve (finished code, one gate run)

Run the D4 gate with `FLUID_LOSASSO_FUSED_SUB_L0=1`, re-bless, delete the env
gate. This is structural in kind — the whole sub-L0 cycle as one cooperative
kernel — and it is already written. Expected: V-cycle 19 → ~4–6 ms. Extend
the same shape upward later (CG iteration tail between wide operator
applications) once SP2–SP4 have moved the bigger walls.

### SP2 — Quiescence: make "nothing changed" the fast path

The advance gets an explicit GPU-computed classification, derived from the
existing refinement-signature compare plus the transport interface delta:

- **Epoch unchanged (steady state, the overwhelming majority):** skip
  topology publish, hierarchy publish, coarse-phi ladder, grading, extension
  band, seeding scan. Field-dependent consumers run *refresh* forms
  (coefficient refresh exists; SP2 adds the coarse-phi field refresh —
  gather phi/ghost values through the retained epoch mapping instead of
  re-running restrict/ghost/finalize).
- **Epoch changed:** run today's full publication path, unchanged.

The skip must be GPU-side — the signature kernel writes zero/nonzero
workgroup counts into the publish passes' indirect records; the host encodes
both paths every advance. No readback, no host branch.

**Implementer must verify:** the refinement signature actually covers every
input the skipped passes depend on (face fields vs topology; the predecessor
doc's WP3 caveat about conditioning retiring a face — a field refresh must
preserve retired faces as zero-coefficient, never drop them); and that
volume correction can move only phi values, never rows, within an epoch.

Expected steady-state deletion: topology 5.5 + hierarchy 3.95 + phi ~4 (of
6.3) + seeding ~4 + band ~2 + grading ~2 ≈ **~21 ms of the 78 becomes
epoch-only**.

### SP3 — Delta-driven derived state (finish the half-built machinery)

For the state that must track every advance even in steady state:

1. **Summary mips:** the warm changed-keys path exists; find why it doesn't
   win (SP0 will show whether the recompute dispatch is delta-sized or
   full/capacity), and make warm the steady case — touched pages and their
   ancestor chain only. A quiescent-interface advance should pay
   ~proportional-to-delta, not 6 ms.
2. **Volume ledger:** steady state keeps the ledger by *carry* — transport
   commit already computes per-page volume change; accumulate the delta and
   run the full measurement at cadence / on epoch change only (hook exists).
   The correction chain's rounds 2–3 become convergence-predicated GPU-side:
   the apply kernel knows the shift magnitude; below the half-fine-cell
   bound it writes a zero indirect record for the follow-up reduction.
   Whether the carry changes summation order decides if this piece lands in
   SP6's solo gate instead.
3. **Band seeding:** consume the transport interface delta (new interface
   leaves only) instead of rescanning; the 4.11 ms
   `Seed global fine bricks from FineSeedLeaf candidates` at 0.2% occupancy
   is a full-scan symptom.

Expected: mips+seed+volume ~22 ms → ~4–6 ms steady state.

### SP4 — Small is fast: live-sized, packed state

Structure packing (bit-identical — indices, directories, layouts; **no field
value changes here**):

1. **Face directory:** replace the 262,144-slot capacity hash with a
   per-epoch directory sized to ~2× live faces (power-of-two, same
   deterministic serial insertion — at 3,616 items the insert is
   microseconds). 2 MiB → ~64 KB; the clear disappears into the epoch
   rebuild; probes become cache hits. This deletes the WP2 "parallel clear"
   approach outright — don't optimize the clear, delete the thing cleared.
2. **Solve vectors:** pack the V-cycle/CG vector arenas by per-epoch live
   row count instead of `rowCapacity` stride (`vcycle-gpu.ts:271`), and pack
   levels contiguously. The working set of the entire solve (4 vectors × ~5
   levels × ~1.5K rows × 4 B ≈ 120 KB + CSR) becomes L2-resident; combined
   with SP1's fusion the solve stops touching DRAM at all.
3. **CSR + row space:** per-epoch compacted u16 row/face id space (live
   counts are ≪65K even when capacity isn't), incidence lists packed u16,
   flags folded into spare bits. Coarse-phi directory entries packed from 4
   words to 2 (key+size share a word; value+hash-tag).
4. **Kill remaining capacity-shaped launches** where the pass survives
   SP2/SP3 (fine transport advect/commit → indirect from `worklist[1]`,
   using the volume module's existing prepare-kernel pattern).

Rebuild-on-epoch makes variable sizing safe: sizes are epoch constants, and
SP2 made epochs rare. Everything here is scheduling/layout — the gates must
stay bit-identical.

Expected: shrinks the byte cost of every surviving pass; transport 9.2 →
~2; the epoch-publish path itself gets cheaper (smaller clears, smaller
scatters). This is the package that attacks the memory-axis wall (per-item
cost inside loop bodies), not lane counts.

### SP5 — Static one-submission advance (host out of the loop)

With SP2–SP4, no per-advance host decision remains that the GPU can't make:

1. One (target; two acceptable) `queue.submit` per advance; zero blocking
   fences or `mapAsync` on the advance path. Audit the blocking sites
   (`webgpu-octree.ts:5248` onward) and classify: diagnostics-gated (leave),
   decision-bearing (move GPU-side — SP2 removed most), evidence-bearing
   (fire-and-forget like `webgpu-water-pipeline.ts:1724`).
2. Cache bind groups and encoded structure across advances keyed on
   (epoch, ping-pong index) — 1,029 bind-group creations per run is host
   time and Dawn churn for identical bindings.
3. Forensics carried from v1: explain the traced 33 ms S1→S2 and 20 ms
   seed→transport gaps *after* SP0's uncontended recapture — if they
   survive, find which `productionBoundary`/`splitProductionPhase` seam
   stalls there.

This is where the wall meets the attributed-GPU number; no honest expected
delta until SP0 reports, but the traced idle (87.9 ms) says this wall is the
same order as the GPU work itself.

### SP6 — Contract renegotiations (each lands alone, own gate run)

Collected here because each changes bits and must be blessed solo, per the
WP5 precedent:

1. **Volume ledger fixed-order live reduction** — replace the
   capacity-shaped exact tree with a fixed-order compensated reduction over
   the compacted worklist (order is already deterministic — scan-compacted,
   not atomic). Deletes the capacity-sized scratch and its clears. Run the
   dry-identity / still-scene oracles explicitly — the class-4 zero-RHS
   contract leans on this ledger.
2. **Value packing** where profitable after SP4: f16 mip summaries,
   quantized open fractions, f16 ghost inverse-distances. Only where the D4
   gate tolerates it; guided by SP0's clean numbers, never speculatively.

## Sequencing and expected trajectory

| Step | Change | Risk | Steady-state effect |
|---|---|---|---|
| 0 | SP0 clean re-baseline | — | real numbers; summary-chain re-attribution |
| 1 | SP1 fused solve gate run | Low (code exists) | solve 24 → ~8–10 |
| 2 | SP2 quiescence gating | Medium (signature coverage verify) | ~21 ms → epoch-only |
| 3 | SP3 delta maintenance | Medium | mips+seed+volume ~22 → ~4–6 |
| 4 | SP4 packing + live sizing | Medium (bit-identical required) | survivors shrink; transport 9 → ~2 |
| 5 | SP5 single-submission | Medium, forensics-led | wall converges to GPU busy |
| 6 | SP6 solo-gated renegotiations | High | ledger ~5 → ~1; further shrink |

Steady-state GPU math after SP1–SP4: transport ~2 + solve ~5 + projection
~0.5 + delta maintenance ~2–3 ≈ **~10 ms attributed**, with epoch advances
occasionally paying a (now smaller) publish. SP5 makes the wall track that
number. That is the 10x — and it comes from advances doing 5× less work,
not from the same work launched better.

Validation per step: fresh dated capture, compare label tables; D4 +
dry-identity gates every step; SP2/SP4/SP5 must be bit-identical (they change
scheduling, sizing, and layout — not values or order); SP6 pieces are the
only intentional numeric changes. The known-benign mini-dam nondeterminism
rules apply (dam-248 corridor finding): judge by gate counters.

## What NOT to do

- **Don't micro-optimize passes that SP2/SP3 delete from the steady path.**
  Fusing the 22-label mip chain or decomposing the seeding scan is wasted
  motion if the steady-state advance shouldn't run them at all. (This is the
  v1 draft's error, preserved here as a warning.)
- **Don't optimize clears of structures SP4 shrinks away** — the parallel
  directory-clear idea from WP2 dies with the capacity directory itself.
- Don't chase occupancy or ALU%; the high-ALU labels are exact-limb
  insurance, and SP6 removes the work rather than feeding it.
- Don't mix structure packing (bit-identical, SP4) with value packing
  (contract change, SP6) in one land.
- Don't build multi-workgroup machinery for ≤4K-item problems; cooperative
  single-workgroup kernels keep determinism trivial — that lesson carries
  over unchanged.
- Don't baseline against the 08-05 capture: contended, and the tree has
  already moved under it.
