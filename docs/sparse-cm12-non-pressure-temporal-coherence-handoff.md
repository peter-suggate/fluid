# Sparse CM12 non-pressure temporal-coherence handoff

**Scope.** Everything in the `adaptive-mass` (Sparse CM12) advance except the
MGPCG pressure solve. On the long dam that is now **63.5% of the frame**, so
the solve is no longer the bottleneck. This document is the measurement, the
diagnosis, and the implementation plan. Every claim carries a verdict:
**CONFIRMED** (measured on device or read in code), **ESTIMATED** (cost model,
not yet measured), or **OPEN** (settled by the work package that names it).

**Direction constraint (standing, do not re-litigate).** Structural do-less-work
before cheaper launches. The lens is "should this pass run at all in the steady
state?" and "how small can this state be?", not "how do I launch this cheaper".

---

## 1. Evidence base

Captured 2026-08-16 on Dawn/Metal from the user's *staged* tree
(`git checkout-index` copy; the worktree carried an unrelated in-flight
pressure-journal edit). Scene `sparse-cm12-long-dam-break`, 192x96x32 finest,
`timeStep: scene`, `balanced` profile, 49 hardware-timestamped advances after 8
warmup frames, medians. Probes:

- `tools/probe-sparse-cm12-stage-cost.ts` — per-phase medians, split by whether
  the frame committed any topology change.
- `tools/probe-sparse-cm12-cell-population.ts` — accepted / liquid / interface
  populations and per-frame topology churn.
- `tools/probe-sparse-cm12-apron-sweep.ts` — receiver-apron policy sweep.

`tools/probe-sparse-cm12-stage-trace.ts` remains the acceptance gate for the
partition *existing*; these are the measurement lanes. Dawn quantizes
timestamps to 65.536 us, so every figure below is a multiple of that tick and
sub-tick stages read as zero — see [[sparse-cm12-advance-partition]].

### 1.1 Where the frame goes

Median advance **21.04 ms**; the hardware phase partition closes at 21.56 ms.
Pressure solve **7.86 ms (36.5%)**. **Non-pressure 13.70 ms (63.5%).** CONFIRMED.

| phase | all | quiescent | changed |
|---|---:|---:|---:|
| One-reduction sparse MGPCG pressure solve | 7.864 | 7.733 | 7.930 |
| Coupled conservative mass + gamma + momentum transport | 1.770 | 1.442 | 1.901 |
| CM12 surface sharpening + gamma conditioning | 1.573 | 0.983 | 1.638 |
| Hysteretic resolution planning + 2:1 candidate grading | 1.573 | **1.049** | 1.573 |
| Composite pressure topology + ghost-fluid rows | 1.376 | 1.114 | 1.442 |
| Composite face preparation + oriented transport rows | 1.311 | 0.983 | 1.376 |
| Budgeted shadow-topology preparation + conservative transfer | 1.245 | **0.066** | 1.245 |
| Transport velocity extension into the sparse air band | 1.180 | 0.852 | 1.311 |
| Encode compact sparse presentation pages | 0.983 | **0.983** | 0.983 |
| Gamma diffusion row-owned snapshot iterations | 0.852 | 0.655 | 0.852 |
| Compact brick activity measurement | 0.721 | 0.590 | 0.721 |
| Composite pressure-gradient projection | 0.459 | 0.328 | 0.459 |
| Projection residual + divergence + energy receipts | 0.262 | 0.197 | 0.262 |
| Finite-volume divergence RHS + compatibility projection | 0.197 | 0.197 | 0.197 |
| Body-force prediction | 0.131 | 0.131 | 0.131 |
| Dry-brick retirement + retained-atlas conditioning | 0.066 | 0.066 | 0.066 |

"quiescent" = the 11 of 47 sampled frames that committed **zero** brick
topology changes. The three bolded cells are the whole argument of this
document: candidate transfer already collapses to nothing when there is nothing
to do (0.066), resolution planning does not (1.049), and presentation
publication does not even try (0.983, identical either way).

There is no hot kernel here. The cost is the *shape* of every stage.

### 1.2 Populations

Median over evolved frames, long dam (1,152 mutable bricks in the domain):

| quantity | value | note |
|---|---:|---|
| accepted cells | 30,640 | what every non-pressure kernel invokes |
| accepted rows | 88,072 | |
| liquid (pressure) cells | 9,984 | **33% of accepted** |
| active pressure rows | 30,753 | 35% of accepted rows |
| resident bricks | 117–130 | of 1,152 |
| bricks changing topology per frame | median 4 | 11 of 47 frames changed nothing |

CONFIRMED. Two thirds of the cells every non-pressure stage sweeps are dry
apron, and the adaptivity machinery spends 2.3 ms/frame (planning + transfer +
activity) to move a median of **four bricks**.

---

## 2. Diagnosis: three shapes of repeated work

### A. Decisions rebuilt at domain scale to move four bricks

`resolution-planning` costs **1.049 ms on frames where nothing changes**. The
chain is 13 dispatches: six brick-scaled neighbour scans (`planBrickResolution`
does 27 directory lookups per brick through `brickRequestedAsReceiver`, plus six
more through `brickDeeplyEnclosed`; `closePlannedResolution` runs three times at
six each; `validateCandidateResolution` six more), a single-lane serial
scheduler, per-brick candidate page synthesis, and two full-template shadow
worklist builds.

**The dominant term is one kernel.** `scheduleTopologyPreparation` is
`@compute @workgroup_size(1)` and walks all 1,152 bricks **twice** with three to
five atomics per brick, with **no early-out**, every frame. Ablating it (scratch
tree, body replaced with an immediate return) moves the phase from 1.049 ms to
**0.262 ms**: it is **~0.79 ms/frame, unconditionally** — 3.7% of the entire
advance, 5.7% of the non-pressure half — spent on a single GPU lane.
CONFIRMED by ablation. Every other dispatch in that stage is brick-scaled and
insensitive to the frozen-topology drift the ablation introduces, so the
attribution is sound; the other phases in the ablated run moved because the
simulation state itself diverged, and are not compared here.

`deferDynamicTopologyPublication` and `validateAndCommitShadowTopology` have the
same single-lane whole-domain shape but both early-out on the quiescent path, so
they are not currently paying. They are the same latent defect.

**Second finding: the epoch machinery is switched off.**
`SPARSE_CM12_ACTIVITY_POLICY.topologyCadenceSteps` is **1**
(`webgpu-sparse-cm12-resident.ts:91`), so `advanceActivityClock` sets
`activity[5]` on *every* frame and every frame is a topology epoch. All the
hysteresis built around epochs — promote/demote epoch counters, the quiet/hot
ladders — is running at frame rate. The `frontLookaheadSteps` hint in
`method.ts:190` still reads "Four matches the default topology cadence", which
is stale against a default of 1. CONFIRMED (code).

### B. Publication that ignores what changed

> **REFUTED by measurement — see WP2.** The reasoning below is sound and the
> gate built from it is exact, but it counts pages rather than work: the pages
> it identifies as redundant are the dry ones, which were already free. Read WP2
> before acting on anything in this section.

`publishSparseLevelSet` dispatches
`globalFineLevelSetSource.plan.maximumResidentBricks` workgroups — **capacity
shaped, not live-sized** — and republishes every page every frame: **0.983 ms,
bit-identical cost on quiescent and changed frames**. With 117–130 of 1,152
bricks resident and a median of four changing, nearly all of that is
recomputing texels that did not move. CONFIRMED.

The signal to gate on already exists and is already computed.
`measureBrickActivity` reduces a per-brick fixed-point density sum and three
moments and *compares them against last frame's stored values* to form its
`temporal` score. Nothing consumes that as a republish gate.

A strictly better gate is available. Density is double-buffered, so at the end
of frame *t*, `sourceDensity()` **is** the buffer published at *t-1*.
`finalizeSharpening` can raise a per-brick dirty bit when the value it writes
differs from the source value — one comparison on data already in registers,
exact rather than thresholded.

**Trap.** Coarse pages (`scale > 1`) read *neighbouring* bricks through
`restrictedPresentationDensity`, so the dirty bit must be dilated by one brick
ring before it may gate a coarse page. The 27-bit support mask already carried
in the activity record is the natural carrier. A gate that misses this freezes
coarse pages next to a moving fine brick.

### C. Band work dispatched over the whole accepted set

> **PREMISE REFUTED — see WP3.** The band is 43.5% of the accepted set, not a
> thin subset of the 33% liquid fraction. Read WP3 before acting on anything in
> this section.

Every non-pressure physics kernel goes through `dispatchAccepted`, covering all
~30,640 accepted cells or ~88,072 accepted rows. The pressure solve does not: it
compacts a stable-ID liquid worklist once per epoch (classify → count →
finalize → compact, then a copied indirect count) and every PCG dispatch
consumes that. The compaction pattern is proven, in-tree, and used by exactly
one consumer.

The kernels that would benefit are the ones that are *already* band operations
in disguise:

- `prepareSharpeningField` / `scatterSharpeningMass` early-out on `rho == 0` and
  on `rho > CM12_LIQUID_ISOVALUE`; they do real work only where
  `0 < rho <= 0.5`, which is the interface itself.
- `extrapolateTransportVelocity` copies-and-returns on a valid cell and writes
  zeros beyond the reach of the sweeps — and it is dispatched **eight times**
  over the full accepted set to fill a band that is at most eight cells thick.

This is where temporal coherence proper enters. Under CFL the interface moves at
most about one fine cell per step, so `band(t+1) ⊆ dilate(band(t), 1)`: the band
is *maintained*, not reclassified. The classify pass then visits last frame's
band plus one ring instead of the whole accepted set, and the compaction cost
amortizes across the ~12 dispatches that consume it.

---

## 3. Work packages

Ordered by (evidence x prize) / risk.

### WP1 — Parallelize `scheduleTopologyPreparation`

**Prize: 0.79 ms/frame unconditional (5.7% of non-pressure). CONFIRMED.**
Risk: low. Not a physics change.

The first loop is embarrassingly parallel: each iteration clears its own brick's
word, tests a predicate on its own record, and increments a counter. Replace it
with a 64-lane brick-scaled dispatch using `atomicAdd` for the urgent/ordinary
counts.

The second loop — the rotating `prepareBricksPerFrame` window — *is*
order-dependent and must stay deterministic. It does not need to walk 1,152
bricks to find at most 64 of them. Have the parallel pass compact pending
*demotion* candidates into a small list (there are 0–75), then select by rank:
candidate `b` is selected iff the number of candidates with a smaller
`(b - cursor) mod brickCount` is below budget. That is a rank over ≤~100 items,
one workgroup, deterministic, and produces the identical selection.

**LANDED.** `scheduleTopologyPreparation` is now `@workgroup_size(64)`: both
walks are 64 lanes striding the ranges the single lane used to, the rotating
coarsening window selects by exclusive prefix over the *rotated* order (the same
set the cursor selected), and a `storageBarrier` orders the clear against the
schedule, which the single-lane version never needed. The chunk loop
deliberately never breaks early — the running count lives in workgroup memory
and is non-uniform to WGSL, so making it a loop condition would put the scan's
barriers in non-uniform control flow.

**Measured (60 frames, long-dam):** the planning phase falls **1.573 → 0.786 ms**
overall and **1.114 → 0.262 ms** on quiescent frames — landing exactly on the
0.262 ms floor the stub ablation predicted. Advance 22.02 → 20.05 ms. CONFIRMED.
The rest of the −1.64 ms non-pressure delta is spread thinly over unrelated
stages and reads as run-to-run drift, not a WP1 effect; only the planning phase
is attributable.

**Acceptance (met, as revised by §3.5):** identical committed-brick sequence
against the baseline over frames 1–117 of 120, where the baseline reproduces
itself only over 1–92; 505 total committed bricks against the baseline run's
505.

### WP2 — Dirty-brick presentation republish — **BUILT, MEASURED, REVERTED**

The cost model in this document was wrong and the implementation bought
nothing. Recorded here so nobody rebuilds it.

**What was built.** Activity word 39 became a presentation stamp holding the
descriptor publication last wrote for a brick (dry-constant, or wet at rung R)
plus two dirty bits. `stampPresentationPages` raised the self bit when the
descriptor changed or when the brick's two density banks disagreed cell for
cell — exact, because only the destination bank is written during an advance,
so the source bank still holds precisely what publication read last frame.
`dilatePresentationDirt` carried the bit one brick ring for pages that sample
coarser than their own cells; macro leaves were exempted and always republished
because their 2*span stencil reaches further than one ring can prove.
`invalidatePresentationStamps` dropped every stamp on the injection and initial
paths, which write density outside the advance parity discipline.

**The gate was exact.** `probe-wp2-skip-equivalence` (same-run oracle, per §3.5)
built publication so it computes every page every frame and, where the gate said
"clean", compares against the atlas instead of writing: **0 mismatched lanes out
of 99,728,896 skipped, over 200 long-dam frames**, skipping **84.4%** of
publication lanes. CONFIRMED.

**And it saved nothing.** Non-pressure **10.093 ms both arms**; advance 18.481 →
18.678; the publication phase itself went *up*, 0.852 → 0.917 ms, which is the
two added dispatches. CONFIRMED.

**Why.** Stubbing publication to an immediate return for every page drops the
phase to 0.0655 ms — one Dawn tick, the measurement floor. So all 9,216 pages'
work is 0.85 ms, and skipping 84.4% of them removed none of it: the pages the
gate skips are the **dry** pages, which write one constant and were already
free. The 0.85 ms lives entirely in the ~1,000 wet pages, whose `scale > 1`
cache fill does `restrictedPresentationDensity` hash lookups — and a wet page on
a running dam changes every frame by construction. **Temporal coherence has no
purchase on the only pages that cost anything.** The document's original premise
— "117–130 resident of 1,152 pages dispatched, so ~89% is redundant" — counted
pages, not work.

**If it is ever revived** it would be for settled-scene idle cost, not for a
running scene, and it should be justified by a measurement of the *wet* page
population that goes unchanged, not by the resident fraction.

### WP3 — Band worklists for sharpening and velocity extension — **PREMISE REFUTED**

This work package assumed a band that is "a strict subset of the 33% liquid
fraction". It is not. Measured branch populations inside
`prepareSharpeningField`, per frame on the long dam (60 frames, medians):

| population | cells | share of accepted |
|---|---:|---:|
| deep liquid, rho > 0.5 — writes zero | 9,319 | 29.9% |
| empty, rho < 1e-5 — writes -rho | 9,002 | 28.9% |
| **diffuse band, 1e-5 < rho < 0.5 — does the full walk** | **13,545** | **43.5%** |

CONFIRMED. The CM12 interface band is **43.5% of the accepted set**, not a thin
shell. `CM12_DRY_CELL_THRESHOLD` is `1e-5`, so "dry" means *numerically empty*,
and every air cell carrying trace density is inside the band.

**The cheap version was tried and is a null.** `prepareSharpeningField` computes
`sharpeningStats` — the composite incidence walk — before the branches that
discard it, so both the deep-liquid and empty cases pay for a walk they throw
away. Hoisting the branches ahead of the walk is **bit-exact** (0 mismatches out
of 1,923,028 cell evaluations over 60 frames, same-run oracle) and skips 58.8%
of cells. It measured **1.0486 → 1.1141 ms** — one Dawn tick *worse*, i.e. no
change. With the in-band population at 43.5%, essentially every 64-lane wave
contains at least one in-band cell, so no wave is ever retired early and lane
predication saves nothing. Reverted.

A compacted band worklist would fix the divergence, but it can retire at most
57% of one 1.1 ms stage while the compaction pass is itself O(accepted). Not
worth building at this ratio. If the band population ever drops well below the
wave width's reciprocal, revisit.

### WP4 — Seed velocity extension from the previous frame

**Prize: up to ~0.8 ms** (eight sweeps to two). ESTIMATED.
Risk: medium — this one *does* change values.

The band moved at most one cell, so last frame's extended velocity is within one
sweep of correct. Needs a quality gate, not an identity gate: front position,
volume conservation and post-projection divergence over a long run. Do this
after WP3, whose band worklist it needs anyway.

### WP5 — Restore a real topology cadence

**Prize: up to ~2 ms on non-epoch frames. OPEN.**
Risk: this is a fidelity decision, not a performance one — it belongs to Peter.

With `topologyCadenceSteps: 1` every frame is an epoch. Raising it to the
designed 4 would let the whole planning and activity block run at a quarter
rate. Whether cadence 1 is deliberate (a response to a front-freeze regression)
or vestigial is **OPEN** and should be answered from history before touching it.
Fix the stale `frontLookaheadSteps` hint either way.

---

## 3.5 The lane is not run-to-run deterministic — read before writing an oracle

Measured while validating WP1. Two runs of the **same binary** on the same scene
produce the same topology decisions for **frames 1–92** and then diverge; over
120 frames they disagree on 9 of them, and the total committed-brick count comes
out 505 against 507. CONFIRMED.

That kills the acceptance criterion this document originally wrote for WP1–WP3
("byte-identical against the unconditional arm over 200 steps"): **no** change
can meet a cross-run identity bar the baseline cannot meet against itself. The
criteria below are the replacements, and the shape of the fix generalises:

- **Prefer a same-run oracle.** Do not diff two runs. Run one binary that
  computes both answers and compares them on device. WP2's oracle
  (`probe-wp2-skip-equivalence`) does exactly this: the instrumented build
  computes every page every frame and, where the gate said "clean", compares
  against what is already in the atlas instead of writing. No second run, so no
  divergence to explain away.
- **Where a cross-run diff is unavoidable, bound it against the baseline's own
  self-diff.** WP1 was accepted because it matched the baseline over frames
  1–117 while the baseline matched *itself* only over 1–92 — it tracks the
  reference more closely than the reference tracks itself, which is the most any
  change in this lane can be asked to show.

The divergence source is not identified and is **OPEN**. It is not caused by
WP1: it reproduces with WP1 absent.

---

## 4. Ruled out

**Receiver-apron over-provisioning.** The obvious hypothesis — that the 3.07:1
accepted:liquid ratio is dry apron sized by `frontLookaheadSteps`, and that
shrinking it scales every non-pressure stage down — is **false**. Sweeping
`frontLookaheadSteps` 4 → 2 → 1 left the accepted population **byte-identical**
(30,640 cells, 88,072 rows in all three arms) and the liquid front at x=61 in
all three. CONFIRMED.

The ratio is not a policy knob: it is the 8³ free-surface floor. Any brick
holding an interface is refined to 512 cells, of which only a fraction are
liquid. Non-pressure cost has to come down structurally — which is what WP1–WP3
do — and not by retuning the apron. Do not re-run this sweep.

---

## 5. Result

| | before | after WP1 | measured |
|---|---:|---:|---:|
| pressure solve | 7.86 | 8.19 | unchanged within noise |
| non-pressure | 13.70 | 11.86 | planning phase −0.79 attributable |
| advance | 21.04 | 20.05 | CONFIRMED |

**Only WP1 landed.** WP2 was built, proven exact, measured at zero, and
reverted. WP3's premise was refuted before it was built. The cost model that
produced the original estimates counted *dispatched items* rather than *work*,
and was wrong in both places it was checked.

### 5.1 Where the non-pressure time actually is

Stubbing five kernel bodies immediately after the accepted-worklist lookup —
the three sharpening kernels and both extrapolation sweeps, so the indirect
dispatch and the worklist indirection still happen but the body does not —
moves the two phases to their floor:

| phase | live | body stubbed |
|---|---:|---:|
| CM12 surface sharpening + gamma conditioning | 1.114 | 0.066 |
| Transport velocity extension into the sparse air band | 0.918 | 0.131 |

CONFIRMED. **Dispatch and indirection are free; the bodies are the cost.** That
rules out the whole "fewer, larger dispatches" family for this lane, and it
confirms the standing direction — do less work — is the right axis. The problem
is that the work is spread over a genuinely dense working set.

### 5.2 The remaining lever is the accepted-cell count

Every non-pressure stage is O(accepted cells) and they all land at 0.85–1.5 ms
over 31,352 cells. Nothing in this document found a subset of those cells that
is both cheap to identify and large enough to matter:

- the apron is not over-provisioned (§4),
- the interface band is 43.5% of the set (WP3),
- 28.9% of accepted cells hold *numerically nothing* (rho < 1e-5) and are still
  walked by every stage.

That last number is the open lead. Those ~9,000 empty cells are not policy —
residency is brick-granular at 8^3, so an empty cell sharing a brick with an
interface cell cannot be dropped. **Cutting non-pressure cost structurally means
cutting the residency granularity, not the policy**, and that is a design
decision for Peter rather than a work package this document can specify.

### 5.3 What is still worth trying

**WP4 (seed velocity extension from the previous frame)** is now the most
promising remaining item and is unaffected by everything above: it does not
depend on a small band, only on the CFL bound that the front moved at most one
cell. It changes values, so it needs the quality gate described below rather
than an identity oracle. **WP5 (topology cadence)** remains a fidelity decision.
