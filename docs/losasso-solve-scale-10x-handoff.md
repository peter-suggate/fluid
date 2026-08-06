# Losasso solve at scale — the true 10x handoff

> **Status 2026-08-06 — S1, S2.0 and S2.1 landed.** See "What landed" at the
> foot of this document for measured results, the one prediction the program got
> wrong, and what S2.2 now has to be about. Headline: 128³ first frame
> **1,957 → 400 ms (4.9×)**; the V-cycle preconditioner turned out to be
> **inert on every scene**, not just at 128³.

**Date:** 2026-08-06 (successor to `docs/losasso-coarse-band-10x-handoff.md`,
whose B0/B1/B2/B3a/B5 landed in `e839807`)
**Captures:**
- Small lane: `artifacts/xctrace-losasso-coarse-band-after-2026-08-06/`
  (symmetric-expansion 32×16×32, fine-factor 1) — **14 ms untraced** (was 36).
- Large lane: `artifacts/xctrace-high-resolution-dam-128-after-2026-08-06/`
  (high-resolution-dam-break 128³, 125,067 rows, leaf 32) —
  **2,078 ms untraced**, 90% GPU-bound.

**Verdict:** the previous program's bets worked where they aimed
(`predictSummaryCells` 19.15 → 0.30 ms; dynamics 3.2 → ~1.0 ms) and the small
lane improved 2.6×. But the frame's center of gravity was never where the
32³ profile said it was. At 128³ the pressure solve is **87% of the frame**,
and 76% of the whole frame is **two serial "finish" kernels running one
thread each**. The true 10x is a solve program.

## Where we went wrong (three honest mistakes)

1. **We profiled only the guide lane.** The standing rule ("symmetric-
   expansion is the guide lane — never the big scenes") was ignored at
   baseline time. At 1,152 rows the solve takes the combined-drain tier
   (one cooperative workgroup, no partials), so the serial-finish
   architecture was invisible in every capture the program was built on.
   The 128³ capture falsifies the whole cost model in one table.
2. **We treated the exact-reduction finish as O(1).** It is
   O(partials × limbs × scalars), single-threaded, twice per iteration.
   Nobody priced it because at 32³ it never appears.
3. **We treated iteration count as an encode-shape problem.** B3a trimmed
   the encoded budget; nobody asked whether *executed* iterations scale.
   They do: 21 at 1,152 rows → 166 at 125K rows for the same 1e-8 target.
   A working MG preconditioner is supposed to make that flat.

## What the 128³ capture says (2,034 ms attributed GPU)

| ms | share | family |
|---|---|---|
| 1,549.8 | 76.2% | **MGPCG serial finish kernels** (`direct curvature finish` 879.6 + `merged reduction finish` 661.8, ×192 each, occupancy 0.000) |
| 220.2 | 10.8% | rest of solve (V-cycle 123.1, closed-form operator 50.4, partial reductions ~39, state updates ~8) |
| 262.6 | 12.9% | everything else (staging ×2 = 55.3, `correctAndAggregateSummaryCells` 24.3, extension adjacency 23.6, row motion 19.7, seeds/grading ~31 cold, …) |

Small lane (16.1 ms attributed): V-cycle 4.52 (34×133 µs), operator 1.47,
combined drains 2.18, direction update 0.47 — **solve ≈ 54%** of the frame;
the B1 targets are now noise (predict 0.30, S1a/b/c ~0.64).

## The mechanism, verified quantitatively

### The serial finish (the 76%)

`lib/webgpu-exact-f32-reduction.ts:187-216` — `fixedScalarValue()` is called
from a **single lane** (`finishMergedReduction` / `finishDirectionCurvature`,
`webgpu-octree-pipelined-mgpcg.ts:1828-1835`, `:1947-1954`:
`if (lane == 0u) { finishMergedTotal(fixedMergedValue()); }`) and serially
folds every partial: `for partial < livePartialCount { for limb < 36 {
atomicLoad } }`.

At 128³: `livePartialCount = ceil(125,067/128) = 978` partials × 36 limbs =
35,208 atomic loads per scalar; `fixedMergedValue()` folds **4 scalars**;
two finish kernels per iteration; 192 encoded iterations:

> 978 × 36 × 4 × 2 × 192 = **54.1 M serial dependent atomic loads**
> measured 1,541 ms ⇒ 28.5 ns/load. The arithmetic closes exactly.

Three compounding wastes inside that:

- **Only lane 0 works** in a kernel already dispatched with 128 lanes.
- **`fixedMergedValue()` is evaluated before the stopped-check can return**
  (argument evaluation precedes `finishMergedTotal`'s `control[1]` early-out
  at `:1750`), and the finish kernels are direct `[1,1,1]` dispatches the
  indirect-retirement machinery cannot zero — so the 26 post-convergence
  iterations (166 executed of 192 encoded) still paid ~8 ms each ≈ 210 ms.
  Same failure family as the previous round's drain bypass, opposite arm.
- **All 4 scalars are folded everywhere**: the curvature finish needs only
  `delta`; the non-initial merged finish needs neither `delta` nor `bb`.

### The iteration count (the multiplier)

166 executed iterations at 125K rows vs 21 at 1,152 rows, same authored
1e-8 relative tolerance (`lib/default-scene.json:29`, honored deliberately
by `octree-solve-tail-policy.ts:354-356`). Preconditioned-CG iteration
growth of ~8× across a 4× resolution step means the V-cycle is not acting
as a multigrid preconditioner at scale. See the convergence-anatomy section
below for the verified cause.

## The bets

### S1 — Parallelize the exact fold in place (bit-identical, days) — THE headline

Integer limb addition is associative and commutative — that is the entire
reason the superaccumulator exists (`docs/losasso-cutover-plan.md`). Any
partition of the partial fold produces identical limb totals, so a
cooperative fold is **bit-identical** to the serial one:

1. In `finishMergedReduction` / `finishDirectionCurvature`, split
   (partial, limb, scalar) across the workgroup's 128 lanes (each lane
   accumulates a strided slice of partials for its limbs into i32
   registers), combine per-limb across lanes via workgroup shared memory,
   barrier, then lane 0 runs the existing carry propagation + decision
   logic unchanged. No ABI change, no new dispatch, no layout change.
   ~275 loads/lane ≈ 20-30 µs per finish instead of 3.5-4.6 ms.
2. Evaluate the fold **after** the stopped/converged early-outs, not as a
   call argument — retired iterations then cost ~µs.
3. Fold only the scalars the call site consumes (delta-only for curvature;
   gamma/rr for non-initial merged).

Expected at 128³: 1,549.8 → **≤ 10 ms** (2,078 → ~540 ms wall, a 3.9× on
its own). Small lane: unaffected (combined-drain tier — already fine).

### S2 — Make the preconditioner actually work at scale (the convergence bet)

**Lead finding: at 128³ the V-cycle is probably not running as multigrid
at all.** The fused sub-L0 kernel's enable predicate requires *every*
sub-level to be published (`vcycle-gpu.ts:203-209`) and returns
immediately otherwise. The 128³ hierarchy trace shows the L2 face refresh
as a zero-workgroup dispatch and transitions 3-5 rebuilding from scratch —
the signature of L2 publication failing and `controlValid()` cascading
ERROR_GEOMETRY into L3-L5 (`losasso-hierarchy.wgsl.ts:71-73`, `:182`).
With `enabled=false`, the preconditioner degenerates to **4 damped-Jacobi
sweeps on L0** — and a smoother-only preconditioner is exactly what makes
CG iterations track resolution (21 → 166). The V-cycle pass time
corroborates: 682 µs median at 128³ is what the six L0-sized dispatches
alone cost; a one-workgroup serial walk of a ~50K-row L1 could not fit in
it. The nonconvergence tripwire that would have flagged this was disabled
in the capture (`FLUID_TRIPWIRES=0`).

**Prime suspect** (`losasso-hierarchy.wgsl.ts:270-273`): aggregated
boundary/free-surface faces measure their coarse distance from the fine
face plane to the parent centre; that distance is exactly 0 whenever the
fine boundary plane coincides with the aggregate mid-plane — ubiquitous in
a dam break — and `!(distance>0)` sets ERROR_GEOMETRY.

**S2.0 — Verify before designing (hours).** Read back the 8 control words
per level (word 3 = published, word 4 = error bits, words 1/2 =
rows/faces) on the 128³ lane. If any sub-level is unpublished, fix the
zero-distance aggregation case (clamp to the fine face's own half-width,
or skip aggregate-internal boundary patches into the diagonal), re-enable
the nonconvergence tripwire in test lanes, and re-measure iterations
before touching anything else. This alone may recover most of the 166→~30.

**S2.1 — Measure contraction, not iterations.**
`FLUID_SYMMETRY_STAGE_AUDIT=1` already captures r₀, M·r₀, and A·M·r₀
(`pipelined-mgpcg.ts:1001-1043`, readback exists in
`tools/webgpu-smoke-readbacks.ts:1531-1661`): report
‖r₀ − A·M·r₀‖/‖r₀‖ on both lanes. Healthy V-cycle ≈ 0.05-0.2;
smoother-only ≳ 0.9. This number is S2's permanent regression metric; add
a per-iteration `rr` ring buffer (written by `finishMergedReduction`) for
the full curve at diagnostics cadence.

**S2.2 — Structural repairs, one land each, in impact order** (verified
anatomy: ω=2/3 Jacobi 2+2 (`vcycle-gpu.ts:282`, `:51-54`), sum-restrict /
constant-prolong transfers (`:72-87`), rediscretized coarse operators,
depth = log2(maximumLeafSize)+1 with an 8-Jacobi-sweep bottom
(`:215-219`)):

1. **Carry the free-surface conditioning to coarse levels.** The coarse
   face refresh inherits only `openFraction`/`normalVelocity` and
   recomputes `inverseDistance` geometrically
   (`losasso-hierarchy.wgsl.ts:369-392`) — the ghost-fluid θ-scaled
   Dirichlet coefficients (up to 10⁴× geometric,
   `losasso-coarse-phi.wgsl.ts:123-137`) never reach any coarse level, so
   the coarse grids solve a different problem exactly where the rows
   concentrate (the surface band IS most of the row set at 128³).
   Aggregate the fine θ-conditioned coefficients instead of recomputing
   from geometry.
2. **Uncap the hierarchy depth and solve the bottom.** Depth stops at
   span = maximumLeafSize regardless of domain (`losasso-hierarchy.ts:
   144-146`): the 128³ bottom is 64 rows "solved" by 8 Jacobi sweeps
   (per-cycle contraction floor ≈ 0.6 from the bottom alone; worse at
   1024³ where the bottom would be 32³). Either extend levels until the
   bottom is ≤ ~16 rows, or replace the bottom sweeps with an exact
   cooperative solve. Cheap; entirely internal to the preconditioner.
3. **Order-2 prolongation.** Constant-constant transfers (m_P + m_R = 2)
   fail the classical > 2m requirement and degrade with level count even
   once 1-2 are fixed. Row cell centers exist per level (`cells`
   buffers); prolong trilinearly from parent centers. Alternatively
   K-cycle/AMLI coarse acceleration. Land after 1-2, judged by the S2.1
   contraction metric.
4. ω = 2/3 → 6/7 (the 3D-optimal weight): a free ~15% per sweep.

All S2 items are preconditioner-only — the converged answer is unchanged,
the iterate sequence is not — so each lands solo, judged by the D4 gate
plus the contraction metric. Also fix the one-workgroup fused sub-L0
scaling hazard while in here: it has no size gate
(`vcycle-gpu.ts:278`), and once L1-L5 publish again at 128³ a single
workgroup would serially relax ~50K L1 rows — gate the fused form to the
small tier (or make sub-L0 levels multi-workgroup) as part of S2.0's
re-measure.

Target: **≤ 30-40 iterations at 128³, ~flat with resolution.** Combined
with S3, executed iterations ~25-35; the remaining per-iteration cost
(~1.3 ms post-S1) puts the whole solve at ~40-90 ms.

### S3 — Tolerance policy renegotiation (per-scene, gated)

The authored default (`default-scene.json:29`, 1e-8) is inherited by every
scene, while the repo's own stated QA floor for f32 projection is 1e-4
(`pipelined-mgpcg.ts` finalize comment; `tools/webgpu-smoke-pressure.ts`).
CG cost is roughly linear in requested digits: 1e-8 → 1e-5 saves ~35-40% of
executed iterations on top of S2. Keep 1e-8 where an oracle depends on it
(D4 guide lanes); author the large scenes at the QA floor they are actually
judged by. Contract change: per-scene authoring + gate runs, never a global
silent switch.

### S4 — Small lane: the resident solve (B3b, unchanged)

The previous handoff's B3b (wire `webgpu-octree-persistent-mgpcg.ts` as the
≤4K-row tier) remains the small-lane fix — the 8.6 ms solve family
(V-cycle 34 × 133 µs + drains + operator) collapses into one cooperative
dispatch. Nothing in this round changes its design; S1's lane-parallel fold
pattern is the same trick at a different tier. Expected: small lane
14 → ~6-7 ms.

### S5 — The post-solve tail at 128³ (only after S1/S2 re-measure)

The 262.6 ms non-solve tail is mostly honest, well-occupied domain work,
part of it first-frame-only (fine-seed publication 20.0, grading closure
11.1). Recurring standouts once solve shrinks:

- Staging runs twice (`stage finest-MAC` 28.7 + `stage predictor
  finest-MAC` 26.5) — the predictor lattice may be derivable from the
  same staging pass or needed only where the predictor actually samples.
- `correctAndAggregateSummaryCells` 24.3 ms — the previous B5
  workgroup-reduce-then-atomic applies at 2M cells.
- Extension-band adjacency + 7 dilation layers ~46 ms — capacity-shaped
  candidates from the previous program (AP3 territory).

Do not plan these in detail until a steady-state 128³ capture exists —
first-frame shares will not survive contact with advance 50.

## Expected trajectory (128³ first frame)

| After | solve | non-solve | frame (est.) |
|---|---|---|---|
| baseline | 1,770 | 263 | 2,078 |
| S1 (parallel fold) | ~240 | 263 | ~540 |
| S2+S3 (~166 → ~25-40 iter) | ~50-90 | 263 | ~330-380 |
| S5 (tail, steady-state) | ~50-90 | ~130-180 | **~200-280** |

That is 8-10× on the first frame and more on steady advances (warm-started
solves converge in fewer iterations; cold seed/grading publication drops
out). Small lane: 14 → ~6-7 (S4) → ~4-5 with the previous handoff's B4/B5
residue — the original 10× target for that lane keeps its path.

## Verification contract

- S1 is Tier A (bit-identical by integer associativity): D4 +
  dry-identity gates must agree exactly; land alone, judged by gate
  counters, fresh dated captures of BOTH lanes (the standing mistake this
  document exists to prevent: never baseline a solve change on one tier).
- S2/S3 change executed iteration sequences — solo lands, judged by the D4
  gate and the pressure QA floor; record per-iteration residual curves
  before/after (see convergence anatomy for the diagnostic hook).
- The retirement fix in S1.2 must keep the publication contract: a
  converged solve still publishes pressure + diagnostics every advance
  (gen-91 rule).
- Noise rules stand: ±5% within-arm, never across tripwire modes; 128³
  WindowServer contention was 229 ms in this capture — recapture quiet.
- **Profiling lanes run `--no-tripwires`, which silenced the MGPCG
  nonconvergence tripwire that would have flagged the dead preconditioner.**
  Any lane used to judge solve health must run at least one arm with
  tripwires on; a hierarchy-publication error word (level control word 4)
  should become a fatal tripwire of its own (an unpublished level silently
  degrading the preconditioner is the gen-91 failure shape in solver
  clothing).

## What NOT to do

- **Don't shrink partial count to "fix" the fold.** Fewer partials means
  more atomic contention in the deposit pass (the parallel stage that is
  currently cheap at 120 µs). The fold is the bug, not the partial layout.
- **Don't replace the superaccumulator with f32 trees to make the fold
  parallel.** Parallelism is free *inside* the integer representation;
  changing the arithmetic is an unforced contract change.
- **Don't judge S2 on the small lane** — 5 levels over 1,152 rows converges
  in 21 iterations regardless; the preconditioner defect only expresses at
  depth. Both lanes, every solve change, always.
- Don't micro-optimize the 128³ tail families (S5) before S1/S2 land — the
  frame they sit in shrinks 6-10× first, and steady-state shares differ.

---

# What landed (2026-08-06)

## S1 — the parallel fold. Shipped, bit-identical, 4.2×.

`fixedScalarValue` is now a cooperative limb-major fold
(`lib/webgpu-exact-f32-reduction.ts`): `limb = lane % 36`, `group = lane / 36`,
so each group of 36 lanes reads one partial record as a contiguous 144-byte
span and the partial axis splits across `FIXED_LIMB_GROUPS = 3` groups. Lane 0
then runs the unchanged carry propagation. Callers select work by **count, not
by branching** — the fold carries workgroup barriers, so `partialCount = 0`
is how a retired iteration folds nothing while staying in uniform control flow.

Both compounding wastes went with it: the fold now happens after the
converged/failed early-outs, and each call site folds only the scalars it
consumes (2 of 4 for the non-initial merged finish, 1 of 4 for curvature —
the rest are cleared limbs that fold to +0.0 anyway).

All walls below are the clean measurement regime (`FLUID_TRIPWIRES=0`, no stage
audit), two reps per arm, so they are comparable with this document's 2,078 ms
baseline. Never compare across tripwire modes.

| lane | before | S1 only |
|---|---|---|
| 128³ high-resolution-dam-break | 1,929 / 1,985 ms | **463 / 585 ms** |
| symmetric-expansion guide lane | 14 / 15 ms | 14 / 14 ms (combined-drain tier, unaffected — as predicted) |

**Bit-identity is measured, not argued**: `iterations`, `relativeResidual`,
`residualSquared` and `rhsSquared` are identical to full printed precision
across arms on both lanes. The `symmetric-expansion:one-step` D4 gate passes
on all ten preconditioner stage vectors; `large-power-dam-runtime` (150
advances) passes. `octree-cutover-fine-factor4` fails identically in both arms
(8333/6213/3610 mesh mismatches, fine-phi symmetry exactly 0) — a pre-existing
render-mesh oracle failure, not a solver one.

## S2.0 — the V-cycle was inert on EVERY scene, not just at 128³

The handoff suspected the 128³ hierarchy. A per-level control-word census
(new: `readLosassoHierarchyCensus`, `FLUID_LOSASSO_HIERARCHY_CENSUS=1`, and
`=gate` to make it fatal) says it was worse than that:

```
128³ : L2 errorBits=2 published=0, L3-L5 cascade to zero rows   cycleEnabled=false
guide: L4 errorBits=2 published=0                               cycleEnabled=false
```

**No scene in this repository has ever run a multigrid preconditioner.** Every
solve was four damped-Jacobi sweeps on L0 wearing a V-cycle's name. That is why
the standing "never judge S2 on the small lane" rule was itself misleading: the
guide lane was not converging in 21 iterations *despite* a working V-cycle, it
was converging in 21 iterations with no V-cycle at all.

### The defect

`buildLosassoCoarseFaces` measured a boundary patch's Dirichlet distance from
the *fine* face plane to the *parent* centre. Aggregation groups either 1 or 2
fine cells per axis (`parentSize = max(fine.w, targetSpan)`, `targetSpan`
doubling), so that expression is **two-valued**: exactly `S/2` when the patch
faces outward and exactly **0** when it faces the aggregate's own mid-plane —
which is where a free surface cuts through an aggregate. Zero distance meant an
infinite coefficient, so the build fail-closed with `ERROR_GEOMETRY`, which
unpublished the level, which cascaded through `controlValid()` to every coarser
level, which made the fused sub-L0 enable predicate false.

The repair (`coarseBoundaryDistance`, both the build and refresh sites) uses the
aggregate half-width, which is bit-identical to the old value wherever the old
value was valid and merely replaces the zeros with the value their sibling patch
already had. **The span is recomputed from the fine cell rather than read from
`coarseCells[parent]`** — a first attempt did read the coarse cell and regressed,
because entries past the live parent count are never cleared, so a stale parent
index yields a zero-span cell. The old arithmetic consumed that same zeroed cell
through `centre()` and merely happened not to produce 0 from it.

The error path also no longer skips its reserved slot: the counting pass
reserves one slot per retained face and the published count is that prefix
total, so `continue` left the tail of each lane's range holding stale records.

### Result: iterations collapse, and then depth becomes the wall

| 128³ | iterations | wall |
|---|---|---|
| V-cycle inert (S1 only) | 166 | 524 ms |
| V-cycle live, full depth 5 | **17** | 850 ms |
| V-cycle live, depth 3 (shipped) | 27 | **398 / 403 ms** |

The iteration count did what the handoff predicted and better — but naively
enabling the V-cycle made the frame *slower*. **The reason is a property of
this hierarchy the program had not priced: aggregation coarsens rows but only
re-parents faces, and the smoother walks faces.**

```
level:  0        1        2        3       4       5
rows:   125,067  30,315   6,883    1,311   200     36
faces:  406,004  243,572  145,348  78,132  53,556  42,585
f/row:  3.2      8.0      21       60      268     1,183
```

The bottom two levels carry 96,000 faces over 236 rows — almost no parallelism
and almost no coarsening. A depth sweep priced them at ~460 ms to take 27
iterations down to 17. Two gates now bound this:

- `MINIMUM_COARSE_LEVEL_ROWS = 1_024` stops the hierarchy where a level can no
  longer be relaxed in parallel (`webgpu-octree-losasso-hierarchy.ts`).
- `FUSED_SUB_L0_MAXIMUM_LEVEL_ROWS = 4_096` keeps the single-workgroup fused
  cycle on the resident tier and hands larger hierarchies to the per-level
  indirect path (`webgpu-octree-losasso-vcycle-gpu.ts`). This gate was dead code
  until now — nothing had ever run the one-workgroup walk at scale, because the
  predicate it guards always returned first.

**Retirement is a secondary term, and it is still open.** The V-cycle's
per-level dispatches read each level's own `rowDispatch`, not the solver's
retirement records, so an encoded-but-retired iteration still launches a
full-size V-cycle whose threads all return on `stopped()`. Dropping the encoded
budget from 192 to 32 measured 798 → 683 ms at 17 executed iterations. The
principled fix is a solve-scoped dispatch mirror the solver can zero; it is
worth ~100 ms and did not land here.

## S2.1 — the contraction metric

`readLosassoPreconditionerContraction` reports
‖r₀ − A·M·r₀‖/‖r₀‖ from the three vectors `FLUID_SYMMETRY_STAGE_AUDIT=1`
already captures, emitted as `phase:"losasso-preconditioner-contraction"`.
Iteration count cannot separate "working V-cycle" from "smoother on an easier
problem"; this can. Use it, not the count, to judge every S2.2 change.

## The correction the next round must carry

**"Don't judge S2 on the small lane" was right for the wrong reason, and the
census is now the cheap way to be right.** Run
`FLUID_LOSASSO_HIERARCHY_CENSUS=gate` on any lane whose solve you are judging;
an unpublished level is a silent wrong-preconditioner, and it fails nothing
else. A publication tripwire in the harness ring is still owed
(`tripwireSources()` samples no hierarchy control buffer).

## What S2.2 now has to be about

The handoff's S2.2 list (θ-conditioning to coarse levels, uncap depth, order-2
prolongation, ω → 6/7) was written believing the levels were cheap and merely
wrong. They are not cheap. Re-order it:

1. **Make coarse levels actually coarsen.** 406K → 243K faces for an 8× row
   reduction is the whole problem; fusing coplanar aggregated patches (or
   folding aggregate-internal boundary patches into the diagonal, which is
   *also* how the θ-conditioning item wants to work) is what buys depth back.
   Only then is "uncap the hierarchy depth" a good idea — today it is strictly
   negative below ~1,024 rows.
2. **Carry the free-surface θ-conditioning to coarse levels** — unchanged, and
   now judged by the contraction metric rather than by iteration count.
3. Order-2 prolongation, then ω = 2/3 → 6/7.

S3 (tolerance policy) is untouched and remains available: at 27 iterations for
1e-8, a 1e-5 large-scene authoring is worth roughly a third of the solve.

## The open cost: the guide lane

The 32×16×32 guide lane went **14.5 → 20 ms** with 21 → 12 iterations. Real,
reproducible, and measured apart into its two causes:

| guide-lane arm | wall | iterations |
|---|---|---|
| before (no S2) | 15 / 14 ms | 21 |
| `MINIMUM_COARSE_LEVEL_ROWS` raised so it builds NO coarse level | 19 / 18 ms | 14 |
| shipped (2-level multigrid) | 20 / 20 ms | 12 |

So **only ~1.5 ms of the 5.5 ms is the coarse levels.** The rest is the
single-level path: with one level the V-cycle falls into its bottom branch,
which is **eight** Jacobi sweeps, where the old effective preconditioner was the
fused schedule's four (2 pre + a fused kernel that returned immediately + 2
post). Raising the floor therefore trades most of one regression for most of
another; the actual fix is to give the single-level case the 2+2 shape, which
means touching `encodedCorrectionDispatchCount`'s validated dispatch accounting.
That did not land here.

**The floor stays at 1,024 deliberately**, so the guide lane keeps a real
two-level V-cycle. It is the only lane with `requirePressureStageAudit`, and its
D4 gate checks all ten preconditioner stage vectors — which, as of this round,
is the first time that gate has ever exercised an actual multigrid cycle rather
than four Jacobi sweeps. Buying back 5 ms by making the correctness lane stop
testing the thing that was just resurrected is the wrong trade.

Judge either route by the contraction metric plus both lanes, never by the guide
lane alone — that rule is what this whole document exists to enforce, and S2.0
showed it cuts both ways.
