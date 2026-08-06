# Losasso coarse-band 10x — implementation handoff

**Date:** 2026-08-06
**Lane:** symmetric-expansion, coarse-band Losasso
(`--coarse-backend=losasso --fine-factor=1 --band=4 --maximum-leaf-size=16`,
32×16×32, 1,152 live rows / 2,048 wet cells / 2,052 interface cells).
**Capture:** `artifacts/xctrace-losasso-coarse-band-first-frame/report.html`
(summary: `summary.json` beside it).
**Baseline:** 36 ms untraced first advance → **target ~3.6 ms**.
**Predecessors:** `docs/losasso-10x-handoff.md` (SP0–SP6, closed out),
`docs/losasso-activity-handoff.md` (AP0–AP6). This document does not re-plan
those; it addresses the coarse-band lane, whose frame shape is different from
both.

## What the capture says

34.02 ms attributed GPU of a 55 ms traced frame (untraced 36 ms; traced
distortion 2.11×). Untraced command audit: 9 submissions, 39 passes, 771
dispatches, **0 completion fences** — the wall tracks GPU busy. This is a
GPU-work problem, not an encode/gap/fence problem.

| ms | calls | family |
|---|---|---|
| 19.88 | 19 | coarse-only summary pipeline — **19.15 is one dispatch, `predictSummaryCells`** |
| 7.99 | 263 | pressure solve (V-cycle 41×81 µs, drains 80×~33 µs, closed-form 42×20 µs, …) |
| 3.19 | 8 | transport/dynamics (S1a 0.95, S1c 0.79, S1b 0.61, coarse-phi advect 0.44, …) |
| 1.23 | 16 | topology/seeds |
| 0.53 | 22 | extension band (7 dilation layers + adjacency) |
| ~1.2 | — | hierarchy, coarse phi, misc |

Caveats: the capture window was contended (Codex service used 21.5 ms of GPU;
55% of our GPU time uncontended) and it is a **first advance** — the solve ran
19 executed iterations of a 40-iteration encoded budget (`iterations: 19`,
converged, in `baseline.log`); warm steady advances execute far fewer. Bet 0
re-baselines both shapes before anything is judged.

## Why the frame costs what it costs (verified anatomy)

### The 19 ms dispatch is four stacked pathologies, and the largest is dead work

`predictSummaryCells` (`lib/webgpu-octree-coarse-summary.ts:588-614`, encoded
at `:295`) is one thread per coarse cell — 16,384 threads at workgroup size
256 = 64 workgroups (`dispatch(..., ownerDirectoryCellCapacity)`, a vestigial
name that equals `domainVolume`). The attribution is exact: this capture ran
with per-pass encoder isolation, so 19.15 ms is this dispatch alone,
~1.17 µs per thread. Where it goes:

1. **Dead work, the largest term.** `var sample=interpolatedCoarseAt(point)`
   at `:601` runs unconditionally, and its result is discarded whenever
   `initialized` is true — which is **always**: `state[16]` is set to 1 at
   construction (`:203`) and `resetSummary`'s clear mask deliberately skips
   it (`:550`). The discarded call is an 8-corner trilinear where each
   corner miss triggers a full 3×3×3 rescan (`extrapolatedCoarseAt`,
   `:426-431`), each probe a dyadic size ladder (`coarseAt`, `:417-421`) over
   a **binary search of the Morton-ordered compact row directory** with
   ~24-ALU-op comparators per step (`coarseSlot`, `:414-416`). Worst case
   ~16K dependent random 32-byte loads per thread — all discarded. The
   compiler cannot remove it (the value feeds a live phi node).
2. **Two exact-arithmetic velocity samples per thread** (RK2 departure +
   midpoint). In Losasso mode `losassoVelocityAtGrid`
   (`lib/webgpu-octree-losasso-velocity-sampler.wgsl.ts:52-98`): 3 axes × 8
   corners = 24 `losassoFace` ladder walks per call — spans 1→16 (5 rungs)
   × up to 32 hash probes × a 4-word `faceGeometry` verify (`:33-50`), wall
   corners doubling the walk (`:87-92`) — then the 36-limb signed-integer
   superaccumulator: `losassoExactValue` per axis, two 35-iteration carry
   loops + a 36-term `ldexp` fold (`:5-18`), ~400-800 serial integer
   divisions per thread.
3. **Private-array spill traffic caps occupancy.** The `array<i32,36>` limb
   accumulator (copied by value in `exactValue`) and five `array<f32,8>`
   canonical-sum buffers are dynamically indexed, so Metal puts them in
   per-thread stack memory. The counter evidence: the kernel's real payload
   is 64 KB (16K × 4 B), yet it measured ~1 GB/s of writes ≈ 19 MB — that
   is spill, not output, and it is what pins occupancy at 4% independent of
   the 64-workgroup launch being too small to fill the machine anyway.
4. **Most threads pay the maximum-cost path to do nothing.** Velocity exists
   only on the W≤7 corridor. A far-field cell's ladder walk misses at every
   rung of every corner — the exhaustive-miss path — then `velocity.w==0`
   keeps phi unchanged (`:608`). ~14K of 16K threads run the most expensive
   probe sequence to conclude "no-op." That is the domain-shaped workload,
   and it violates the repo's own Bet-1 rule
   (`lib/webgpu-octree-work-accounting.ts:1046-1069`: "Capacities may size
   buffers; they may never size launches").

**The same sampler is instantiated in the S1 dynamics shader**
(`lib/webgpu-octree-losasso-dynamics.wgsl.ts`: `addExact`/`exactValue`/
`exactFace`/`containingFace`/`velocityAtGrid`) and in coarse-phi advection —
S1a/S1b/S1c (2.35 ms at 2–4% occupancy) and `advectLosassoCoarsePhi` are the
same latency shape at smaller item counts.

Why nobody saw this before: commit `a56ddd0` revived the coarse-only
tracker — a frozen layout-version literal had made `predictSummaryCells`
early-return without writing a cell (`:451-457`), so **this pass had never
been profiled while doing work.**

The superaccumulator is not an accident — it is the D4 primitive the cutover
plan introduced (`docs/losasso-cutover-plan.md`: integer accumulation is
permutation/partition invariant; mirrored sample points visit the same 8
corner values in different loop order). Fixes must preserve that property,
not delete it.

The rest of the summary pipeline: 11 of its 19 dispatches are O(domain).
`ensureSupportSummaryPages`/`Ranks` launch 16,384 threads to claim **293
hierarchy entries** — all 64 threads of each 4×4×4 block CAS the same
directory word through a 4-deep retry loop (`:559-580`).
`correctAndAggregateSummaryCells` fans 16K threads of atomicMin/Max into the
same 293 words (`:662-674`). The five dense redistance sweeps (`:620-649`)
are honest work and cheap. `FLUID_COARSE_SUMMARY_INDIRECT_DISPATCH` changes
none of this — it is launch-shape only and still publishes
`(domainVolume+255)/256` workgroups.

### The solve is 8 ms of tiny serialized links for a 1,152-row problem

The loop is host-unrolled at `lib/webgpu-octree-pipelined-mgpcg.ts:1057-1116`
(entered via `webgpu-octree-losasso-backend.ts:628-636`). Per encoded CG
iteration, six labelled requests / 14 dispatches: outer-state advance,
V-cycle application, merged reduction drain (direct `[1,1,1]`), direction
update, closed-form operator, direct-curvature drain (direct `[1,1,1]`) ≈
**178 µs × 40 encoded ≈ 7.1 ms**, at ≤1% occupancy throughout. (The 352
encoders in the trace are a capture artifact of
`FLUID_GPU_ISOLATE_PASS_ENCODERS`; production encodes this as ~1 pass — the
cost is 579 barrier-separated dispatches, not encoder creation.)

The pass labeled `Losasso V-cycle - initialize levels`
(`lib/webgpu-octree-losasso-vcycle-gpu.ts:371`) is mislabeled in effect: it
contains the **entire V-cycle preconditioner application**, and the fused
sub-L0 form is already the default (`:274`; 9 dispatches per application).
The 81 µs is dominated by the fused kernel itself: **one 256-lane workgroup**
(`:398`, `dispatchWorkgroups(1)`) serially walking the L1→L4 ladder with ~31
barrier pairs, where L1 alone is a 2,048-row-capacity level relaxed 4× on
one GPU core (`losasso-hierarchy.ts:141-144`).

Three encode-shape facts multiply the waste (verified in
`pipelined-mgpcg.ts` / `webgpu-octree.ts`):

1. **The encoded budget is the hard ceiling, not the expected tail.** The
   Losasso backend is constructed with `maximumIterations =
   hardOuterIterationCeiling` = 40 (`webgpu-octree.ts:2655-2658`,
   `octree-solve-tail-policy.ts:16`), where the structured lane encodes 10.
2. **Retirement is bypassed on this lane.** The solve *did* converge at
   iteration 19 (`baseline.log`: `converged: true, iterations: 19`,
   relative residual 9.6e-9), and convergence zeroes `outerDispatch`
   records — but the combined-drain arm dispatches `advancePCGState` /
   `updateDirections` via `runAcceptedRowsIndirect` off the immutable
   `source.rowDispatch` (`pipelined-mgpcg.ts:1258-1268`), both drains are
   direct `[1,1,1]`, and the V-cycle/operator dispatch off hierarchy-owned
   records. The zeroed records are dead in this schedule: **iterations
   20–40 launched the full per-iteration graph** (threads exit via
   `stopped()`, the ~31-barrier fused walk still executes its ladder) ≈
   3–4 ms of retired-but-launched work in this capture.
3. AP5's encoded-tail predictor (`selectOctreeFactorOneEncodedSolveTail`)
   is dead code, gated off, and clamps to ≤10 — it cannot express this
   lane's shape as-is.

The entire solve state — 4 vectors × 5 levels × ~1.5K live rows × 4 B ≈
120 KB packed, L0 vectors alone ~18 KB — fits in or near threadgroup
memory. Every inter-pass boundary in the chain pays full price to
synchronize a problem that fits in one workgroup's shared memory. **An
end-state template already exists in-tree and is wired to nothing:**
`lib/webgpu-octree-persistent-mgpcg.ts` runs the whole solve — V-cycle,
operator applies, CG updates, compensated dot products, all iterations —
inside ONE dispatch of one 256-lane workgroup, with a 65,536-row envelope
(`section43-contract.ts:25`). The Losasso problem sits far inside it.

### The summary pipeline rebuilds its directory from scratch every advance

`resetSummary` clears the top-word table wholesale (comment at
`webgpu-octree-coarse-summary.ts:280-282` already calls this "Bet 1
residue"); `ensureSummaryPages`/`ensureSupportSummaryPages`/
`ensureSummaryRanks`/`ensureSupportSummaryRanks` re-claim every page and rank
with atomic CAS retry loops and per-claim `PAGE_SIZE` clear loops
(`:551-580`), all dispatched at `ownerDirectoryCellCapacity` or
`rowCapacity`. ~0.7 ms/advance re-deriving a directory whose keys change only
when topology changes. The five dense redistance sweeps (`:620-649`) are
cheap by comparison (6 neighbor loads each) and are not a target.

## The bets

Ordered by expected yield. B1 and B3 carry the program; a 10x that misses
either does not exist. Every bet is better-algorithm / better-data-structure /
better-utilization — none is "fewer dispatches" as a goal; dispatch counts
fall out where they fall out.

### Bet 0 — Clean re-baseline, two arms (prerequisite, half a day)

The capture is contended and first-advance-only. Recapture uncontended
(everything else closed), two arms: `--first-frame` (this shape) and a
50-step steady arm of the same lane. Record the per-label table and
`iterations` for both. All bets are judged against these tables; the noise
rules stand (±5% within-arm; never compare across tripwire modes).

### Bet 1 — Make `predictSummaryCells` compute only what it publishes

Three sub-bets, in landing order. B1a is a one-line fix worth more than any
other single change in this document.

**B1a — Guard out the dead interpolation (hours, output-identical).**
`:601` becomes: compute `interpolatedCoarseAt(point)` only when
`!initialized`. Since `initialized` is constructionally always true, the
branch never executes in production — the change is output-identical by
inspection (the overwrite at `:602-603` is unconditional for every in-range
item). This deletes the 8-corner × 27-rescan × size-ladder × binary-search
pointer chase from every thread. If the bootstrap path genuinely needs the
interpolated seed on advance 0, it still gets it. Expected: 19.15 →
**~4-7 ms** on its own. Verify with a fresh capture before stacking B1b —
this re-ranks everything after it.

**B1b — Stage the staggered velocity lattice once per advance
(bit-identical).** `losassoFace(axis, q)` is a pure function of the
published face directory — per advance, per staggered node, it always
resolves to the same face and the same `extendedVelocity` value (including
the closed-wall extension logic, also a pure node function: sampler
`:76-92`). Today it is re-evaluated 24× per velocity sample, per consumer,
per RK2 stage — and for far-field nodes the evaluation is an exhaustive
miss.

Build a dense staged staggered velocity immediately after projection
publishes `extendedVelocity`:

- Three banks, one per axis, `(dims[axis]+1) × tangent dims` ≈ 3 × ~17K
  nodes ≈ 51K threads ≈ 200 KB f32 + validity (pack value+valid or use a
  sentinel encoding). One thread per node does ONE ladder walk and one
  wall-extension resolve, writes the value. No atomics; deterministic; a
  wide, well-occupied dispatch.
- Rewrite `losassoVelocityAtGrid` (and the dynamics shader's
  `velocityAtNode`) to read the staged banks: 8 dense loads per axis
  instead of 8 ladder walks. **Weights, corner order, and the
  superaccumulator fold are untouched — the per-corner value is identical
  by construction, so results are bit-identical.** D4 gate must confirm; no
  renegotiation is being made.
- All consumers switch: `predictSummaryCells`, S1a/S1b/S1c, coarse-phi
  advection, `reconstructLosassoRowMotion` — every instantiation of the
  sampler in this lane.

Arithmetic: predict does 16K × 2 samples × 24 walks ≈ 786K ladder walks
(mostly exhaustive misses); S1a/b/c add ~24 walks per RK2 stage per face
over ~3.6-10K faces × 3 passes. Staging does 51K walks total, once.

**B1c — Stop spilling (bit-identical, mechanical).** Pass the
superaccumulator by pointer everywhere (`losassoExactValue` currently
copies its 144-byte argument by value), restructure `canonicalSum8`'s
sort to avoid a dynamically-indexed private copy where possible, and put
the per-advance interface telemetry atomics (`state[27..30]`, `:605-611`)
behind the diagnostics flag so production threads stop contending four
device-scope words. Target: private stack traffic ≈ 0, occupancy no longer
spill-capped.

Expected after B1a+B1b+B1c: predict 19.15 → **≤ 1 ms**, S1a/b/c +
coarse-phi advect 3.19 → **≤ 0.8 ms**, staging pass ≤ 0.3 ms.

Scaling note (droplet-in-a-vast-domain direction): a dense staged lattice is
domain-shaped in *bytes*. That is acceptable at 32³ (200 KB) and stays
acceptable while the coarse domain is the small lattice of the two-tier
design; if the coarse domain ever grows past ~128³, stage per extension-band
tile instead (the corridor worklist already exists). Do not build the tiled
form speculatively.

### Bet 2 — Cheapen the permutation-invariant fold (D4-renegotiated, lands solo)

After B1, the predict kernel's remaining serial cost is the superaccumulator:
6 `exactValue` evaluations per thread (~100+ serial integer loop iterations
each). Two options, in preference order:

1. **Sorted-fold:** bitonic-sort the 8 corner terms by total order
   (bitcast-u32 ordering breaks sign/magnitude ties), then sum sequentially.
   19 compare-exchanges + 7 adds. Permutation-invariant by construction —
   the D4-mirrored thread sums the same multiset in the same sorted order —
   which is the only property the superaccumulator is buying here. It is
   *not* the same bits as the exact sum, so this is a contract change:
   D4 re-bless, lands alone, judged by the gate.
2. **Keep the superaccumulator but hoist it:** accumulate the 8 corners'
   limb contributions with `losassoExactAdd` (cheap, 4 limb updates each) as
   today, but replace the per-axis `losassoExactValue` normalization with a
   single deferred normalization after all three axes — 3× fewer carry
   chains, bit-identical output. Smaller win, Tier A. Do this regardless;
   attempt (1) only if the profile still shows the fold after B1+B3.

Expected: with (2) alone, predict ≤ 1.2 ms; with (1), predict ≤ 0.5 ms.

### Bet 3 — Resident pressure solve: the whole loop in one cooperative kernel

**Claim: the solve family (7.99 ms) collapses to ~1 ms when the loop stops
crossing dispatch boundaries. The kernel already exists in-tree.**

Two stages:

**B3a — Stop launching retired work (days, encode-shape only).**
1. Encode `encodedOuterIterations`-style budget instead of the 40-iteration
   hard ceiling (`webgpu-octree.ts:2655-2658` is the one line that differs
   from the structured lane's 10). Keyed to last-executed+margin per AP5's
   design, falling back to the ceiling on epoch change/nonconvergence —
   this lane converged at 19 cold and will sit far lower warm.
2. Fix the retirement bypass: in the combined-drain arm, route
   `advancePCGState`/`updateDirections` through zeroable dispatch records
   (or make the drains and the V-cycle honor a solve-stopped indirect
   record) so convergence actually retires the suffix — today the zeroed
   `outerDispatch` records are dead and iterations 20–40 launched the full
   graph. This is the mechanism `docs/losasso-gpu-speedup-handoff.md:48-51`
   already flagged.
Expected: ~7.1 → ~3.5 ms first frame (19 executed × 178 µs), much less warm.

**B3b — Wire the persistent solve.** `webgpu-octree-persistent-mgpcg.ts`
runs the entire preconditioned CG loop — V-cycle, operator, reductions,
convergence — in ONE dispatch of one 256-lane workgroup, barrier-for-
dispatch, and is instantiated nowhere in production. Port it to the Losasso
operator/hierarchy (the fused sub-L0 kernel already proves the
barrier-for-dispatch substitution passes the D4 gate on this exact operator;
the closed-form operator has no power-descriptor arithmetic to transcribe):

- Feed it the packed vector arenas (already contiguous,
  `losasso-vcycle-gpu.ts:271-305`) and the per-level CSR/transfer views.
  Size level loops by **live** row counts, not `levelRowCapacities` — the
  current fused sub-L0 walks capacity-shaped levels (L1 = 2,048 capacity
  for ~288 live).
- Reductions: integer superaccumulation is partition-invariant, so the
  exact-reduction contract holds trivially inside one workgroup with a
  fixed fold tree — the cutover plan's own argument, applied in the
  opposite regime.
- **Selection is a size tier, not a replacement.** Small problems (fits
  the persistent envelope comfortably; this lane is 1,152 rows) take the
  resident kernel; larger keep the pipelined path. The cutover plan retired
  the old one-workgroup executor because it was the wall at 65K rows — that
  verdict stands; this is the regime where "cooperative single-workgroup
  kernels keep determinism trivial" applies.
- Consider 2–4 SIMD-groups' worth of lanes (512–1024 threads) only if the
  single-workgroup profile shows ALU starvation on L0/L1 sweeps; don't
  build multi-workgroup machinery.

Numeric contract: aim for the executed iteration sequence of the pipelined
path (same operator, same fold orders, same convergence predicate) — then
D4 agreement is expected, but the land still gets its own gate run. If
bit-agreement proves impractical, it becomes an SP6-class solo land judged
by the D4 gate directly.

Expected from B3a+B3b: 8 → **~1 ms** first frame, less warm.

### Bet 4 — Corridor-shaped prediction: stop visiting the far field (scaling bet)

After B1 the far-field cells are cheap, but they are still *visited* by
every dense stage (predict, 5 redistance sweeps, volume summarize, correct)
— acceptable at 16K cells, domain-shaped in principle. Make the prediction
corridor explicit:

- Cells whose staged-velocity support is entirely invalid (outside the W7
  corridor + one trilinear halo) provably keep `predicted == current` (the
  `:608` guard). Build a corridor cell worklist from the extension-band
  face worklist (it already exists GPU-side: the band publishes per-layer
  face lists) dilated by the trilinear support; dispatch predict + volume
  correction over the corridor indirect count; a trivial carry/restamp
  kernel covers the complement (or nothing at all, if the bank-flip
  copy is fused into redistance sweep 1).
- The redistance sweeps stay dense for now (they are the mechanism that
  keeps far-field distances sane, and they are cheap); if a later profile
  shows them, band them with the same worklist + per-sweep halo shrink.

Expected at 32³: ~0.3 ms further off the summary pipeline. The real payoff
is structural: the coarse-band advance stops having any pass whose cost is
proportional to domain volume with a per-cell constant bigger than a few
loads — which is the property the droplet-in-a-vast-domain program needs
from this lane. Cost ∝ corridor (n^2/3-shaped), not domain.

### Bet 5 — Right-size and retain the summary hierarchy maintenance

Two independent fixes to the non-predict pipeline stages:

1. **Launch the hierarchy by its own cardinality.** The whole summary
   hierarchy for this domain is **293 entries** (~14 KB), yet
   `ensureSupportSummaryPages`/`Ranks` launch 16,384 threads whose 4×4×4
   blocks all CAS the same word, and `correctAndAggregateSummaryCells` fans
   16K threads of atomicMin/Max into those 293 words. Enumerate hierarchy
   keys directly (293 threads) for the ensure passes; for the aggregate,
   workgroup-reduce first — a 256-thread workgroup covers exactly four B4
   blocks — and emit one atomic per (entry, level), preserving the
   min/max/mask semantics (order-free operations, bit-identical).
2. **Retain the directory across advances (epoch-keyed).** The page/rank
   assignment is a pure function of coarse topology (row set + support base
   set), which changes only on topology epochs. Apply the predecessor's
   landed pattern (exact-topology reuse keyed on `candidatePowerGeneration`):
   keep directory, pages, and ranks across unchanged epochs; run the
   reset/ensure ladder only on epoch change. This also delivers the
   touched-key worklist that `resetSummary`'s "(Bet 1 residue)" comment is
   waiting for. Retention is "no rebuild", never "no publish" — the receipt
   counters and bank flip keep their per-advance semantics (gen-91 /
   dry-identity rule).

Expected: ~0.7 → ~0.15 ms, and the summary pipeline stops being a
per-advance allocator. Small in ms; large in pass pressure and in scaling
hygiene.

### Bet 6 — Close the wall-to-GPU gap (only after B1–B3 re-measure)

Untraced: 36 ms wall vs ~30–32 ms estimated GPU busy — a few ms of host
encode + 9 submissions + inter-submission idle. After B1–B3 shrink GPU busy
to ~4 ms, the residual host cost bounds the wall. Inherited SP5 applies
unchanged: submission consolidation toward 1–2 per advance, bind-group
caching (the V-cycle already has `cachedBindGroup`; the summary module
rebuilds bind groups per dispatch — `webgpu-octree-coarse-summary.ts:240`),
and no new fences (there are already zero). Don't pre-optimize this: measure
after B1/B3, fix what the trace shows.

## Expected trajectory

| After | summary | solve | dynamics | rest | GPU total | wall (est.) |
|---|---|---|---|---|---|---|
| baseline | 19.9 | 8.0 | 3.2 | 2.9 | 34.0 | 36 |
| B1a (dead-work guard) | ~5–8 | 8.0 | 3.2 | 2.9 | ~19–22 | ~21–24 |
| B1b+B1c (staging, spills) | ~1.7 | 8.0 | ~0.8 | 2.9 | ~13.4 | ~15 |
| B3a (retire + budget) | ~1.7 | ~3.5 | ~0.8 | 2.9 | ~8.9 | ~10 |
| B3b (resident solve) | ~1.7 | ~1.0 | ~0.8 | 2.9 | ~6.4 | ~8 |
| B2+B4+B5 | ~0.8 | ~1.0 | ~0.8 | ~2.3 | ~4.9 | ~6 |
| B6 + tail cleanup | — | — | — | — | ~4.3 | **~4–5** |

First-frame 36 → ~4–5 ms is 7–9×; the steady arm (warm solve, quiescent
topology already landed) should cross 10× first. If the first-frame 10× is
the hard target, the remaining ms lives in the `rest` row: topology/seeds
1.23 + extension band 0.53 + hierarchy/phi ~0.6 — same treatment (live-sized
dispatches, retained-across-epoch structures), only worth planning once
B1/B3 have landed and Bet 0's steady arm shows what actually recurs.

## Verification contract

- Per land: symmetric-expansion D4 gate (exact topology/volume/diagonal;
  blessed tolerances on velocity/pressure/RHS), dry-identity/still-scene
  oracles, fresh dated capture, both Bet 0 arms. Compare label tables, never
  walls across tripwire modes.
- B1 and B2-option-2 and B5 are Tier A (bit-identical) — gates must agree
  exactly. B2-option-1 and any B3 reduction-shape change are contract
  renegotiations — solo lands, judged by the D4 gate.
- B4/B5 touch publication-adjacent paths: the standing rule — skipped work
  is never skipped publication; receipts/stamps written every advance;
  tripwires (`state[21..26]` family, `control[13]`) stay on in test lanes.
- Noise: ±5% within-arm; single-run 2–3% deltas measure nothing.

## What NOT to do

- **Don't chase dispatch counts.** The 771 dispatches are not the wall; the
  wall is per-thread latency chains (B1), pass-boundary-serialized tiny
  solves (B3), and from-scratch rebuilds (B5). Dispatch reduction is a
  side effect, not a goal.
- **Don't micro-optimize the ladder walk** (probe counts, hash tweaks,
  early-outs). B1 removes 90%+ of ladder walks; tuning the surviving 51K is
  noise. Same for fusing the redistance sweeps before B4's worklist exists.
- **Don't delete the superaccumulator casually.** It is the D4 primitive.
  B2 gives the two sanctioned paths; anything else is an unplanned contract
  change.
- **Don't generalize the resident solve upward.** It is a ≤4K-row tier. The
  wide pipelined path remains the scaling story; the cutover plan's verdict
  on one-workgroup executors at 65K rows stands.
- **Don't apply the activity handoff's "don't optimize solver iterations"
  advice to this lane.** That claim (`docs/losasso-activity-handoff.md:36-49`)
  is true of the structured/Power lane (budget 10, retirement live). This
  lane encodes the 40-iteration hard ceiling and its retirement path is
  bypassed (see anatomy) — B3a is real work here, not folklore-chasing.
- **Don't judge steady-state wins on the first frame** (19 cold iterations)
  or first-frame wins on the steady arm. Bet 0 gives both tables.
- Don't baseline against the 08-06 capture — it was contended (Codex 21.5 ms
  GPU in-window).
