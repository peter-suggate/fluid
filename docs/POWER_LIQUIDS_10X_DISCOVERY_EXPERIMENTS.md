# Locating the 10x — discovery experiments handoff

Date: 2026-07-30. Companion to `POWER_LIQUIDS_FINE_BAND_10X.md` (the
implementation ladder for *known* waste). This document is different in kind:
nothing here optimizes anything. Each experiment is an **oracle, ablation, or
speed-of-light bound** that puts a hard number on one hypothesis about where
the frame's time is architecturally unnecessary. The precedent is band 0 in
the 2026-07-29 band sweep: a deliberately quality-invalid probe that bounded
the fine-band ceiling (23% clean, 43% attributed) in one run and re-priced a
whole family of proposals. We need that instrument for every hypothesis, not
just band width.

Rules of the genre, up front:

- A probe may be quality-invalid **by design** (like band 0). It must say so,
  and its number is a *bound*, never a shipped result.
- Probes may use host-side encode skips and other mechanisms the product
  forbids (`hostSchedulingUsesReadback = false` is a product contract, not a
  probe contract). The product form of any winning probe must be re-derived
  under the real constraints.
- Every wall number comes from clean lanes, interleaved, fixed `--steps`,
  tripwires on — same as always. Attribution numbers come from the xctrace
  skill and are rankings.
- Each experiment ends with a **decision rule**: which architecture bet it
  funds or kills. An experiment without a decision rule is tourism.

## 0. The three bets

A real 10x on the non-pressure frame (~28.5 ms attributed, ~9% occupancy)
must come from one or more of:

- **Bet R — Redundancy.** Most of what is computed each frame is bit-identical
  (or below-quantization-identical) to last frame's output. If true, the
  winning architecture is *persistent everything, delta repair* — the fine
  lane and maintenance families join the fingerprint/fail-closed regime the
  SPGrid hierarchy already lives in. Evidence for: 2.6% membership churn
  census, 0.004 s frames, band-1's outsized win. Evidence against: the 95%
  dirty classification might be *honest* (fractions/velocities really change
  everywhere wet).
- **Bet L — Lean rebuild.** The rebuilds are semantically necessary but run
  10–30x above their memory-bandwidth floor because of per-tap validation,
  scattered pointer chases, capacity-shaped launches, and division-heavy
  addressing. If true, the winning architecture keeps the rebuild schedule
  but runs it on packed, pre-resolved, shift-addressed streams (the
  "cache-efficient mirror", in its sanctioned addressing-only form).
- **Bet C — Critical path.** The frame is latency-bound, not work-bound: a
  ~170-pass dependency chain in which most passes hold the machine at <10%
  occupancy means wall ≈ (chain length × per-stage latency), and no amount of
  per-kernel leanness fixes a serial chain. If true, the winning architecture
  is dependency-shortening: phase-packing (`POWER_LIQUIDS_5X_GPU_PLAN.md:185-218`),
  wider fused phases, speculative overlap of independent families.

These are not exclusive; the experiments below produce a split of the 28.5 ms
across R/L/C. The 10x exists iff R + L + C headroom multiplies to ~10 on the
slices the product needs. The experiments are ordered by information per hour.

## X-1. Hydrostatic still-lane itemization — the coherence tax ledger

**Hypothesis (Bet R).** In still water nothing changes; a perfectly coherent
architecture's advance approaches the pressure floor. Every attributed
millisecond a hydrostatic lane spends outside pressure is, *by construction*,
computation of an unchanged result — a line-item inventory of the 10x.

**Method.** The scenes already exist and are authored oracles:
`large-hydrostatic-offset` (intentionally cell-cut free surface,
`lib/scenes.ts:105-111`) and `tiny-hydrostatic-two-level` (`:76-82`).

1. Add a `hydrostatic` lane to `tools/power-dam-lane-environment.ts` mirroring
   the `large` lane's grid/band/factor so brick counts are comparable
   (the lane table is the single source of truth; both consumers pick it up).
2. Clean wall: `node --import tsx tools/benchmark-power-dam.ts
   --lane=hydrostatic --steps=240`, interleaved against `--lane=large`.
3. Attribution: `node --import tsx tools/profile-mini-dam-xctrace.ts
   --counters --isolate-pass-labels --lane=hydrostatic --steps=240
   --out=artifacts/discovery/hydrostatic-itemized`.
4. Deliverable: the §1.2-style family table for still water, side by side
   with the churn table, plus the same for `FLUID_OCTREE_ROW_DELTA_CENSUS=1`
   (expect added/retired ≈ 0; record what dirty does — if dirty is still ~95%
   on a *still* scene, M-2 is answered for free and Bet R's classification
   arm is funded immediately).

**Reading it.** Watch for the trap the ULTIMATE doc documents: the mini lane
has "no settled regime" (~110 ms flat) — that flatness measured on a *churn*
scene that never settles is itself weak evidence for R being large; the
hydrostatic lane makes it rigorous because the ground truth (zero change) is
authored, not assumed.

**Decision rule.** Let `S` = still-lane non-pressure attributed ms, `K` =
churn-lane non-pressure attributed ms (~28.5).
- `S/K > 0.6`: Bet R is the main event; the coherence architecture is worth
  the full Wave-2+ investment. Every family with still-cost ≈ churn-cost goes
  on the delta-repair list with its measured ceiling.
- `S/K < 0.2`: the frame already collapses when nothing changes — the 10x is
  not in redundancy; fund Bets L/C.
- JFA/transport specifically: their still-lane cost is the exact ceiling of
  E-7 (unchanged-page carry) with zero soundness risk in the measurement.

**Cost:** ~half a day (lane entry + two runs + one capture). No product code.

## X-2. Frame-redundancy hash census — how much output is literally identical?

**Hypothesis (Bet R).** X-1 measures a scene where everything is redundant.
This measures redundancy on the *churn* lane — the fraction of each buffer
whose generation-N content equals generation N-1, per family.

**Method.** A diagnostic-arm pass family (`FLUID_REDUNDANCY_CENSUS=1`), one
64-lane hash kernel per arena, patterned exactly on the lag-1 census drain
(`censusTick`, `lib/webgpu-octree-fine-levelset-transport.ts:457-486`; its
mapAsync-lag rationale applies verbatim):

- fine φ pages: hash 64 words/page → 1 u32/page; compare against last
  generation's hash keyed by **logical brick key** (physical ids are
  recycled; the remap tables from the JFA seed carry show how to follow
  identity across the A/B transaction);
- sample flags pages (carries the CPT codes — JFA's true output);
- structured face-velocity rows (hash per row);
- transport class worksets, summary entries, power descriptors, boundary
  slots, air-support identity words: hash per record.

Two counters per family per generation: `identicalPages/totalPages` and
`identicalBytes/totalBytes`, drained lag-1 like every census, printed as
NDJSON. Run 500 steps on `large` and `mini`, plot fraction-identical vs step
index (early churn vs late pool matters — the answer is a curve, not a
number).

**Also measure the ε-redundant fraction:** a second variant quantizes φ to
the redistance tolerance and velocity to the projection tolerance before
hashing. Bit-identical is the floor; below-quantization-identical is the
honest ceiling for skip-with-carry schemes.

**Decision rule.** Per family: redundancy > 70% funds that family's
delta-repair item (E-7/E-9 in the ladder) with a measured ceiling; < 30%
kills it and moves that family's budget to Bet L. The gap between
bit-identical and ε-identical prices how much of the win needs quantized
fingerprints (harder, Gate B) vs exact ones (easier).

**Cost:** 1–2 days (one WGSL hash kernel, N bind groups, census drain). Zero
product-path risk (diagnostic arm only).

## X-3. Dirty-oracle probe — is the 95% dirty set honest?

**Hypothesis (Bets R+L).** The coarse census says membership churn is 2.6%
but 95% of rows are dirty and `meanAffected == meanDirty` exactly. If dirty
over-inclusion is conservative slop (or a counter aliasing bug), fixing
*classification alone* shrinks every downstream dirty-shaped dispatch by an
order of magnitude without touching any kernel.

**Method.** Two arms:

1. **Counter forensics (no GPU change, do first):** extend the row-delta
   census drain (`lib/webgpu-octree.ts:3186-3230`) to decompose dirty by
   promotion reason — the fine lane already packs 6 reason codes into
   `supportCandidates` bits 26+ (`lib/webgpu-octree-fine-levelset-topology.ts:2033`,
   counted at `:2072`) and nobody has ever printed the histogram. Also
   determine whether `meanAffected == meanDirty` is a real equality or the
   same counter read twice.
2. **Probe arm (quality-invalid allowed):** `FLUID_DIRTY_ORACLE=membership`
   replaces the dirty classification with membership-delta + 1 ring, on both
   the coarse rows and `classifyFineAffectedPages`. Run the 500-step gates
   anyway and *record* which ones fail — the failure pattern says which
   consumers genuinely need the wide dirty set (that is design input, not
   noise). Measure the wall with everything downstream unchanged.

**Decision rule.** Wall drop > 15% with only ring-1 dirty ⇒ classification is
the cheapest structural lever in the codebase; a sound narrowing (per-reason
radii, fingerprint-gated promotion) gets specified next. Wall drop < 5% ⇒
dirty width is not what the downstream cost scales with (launch shapes are —
E-6), and M-2 closes.

**Cost:** arm 1 hours; arm 2 ~a day.

## X-4. Maintenance-freeze probe — the full price of rebuilding topology

**Hypothesis (Bet R).** Grading closure, refinement, worksets, descriptors,
boundary slots, owner pages, air-support identities: the entire maintenance
superstructure exists to track a surface that moves ≤ 1–2 cells/frame. Freeze
all of it and the residual frame (transport + JFA + pressure on frozen sets)
is the floor the maintenance families must be judged against.

**Method.** `FLUID_FREEZE_TOPOLOGY_AFTER=N` (probe flag): after generation N,
the encode path skips the octree reset/refinement/grading block, fine
topology delta chain, power descriptor/topology/boundary/air-support encodes
— host-side skip is fine for a probe. Interface transport and redistance keep
running on the frozen page set (the band is wide enough for a few hundred
frozen steps of a *near-settled* scene; on a churn scene the fluid escapes
the band and the run is invalid — pick N after the impact, steps 300+, and
keep runs short). Compare per-advance wall N±60 steps, and capture one
itemized trace of the frozen regime.

**Decision rule.** This is the direct measurement of "maintenance tax" `T`.
The delta-repair architecture's whole promise is ≈ `T × (1 − churn%)`. If
`T` < 20% of the frame, Wave-2 coherence work on maintenance is deprioritized
regardless of how elegant it is; if `T` > 40%, it is the program.

**Cost:** ~1–2 days (encode-skip flag + careful validity window). Probe only;
never a product mechanism.

## X-5. Speed-of-light rooflines — how far from physics are the hot kernels?

**Hypothesis (Bet L).** The irreducible cost of the fine band is small:
~6–18k bricks × 64 samples × a handful of channels. If a stripped kernel
doing only the irreducible memory traffic runs 10–30x faster than the real
stage, the gap *is* the addressing/validation tax, and packed-stream rebuilds
buy that gap without any coherence risk.

**Method.** Standalone tool patterned on
`tools/benchmark-octree-section63-bandwidth.ts` (the precedent for exactly
this genre), running against real captured arenas (sizes from the large lane:
32,768-page capacity, 14–18k active):

- **SOL-0 stream:** read+write every active φ page once, packed compact,
  shift-addressed. This is the bandwidth floor for *any* per-voxel pass.
- **SOL-1 gather:** per sample, 8-tap trilinear from pre-resolved page slots
  (halo table resolved once per workgroup, JFA-style) — the transport
  `sampleFine` floor without directory validation.
- **SOL-2 flood:** the actual 28-candidate JFA inner loop on packed bricks
  with B4 shift addressing — the redistance floor.
- **SOL-3 backtrace:** SOL-1 iterated `substeps` times with the real
  air-owner record loads replaced by a flat pre-gathered array — the
  "transport with staged addressing" (E-5) end state, measured before anyone
  builds it.

Report each as µs/Mvoxel and GB/s against the ~400 GB/s part, next to the
measured stage times from `artifacts/scene-size-overhead/large-current/`.

**Decision rule.** Per kernel: `actual/SOL > 8` funds the Bet L rework of
that kernel with a measured ceiling (and SOL-3 specifically prices E-5 before
it is implemented); `actual/SOL < 3` means the kernel is already near
physics and *only* Bets R (skip it) or C (overlap it) can help — do not fund
micro-optimization there.

**Cost:** ~2 days. Zero product risk (standalone tool).

## X-6. Critical-path decomposition — is the frame long or is it thin?

**Hypothesis (Bet C).** 48.75 ms attributed at 9% mean occupancy could be
either "too much work" or "a serial chain of small stages". They need
opposite architectures. The discriminator: compare wall against
`Σ(stage_ms)` (work) and against `Σ(critical-path stage_ms)` (chain).

**Method.** Clean-mode (no isolation) per-pass GPU timestamps already exist —
`FLUID_GPU_PASS_TIMESTAMPS=1` injects begin/end into passes the frame already
opens, no extra passes (`tools/webgpu-smoke-gpu-audits.ts:78-262`, coverage
honesty via `coverageRatio`). The dependency DAG is derivable from
`tools/webgpu-data-flow-manifest.ts` (pass labels → pipelines → buffers)
— an edge wherever pass B reads a buffer pass A wrote, within one advance.

1. Capture per-pass begin/end for ~25 advances on `large`, clean mode.
2. Build the DAG from the manifest; longest path by measured duration.
3. Deliverables: (a) `wall`, `Σwork/machine-width`, `critical-path` triple;
   (b) the top-10 chain segments (which families own the spine); (c) overlap
   already achieved (passes whose GPU intervals overlap — Metal does overlap
   independent encoders sometimes; measure, don't assume).

**Decision rule.** `critical-path / wall > 0.7` ⇒ the frame is chain-bound:
phase-packing and family-overlap (encode independent families into
concurrently-schedulable spans) get promoted from "last" to "the plan", and
per-kernel work removal is re-priced (it only helps stages *on the spine*).
`critical-path / wall < 0.4` ⇒ Bet C dies; the 5X plan's "phase-packing last"
ordering stands.

**Cost:** 1–2 days, mostly the DAG join. No product changes.

## X-7. dt-elasticity — does cost scale with change or with existence?

**Hypothesis (Bets R+C).** In a coherence-respecting architecture,
per-advance cost scales with per-frame displacement (∝ dt); in a
rebuild-everything architecture it is dt-invariant. The elasticity
`d(cost)/d(dt)` per family is a one-afternoon measurement of how coherent
each family *already* is.

**Method.** `FLUID_MAX_DT=0.004 / 0.002 / 0.001` on the large lane at fixed
simulated time (`FLUID_TARGET_S=2`, so 500/1000/2000 advances), clean walls
per advance + one itemized capture at 0.004 and 0.001. Families whose
ms/advance is flat across dt are rebuild-dominated; families that shrink
with dt already track change. (JFA should shrink — the displacement ladder
exists; transport should shrink via `ceil(vmax·dt/h)` substeps; if either is
flat, its coherence mechanism is not engaging, which is itself a finding.)

**Decision rule.** Flat families inherit Bet R budget in proportion to their
size. A family that is flat *and* near its X-5 roofline *and* off the X-6
spine is fundamental cost — subtract it from the achievable-10x arithmetic
honestly.

**Cost:** hours.

## X-8. Cost-vs-change correlation — free, from logs that already exist

**Hypothesis (Bet R).** Same question as X-7 asked within a single run: the
dam evolves from impact chaos to sloshing pool; per-advance measured change
(`pageDelta[9]` displacement, census added/dirty counts) varies by orders of
magnitude across the 500 steps. Does per-advance wall follow it?

**Method.** Pure post-processing: the progress heartbeat
(`windowWallPerStep_ms`, the mechanism the ULTIMATE doc endorses over the
quiescent lane), the row-delta census NDJSON, and (after M-1 fixes it) the
fine workset census, joined on step index. Scatter wall vs change; compute
the elasticity. The +46–63% late-run drift already documented for
`Advect fine phi common` / `Stage exact power topology carry` suggests cost
correlates with *band size*, not with *change* — confirm and quantify per
family.

**Decision rule.** Correlation with band size but not with change is Bet R's
cleanest indictment: the frame prices existence, not motion. Families where
even band-size correlation is absent are fixed-overhead (Bet C spine
candidates — cross-reference X-6).

**Cost:** hours. No GPU work at all.

## X-9. Clean-room band-pipeline spike — the existence proof

**Hypothesis (all bets, terminal).** The definitive question: on this GPU,
what does the fine-band update *cost when written from scratch* with
everything this document's evidence recommends — packed brick stream in
brick-key order, apron-free but halo-table-resolved addressing, shift-only
index math, per-brick substep bounds, JFA warm ladder, no capacity-shaped
anything, worksets carried not rebuilt?

**Method.** Throwaway standalone tool (2–3 days, explicitly not product
code): snapshot one real mid-run frame's arenas (worklist, metadata, φ,
flags, velocities, air-owner records) from the large lane; implement the
minimal pipeline — classify (delta-driven) → advect (per-brick substeps,
staged addressing) → redistance (B4 floods, displacement ladder) → summarize
— as ~6 kernels; step it 60 times against the frozen velocity field; measure
ms/advance. It is *wrong physics* (frozen coupling, no pressure, no topology
churn) — that is fine; it is the measured lower bound for the architecture
family, the number the incremental ladder converges toward.

**Decision rule.** Spike ≤ 2 ms/advance for the transport+JFA+summary slice
(vs ~10 ms attributed today): the 10x exists and the gap decomposes into the
X-2/X-4/X-5 numbers, which then order the product work by measured value.
Spike ≥ 6 ms: a 10x on these slices does not exist on this hardware at this
resolution — publish the number, and the program target becomes the honest
multiple the probes support (with the remainder sought in Bet C overlap and
the pressure lane instead). Either outcome ends the guessing.

**Cost:** 2–3 days. The most expensive experiment here — run it last, after
X-1/X-3/X-5 have sharpened what "minimal" means.

## Sequencing and the decision matrix

Recommended order (information per hour, dependencies respected):

| order | experiment | cost | primarily prices |
|---|---|---|---|
| 1 | X-1 hydrostatic ledger | ½ day | R (global) |
| 2 | X-8 cost-vs-change join | hours | R (per family) |
| 3 | X-7 dt-elasticity | hours | R + engagement check |
| 4 | X-3 dirty oracle | 1 day | R (classification arm) |
| 5 | X-5 rooflines | 2 days | L (per kernel) |
| 6 | X-6 critical path | 1–2 days | C |
| 7 | X-2 redundancy hashes | 1–2 days | R (exact ceilings) |
| 8 | X-4 maintenance freeze | 1–2 days | R (maintenance tax) |
| 9 | X-9 clean-room spike | 2–3 days | existence proof |

M-1/M-2 from the companion ladder are prerequisites where noted and are
absorbed by X-3/X-8.

Reading the matrix when the numbers are in:

- **R large (still-cost high, redundancy high, cost flat vs change):** the
  10x is *persistent everything, delta repair*. Product order: dirty
  classification fix → fine-lane fingerprint carry (E-7/E-9) → maintenance
  delta cones (E-3, grading/descriptors/boundary) — each with its ceiling now
  measured, not estimated.
- **L large (rooflines 8–30x, redundancy low):** the 10x is *lean rebuild*:
  packed streams, staged addressing (E-5), shift math (E-1), compacted
  launches (E-6), per-brick substeps (E-4). X-9's spike is the reference
  implementation.
- **C large (critical path ≈ wall):** promote phase-packing; group the frame
  into few wide phases; overlap independent families; only spine stages get
  kernel-level attention.
- **All three moderate:** the 10x is a product of ~2x from each — which is
  still a 10x program, but one that must land all three waves; the companion
  ladder's ordering already reflects that composition.

One warning to carry into every reading: refutation #9 and the encoder-
isolation measurement prove this frame hides *negative* space — removing
provably-dead dispatches has measured slower. Any probe whose win is small
(< 5%) is inside the machine's drift and the architecture's nonlinearities;
only the big, replicated, interleaved numbers move bets.
