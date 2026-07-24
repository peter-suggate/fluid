# Algorithmic Simplification Plan — one algorithm per quantity

**Written 2026-07-25.** Baseline: 98.03 ms/advance free-run, mini dam UI lane
(engine-collapse tree on e0441ee). Companions:
`docs/STAGE_COLLAPSE_PLAN.md` (landed), `docs/DATA_MODEL_COLLAPSE_PLAN.md`
(D1 mask authority / D2 state diet / D3 minimal trust — runs in parallel
with this plan and shrinks further as this plan deletes subsystems).

## 0. Thesis

The frame's remaining cost is not any algorithm being slow — it is the same
physical quantity computed by two or three parallel subsystems that then
need reconciling (restriction passes, authority gates, generation pairing).
Each item below picks ONE algorithm for one quantity and deletes the others.
Target steady-state frame: **transport → warm CPT → face sampling → one
conditioned solve → harvest.**

These are algorithm changes, not refactors: they deviate from the current
code's (and in A2's case the paper's) letter while preserving its intent —
∇q·∇φ = 0 extension, |∇φ| = 1 distances, volume conservation. Every item
ships behind an env A/B gate and passes the accuracy bars in §3 before its
legacy path is deleted.

## 1. The four simplifications

### A1 — One distance field (delete two of three Eikonal solvers)
Today: fine JFA-CPT (warm, 8.7 ms, authoritative) + coarse 8-pass Jacobi
redistance schedule (power-coarse-levelset) + face-band 16-round
tetra-Eikonal phi extension. All compute distance to the same interface.
- Coarse phi in-band := restriction of fine phi (already computed in
  harvest). The coarse schedule's audited consumers need only: seeds for
  bricks entering the band + far-field boundary authority.
- Far field := frozen between topology epochs; refreshed by a few dilation
  passes on epoch commit (far-field phi in a dam break changes at the pace
  of the coarse interface, not per frame).
- Band phi at face-band rows := sampled from the fine CPT (one level down),
  deleting the 16-round relaxation loop, its frontier machinery, and the
  tetra Eikonal solves.
Deletes: advect/correct/redistance×8/publish/commit coarse schedule (as a
per-frame subsystem), extendBandPhi loop, bandPhi frontier.
Env gate: `FLUID_DISTANCE_AUTHORITY=triple|cpt`.

### A2 — CPT velocity extension (retire the Section-5 mesh-graph machinery)
Today: band-row directory + Delaunay transition adjacency + incidence +
transient power graph + 8×2 repair waves propagate face velocities into air
by marching mesh connectivity. Replacement (Track G, now cheap because the
CPT is warm and validated): per face needing velocity,
`cp = x − φ∇φ` from the fine CPT → sample the liquid-side velocity
interpolant at cp → project on the face normal. Also serves transport's
air-side sampler (the 5×5×5×levels fallback scan dies with it).
The paper fast-marches the power diagram, but its goal is ∇q·∇φ = 0
extension; the closest-point method is the standard equivalent, and the
paper's Table 2 treats this step as near-free — that is the property being
bought. Retires most of webgpu-octree-face-closest-point.ts (~4,600 lines).
Env gate: `FLUID_SECTION5_EXTENSION=march|cpt` (name reserved by the V2
plan; implement it now).

### A3 — One solver, one operator, better conditioning
Today: Galerkin (+ per-solve RAP refresh) AND MGPCG/SPGrid AND a disabled
persistent3 lane; the operator assembled 2-3×. Choose one survivor —
catalog-stencil operator + mask-based V-cycle aligns with D1 — and delete
the rest. Then cut iteration COUNT, not iteration cost:
- Residual-gated cycles (stop encoding fixed 20; coarsest CG breaks on
  convergence).
- **Hydrostatic split**: solve for deviation p′ = p − ρg·h; the dominant
  field moves into the RHS. For gravity-dominated scenes this is worth
  several V-cycles and improves warm-start quality between frames.
Env gate: `FLUID_PRESSURE_SPLIT=absolute|hydrostatic`; solver selection
already env-driven.

### A4 — Volume control as a solve source term
Today: measure (full-lattice reductions) → uniform shift → re-measure.
Replacement: one scalar drift per frame folded into the divergence RHS
(standard volume controller), or equivalently the analytic shift applied
during the CPT commit already writing phi. The volume module stops existing
as a pipeline stage; keep its measurement kernel behind the debug lane for
the accuracy bars below.
Env gate: `FLUID_VOLUME_CONTROL=sweep|rhs`.

### Explicitly NOT taken
Factor-m transport → single RK2: post-hint-carry the segments are cheap and
this genuinely changes the paper's advection semantics for little return.

## 2. Sequencing

1. **A2 first** (biggest code deletion; depends only on the warm CPT, which
   is already validated; unblocks A1's band-phi deletion).
2. **A1** (needs A2, since the band-phi consumers go through the same
   sampling path).
3. **A4** (small, independent; do alongside A1).
4. **A3** last (touches the solve; land after the extrapolation changes so
   accuracy regressions are attributable).
Keep both paths per item until its bars pass at BOTH scales (§3.6), then
delete the legacy path in the same PR that flips the default — dead
alternates left in tree are how the current three-solver state happened.

## 3. Assessing accuracy

### 3.1 The prime lesson: a faster number can mean a broken stage
Redistance "regressed" 17→60 ms because the fix made it actually run; the
17 was a silently under-working stage (fail-closed gate publishing zero
work). And `benchmark:*-performance` lanes skip quality gates and exit 0
while broken. Therefore: **every timing claim ships with work counters** —
seeds planted, pages flooded, faces extended, cycles executed, rows
carried. A perf win with a work-counter drop is a bug until proven a
deletion. Wire the counters into the physics-trace JSON so the A/B diff is
mechanical.

### 3.2 Field-level bars (run on both A/B arms, same seed)
- **Interface IoU per step vs the pre-change baseline** (existing harness
  bar): sliding window, alarm on trend not single steps. Expect small
  divergence from reordered float reductions; alarm at the bar you already
  use for cross-backend parity.
- **|∇φ|−1 residual histogram** in the band (A1): the JFA-CPT and the
  deleted solvers must agree; the histogram must not fatten at the seam
  where fine-sampled band phi replaces the tetra-Eikonal values.
- **∇q·∇φ residual** sampled on air-side faces (A2): the extension PDE's
  own residual is the direct quality metric for CPT extension vs marching;
  compare distributions, not maxima (marching has its own error).
- **Volume drift over 500 steps** (A4 and globally): the existing 500-step
  minimal gate; drift must be ≤ the sweep-based controller's. Watch for
  low-frequency volume *oscillation* — an over-eager RHS controller rings.
- **Energy ledger** (existing): extrapolation is the classic place energy
  injection hides. A2's bar: post-projection KE trend must not rise
  relative to baseline; attribute with the ledger's loss accounting.

### 3.3 Behavioral tests that catch what field metrics miss
- **Still-water equilibrium** (A3-hydrostatic, A2): a resting tank must
  stay at rest to solver tolerance — velocities ~0, interface flat, no
  creeping current at the walls. Run 500 steps. This test is nearly free
  and catches extension bias, hydrostatic-split sign errors, and volume
  controller ringing in one scene.
- **Dam-break front position vs time** against the pre-change baseline
  (and the Martin–Moyce data the literature uses): front-stall history
  ([[quadtree-front-stall]] class of bug) shows this catches coupling
  errors that IoU smears over.
- **Thin-sheet/droplet survival**: the mini dam's late-stage splash;
  CPT extension near thin features is where cp = x − φ∇φ degrades (∇φ
  ill-defined at skeleton points). Count disconnected-component births vs
  baseline; A2 keeps the carrier fallback for |∇φ| far from 1.
- **Authority-health assertions on every run**: coarse/fine/power
  generation pairing must stay locked (the 30-100× stall regime was a
  generation stall). The quality lane must FAIL, not warn, on
  rolledBack/MALFORMED — never assess accuracy on a run that entered
  degraded authority.

### 3.4 A/B discipline
- One env flag per item; flip exactly one per measurement run.
- Same seed, same step count; compare per-step traces, not just endpoints.
- Keep UI-lane and Dawn-lane configs verified identical before comparing
  ([[octree-damping-investigation]]: UI-vs-Dawn config divergence produced
  a false regression once already).
- Reordered reductions mean bit-parity is dead; do NOT chase it. Quantify
  instead (3.2). Where determinism is claimed (Morton bit-scan ordering),
  test *that* — same input → same output across runs — not equality with
  the legacy path.

## 4. Ensuring the performance wins are real

- **Free-run numbers only** for cross-build claims. The segmented probe
  adds a per-advance-constant ~63 ms that concentrates into the biggest
  segments — per-stage attribution is for *locating* cost
  (`FLUID_ENGINE_SPLIT=fine`), never for cross-build totals.
- **Measure the late-step regime.** Historic pathologies start ~step 15+
  (post-stall) and were nondeterministic (15.8 s vs 46 s for the same run).
  Report median and p90 over steps 20-60, not the 62-advance mean alone;
  any A/B where variance changes materially needs a repeat run before
  drawing conclusions.
- **Pair wall time with dispatch/pass counts** (the existing
  FLUID_MAX_DISPATCHES/PASSES gates) as *sanity* metrics — but remember the
  measured lesson that dispatch cuts alone were wall-neutral; the claim to
  verify for each deletion is wall ms AND the disappearance of the deleted
  kernels from the trace.
- **Clean-room measurement**: no `vinext dev`, no browser WebGPU tabs, no
  concurrent agent editing the tree (verify `git status` before/after),
  clear stale `/tmp/fluid-webgpu-exclusive.lock` only when its owner PID is
  dead. Dawn occasionally SIGABRTs (absl map) — a crash is a rerun, not a
  data point.
- **Two scales, always**: the mini dam (16³, where "narrow band" ≈ whole
  domain) and a large lane (vast-ocean class). A1/A2 wins are *understated*
  at 16³ (the deleted subsystems scale with rows); a change that wins at
  16³ but not at scale is suspect of measuring launch overhead.
- **Watch the gates you weaken.** A1 removes redundant distance solvers
  that today mask each other's failures; after it lands, the epoch
  validation reduce (D3) is the only net. Land D3's reduce before deleting
  the legacy solvers, and run the deep-validate lane in CI nightly.
- **Declare the expected number before the run** (per item: A2 deletes the
  15.6 ms topology-build + 5.3 adjacency + repair waves ⇒ expect row-b
  free-run share to drop by roughly their deflated sum). A win that arrives
  without its predicted mechanism (counters, trace) is investigated, not
  banked — that rule is what caught the redistance fidelity bug.

## 5. End state

Frame = transport (banded, hint-carried) → warm CPT (collar repair) → face
sampling (CPT extension) → hydrostatic-split V-cycle (residual-gated) →
harvest (one fused sweep). One distance algorithm, one extension algorithm,
one solver, one volume controller; the trust model is D3's epoch reduce.
On current arithmetic that is a single-digit-to-low-teens ms frame at the
mini dam scale, with the deleted subsystems' capacity-scaling gone at size.
