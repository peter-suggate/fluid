# Power Liquids Without a Fine Band — Coarse-Octree-Only Plan

Status: proposal (2026-07-31). Companion to `docs/papers/aanjaneya-2017-power-liquids.txt`,
`POWER_LIQUIDS_FINE_BAND_10X.md`, `POWER_LIQUIDS_10X_DISCOVERY_RESULTS.md`.

Goal: adapt the Aanjaneya 2017 pipeline so the simulation no longer *needs* the
4×/8× fine narrow band — the sparse coarse octree becomes a self-sufficient
level-set authority, the raster surface is reconstructed faithfully from
octree-resolution phi, and the pressure solve and octree topology updates consume
coarse phi directly.

End-state architecture (per direction from 2026-07-31): the fine band returns
later as a **regional overlay** — enabled only where it earns its cost (low
liquid-volume regions, high energy/velocity regions), correcting the coarse
tracker where resident. This constrains the plan in two ways:

1. The band machinery must **survive**, not be deleted — decoupling means the
   simulation is *correct with zero fine coverage*, and gracefully better with
   partial coverage. (An earlier row-native "delete the band" route is kept in
   §5.4 only as an analyzed-and-rejected alternative; its problems P7/P8 are why.)
2. Every simulation-critical consumer contract must acquire a legitimate
   "intentionally uncovered — coarse authority" state, distinct from today's
   fail-closed "missing coverage is an error" state.

---

## 1. What the fine band actually does today (removal inventory)

The paper (§5) and our implementation agree on the division of labor: **velocities,
pressure, and the Poisson discretization never touch the fine band.** MGPCG, the
SPGrid V-cycle, the Section 4.3 preconditioner and the Section 6.3 operator have
zero fine-band reads. The band feeds exactly these consumers:

| # | Consumer | Where | Criticality |
|---|----------|-------|-------------|
| 1 | Ghost-fluid θ + liquid row mask | `lib/webgpu-octree-structured-boundary.ts:206-211` (`fineSample`, `classifyStructuredLiquidRows`, `resolveStructuredBoundarySlots`) | simulation-critical |
| 2 | Wet-owner override + refinement evidence | `lib/webgpu-octree.ts:6548-6628, 6665-6691, 7069-7089` via the fine summary mip | simulation-critical |
| 3 | Coarse-phi correction (restriction) | `lib/webgpu-octree-fine-to-coarse-levelset.ts` → `encodeCoarsePhiCorrection` (`webgpu-octree.ts:3723-3761`) | simulation-critical |
| 4 | Interface tracking itself: transport, topology, JFA-CPT redistance, volume correction | `lib/webgpu-octree-fine-levelset-{transport,topology,redistance,volume}.ts` | simulation |
| 5 | Air-velocity support demand | `lib/webgpu-octree-air-velocity-support-gpu.ts:487-527` | simulation |
| 6 | Raster surface (marching tetrahedra) | `lib/webgpu-water-global-fine-{classify,tetra,polygonise}.ts` | rendering |

Cost being removed (large lane, 48.75 ms attributed frame, `POWER_LIQUIDS_FINE_BAND_10X.md:44-53`):
fine transport 5.13 ms + fine topology/summary/volume 4.12 ms + fine JFA 4.01 ms
≈ **13.3 ms (~27%)**, plus the fine share of air support (3.47 ms total) and
~94–210 MB of arenas.

Honesty check before committing: X-9 in `POWER_LIQUIDS_10X_DISCOVERY_RESULTS.md:196-206`
demonstrated a clean-room packed band pipeline at **0.43 ms/advance at 14k pages** —
i.e. most of the 13.3 ms is recoverable *without* any quality loss by repacking the
band lane. Removal buys structural simplification and memory, not much more speed
than the repack ceiling. Section 8 makes this a decision gate.

---

## 2. Key structural insight: two different things called "the fine band"

The band conflates two independent properties, and only one of them is the target:

1. **Resolution** — phi sampled at `fineFactor` = 4–8× the finest octree lattice.
2. **Structure** — a brick-addressed narrow band whose keys come from the global
   finest lattice, *independent of octree rows* (`lib/octree-fine-levelset-bricks.ts:1-8`).
   This is what decouples tracking from octree topology churn: bricks survive
   octree rebuilds, carry their own A/B generations, rollback, and fail-closed
   publication gating.

**Dropping property 1 gives the entire quality trade being accepted.
Dropping property 2 is where almost all of the new analytic risk lives** (Section 6:
P4, P7, P8, P11), because the octree must then absorb the band's roles — phi carry
across every topology rebuild becomes lossy resampling of the *only* copy, and the
uniformly-finest refinement shell must widen to cover backtrace + redistance
support, growing the pressure solve (the most expensive lane) to shrink a cheaper one.

This split maps directly onto the hybrid end-state:

- **Stage 1 (decouple): a factor-1 band instance becomes the always-on baseline
  tracker.** Extend `FineLevelSetFactor` to `1 | 4 | 8` and run the existing band
  machinery at octree resolution. The simulation then works with zero *fine*
  (4×/8×) coverage: surface, θ, and tracking are all octree-resolution, sample
  counts drop ~16–64×, and every existing fail-closed contract carries over
  unchanged. This is "decoupled from the fine band" in the resolution sense while
  deliberately keeping the band *structure* — which the regional future needs
  anyway, and whose brick keys are octree-independent (no P7/P8 exposure).
- **Stage 2 (regional overlay, future): a second band instance at factor 4/8,
  gated by a coverage policy** (low liquid volume, high energy/velocity),
  correcting the factor-1 baseline where resident. §5 designs this and its seam
  problems; nothing in Stage 1 has to be undone for it.

One deliberate consequence: there is **no coarse tracker to build from scratch**.
Today coarse phi is *derived* — restriction from the fine band re-creates it every
step (`encodeCoarsePhiCorrection`). Simply switching the band off freezes the
surface (the known silent failure mode). The factor-1 instance *is* the
self-sufficient octree-resolution tracker, obtained by parameter change rather
than by writing new advection/redistance kernels on the row graph.

Stage 1 answers every quality question (volume drift, θ jitter, topology-event
timing, drop oracles) at ~10% of the engineering cost of any rewrite, and is
reversible behind a flag.

---

## 3. Stage 0 — Baselines and quality harness (prerequisite)

No cutover work until these exist; several failure modes here are of the
"faster frame, frozen surface, still prints PASS" family
(`POWER_LIQUIDS_ULTIMATE_M1MAX.md:216-224`).

1. Capture paired baselines on mini (500-step), large (60/120-step), hydrostatic,
   ceiling-drop lanes: wall clock, attributed GPU per family, terminal pressure
   iterations, active brick/page counters.
2. Quality metrics added to the trace harness:
   - per-step and cumulative volume error (the volume corrector's measured
     pre-correction residual is the signal — log it, don't just consume it);
   - surface Hausdorff / mean distance between the extracted mesh and the
     current factor-4 mesh on identical trajectories (export both for N steps);
   - θ histogram temporal variance — `control.theta` already buckets θ into 8 bins
     (`webgpu-octree-structured-boundary.ts:416-428`); add per-step L1 delta of the
     histogram as a cheap parasitic-jitter proxy;
   - deforming-sphere / rigid-rotation tracking test at octree resolution
     (paper Fig. 7 setup) for dissipation measurement.
3. Re-run the wall-sticking free-fall drop oracles (see memory: ceiling/seam
   sticking fixes) — contact-line behavior is resolution-sensitive.
4. Tripwires (checked every N steps, fatal in tests):
   - surface displacement ≈ 0 while liquid kinetic energy > threshold (frozen surface);
   - terminal pressure iterations = 0 on a churn lane;
   - fine/coarse band count hitting `0xFFFFFFFF` sentinel;
   - cumulative volume drift budget per 100 steps.

---

## 4. Stage 1 — Factor-1 band (`FLUID_FINE_FACTOR=1`)

The band-width formulas are already factor-covariant — bands are authored in
*finest* cells and multiplied by `fineFactor` internally
(`planFineLevelSetBandFineCells`, pinned by
`tests/webgpu-octree-fine-levelset-bricks.test.ts:215`: `(3,8).transport ==
2×(3,4).transport`). At factor 1 the physical band widths are unchanged;
only sample density drops. Concrete work items:

### 4.1 Type/plumbing
- `FineLevelSetFactor = 1 | 4 | 8` (`lib/octree-fine-levelset-bricks.ts:62`);
  lift the hard factor-4 requirement at `gpu-startup.ts:77`.
- `planFineLevelSetLeafBrickBounds` (`webgpu-octree-fine-levelset-topology.ts:193-213`):
  `bricksPerFinestCell = fineFactor/4` becomes fractional. Invert the mapping —
  introduce `finestCellsPerBrick = 4/fineFactor` (4 at factor 1); brick key =
  `finestCell >> 2`. Leaf seeding becomes many-to-one (64 finest cells per brick);
  duplicate desired keys already collapse in `validateDesiredSeeds` / identity
  assignment — add a test pinning dedup at factor 1.
- Summary mip leaf→node mapping (`webgpu-octree-fine-levelset-summary-direct.ts`,
  pinned by `tests/octree-fine-summary-consumer.test.ts:17`): one summary leaf node
  now covers a 4³-finest-cell brick. Extend the dyadic-alignment table for factor 1.
- `planGlobalFineNarrowBandBrickCapacity` (`webgpu-octree.ts:723-784`) and lane
  configs: capacities shrink ~16–64×; keep planner clamps.

### 4.2 Transport
- Multi-step count `m = fineFactor` → 1. Replace the m-step forward-Euler backtrace
  with a single **RK2 (midpoint) backtrace** — at ratio 1 the paper's multi-stepping
  rationale disappears, but first-order single-step backtrace is too dissipative at
  the resolution that is now also the surface (see P2). Add optional
  BFECC/MacCormack behind `FLUID_COARSE_PHI_BFECC` for the quality ladder.
- `maximumBacktraceFineCells = 2*fineFactor` → 2 cells = same physical reach; OK.

### 4.3 Redistance / volume
- JFA-CPT runs verbatim (B4 brick geometry unchanged — bricks are still 4³
  samples). Clamp the warm ladder `8,4,2,1,+1,+1` to the now-narrower band in
  fine cells.
- Volume correction unchanged mechanically; its per-step correction magnitude
  becomes a first-class logged metric (P3).

### 4.4 Pressure solve
- **No binding or operator changes.** `fineSample` in the structured boundary now
  returns finest-lattice trilinear phi; θ formula, floor, aperture handling
  unchanged. Wet-owner override (`currentPressureOwnerWet`) now compares
  finest-center phi against coarse restriction — near-degenerate; leave in place
  in Stage 1 (it is the cross-check that tells us restriction and band agree).
- Refinement evidence keeps the summary mip, now at octree resolution. Verify
  the refine threshold
  `minimumAbsolutePhi <= (band + max(2,size)) * cellWidth` still refines *ahead*
  of the interface by ≥ CFL displacement — at factor 1 the summary can no longer
  see a zero crossing at sub-cell resolution before refinement (P5); widen the
  evidence band by the measured `maximumDisplacementFineCells` if the mini lane
  shows refinement lag.
- Factor-1 refinement must treat an observed sign crossing or finite
  near-interface sample as positive evidence even when the size-8/16 summary is
  only partially covered. Requiring whole-leaf completeness prevents the
  coarse-to-fine passes from reaching resident B4 surface bricks at all. Missing
  or invalid entries remain false; the relaxation can only over-refine where
  current surface samples actually exist.
- Progressive pressure grading: keep `surfaceRefinementGradingLayers = 1` as
  the exact legacy A/B, then test 2 and 3. The refinement threshold becomes
  `minimumAbsolutePhi <= (band + gradingLayers * max(2,size)) * cellWidth`, so
  each intermediate dyadic level occupies several layers rather than the
  sharpest legal `1 → 2 → 4 → …` transition. This does not widen the factor-1
  tracking band; it isolates pressure-transfer dissipation from interface
  advection dissipation. Accept a larger default only if the mini lane's compact
  dissipation ratio improves without exhausting the pressure-row capacity.

### 4.5 Rendering
- `webgpu-water-global-fine-classify.ts` runs unchanged: marching-tetrahedra cubes
  now at unit *finest* lattice; the `!fineOwnsCube` coarse complement is
  automatically conforming (same resolution at the seam). This is the "faithful
  raster surface from the coarse octree" deliverable.
- Fidelity compensators (rendering-only, no physics feedback):
  - gradient/Hermite vertex placement is already implicit in MT edge interpolation;
  - tricubic (or 3×3×3 Gaussian-filtered) normals at shade time to kill 4× larger
    faceting;
  - optional volume-preserving Taubin smoothing pass on the extracted mesh.

### 4.6 Acceptance gates for Stage 1
- Use the `minimal-power-dam-break-32` and `minimal-power-dam-break-64`
  comparison presets to inspect pressure refinement directly. They preserve the
  mini dam's 0.8 m tank and physical reservoir while changing the lattice from
  16³ at 0.05 m to 32³ at 0.025 m and 64³ at 0.0125 m. Both pin factor-1
  tracking, maximum leaf 16³, band reach 3, and three progressive grading
  layers. The original 16³ preset remains the quality-gate baseline rather than
  silently changing resolution.
- 500-step mini quality gates pass (same set that validated band 1 in
  `POWER_LIQUIDS_TEMPORAL_COHERENCE_HANDOFF.md:57-73`).
- Volume drift ≤ 4× the factor-4 baseline per 100 steps (analytic expectation,
  Section 6 P3); if it exceeds this, BFECC goes default-on before proceeding.
- Free-fall drop oracles pass; hydrostatic lane bit-stable pressure (hydrostatic
  is exactly representable at any resolution — linear phi + ghost-fluid θ — so
  any hydrostatic regression is a bug, not a resolution effect).
- Large-lane wall clock: expect ~36–38 ms (from 48.75). If the repacked-band
  alternative (Section 8) has landed meanwhile, compare against it here.

---

## 5. Stage 2 — Regional fine-band overlay (future)

The hybrid end-state: the factor-1 baseline tracker from Stage 1 stays always-on;
a second band instance at factor 4/8 runs only over policy regions and corrects
the baseline where resident. The codebase already instantiates the band machinery
more than once (two topology objects AB/BA, two transports, two redistancers —
`webgpu-octree.ts:2537-2570, 2710-2735`), so a second instance is parameterization,
not new machinery.

### 5.1 Coverage policy
- Inputs (all already computed or cheap): per-row liquid fraction / physical
  volume (low-volume ⇒ thin features ⇒ fine coverage pays), per-row max |u| or
  kinetic energy from the structured velocity lane, optionally authored volumes.
- Policy evaluates on octree rows, then dilates to brick keys (the band's
  existing seed/dilation path — `insertExternalSeeds` and the ring expansion in
  `webgpu-octree-fine-levelset-topology.ts:1104-1111` — is exactly the injection
  point; `FineSeedLeaf` candidates just become policy-filtered).
- **Hysteresis and dwell are mandatory** (P14): a region must stay covered for
  ≥N steps once admitted, and admission/eviction thresholds must be separated,
  or coverage flicker becomes visible popping in both the surface and θ.

### 5.2 Mixed-authority semantics (the real decoupling work)
Every consumer already has a coarse fallback; what changes is its *meaning*:

- `fineSample` in the structured boundary (`webgpu-octree-structured-boundary.ts`)
  falls back per-tap today. Under partial coverage this must be all-or-nothing per
  θ evaluation (mixing fine and baseline taps inside one trilinear stencil creates
  O(Δx) phi discontinuities *within* a single face coefficient). Rule: if any of
  the 8 taps is uncovered, evaluate the whole sample from the baseline tracker.
- Summary mip: `COARSE_AUTHORITY` (0x80000000) already exists as an ABI flag.
  Split its interpretation: "intentionally uncovered — baseline is authoritative"
  (legal, never refines/coarsens on absence) vs "expected coverage missing"
  (fail-closed error). The policy mask is the arbiter; tests at
  `octree-fine-summary-consumer.test.ts:28` extend rather than relax.
- Restriction (`fine-to-coarse-levelset.ts`): overlay corrects baseline phi only
  where all 8 corners are overlay-resident; `unacceptedRows` becomes the seam
  diagnostic rather than a regression signal.
- Surface extraction: the `fineOwnsCube` partition already renders a band/no-band
  seam conformingly at today's global band edge; the overlay boundary is the same
  seam, just placed by policy. The resolution step across it is P13.

### 5.3 Blending at policy seams (P13)
Where overlay coverage ends along the interface, the surface position estimate
jumps by up to O(Δx − Δx/4) and θ jumps with it — a static kink in the surface
and a pressure-coefficient discontinuity pinned to the seam. Mitigations, in
order of cost: keep seams away from view-critical regions via policy shaping;
feather — in a 2–4 brick collar, publish `phi_blend = α·phi_overlay +
(1−α)·phi_baseline` into the overlay before consumers read it; accept it (the
seam is temporally stable if P14 hysteresis holds).

### 5.4 Rejected alternative: row-native removal (recorded for the record)
Deleting the band structure entirely and moving phi authority onto the compact
coarse directory (`lib/webgpu-octree-power-coarse-levelset.ts`) was analyzed and
rejected for this roadmap — it forecloses the regional future and carries the
worst analytic problems:
- It needs an octree-native redistance (serial FMM is a GPU nonstarter; JFA-CPT
  on the row graph loses the uniform-lattice stride-halving completeness
  guarantee and needs per-level ghost exchange — the repo has hit JFA stride bugs
  even on a *uniform* lattice, `POWER_LIQUIDS_ULTIMATE_M1MAX.md:999-1014`).
- Every octree topology rebuild becomes a lossy resample of the only phi copy
  (P7) — today restriction self-heals carry damage next step.
- The octree must hold uniformly-finest rows across the whole tracking support
  (backtrace + interpolation + redistance ≈ 4–6 cells each side vs
  `interfaceRefinementBandCells = 1` on the large lane), growing the pressure
  lane — already 20.28 ms of 48.75 — to pay for removing a cheaper one (P8).
If a future push wants the band pipelines deleted for simplification, this
section is the checklist of what it actually costs.

---

## 6. Analytic problem register

P1. **θ quantization jitter feeds the pressure solve.** Interface position error
grows from O(Δx/4) to O(Δx); as the zero crossing sweeps a cell, θ (and the 1/θ
face scaling, floored at 1e-2) changes 4× faster per frame. Effect: grid-scale
parasitic velocities at the surface — *dynamics* noise, not just visuals — plus
noisier PCG convergence and weaker warm-start coherence for persistent MGPCG.
Severity: medium. Mitigation: RK2/BFECC advection (less staircase jitter), CPT
redistance (anchors the zero crossing at interpolated positions, preserving it to
interpolation accuracy); measure via the θ-histogram delta metric.

P2. **Dissipation at the authority resolution.** Today's coarse phi is re-derived
from the fine band every step; errors do not accumulate in it. Coarse-only, SL
advection + redistance error compounds in the only copy at 4× the cell size.
First-order single-step SL is not acceptable here. Severity: high without
mitigation. Mitigation: RK2 backtrace mandatory, BFECC optional ladder, CPT
(not PDE-reinit) redistance, deforming-sphere metric in CI.

P3. **Volume conservation degrades and the corrector's artifacts amplify.**
Per-step volume loss scales with Δx; the global corrector must push a ~4× larger
scalar phi shift, which reads as surface "breathing" and rounds concavities.
Worse: when a sub-Δx droplet or air pocket is deleted by resolution, the corrector
re-inflates the *remaining* liquid — spatially misattributed volume. Severity:
medium-high. Mitigation: log pre-correction residual, cap per-step correction,
consider per-connected-component volume budgets later.

P4. **Topology events move to coarse Nyquist and feed back into pressure
connectivity.** Two bodies merge when the gap < Δx (4× earlier): air cushions
collapse early, momentum exchange changes; sheets tear when thinner than ~Δx.
These change the *simulation*, not the picture. Severity: inherent to the ask —
document, don't fight. (The paper itself notes dynamics were always driven by the
octree-sampled level set; what's new is that *tracking* now agrees with the coarse
dynamics instead of out-resolving them.)

P5. **Refinement-oracle circularity.** Refinement evidence now reads phi that only
exists at the resolution topology previously granted. A fast interface can enter a
coarse region before evidence triggers; its shape is then permanently degraded to
whatever the coarsest traversed cell resolved (the fine band used to track through
octree lag; now tracking inherits topology latency). Severity: medium. Mitigation:
dilate the evidence band by measured per-step displacement + safety; tripwire on
"interface zero crossing observed in a non-finest row".

P6. **Refine/coarsen thrash.** Coarsening decisions lose their 4×-resolved
sub-cell data; a coarsened region whose invisible detail re-triggers refinement
next step oscillates. Mitigation: hysteresis (coarsen threshold strictly wider
than refine), plus the existing fail-closed rule (absent data never authorizes
coarsening).

P7. **(Row-native route only — rejected, §5.4) Topology rebuilds become lossy
resamples of the only copy.**
Today any phi damage from an octree rebuild self-heals next step via fine
restriction. Row-native, every refine/coarsen cycle resamples the authority;
repeated cycles diffuse the interface even with a perfect advection scheme.
Severity: high for Stage 2. Mitigation: exact center-sample coarsening, gradient-
augmented prolongation, post-carry redistance repair, hysteresis to cut cycle
count — and the Stage 1/Stage 2 split itself (band keys are octree-independent).

P8. **(Row-native route only — rejected, §5.4) The pressure lane grows to absorb
the band's support roles.** Uniform-finest shell must widen from ~1 to ~4–6 cells → finest pressure
rows in the shell grow ~linearly with band width, pressure+SPGrid is already
20.28 ms of 48.75. This can eat the entire removal saving. Severity: high;
this is the main reason Stage 2 may be a net performance loss. Mitigation:
§5.2 options; hard decision gate on measured row growth.

P9. **Redistance quality is now a pressure-accuracy input with no backstop.**
Ghost-fluid second-order pressure (Gibou 2002) assumes phi is a good signed
distance at cell centers; redistance error lands directly in θ. The fine band's
higher-resolution CPT previously hid octree-side reinit sloppiness. Mitigation:
keep CPT (geometric, zero-crossing-anchored) rather than iterative reinit; the
fail-closed closure oracles carry over.

P10. **Curvature-consuming futures get 16× noisier.** κ = ∇·(∇φ/|∇φ|) error at
coarse resolution makes future surface tension (paper §8 lists it as the natural
extension) substantially harder. Not a today-problem; a door being closed.

P11. **Loss of redundancy and observability.** Today fine-vs-coarse disagreement
is a free cross-check (`unacceptedRows` band-coverage regression signal,
restriction validity gating). Coarse-only, a single corrupted field propagates to
θ, wet classification, refinement, rendering with no independent witness — and the
known failure mode is *silent* (frozen surface, faster frame, PASS). Mitigation:
Stage 0 tripwires are not optional; keep the wet-owner cross-check in Stage 1.

P12. **Contact lines and walls.** Wall-adjacent phi at 4× coarser resolution:
contact-line stepping, and the ceiling/seam sticking class of bugs are
resolution-sensitive. Mitigation: drop oracles in the acceptance gates.

P13. **(Stage 2) Policy seams are surface kinks and pressure-coefficient steps.**
Where regional overlay coverage ends along the interface, the surface position
estimate jumps by up to O(Δx − Δx/4), and θ jumps with it — a static kink plus a
boundary-coefficient discontinuity pinned to the seam. Mitigation: §5.3 feathered
blend collar; keep seams temporally stable (P14).

P14. **(Stage 2) Coverage flicker.** Velocity/energy-driven policy inputs are
noisy; regions oscillating in and out of coverage make the surface pop between
resolutions and modulate θ frame-to-frame. Mitigation: admission/eviction
hysteresis with a minimum dwell; smooth policy inputs over a window; never evict
a region the interface currently crosses within its blend collar.

---

## 7. What we gain

- Large lane ~13 ms/advance of band-lane GPU time plus a slice of air support;
  many small low-occupancy dispatches disappear (the fine lane is a major
  contributor to the 9% frame-mean occupancy).
- ~94–210 MB of arenas on the large lane; capacity planning simplifies.
- dt is unchanged (it was always set by the octree).
- The regional future becomes cheap: once the sim is correct at zero fine
  coverage, fine resolution turns into a dial that spends GPU time only where the
  policy says it pays — the band's cost scales with covered area, not with the
  whole surface.

## 8. Decision gates

1. **Before Stage 1:** confirm the objective. If it were wall-clock only, the X-9
   repack (0.43 ms clean-room band pipeline) reaches nearly the same frame time
   with zero quality loss. With the regional-hybrid end-state as the stated goal,
   decoupling is justified independently of the perf comparison — but X-9-style
   packing remains complementary (it would make both band instances cheaper).
2. **After Stage 1:** quality verdict on real lanes (volume drift, θ jitter,
   drop oracles, 500-step gates) with the fine (4×) band fully off. If baseline
   quality is unacceptable even as a floor, the regional policy must be inverted
   (fine-by-default, coarse-only in calm regions) — same machinery, opposite
   default.
3. **Before Stage 2:** the seam design (§5.3) must have a validated blend or an
   explicit decision to accept kinks; the policy must demonstrate temporal
   stability (P14) on the large lane before overlay correction is allowed to
   touch θ.
