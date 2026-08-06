# Rigid coupling repair + minimal Dawn coupling oracle

Successor to `docs/rigid-coupling-fine-band-handoff.md`. The R-series (R1–R5)
landed and was re-audited against two regressions; this doc carries (a) the
repair items with their root causes, and (b) the definition of a minimal Dawn
lane that asserts *plausible* rigid/fluid coupling physics so this subsystem is
never again exercised only by eyeball.

## Evidence base (what was proven, 2026-08-05)

All by deterministic A/B bisect on the `corner-brick-drop` seam-shortfall
oracle (identical values run-to-run), scratch worktree at HEAD `46f7ff5` +
working-tree diff, nine runs:

- **X1 root cause, proven.** `rigidPressureReaction` is constructed
  unconditionally (`webgpu-octree.ts:2660`), so R5's
  `encodeRigidBoundaryRefresh` fires every substep in EVERY Losasso scene
  (zero bodies included). It re-runs `conditionLosassoFaces` over the accepted
  faces, resetting `openFraction = 1 − solid` — with the zero-solid sentinel
  that is `1.0` on every face, including the closed-boundary wall/lid faces
  the ghost pass had zeroed (`webgpu-octree-losasso-coarse-phi.wgsl.ts:181`).
  The refresh rebuilds operator coefficients without re-running
  `publishLosassoGhostDistances`, so wall contact is destroyed for the rest of
  the advance and walls act as pressure anchors. Effect: HEAD seam stick
  0.51 shortfall for 0.08 s → with refresh 0.65–0.76 held through 0.20 s
  (the visible "wall at the dam front"). Disabling only the refresh restores
  the HEAD baseline exactly.
- **Exonerated by the same bisect:** the Ando-Batty sizing chain (production
  and consumption), the backend conditioning rewrite, both identity-row
  fallbacks, the wet-owner exclusion (compiles out unless
  `hasTerrain || rigidBodyCount > 0`, `webgpu-octree.ts:874`).
- **X2 root cause, code-confirmed.** The R2 carve writes positive phi inside
  bodies (`webgpu-octree-fine-levelset-rigid-carve.ts:49`). The ghost pass's
  missing-neighbor branch has no solid check
  (`webgpu-octree-losasso-coarse-phi.wgsl.ts:214-215`): faces into the
  R1-excluded body interior see `liquid<0 && sampled>0` and install a
  Dirichlet p=0 ghost with theta at the body wall. Shell apertures ≈ 0.5 make
  the anchor strong. The body's wetted surface is pinned to atmosphere: no bow
  wave, and the reaction impulse (∝ p at the wall,
  `webgpu-octree-losasso-rigid-pressure-reaction.ts`) collapses toward zero —
  bodies "cut a path".
- **X3, code-confirmed.** `measureOnlyLosasso` gained
  `|| scene.rigidBodies.length > 0` (`webgpu-octree.ts:4930`): volume
  correction is fully OFF in rigid scenes, so carved/splashed volume never
  returns (persistent craters).
- Pre-existing at HEAD, out of scope here but recorded: a mild 0.51/0.08 s
  seam stick — the HEAD "wall release" fix is incomplete.

## Repair items

**X1 — boundary refresh must not clobber contact state.**
Two parts, both required:
1. Gate: skip the refresh when `scene.rigidBodies.length === 0` (do not rely
   on `rigidPressureReaction` being absent — it never is). This alone restores
   every rigid-free scene.
2. Correctness in rigid scenes: after the refresh's conditioning pass, re-run
   `publishLosassoGhostDistances` before
   `losassoConditionedOperator.encodeAfterGhostDistances` +
   `encodeHierarchyCoefficientRefresh`, so wall-contact zero apertures and
   free-surface theta are rebuilt against the refreshed apertures. (Alternative
   if the ghost re-run is too expensive per substep: make the refresh kernel
   skip `FACE_CLOSED_BOUNDARY` faces entirely; but the ghost re-run also
   serves the freshness goal R5 wanted.)
Also: the refresh dispatches `conditionLosassoFaces` bound to the ACCEPTED
authority control; its `fail(ERROR_HEADER)` path can poison a live accepted
graph. Route refresh-time validation failures to a diagnostic word, never to
`authority[4]` of the accepted set.
*Acceptance:* `corner-brick-drop` seam-shortfall returns to ≤ HEAD baseline
(violations only at t ≤ 0.08 s, magnitude ≤ 0.55); new O-lane (below) green.

**X2 — solid-aware ghost classification.**
Bind `solidCells` into the coarse-phi/ghost pass. In the missing-neighbor
branch, before installing a Dirichlet ghost, probe the neighbor cell's solid
fraction; at ≥ 0.999999 (same constant as R1's wet exclusion) take the
no-ghost path (`flags=4`): the face stays Neumann and the four-sample aperture
carries the blocking. Dirichlet remains only for genuine liquid–air. This is
the Ando-Batty split: `[A]` face fractions own solids, ghost fluid owns the
free surface. The carve itself is correct and stays — it only stops leaking
into the pressure classification.
*Acceptance:* hydrostatic oracle (O1) measures buoyancy ≈ ρVg instead of ≈ 0.

**X3 — volume correction target, not disablement.**
Revert the `rigidBodies.length > 0` clause. Instead subtract the submerged
solid volume from the corrector's target: Σ over bodies of
V_body · immersed-fraction (the integrator already computes `immersed`
per body — publish it, or recompute analytically from body pose vs the
measured surface height). Correction cadence unchanged.
*Acceptance:* O-lane volume drift ≤ 0.5% through each scenario.

**X4 — tighten the conditioned-operator identity fallback.**
`webgpu-octree-losasso-conditioned-operator.ts:46-55` currently converts any
row with zero faces OR all-closed faces into an identity row. Restrict the
identity to rows that are provably solid-enclosed (at least one face, all with
`openFraction ≤ 1e-7`); a zero-face row is malformed topology and must keep
raising `solverAuthority` error 16. Mirror the same distinction in
`sortLosassoIncidences` (`webgpu-octree-losasso-backend.wgsl.ts:608-620`,
whose `allSolidBlocked` init already requires `end > begin` — keep it that
way).

**X5 — coupling tripwires (cheap, permanent).**
1. Sealed-plug counter: in the wet classifier, count wet owners whose cells
   are all ≥ 0.999999 solid into a diagnostic word (post-R1 this must be 0).
2. Surface-source counter (F0 from the previous handoff): checkpoints already
   expose `raster.geometrySource`; the lane asserts on it (below).

## O — the minimal Dawn coupling oracle

One new scene family in `lib/scenes.ts` (pattern: `createFreeFallDropScene`),
three variants sharing one container, wired as smoke-catalog suites with
diagnostic hooks (pattern: the `free-fall-contact` hooks,
`lib/scene-webgpu-smoke-catalog.ts:1334`), run under Dawn via
`FLUID_SCENE=<id> FLUID_METHOD=octree node --import tsx tools/run-webgpu-smoke.ts`.

**Shared container** — deliberately tiny and analytic:
- Grid 16×16×16, `finestCellSize_m = 0.05` (0.8 m cube), `fineFactor` default 4
  (12.5 mm fine cells). `top: "closed"`, `fluidWallMode: "free-slip"`, no
  terrain, no inflow, `surfaceTension_N_m = 0`.
- Still water fill to depth 0.4 m (fillFraction 0.5), `fixedDt_s = 0.004`
  (deterministic, same as the drop oracles).
- Tank cross-section A = 0.64 m²; water ρ = 1000 kg/m³; g = 9.81.

### O1 `rigid-hydrostatic` — static submerged sphere (the core case)

Sphere r = 0.10 m (V = 4.189×10⁻³ m³), `motion: "static"`, center at
(0, 0.20, 0) — fully submerged, top 0.10 m (2 coarse cells) below the surface,
bottom 0.10 m above the floor. Nothing moves, everything is analytic. Duration
0.5 s, checkpoints every 0.05 s.

Oracles (warmup: skip t < 0.1 s):
- **Buoyancy from pressure.** Per checkpoint, read the rigid exchange:
  `force = impulse/dt`. Assert `force_y ∈ ρ g V · [0.85, 1.15]` (41.1 N ± 15%;
  first-order operator + ⅛-fraction rasterization earn the margin),
  `|force_xz| ≤ 0.05·ρgV`, `|torque| ≤ 0.05·ρgV·r`. **This is the assertion
  that fails ≈ 0 under the X2 bug and would have caught it.**
- **Still water stays still.** `maxSpeed_m_s ≤ 0.05` at every checkpoint.
- **Volume.** Measured liquid volume within 0.5% of authored (fill minus
  V — the carve removes the sphere's interior once; it must remove it exactly
  once and never again).
- **Liveness.** Authority generation strictly increases between checkpoints;
  candidate error flags 0; sealed-plug counter 0.
- **Rendering.** `raster.geometrySource == "global-fine"` for ≥ 90% of
  checkpoints (catches silent fallback).

### O2 `rigid-float` — dynamic buoyant sphere

Sphere r = 0.15 m, ρ_body = 500 kg/m³, `motion: "dynamic"`, released at rest
with its center exactly at the waterline (y = 0.40). Archimedes: settles with
immersed volume fraction = ρ_body/ρ_water = 0.50, i.e. center at the surface.
Duration 2.0 s, checkpoints every 0.1 s.

Oracles:
- **Settle depth.** Over t ∈ [1.5, 2.0]: mean immersed fraction (from body
  center vs measured surface height, sphere-cap formula) ∈ [0.40, 0.60];
  center-height oscillation amplitude decaying (last-half amplitude < half of
  first-half amplitude) — asserts drag/added-mass act at all without pinning
  their values.
- **No plunge-through:** body minimum y > floor + r at all times (fails if
  buoyancy is dead, as in the X2 bug or the un-wired exchange words pre-R3).
- Volume ≤ 0.5% drift, liveness, plug counter, geometry source as in O1.

### O3 `rigid-sink` — the "far and wide" tripwire

Sphere r = 0.10 m, ρ_body = 2400 kg/m³, dynamic, released from rest with its
bottom 0.10 m above the surface (center y = 0.60). Duration 1.0 s.

Oracles:
- **Bounded splash.** `maxSpeed_m_s ≤ 2·√(2g·0.2) ≈ 4.0` at every checkpoint
  (impact speed ~1.98 m/s; a divergence pump blows far past this), and no wet
  sample above y = 0.55 after t = 0.5 s (bounded crown, settled).
- **Displacement is conserved, not doubled.** Final volume within 0.5% of
  authored; measured surface rise once the body rests on the floor
  = V/A = 6.5 mm ± one fine cell (12.5 mm) — loose, but kills the ~2×
  injection mode of the sealed-plug era.
- **It sinks.** Body reaches the floor (bottom within one coarse cell of it)
  by t = 0.8 s and stays; body speed at t = 1.0 s ≤ 0.2 m/s.
- Liveness across the entry event (this is where A1-style publication
  poisoning appeared), plug counter 0, geometry source ≥ 90%.

### Harness work

- Scene ids + `createRigidCouplingOracleScene` in `lib/scenes.ts`; suites +
  hooks in `lib/scene-webgpu-smoke-catalog.ts`; package script
  `test:webgpu:rigid-oracles` running all three.
- New readback (small): per-checkpoint rigid snapshot — body pose/velocity
  (32 floats/body from the bodies buffer) and the exchange accumulators before
  clear. Everything else (volume measurement, maxSpeed, authority diags,
  raster source) already exists in the smoke path.
- All three lanes are enforced (not diagnostic-only) once X1–X3 land; until
  then they run red and document the gap.

## Sequencing

1. **X1** (gate + ghost re-run + fail-path isolation) — unblocks everything;
   re-run `corner-brick-drop`, expect HEAD baseline.
2. **O1** harness + scene (red against current tree ≈ proof it bites), then
   **X2** → O1 green.
3. **X3** → O1/O3 volume oracles green; **O2/O3** enabled and enforced.
4. **X4, X5** alongside; they are small and independent.

Baseline numbers to hold: corner-brick-drop seam-shortfall at HEAD = 0.5081 /
0.5505 / 0.5506 / 0.3669 at t = 0.02–0.08 s, clean after. Better is welcome
(the HEAD residual is itself a known incomplete fix); worse is a regression.
