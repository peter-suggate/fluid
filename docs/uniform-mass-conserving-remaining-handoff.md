# Uniform (CM12) performance — remaining work handoff

Successor to `docs/uniform-mass-conserving-performance-handoff.md` (the full
findings ledger F1–F11 and the original WP definitions live there; this document
is what a fresh session needs to continue). Written 2026-08-12.

## Where this stands

- **Landed, bit-identical, verified:** WP1 (coarsest-solve convergence break via
  `workgroupUniformLoad`) and WP3a (`mgProjectMinimum` deleted; projection folded
  into the smoother's pass-through write as `max(old, p_min)`). Median wall
  **59.2 → 12.3 ms/step (4.8×)** on the benchmark arm. The 4,096-iteration
  coarse spin was 80% of the GPU frame.
- **Implemented in the successor worktree:** WP3b now bakes each level's three
  positive-face coefficients and liquid flag once after phi continuation.
  `mgSmoothColour`, `mgResidual`, and `mgMeasureFineResidual` consume that
  immutable stencil; the coarsest solve keeps its existing shared-memory path.
  Dawn shader validation and one uniform symmetric-expansion advance pass. The
  project regression lanes are the authored mini-octree dam break at 32³ and
  octree symmetric expansion; both two-step Dawn gates pass (2026-08-12).
- **Commit state (as of c1f4c51):** Peter's commits caught the session mid-flight.
  `0f461c0` is the exact pre-WP1 state of both multigrid files (the clean revert
  target — never stash/checkout in this worktree; `git show 0f461c0:<path>` and
  overwrite is the sanctioned method for these two files only). HEAD `c1f4c51`
  contains WP1 and the WP3a comments; the only uncommitted remainder is the WP3a
  fold itself: 6+/11− across `lib/webgpu-uniform-pressure-multigrid.ts` (drop
  `mgProjectMinimum` from `ENTRY_POINTS`/`ENTRY_BINDINGS`, 2-pass `sweep`) and
  `.wgsl.ts` (pass-through writes `max(old, p_min)`, kernel deleted). **Commit
  this remainder first** — HEAD alone is an inconsistent midpoint (comments
  describe a fold that isn't applied at HEAD; it still runs correctly, but the
  228 projection passes are still paid).
- **Remaining frame is launch-depth bound.** No single pass dominates anymore;
  the cost is ~700 dependent sub-ms passes.

## Measured frame anatomy (post-WP1, instrumented sample)

One sampled advance under `FLUID_GPU_PASS_TIMESTAMPS=1` (1,024 passes, summed
21.2 ms — instrumented samples run heavier than the 12.x ms median wall; trust
the *shares*, not the absolutes). Capture predates WP3a, so `mgProjectMinimum`
still appears; it is now gone.

| Block | ms | invocations | note |
| --- | --- | --- | --- |
| `mgSmoothColour` | 9.67 | 455 × 21 µs | the wall. F2's fixed 3F+4V schedule |
| FIM extension (both invocations) | ~5.0 | ~110 | 2.15 ms is ONE front iteration (all-or-nothing indirect dispatch, F8); ~90 passes are 12–13 µs of `prepare`/update links |
| mg misc (residual, restrict, prolongate, clears, downsamples) | ~2.5 | ~180 | 6–18 µs each |
| `mgProjectMinimum` | 2.16 | 228 | deleted by WP3a |
| `mgSolveCoarsest` | 0.93 | 15 × 62 µs | was 51.05 ms before WP1 |
| transport + sharpening + gamma diffusion | ~0.3 | ~30 | F5/F6 confirmed irrelevant on this body-free scene |

So the honest post-WP3a split is roughly: **~55% smoother passes, ~25% FIM
extension, ~15% multigrid plumbing, ~5% everything else.** Every further win on
this benchmark is either fewer dependent passes (WP3c), a cheaper front (WP7),
or fused stages (WP5).

## The decision menu — RE-BLESS, needs Peter's call before starting

These change numerics; benchmark gates must be consciously re-accepted. Do not
bundle them with each other or with bit-identical work. Ranked by expected
effect on the current frame.

### WP3c — Jacobi/Chebyshev smoother + residual-steered cycle count

Replace PRBGS with damped or Chebyshev-accelerated Jacobi (projection folded in,
as WP3a already does) and let the already-measured-and-read-back fine residual
(`webgpu-uniform-reference.ts:1190` — currently display-only) steer how many
cycles run instead of the fixed 3 Full + 4 V. Attacks the 455-pass smoother
block from both ends: fewer passes per sweep is not the point (Jacobi is 1 pass
vs 2, but needs more sweeps); the point is **cycle count becomes adaptive** —
most advances on a settling scene need far fewer than 7 cycle-equivalents.
Coloring is D4-symmetric either way. Expected: the ~10 ms smoother block scales
with actual convergence need; plausibly 2–3× on the pressure stage. Acceptance:
the §Verification gates plus the convergence telemetry (`mgConvergence` slots,
`uniformCM11aCoarseIterations`) showing residuals still meet
`UNIFORM_CM11A_COARSE_RESIDUAL_TOLERANCE`-class targets at the fine level.

### WP7 — Extension restructure (pick 7a or 7b first)

The ~5 ms front is two full extension invocations per advance (MacCormack needs
the predicted field extended too), each with a 24-slot fixed FIM encode whose
live iterations sweep the whole grid to advance a ≤2-cell band.

- **7a (non-conforming, REVERTED):** dropping MacCormack made velocity advection one
  semi-Lagrangian pass and deleted the second extension
  (~61 passes), reverse, and correct. More velocity dissipation — the lane is
  already labeled "very dissipative", so this is squarely Peter's call. A/B via
  `lateToMiddleKineticEnvelopeRatio` / `normalizedLateMechanicalEnergySlopePerSecond`.

  This path was based on an incomplete reading of the primary paper. CM12
  delegates velocity transport to CM11b, whose Sec. 3.5 explicitly requires
  modified bounded MacCormack. The predicted-field extrapolation bind groups
  and schedule have therefore been restored. The former experiment's result
  was: on the 32x16x32 symmetric-expansion Dawn arm, 20 exact
  4 ms steps pass all diagnostics with finite state, 0.572% represented-volume
  drift, and mechanical-energy retention 1.00785. The velocity-advection stage
  measures 65.5 us; final kinetic-energy proxy is 0.04464 versus 0.04561 before
  the cutover, confirming the expected additional dissipation. The authored
  32-cubed mini dam two-step gate and octree symmetric-expansion three-step D4
  gate also pass; the latter retains exact topology/volume symmetry and a
  maximum velocity error of 1.54e-8.

  A separate `uniform-16ms` still-water lane now guards the hydrostatic control
  after a visibly unsettled post-motion surface prompted investigation. The
  planar 16-cubed tank remains at rest for 20 exact 16 ms steps: zero raw and
  represented-volume drift, maximum speed 2.57e-6 m/s, and column-height
  standard deviation 8.04e-7 cells. This distinguishes a history-dependent
  interface/reconstruction problem from an inherent inability of CM11a to hold
  hydrostatic balance.
- **7b (keep MacCormack):** build the extension *operator* once from
  post-advection density, apply to both current and predicted fields. Slight
  result change vs today's two-geometry extension.
- **Either way:** replace the 24-slot wavefront with a direct radius-2 gather
  (the band is ≤2 cells, distances and upwind values are computable in one
  bounded-stencil pass), or at minimum make the indirect dispatch a compacted
  worklist instead of all-or-nothing (`webgpu-uniform-velocity-extrapolation.wgsl.ts:325`).

### WP5 — Gamma diffusion as one snapshot flux-gather pass — IMPLEMENTED

Antisymmetric pairwise flux computed from a pre-pass snapshot (both sides of
each face compute the identical flux; one adds, one subtracts) is conservative
and order-free by construction — the 12 mirrored ordered passes + average pass
+ 4 staging copies collapse to one pass per repetition. **Note the measured
reality:** on this benchmark the whole block is ~0.3 ms, so this is a
depth/cleanliness win (deletes ~16 dependent passes and the mirrored-order
machinery), not a wall win. D4 symmetry should get *stronger*; re-bless
dissipation via the energy-retention indicators.

Implemented with a degree-normalized six-face gather. A 20-step Dawn comparison
changed total density across the diffusion stage by 2.78e-7 cells (1.36e-10
relative), retained finite state, reported 0.585% represented-volume drift and
1.0093 mechanical-energy retention, and measured the replacement as one 65.5 us
pass. The documented post-pressure D4 growth remains visible and is not silently
reclassified as a gamma-stage regression.

## Bit-identical backlog — no decision needed, ranked by when they matter

- **WP3b — per-level coefficient bake. IMPLEMENTED.** One setup pass per level stores
  (a_x⁺, a_y⁺, a_z⁺, flags); smoother/residual read one texel per neighbour
  instead of re-deriving via `mgFaceV`/`mgTheta` (3–5 loads per neighbour).
  Same formula, same order. Cuts per-pass cost of the dominant block; matters
  more at 128³ (L=7, 2,634 passes) than at benchmark size. Watch for compiler
  float-reassociation breaking byte-identity (known legitimate failure mode —
  fall back to gates + small absolute bound, and say so).
- **WP6 — boundary-velocity compaction + bind-group parity. IMPLEMENTED.** Four dense
  n³×16 B buffers hold O(n²) payload; `carryBoundaryVelocity` rewrites all of
  them in every advect/reverse/correct/project. Face-sized arrays + both-parity
  bind groups. Implementation scalar-packs the three negative face planes while retaining
  A/B/C/D parity, updates audit readback to the same layout, and reduces the
  32×16×32 symmetric arm's reported allocation by 1,015,824 bytes. The evaluated
  negative-boundary reflection audit remains exact (`maximumAbsoluteError = 0`).
  Correction to the original ledger: the nearby texture copies carry density,
  render, or rigid-coupling state rather than boundary-velocity staging, so WP6
  does not claim their removal.
- **WP8 — field packing.** (ρ, γ, ρ′, flags) into one rgba; (p, rhs, φ, p_min)
  per level; stop writing velocity's dead `w` lane. Do after WP3c/WP5/WP7 —
  those change who reads what.
- **WP2 — solid topology bake.** Bakes the analytic-primitive chains (≤12-body
  loops per texel) into textures once per advance; `mgBuildFinestTopology` is
  the in-tree template. **Zero effect on the body-free benchmark** — do it when
  body/terrain scenes matter, or when cost-model honesty at production scenes
  is the goal.
- **WP4 — hardware trilinear (RE-BLESS, listed here for completeness).**
  Transport is ~1 ms on this benchmark; the paper-shaped motivation is gone.
  Only worth it bundled with a resolution jump. Requires `float32-filterable`
  (never requested today — sampling would fail validation).

## Verification protocol (unchanged rules + two hard-won traps)

- **Identity A/Bs MUST run with `FLUID_AWAIT_EVERY_STEPS=1`.** Under the
  default every-30-step fence a mid-run checkpoint deterministically includes
  state from in-flight later advances; any change that shifts frame timing
  produces phantom physics diffs (WP3a's first A/B showed 82 spurious
  checkpoint differences this way). Tell: same code, different mid-run
  observations under different checkpoint cadences. Physics is deterministic;
  the capture lies.
- **Attribution: `FLUID_GPU_PASS_TIMESTAMPS=1`, never `FLUID_GPU_FINE_TIMESTAMPS`**
  on this lane — the seam-level recorder charged ~35 ms to sub-ms extension
  phases and its sum exceeded measured wall.
- Bit-identical WPs: `FLUID_UNIFORM_SYMMETRY_STAGE_AUDIT=1`, byte-compare
  stage-audit textures against control at fixed steps.
- Wall claims: fresh process per arm, ≥3 runs, medians; never across tripwire
  modes; single-run 2–3% deltas measure nothing.
- RE-BLESS WPs: full `npm run benchmark:symmetric-expansion:uniform-vs-losasso`;
  gates: D4 volume ≤1e-3, D4 velocity ≤1e-4, ≤1% conservative drift, dominant
  component, boundary residue. Energy indicators are reported A/B, not gated —
  regressions go to Peter, never silently accepted.
- GPU exclusive lock `/tmp/fluid-webgpu-exclusive.lock` (a directory): wait for
  release; check `owner.json` pid liveness before ever considering it stale.

## Known reds — do not chase as regressions

- **The canonical 250-step uniform arm fails its D4 gates pre-existing:** 48
  failures (velocity D4 from step 30, volume D4 from step 60, dominant
  component ~100–210). Proven unrelated to WP1/WP3a twice over: a full revert
  to `0f461c0` reproduced the failure list **line-for-line identical** at
  74.9 ms/step (2026-08-11, scratchpad `revert-failures.txt` vs
  `postchange-failures.txt`), and
  `artifacts/canonical-symmetric-expansion-250-projected-face-20260811.json`
  fails the same 48 gates at pre-WP1 speed (61.6 ms/step). Whoever fixes the
  D4 asymmetry does it as its own investigation, on its own branch of
  evidence — perf work should not inherit the blame or the fix.
- The losasso arm of the same benchmark crashes on its cutover oracle
  ("adaptive publication exceeds D4 tolerance", "2 enclosed surface holes") —
  independent, pre-existing, not touched by any of this work.
- The CPU suite is red at HEAD (~124 failures, unrelated; see memory
  `cpu-suite-is-red-at-head`).

## Reproduction quick reference

- Benchmark arm alone (the uniform env is what
  `tools/benchmark-symmetric-expansion-comparison.ts` spreads):
  `FLUID_LANE=comparison-uniform`, `FLUID_UNIFORM_SYMMETRY_STAGE_AUDIT=1`,
  `FLUID_UNIFORM_DENSITY_POSTPROCESSING=0`, 250 steps, dt=0.004, checkpoint
  every 10 — plus `FLUID_AWAIT_EVERY_STEPS=1` for any identity comparison.
- Full gate: `npm run benchmark:symmetric-expansion:uniform-vs-losasso`.
- Plan arithmetic (post-WP3a): sweep = 2 passes; benchmark (L=4) total 711
  dependent pressure passes, 16 coarse solves; recompute for other sizes from
  the appendix of the predecessor doc.
