# Vast oceans: the activity-sparsity unlock

Status: analysis + staged plan, 2026-07-19. Companion to
`docs/SPARSE_FLUID_OCEAN_MIGRATION.md` (the brick-sparse dispatch milestone).
This document explains why that milestone plateaued at 1.11× whole-step and what
the fundamental unlock is: deep or quiescent water must be cheap because of how
it is *represented and formulated*, not merely how it is dispatched.

## 1. Why brick-sparse dispatch hit a ceiling

The measured profile (ocean-seiche, 384×96×64, sparse defaults, median of three
Dawn runs, timestamp buckets exhaustive to 0.07 ms):

| Stage | ms/step | % | Domain today |
| --- | ---: | ---: | --- |
| advection | 14.221 | 42.5% | wet bricks (≈ whole box) + 2 dense transport builds |
| pressure | 6.423 | 19.2% | 213,844 rows × 32 Chebyshev passes |
| projection | 5.112 | 15.3% | rows |
| topology rebuild | 3.867 | 11.5% | worklist tiles |
| surface update | 2.294 | 6.8% | sparse surface worklist |
| extrapolation | 0.852 | 2.5% | dense box (4 sweeps) |
| diagnostics + spray | 0.655 | 2.0% | dense reduction + particles |
| **total** | **33.489** | | |

Three structural facts cap the current approach:

1. **Wetness is the wrong sparsity axis for an ocean.** The bulk velocity
   residency deliberately retains all deep liquid
   (`includeLiquidInterior: true`, `lib/webgpu-octree-sparse-bricks.ts:458-470`;
   classify keeps any brick with one negative phi sample,
   `lib/webgpu-fluid-brick-residency.ts:248-279`). 75–83% of bricks are wet, so
   "sparse" removes only air. Amdahl closes the door: the observed advection
   17.3 → 15.5 ms is roughly the air fraction, and it can never be much more.
2. **Velocity has no coarse representation.** The octree grades *pressure* DOFs
   only (interface band `leafNeedsRefinement`, `lib/webgpu-octree.ts:1784-1791`);
   velocity is four dense rgba32float finest-grid ping-pongs plus two padded
   rgba16float transport textures (`lib/webgpu-uniform-eulerian.ts:196-210`),
   advected with MacCormack at finest resolution in every wet cell, including
   4096-cell-deep 16³ leaves whose pressure needs only 8 rows. Both
   `buildTransport` passes still dispatch over the full padded box
   (`lib/webgpu-uniform-eulerian.ts:940,952`).
3. **The formulation forbids skipping calm water.** Gravity is added to every
   wet face every step (`lib/webgpu-eulerian.ts:721`) and pressure is solved in
   absolute form — the solve's job in calm water is to reconstruct hydrostatic
   pressure so projection can cancel the `g·dt` just injected. Equilibrium is
   maintained *dynamically*, at full cost, everywhere, forever. Masking any deep
   brick breaks the cancellation on its faces the next step. This is the real
   content of the earlier "velocity-masking unsafe" finding.

Pressure topology, by contrast, is already area-scaling: deep 16³ leaves
contribute ~8 rows per 4096 cells (~4.6k of the 213.8k rows); the rest is the
interface band plus the 2:1 graded ring. The velocity side has no equivalent.

Conclusion: the unlock is not more indirect dispatches over the same finest-wet
set. It needs three changes of representation, in dependency order:

- **A. A rest-state-zero formulation** (hydrostatic split) so still water has
  identically zero state — making "skip" *exact*, not approximate.
- **B. Activity gating** so bricks with nothing happening do no work.
- **C. Resolution grading of velocity** so deep *moving* water (swell, seiche)
  is represented coarsely, like pressure already is.

A enables B; C covers the motion B cannot skip. Together they change the
asymptotic from O(wet volume) to O(surface area × band thickness).

## 2. The hydrostatic split (Stage 1 — the enabler)

Decompose pressure as `p = p_h + p′` with an analytic per-column reference
`p_h(x,y,z) = ρ·g·(η(x,z) − y)` for `y < η`, else 0, where `η` is the column
surface height (the occupancy pass already computes column heights,
`buildOccupancy`, `lib/webgpu-eulerian.ts:753-762`; keep a filtered `η` texture,
r16float, nx×nz).

Instead of adding `g·dt` in `applyVelocityForces` and letting the solve rebuild
`p_h` each step, apply the *combined* body force `g − ∇p_h/ρ` analytically:

- For `y < η`: the vertical parts cancel exactly and what remains is a
  horizontal force `−g·(∂η/∂x, 0, ∂η/∂z)` — the shallow-water surface-slope
  force, applied through the column. Zero wherever the surface is flat.
- For `y ≥ η` (ballistic droplets/spray above the column surface): `p_h ≡ 0`,
  so plain gravity applies unchanged.

The solve then produces only the deviation `p′`. Free-surface ghost BC is
unchanged (`p_h(η) = 0` by construction, so `p′ = 0` at the surface); the
bottom Neumann condition for `p′` no longer carries gravity because gravity no
longer enters `v*` below `η`.

**The key robustness property:** `p_h` is only a *preconditioner of sparsity*.
Any continuous choice of `η` — stale by a frame, smoothed, clamped through
splash regions — yields correct physics, because `p′` absorbs whatever `p_h`
got wrong, locally. Exactness of `η` affects how sparse the deviation is, never
correctness. So `η` can be maintained lazily and smoothed aggressively.

What this buys immediately:

- Still water is exactly invariant: MacCormack/semi-Lagrangian advection of a
  zero field with zero force is zero. `p′ ≈ 0` at depth, so the warm-started
  adaptive Chebyshev budget (`updateSolveBudget`,
  `lib/webgpu-octree.ts:1173-1188`) floors at 2 passes in calm scenes instead
  of re-deriving hydrostatic balance.
- Skipping quiescent bricks becomes *exact* modulo incoming flux — precisely
  the wake conditions the residency scheduler already computes
  (`classifySwept` brickMaxSpeed + `expandDownstream`,
  `lib/webgpu-fluid-brick-residency.ts:277-303`).
- The residual force field is per-column, derivable from `η` alone, and smooth
  in depth — which is what makes Stage 3's coarse deep velocity accurate.

This is the standard reference-state subtraction of ocean/atmosphere codes and
the implicit assumption behind restricted-tall-cell game water
(Irving et al. 2006; Chentanez & Müller 2011). The repo already contains
tall-cell machinery on the quadtree backend (`lib/webgpu-quadtree-tall-cell.ts`,
`lib/tall-cell-*.ts`) — prior art for Stage 3, not currently on the octree path.

Gate for Stage 1 (behind `FLUID_HYDROSTATIC_SPLIT=1`):

- Calm-tank invariance: 100 steps, max|v| stays ~0 (< 1e-6·h/dt), pressure
  budget at floor, zero far-half disturbance.
- Seiche parity: period and far-half disturbance within tolerance of the
  current formulation; dam-break IoU vs referee ≥ existing bar.
- Watch items: volume-correction divergence source, surface tension terms
  (both live in the same force kernel), inflow columns, enclosed air pockets
  (define `η` per connected surface column; fall back to plain gravity + full
  `p` inside sealed regions).

## 3. Activity-gated velocity work (Stage 2)

With rest state = zero, split the bulk worklist into ACTIVE and STILL bricks:

- Activity predicate per brick: `brickMaxSpeed > ε` (already computed for swept
  support) OR column slope `|∇η| > ε` OR incoming `p′` face gradient OR swept
  support from an active neighbor. Everything else is STILL: skip predictor,
  reverse, correct, transport writes, and extrapolation for those bricks.
- Ping-pong hazard: skipping a brick in an A→B pass leaves stale B. Two
  resolutions, in order of preference: (1) make atlas pages authoritative for
  bulk velocity (already the stated roadmap) so pages don't ping-pong and a
  STILL page is simply untouched; (2) interim: a cheap per-brick copy pass for
  STILL bricks (bandwidth-only, ~4× cheaper than the MacCormack chain, and a
  correct stepping stone).
- Re-run the shelved sparse transport (−0.42%) and occupancy/flux (−2.7%)
  A/Bs against the *active* set. They lost only because their worklist was the
  whole wet box; over an activity band they should flip to wins, killing the
  last dense O(box) builds (`buildTransport` ×2).
- Diagnostics reduction goes behind the same worklist or an opt-in.

Gate: a "calm expanse + one splash" scene (e.g. 768-wide, disturbance confined
to one end) shows whole-step cost tracking the disturbed region, not the tank;
ocean-seiche unchanged (its whole surface is active — that's Stage 3's job).

## 4. Coarse deep velocity (Stage 3 — cheap *moving* depth)

Seiche/swell motion penetrates the full depth, so gating alone cannot help the
deep bulk of a wavy ocean — but that motion is smooth in depth (the residual
slope force is depth-uniform). Represent it coarsely:

- Two-tier velocity: fine bricks only inside the activity/interface band;
  below it, velocity DOFs on the same 16³ leaves the pressure solve already
  uses (or restricted tall cells per column, reusing the quadtree tall-cell
  kernels). Advect the coarse tier over the coarse worklist — 4096× fewer
  cells than finest — with 2:1-style interpolation at the tier seam, the same
  grading discipline the pressure octree already enforces.
- This is the step that finally decouples cost from depth: doubling the water
  depth adds only coarse cells (a few percent), not another half of the
  advection bill.

Gate: seiche parity at coarse-deep vs finest-deep; a doubled-depth scene
(384×192×64) whose step time grows < 10%.

## 5. Unbounded extent (Stage 4 — existing roadmap, now viable)

Atlas-authoritative fine band + owner pages + scene-scale two-level addressing
(`lib/scene-scale-fluid-bricks.ts`) as already planned — but after Stages 1–3
the resident fine set is genuinely thin (∝ surface area), so memory and
worklists scale with area too, and "vast" stops meaning "vast × depth".

## 6. Independent quick wins (Stage 0, do anytime)

- **Projection 5.1 ms is suspicious**: one Chebyshev pass over the same row
  domain costs ~0.2 ms, and projection is ~2–3 row passes
  (`reconstructGradients` + `projectLeaves`, `lib/webgpu-octree.ts:1069-1074`).
  Profile per-pass (`FLUID_PRESSURE_PHASE_TIMINGS=1`) — there may be 3–4 ms of
  scratch copies or mis-bucketed work here.
- **Topology 3.9 ms**: verify change-driven rebuild
  (`FLUID_OCTREE_CHANGE_DRIVEN`) is engaged on ocean-seiche; deep tiles are
  clean in a seiche and the calm-scene result was 5.18 → 0.20 ms. If it is
  engaged, find what keeps deep tiles dirty (wetness-flip sensitivity near the
  band was a known cause).
- **Assemble split**: matrix coefficients depend on topology + solids only;
  RHS on velocity. Assemble the matrix only for dirty tiles (change-driven
  signal already exists), rebuild RHS every step. Saves a per-row full
  assemble each step in calm regions.

## 7. Expected arithmetic

At 384×96×64 (33.5 ms today): advection 14.2 → ~3–4 ms (fine band ~0.6 M cells
+ coarse deep), transport builds fold into the active set, pressure 6.4 →
~4–5 ms (deviation warm start; assemble split), projection pending the Stage-0
profile, topology → ~1–2 ms with change-driven engaged at depth. Whole step
plausibly ~15 ms at this size — but the real point is the scaling law: after
Stage 3, widening or deepening calm water adds near-zero cost, and a live
region pays in proportion to its activity. That is the "deep areas or areas
with little activity should be cheap" property, made exact by the formulation
rather than approximated by dispatch culling.

## 8. Verification harness notes

- Use `docs/GPU_STAGE_DIAGNOSTICS_CAPTURE_PLAN.md` to visualize launched versus
  active domains, resource deltas, worklists, and per-stage summaries. Its
  instrumented frames are explanatory evidence only; all speedups must still be
  measured capture-off with the interleaved A/B harness below.
- Full stage table prints from the standard smoke run (no extra env): the
  result JSON carries all 15 `GPUPhysicsTimings` buckets
  (`lib/webgpu-eulerian.ts:34-54`); the published benchmark cherry-picked 4.
- Only trust in-process interleaved A/B (`FLUID_AB_ENV` in
  `tools/benchmark-octree-leaf-sizes.ts`); process-to-process swings ±30%.
- Bit-determinism: smoke runs are bit-deterministic, so Stage-2 STILL-brick
  skipping can be gated on *bit-identical* fields for a calm tank, the same
  technique that caught the change-driven topology bugs.
