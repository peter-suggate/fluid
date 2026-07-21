# Hydrostatic split + two-tier velocity: detailed plan and bottleneck analysis

Status: design, 2026-07-19. Successor detail to
`docs/VAST_OCEAN_ACTIVITY_SPARSITY_PLAN.md` §2–§4. Scope here: (A) the
hydrostatic pressure split, (B) leaf-face velocity DOFs below the fine band
("two-tier velocity"), and an honest answer to *will this work or just move
the bottleneck?*

## 0. Verdict up front

**At today's 384×96×64 this buys ~1.4–1.6×, not 5×. Its real product is the
scaling law: after it lands, step cost stops growing with water depth.** The
profile is advection 14.2 / pressure 6.4 / projection 5.1 / topology 3.9 /
surface 2.3 / other 1.5 (33.5 ms). This plan attacks only the velocity-domain
work (advection + transport builds + extrapolation ≈ 15 ms); Amdahl caps the
fixed-size gain regardless of how well it executes. What changes qualitatively:

- Today, doubling depth roughly doubles advection and grows topology; a
  384×384×64 tank would step at ~75–90 ms. Post-plan it steps at ~24–26 ms —
  fine-tier work is pinned to the surface band (∝ area × band thickness), and
  the deep tier adds thousands of DOFs, not millions of cells.
- The bottleneck then moves, by design, to the **area-scaled** stages:
  pressure rows (~213k, band-dominated already), projection, topology, surface
  — roughly 25/22/17/10% of a ~22 ms step. Those have their own known levers
  (Stage-0 projection audit, assemble split, change-driven topology at depth,
  band-thickness tuning) and, for vast calm *width*, activity gating.
- What this plan does **not** deliver: memory reduction (dense velocity
  ping-pongs stay allocated O(box) until atlas-authoritative lands) and cheap
  calm *width* (that is activity gating, a separate slice).

New bottlenecks it could introduce, with mitigations designed in: seam-exchange
passes (§B5, budget ≤ 2 small indirect passes per substep — the solver has a
history of being dispatch-overhead-bound), software velocity sampling in the
deep tier (§B4, bounded by tiny DOF counts and a semi-Lagrangian-only rule),
and two representations that must not drift (§C, bit-deterministic parity
harness). Failure conditions that would falsify the design are listed in §D —
each has a measurable gate before we commit to the next milestone.

Risk asymmetry: Part A is low-risk, standard practice (reference-state
subtraction), lands alone, and is a prerequisite; Part B is the larger lift and
is built mirror-first so every step is A/B-checkable against the dense path.

---

## Part A — Hydrostatic split

### A1. Formulation

Split `p = p_h + p′` with the analytic per-column reference

```
p_h(x, y, z) = ρ · g · (η(x,z) − y)   for y < η(x,z),   else 0
```

`η` = column surface height. Instead of `v.y += g·dt` on every wet face
(`applyVelocityForces`, `lib/webgpu-eulerian.ts:721`) followed by a pressure
solve that reconstructs `p_h` so projection can cancel it, apply the combined
body force `g − ∇p_h/ρ` directly:

- `y < η` (columnar water): vertical parts cancel exactly; remaining force is
  horizontal, `f = −g · (∂η/∂x, 0, ∂η/∂z)`, applied on x/z faces through the
  whole column. Zero where the surface is flat.
- `y ≥ η` (droplets/spray above the column surface): `p_h ≡ 0`, plain gravity,
  unchanged.

The solver mechanically still solves the same system — it just converges on
`p′`. Free-surface ghost BC unchanged: `p_h(η) = 0` by construction so the
ghost condition `p′ = 0` is the existing one. Bottom/wall Neumann is consistent
because gravity no longer enters `v*` below `η`.

**Correctness invariant (the load-bearing property):** any continuous `p_h` is
valid — `p′` absorbs reference error *locally*. `η` staleness, smoothing, or
clamping affect only how sparse/small `p′` is, never the converged physics.
This is why `η` can be lazy and why no special-case fallback is needed for
correctness anywhere.

### A2. η source and conditioning

- `η` already exists: `buildOccupancy` writes the per-column highest occupied
  cell to the height texture (`lib/webgpu-eulerian.ts:753-762`), and that
  texture (`heightIn`) is **already bound in the advection kernels** — the
  `aboveOccupancy` skip reads it (`:785-791`). Stage A needs no new bindings on
  the predictor path. Optional sub-cell refinement (occupancy fraction of the
  top cell) can come later; it only sharpens `p′` sparsity.
- Conditioning, not correctness: clamp face slope `|Δη|` (e.g. ≤ 2 cells per
  cell) and use the face-straddling difference `(η(i+1) − η(i))/Δx` on the
  i-th x-face. Multi-valued columns (splash sheets, sealed air pockets) need
  **no special path** — the clamped single-valued `η` is simply a worse
  reference there and `p′` carries the difference, which is exactly what the
  solver already resolves today (today `p_h` is effectively 0 everywhere and
  `p′` is everything).
- Ordering is already correct: occupancy is rebuilt at the top of the advection
  stage from post-projection state (`lib/webgpu-uniform-eulerian.ts:924-947`),
  so the force uses a fresh `η` in the same substep.

### A3. Code touch list

1. `applyVelocityForces` (`lib/webgpu-eulerian.ts:717-729`): replace the
   unconditional `v.y += cellGravity.w*dt` with the split force (η lookup +
   below/above-η branch + clamped slope force on x/z faces). Gravity is
   y-only in this engine (`cellGravity.w` scalar packed from
   `gravity_m_s2.y`), so no generalized-direction work.
2. `correctAdvection` inherits the change (it calls the same helper,
   `lib/webgpu-eulerian.ts:838`). Surface tension, inflow, volume-correction
   divergence: untouched.
3. Option plumbing: `hydrostaticSplit` through `methods/octree.ts` →
   `webgpu-uniform-eulerian` (the field-by-field copy trap), env
   `FLUID_HYDROSTATIC_SPLIT`, default off until gates pass.
4. Warm start: cold-started runs need nothing. If ever toggled live, snap the
   pressure budget to cap for a few steps (the >1%-row-change snap mechanism
   already exists, `lib/webgpu-octree.ts:1178-1187`) rather than converting the
   stored field.

### A4. Gates (all in-process A/B, `FLUID_AB_ENV` interleaved)

- **Calm tank**: 100 steps, max|v| < 1e-6·h/dt, far-half disturbance ~0, and
  the adaptive pressure budget decays to its floor (2 passes) — the direct
  observable that the solve stopped re-deriving hydrostatics.
- **Seiche parity**: far-half disturbance within 5%, oscillation period within
  2% of the current formulation; 400-step gate still passes.
- **Dam break**: harness IoU vs referee ≥ existing bar; no new validation
  errors; bit-determinism of the *off* path preserved.
- **Perf**: whole-step within ±1% (A alone is an enabler; pressure may improve
  slightly in calm scenes via the budget floor).

Effort: ~2–4 days including harness time. Watch items: wave-amplitude parity
(the split changes the surface cell's force discretization at the η kink —
this is where the ghost-fluid machinery already lives) and inflow columns.

### A5. Implementation finding (2026-07-19)

The default-off octree A/B is implemented, but Stage A has **not** cleared its
gates. Three formulations were tested on Dawn/Metal:

- A local connected-column `η(x,z)` force made a settled tank exactly still,
  but the 2 s ocean far-half disturbance rose from the absolute-pressure
  baseline's 0.408 cells to 1.457 cells.
- Restoring analytic `p_h` through octree leaf pressure reduced neither issue:
  vertically nonconforming coarse/fine neighbours have pressure centres at
  different heights, so a leaf-constant hydrostatic field creates horizontal
  gradients. Settled liquid speed remained about 1.13 m/s and ocean
  disturbance was 1.381 cells.
- The retained fixed-rest-surface variant keeps `p_h` out of the leaf basis.
  It cancels only vertical rest gravity and supplies the perturbation Dirichlet
  value `p'=-p_h` at the actual level-set crossing. After 100 settled-tank
  steps it gives 0.00143 m/s maximum liquid speed, 0.00465 1/s RMS divergence,
  zero volume drift, and no validation errors. The 2 s ocean remains stable
  and connected, but its 1.345-cell far-half disturbance still fails the 5%
  parity gate; the pressure budget also remains at 32 passes rather than its
  floor.

This falsifies A1's load-bearing claim for the current discrete octree: an
arbitrary continuous reference is not merely a change of variables when its
gradient is applied on fine faces but `p'` lives in the nonconforming
leaf-constant/affine space. The flag therefore remains off by default and
fails closed for dam-break initial conditions, inflows, and pressure-coupled
bodies.
Before Stage B, either enrich the octree pressure space with an explicit
hydrostatic mode and demonstrate operator parity, or revise the seiche
reference against an independent analytic/referee solution rather than the
numerically damped absolute-pressure baseline.

---

## Part B — Two-tier velocity (leaf-face DOFs below the band)

### B1. Representation and tier rule

- **Fine tier**: finest-resolution velocity, exactly today's storage (dense
  ping-pongs + atlas mirror + hardware trilinear), but **dispatched only over
  a band-limited worklist**: a new bulk-residency variant with
  `includeLiquidInterior: false`, phi band + swept support + N-cell dilation
  below the surface (all machinery exists in
  `lib/webgpu-fluid-brick-residency.ts` — this is configuration, not new code).
  Band thickness starts at ~16–24 fine cells below η; tunable, and the main
  quality/cost dial of the whole design.
- **Deep tier**: every liquid leaf not covered by the fine band carries MAC
  face DOFs — 3 f32 (positive x/y/z faces) per row, in row-indexed buffers
  beside the pressure arenas. At the 406,272-row capacity this is ~4.9 MB —
  negligible. The compaction stream already enumerates these rows and already
  separates cooperative (size ≥ 8) rows (`lib/webgpu-octree.ts:2460-2467`).
- Tier membership is derivable per row from the fine-band worklist (a brick
  covers the row's origin or not) — one flag bit in `LeafHeader` set at emit.

### B2. Deep-tier dynamics

- **Advection**: semi-Lagrangian only, one kernel over the deep-row worklist.
  Backtrace from each leaf-face center through the composite sampler (§B4).
  **No MacCormack in the deep tier** — the field there is smooth by
  construction (post-split the deep forcing is the depth-uniform slope force;
  wave orbital motion decays and smooths with depth), and MacCormack's
  neighbor-extrema limiter is what makes coarse non-uniform stencils ugly.
  This is a deliberate accuracy/complexity trade with a parity gate (§B7).
- **Forces**: the same split slope force, evaluated at the leaf-face center
  from the η texture. (Without Part A this tier is untenable — absolute-form
  gravity/pressure cancellation at coarse resolution would ring. This is the
  dependency.)
- **Projection**: `reconstructGradients` already produces per-row affine
  gradients; deep rows apply `−dt·∇p′/ρ` directly to their 3 face DOFs and
  **skip writing finest velocity textures entirely**. `projectLeaves` /
  `projectSmallLeaves` (`lib/webgpu-octree.ts:1073-1074`) gain the tier branch.
- **CFL/max-speed**: the reduction must take `max` over deep DOFs too — one
  small extra reduce over rows.

### B3. Divergence assembly gets *cheaper*

`assembleSystem` currently sums `size²` finest sub-face velocities per coarse
leaf face (`lib/webgpu-octree.ts:2489-2494`). Post-change:

- deep–deep faces: read the single shared leaf-face DOF (area-weighted at 2:1
  seams: the four fine faces vs one coarse face use the existing
  merged-neighbor bookkeeping — the matrix side already handles exactly this
  grading).
- deep–fine seam faces: the fine sub-faces are authoritative; the coarse face
  value is their area mean (restriction, §B5). This keeps the RHS exactly
  conservative across the seam — the flux the fine side exports is the flux
  the coarse side imports.
- fine–fine faces: unchanged.

A 16³ leaf face goes from 256 velocity reads to 1. Deep assemble cost drops.

### B4. Composite velocity sampler

One WGSL helper, `compositeVelocity(p)`:

- `p` inside the fine band → today's path (hardware-filtered transport/atlas
  sample).
- `p` below the band → owner-map lookup (`ownerAt`) + per-axis linear
  interpolation between the owning leaf's opposing face DOFs, blended with
  face-neighbor leaves' DOFs across leaf boundaries (the same
  neighbor-resolution rules the pressure gradient cases use). Start simple
  (per-leaf linear + neighbor average at faces); the deep field's smoothness is
  what makes low-order interpolation acceptable, and the parity gate decides.

Consumers: deep-tier advection backtraces; fine-tier backtraces that dip below
the band (long vertical characteristics); the seam skirt fill (§B5). The
surface stage's RK2 phi advection samples near the surface — always in-band,
unchanged. Overlay/diagnostic materialization reads deep DOFs only when
overlays are on (existing lazy path).

### B5. Seam exchange — the pass-count budget

Two small indirect passes per substep, and that is the budget:

1. **Skirt fill (prolongation)**: populate a 2-cell finest "skirt" below the
   band bottom from deep DOFs, into the transport texture, over a
   skirt-brick worklist (bottom layer of the band). This keeps *hardware*
   filtering valid for all shallow in-band backtraces so the fine tier's hot
   path never branches into software sampling.
2. **Restriction**: area-average fine band-bottom faces into the adjacent deep
   leaf-face DOFs (feeds §B3's conservative seam rule).

Both are band-area-sized (thousands of workgroups). The dense transport builds
shrink to the band worklist (`dispatchTransport` already has the sparse route,
`lib/webgpu-uniform-eulerian.ts:597-603` — it currently loses only because its
worklist is the whole wet box); occupancy keeps its 2D column form (already
area-scaled); extrapolation runs its existing sparse variant over the band
(`FLUID_BRICK_SPARSE_EXTRAPOLATION` becomes a win once the domain is the band,
not the box — re-run the shelved −3.4% A/B here).

### B6. What stays dense (explicitly deferred)

Dense velocity/transport allocations remain (unwritten below the band after
the flip). Memory reduction is the atlas-authoritative/owner-page track, which
this plan composes with but does not depend on. Do not bundle them: the parity
story here relies on the dense mirror existing.

### B7. Gates

- **Mirror-mode equivalence** (before any flip): deep DOFs written by
  restriction from dense each step; assemble-from-DOFs vs assemble-from-dense
  RHS difference below tolerance on seiche + dam break.
- **Physics**: seiche far-half disturbance within 5%, period within 2%, single
  liquid component, 400-step gate; dam-break IoU ≥ bar (dam break mostly runs
  fine-tier — the band covers the churn; the gate verifies the band/refinement
  criterion is catching it).
- **Refinement safety net**: extend `leafNeedsRefinement` with a deep strain
  term (its `detailActivity` velocity sampling currently only engages within
  the surface band, `lib/webgpu-octree.ts:1793-1794`) so genuine fine-scale
  momentum at depth forces local refinement back to the fine tier. Gate: a
  "deep jet" scene (inflow at depth) must locally refine and match the dense
  path's plume within IoU bar.
- **Perf at 384×96×64**: advection bucket ≤ 6 ms; whole step ≤ 26 ms; no new
  bucket > 1 ms unaccounted.
- **The scaling gate (the point of it all)**: 384×192×64 and 384×384×64 calm
  and seiche variants — whole-step growth ≤ 15% per depth doubling
  (today: roughly linear in depth).

---

## C. Parity and safety harness

- Everything behind two flags: `FLUID_HYDROSTATIC_SPLIT`,
  `FLUID_TWO_TIER_VELOCITY` (B requires A at plan level; assert at options
  validation). Both plumb through the `methods/octree.ts` field-by-field copy.
- Smoke runs are bit-deterministic: the off-path must remain bit-identical
  throughout (this caught all three change-driven-topology bugs).
- Only in-process interleaved A/B for timings (`FLUID_AB_ENV`); ±30%
  process-to-process noise makes anything else worthless.
- Dawn-node lifecycle tests for every new pipeline/binding (the 9-storage-
  buffer Chrome limit incident: request `requiredLimits`, verify in lifecycle
  tests, never repro GPU crashes in a live browser).

## D. Failure conditions (what would falsify this)

1. **Seiche parity fails at any reasonable band thickness** → deep flow is not
   as smooth as argued; fallback: thicken band adaptively by wave amplitude
   (cost degrades gracefully toward today's, never below it).
2. **Seam artifacts** (reflection/damping of long waves at the tier boundary)
   → visible as period drift/amplitude loss in the seiche gate. Mitigations in
   order: deepen band; higher-order restriction; move seam off the wave's
   high-shear region. This is the highest physics risk — it is why the seiche
   gate runs at every milestone, not at the end.
3. **Dispatch overhead eats the win** (history: quadtree CG was
   dispatch-overhead-bound) → visible immediately in the M3 advection bucket;
   budget is 2 seam passes; fuse skirt fill into the transport build if needed.
4. **Deep-tier sampling too crude for rigid-body scenes at depth** → the
   refinement safety net (B7) is the answer; if a body at depth forces the
   whole region fine, cost degrades to today's, not worse.

If (1) and (2) both hold at no acceptable band thickness, the design is wrong
and the fallback position is Part A alone + activity gating (still exact for
calm regions) — Part A is independently valuable either way.

## E. Milestones

| # | Deliverable | Gate | Est. |
| --- | --- | --- | --- |
| M0 | Stage-0 audits: projection 5.1 ms per-pass profile (`FLUID_PRESSURE_PHASE_TIMINGS`); confirm change-driven topology engages on seiche | numbers in hand; informs E-arithmetic | 0.5–1 d |
| M1 | Part A behind `FLUID_HYDROSTATIC_SPLIT` | §A4 all pass | 2–4 d |
| M2 | Deep-row DOF buffers + tier flag + mirror-mode restriction + assemble-from-DOFs parity | §B7 mirror equivalence | 3–4 d |
| M3 | The flip: band-limited fine worklist, deep advection/projection on DOFs, 2 seam passes, composite sampler | §B7 physics + perf gates | 5–8 d |
| M4 | Band cutovers: transport builds, extrapolation, occupancy sparse re-A/B | each slice wins or stays off | 2–3 d |
| M5 | Depth-scaling benchmark + doc + defaults flip | scaling gate; long-horizon gates rerun | 1–2 d |

Rough total: 2–3 weeks of sessions. After M5 the expected profile at
384×96×64 is ~21–24 ms dominated by pressure/projection/topology — at which
point the next campaign is the area-scaled stages and activity gating for
width, per `docs/VAST_OCEAN_ACTIVITY_SPARSITY_PLAN.md` §3/§6.
