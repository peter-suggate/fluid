# Advance slice fidelity contract

The Advance lab CPU solver is a two-dimensional port of the production Sparse
Geometric (CM12) numerical stages. Its visible and transported authority is an
X/Y slice; pressure retains the source-derived 3-D cells and rows needed to
reproduce the production recurrence before projecting the centre-Z X/Y faces.
Its purpose is to make changes cheaply in two dimensions and then port
successful changes back to the WebGPU solver.
Parity therefore includes production quirks and known failure behaviour. An
analytically better result is a regression when the production solver still
produces the old result.

## What parity means

The CPU and GPU paths can be compared step for step when the three-dimensional
state is invariant in Z: identical Z planes, zero Z velocity, no Z-dependent
solid aperture, and no Z source or boundary flux. Z flux and divergence cancel
in that case, and a two-dimensional control volume is the unit-depth
restriction of the three-dimensional one. Exact pressure-iteration parity
still requires the extruded 3D incidence graph: its Z rows and separated
mixed-rung terms remain in the positive Jacobi diagonal and therefore change
the f32 preconditioned recurrence even when their operator image cancels.

For any other catalog scene, the center-Z slice is an exact initial-state
sample and a useful dimensionally reduced experiment. It is not a prediction
of the center plane of the later 3D run. The 3D update contains Z face flux and
`dw/dz`; those terms cannot be recovered from one plane.

The numerical comparison uses fields and receipts, never canvas pixels. A
matched probe records, in order:

1. initial open capacity, extensive liquid volume, staggered face velocity,
   acceleration, cell size, and time step;
2. the eight synchronous velocity-extension generations;
3. forced face velocity;
4. pressure membership, row membership and ghost-fluid fraction;
5. pressure RHS, pressure, true residual and projected divergence;
6. the transport microstep count and every shared low, high and limited face
   flux;
7. committed volume, total volume, center of mass, second moments and kinetic
   energy after each microstep and outer step.

Comparison tolerances must be derived from f32 evaluation and reduction order.
They must not be widened until a failing field has been localized. Signed and
bit-level comparisons are preferred for classification, membership, stage
counts and branch decisions.

## Production behaviour carried by the CPU port

The production order relevant to the two-dimensional port is velocity
extension, face preparation, body force, pressure topology, pressure RHS,
pressure solve, velocity projection and collocation, then conservative volume
transport. Production has no separate semi-Lagrangian face-velocity advection
stage in this path.

Velocity extension seeds cells whose density is strictly greater than `0.5`.
It advances through eight synchronous generations. A dry cell reads only the
opposite side of each physical face and weights a neighbour by subface area
divided by center distance. Pressure membership uses density greater than or
equal to `0.5`, and can retain a previous member when every accepted-row
neighbour was also a member and the cell has no one-term air port. The strict
and inclusive comparisons are intentionally different.

Gravity is applied to accepted face rows. Pressure uses the production
incidence orientation, dual weights, a minimum ghost-fluid fraction of `0.05`,
and the face-jump form of `G^T W G`. The solve is warm-started f32 pipelined CG
with a diagonal Jacobi preconditioner and a true-residual check every eight
iterations. Gauss-Seidel, a cold solve, or a float64 recurrence is a different
algorithm even when its final divergence is smaller.

Transport holds extensive liquid volume. It freezes the projected face field
for an outer step and chooses

```
microsteps = max(1, ceil(2 * maximumCellCfl))
```

with a production failure above 128. Donor outgoing rate and per-face prism
travel determine the cell CFL; incoming through-flow does not double it. Each
microstep reconstructs the interface, publishes one shared oriented subface
flux, constructs a bounded upwind low state, applies the synchronized receiver
low-flux limiter, limits the anti-flux with both endpoint budgets, and gathers
that same shared flux into both cells. Authoritative volume is not clamped.

The production data path is f32. Operation grouping, face order and reduction
order can change a boundary decision by one ulp, so parity tests preserve them
where the GPU makes them deterministic. GPU atomics or workgroup reductions
whose order is intentionally unspecified require bounded numerical comparison
plus exact comparison of the resulting branch and failure receipts. Runtime
WGSL division is also backend-bounded: the [WGSL accuracy table](https://www.w3.org/TR/WGSL/#floating-point-accuracy)
allows 2.5 ULP for ordinary f32 division. A raw-word mismatch confined to such
an operation is recorded with its exact operands, backend result and downstream
branch/material receipt; it is not hidden by a general tolerance.

## Scene extraction

The selector is sourced from the production scene catalog. The selected source
plane is `floor(nz / 2)`. Scalar cell samples use cell centers; X and Y
velocities use their staggered face centers. If the numeric slice uses a
canvas-style Y-down axis, extraction reverses Y indices and negates every
Y-directed vector component, including Y face velocity and acceleration.

Initial velocity in scene documents is metres per second. The resident solver
stores velocity in finest cells per second, so extraction applies the same
cell-size conversion. Static center-slice extraction does not by itself certify
a continuous inflow, moving rigid body, moving capacity, live edit, or sparse
frontier transaction. The CPU path contains reduced forms of these stages, but
the selector labels them as reduced until the corresponding differential
receipt passes.

## Sparse activity and topology lifecycle

The numerical authority is a sparse accepted brick directory, compact cell and
row image, physical subface stream and incidence CSR. A candidate generation is
planned separately. Its fields are conservatively transferred and the whole
generation is either published or refused; a dense display lattice is readback
only.

The activity policy retains the resident shader's f32 fixed-point density
moments, reason bits, score bytes, hot/quiet/proof history, topology cadence,
velocity rung thresholds, coarse-first curvature and incoming-motion floors,
surface-proof gate, directional support masks and exact-zero retirement rule.
The Z-removed support mask is 3x3 rather than 3x3x3. Missing in-domain support
gets a stable sparse page identity at B8. Hard cell-size bounds propagate an
outward grading cap before refine-only physical 2:1 closure. Promotions and
lifecycle changes are urgent; ordinary demotions use the production rotating
budget. Leaf/cell capacity failure keeps the accepted generation authoritative.

The persistent runtime authority uses the production WDR1, TEI2 and HTP1
constructors and word ABIs directly. The Z-reduced atlas has physical depth one;
HTP cells have `centerZ=0.5` and `widthZ=1`, while TEI keeps the production
stable `leaf * 64 + packet` address and publishes only the `z=0` lane plane.
Accepted and candidate HTP/WDR images remain separate, TEI uses opposite slots,
and commit or cancel changes all banks together. Post-presentation retirement
tombstones a dynamic WDR coordinate, invalidates its leaf generation, updates
the free list and bounds, and scrubs both TEI banks before that stable leaf ID
can be reused. The accepted HTP remains intact until the next topology commit.

## Live liquid injection

A dropped ball is an intervention, not a stage. `injectAdvanceSliceLiquid` is
the CPU counterpart of `injectLiquidBall` and the resident `injectLiquid`
kernel, and it reproduces the production two-phase shape: one topology
generation planned with the drop in `injectionDemandedBrickKeys`, then the dose
applied once onto the committed graph. A refused transaction refuses the whole
drop, matching the kernel's opening `sparseCM12TopologyLifecycleAccepted()`
gate; a partially applied ball would be silently missing whichever part needed
a page.

The dose is the resident arithmetic term for term. Coverage is the smoothed
one-cell indicator `clamp(0.5 - signed * radius / width, 0, 1)`, clipped by the
cell's open fraction, then `max` against the density already there. It is not
an exact clipped area even though the slice has that geometry available,
because production's is not: an analytically better coverage here is a parity
regression. Gamma goes to 1 on any covered cell. Faces are untouched, because
`injectLiquidFaces` returns early for anything that is not a hose, so a dropped
ball arrives at rest. Brick demand is the ball's bounding box, not the disk,
which is `injectionReachesBrick` exactly; the extra bricks take no liquid
because their coverage is zero.

Three reductions are deliberate. There is no deferral, because `advanceSlice`
is synchronous and no pointer event can land mid-advance, so there is no frame
in flight to queue behind. There is no page allocation, because the slice's
brick roster is the whole atlas plane and activation is the only thing a drop
ever needs. There is no open world: a 2-D drop is inside the lattice or it is
nothing.

The dimensional reduction differs by scene, and the lab states which one it is
giving. On a `symmetry` z boundary the disk is the exact unit-depth reduction
of the cylinder the app drops into a 2-D case, so the two runs stay comparable
step for step. On a bounded z boundary it is the centre-plane sample of a
dropped ball at the instant it lands and no later, because a ball's slice is
not z-invariant and the fall diverges immediately.

Injected volume joins `seededVolume`. Drift is conservation error against what
the run was given, and a reader's own water is something the run was given, not
a conservation failure.

## Validation coverage

The CPU advance calls the same 15 named stages in the resident ABI order. All
15 now have a concrete authority and an execution path; the last column keeps
the distinction between implemented source parity and live GPU evidence.

| Resident stage | CPU authority | Validation status |
| --- | --- | --- |
| `transport-velocity-extension` | `extendSliceVelocity` over physical mixed-port pieces and the production eight-generation f32 axis tree | CPU mixed-port fixture and the strict mixed/hydro whole-step Metal receipts pass |
| `face-preparation` | accepted-row face preparation plus the production staggered RK2/support sampler | CPU sampler fixtures and mapped Metal face area/sweep receipts pass |
| `body-forces` | accepted-face gravity, moving-wall and inflow forcing | Strict hydrostatic Metal pressure/projection receipt passes |
| `pressure-topology` | source-derived 3-D embedded graph plus retained PCM1/PCF1 images | Fixed, mixed and rerung mapping/cache CPU tests pass; live rerung bank words remain open |
| `pressure-rhs` | source-graph density, source, capacity-motion and hydrostatic classification | Mixed translation and nonzero hydrostatic Metal RHS/diagonal words pass |
| `pressure-solve` | retained PEI1 execution order and f32 Chronopoulos-Gear recurrence, recovery and eight-iteration true-residual cadence | 17-record Metal PCG journal and whole-slice nonzero solve pass |
| `velocity-projection` | source-graph pressure projection followed by centre-Z X/Y extraction, inflow enforcement and collocation | Mixed and hydrostatic projected face words pass on Metal |
| `conservative-transport` | PLIC reconstruction, shared low/high flux, receiver limiter, anti-flux limiter, moving FISTA/closing and source commit | Two mixed B8:B4 Metal frames and one hydrostatic frame pass; one bounded division scratch difference is recorded below |
| `tracer-advection` | fixed-lane tracer authority with production wet seeding and dry retirement | `slice-tracer-authority.test.ts` passes; live whole-frame tracer words remain open |
| `scalar-publication` | FSM1 stable TEI packet bits, source-bank facts and bulk certificate | `slice-scalar-authority.test.ts` passes; live FSM1 words remain open |
| `activity-measurement` | f32 density/activity moments, history, reason and support masks | Source-fingerprinted policy fixtures pass; live activity words remain open |
| `resolution-planning` | cadence, hysteresis, region caps, frontier growth, retirement, rotating budget and 2:1 closure | Policy and integrated automatic-rerung CPU fixtures pass; live automatic-rerung receipt remains open |
| `candidate-transfer` | accepted/candidate WDR1/TEI2/HTP1 images and conservative field transfer with commit/cancel | CPU generation, rollback, new-air and stable-slot tests pass; live candidate image remains open |
| `brick-retirement` | post-topology retirement authority followed by post-presentation WDR tombstone/free-list release | CPU change-set and allocate-retire-reuse tests pass; live release words remain open |
| `presentation-publication` | persistent page publication, metrics/rigid receipts and shared RDF output | CPU publication tests, the actual-Metal planar/point-neighbour RDF receipt and the shipping curved RDF-to-FPP-to-mesh gate pass; the unchanged 2-D Figure 3 area bound has the failure recorded below |

| Area | Evidence | Current scope |
| --- | --- | --- |
| Center-plane seed | `production-scene-slice.test.ts` across catalog metadata and focused canonical samples | Exact initial state and coordinate/sign conversion |
| Sparse graph | `slice-topology.test.ts` compares complete X/Y center-plane row streams with `buildSparseAtlasCompositeGrid`, including B8:B4 seams | Exact static graph/order for the compared accepted generations |
| Generation transfer | `slice-topology.test.ts` covers PLIC split, capacity fallback, coarsening weights, face remap, new air and f32 terminal tolerance | CPU source-derived fixtures; live GPU transfer differential remains open |
| Pressure recurrence | `advance-slice-pressure-parity-dawn.test.ts` compares 17 raw resident journal records bit for bit | Exact CPU PCG recurrence against the actual 3D resident operator callback; this does not certify 2D orchestration |
| Activity/lifecycle | `slice-resolution-policy.test.ts` pins resident/policy source fingerprints and exercises activity reasons, frontier activation/allocation, exact-zero retirement, grading, admission and rollback | Source-derived CPU port; live resident activity-word differential remains open |
| WDR/TEI/HTP banks | `slice-runtime-authority.test.ts` builds literal production WDR1/TEI2/HTP1 images, checks unit-depth cells, stable packet/lane addressing, candidate slot commit/cancel, capacity refusal and dynamic allocate-retire-reuse | Exact production host constructors and CPU execution of the production WDR free state over the Z-reduced topology; live candidate-word differential remains open |
| Embedded pressure graph | `slice-pressure-embedding.test.ts` proves unique source-cell/source-row mappings for fixed, mixed and automatically rerung centre slices; `slice-pressure-embedding-authority.test.ts` exercises retained PCM1/PCF1/PEI1 generations, cache reuse and topology invalidation | Full source-derived 3-D pressure graph, persistent warm state and literal pressure bank images are authoritative on symmetry extrusions; live rerung bank-word comparison remains open |
| Reconstruction/FCT | `advance-slice-transport-parity-dawn.test.ts` runs two actual Metal frames on a mixed B8:B4 Z extrusion | Initial density and every mapped area, sweep, low/high/limited flux and committed volume are bit exact. Metal's limiter division scratch is one ULP below the CPU result for two cells from identical recorded operands, within WGSL's division bound; downstream material words are exact |
| Nonzero pressure step | `advance-slice-transport-parity-dawn.test.ts` runs the Z-invariant tiny hydrostatic tank through actual Metal | Pressure membership, RHS, diagonal, nonzero pressure, projected X/Y velocities, transport fluxes and final volumes are bit exact; the retained PCG executes two iterations |
| Dynamic source | `advance-slice-source-ledger-dawn.test.ts` exercises the complete staged resident source commit | Actual Metal and CPU agree across all 11 commit dispatches and 20 persistent/event ledger lanes |
| Rigid integration | `advance-slice-rigid-parity-dawn.test.ts` reads the packed 32-f32 body state and 12-i32 exchange record | Actual Metal golden covers a free sphere and off-centre cuboid exchange |
| Live liquid injection | `slice-liquid-injection.test.ts` pins the resident `injectionCoverageAt` / `injectionReachesBrick` / `injectLiquid` terms this port copies, and exercises open-fraction clipping, the non-accumulating `max`, bounding-box brick demand, frontier activation to the finest rung, refusal and conserved-total bookkeeping | Source-derived CPU port of the two-phase interaction; live resident injection-word differential remains open |
| Whole step | The same transport Dawn test covers two mixed-rung translating frames and one nonzero hydrostatic frame | Exact for those fixed accepted generations; automatic rerung is still pending, so there is no general end-to-end parity claim |
| RDF cache | `rdf-3d-plane-metal-point-neighbours.json` records six signed/oblique actual-Metal cases plus a manufactured diagonal-only contributor after the publication-only dispatch refactor | GPU and independent CPU agree for the full uniform point-neighbour least-squares stencil, including a value that cannot be obtained from the six face neighbours; maximum RDF-value and gradient errors are `1.49e-6` and `1.84e-7` finest-cell units, density banks are unchanged, and the source stayed unchanged throughout the final receipt (repository-source fingerprint `c5324af5...`, geometric WGSL `716fa1d7...`, resident WGSL `62acd51f...`) |
| Shipping curved RDF | `rdf-shipping-curved-metal.json` records production VOF spheres and tori whose partial cells cross an authored B8:B4 join, then republishes the same resident RDF → PLIC → RDF without advancing physics | Exact Equation 10 weighting plus the shared topology-vertex least-squares cache produces closed, manifold meshes and a word-identical RDF republish. Sphere volume is `2.8990%` from accepted VOF with RMS/maximum surface error `0.004827/0.010661 m`; torus volume is `0.2582%` from accepted VOF with RMS/maximum `0.005936/0.023459 m`. RDF and PLIC differ in 7,188/5,616 packed sample words, while accepted density, time and topology generation remain unchanged |
| Adaptive mesh consumer | `sparse-cm12-adaptive-mesh-dawn.test.ts` directly supplies analytic signed-distance samples across X/Y/Z 2:1 joins | Sphere, torus and ellipsoid meshes are closed, manifold and finite in the measured cases. This bypasses the resident RDF cache and is only a downstream meshing control. The unchanged torus volume bound still fails at coarse ratio 1 (`14.287%`) and ratio 2 (`10.856%`) while full resolution is `4.053%`; that is an existing resolution limit, not an RDF result |

The literal Equation 10 self contribution and in-domain sparse-air support
change the 2-D CM12 Figure 3 shared-contour area error to
`9.1566 / 2391.080017 = 0.383%`. The contour has no unresolved or ambiguous
fine cells, but it misses the existing `0.2%` area cap. The formula is not
special-cased and the cap was not raised; the focused CPU test remains a
recorded failure.
The final 15-file advance-slice CPU sweep therefore passes 73 of 74 tests; its
sole failure is this named presentation bound. The current Water Box fixture
also proves the source-authored brick 8 promotion from B4 to B8, candidate
transfer, generation advance and runtime-bank identity.

The paired mini32 Metal performance receipt in
`artifacts/sparse-cm12-rdf-presentation-performance-final/paired-summary.json`
uses 8 warm-up and 24 hardware-timestamp samples per mode. Accepted density,
gamma, velocity, pressure RHS/diagonal/solution and terminal work hashes match.
Presentation median changes from `5.8982` ms for PLIC to `7.3400` ms for RDF
(`+1.4418` ms, `+24.45%`); whole-advance median changes by `+0.9830` ms
(`+0.592%`). This pair pins the final cached RDF implementation but predates
the separate swept-air-support edit in the canonical snapshot, so it is not a
full-frame baseline for that later source.

The pressure-journal receipt uses a full 3D `G^T W G` callback reconstructed
from the resident graph and certifies the solver recurrence and reduction
schedule. The whole-slice hydrostatic receipt additionally certifies that the
slice orchestration constructs the same member set, RHS, diagonal, nonzero
pressure, projection and transport inputs for that Z-invariant case.

The first integrated mixed-rung comparison exposed why the full graph is
required. Pressure membership and RHS matched, but coalescing a Z extrusion
into 2D terms before squaring incidence coefficients changed the X/Y Jacobi
weight. Adding only a guessed Z diagonal also admitted closed depth-boundary
rows whose live ghost-fluid theta was zero. The CPU now runs the source-derived
extruded cells, rows, masks, persistent pressure images and execution order,
then extracts the centre-plane result. Mixed translation and nonzero
hydrostatic receipts both pass. `water-box-dam-break` is deliberately excluded
from evolution parity because its authored dam has finite Z extent and its
source rungs differ between depth layers; its centre plane remains an exact
initial-state and dimensional-reduction case.

## Canonical production gate receipt

The final mandatory `npm run test:dawn:sparse-cm12` gate ran on an immutable
snapshot because a separate task was editing the shared checkout. The snapshot
pins host `0bf6ac5c...`, resident WGSL `11952062...`, geometric WGSL
`716fa1d7...`, and stage-cost baseline `87f2463d...`. It completed on Metal in
362183.3 ms, inside the 480000 ms suite budget, with 4 of 17 lanes passing. The
machine-readable receipt and full output are
`artifacts/advance-slice/sparse-cm12-dawn-final.json` and
`artifacts/advance-slice/sparse-cm12-dawn-final.log`.

`simulation-failure-halt`, `mixed-ratio-topology`,
`topology-generation-storage`, and `mini64-min8-surface` passed. Seven lanes
timed out without changing their limits: `symmetric-expansion`,
`topology-page-budget`, `clipped-topology-transfer`, `mini32-correctness`,
`mini32-performance`, `live-rigid-body-coupling`, and
`live-liquid-injection`. Six lanes exited with concrete failures:
`hydrostatic-adaptivity`, `mini64-performance`, `tall-cells-hills-far-wall`,
and `outside-tank-symmetric-collapse` hit geometric-volume coverage faults;
`long-dam-far-wall` published 66 of 106 generation-zero pages; and
`min8-region-surface` changed a published RDF sample by `0.874643` cells
against a `0.02` cap. The last failure is localized to a rank-deficient
boundary vertex least-squares fallback while accepted density remains planar.
The captured snapshot includes an unfinished swept-air-support policy from a
separate active task, so this receipt does not certify that policy or label its
failures as RDF or proxy regressions without a differential.

An earlier moving-checkout run recorded 7 of 17 lanes passing in 467632.3 ms in
`artifacts/advance-slice/sparse-cm12-dawn-gate-2026-09-12.json`; it is retained
as history rather than a matched baseline. The focused source-ledger, rigid,
pressure-recurrence, transport and RDF receipts above remain separate evidence
and do not turn either full gate into a pass.

## Open parity gates

The following checks must pass before the lab can claim production parity:

1. Compare live resident activity words, requested/candidate rungs, lifecycle
   membership, grading result and admission/fault bits against the CPU policy
   for fixed, mixed and world-growth inputs.
2. Compare the exact-zero closing-capacity FISTA/allocator receipt against the
   resident (`K:1→0`, one open receiver), including the zero terminal amount.
3. Compare two-frame PCM/PCF/PEI generation headers, dirty worklists and stable
   pressure-cell stream against the live no-solid resident.
4. Compare the staged WDR/TEI/HTP candidate words and slot flip with live page
   growth, rerung and rollback transactions.
5. Run an automatic-rerung Z-invariant step with identical scene, time and
   inputs; compare volume/flux, pressure/divergence, shape/energy, timestep
   counts, topology decisions, bank generations and failure receipts.
6. Add a shipping-path ellipsoid receipt if parity is later claimed for that
   curved field. Sphere and torus now pass the actual RDF-to-FPP-to-mesh gate;
   ellipsoid is covered only by the analytic-field downstream mesh control.

## Deliberate dimensional omissions

The CPU slice cannot validate general 3D PLIC polyhedron clipping, Z subfaces
or any Z-dependent flux/derivative. The production reconstruction retains a
certified invariant-axis branch so an adaptive Z extrusion stays an extrusion;
genuine 3D stencils continue through the ordinary 3D candidate path. GPU
dispatches are reproduced by their observable stage and reduction order where
that order is defined; hardware scheduling itself is not a portable CPU
property. The display lattice is evidence about presentation unless its cells
and subfaces are also the numerical control-volume graph.

These omissions limit where parity can be claimed; they do not permit changing
the two-dimensional form of a production formula. Experimental fixes belong
behind an explicit mode until the production WebGPU implementation adopts the
same change.
