# Sparse Geometric: large-timestep conservative remap plan

Date: 2026-09-12.

Status: restricted-map Dawn feasibility implementation and a standalone shared
repeated-remap UI session are completed; production remap and production
velocity bridge remain unimplemented. The active simulation is unchanged. This
document does not authorize concurrent GPU work.

Review update: 2026-09-12. Large translation is feasible in principle; general
nonlinear capacity-compatible remapping and production velocity coupling remain
unproven. See the measured first increment at the end of this document.

## Recommendation

Move toward a conservative, large-step geometric remap on the existing sparse
bricks. Use adaptivity to reduce geometry work without sacrificing fine interface
detail merely to satisfy a transport CFL limit.

Resolve one substantial feasibility question first: can the remap conserve
liquid and respect receiving-cell capacity without an expensive global
correction? Establish that before committing to a full rewrite.

## 1. Research findings

| Approach | Useful property | Limitation for this solver |
| --- | --- | --- |
| Existing geometric substeps | Shared transfers, bounded liquid amounts, functioning implementation | Work grows with the worst transport CFL |
| Adaptive-mass conservative semi-Lagrangian transport | Direct transport across multiple cells | Donor conservation alone does not guarantee bounded receiving volumes |
| Conservative cell-integrated semi-Lagrangian remapping | Integrates material over departure regions; supports long transport distances | Compatible geometry in 3D is difficult |
| Directional long-step remapping | Structured scans and range sums suit GPUs | Multidimensional compression, bounds and splitting require careful treatment |
| Unsplit geometric VOF | Explicit interface geometry with strong conservation properties | “Semi-Lagrangian” does not automatically imply unrestricted Courant numbers |

Lentine–Grétarsson–Fedkiw establishes the useful principle behind adaptive-mass:
conservative transport can avoid the usual advective CFL restriction. Its
normalization machinery does not, by itself, supply the geometric occupancy
guarantees needed here. [Original paper](https://physbam.stanford.edu/~fedkiw/papers/stanford2010-01.pdf).

Long-step flux-form schemes also demonstrate that transport distance need not
imply proportionally more interpolation work. A published comparison describes
directional transport whose arithmetic cost changes little when the Courant
number exceeds one. This is promising for brick scans, although it is atmospheric
transport rather than a ready-made free-surface solver.
[Chen et al., 2017](https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.3125).

For 3D geometry, Comminal–Spangenberg and Fröde et al. provide relevant
conservative cell-based constructions. The latter corrects projected-cell
geometry using conservative flux volumes. These are construction references;
they do not establish the desired high-Courant, adaptive, cut-cell implementation
without further derivation.
[Comminal–Spangenberg, 2021](https://orbit.dtu.dk/en/publications/three-dimensional-cellwise-conservative-unsplit-geometric-vof-sch/),
[Fröde et al., 2022](https://www.sciencedirect.com/science/article/abs/pii/S0021999122004363).

The algorithm below is a proposed synthesis. Its performance and capacity
compatibility remain hypotheses to validate, not conclusions established by
these papers.

## 2. Numerical contract

Retain the current authority:

- `V_i`: liquid volume.
- `K_i`: available cell capacity after solids.
- `0 <= V_i <= K_i`.
- Physical mass is constant liquid density times `V_i`.
- Interface geometry is reconstructed from `V_i`; it cannot independently change
  liquid amount.

For an ideal incompressible flow map `Phi`, destination liquid volume is:

```text
V_j^(n+1) = volume(Phi^(-1)(Omega_j^(n+1)) intersect L^n)
```

Here `L^n` is the reconstructed liquid region. Trace the receiving region
backward and measure the liquid it contains.

Two properties are essential:

1. Departure regions partition the source domain without gaps or overlap.
2. Their volumes agree with destination capacities, with explicit accounting for
   boundaries and moving solids.

Together these give conservation and bounds. A discretely divergence-free
velocity field does not automatically make independently traced polyhedra
satisfy both properties.

This is the main research gate. Normalizing donor totals, clamping overflow, or
redistributing excess afterwards would conceal a failure of this contract.
Shared geometry and any geometric correction must have an explicit derivation,
measured cost, and an accuracy check.

Shared vertices alone are insufficient. For rigid rotation, an explicit backward
Euler trace has determinant `1 + (omega * dt)^2` in the rotation plane, despite
zero velocity divergence. Higher-order tracing reduces this defect but does not
generally enforce unit determinant. Nonlinear flow also bends faces between
traced vertices. Specify the face representation, trajectory integrator,
partition construction and volume-preservation mechanism before expanding the
implementation.

For a closed or periodic domain, audit geometric transfer volumes `A_ji`:

```text
A_ji >= 0
sum_j A_ji = K_i^n       (source coverage)
sum_i A_ji = K_j^(n+1)   (receiver capacity)
```

For liquid transfers, independently audit each donor's total against `V_i`,
and each receiver against its capacity. Check signed global error, absolute
local error and the maximum local error; cancellation is not evidence of
compatibility. Open boundaries, sources and live additions/removals require an
explicit ledger. These equations use open cell regions, not the enclosing solid
cell boxes. Tolerances must be declared before observing candidate results.

## 3. Meaning of large-timestep support

Separate translation from deformation:

```text
Co = |u| * dt / h
D  = ||gradient(u)|| * dt
```

- Large `Co` can simply mean a shape translates many cells.
- Large `D` indicates substantial deformation during the step.

The target is cheap large translation Courant numbers. Trajectory accuracy,
deformation, wall interactions and changing pressure can still require extra
work.

Translating a cell by 25 cell widths need not involve 25 successive volume
updates. Under uniform translation, its final image overlaps only a small number
of destination cells. Direct destination lookup can find them.

Increasing cell width reduces Courant number. The goal is to remain efficient
when fine cells have large Courant numbers, rather than coarsening them until
their Courant numbers become small.

## 4. Proposed architecture

Retain sparse bricks, current volume authority and the production method identity.

1. **Compile topology when it changes.** Store direct cell/face ownership,
   coarse–fine relationships, reconstruction stencils, solid geometry references
   and brick lookup structures.
2. **Predict transport velocities for the outer step.** Provide temporally
   appropriate velocities and valid support wherever characteristics travel.
3. **Construct a compatible geometric map.** Share geometry across neighboring
   cells and coarse–fine boundaries. Trace shared entities consistently.
4. **Build overlap work lists.** Locate relevant source/destination bricks
   directly, then enumerate actual intersecting cells.
5. **Integrate liquid geometry.** Use simple full/empty-region calculations where
   valid. Clip reconstructed interfaces only where necessary.
6. **Reduce transfers and commit once.** Use explicit ownership and segmented
   reductions. Validate conservation, capacities and coverage before accepting
   the state.

Topology can be cached; velocity-dependent trajectories and overlaps generally
cannot. Give them separate generation counters to prevent stale geometry reuse.

Desired cost model:

```text
topology changes + characteristic work + actual overlaps + interface clipping
```

Current cost pattern to eliminate:

```text
global CFL substeps * whole-domain transport
```

For uniform translation, direct endpoint lookup should avoid stepping through
every crossed cell. General deformation can increase overlap complexity; this
is an explicit measured cost, not something the design assumes away.

## 5. Role of adaptivity

Use adaptivity in three distinct ways:

- **Representation:** retain fine cells around interfaces, thin features and
  obstacles; coarsen smooth bulk where appropriate.
- **Computation:** aggregate compatible full-fluid regions and use brick-level
  bounds or range sums without changing fine-cell state.
- **Geometry:** subdivide trajectories or mapped regions where deformation makes
  the approximation inaccurate.

Computational aggregation can save work without coarsening the surface.

A uniform velocity boost should not cause refinement everywhere. Resolution
criteria should respond to geometry and velocity variation, not absolute speed
alone.

Endpoint lookup is insufficient around solids: a trajectory must not pass
through a wall. Hierarchical traversal can skip unobstructed regions, but
collision handling remains part of the transport contract.

## 6. Staged delivery

Each production increment must remain inspectable in the ordinary UI and use
Sparse Geometric as the default. Numerical feasibility work uses Dawn probes;
it must not create another permanent user-facing method.

| Stage | Work | Acceptance condition |
| --- | --- | --- |
| A. Establish measurements | Capture transport cost, substep count, reconstruction count, limiter iterations, volume error and bounds | Reproducible baseline with source/settings fingerprints |
| B. Prove long-range transport | Small all-fine translation cases; direct geometric overlaps across multiple cells | Correct volume, bounds and displacement without global CFL substeps |
| C. Prove map compatibility | Divergence-free deformation; shared mapped geometry; full-domain capacity checks | No gaps, overlaps or systematic capacity defects hidden by repair |
| D. Add brick acceleration | Compiled ownership, overlap lists, full/empty fast paths, deterministic reductions | Measured reduction in total transport time and bounded memory use |
| E. Extend across adaptivity and solids | Coarse–fine geometry, refinement/coarsening, static cut cells, then moving boundaries | Same conservation contract across every supported boundary |
| F. Integrate coupled dynamics | Pressure/velocity timing, newly wetted support, momentum consistency | Accurate large-step fluid motion, beyond volume conservation |
| G. Complete scene qualification | Geometric ladder scenes to 3 seconds; canonical Dawn gate | Receipts for every scene and explicit remaining failures |

Stages B–C are the decisive feasibility work. Production improvements from A and
D can land independently while that derivation is resolved. Replace the
transport core only when it covers supported production behavior; a restricted
default would violate the existing working agreement.

Split C into three separate gates:

- **C1: exact map.** An analytic volume-preserving map isolates overlap geometry
  from trajectory error. Translation alone does not satisfy this gate.
- **C2: nonlinear map.** Spatially varying deformation uses the proposed face
  representation and map construction. Include a deliberately incompatible
  corner-traced control to establish that the audits detect local defects.
- **C3: production velocity bridge.** Drive the map with actual discrete
  production velocities and their interpolant. Establish departure support,
  receiver residency, projection residual compatibility and full-cell
  preservation before investing in Stage D. A prescribed analytic field bypasses
  these issues and cannot pass C3.

The current eight-sweep velocity extension is not by itself proof of support
for a 25-cell departure trace. Include support generation, sparse allocation,
trajectory work and projection in early end-to-end cost measurements.

First construction to investigate: compositions of volume-preserving shears.
For `x' = x + f(y)` the Jacobian determinant is one. A continuous piecewise
linear `f` gives a piecewise affine map. Split geometry at every crossed knot
plane before applying the shear; shared boundaries then have identical images.
Compose shears on different axes without intermediate liquid reconstruction or
volume commits. Each piece retains unit determinant and the composition remains
bijective. Merely applying the nonlinear composition to original cell corners
does not supply these properties. This is a restricted candidate map family,
not yet a derivation for arbitrary production velocities. Measure geometric
fragment growth and map approximation error as well as conservation.

CPU geometry compilation is acceptable for the initial Dawn feasibility probe,
with CPU and GPU costs reported separately. It is not production GPU acceleration
and must not be presented as Stage D completion.

Each completed production increment cuts over immediately. Do not defer
promotion behind a preview mode, silently fall back to adaptive-mass, or weaken
existing acceptance thresholds. Preserve a functioning simulation throughout.

## 7. Correctness ladder

Run cases in order. Investigate failures before adding complexity. Use analytic
or prescribed-velocity cases to isolate transport before evaluating full fluid
dynamics.

### 7.1 Uniform translation

Start with the existing tiny all-fine plug. Add axis-aligned and diagonal
movement, integer and fractional displacements, and target Courant numbers near
0.5, 2, 8 and 25.

Keep enough domain length to avoid boundaries. Measure:

- Total volume and per-cell bounds.
- Analytic cell-volume error.
- Centre-of-mass displacement.
- Transfer count and GPU time.

An integer-cell shift cannot be the only success criterion; it can conceal
interpolation and reconstruction errors. Record executed simulation time as
well as requested time.

### 7.2 Oblique planar interface

Use exact initial cell intersections and prescribed constant velocity. This
isolates interface reconstruction from pressure and dynamics.

Reconstruction should match neighboring volumes while enforcing central-cell
volume. Raw volume-fraction gradients are insufficient as the general accuracy
contract. Pilliod–Puckett provides the relevant line-preserving reconstruction
framework; a 3D implementation must establish its own corresponding behavior.
[Pilliod–Puckett, 2004](https://www.math.ucdavis.edu/~egp/PUBLICATIONS/JOURNAL_ARTICLES/APPEARED/2004/JEP-EGP-2004.pdf).

Do this before polishing detached-box corners: a corner cannot generally be
represented exactly by one plane. Avoid axis snapping as a generic remedy.

### 7.3 Rotation and reversible deformation

Use prescribed divergence-free velocity to separate transport from the fluid
solver. Check boundedness, volume, return-shape error, mapped-cell validity and
convergence. These cases expose errors uniform translation cannot reveal.

Include a fully filled domain to test capacity preservation separately from
interface clipping. Log departure coverage and geometric capacity residuals,
not just the final liquid-volume sum.

Include partial occupancy and a forward/inverse shape test. Separate refinement
of the flow-map representation from refinement of the liquid reconstruction.
An exact affine shear or rotation cannot be the only deformation success.

### 7.4 Adaptive crossings and cut cells

Move the same objects across refinement boundaries, then around static solids.
Include very small open capacities. Refinement and coarsening need an explicit
volume ledger.

Use the same solid geometry for capacity and transport clipping. Moving solids
come afterwards and require swept-boundary accounting consistent with changing
capacity.

### 7.5 Coupled fluid cases

Run the steady vortex, thin moving material and dam front. These test velocity
transport, pressure coupling and wetting rather than only geometric remapping.

Finally run every geometric ladder scene to 3 seconds. Record actual selected
method, settings, source fingerprint, final simulation time, correctness metrics
and timing for each result.

## 8. Dissipation remains a separate acceptance problem

A better volume algorithm does not automatically fix velocity interpolation or
pressure-splitting losses.

Record energy changes separately across:

- Velocity preparation/advection.
- Forces.
- Projection.
- Topology transfer.

The steady vortex removes the free surface. If it dissipates, changing interface
transport cannot be the complete fix.

For large outer steps, investigate consistent midpoint or predictor–corrector
coupling. Do not assume frozen velocity through a long remap gives accurate
newly wetted flow. A pressure solve per cell crossing must not become the
default solution; that would surrender the intended performance benefit.

Dam-front timings provide downstream evidence, but compare shallow-water
analytic fronts only within their applicable assumptions. Do not treat finite
wall-wetting thresholds after impact as exact Ritter-front targets.

## 9. Performance gates and integration points

Measure the same active grid and geometry while varying displacement, wherever
boundaries permit. The first architectural gate is:

> Increasing translation Courant number must not increase the number of
> whole-domain volume updates.

Measure:

- Total GPU transport time, including geometry construction and reductions.
- Total frame time.
- Overlaps and clipping operations per cell.
- Temporary memory and work-list overflow.
- Topology compilation cost separately from steady-state cost.
- Scaling with interface area and deformation.

Set numerical timing targets after collecting the baseline; do not invent a
millisecond budget before measurement. Existing regression timing ceilings stay
unchanged. Compare alternative measurements on the same hardware and settings,
with topology changes and warm/cold costs identified separately.

Main integration points:

- [Resident transport encoder](../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts).
- [Geometric volume planner and kernels](../lib/methods/adaptive-volume/resident-volume.wgsl.ts).
- [Directional transport notes](geometric-split-transport.md).
- [Existing incremental production plan](adaptive-geometric-volume-plan.md).

Activating the existing directional helpers alone does not meet the large-step
goal: their documented donor-prism restriction still requires short steps.

Use Dawn integration probes rather than unit tests. After substantial production
changes, run the canonical gate:

```bash
npm run test:dawn:sparse-cm12
```

Serialize Dawn with browser/GPU use and other Dawn processes. Do not silently
weaken a lane or raise a timing ceiling. Distinguish pre-existing failures from
regressions using comparable receipts.

## 10. Reasons to reconsider the method

Pause the design if:

- Capacity compatibility requires repeated global redistribution.
- Ordinary translation produces work proportional to every crossed cell.
- Mild deformation causes widespread geometry subdivision.
- Fine interfaces require coarsening to run efficiently.
- Conservation depends on corrections that visibly distort transport.
- Large-step coupled dynamics remain inaccurate despite accurate prescribed
  transport.

If unsplit polyhedral remapping fails these gates, the next candidate is
directional conservative long-step transport using brick range sums, with an
explicit derivation of multidimensional bounds and compression. It offers a more
regular GPU workload, but must satisfy the same tests.

## First concrete deliverable

A tiny Dawn case demonstrating bounded, conservative fractional-cell translation
at large Courant number, followed immediately by a deformation case proving
capacity compatibility. These determine whether the direction preserves the
original large-timestep advantage before broad implementation proceeds.

Predeclare numerical tolerances, record raw overlap volumes without repair, and
include integer/fractional axis and diagonal translations through Co 25. Use
periodic domains where necessary and identify them explicitly. Report requested
and executed displacement, one final volume commit, overlap counts, fragment
counts, temporary storage, CPU geometry time and GPU execution time. Mark C1,
C2 and C3 independently; a restricted-map success does not validate the complete
solver. Update this document with measured results and remaining obstacles.

## First implementation and verification — 2026-09-12

Implemented a reference map compiler and a standalone Dawn probe:

- [CPU geometry construction](../lib/core/geometric-remap/geometry.ts): split convex
  cells at shear knots, transport the original affine liquid attribute through
  every piece, tetrahedralize only the final geometry, then enumerate destination
  cells directly from final bounds. Intermediate shears do not reconstruct or
  commit liquid volumes. Periodic indexing permits long translations.
- [Dawn intersection kernel](../lib/core/geometric-remap/overlap.wgsl.ts): bounded
  indexed convex clipping, raw bulk/liquid intersections, explicit overflow
  sentinels and deterministic segmented receiver reduction. No volume clamps,
  donor normalization or receiver redistribution.
- [Verification driver](../tools/probe-geometric-large-remap-dawn.ts): independent
  donor and receiver ledgers, exact translation oracle, independent inverse-map
  shape quadrature, map-resolution convergence, negative control, fingerprints,
  work counts, memory and separated CPU/GPU timing.

Run serially with Fluid Lab browser tabs unloaded:

```bash
npm run probe:dawn:adaptive-volume:large-remap
npm run probe:dawn:adaptive-volume:large-remap -- --list
npm run probe:dawn:adaptive-volume:large-remap -- --case=nonlinear-composition-8
npm run probe:dawn:adaptive-volume:large-remap -- --size=4
```

The default matrix has 16 cases: integer and fractional axis/diagonal translation
through 25.5 cells; one nonlinear shear; three composed shears at two map
resolutions; full occupancy; forward/inverse composition on retained geometry;
the 4³ coplanar regression; and an intentionally incompatible corner-only map.
Most cases use an 8³ periodic all-fine lattice. The separate 4³ matrix changes
the domain and deformation strength relative to its size; it is an additional
robustness test, not a fluid-grid convergence study.

Receipts:

- [Default matrix](../artifacts/analytic-motion/large-remap-verified.json).
- [4³ matrix](../artifacts/analytic-motion/large-remap-verified-4.json).
- [Existing production translation baseline](../artifacts/analytic-motion/large-remap-production-baseline.json).
- [Retained failing coplanar case before its fix](../artifacts/analytic-motion/large-remap-inverse-coplanar-before.json).

The first complete 8³ run with the independent shape oracle gave:

| Measurement | Result |
| --- | --- |
| Maximum analytic translation cell-volume error | 2.76e-7 cell volumes |
| Maximum receiver capacity error, refined nonlinear map | 9.54e-7 cell volumes |
| Relative total liquid error, accepted 8³ cases | Below 1e-7 |
| Diagonal translation candidate count, Co 0.5 / 2.5 / 8.5 / 25.5 | 32,768 for every displacement |
| Nonlinear smooth-map position error, 8 to 16 knots | 0.08054 to 0.02095 cell widths; 3.84× reduction |
| Nonlinear composition versus independent 32³-per-cell inverse quadrature | Maximum cell discrepancy 0.00279; mean below 0.00027 |
| Corner-only control donor coverage defect | 0.02122 cell volumes; correctly rejected |

The corner-only control still fills receivers to roundoff. Its **donor** ledger
exposes the incompatible map; checking receiver bounds and a global liquid sum
alone would be inadequate. The negative control must have finite, bounded raw
intersections and valid receiver capacity, so a kernel failure cannot count as
successful rejection of the numerical method.

The independent shape oracle uses midpoint quadrature at 16³ and 32³ samples per
cell. Its discretization differences are recorded; its agreement is not a claim
of exact nonlinear shape integration. The inverse case composes forward and
inverse maps before one volume update. It does not establish reversibility after
repeated PLIC reconstruction. These probes execute prescribed maps, not a physical
fluid clock; production timestep accuracy and unwrapped long-range residency are
still outside their scope.

Two implementation obstacles were found and resolved:

1. Composing the existing by-value polyhedron clipper exceeded Metal's available
   compute stack. An indexed polyhedron with pointer-based clipping compiled and
   ran. It retains explicit vertex/face capacity checks and returns a fault
   sentinel on overflow instead of truncating geometry.
2. A 4³ forward/inverse case duplicated a cap when f64 roundoff put an existing
   face infinitesimally across a knot. It gained 1/12 cell volume on both CPU and
   GPU. A shared coplanar predicate within 32 f64 ulps of coordinate scale fixes
   that topology error. State amounts are never corrected. The default matrix
   permanently includes the triggering case.

**Performance is not production-ready.** On this machine the initial 512-cell
measurements were roughly 3 ms GPU for fractional diagonal translation, 5–6 ms
for three nonlinear shears, and 11–12 ms for the refined map. CPU map construction
was roughly 0.1 s for the three-shear map and 0.6 s after refinement. Refinement
raised maximum pieces per cell from 2 to 14 and total pieces from 1,024 to 6,656.
GPU buffer allocation including QA readback was about 5 MB and 13 MB respectively.
These are probe timings, with immutable-input warmup and quantized timestamps;
zero reported gather duration means below timestamp resolution. Whole-case time
also includes expensive CPU shape oracles. No production speedup is claimed.

The production plug baseline still passes through 0.05 s, using four transport
microsteps per outer step at CFL 2. Its grid, boundaries and coupled physics differ
from the reference probe, so its timing is not an apples-to-apples speed ratio.

### Assessment and next verification target

- **B:** passes for prescribed periodic translation with direct destination
  lookup and one volume gather. It does not yet establish sparse growth.
- **C1:** passes for the exact piecewise affine shear maps.
- **C2:** passes for the tested nonlinear shear compositions, including shape
  checking, map-resolution convergence and the negative control. This is a
  restricted map family, not arbitrary velocity-field transport.
- **C3:** open. No bridge from arbitrary production face velocities to this
  compatible map has been implemented or validated.
- **D–G:** open. CPU geometry compilation, dense periodic indexing and prescribed
  shears are feasibility scaffolding. Adaptivity, solids, sources, physical time
  integration, momentum and production performance remain required.

The next decisive deliverable is C3: define an approximation of actual production
velocity fields by volume-preserving maps, measure velocity/trajectory error as
well as occupancy, and test the production interpolation and support boundary.
The three coordinate shears tested here cannot represent an arbitrary
divergence-free field. Do not fit an arbitrary field into that family and declare
success based only on conservation. Investigate a richer volume-preserving map
representation or reconsider the directional long-step candidate if fidelity or
fragment cost defeats the intended advantage. Optimize broad GPU geometry only
after that compatibility/fidelity question has a concrete passing probe.

No production simulation files changed in this increment. The targeted Dawn
probes and existing production translation baseline are its verification scope;
the full canonical production regression gate remains required for subsequent
substantial production changes.

Final verification: both 16-case matrices pass, including the expected rejection
of each corner-only control, with no WebGPU validation errors and unchanged
source fingerprints during each run. The production translation baseline also
passes. Targeted TypeScript checking of the three new probe files passes;
repository-wide TypeScript checking reports the same pre-existing errors outside
these files as before this increment.

## Shared repeated session and small UI scene — 2026-09-12

The reference geometry and GPU kernels now live under
[`lib/core/geometric-remap`](../lib/core/geometric-remap/) so the Dawn probes and
the standalone UI use the same implementation. The shared
[`GeometricRemapSession`](../lib/core/geometric-remap/session.ts) reconstructs
the **last accepted** cell volumes, compiles one prescribed map, validates raw
donor and receiver transfers, and publishes exactly one new volume state per
step. It does not repeatedly resample the authored initial shape.

Run its integration probe serially, with Fluid Lab browser tabs unloaded:

```bash
npm run probe:dawn:adaptive-volume:remap-session
```

The verified session receipt is
[`large-remap-ui-session.json`](../artifacts/analytic-motion/large-remap-ui-session.json).
It records 24 consecutive accepted steps for fractional translation, 25.5-cell
translation, nonlinear slab deformation and nonlinear oblique deformation, plus
16 full-domain deformation steps. Across those 112 steps, maximum cumulative
relative volume drift was `8.5831e-7`; maximum cell error against the exact
translation oracle was `2.0862e-6`. The full-domain case remained bounded and
capacity compatible. The probe reported no WebGPU validation errors and its
source fingerprint remained unchanged during that run. The receipt is pinned to
the source snapshot it records; the later package-script and documentation edits
were not part of the fingerprinted run.

Resetting while GPU readback is in flight increments the session generation.
The completed work checks that generation immediately after readback and before
consulting or publishing mutable session state. The stale step is rejected as
cancelled, GPU buffers are released, and the reset state can advance normally.
The probe exercises this race explicitly.

The original 16-case prescribed-map matrix was rerun against the shared paths:

```bash
npm run probe:dawn:adaptive-volume:large-remap \
  -- --out=artifacts/analytic-motion/large-remap-ui-core-verified.json
```

All 16 cases passed with no WebGPU validation errors and an unchanged source
fingerprint. The nonlinear map convergence ratio was `3.8448`. See
[`large-remap-ui-core-verified.json`](../artifacts/analytic-motion/large-remap-ui-core-verified.json).
These results are correctness evidence for the reference prescribed-map kernel;
they do not establish a production speedup. The repeated nonlinear session
checks establish conservation, receiver capacity and cell bounds. They do not
provide an independent long-time interface-shape accuracy oracle; the exact
per-cell oracle in the repeated session applies only to translation.

The small scene is available at `/remap-lab`. It uses an 8³ periodic grid and
offers slab, oblique-interface and full-domain initial shapes; travel from 0 to
25.5 cells per step; optional prescribed volume-preserving shear deformation;
single-step, play, pause and reset controls; and volume-cell or XY-slice views.
The live panel reports accepted liquid volume, drift, capacity and donor errors,
one volume update per step, overlap candidates, geometry pieces, CPU geometry
time and GPU submission/readback time. A long step wraps across periodic faces.

This route is a bounded inspection scene, not a second fluid method. It uses CPU
geometry construction, dense periodic cells, prescribed motion and GPU overlap
integration. Pressure projection, gravity, momentum transport, production
velocity interpolation, sparse adaptivity, solids and physical timestep
coupling are absent. Repeated interface reconstruction can diffuse shape. The
scene therefore demonstrates the B/C1/C2 feasibility increments interactively;
it does not close C3 or support a production-performance claim.

Browser verification passed in Chrome at `1728 × 938`, ending at
`http://localhost:3000/remap-lab` in the default reset/Ready state with the
remap scene as the sole active GPU scene. The run exercised all four travel
presets (`0.5`, `2.5`, `8.5` and `25.5` cells), 100% deformation, play through
12 accepted steps, pause and reset, all three initial shapes, view angle `1.2`,
and XY slice Z7. Oblique and full states both reset and advanced successfully.
The largest observed browser drift was `4.77e-8`; maximum capacity error at full
deformation was `4.17e-7`. Browser warning and error logs were empty.

The isometric and slice views were visually inspected at that viewport and were
crisp, legible and unclipped. A route and lease-lifecycle round trip also passed:
remap scene → library → Small room standard studio with an interactive viewport
and FPS display → Browse library → remap scene → Ready. The worker distinguishes
recoverable step errors from fatal device/lifecycle failures and guarantees a
close acknowledgement during shutdown. Repository-wide TypeScript checking
still reports pre-existing unrelated errors, with no diagnostics in the remap or
UI files. No numerical core changed after the Dawn receipts, so the Dawn probes
were not rerun for this browser/lifecycle increment.

## C3 rejected construction: piecewise-constant tetrahedral flux lift

The first arbitrary-production-velocity construction investigated after the
restricted shear maps was a piecewise-constant `H(div)` lift on a conforming
Freudenthal six-tetrahedron subdivision of every Cartesian cell. It has an
attractive exact local argument. A constant velocity is divergence free inside
each tetrahedron. If adjacent tetrahedra share the same normal velocity across a
face, crossing from velocity `v_a` to `v_b` through `n.x = c` gives, on one fixed
itinerary,

```text
t_cross(x) = (c - n.x) / (n.v_a)
y(x) = x + v_b dt + (v_a - v_b) t_cross(x)
det(dy/dx) = 1 - n.(v_a - v_b) / (n.v_a) = 1.
```

Successive crossings therefore compose affine determinant-one maps. Pulling
the next-exit comparisons back to source coordinates gives planar clipping
predicates. The prototype retained source coordinates plus affine current-
position and elapsed-time maps for every piece; it did not collapse a
three-dimensional piece onto its two-dimensional crossing face or reconstruct
liquid between crossings.

An early version constrained only domain-boundary triangles. That was invalid:
it could satisfy capacity while ignoring the production velocities on internal
MAC faces. The corrected construction prescribed the physical volume flux on
every Cartesian face triangle, constrained both incident tetrahedra to that
same canonical normal flux, and left only intra-cell diagonal fluxes free. A
balanced 2 × 2 × 1 circulation test then changed the lifted field when its
interior MAC circulation changed, while retaining `6.66e-16` maximum shared
normal mismatch and `1.94e-16` maximum boundary-flux error. Unbalanced boundary
flux was rejected rather than repaired in volume space.

This construction nevertheless fails the C3 cost and completeness gate. The
normal-continuous field has tangential jumps across the artificial tetrahedral
faces. In a closed 2 × 2 × 1 manufactured vortex, donor zero repeatedly followed
the internal-face itinerary

```text
1, 8, 11, 14, 5, 2, 1, 8, 11, 14, 5, 2, ...
```

around the cube body diagonal. After 30 crossings its affine crossing-time
range still covered `0` through `0.05000000000001137`. The largest lifted speed
was `0.1275611441` cell widths per unit time, so maximum travel over the step was
below `0.00638` cell widths. This is not a large-Courant problem or an
independently chosen face-sign error. Canonical shared face-flux signs and
explicit exclusion of the entry face did not remove it. Tangential
discontinuities admit arbitrarily rapid cycling near the codimension-two
tetrahedral edge fan, producing unbounded itinerary fragmentation over a finite
step.

Raising an event cap, discarding small pieces, or selecting an arbitrary
continuation at the edge would not establish a partition of the source volume.
The piecewise-constant tetrahedral event map is therefore rejected for
production. It was not added to the active simulation, and it does not close
C3. The CPU evidence receipt is
[`artifacts/analytic-motion/c3-p0-hdiv-rejection.json`](../artifacts/analytic-motion/c3-p0-hdiv-rejection.json).

### Next C3 choice

A smoother divergence-free lift avoids the tetrahedral edge fan, but does not
immediately retain polyhedral exactness. Continuous piecewise-linear or
Fourier/vector-potential fields have curved finite-time images; ordinary
trajectory integration restores a local Jacobian error and independently
traced cells no longer provide an exact partition. Compositions of exactly
volume-preserving shears remain useful on a periodic uniform domain, but fitting
enough globally oriented shears to an arbitrary projected field can cause the
same fragment-growth problem, and coordinate shears do not preserve general
solid walls.

The strongest next bounded experiment is a conservative directional long-step
remap on a periodic all-fine grid, before adding walls. Each one-dimensional
pass should transport both liquid volume and an explicit geometric
capacity/Jacobian measure with identical positive long-step weights. The pass
must audit donor coverage and receiver capacity before any multidimensional
commit. A symmetric composition then has to return the transported capacity
measure to the fixed Eulerian cell capacities; failure to do so is a failed
multidimensional compatibility test, not overflow to clamp. Test constant and
balanced variable production face fluxes, a full domain, a planar interface and
forward/reverse deformation through Courant numbers 0.5, 2, 8 and 25. Compare
against the production trilinear cell-centre interpolant with its RK2 timing,
while reporting face-flux mismatch separately.

There is a real cost-versus-fidelity limit behind the original large-step goal.
Uniform translation can and should remain nearly independent of Courant number.
An arbitrary velocity field that varies along a 25-cell path contains physical
information from those crossed regions; a faithful method must either inspect
that information, use a certified hierarchical aggregate, or accept a measured
approximation. The performance target should therefore remain independent of a
uniform velocity boost and scale with resolved deformation and encountered
field complexity, rather than promise constant work for every arbitrary
large-Courant field.

## Production presentation repair found during C3 integration work

The production Sparse Geometric path remains the existing adaptive-volume
solver; no experimental large-step map is selectable there.  Exercising that
path exposed a separate presentation defect in the `min8-region-surface` Dawn
lane.  After 48 quiescent steps the conservative density column moved by at
most `3.8147e-6` finest cells, while the published scalar-isovalue surface moved
by `0.3245869` cells and formed a `0.4222`-cell boundary bump.  Pressure and
post-projection divergence remained clean.  A one-step receipt had exactly zero
before/after publication drift, excluding reset-versus-frame publication and
accepted-bank parity as the cause.

The coarse presentation stencil contours cell-centred volume fractions at
`rho=0.5`.  Redistribution between vertically adjacent coarse authorities can
move that isovalue even when their summed liquid volume, and therefore the
physical waterline, is unchanged.  Sparse Geometric now defaults the Column
height control to `Auto`.  Auto attempts the existing volume-integrated height
receipt only on adaptive coarse bricks.  That receipt still requires complete
support, a floor-connected monotone column, a waterline inside the page, and a
locally calm height field.  A failed certificate retains the geometric scalar
path.  `Off` explicitly disables column presentation; `On` requests the prior
broader certified-column policy.  Auto does not select the column path for
all-fine surfaces.

The first Auto prototype made `min8-region-surface` report zero published drift,
but later negative testing showed that it reached an unsafe local-bracket
fallback for a floating coarse slab. That receipt is superseded and is not
acceptance evidence. Strict Auto now stops when the complete floor-connected
receipt fails. The positive B2/B1 defect was at the request site: the accepted
surface pages are at brick y=1 and reset marks them with surface reason bit 64,
while the first policy attempted the adaptive floor walk only on brick y=0 and
omitted bit 64. The corrected policy lets any accepted surface/thin page request
the same whole-world floor walk; the walk still decides whether the column is
eligible.

With that correction, the unchanged `min8-region-surface` lane passes after 48
steps with zero published height drift and zero B2/B1 boundary bump; conservative
density-height drift remains `3.8147e-6` finest cells. Controlled Auto/Off runs
of both the floating coarse/fine translation and free-fall scenes are exactly
equal for reset and final published volume and surface-error receipts, proving
that Auto rejects their dry floor gap. The machine-readable receipt is
[`artifacts/analytic-motion/strict-auto-column-certificate.json`](../artifacts/analytic-motion/strict-auto-column-certificate.json).
An adaptive sphere probe also passed with symmetric X/Y/Z surface receipts; its
liquid-bearing surface bricks were all-fine, so Auto retained the non-heightfield
geometry path.

This repairs an ordinary production presentation failure. It does not implement
C3: the general large-timestep transport problem and the directional-remap
experiment described above remain open.

The moving-solid production path also received a bounded feasibility repair.
Its capacity-constrained dual now uses moving-only diagonally preconditioned
FISTA and a fail-closed 1024-pass work budget; static transport retains its
existing limiter. Pressure membership anticipates moving-solid filling from
authoritative raw liquid volume and supported net incident flow, rather than
the open-fraction-normalized pressure density. This predictor is not a
post-projection capacity guarantee: the qualified 30-frame rigid run still
observed three nonmember endpoint-risk cells, which the conservative transport
legally rerouted without a bound or feasibility fault. The independent graph
feasibility evidence, rejected broader predictor, pass counts, final bounds and
scope limits are recorded in
[`artifacts/analytic-motion/rigid-moving-volume-qualification.json`](../artifacts/analytic-motion/rigid-moving-volume-qualification.json).

The first full Sparse CM12 Dawn matrix after these repairs completed in
`477.368 s` of its `480 s` overall budget. Seven lanes passed, including both
`min8-region-surface` and `mini64-min8-surface`, the repaired live rigid-body
coupling lane, and live liquid injection. Ten lanes reached their existing
per-lane time ceilings; none failed an assertion. That historical chunk-eight
receipt, exact lane list, raw log and frozen-source hashes remain recorded in
[`artifacts/dawn-regression/full-canonical-2026-09-12.json`](../artifacts/dawn-regression/full-canonical-2026-09-12.json).

The current production schedule batches 32 transport packets between host
continuation receipts while device readiness still decides the exact packet
that completes the numerical step. In a matched mini64 diagnostic, chunk 8 and
chunk 32 produced identical density, velocity and pressure hashes, 649 total
limiter passes, a maximum of 74 passes per microstep, ten completed substeps and
zero transport faults. Chunk 32 reduced the continuation batches from about 82
to 21 and the single retained hardware-sample advance from `374.604 ms` to
`345.309 ms`; that small diagnostic sample is not a production performance
qualification. Its full receipt and limitations are in
[`artifacts/dawn-regression/chunk8-vs-chunk32-mini64-2026-09-12.json`](../artifacts/dawn-regression/chunk8-vs-chunk32-mini64-2026-09-12.json).

The final unfiltered matrix on chunk 32 completed in `477.255 s` with the same
seven passes, ten existing timeout classifications and zero assertion failures.
It supersedes the chunk-eight matrix as the current whole-source result; the
machine-readable receipt is
[`artifacts/dawn-regression/full-canonical-chunk32-2026-09-12.json`](../artifacts/dawn-regression/full-canonical-chunk32-2026-09-12.json).
The unchanged timeout set and the limited A/B sample do not establish a general
performance improvement. C3 is still not integrated: arbitrary production
velocity coupling, long-timestep accuracy and production performance remain
open work.
