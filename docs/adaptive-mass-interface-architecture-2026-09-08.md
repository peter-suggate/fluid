# Persistent adaptive liquid interfaces: architecture and validation

Status: research proposal with a CPU information/remapping experiment. No
production solver change. The representation, coupled dynamics, and GPU cost
are not yet validated. This is the long-term direction following the paused
region-edit distortion report; it does not propose interim surface patches.

The subsequent [density-field contract](adaptive-mass-density-field-contract.md)
clarifies the governing requirement: density means are integrals of retained
implicit state, and splitting physics cells does not change that state. The
user's sharp-feature insight means that "continuous field" below must not be
read as a requirement to smooth edges or to impose a globally smooth scalar.
The field may be piecewise smooth or have a sharp indicator interpretation.
This contract takes precedence over the provisional representation choices here.

## Decision being investigated

Make liquid interface detail persistent simulation state, with an adaptive
spatial partition that need not match the pressure/velocity partition. A
physics-region edit must repartition physical work without discarding the
represented interface. Interface coarsening is a separate operation justified
by geometric error, not a consequence of pressure-cell coarsening.

This is a hypothesis to validate, not a selection of a finished algorithm.
The leading candidate is a sparse geometric volume-fraction hierarchy coupled
to a continuous interface field. Compare a moment-enriched geometric hierarchy
before choosing its local reconstruction. Neither plain PLIC nor an advected
level set by itself satisfies all the requirements below.

The user explicitly rejected quick fixes. Do not first ship a special pool
height reconstruction, region exception, t=0 reseed, surface smoothing pass,
or transfer tweak as the proposed solution. Diagnostic and isolated research
implementations are useful; production migration follows evidence.

## Why a new representation contract is necessary

Current accepted cells retain density means. Conservative restriction loses
subcell location and topology. The resident transfer has special reconstruction
for B4-to-B8; most other refinements copy the parent density. Presentation uses
a separate, neighbour-dependent sampling width, and fine/coarse publication
uses different reconstruction paths. Paused region planning enforces bounds
without the automatic coarsening geometry receipt. These observations concern
the inspected working tree, not an isolated reproduction of the screenshot.

Relevant source anchors:

- `webgpu-sparse-cm12-resident.wgsl.ts`: `transferCandidateCellsWork`,
  `presentationHorizontalVolumeScale`, `cm12PresentationPreparePage`,
  `planEditedRegionResolution`, `closePlannedResolution`.
- `tests/sparse-cm12-paused-region-dawn.test.ts`: checks conserved mass, unchanged
  time, and a nonempty published field; does not bound interface displacement.
- [Static grid-imprint investigation](coarse-surface-grid-imprint-investigation-2026-09-07.md):
  an independently verified density field can still publish a distorted surface.
- [Analytic motion investigation](analytic-coarse-surface-motion.md): correcting
  static appearance alone does not establish transport or pressure correctness.

An average specifies an integral, not a shape. Even volume plus first spatial
moments is insufficient for arbitrary topology: one central slab and two
separated slabs can have identical volume and centroid. This is an information
limit; smoothing, more triangles, and global volume adjustment cannot remove it.

## What the literature establishes

**CM12:** [Mass-Conserving Eulerian Liquid Simulation](papers/massConservingLiquids.txt),
sections 3.1, 3.4–3.8 and 5, uses a regular staggered grid and a transported
surface-density field. Its conservative advection and local sharpening do not
provide adaptive geometric remapping. The paper explicitly allows visible
volume changes despite conserved mass. Excess density is legal intermediate
state, so converting its density directly into a bounded VOF fraction is not
a semantics-preserving change. Keep CM12 as a conservative baseline, not as
proof that the proposed coupling is correct.

**Ando–Batty 2020:** [author paper](https://cs.uwaterloo.ca/~c2batty/papers/Ando2020/Ando2020.pdf),
also [local text](papers/ando-batty-2020-octree-liquid.txt), supplies surface
adaptivity, propagation of fine-feature requirements before remeshing, and
mixed-grid pressure/interpolation techniques. Its limitations include C0
interpolation kinks and nonconservative semi-Lagrangian advection. It is useful
for adaptive geometry and coupling, not a complete conservation solution.

**Geometric VOF:** [Basilisk fractions implementation](https://basilisk.fr/src/fractions.h)
reconstructs a plane and intersects it with children during refinement.
Retaining a correct plane makes exact planar subdivision possible. Re-estimating
its normal from changed neighbours is a different operation and needs its own
accuracy proof. One plane cannot encode two disconnected interfaces in a cell.

**Moment of fluid:** [Dyadechko–Shashkov](https://cnls.lanl.gov/~shashkov/papers/main.pdf)
adds material centroids to volume constraints and fits a volume-preserving
plane that approximates those centroids. This reduces dependence on neighbour
samples and provides a useful geometric residual. It does not make a finite
set of moments a lossless representation of arbitrary geometry. Its local fit
and moment transport add cost that must be measured.

**Coupled interface fields:** [Basilisk CLSVOF](https://basilisk.fr/src/two-phase-clsvof.h)
advects a distance field and relaxes it toward a VOF reconstruction. This is a
precedent for separate geometric and volume roles. Its relaxation and
redistancing are not a proof of an unchanged rendered contour, and its two-phase
solver is not a drop-in replacement for our free-surface solver.

## Proposed state and authority

Use one physical material state with multiple discretizations, not two
independently authoritative fluid volumes.

| State | Role | Adaptation authority |
| --- | --- | --- |
| Material/interface leaves | Conserved liquid amount and retained subcell geometric information | Surface error, topology and transport requirements |
| Continuous interface field or patches | Surface position, normals and interface intersections, constrained by material data | Geometry tolerance; shared by physics and rendering |
| Pressure cells and native velocity faces | Incompressibility and resolved motion | Physical error, coupling representability and region requests |
| Presentation pages/mesh | Samples/triangles of the accepted interface | Display error and budget; no independent fluid evolution |

Bulk full/dry material can collapse to implicit or macro records. Interface
leaves remain coarse where a low-order patch describes a broad plane or gently
curved surface. Local refinement handles droplets, sheets, folds and contact
regions. This must not become an always-finest band around every interface.

The exact local state is the first design comparison: volume plus retained
plane/continuous field, versus volume plus spatial moments and reconstructed
patches. Do not allocate both indefinitely merely to avoid choosing. A small
level-set band can provide normals, but a renderer must not evolve geometry
independently from the material update.

At rest and away from solids, target a geometric occupancy `chi_G` satisfying

```
material volume in C = integral over C of chi_G(x) dx
physics-cell liquid volume = sum of intersecting material-leaf contributions
```

Inside cut cells the integral is over open space. Liquid mass is reference
density times liquid volume for the incompressible geometric model. Fractions
remain in [0,1]. Existing CM12 tracking-density excess cannot be silently
absorbed into this equation: retaining CM12 requires an explicit conservative
residual/coupling formulation and a demonstrated bound on its geometric
effect, or the geometric method replaces its scalar transport semantics.

This choice is a blocking design question before production integration.

## Four contracts the architecture must satisfy

### 1. Zero-time repartitioning

Let `G` be accepted interface geometry and `P` the pressure partition. For a
physics edit `P -> P'`, `G` remains unchanged. Compute new control-volume
integrals from retained material geometry. Both old and new pressure grids
must query the same surface intersections. Merely remapping density and then
inventing a new zero contour is not the contract.

If the material partition itself changes, refine by integrating existing
reconstruction; merge only after measuring candidate displacement, normals,
local moments and topology. If a candidate cannot satisfy the geometric
tolerance, preserve the necessary geometric leaves. Repeated edit cycles must
not accumulate unbounded error.

Initial analytic shapes may seed and independently score an experiment. After
initialization they are unavailable to the candidate remap. The same method
must operate after impact and breakup. Rendering the original sphere at t=0
would bypass the problem.

### 2. Conservative, geometrically consistent motion

An accepted velocity update must advance both material amount and interface
geometry consistently. Test local flux/transfer receipts as well as total
mass, material centroid and shape. A local geometric correction is still
artificial motion unless bounded and accounted for; a conservative global
correction can transport liquid between unrelated bodies.

Compare two transport branches in isolation:

- CM12 conservative transport on the material hierarchy, with an explicit
  geometry coupling that handles diffuse/excess density. Reject this branch
  if volume constraints repeatedly require large geometric displacement.
- Geometric transport of bounded material fractions, with compatible
  interface/moment transport. Declare the numerical model change. Derive
  the stability/CFL requirements before assuming CM12's large steps remain
  possible; include all required substeps in timing.

Neither branch is pre-approved. Preserving CM12's name is not a reason to
retain an incompatible interface model. Conversely, VOF conservation alone
does not establish momentum, pressure or large-step behaviour.

### 3. Compatible pressure and boundary geometry

Classification, free-surface distances, open subface areas and rendering must
derive from the same accepted geometry generation. Velocity extension and
transport support must carry detached or dilute material correctly. Refining
material leaves must not create fictitious pressure coupling between separate
liquid regions.

Independent geometry cannot make an arbitrarily coarse pressure cell valid.
Two disconnected wet portions, a narrow air gap, multiple surface crossings
or incompatible boundary constraints can require more pressure degrees of
freedom. Initially retain/refine pressure cells when their representation
cannot support these configurations; multi-component cut-cell pressure is a
separate substantial algorithm, not assumed available.

A requested region width therefore cannot override a mathematical validity
requirement. Establish and expose the distinction between requested physical
width and required interface/coupling resolution in the eventual product.
Do not silently destroy geometry to satisfy a region.

### 4. Sparse, transactional execution

Allocate work from active material/interface patches, changed pressure cells,
and bounded coupling support. A local edit must not rebuild or scan the entire
possible world, allocate a domain-wide fine volume, or read fluid fields to the
host. Measure occupied counts and actual work, not only accepted pressure cells.

Publish compatible material, geometry, pressure-map and presentation generation
references together. Handle reserve/validate/commit/retire, rejection,
superseding edits, newly wet pages and memory exhaustion. A failed transaction
retains the entire prior accepted state. Count old-plus-new peak storage.

## Validation ladder and decision gates

Freeze fixtures, source receipts and numerical budgets before comparing
candidates. Keep representation, temporal discretization and production
regressions separate. An all-fine solver is a reference discretization, not
analytic ground truth. A watertight mesh is not proof of a correct surface.

| Gate | Experiment | Required evidence before proceeding |
| --- | --- | --- |
| 0: information and algebra | Known planes under uniform/mixed subdivision; equal moments with different topology | Identify necessary retained state; local conservation and planar identity |
| 1: candidate geometry | Full 3D planes, quadratic patches, spheres/ellipsoids, detached droplets, folded/thin sheets, two bodies in one pressure cell, solid cuts | Shape convergence, independent volume/moment integrals, topology retention, normals, shared-face continuity; no access to initial analytic geometry after seeding |
| 2: zero-time adaptation | Partial/moving regions, each supported rung, seams on all axes/edges/corners, macro and clipped cells; 1/10/100 coarse-fine cycles | Same current-time geometry, local receipts, unaffected neighbours and cumulative-error bound; actual accepted changes asserted |
| 3: prescribed transport | Translation, rigid rotation, reversible divergence-free deformation, breakup-resolution stress; fixed and changing partitions | Conserved material, convergence of displacement/shape, no seam locking, no spurious components; include high-CFL cost |
| 4: coupled physics | Flat hydrostatics, free fall, standing wave, ball impact, thin falling sheet and moving solids | Shared pressure geometry, preserved motion, local volume consistency and finite stable evolution; geometry-only successes are insufficient |
| 5: sparse GPU feasibility | Broad pool, local ball/region edit, larger empty world, many interfaces; production publication | Bounded memory and locality, end-to-end time including transport/substeps/reconstruction/extraction, coherent atomic publication |
| 6: production cutover | Existing focused regressions and canonical Sparse CM12 matrix, then browser inspection | Existing requirements pass without relaxed lanes/ceilings; known baseline failures remain explicit blockers to a fully green claim |

For gate 1, build an independent CPU oracle and then use the same fixtures in
WGSL. A known normal is permitted only in a labelled analytic subtest; actual
normal estimation and curved reconstruction must be scored separately. Measure
both sides of every interface, not just the highest crossing in a column.
Rotated versions must pass, so height-field-only methods cannot qualify.

Initial proposed new accuracy targets, fixed before candidate captures:

- CPU float64 represented-plane remap: mass error <1e-10 unit-cube volumes
  and plane displacement <1e-9 unit-cube lengths (gate 0 only).
- Production zero-time physics-only edit: unchanged canonical geometry data;
  compare publication against an unchanged-state re-publication control and
  explicitly account for packing/meshing error. A topology edit must add no
  geometric approximation stage.
- Material coarsening: start the accuracy/cost sweep with world-space shape
  tolerance `0.02 * reference finest cell width` (1 mm for 5 cm cells) and
  normal tolerance 1 degree on smooth patches; sharp edges use feature-aware
  comparisons. These are proposed design targets, not existing passing claims.
  Measure cumulative drift against the pre-cycle surface, not only each parent.
- Topology: unchanged component count and no new bridges/holes for topology-only
  edits of fixtures whose components and gaps are explicitly resolved in the
  retained geometry. Do not infer this from net volume.
- Dynamic tests: retain existing analytic-motion and canonical budgets; add
  independent surface/phase-error convergence measurements. Do not tighten a
  physics claim around an inaccurate analytic approximation.

If accuracy cannot fit available surface resolution or storage, report that
constraint. Choose an explicit accuracy/cost point from measured curves before
gate 5 acceptance. No numeric speedup or memory ceiling for this unimplemented
architecture is established yet; settling those budgets is required to select
the final design, not something to defer until after production cutover.

## First experiment delivered

[`probe-interface-representation-contract.py`](../tools/probe-interface-representation-contract.py)
is a standard-library CPU oracle. It integrates a plane clipped to boxes by
inclusion-exclusion, checks the integrator against independent indicator
quadrature, remaps through three dyadic levels, and repeats 32 round trips on
uniform and mixed partitions. Each refined mixed cell recovers its plane offset
from its volume and retained normal; it does not fetch the initial shape.
The normal is supplied, not estimated. Empty/full cells retain phase only.

It also constructs a single central slab and two separated slabs with exactly
the same volume and first spatial moments. The two material sets have a
symmetric-difference volume of 0.5 in a unit cube. This demonstrates why a
single mean, or a mean and centroid, cannot promise arbitrary topology retention.

Run without a browser or GPU dependency:

```sh
python3 tools/probe-interface-representation-contract.py \
  --output artifacts/interface-representation/contract.json
```

The result records the probe's SHA256 and labels its scope. This is preliminary
gate-0 evidence only. It does not validate curved reconstruction, neighbour
normal estimation, a persistent sparse hierarchy, CM12 coupling, pressure,
momentum, GPU precision, or performance. The quadrature sanity-check error is
reported separately from the conservative remap error.

Observed on 2026-09-08, source SHA256
`3303c52590c02bd44d7bf25fd22e7f96db11ca9b4450c3c7d75a05211139cf60`:

| Measurement | Result |
| --- | ---: |
| Plane configurations / round trips per configuration | 80 / 32 |
| Maximum local or total mass error, unit-cube volume | 1.055e-15 |
| Maximum retained-plane displacement, unit-cube length | 2.244e-14 |
| Maximum integrated child-fraction L1 error from copying parent means | 0.5 |
| Independent 24-cubed indicator quadrature discrepancy | 0.020834 |
| Single/two-body shared volume and centroid | 0.25 / (0.5, 0.5, 0.5) |

The large quadrature discrepancy is a separate coarse indicator-integration
error; it is not the geometric remap's error or a production acceptance target.
The evidence supports carrying geometry through a remap and demonstrates the
information limit of moments. It does not select between the two proposed
geometry candidates. [Machine-readable receipt](../artifacts/interface-representation/contract.json).

## Next implementation milestone

Build a standalone 3D CPU material/interface prototype with two candidates:
geometric fractions plus a continuous interface field, and moment-enriched
patches. Use identical initialized material, pressure partitions, accuracy
targets and storage accounting. Start with static reconstruction and zero-time
edits, then prescribed translation; do not modify the production renderer to
make these experiments appear successful.

Deliver a comparison report with actual per-component surface errors, local
volume/moment errors, topology, retained leaves/bytes and remap work. Stop a
candidate that needs initial analytic shapes, a permanently fine band, or
unbounded correction. Select the representation only after it passes the
static/transport gates and has a credible sparse GPU cost model. Then implement
a vertical GPU slice through material transfer, shared pressure intersections,
and publication before migrating the full solver.

Production migration may retire current rung-specific reconstruction and
density-derived pressure geometry once the replacement covers their contracts.
Do not leave two competing surface authorities active indefinitely.

After substantial Sparse CM12, topology, boundary, editing or publication
changes, run `npm run test:dawn:sparse-cm12` unchanged, serially under the
repository WebGPU lease with Fluid browser simulation unloaded. This document
and CPU oracle make no such production changes, so they do not require or
claim a new Dawn result.
