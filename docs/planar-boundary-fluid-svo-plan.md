# Archived planar-fluid proposal and retained SVO embedded planes

Date: 2026-08-26

Status: **The planar-fluid path is retired. Sparse CM12 uses SolidWorld voxels
as its only authored static-boundary authority. Explicit embedded planes remain
a render/SVO geometry representation.**

Current decision (2026-08-27):

- Container shells, terrain and authored edits reach fluid physics only through
  the unified SolidWorld occupancy/aperture path.
- Sparse CM12 has no inferred tank-face compiler, face mask, coordinate-plane
  closure, or alternate generic-solid routing switch.
- Explicit finite embedded planes remain valid for SVO acceleration and authored
  thin geometry, but they do not override or replace fluid-solid voxel authority.
- The material below is retained as historical design context, not as an active
  implementation plan.

Primary target: static tank walls and other large, flat, featureless solid
surfaces that currently force finest-rung Sparse CM12 bricks and expensive SVO
primary traversal.

Implementation snapshot (2026-08-26):

- `lib/core/planar-boundary.ts` now defines and packs the shared oriented finite
  slab, including exact CPU and WGSL ray intersection and material/owner identity.
- The four-word SVO leaf ABI has cut over from redundant Morton words to
  `[node, voxelOffset, terminalKind, terminalIndex]`; every former Morton reader
  now obtains the authoritative address from the node record.
- Thin, static authored boxes with at least 8:1 aspect ratio are admitted as
  planar terminals only when no second geometry bound overlaps the leaf.
  Intersections, corners, moving proxies, and mixed leaves remain voxel residuals;
  planar residuals are forced to finest depth so the relaxed thickness rule
  can never leak into a voxel fallback.
- Terrain-free `SolidWorld` thin fill patches now compile into that same compact
  catalogue with exact realized, potentially anisotropic voxel thickness.
  Clear edits, non-planar fills, terrain pages, and overlapping patch regions
  remain conservative voxel residuals.
- Planar terminal thickness no longer sets environment voxel resolution. The
  in-plane feature does, while the exact primitive record retains thickness.
- Canonical, compact, wide, split/raster-primary, macro-HDDA, and secondary
  visibility leaf consumers resolve the terminal exactly and skip the brick DDA.
  Terminal indices now address a dedicated immutable 64-byte planar catalogue
  carried by the structural source; source-primitive ordering is no longer part
  of the render hit contract.
- A world containing macro terminals rejects in-place geometry rescaling or
  authored geometry edits that would invalidate its baked classification; those
  changes require a sparse-world rebuild. This is the fail-closed lifecycle for
  the immediate ABI cutover, not a legacy voxel fallback hidden behind it.
- The structural source publishes initial voxel/planar terminal counts. Focused
  CPU tests pass. A guarded Dawn test compiles the full canonical production
  fragment shader and executes packed-record ray parity on a tiny compute
  dispatch.
- The former inferred tank-face compiler and Sparse CM12 mask path have been
  removed. Static-solid resolution evidence follows SolidWorld transitions.
- The canonical water-box reservoir is explicitly 12x14x10 finest cells so the
  B8/P8 method contains one genuinely featureless wet corner page. Its accepted
  atlas is one coarse 4^3 page plus seven interface 8^3 pages; the rule that a
  B8 page containing a liquid-air interface stays fine is unchanged.
- The isolated Dawn acceptance retains that 4^3 page through two paper steps
  beside the floor, x-low, and z-low planar enclosure faces: zero topology
  faults, zero represented-volume drift, converged pressure, no iteration cap,
  materialized relative mass drift below 1e-5, and no validation errors.
- The ordinary browser UI shows `Eulerian · Sparse CM12`, `7 fine · 1 coarse`,
  eight resident bricks, and zero represented-volume drift after two steps.
- Stationary primary-G-buffer reuse has been retired universally. The product
  publishes no coherence key, exposes no tuning/UI switch, and the low-level
  decision always retraces.
- With real primary traversal active, paused `Sparse CM12 · symmetric expansion`
  in the performance profile sustained 58.0–60.2 FPS over ten browser samples.
  Primary traversal was normally 2.0–2.7 ms and the live structural census
  reported 59 planar terminals versus 191 voxel terminals. The smooth
  spotlight-lit floor remains intact.
- Still pending beyond this accepted E0 scope: omit redundant planar voxel
  payload; patch-owned liquid-contact mesh closure; E1/E2/I1/I2 oblique and
  multi-fragment fluid topology; derived opacity/radiance macro coverage; and
  the conservative beam prepass.

---

## 1. Decision

Introduce a shared, compiled boundary-patch representation whose first geometry
kind is an exact finite plane or slab. One canonical patch supplies:

- the solid boundary seen by Sparse CM12;
- the liquid-contact condition and exact planar closure seen by free-surface
  reconstruction;
- the fine physical/render thickness of the wall;
- the SVO macro-surface records used for primary and secondary visibility;
- stable material, boundary-condition, generation, and provenance identities.

Planar geometry becomes a coarse-node property rather than a reason to publish
every intersected finest voxel. Fine resolution remains mandatory around patch
edges, corners, holes, unsupported overlaps, free-surface detail, and genuine
flow activity.

The fluid and renderer share semantic geometry, not necessarily one runtime
acceleration structure:

```text
authored solid geometry
        |
        v
canonical BoundaryPatch catalogue
        |
        +--> Sparse CM12 boundary-face / cut-fragment compilation
        |
        +--> liquid free-surface contact / exact planar closure compilation
        |
        +--> SVO macro-node / planar-slab compilation
        |
        +--> residual fine SolidWorld/SVO pages at unresolved features
```

The representation is fail-closed. If a node or solver cell cannot prove that a
supported planar patch completely describes its relevant boundary geometry, it
uses the existing fine-voxel path. No tolerance failure, capacity failure, or
unsupported topology may silently turn solid into fluid or omit visible
geometry.

## 2. Why this project exists

The current static-boundary resolution rule explicitly makes every brick that
touches a change in `SolidWorld` finest-rung:

- `brickRequiresStaticSolidBoundaryResolution` in
  `lib/methods/adaptive-mass/sparse-brick-atlas.ts` scans finest solid samples;
- any partial solid fraction or any six-neighbour fraction change returns
  `true`;
- `createInitialSparseAdaptiveMassAtlas` promotes that brick to
  `brickFineResolution`;
- strong grading then propagates part of that cost into adjacent liquid.

The behavior is correct for the existing representation. A coarse repeated
sample cannot reproduce a fine occupied wall, so retaining the wall requires
retaining the fine brick. It is also the source of the performance loss: an
otherwise calm tank inherits the wall's voxel resolution as its simulation
resolution.

The SVO has the corresponding representation problem. A one-fine-voxel-thick
surface of area `A` contains approximately `A / h^2` occupied fine voxels. A ray
that sees the surface still descends to and traverses those leaves. The retired
static-primary cache avoided recomputing a settled camera, but hid the actual
traversal cost and never helped a moving camera.
`docs/svo-primary-visibility-handoff.md` records that primary
visibility accounted for about 46% of the measured SVO frame in the dam scene
and identifies a conservative beam/depth prepass as the next spatial step.

The desired change is not "ignore solids when selecting resolution." It is:

> Preserve solid geometry and fluid-domain topology without requiring the
> solver and renderer to express a flat wall as fine occupancy voxels.

There is also a fidelity regression to remove. Before signed sparse-world
addressing, the water extractor treated a tank wall as an exact domain face:
it emitted a planar wetted closure and mirrored the interior liquid scalar for
the free-surface contour. The signed-sparse classifier instead takes the
ordinary eight-sample contour path and returns before the exact wall-plane,
wall-mirroring, and sharp box-feature paths. Missing or dry samples beyond the
wall therefore behave like exterior air. Marching tetrahedra interpolates
towards them, which moves the visible contact surface inward and rounds or
chamfers wall contacts and corners by an amount that depends on sample span.

This is primarily a reconstruction-semantic difference, not evidence that a
fully occupied, axis-aligned SolidWorld wall has become permeable in pressure.
The planar-boundary project must nevertheless fix both consumers from one
patch. It is not complete if the pressure face is exact but the visible liquid
still closes against an implicit dry voxel.

## 3. Literature basis

This plan is a synthesis of established techniques rather than a claim of a
single existing unified method.

### 3.1 Fluid boundary treatment

- Chentanez and Muller already use non-solid cell volumes and open face-area
  fractions for non-axis-aligned and moving boundaries in the CM12 method. The
  current repository's solid fraction, SDF, face preparation, and projection
  authorities are descendants of that model. See
  [Mass-Conserving Eulerian Liquid Simulation](https://matthias-research.github.io/pages/publications/masscon_sca.pdf).
- Batty, Bertails, and Bridson show that embedded sub-grid solid geometry can be
  incorporated into a variational pressure projection on coarse Cartesian
  grids while retaining a symmetric positive-semidefinite system. See
  [A Fast Variational Framework for Accurate Solid-Fluid Coupling](https://uwspace.uwaterloo.ca/items/f5cb877f-1b56-4ff5-81d9-f7473a5e7f1a).
- Azevedo, Batty, and Oliveira provide the direct topology reference: clipping
  a coarse background cell by thin solid geometry may create multiple distinct
  fluid sub-cells, each represented in a graph-based pressure projection. This
  prevents opposite sides of a sub-cell-thin wall from coupling. See
  [Preserving Geometry and Topology for Fluid Flows with Thin Obstacles and Narrow Gaps](https://uwspace.uwaterloo.ca/bitstreams/9459a480-4969-40c2-9ef2-11a7244ee7e6/download).
- Ando and Batty provide the closest modern octree liquid operator foundation,
  including free-surface and solid boundary conditions across adaptive level
  transitions. See
  [A Practical Octree Liquid Simulator with Adaptive Surface Resolution](https://doi.org/10.1145/3386569.3392460).

### 3.2 SVO representation and traversal

- Laine and Karras attach an oriented contour slab, represented by two parallel
  planes, to voxels. Their contour permits tighter surface placement and more
  aggressive hierarchy pruning. They also describe a conservative 4x4/8x8
  primary beam optimization that skips a shared empty prefix. See
  [Efficient Sparse Voxel Octrees](https://research.nvidia.com/publication/2010-02_efficient-sparse-voxel-octrees-analysis-extensions-and-implementation).
- Sparse voxel DAGs share identical occupancy subtrees and can dramatically
  reduce storage for repetitive geometry. They are complementary to this plan:
  a DAG saves repeated structure, while a macro planar node also shortens ray
  traversal. See
  [High Resolution Sparse Voxel DAGs](https://doi.org/10.1145/2461912.2462024).
- Adaptively sampled distance fields establish the broader precedent that a
  spatial hierarchy can stop refining where a coarse analytic/interpolated
  field already represents the surface within an error bound. See
  [Adaptively Sampled Distance Fields](https://graphics.stanford.edu/courses/cs468-03-fall/Papers/frisken00adaptively.pdf).

## 4. Outcome

When complete, the system will have these properties:

1. A large flat tank wall can remain one-fine-voxel thick for rendering and
   collision semantics while calm adjacent fluid uses the coarsest otherwise
   admissible Sparse CM12 rung.
2. An oblique planar vessel wall can cut coarse fluid cells without being
   voxel-inflated or forcing finest-rung simulation.
3. An internal thin wall may divide one coarse background cell into multiple
   disconnected fluid fragments without pressure, interpolation, or advection
   leaking between them.
4. SVO primary rays may hit a coarse planar macro record analytically and stop
   without descending through the fine wall occupancy.
5. Coherent primary tiles may conservatively reuse a near distance or, for a
   fully certified plane tile, resolve visibility without independent hierarchy
   traversal for every pixel.
6. Unsupported geometry follows the current fine path with no loss of fidelity.
7. Fluid and render consumers can prove that they compiled the same patch
   generation and canonical coefficients.
8. At an admitted planar wall, the visible liquid closure lies on the exact
   authored plane at every Sparse CM12 rung; it is never displaced by half a
   contour span or rounded by a missing exterior sample.
9. The first delivery reproduces the previous orthogonal/neutral tank-wall
   contact convention. Slip, surface tension, normal filtering, and future
   wetting/contact-angle controls remain separately attributable.

## 5. Non-goals for the first production delivery

- Do not make all solid-adjacent fluid coarse. Flow and free-surface evidence
  remain independent refinement causes.
- Do not add no-slip wall functions or claim resolved viscous boundary layers.
  The first target is the existing inviscid/free-slip CM12 boundary behavior.
- Do not add a general capillary wetting model in the first delivery. Preserve
  the previous neutral, orthogonal liquid-contact reconstruction exactly;
  authored contact angles are a later, explicit physical extension.
- Do not support moving planar patches initially. Moving cut cells add swept
  volume, newly uncovered states, geometric conservation, and temporal
  topology changes.
- Do not automatically fit arbitrary noisy voxel surfaces in the first phase.
  Authored box-vessel planes provide an exact and valuable initial source.
- Do not support intersecting arbitrary patch arrangements inside one coarse
  cell initially. Unsupported multiplicity forces local refinement.
- Do not change CM12 convergence tolerances according to wall planarity.
- Do not weaken the current accepted-generation, exact dependency, or
  fail-closed publication rules.
- Do not reintroduce source-primitive shading dependencies into the static dry
  renderer. A macro surface is accepted SVO geometry, with material and normal
  data sufficient for shading.
- Do not claim voxel-exact reproduction of an arbitrarily oriented staircase
  with one analytic plane. Axis-aligned authored walls can be voxel-exact;
  oblique patches are source-geometry-exact with an explicitly bounded
  difference from their former voxelization.

## 6. Terminology

**Boundary patch**
: A canonical, finite piece of solid boundary with geometry, sidedness,
  thickness, material, physical boundary conditions, and provenance.

**Plane patch**
: A zero-thickness impermeable sheet, finite in two in-plane dimensions.

**Slab patch**
: The solid region between two parallel planes, clipped to a finite in-plane
  extent. Its thickness is independent of the solver cell size and SVO node
  size.

**Macro node**
: An SVO node that terminates traversal using a boundary macro record instead
  of descending to ordinary 8x8x8 voxel payload leaves.

**Residual geometry**
: Fine `SolidWorld` and SVO occupancy retained around patch boundaries,
  corners, holes, overlaps, non-planar regions, or failed certificates.

**Background cell**
: A normal Sparse CM12 adaptive cell before clipping by embedded boundary
  geometry.

**Fluid fragment**
: One connected portion of fluid volume remaining inside a background cell
  after clipping. One background cell may own more than one fragment.

**Open fragment face**
: A connected face polygon through which two fluid fragments exchange flux.

**Solid fragment face**
: A clipped patch polygon that imposes the wall-normal boundary velocity and
  has no pressure neighbour across it.

**Liquid-contact condition**
: The boundary rule used to extend the liquid scalar at a solid face and to
  determine how the free surface meets that face. It is independent of
  tangential velocity slip and of the surface-tension coefficient.

**Planar wetted closure**
: The portion of an exact boundary plane covered by liquid, clipped in the two
  tangential directions by interior liquid samples. It closes the rendered
  liquid mesh without treating solid or missing samples as exterior air.

## 7. Architectural boundary

The new semantic owner should live with `SolidWorld`, not under Sparse CM12 or
the renderer. Proposed public construction shape:

```ts
export interface SolidBoundaryWorld {
  readonly version: number;
  readonly patches: readonly SolidBoundaryPatch[];
  readonly residual: SolidWorld;
  readonly contentStamp: string;
}

export type SolidBoundaryPatch = PlanarBoundaryPatch;

export interface PlanarBoundaryPatch {
  readonly kind: "plane" | "slab";
  readonly patchId: number;
  readonly generation: number;
  readonly normal: readonly [number, number, number];
  /** n dot x = frontOffset_m. */
  readonly frontOffset_m: number;
  /** Equal to frontOffset_m for a plane. */
  readonly backOffset_m: number;
  readonly tangentU: readonly [number, number, number];
  readonly tangentV: readonly [number, number, number];
  readonly minimumUV_m: readonly [number, number];
  readonly maximumUV_m: readonly [number, number];
  readonly fluidSidedness: "negative" | "positive" | "both";
  readonly materialId: number;
  readonly boundaryCondition: "static-free-slip";
  /** Phase one preserves the former mirrored-scalar, 90-degree convention. */
  readonly liquidContactCondition: "legacy-orthogonal";
  readonly sourceRevision: number;
  readonly certificateHash: number;
}
```

The exact packed ABI will be designed after the CPU oracle is accepted. The
semantic requirements are fixed:

- normals are unit length within an explicit construction tolerance;
- offsets use metres in the same world frame as SVO primitives and the physical
  lattice header;
- `frontOffset_m <= backOffset_m` under one documented normal convention;
- tangent axes form a right-handed orthonormal frame with the normal;
- finite UV bounds are ordered and non-empty;
- patch IDs are stable within a `contentStamp`;
- geometry, material, and boundary-condition changes advance generation;
- liquid-contact-condition changes advance generation and are covered by the
  same certificate as the plane coefficients;
- the certificate covers every bit consumed by fluid and SVO compilers.

`SolidWorld` remains the sparse voxel fallback and edit representation. The
first implementation does not delete its pages. It changes their role:

```text
before: every solid is expressed through finest voxel pages
after:  supported planar interiors are patches;
        everything else is residual voxel geometry
```

This keeps authored voxel edits, terrain, arbitrary vessels, and partial
migration valid.

## 8. Authoring and compilation

### 8.1 Phase-one authoring source

`boxSolidVoxelShell` already authors the six walls of a rectangular vessel as
one-voxel-thick axis-aligned boxes. Extend vessel construction so it publishes
the corresponding exact patches at the same time:

- floor;
- negative/positive X walls;
- negative/positive Z walls;
- optional lid.

The existing voxel patches remain available as an oracle and fallback during
migration. They must not independently define a slightly different wall.
Create both products from one canonical vessel specification.

The same vessel specification assigns `legacy-orthogonal` liquid contact to
the initial patches. This is the former tank behavior: the liquid scalar is
mirrored/extended normally across the boundary for free-surface extraction,
and the wetted closure is emitted on the exact plane. It is not inferred from
whether the wall happens to occupy a neighbouring voxel.

For the common open box tank, the liquid-facing plane lies exactly on the
interior domain face. The fluid does not need an irregular cut volume at all;
it only needs a closed boundary face at any selected rung. This is deliberately
the first delivery because it removes the wall-induced finest-rung floor
without introducing multi-fragment pressure rows.

### 8.2 Oblique authored patches

The next authoring source accepts an oriented rectangle or slab with explicit
world-space thickness. It is canonical source geometry, not a fit to the
voxelized staircase.

The residual voxelization remains available for comparison. Admission records
the maximum Hausdorff disagreement introduced by replacing the old digital
staircase. The target is at most half of the former finest voxel's body
diagonal unless the author explicitly supplies a tighter bound.

### 8.3 Automatic extraction, later

Automatic extraction from static fine solids is a later optimization. It may
use region growing and plane fitting, but a fitted patch is accepted only when
all of the following hold:

1. maximum signed-distance residual is within the configured geometry error;
2. the normal cone is within the configured angular error;
3. the inferred thickness interval is non-empty and within the thickness error;
4. material and physical boundary condition are uniform;
5. the patch's finite coverage and holes are exactly represented;
6. the patch does not merge disconnected surfaces;
7. every source solid sample is accounted for by either the patch or residual
   geometry;
8. reconstructing the patch plus residual passes a source-coverage oracle.

Automatic extraction is construction work. It does not run every frame for a
static world.

## 9. Planarity and macro-node admission certificate

Every SVO node and fluid background cell decides independently whether a patch
may replace fine occupancy there. The shared certificate inputs are:

- canonical patch ID and generation;
- patch plane/slab coefficients;
- node or cell world-space bounds;
- finite UV coverage;
- set of residual geometry overlapping the bounds;
- material and boundary-condition identity;
- supported patch multiplicity and topology class;
- requested geometric error and former finest spacing.

A node or cell is an **interior planar admission** only if:

1. exactly one supported patch is relevant in the first implementation;
2. the intersection of the infinite plane/slab with the box lies wholly inside
   the patch UV extent with a conservative numerical margin;
3. no patch edge, corner, hole, or residual solid enters the box;
4. the patch divides the box into one of the supported topology signatures;
5. the material and boundary condition are uniform over the intersection;
6. recomputed coefficient and bounds hashes match the catalogue certificate;
7. all quantization error stays within the declared bound.

Otherwise the box is a **feature admission** and descends/refines until the
existing residual representation is sufficient.

The first topology signatures are:

```text
E0  axis-aligned external wall coincident with a cell face
E1  one-sided plane cutting one fluid fragment from a cell
E2  one-sided slab cutting one fluid fragment from a cell
I1  two-sided plane producing two fluid fragments
I2  two-sided slab producing up to two exterior fluid fragments
```

`E0` lands first. `E1/E2` require cut volumes. `I1/I2` require multi-fragment
state and pressure connectivity.

## 10. Resolution policy change

Replace the geometric meaning of
`brickRequiresStaticSolidBoundaryResolution` with a narrower question:

```text
Does this brick contain solid boundary geometry that cannot be represented by
an accepted boundary patch at the candidate rung?
```

Proposed decomposition:

```ts
interface StaticBoundaryResolutionEvidence {
  readonly requiresFineResidual: boolean;
  readonly reason:
    | "none"
    | "patch-edge"
    | "patch-corner"
    | "patch-hole"
    | "unsupported-overlap"
    | "unsupported-topology"
    | "certificate-failure"
    | "residual-solid"
    | "open-exterior";
  readonly patchId?: number;
  readonly admittedTopology?: "E0" | "E1" | "E2" | "I1" | "I2";
}
```

Resolution selection order becomes:

1. hard authored minimum/maximum resolution bounds;
2. unresolved solid geometry and patch-feature refinement;
3. free-surface/interface refinement;
4. flow/activity resolution request;
5. predictive transport/forcing support;
6. strict grading closure.

A planar patch interior contributes boundary metadata but no finest-rung
request. A free surface touching that wall may still request fine cells. A
corner or patch end retains a configurable fine feature ring.

The overlay must show why a wall-adjacent brick is fine. "Planar wall" is not a
valid fine reason after the cutover; "contact line," "patch edge," "vorticity,"
or another explicit cause is.

## 11. Fluid discretization

### 11.1 Delivery A: axis-aligned exterior tank walls

For `E0`, retain ordinary background cells and pressure rows. Compile the patch
to closed face apertures at every rung:

- open area is zero across the tank wall;
- wall normal velocity is zero for a static wall;
- the fluid cell has full volume;
- there is no pressure neighbour outside the vessel;
- transport tracing clips at the exact plane;
- velocity extension and interpolation cannot read through the closed face.

This path should be expressible largely through the existing boundary and
solid-open inputs consumed by face preparation and projection. The material
thickness is irrelevant to the fluid interior as long as the wall is
impermeable and the exterior is not simulated.

The exit condition is that removing boundary-driven fine promotion changes no
fluid equation for the aligned tank; it changes only which rung owns that
equation.

#### 11.1.1 Exact liquid contact and mesh closure

`E0` also compiles an exact free-surface boundary record. This is a separate
consumer of the same accepted patch, not a renderer-side inference that an
outer lattice coordinate must be a tank wall.

For `legacy-orthogonal` contact:

- sample the live liquid scalar only on the fluid side of the patch;
- mirror or constant-extend that scalar in the patch-normal direction when
  classifying free-surface cubes;
- emit one analytic planar wetted-closure owner on the boundary plane;
- clip its tangential polygon from the interior scalar without inventing an
  all-wet quad;
- never interpolate the liquid surface from an interior sample to a missing,
  dry, or solid sample across the wall;
- at two- and three-plane corners, give each plane an independent Cartesian
  face owner with deterministic seam ownership rather than allowing the fixed
  marching-tetrahedra diagonal to chamfer the corner;
- keep the plane-normal coordinate exact under rung/sample-span changes; and
- derive geometric normals for the planar closure from the patch. Optional
  filtered normals may smooth free-surface shading but must not move vertices,
  alter closure ownership, or blend a wall plane into an apparent geometric
  fillet.

The classifier must look up accepted patch coverage and generation. Signed
sparse addressing is not, by itself, permission to bypass exact boundary
handling. A cube without a valid planar-contact certificate retains the
ordinary implicit contour path.

This phase should reuse the intent of the current exact-wall emission helpers
in `webgpu-water-global-fine-classify.ts`, but replace their bounded-domain
coordinate assumption with patch identity and coverage. Restoring a global
"x/z domain edges are walls" special case would be incorrect once sparse pages
can legitimately exist outside the original box.

The physical and presentation A/B must be separable:

- compare density, gamma, velocity, pressure, open fractions, and topology
  with water-mesh generation disabled or ignored;
- compare water vertex positions and indices with flat/geometric normals;
- compare shaded images last, with filtered normals explicitly off and on.

If only highlights change, the cause is shading. If the contact silhouette or
plane-normal vertex coordinate changes while the fluid fields agree, the cause
is extraction. If the accepted fluid fields change, investigate boundary
aperture, trace clipping, or conditioning independently.

### 11.2 Delivery B: one-sided oblique vessel walls

For `E1/E2`, clip one background cell by the patch and retain the fluid-side
polyhedron. Publish:

- fluid volume `V_c` and volume fraction;
- fluid centroid;
- each open face polygon's area and centroid;
- solid face area, centroid, and normal;
- neighbour background cell and fragment identity;
- canonical distance used by the pressure gradient;
- topology class and patch provenance.

The discrete divergence for fragment `c` is:

```text
div(c) = (
  sum over open faces f: A_f * u_f
  + sum over solid faces s: A_s * dot(u_wall_s, n_s)
) / V_c
```

For a static wall the solid term is zero. The pressure graph edge between two
fluid fragments has weight proportional to:

```text
w_f = dt * A_f / (rho * l_f)
```

where `l_f` is the accepted pressure-sample separation for that face. Both
incident rows must use the same `A_f`, `l_f`, and weight bits with opposite
incidence signs. The diagonal is the sum of accepted incident weights plus
valid free-surface terms.

CM12 density capacity becomes the fragment fluid volume rather than the whole
background-cell volume. The normalized phase value used for pressure
membership is `rho_c / V_c`, following CM12's existing partial-solid logic.

### 11.3 Delivery C: internal thin barriers and multiple fragments

For `I1/I2`, one background cell may contain disconnected fluid on both sides
of a wall. One cell-centred density/pressure value is invalid because it would
couple those sides.

Introduce a fragment indirection:

```ts
interface SparseCM12BackgroundCellRecord {
  readonly fragmentOffset: number;
  readonly fragmentCount: number;
  readonly topologySignature: number;
  readonly geometryGeneration: number;
}

interface SparseCM12FluidFragmentRecord {
  readonly backgroundCell: number;
  readonly localFragment: number;
  readonly volume_m3: number;
  readonly centroid_m: readonly [number, number, number];
  readonly pressureRow: number;
  readonly scalarState: number;
  readonly faceOffset: number;
  readonly faceCount: number;
}
```

Each connected fragment owns its own:

- surface density and gamma state;
- pressure row and warm start;
- divergence target;
- collocated velocity reconstruction inputs;
- phase/membership classification;
- transport receivers and donors.

There is no graph edge across a solid fragment face. Two fragments in the same
background cell are no more connected than fragments in distant cells.

The first multi-fragment implementation supports at most two fragments and one
plane/slab per background cell. Greater multiplicity or ambiguous clipping
forces refinement until every accepted cell meets the bound.

### 11.4 Small cut fragments

Arbitrarily small cut volumes can damage conditioning and, for explicit flux
methods, the stable time step. CM12's conservative semi-Lagrangian transport
avoids some explicit CFL restrictions, but tiny pressure volumes and poor
interpolation geometry remain problematic.

Use deterministic agglomeration when `V_c / V_background < minimumFragmentFraction`:

- merge only with a face-connected fragment on the same side of every patch;
- never merge across a solid face;
- preserve total volume and open-face area;
- publish an exact source-to-aggregate receipt;
- use the aggregate as one pressure/scalar state owner;
- retain the original geometry for collision and visualization.

If no legal same-side merge exists, refine. Do not clamp the volume upward or
delete the fragment silently.

## 12. CM12 transport and scalar conditioning

Pressure projection alone is insufficient. Every operation that currently
assumes one Cartesian cell or samples through a `SolidWorld` fraction must be
made patch/fragment aware.

### 12.1 Characteristic tracing

- Intersect each trace segment against analytic patch planes/slabs.
- Stop or clip at the first impermeable crossing.
- Resolve the source fragment containing the clipped backtrace point.
- Reject a donor that is geometrically close but disconnected by a wall.
- Include patch generation and fragment topology in exact reuse dependencies.

The existing seams in
`sparse-cm12-transport-home-frame-halo.wgsl.ts`—notably
`clampInsideEmbeddedBoundary` and `clipBoundarySegment`—are the first consumer
audit points.

### 12.2 Conservative transport

- Receiver capacity is fragment volume.
- Donor/receiver weights are valid only across open fragment connectivity.
- Aggregate weights remain locally and globally mass conserving.
- Scatter/gather normalization cannot send excess density through a wall.
- Gamma and sharpening neighbourhoods exclude disconnected fragments.

### 12.3 Velocity extension and interpolation

- Extension must use the fragment graph, not Euclidean adjacency alone.
- Reconstruction near a clipped polyhedron uses accepted open/solid face
  normals and areas.
- A one-sided interpolation stencil must not include a velocity sample behind
  the wall.
- For the first `E0` path, existing closed-face topology should make this a
  much smaller change; `I1/I2` require explicit fragment visibility.

### 12.4 Free-surface conditioning at a patch

- Surface sharpening and gamma diffusion use the same-side fragment
  neighbourhood and the accepted liquid-contact condition.
- `legacy-orthogonal` extends the scalar normally without adding or deleting
  liquid capacity; it is a boundary stencil, not a mass source.
- Surface tension remains an independent force coefficient. A zero coefficient
  must still produce the exact planar closure, and a nonzero coefficient must
  not silently select a wetting angle.
- `free-slip`/`no-slip` remains an independent velocity condition. Switching
  slip mode must not change the static extracted contact geometry.
- A future authored contact angle must change the canonical contact condition,
  patch generation, scalar extension, and reconstruction certificate together.
  It must not be implemented as a shading-normal adjustment.

### 12.5 Topology transition

When the adaptive rung changes under a static patch:

- rebuild only changed background cells, fragment faces, incident rows, and
  pressure ancestors;
- transfer scalar mass by exact overlap of source and candidate fragments;
- transfer normal flux by shared open-face area;
- seed pressure by volume-weighted same-component restriction/prolongation;
- reject candidate publication if component mapping is incomplete;
- include boundary-patch generation in the topology transaction provenance.

This work belongs inside the sparse-world topology transaction described by
`docs/sparse-world-module-architecture.md`, not in renderer-driven or
consumer-specific mutation.

## 13. Pressure-authority integration

The repository's pressure and face work is built around stable row IDs,
interned boundary operators, exact dependency generations, and local repair.
Planar boundaries must extend those authorities rather than bypass them.

Required changes by delivery:

### E0 authority

- Compile a rung-independent patch-face closure into the accepted boundary
  image.
- Treat patch generation as an embedded-boundary dependency for FPA
  preparation and projection.
- Preserve the existing pressure endpoint shape; the outside endpoint is
  absent/solid.
- Mark only rows incident to changed topology when an adaptive brick rerungs.

### E1/E2 authority

- Make pressure-row geometry reference the compiled fragment, not only the
  background cell.
- Intern common planar cut signatures where coefficients and orientation permit
  reuse, while retaining exact per-cell area/volume data where they differ.
- Extend PCM/PCF membership and effective-edge repair to fragment rows.
- Preserve reciprocal off-diagonal bits and accepted generation across both
  incident fragments.

### I1/I2 authority

- Stable row identity becomes `(backgroundCellStableId, localFragmentId,
  geometryGeneration)` or an equivalent canonical rank.
- A fragment reorder is a topology change even if the background cell remains.
- Incidence maps, face authority, pressure cache, projection, collocation, and
  diagnostics consume the same fragment row catalogue.
- No compatibility path may fold two fragment rows back into one cell row.

The exact certificate in `docs/sparse-cm12-face-projection-authority.md`
already names embedded boundary and moving-solid words as preparation
dependencies. This project replaces those words' geometric expressiveness; it
does not relax exact reuse.

## 14. SVO representation

The dry/static SVO macro surface and the dynamic water mesh are distinct
compiled consumers. A correct planar dry wall does not by itself fix liquid
contact reconstruction. Both must accept the same patch generation, but the
water extractor additionally consumes the live liquid scalar and the patch's
liquid-contact condition.

### 14.1 Preserve the voxels-only publication contract

The static dry primary currently shades accepted voxel geometry and deliberately
does not consult the source primitive after voxelization. A planar macro record
must be part of the accepted SVO publication:

- traversal obtains its geometry from SVO node/macro storage;
- the hit carries normal, feature/material identity, and exact distance;
- deferred shading reads the existing material publication;
- no per-pixel lookup back into `SvoPrimitiveDescriptor` is introduced.

The canonical boundary patch may also be expressible as an authoring primitive,
but that is not the runtime acceleration. Adding a `plane` descriptor while
still voxelizing it to ordinary fine leaves would not solve the problem.

### 14.2 Macro record

Proposed semantic record:

```ts
interface SvoPlanarMacroRecord {
  readonly patchId: number;
  readonly patchGeneration: number;
  readonly normal: readonly [number, number, number];
  readonly frontOffset_m: number;
  readonly backOffset_m: number;
  readonly materialId: number;
  readonly featureId: number;
  readonly nodeMinimum_m: readonly [number, number, number];
  readonly nodeMaximum_m: readonly [number, number, number];
  readonly certificateHash: number;
}
```

The node AABB clips the infinite slab. Patch UV bounds do not need to be loaded
per ray for an interior admission because the certificate has proved the node's
complete plane/slab intersection lies inside the patch. Nodes containing a
finite patch edge descend to residual geometry.

The packed record should use f32 plane coefficients initially. Compact normal
and offset encodings are a later measured memory optimization; quantization
must not precede correctness.

### 14.3 Node topology

The octree requires a terminal macro-node kind in addition to ordinary empty,
interior, and brick-leaf states. Traversal behavior:

1. intersect the ray with the node AABB as today;
2. load the planar macro record;
3. intersect against the two slab planes;
4. clip the result to the node interval;
5. return the nearest valid surface hit with the accepted normal/material;
6. do not push children.

A plane record uses equal offsets and a sided intersection rule. A slab record
selects front or back normal according to the entered surface. Thickness is
therefore exact even when it is much smaller than the node.

Mixed macro-plus-residual nodes are out of scope initially. If any competing
geometry enters the node, it descends. A later mixed node may test a small
primitive list plus child occupancy, but it is not needed to prove the core
win on large wall interiors.

### 14.4 Construction

Construct macro nodes top-down or bottom-up after ordinary conservative node
coverage is known:

- begin with the coarsest node intersecting a patch;
- admit it if the macro certificate passes and no residual/competitor overlaps;
- otherwise visit its children;
- ordinary fine voxelization remains beneath failed nodes;
- publish counts and first rejection reason by patch.

Node construction must retain the existing conservative coverage guarantee:
every source wall point is represented by exactly one accepted macro region or
ordinary residual geometry.

## 15. SVO traversal and coherent primary visibility

### 15.1 Macro-hit path

The first render win is per-ray traversal shortening. A ray hitting a macro
wall stops at the admitted node depth rather than descending to the former fine
brick and walking its voxel payload.

Add counters for:

- macro nodes visited;
- macro hits and misses;
- ordinary node visits avoided;
- former fine-equivalent voxels covered;
- macro rejection reasons;
- primary pixels terminating on macro geometry.

The leaf-visit histogram experiment already present in
`webgpu-svo-dry-scene.ts` is the primary comparison instrument.

### 15.2 Conservative beam/depth prepass

Implement the general Laine-Karras-style prepass after macro hits are correct.
For each `N x N` screen tile:

1. trace conservative corner/extent rays against the accepted SVO;
2. compute a conservative earliest possible hit distance for the tile;
3. store that `tMin` in a small texture/buffer;
4. start each full-resolution primary cursor at `tMin` rather than root entry.

The integration seam already identified in
`docs/svo-primary-visibility-handoff.md` is `traceStatic` and
`svoTraversalContinuationBegin`, where `ray.tMin` is already respected by
continuation traversal.

Correctness requirements:

- the stored value may be too early but never too late;
- a small occluder inside the tile cannot be skipped because corner rays missed
  it;
- reverse-Z depth behavior remains unchanged;
- a disabled or invalid prepass uses `tMin = 0` and reproduces the control;
- camera motion does not invalidate the result because it is rebuilt for the
  same frame.

Start with 4x4 and 8x8 arms. Select only after measured primary time and
divergence on the target GPU.

### 15.3 Plane-certified tile resolution

The stronger planar special case is a later phase. A tile may resolve primary
visibility from one patch when it proves:

1. the tile frustum intersects one planar macro patch over its full footprint;
2. no competing SVO node can contain a nearer hit anywhere in the tile;
3. patch coverage contains the full projected tile;
4. front/back orientation and material identity are homogeneous;
5. generated depth stays within the node and camera intervals.

Then depth and normal may be generated analytically for the tile. Shading may
still execute per pixel because lighting, texture, output, and water
composition vary. This optimization targets traversal only.

The existing raster-primary infrastructure may be reusable, but the source
must be the accepted SVO macro publication rather than the pre-voxelization
scene primitive catalogue.

## 16. Shared generation and provenance

Fluid, water-extraction, and SVO products need not share one GPU buffer, but
they must prove that they derive from the same canonical patch bits.

Publish per consumer:

```text
boundaryContentStamp
patchCatalogueGeneration
patchCount
catalogueCertificateHash
consumerCompileGeneration
consumerCoverageHash
firstRejectedPatch / reason
```

At world creation or static-boundary update:

- canonicalize once;
- pack one immutable patch catalogue;
- compile the fluid boundary image, water-contact image, and SVO macro image;
- validate each against the canonical certificate;
- accept presentation/simulation only when their requested generation is
  complete.

Dry rendering may continue from a previously accepted SVO generation while a
new static world compiles, water extraction may retain its previous accepted
contact generation, and simulation may retain its previous accepted boundary
generation. None may mix new patch coefficients with old residual geometry or
liquid-contact policy.

## 17. Implementation phases

### Phase 0 — Freeze wall-driven baselines

Purpose: measure the exact cost and fidelity before changing representation.

Work:

- Add a minimal open rectangular tank containing quiescent full-depth liquid.
- Add the same tank without solid-wall resolution promotion as a diagnostic-only
  topology oracle; do not run physics with the unsafe topology.
- Add a large axis-aligned dry plane and one oblique dry slab render scene.
- Capture the same partially filled aligned tank through the pre-signed-sparse
  exact-wall extractor and the current signed-sparse generic contour path.
- Record water-mesh positions/indices with geometric normals, then shaded
  images with filtered normals disabled and enabled. Do not use a shaded image
  alone as evidence that fluid geometry changed.
- Record solid-induced requested rungs, grading-induced rungs, accepted bricks,
  cells, pressure rows, faces, transport work, SVO nodes/leaves/voxels, primary
  visits, and per-stage GPU time.
- Capture all-fine visual and physical reference artifacts.
- Record primary traversal on every measured frame. Temporal primary-G-buffer
  reuse is not an accepted benchmark arm.

Likely files:

- `tools/benchmark-power-dam.ts` or a dedicated planar-wall benchmark;
- sparse CM12 trace/artifact schema modules;
- `tools/benchmark-svo-dry-frame-gpu.ts`;
- focused artifact tests under `tests/`.

Exit gate:

- repeated runs publish stable enough medians and exact topology/census values
  to attribute later changes.

### Phase 1 — Canonical authored planar patches

Purpose: create one exact source for tank walls without changing consumers.

Work:

- Add `SolidBoundaryWorld` and planar patch CPU types.
- Generate exact axis-aligned patches from the same vessel specification that
  generates `boxSolidVoxelShell`.
- Add canonicalization, validation, stable IDs, content stamp, and certificate
  hashing.
- Include `liquidContactCondition` in canonicalization, hashes, generation,
  packing, and validation; author box vessels as `legacy-orthogonal`.
- Retain the old `SolidWorld` pages and compare patch sampling to voxel sampling
  on axis-aligned walls.
- Add geometry-oracle tests for plane/slab point classification, segment
  intersection, UV coverage, and thickness.

Likely files:

- `lib/core/solid-world.ts`;
- a new `lib/core/solid-boundary-patches.ts`;
- `lib/core/webgpu-solid-world-pages.ts` or a companion patch-packing module;
- vessel/scenery construction modules;
- focused CPU tests.

Exit gate:

- every rectangular tank wall has one canonical patch and an exactly matching
  legacy voxel oracle; no physics or rendering behavior changes.

### Phase 2 — E0 fluid boundary and resolution-policy cutover

Purpose: allow calm fluid to remain coarse against axis-aligned exterior tank
walls.

Work:

- Compile E0 patches into closed faces at every candidate rung.
- Compile the same E0 patches into exact planar wetted-closure records and
  patch-normal scalar extension for water extraction.
- Replace the hard boundary promotion predicate with patch-aware evidence.
- Keep patch edges/corners in a configurable feature ring.
- Include patch generation in boundary/FPA dependencies and topology receipts.
- Verify transport clipping and extension against the compiled closed faces.
- Replace the signed-sparse extractor's unconditional generic-contour bypass
  with certificate-based planar contact handling. Preserve generic contouring
  for uncovered, residual, and non-planar solids.
- Add deterministic two-plane edge and three-plane corner ownership so the
  tetrahedral body diagonal cannot chamfer an exact box contact.
- Add an overlay distinguishing admitted planar boundary, residual fine
  boundary, and non-boundary refinement causes.
- Use benchmark branches/builds for A/B comparison; do not retain a production
  legacy representation flag after cutover.

Primary files:

- `lib/methods/adaptive-mass/sparse-brick-atlas.ts`;
- Sparse CM12 boundary image/operator modules;
- `lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts`;
- `lib/core/webgpu-water-global-fine-classify.ts` and
  `lib/core/webgpu-water-global-fine-tetra.ts`;
- grid overlay and trace modules;
- focused physics and topology tests.

Exit gate:

- the aligned tank is leak-free and matches the legacy all-fine physical
  reference within the agreed coarse-discretization tolerance;
- no brick is fine solely because it touches the interior of an admitted wall;
- patch corners and free-surface contact remain refined for explicit reasons;
- every planar wetted-closure vertex lies exactly on its authored wall plane,
  independent of rung, and no closure is displaced inward by contour span;
- the `legacy-orthogonal` arm matches the former exact-wall contact silhouette
  and corner topology, with filtered normals disabled for the geometry gate;
- changing only filtered-normal policy changes no water vertex position or
  index, and changing slip mode changes no static contact geometry;
- pressure/operator receipts remain valid and the system remains SPD.

### Phase 3 — SVO planar macro nodes

Purpose: remove fine planar wall traversal while preserving SVO-only static
geometry publication.

Work:

- Add a planar macro record buffer and terminal node state.
- Compile admitted patch interiors into the coarsest valid nodes.
- Retain ordinary voxels at patch features and overlaps.
- Add exact ray-plane/slab intersection to canonical and split primary paths,
  and to visibility paths that consume the same traversal.
- Publish material/normal/feature identity without source-primitive lookup.
- Extend construction coverage validation and primary visit histograms.

Primary files:

- `lib/svo/sparse-brick-octree.ts`;
- `lib/svo/webgpu-svo-traversal.ts`;
- `lib/svo/webgpu-svo-dry-scene.ts`;
- SVO publication/build modules;
- CPU/GPU traversal and voxel-coverage tests.

Exit gate:

- axis-aligned macro rendering is pixel-identical to the finest voxel reference
  under the matching face/thickness convention;
- oblique source-geometry mode stays within its declared image/geometry error;
- every source surface sample is covered by macro or residual geometry;
- primary leaf visits and time improve on the moving-camera plane benchmark.

### Phase 4 — Conservative primary beam prepass

Purpose: skip the common empty prefix for coherent primary rays, including
non-planar geometry.

Work:

- Add 4x4 and 8x8 conservative-distance prepass arms.
- Thread per-tile `tMin` through split primary traversal.
- Add small-interior-occluder, silhouette, sky, reverse-Z, and camera-motion
  adversarial tests.
- Measure prepass construction cost separately from visits/time avoided.

Exit gate:

- the output G-buffer is byte-identical to ordinary full traversal for exact
  arms;
- no adversarial occluder is skipped;
- the selected arm reduces total primary GPU time, not merely traversal
  counters.

### Phase 5 — One-sided oblique cut cells

Purpose: let oblique planar vessels use coarse fluid cells.

Work:

- Add robust cube-plane/slab clipping and polyhedral geometry oracles.
- Publish E1/E2 fragment volume, centroid, open faces, and solid faces.
- Extend pressure coefficients, divergence, projection, transport capacity,
  interpolation, and topology handoff.
- Add deterministic same-side small-fragment agglomeration/refinement policy.
- Compare to an all-fine oblique tank and a manufactured planar-cut operator.

Exit gate:

- mass cannot enter solid or leave the vessel;
- hydrostatic pressure and zero velocity remain stable;
- pressure symmetry/reciprocity and divergence gates pass;
- runtime and row count improve against the wall-forced all-fine arm.

### Phase 6 — Multi-fragment internal barriers

Purpose: preserve domain topology when a thin wall has fluid on both sides.

Work:

- Add background-cell-to-fragment indirection and stable fragment rows.
- Extend all scalar, face, pressure, collocation, interpolation, and transport
  authorities to fragment identity.
- Implement I1/I2 clipping and canonical fragment ordering.
- Add exact overlap handoff for fragment topology changes.
- Force refinement for unsupported multiplicity.

Exit gate:

- two differently initialized fluid regions separated by a sub-cell wall show
  zero scalar, pressure, and velocity cross-talk within numerical tolerance;
- removing the wall connects the graph in the next accepted topology generation;
- thin gaps remain connected rather than voxel-closing;
- SPD, mass, and generation gates pass.

### Phase 7 — Plane-certified tile visibility

Purpose: share more than the near distance when a screen tile is completely
covered by one planar macro patch.

Work:

- Add tile-frustum/patch coverage and nearest-geometry certificates.
- Resolve homogeneous plane depth/normal/identity per tile.
- Shade/material-evaluate at full pixel resolution.
- Fall back per tile to the general beam/full traversal path.

Exit gate:

- certified tiles reproduce full traversal exactly within the chosen depth
  representation;
- false-positive certification is zero in adversarial overlap scenes;
- net primary time improves beyond the general beam prepass on wall-dominated
  frames.

### Phase 8 — Automatic planar extraction and moving boundaries, optional

Automatic fitting and moving patches are separate projects after the authored
static path is production-safe. Each requires its own plan amendment and
acceptance matrix.

## 18. Test scenes

### Geometry-only

1. **Axis box shell** — every wall, corner, and optional lid; exact comparison
   to `boxSolidVoxelShell`.
2. **Single oblique slab** — several normals, offsets, and sub-voxel
   thicknesses.
3. **Finite patch edge** — rays/cells crossing interior, edge, corner, and just
   outside UV coverage.
4. **Patch plus residual obstacle** — competitor inside a candidate macro node
   must force descent.
5. **Two crossing patches** — unsupported initial topology must fail closed.

### Fluid

1. **Quiescent aligned tank** — full-depth and partially filled.
2. **Aligned hydrostatic column** — pressure probes at several depths/rungs.
3. **Dam break in aligned tank** — wall interior coarse, moving front adaptive.
4. **Oblique hydrostatic tank** — several angles and cut-volume fractions.
5. **Oblique wall jet impact** — tests trace clipping and wall-normal flux.
6. **Thin internal divider** — different density/velocity state on each side.
7. **Divider removal** — one explicit topology-connectivity transition.
8. **Narrow planar gap** — gap thinner than a background cell but physically
   open.
9. **Tiny cut fragment** — exercises agglomeration/refinement policy.
10. **Patch edge contact line** — verifies edge/free-surface refinement causes.

### Water reconstruction

1. **Uniform pool against one wall** — factors/rungs 1, 2, 4, and 8; every
   closure vertex has the exact plane-normal coordinate.
2. **Pool in a box corner** — two perpendicular closures meet without a
   diagonal tetrahedral chamfer, crack, duplicate, or inward fillet.
3. **Closed trihedral corner** — deterministic three-plane ownership and
   watertight indices.
4. **Dam-break wall contact** — compares accepted fluid fields separately from
   mesh geometry and shaded output over time.
5. **Surface-tension/slip matrix** — zero/nonzero tension and free/no slip prove
   that reconstruction policy, capillary force, and velocity condition remain
   independently controlled.
6. **Residual voxel obstacle** — no planar certificate; retains generic
   implicit-solid contouring without pretending the obstacle is a tank plane.
7. **Filtered-normal A/B** — identical positions and indices, with only normals
   and shading permitted to differ.

### Rendering

1. **Full-screen axis plane** — direct macro-hit throughput.
2. **Grazing stage floor** — matches the existing primary bottleneck.
3. **Full-screen oblique slab** — front/back thickness and normal.
4. **Plane with small foreground occluder** — beam/tile conservatism.
5. **Plane with a hole** — residual coverage and tile rejection.
6. **Plane meeting detailed geometry** — macro/residual seam.
7. **Moving camera orbit** — excludes temporal-cache credit.
8. **Settled camera** — verifies macro/beam still retraces and meets its budget
   without temporal primary-G-buffer reuse.

### Integrated north-star gate — Water box dam break

The eventual end-to-end acceptance test is the existing
`water-box-dam-break` scene running through Dawn and the normal UI presentation
path. It must prove all three layers in one isolated run:

1. **Boundary compilation** — the tank floor, walls, and enclosure are admitted
   into the canonical planar catalogue, with their exact thickness/material
   records and non-zero planar SVO terminal coverage reported in the artifact.
2. **Fluid adaptivity at the wall** — after the dam-break front contacts the
   enclosure, the Sparse CM12 receipt enumerates liquid cells sharing an
   impermeable face with planar-interior wall regions and proves that coarse
   rungs remain present there. Edge, corner, opening, and moving-interface cells
   may refine for their own explicit reasons. Zero wall flux, the existing mass
   gate, and pressure/topology validity remain mandatory; a coarse-cell count by
   itself is not acceptance.
3. **UI presentation** — the ordinary UI frame shows the water, tank enclosure,
   floor, and walls correctly, without missing macro faces, double-rendered
   voxel remnants, cracks at macro/residual seams, or incorrect contact
   silhouettes. Save the final image plus G-buffer/terminal counters alongside
   the physics receipt.

This gate comes after the small Dawn shader/parity and pipeline-construction
rungs. It cannot pass on render-only Phase 3: coarse liquid cells adjacent to an
impermeable planar wall require the E0 Sparse CM12 boundary cutover from Phase 2.

Current E0 receipt (2026-08-26): the isolated Dawn test passes at t=0 and after
two paper steps with one coarse 4^3 wet page beside the floor/x-low/z-low planar
faces and seven 8^3 interface pages. The ordinary UI independently reports
`7 fine · 1 coarse` after two steps. Render acceptance is measured separately
on paused Sparse CM12 symmetric expansion with primary traversal forced live:
ten samples at 58.0–60.2 FPS, normally 2.0–2.7 ms primary traversal, and 59 planar /
191 voxel terminals.

## 19. Quantitative acceptance matrix

Initial values are deliberately stated as gates or reporting requirements.
Phase 0 may tighten thresholds after baseline variance is known.

| Concern | Acceptance |
| --- | --- |
| Canonical axis geometry | exact plane location and thickness relative to the authored vessel specification |
| Oblique geometry | maximum declared Hausdorff error no greater than the configured former-fine-voxel bound |
| Patch coverage | every source solid sample covered by exactly one macro/residual route; zero uncovered samples |
| Fail-closed behavior | every unsupported patch overlap/topology retains legacy residual refinement |
| Wall leakage | zero accepted open flux edge crossing an impermeable patch |
| Planar liquid closure | every wetted-closure vertex has the authored plane coordinate exactly in the accepted f32 convention; zero scale/2 inward displacement |
| Contact topology | perpendicular planar closures have zero tetrahedral chamfers, cracks, and duplicate owners at certified edges/corners |
| Legacy contact fidelity | `legacy-orthogonal` reproduces the former exact-wall contact silhouette/topology with filtered normals disabled |
| Physics/presentation separation | a reconstruction-only arm has matching density, gamma, velocity, pressure, aperture, topology, and mass artifacts |
| Normal-filter isolation | filtered-normal arms have identical water positions and indices; only vertex normals/shaded pixels may differ |
| Multi-fragment isolation | no pressure/scalar/velocity graph path across an internal barrier |
| Mass | current Sparse CM12 conservation contract retained; no transition-attributable loss |
| Pressure structure | reciprocal off-diagonal coefficients; diagonal sum and SPD/gauge-fixed gates retained |
| Divergence | no worse than the matched coarse reference under the repository's established absolute ceiling |
| Hydrostatics | no monotonic parasitic-current growth; pressure-depth error reported against all-fine and matched-coarse controls |
| Wall-driven resolution | at least 90% of planar-interior finest bricks removed on the aligned-tank oracle |
| Fluid value | at least 4x fewer wall-attributable pressure rows/transport cells before claiming success |
| Fluid runtime | at least 2x speedup on a wall-dominated calm-tank step, with topology work included |
| SVO storage | macro plus residual bytes reported against fine-equivalent planar voxel bytes; target at least 8x reduction on the plane oracle |
| SVO visits | at least 4x fewer primary leaf/voxel visits for plane-hit pixels |
| SVO runtime | at least 2x primary-band speedup on the moving-camera plane/stage scene before default enablement |
| Browser paused performance | Sparse CM12 symmetric expansion, performance profile, primary traversal always live: sustained at least 50 FPS after warm-up while preserving the smooth spotlight-lit floor; report viewport, profile, primary time, and planar/voxel terminal census |
| Beam correctness | byte-identical exact G-buffer versus full traversal; zero missed adversarial occluders |
| Beam value | prepass plus primary total faster than primary control; prepass time reported separately |
| Scaling | recurring fluid work follows live cells/fragments/faces; render work follows accepted nodes/pixels, not fine wall area |

All runtime claims require interleaved runs and medians because the existing SVO
benchmark lane has shown substantial run-to-run drift. A counter reduction is
not a speedup until total GPU time falls.

## 20. Telemetry and artifacts

Every boundary compilation artifact should contain:

- canonical patch count by kind and sidedness;
- patch area, perimeter, physical thickness, material, and generation;
- admitted E0/E1/E2/I1/I2 counts;
- macro-admitted area and residual area;
- rejection count and first reason by patch;
- canonical, fluid, and SVO certificate hashes;
- former fine-equivalent solid voxels;
- residual `SolidWorld` pages and bytes;
- SVO macro records, nodes, residual leaves, payload bytes, and construction
  time.

Every Sparse CM12 step artifact should add:

- bricks requested fine by unresolved solid, patch edge, contact line, flow
  activity, and grading closure separately;
- background cells, fluid fragments, open fragment faces, and solid faces;
- fragment-count histogram and minimum volume fraction;
- agglomerated/refined tiny fragment counts;
- patch-incident pressure rows and graph edges;
- rejected fragment topology publications and first receipt;
- boundary compile/repair, pressure setup/solve, projection, transport, and
  topology transaction GPU time.

Every water-extraction artifact should add:

- patch catalogue generation and liquid-contact-condition hash;
- planar-contact cubes, emitted closure faces, ordinary contour cubes, and
  residual/fallback cubes;
- two-plane and three-plane seam-owner counts;
- maximum and first plane-normal vertex error by patch;
- position, index, geometric-normal, filtered-normal, and shaded-image hashes
  as separate fields;
- rejected planar-contact certificate count and first reason;
- extraction classify/scan/emit GPU time separated from SVO dry visibility.

Every SVO frame artifact should add:

- macro node visits/hits/misses;
- primary termination type histogram;
- ordinary leaf visits avoided;
- beam tile count and accepted/rejected reason histogram;
- plane-certified tile count;
- prepass time, primary traversal time, deferred shading time, and total frame
  time;
- exact primary G-buffer hashes.

## 21. Risk register and containment

### Flat geometry mistaken for flat flow

A planar wall can host a contact line, impact, vorticity, or a viscous boundary
layer. Keep geometry admission separate from flow/activity resolution. The patch
removes only the geometric hard floor.

### Exact pressure wall but rounded visible liquid

Closed pressure apertures do not guarantee correct free-surface closure. The
water extractor must consume the accepted patch and contact condition rather
than treating missing signed-sparse samples as air. Gate physical fields and
mesh geometry separately so a presentation regression cannot be hidden by a
leak-free pressure test.

### Contact angle confused with slip, tension, or shading

Tangential velocity slip, capillary surface tension, liquid-contact condition,
and filtered normals are four separate controls. Hash and report them
separately. The first planar path is `legacy-orthogonal`; a future contact-angle
model must modify scalar boundary conditioning and reconstruction, never only
the vertex normal.

### Exact planes acquire tetrahedral corner chamfers

A fixed marching-tetrahedra diagonal cannot represent the reflection symmetry
of a two- or three-plane box corner. Certified planar faces need independent
owners and deterministic seams. Any corner without a complete certificate
falls back to residual refinement rather than mixing exact and implicit owners.

### Opposite sides coupled through one cell state

This is the central topology failure. E0/E1 are one-sided. Do not enable
two-sided patches until fragment-specific scalar, pressure, interpolation, and
transport state is complete.

### Advection crosses a wall despite a correct pressure graph

Pressure non-penetration does not make semi-Lagrangian tracing or interpolation
topology-aware. Segment intersection and fragment-visible stencils are mandatory
gates, not later quality work.

### Tiny fragments damage conditioning

Use deterministic same-side agglomeration or refine. Report minimum volume and
solver iterations. Never inflate volumes or merge across a wall.

### Analytic oblique wall changes the voxel aesthetic

Axis-aligned patches are exact. Oblique patches default to source-geometry
accuracy and publish their difference from the former staircase. If a scene
requires the staircase, it can retain residual voxels or later use a 2-D digital
coverage/height representation.

### Macro node misses a finite patch edge or competitor

Admit macro nodes only when the complete node/patch intersection lies within UV
coverage and no residual geometry overlaps. Edge nodes descend.

### Beam corner test skips an interior occluder

Corner rays alone are not a certificate. Use conservative hierarchy/frustum
bounds, and retain adversarial small-occluder tests as default gates.

### Material variation forces geometric refinement

The first path accepts uniform material. Later, decouple planar geometry from a
2-D material page rather than returning to fine geometry solely for colour.

### Patch and residual generations diverge

Publish canonical and per-consumer certificate hashes. Accept atomically by
generation; never mix new macro records with old residual pages.

### Boundary metadata costs more than saved work

Use one catalogue and compact per-changed-tile compilation. Static patch
construction must not run per frame. Measure metadata bytes and construction
time separately.

### Strong grading retains most of the fine region

Feature rings around a large rectangular patch scale with perimeter rather than
area, but an overly wide ring can still dominate small tanks. Report direct and
grading-forced rung counts and tune from geometry/flow error, not appearance.

### Current sparse-world migration changes ownership

The repository is moving toward `lib/sparse-world` as the sole simulation
module. Keep the semantic boundary catalogue under `lib/core`; implement
consumer compilation behind the sparse-world boundary when that cutover exists.
Do not add new external callers of resident sub-pipelines.

## 22. Pull-request sequence

Keep representation, physics, and acceleration changes independently
reviewable:

1. **Baseline scenes, counters, and artifact schema** — no behavior changes.
2. **Planar patch CPU model and exact box-vessel authoring** — legacy consumers
   only.
3. **Patch GPU packing, validation, and cross-consumer certificate ABI**.
4. **Observational patch-aware resolution evidence and overlay** — no topology
   changes.
5. **E0 closed-face compilation and aligned-tank topology A/B flag**.
6. **E0 exact liquid-contact reconstruction** — patch-owned planar closure,
   mirrored scalar extension, deterministic corners, and legacy visual A/B.
7. **E0 transport/extension audit and physical acceptance suite**.
8. **SVO macro record and CPU traversal oracle**.
9. **SVO macro node construction and GPU canonical traversal**.
10. **Split-primary/secondary traversal integration and render acceptance**.
11. **Conservative 4x4/8x8 primary beam prepass**.
12. **Robust one-sided plane/slab clipping and E1/E2 geometry oracle**.
13. **E1/E2 pressure, transport, scalar, and topology handoff**.
14. **Tiny-fragment agglomeration/refinement policy**.
15. **Fragment state/row ABI and I1/I2 pressure graph**.
16. **I1/I2 transport, interpolation, and topology handoff**.
17. **Plane-certified tile primary visibility**.
18. **Default enablement decision** from the complete acceptance matrix.

Every recurring physics PR must retain:

- TypeScript checking for touched modules;
- focused CPU geometry/topology tests;
- focused Dawn/WebGPU publication and equation tests;
- current CM12 mass, projection, hydrostatic, and exact-authority gates;
- legacy fallback A/B until its replacement phase is accepted;
- byte-identical control on scenes with no admitted planar patch.

Every water-reconstruction PR must retain:

- exact plane-coordinate and deterministic corner/seam geometry tests;
- position/index hashes with filtered normals disabled and enabled;
- separate physical-state hashes proving presentation-only changes do not
  alter simulation;
- generic-contour fallback tests for non-planar and uncertified solids;
- the pre-signed-sparse exact-wall A/B until patch-owned contact reconstruction
  is accepted.

Every SVO PR must retain:

- source-to-publication coverage tests;
- canonical/split traversal agreement;
- primary G-buffer hashes on exact arms;
- small-occluder and residual-seam adversarial scenes;
- measurements with primary traversal encoded on every frame.

## 23. Initial file map

This is a planning map, not authorization to place every responsibility in the
named legacy module. Prefer the sparse-world boundary as that migration lands.

| Responsibility | Initial integration points |
| --- | --- |
| Canonical patch model | new `lib/core/solid-boundary-patches.ts`; `lib/core/solid-world.ts` |
| Vessel patch authoring | `lib/core/solid-world.ts`; vessel/scenery builders |
| GPU patch catalogue | new companion to `lib/core/webgpu-solid-world-pages.ts` |
| Boundary resolution evidence | `lib/methods/adaptive-mass/sparse-brick-atlas.ts`; resolution policy modules |
| Sparse CM12 boundary consumption | boundary image/operator modules; `webgpu-sparse-cm12-resident.ts` during migration |
| Liquid-contact reconstruction | `lib/core/webgpu-water-global-fine-classify.ts`; `lib/core/webgpu-water-global-fine-tetra.ts`; fine-level-set publication ABI |
| Trace clipping | `sparse-cm12-transport-home-frame-halo.wgsl.ts` and shared boundary helpers |
| Fragment geometry | new sparse-world/internal cut-fragment compiler |
| Pressure fragment rows | hot topology, pressure membership/cache, face/projection authority modules |
| SVO macro storage | `lib/svo/sparse-brick-octree.ts` and publication layouts |
| SVO traversal | `lib/svo/webgpu-svo-traversal.ts`; `lib/svo/webgpu-svo-dry-scene.ts` |
| Beam prepass | split primary pipeline in `webgpu-svo-dry-scene.ts` |
| Overlays/telemetry | grid overlay, SVO probe, sparse-world trace, benchmark artifact modules |

## 24. Open design decisions

These questions should be answered by Phase 0-2 prototypes, not guessed into
the final ABI:

1. Should the canonical geometry expose a zero-thickness `plane`, or represent
   every wall as a slab with possibly negligible thickness? The fluid wants a
   boundary sheet; rendering often wants two surfaces.
2. Is patch UV extent always rectangular for the first production ABI, or do
   authored polygon extents justify an indexed 2-D boundary immediately?
3. What stable fragment-ID scheme minimizes row churn across rerungs while
   remaining canonical under clipping?
4. What minimum fragment fraction gives acceptable pressure conditioning on
   the current solver without excessive feature refinement?
5. Can common planar cut signatures reuse the existing interned boundary
   catalogue efficiently, or should exact cut geometry be a separate compact
   per-tile lane?
6. Does the SVO node encoding have spare state for a macro terminal, or should a
   companion node-kind/record-index lane be introduced?
7. Is a macro record per admitted node acceptable, or should nodes reference
   one patch plus derive their AABB from traversal state?
8. Which exact paths beyond primary—hard shadow, cone, AO, GI—must consume macro
   records before publication is considered complete?
9. Does the existing raster-primary infrastructure provide a net win for
   plane-certified tiles on the target GPUs, or is compute depth generation
   simpler and faster?
10. How wide must the patch feature ring be for corners and contact lines under
    current CM12 interpolation?
11. After `legacy-orthogonal` is accepted, should physical wetting use one
    static contact angle per patch/material pair, and how should it couple to
    the existing surface-tension force without affecting zero-tension scenes?
12. Should exact wetted-closure records be packed beside the boundary catalogue
    or compiled into each accepted fine-level-set publication? Either choice
    must retain patch generation, coverage, and deterministic seam ownership.

Defaults until measured:

- authored static rectangular slabs only;
- f32 plane coefficients;
- one patch per admitted node/cell;
- E0 only for the first fluid cutover;
- `legacy-orthogonal` liquid contact for every phase-one vessel patch;
- exact patch normals on planar wetted closures; optional filtered normals only
  on the non-planar free surface until an explicit shading policy is accepted;
- uniform material per macro node;
- residual fine geometry at every edge/overlap;
- no plane-certified tile fill before the general conservative beam prepass.

## 25. Definition of done

The project is production-complete when:

- large admitted planar wall interiors no longer force finest-rung Sparse CM12
  bricks;
- flow/free-surface/feature evidence still refines where physics needs it;
- axis-aligned and oblique vessel walls are impermeable on coarse cells;
- the water surface at a certified planar wall closes on the exact authored
  plane at every rung, with no ghost-air displacement or tetrahedral corner
  chamfer;
- `legacy-orthogonal` contact reproduces the previous tank-wall result, while
  slip, surface tension, and filtered-normal controls remain independent;
- internal sub-cell barriers preserve disconnected fluid components across
  pressure, transport, interpolation, and topology transitions;
- patch thickness is independent of fluid rung and SVO macro-node size;
- SVO macro nodes cover every admitted planar interior with residual voxels at
  all unsupported features;
- all visibility paths required by the selected render preset consume accepted
  macro geometry consistently;
- conservative beam/tile acceleration cannot miss an interior occluder;
- fluid, water extraction, and SVO rendering publish matching boundary
  catalogue certificates;
- failure retains the legacy/residual representation or the prior accepted
  generation;
- the quantitative fluid and render runtime gates are achieved with all quality
  and conservation gates green;
- the legacy rule "every solid fraction change forces finest resolution" can be
  removed rather than merely bypassed by a scene-specific special case.

Until these conditions hold, expose the work as an experimental
`planar-boundary` path with legacy voxel boundary fallback.
