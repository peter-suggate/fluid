# Procedural solid boundaries

The scene generator supplies a continuous implicit solid before voxelization.
`bakeProceduralSolid(dimensions, cellSize_m, sample, materialId)` produces both
`solidVoxels` and `solidBoundary`. Assign both results to the scene. `sample`
takes lattice coordinates and returns a finite scalar, negative inside solid.
An SDF is suitable; a regular implicit field also works. Gradients are derived
from the sampled field, rather than separately authored directions that could
disagree with the surface. Cell sizes convert them into physical normals.

## Representation and authority

- The boundary contains sparse 8³-cell bricks with 9³ node samples. Shared
  nodes must agree exactly. Unlike a single heightfield, this can retain
  floors, ceilings, walls and multiple solid layers in the same column.
- The continuous representation used by simulation is the trilinear field
  inside each cell. Features must be resolved by the authoring lattice;
  features entirely between node samples are not recoverable. Curved surfaces
  and corners have the approximation error of that sampled representation.
- Generated voxel fractions and distances come from this same field. The
  voxels remain the source for existing voxel presentation and editing;
  continuous simulation queries use the boundary where it is valid.
- The document carries ordinary JSON data, not callbacks. Its generated
  patch-prefix count and content key identify which voxels it describes.
  Replaced or reordered generated patches invalidate the boundary. Scene
  finishing can prepend a container shell while preserving this provenance.
- A content key participates in the solver's seed/reset key. Changing the
  source boundary or lattice requires rebuilding the solver. Ordinary voxel
  edits can be applied live without reallocating boundary storage.

## Solver integration

The boundary uses a hashed brick directory appended to the existing solid
storage buffer. Existing scenes without boundary data keep their previous
query path. No per-frame generator callback or new bind group is needed.

Particle seeding, initial fluid capacity, particle contact, solid-centre
classification, pressure cell capacity and pressure-face dual volumes query
the boundary. Capacity integration uses eight local plane integrals, exact
for planar fields; it is a quadrature approximation for curved trilinear
fields. CPU initialization and GPU integration use the same rule.

Contact walks the cells crossed by a particle's existing half-cell strides.
The restriction of a trilinear field to a line is cubic. Bernstein bounds
find a crossing even when both endpoints are outside, with a maximum
parameter uncertainty of 2^-14 within a crossed cell. Contact removes inward
normal velocity and preserves tangent velocity. The inferred voxel bed force
is disabled where an authoritative boundary is present, avoiding two slope
responses for one surface.

The retained native-terrain experiment remains a separate path. This work
does not establish that its earlier volume-growth problem is solved.

## Edits and undo

Each brick has a per-cell validity mask. A later fill or clear invalidates
only cells touched by that patch's rasterizer. Those cells use the edited
voxel geometry; unaffected cells retain their continuous boundary. Paint
preserves geometry. Undo regenerates validity from the immutable source and
the surviving edit suffix, restoring the boundary without modifying samples.

The GPU boundary image and solid records are refreshed even when an edit
leaves the binary occupancy mask unchanged. This matters for a fill in a
previously cut cell: the mask may already say solid, while its geometry has
changed from a sloping partial cell to a whole voxel.

## Water slide and verification

`nbflip-water-slide` now generates its spiral floor, walls, hood and pit as
one continuous implicit solid. The floor falls without integer rounding.
The original whole-voxel staircase remains available through
`createNbflipWaterSlideVoxelReference` for inferred-gradient regression tests.
The long and theme-park slides retain their existing definitions; other
generators can adopt the same boundary API.

Tests cover planes across brick seams, floor/ceiling/wall orientation, stacked
surfaces, JSON round trips, edit invalidation, undo and initial fluid capacity.
`tools/probe-procedural-solid.ts` checks GPU values, derivatives, capacities
and contacts against CPU expectations, including an interior cubic crossing.
`FLUID_CLEARANCE_BOUNDARY=procedural` selects the same sampled-boundary path
in `tools/probe-slope-clearance.ts`. Its existing 1% drift gate remains a
diagnostic convention, not Peter's acceptance threshold.

Full clearance, volume drift and performance must be reported from measured
fluid runs. Geometry contracts and unit tests alone do not establish them.

### Measurements on 14 October

- The real-GPU contract probe passes: four queries agree with the CPU field
  to 1.91e-7; contact time is 0.693726 versus 0.693750 analytically, and the
  cubic interior crossing is 0.399963 versus 0.400000. Remaining normal
  velocity is 7.21e-8 m/s. Log: `/tmp/fluid-procedural-contract-gpu.log`.
- Type checking passes. Latest full unit run: 819 tests pass (the shared
  checkout's other session removed older test files after the earlier
  896-test run). Current-checkout shader preflight passes all modules.
- An isolated copy containing these changes completes 30 fluid steps at
  60 Hz without a solver error. GPU volume accounting rises from 384 to
  1105.885 cell-volumes (+187.99%) at 0.5 s; 4366 particles remain on the
  slope. This is not a clearance acceptance run or a volume-conservation
  success. The earlier reconstruction-growth investigation remains open.
  Log: `/tmp/fluid-procedural-slope-flow.log`.
- That run also exposed an error in the older diagnostic: it applied the
  binary voxel mask to already integrated owner volumes, excluding water in
  partial cells. Its `total_cells`/`ramp_cells` values are invalid for this
  run; the GPU `stats_cells` values above are the authoritative measurement.
  The probe now sums the expanded owner averages without that extra mask and
  checks their sum against GPU accounting. No fluid behavior was changed to
  compensate for this measurement error.
