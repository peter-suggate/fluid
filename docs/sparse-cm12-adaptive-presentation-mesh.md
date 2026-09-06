# Adaptive presentation mesh

Sparse CM12 now uses the accepted surface-cell width to choose presentation
mesh spacing. **Surface mesh refinement** defaults to **×2**; **×4** retains more
triangles. The control is live and appears under **Presentation pages → Mesh
refinement**. It does not reset physics or request a simulation refinement.

## Geometry and ownership

For accepted cell width `h`, the target extraction-group width is `h / ratio`,
clamped to the existing finest presentation sampling lattice. This change
coalesces the presentation mesh; it does not add sub-finest scalar samples.
Dyadic groups align to the solver's integer brick lattice, including when a
group spans several presentation pages. One lower-anchor invocation owns the
whole group. The remaining invocations relinquish ownership.

Each group retains the complete unit-sampled contour on its six faces. Shared
faces use the same packed phi, edge interpolation, quantization and ambiguous
marching-squares decision as the unit contour. Thus adjacent groups can differ
in size without T-junctions, skirts, overlaps, or independently reconstructed
boundary vertices. The group joins its boundary loop to an interior centre.
Normals use the existing common filtered scalar reconstruction.

Simplification requires all of the following:

- A complete, current, ordinary-page sample neighbourhood.
- One non-branching closed boundary loop.
- A monotone scalar direction normal to a convex projected loop.
- A bounded contour that fits the local scratch budget.

These conditions prevent a fan from joining disconnected components or
folding over itself. Failure retains the unit children, with their established
wall, macro-transition and missing-page ownership. Uniform groups produce no
surface. Floor films keep their dedicated height receipt. For ordinary-height
water, the adaptive volume groups replace the row-zero height shortcut so it
cannot publish a duplicate surface.

The ratio is a target, not a promise to discard unresolved features. Finest
cells, thin films, boundaries, native macro transitions and failed geometric
proofs can retain finer triangles. The existing packed scalar pages remain
allocated at their current resolution; this is mesh adaptation, not a reduction
in the scalar publication's storage.

## Publication and rendering

Compact CM12 reserves packed-sample bits 24–27 for `log2(h)` in finest-cell
units. Both the transactional publisher and the reference publisher write
these bits with phi, so extraction never consults a newer solver topology or
performs a CPU readback to choose resolution. Other publishers do not opt in.

The renderer carries the ratio separately from accepted physical state, in
its own compact uniform. A ratio change invalidates the retained mesh even
when buffer bindings and accepted generation are unchanged.

An explicit triangle uses reserved descriptor 193 without growing the existing
record: the two vec4 payloads hold `a.xyz/c.x` and `b.xyz/c.y`; the unused packed
cube x word holds the bits of `c.z`. Descriptor bits 8–9 select the proven
monotone axis and bit 10 selects its negative direction. Winding follows that
axis, independently of smoothed shading normals. Both scan and emission paths understand
this descriptor, including the indirect combined emitter. Existing vertex and
record capacity bounds remain unchanged.

## Correctness checks

```sh
node --import tsx --test tests/sparse-cm12-adaptive-mesh.test.ts
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_WEBGPU_BACKEND=metal \
  node --import tsx --test tests/sparse-cm12-adaptive-mesh-dawn.test.ts
npm run test:water-shaders
npm run test:dawn:sparse-cm12
npm run test:dawn:sparse-cm12:coarse-first
```

The focused Dawn test executes the production classifier, scan and emitter at
both ratios. Fixtures cover mixed accepted widths, curved, toroidal and box
surfaces,
disconnected liquid, marked pool columns, thin films, and a film rising into a
volumetric surface. It checks internal edge closure, manifoldness, finite
vertices, unchanged component counts, Euler characteristic and winding, and no
additional degenerate triangles.
The real coarse-first solver is also checked at reset and after advancing;
its live ratio change preserves the physics generation.

The existing floor/wall clipping emits some collapsed boundary triangles in
both reference and adaptive modes. The interior closed fixtures have zero;
this change does not claim to remove those pre-existing boundary degeneracies.

## Missing-pool-top regression (2026-09-06)

The default coarse-first pool exposed a Chrome/Dawn-on-Metal control-flow
failure in the new extraction path. Conditional returns inside the boundary
proof loops truncated segment accumulation and rejected valid patches. The
height-patch fallback similarly admitted unmarked columns, suppressing their
volume cubes. Winding was not the cause: the missing triangles were absent
from the vertex buffer before rasterization.

The proof loops now accumulate validity and return after the loop, before
publishing any triangles. Boundary interpolation, the single-loop/convexity
requirements, fan orientation, and unit-child fallback are unchanged.

A captured browser publication was replayed through the production classifier,
scan and emitter. The installed Node/Dawn build produced the complete surface
with the old code, while Chrome's Dawn backend reproduced the failure with the
same buffers and uniforms. A single-group Chrome probe reduced it to segment
collection and loop rejection. The repaired full-pool Chrome extraction emits
14,400 adaptive triangles and an upward projected top area of 40.960001 m²
(expected 40.96 m²), with zero downward projected top area. Before the repair,
the adaptive triangle count was zero and only 0.637502 m² remained at the top
boundary.

The focused Dawn gate now also exercises all adjacent 8/4/2/1 accepted widths,
resolution changes along each axis, clamped pool samples, an ellipsoid, and
per-triangle outward sphere winding. Both ratios are checked on the authored
full-size pool. The original coverage test made three advances; the follow-up
regression below pins actual solver step counts. Coverage assertions are always
active, including when `FLUID_FULL_POOL=1` selects just the full-pool cases;
edge closure alone must not accept an empty surface.

The full-size pool retains the unit-mesh baseline's floor/cap edge defects
below the free surface. The closed synthetic seam fixtures require zero
interior open edges and zero non-manifold edges. This fix does not claim to
repair the separate native floor/cap geometry.

Validation of the missing-top repair: focused Dawn mesh tests, unit tests and
shader validation pass. The full Sparse CM12 gate completed in 94.73 seconds:
15 of 16 lanes passed, including every correctness lane. The mini64 performance
lane measured 64.1597 ms against its unchanged 50 ms ceiling. Repository type
checking still reports errors in unrelated Losasso audit and existing test/probe
files; no adaptive-mesh file appears in those diagnostics.

## Page-boundary terraces after advancing (2026-09-06)

Chrome and native Dawn reproduce the blocky strips after three actual solver
steps (0.05 s). GPU readback shows no internal interfaces. The free surface
itself develops terraces: the third-step maximum is 1.615103 m for the authored
1.6 m pool, and step 30 spans 1.577629–1.616828 m.

The local height proof previously integrated only its own vertical brick. A
small amount of liquid entering the air brick made that page eligible for
height-derived signed distance, while the full brick below retained its
density-derived scalar. For example, the shared contour edge interpolated
between approximately -0.1 and +0.02467 m. Those samples use incompatible
scales; they move the zero crossing and tilt the shading normals. Adaptive
meshing faithfully exposed that upstream scalar discontinuity.

The height proof now finds the accepted liquid-to-air crossing, then integrates
one canonical vertical bracket around it, walking accepted cell widths. Both
sides of a page face therefore test the same endpoint conditions, include the
same partial liquid, and reconstruct the same physical height. A partial
coarse cell at the bracket's lower endpoint is anchored by its full neighbour
below, preserving off-grid waterlines. Monotonicity, open-volume consistency,
and wet/dry endpoint proofs retain the volumetric fallback for cavities,
detached liquid and cut columns.
The shared helper serves both publishers and the surface representability
proof. It does not change mesh winding or conceal interior geometry.

The Dawn regression now advances once per physical timestep and asserts the
solver's actual step count. `advanceTo` caps each call to one step, so jumping a
requested timestamp did not reach the named checkpoint. Reset and steps 1, 3,
and 30 now check upward coverage, absence of interior triangles, both mesh
ratios, and no added seam defects against the unit contour. The waterline must
stay within 1 mm at every checkpoint. The mixed-rung synthetic fixtures remain
strict about closure.

## Coarse-grid pressure imprint at 0.5 s (2026-09-06)

The remaining central rings were physical mass motion, also visible with the
unit mesh. After repairing publication, step 30 still had a 19.3 mm deficit in
integrated pool height and spurious velocities up to 0.131 m/s. The falling
ball was still above the pool; its adaptive support changed the cell widths
around the waterline. Removing the ball removed the central disturbance.
Tightening the pressure tolerance from 1e-3 to 1e-6 still left 0.115 m/s of
unwanted flow, ruling out pressure convergence as the primary cause.

The ghost-fluid pressure boundary interpolated `0.5 - density` without
accounting for the size of each cell. Full 8h liquid beside empty 4h air then
placed the zero-pressure boundary halfway between their centres, 5 cm below
the 1.6 m surface in this scene. The correct fraction is 2/3. Adjacent finer
pairs placed it at a different height, driving a grid-shaped pressure error.

Pressure classification now scales each interior-row density distance by its
cell width along the row axis before computing the ghost-fluid fraction. This
keeps the boundary in common physical units across 2:1 seams, for every
selector mode and axis. Same-width pairs retain their fraction. Exterior rows
retain their existing dimensionless ghost convention, and the separate
authored-region planar-height correction remains unchanged.

The full-pool regression reads accepted density and velocity independently of
the mesh. Before impact, column height must remain within 1 mm of 1.6 m and
pool speed below 0.003 m/s through 30 actual timesteps. This prevents a
presentation-only change from concealing the simulated disturbance.

The repaired full-pool Dawn extraction passes at both mesh ratios through
0.5 s. At step 30, accepted column heights span 1.599806–1.600100 m and maximum
pool speed is 0.000840 m/s. Emitted top vertices span 1.599916–1.600511 m, with
40.960001 m² of upward projected area, zero downward area and zero interior
triangles. At step 3, the top remains at 1.600000 m. All closed synthetic
mixed-rung fixtures retain zero open or non-manifold edges. Chrome verification
at 0.05 s and 0.5000 s clears the perimeter terraces and central rectangular
pattern, with no browser validation errors.

The canonical gate passes all 14 correctness lanes, including the offset
waterline, terrain front and live edits. Mini32 passes at 27.3285 ms; mini64
remains above its unchanged 50 ms ceiling at 64.3564 ms (64.1597 ms before these
follow-up fixes). The full run takes 102.86 seconds. The separate coarse-first
gate passes all four still-pool, impact, settling and macro re-rung tests.
The terrain test now retains its Dawn GPU instance through large readbacks;
otherwise garbage collection could destroy the native instance while map
callbacks were pending. No assertions, timing ceilings or lane selections were
weakened. Type checking still reports unrelated repository errors, with none
in the changed files.
