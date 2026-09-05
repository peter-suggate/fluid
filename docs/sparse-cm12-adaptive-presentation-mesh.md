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
full-size pool at reset and steps 1, 10 and 20. Coverage assertions are always
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
