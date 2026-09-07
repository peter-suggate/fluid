# Retained density production acceptance

This is an implementation record, not a claim that arbitrary curvature transport
or every boundary interaction is complete. The authored plane, quadratic height,
box and sphere sources now compile to a physical density field used by both the
production native density initialization and surface publication.

The field is `q = clamp(0.5 - min(phi_i) / w, 0, 1)`, with fixed physical `w`.
Native densities are volume integrals of that field. Neither a region edit nor
native rung selection changes its coefficients or transition width. Fine support
moments are integrated once; native rungs restrict those same moments. Solids
use actual open voxel boxes or the authoritative quantized terrain bottom slab,
not a product of an average density and an average aperture.

## Evidence recorded so far

The independent CPU ladder measures the zero set and normals after 100
split/merge cycles. Its ten examples include planes, subtle quadratic curvature,
spheres and sharp branch intersections. Maximum surface displacement is
`9.40e-14 m`; maximum normal-vector error is `5.08e-15`.

The production quarter pool-impact scene uses the literal original region query
`0_0_0_25_66.6667_100_8_8`. The GPU acceptance changes the actual native topology
13 times, including widths 1, 2, 4 and the original local min/max-8 region, then
resumes simulation. On the successful run:

| Measurement | Result |
| --- | --- |
| Analytic crossing count | All 1,184 crossings present |
| Paused published scalar change | Exactly zero |
| Native restriction error | At most `4.48e-8` density |
| Difference from independent diffuse amount | At most `3.57e-9 m³` |
| Published pool-plane height error | `5.56e-17 m` |
| Extracted pool-plane height error | `5.97e-9 m` |
| Extracted sphere vertex radial error | `3.774 mm` |
| Extracted sphere triangle-interior radial error | `8.754 mm` |
| Extracted sphere normal-vector error | `0.03381` |
| Non-manifold / interior open edges | Zero / zero |
| Resume | One physical step accepted; velocity and field evolve |

These mesh errors are finite polygon and field-sampling errors, not an exact
sphere claim for planar triangles. The initial implicit boundary is the sphere;
the actual shipping mesh is independently measured against it. Its previous
averaged polygon centers shrank the sphere by up to 35.4 mm. Centers now project
onto the sampled field, and polygon fans must satisfy a geometry error budget.
Normals differentiate a quadratic interpolant rather than a smoothing kernel.

Quarter run: `/tmp/fluid-retained-quarter-production-4.log`. Saved raw production
mesh, scalar samples and receipts are under
`artifacts/retained-density-production/coarse-first-pool-impact-quarter`.
The actual mesh cross-section comparison is
`artifacts/retained-density-production/analytic-mesh-sections.png`.

## Evolution and remaining validation

An actual scalar update creates the next retained generation on persistent
physical support. Each support retains `a*q_seed + b`; draining scales density,
filling mixes toward capacity, and compression retains positive excess. Native
integrals are rebuilt from these coefficients and checked against accepted
solver amounts before the generation publishes. Topology edits only integrate
the accepted generation. Replacement residents copy physical support and then
re-integrate it into the new native partition.

This first conservative evolution basis is discontinuous between support boxes.
It preserves the initial curved and sharp source geometry, but does not yet
provide high-order transport of curvature into initially empty supports. The
initial support arena also retains a dense bounded domain prefix, followed by
sparse growth slots. It is not the final sparse curvature representation.
General partial solid reopening needs further geometric support work; it must
not silently re-author source liquid.

The half-size original-region acceptance, resumed partition ladder, mesh
regressions and canonical Sparse CM12 Dawn suite remain mandatory before
declaring this implementation milestone complete. No acceptance threshold or
performance ceiling is to be relaxed to obtain a pass.
