# Current density: bounded 3D tensor investigation

Status: CPU mathematical validation, not a selected production replacement. The isolated current-quadric GPU prototype establishes useful affine transport cases; it does not represent general evolving fluid geometry. This note proposes the next smooth-field carrier and records its demonstrated failure cases. Neither prototype has fixed shipping fluid motion.

## Decision and rejected shortcut

Investigate a full tensor of the 1D conservative quartic functionals. In each coordinate, a cell polynomial matches its two endpoint values, two endpoint derivatives, and integral. Shared functionals give shared value and first-derivative traces; the volume integral belongs to the same current polynomial queried for density. The original CIP-CSL4 paper motivates the 1D combination of characteristic point data and conservative integral constraints. The tensor construction below is our mathematical extension, not a claim to implement that paper's complete solver. [Yabe et al., 2001](https://journals.ametsoc.org/view/journals/mwre/129/2/1520-0493_2001_129_0332_aecsls_2.0.co_2.xml)

A smaller candidate fails dimensional consistency. Start with shared tricubic Hermite vertex jets, then correct each cell's mass by adding `δ β(x)β(y)β(z)`, where `β(t)=30t²(1−t)²` and its integral is one. This preserves C1 face traces, but a residual for a field depending only on x invents dependence on y and z. For `q=.5+.1x⁴` on the unit cube, the tricubic part is `.5+.1(2x³−x²)` and `δ=.1/30`. At x=.5, the corrected density is .5 on y=0 and .52197265625 at the cube center; the actual density is .50625 everywhere on that section. Thus the 64 local Hermite constraints plus one volume bubble are a negative control, not the chosen next carrier. Local C1 tricubic interpolation itself is well established; it does not solve this extra conservative constraint. [Lekien and Marsden, 2005](https://onlinelibrary.wiley.com/doi/10.1002/nme.1296)

## The full tensor and its cost

Use normalized coordinate t in each support and functionals `[V0,hD0,V1,hD1,I]`, with I the cell average. Their five cardinal polynomials have power-coefficient columns:

```text
V0 : [1, 0, -18, 32, -15]
hD0: [0, 1, -4.5, 6, -2.5]
V1 : [0, 0, -12, 28, -15]
hD1: [0, 0, 1.5, -4, 2.5]
I  : [0, 0, 30, -60, 30]
```

Their tensor product has 125 local coefficients. Canonical shared entities store only 27 scalars per periodic bulk cell: eight vertex mixed derivatives, twelve edge partial integrals/derivatives, six face partial integrals/derivatives, and one volume mean. No 125-coefficient expansion is retained globally. Every edge and face has one authority; independent per-leaf interpolation would break this property.

Two f32 banks cost 216N bytes: 40.5 MiB at 196,608 periodic bulk cells. A finite 64×48×64 support grid, counting outer boundary entities, costs about 41.92 MiB before directories, admission receipts, staging, branch records, and velocity data. The CPU reference uses 27N f64 values per state: 108 KiB at 8³ and .844 MiB at 16³. A directional translation pass uses at most five source slots for a point/derivative and ten for an interval average. Three sparse passes require roughly 500–800N multiply-adds; polynomial range checks require roughly 750N useful multiply-adds. These are arithmetic/storage estimates, not GPU performance results.

Quadratic B-splines are a smaller alternative: one coefficient per control point, 27 local basis values and C1 smoothness on an ordinary grid. Bounded coefficients provide a sufficient range condition. However, matching prescribed local masses with bounded shared coefficients is a coupled and potentially infeasible constraint; simply advecting coefficients does not establish local conservation. Hierarchical splines require an actual nonnegative partition of unity across refinement, not independent octree fits. They remain an alternative to benchmark if the tensor memory or remap cost is excessive. [Giannelli et al., THB-spline methods](https://gs.jku.at/pubs/NFNreport30.pdf)

## What transport and admission mean

For uniform translation, the reference measures every destination mixed functional directly on the old **current** field: derivatives at departure points and exact partial integrals over departure intervals. The three tensor operators commute mathematically. Reconstructing a destination polynomial from those measurements remains an approximation: fractional translation moves old polynomial joins inside destination cells. Exact moment transfer is not exact pointwise translation of an arbitrary piecewise polynomial. Integer lattice translations and polynomials contained in the target space are special exact cases. A global quadric is closed under affine pullback; a general tensor quartic is not closed under rotations or nonlinear deformation.

For one local quartic, Bernstein coefficients are `[V0, V0+hD0/4, −2V0−hD0/4−2V1+hD1/4+5I, V1−hD1/4, V1]`. Apply this conversion in all three axes. Because tensor Bernstein basis functions are nonnegative and sum to one, all 125 coefficients in [0,1] suffice to bound the whole cell. Failed bounds require rejection or refinement; they do not alone prove actual negative density. Subdivision can tighten conservative bounds. Float64 or f32 arithmetic is not outward-rounded interval certification, so a production certificate needs a rounding-error enclosure. [Foufou and Michelucci, 2012](https://interval.louisiana.edu/reliable-computing-journal/volume-17/reliable-computing-17-pp-192-208.pdf)

Two actual positivity failures are already explicit in the CPU tests:

* A positive clipped plane goes from one to zero over x=[.015,.035] inside a support [0,.25]. Exact endpoint values are 1/0, endpoint derivatives vanish, and the mean is .1. Its constrained quartic evaluates to −.25 at x=.125. Exact authored moments do not make this representation admissible.
* An initially admissible periodic C1 field has four unit-width cells with vertex values `[0,0,1,1]`, zero slopes, and means `[0,.5,1,.5]`. Its Bernstein range is exactly [0,1]. After a half-cell translation, the candidate has Bernstein range [−.15625,1.15625] and actual sampled values below −.028 and above 1.028. Mass and C1 continuity still hold. Forming this rejected candidate leaves the accepted source unchanged.

Independent cell blending toward a mean generally breaks shared C1 traces. Clipping density changes its integral and introduces new branches. Neither is an acceptable hidden repair. The prototype must reject a failed candidate; production needs a demonstrated shared refinement or branch mechanism to recover admissibility. Refinement alone does not guarantee strict positivity at a finite resolution or exact saturation geometry: a single polynomial cannot be identically zero on part of a cell and nonzero elsewhere. A misaligned saturation boundary therefore requires explicit handling, not a promise that a sufficiently fine grid will exactly reproduce it.

## General geometry remains a separate gate

A smooth scalar can contain multiple disjoint components and level-set topology changes. It supplies a smooth interface only where its gradient is nonzero. A merger can have a critical point with zero gradient; a regular-normal assertion there is mathematically inappropriate. Pure invertible material advection preserves component topology until contact or another physical operation intervenes.

The existing sphere density `clamp(.5−phi/w,0,1)` is only piecewise smooth at its saturation boundaries. A within-cell clipped quadratic is not a tensor polynomial. The smooth bounded wave below does **not** establish clamped-sphere initialization, exact t=0 curvature, dry/full plateau transport, or general primitive acceptance. An explicit current numeric quadric/cut branch, or a measured convergent approximation admitted before adoption, remains necessary.

Intentional sharp intersections require canonical selectors with separate one-sided traces. A finite union uses the selected maximum density, not summed branch masses. Integrals must partition winning regions without double counting; gradients at a tie remain branch-specific. An arbitrary fixed cap of two or four branches cannot represent every possible component arrangement. Overflow needs refinement or transaction rejection. This note has no demonstrated universal sharp-branch representation.

Sparse physics cells may integrate a separate canonical fine support field, but support splits/merges must conserve its moments and preserve shared traces. Exact box integrals do not solve solid cut cells, nonlinear departure footprints, or branch intersections; those need bounded integration with complete worklists and error receipts. A divergence-free sampled velocity alone does not make an RK2 departure map volume preserving. See the [independent map oracle](../tools/implicit-density/volume-preserving-map-oracle.ts) and the separately owned cutover audit for immutable source-velocity staging. This CPU tensor oracle assumes a prescribed uniform translation and does not implement general characteristic jets or a production flux scheme.

## Evidence and the next ladder

Command: `node --import tsx --test tests/tensor-csl4-oracle.test.ts tests/conservative-hermite-motion-oracle.test.ts`.

The tensor tests verify independent 1D reduction to roundoff, all 27 translated mixed functionals, exact current-cell integrals, complete face coefficient identities, and explicit rejected cases. The smooth fixture is `.5+.1 sin(2πx)sin(2πy)sin(2πz)+.07 cos(2πx)cos(2πy)cos(2πz)`. Each step translates by `(h/3,2h/3,−h/3)` for 3N steps, completing a periodic orbit. Range bounds are checked at every generation; total mass error stays below 3e-12. The oracle reports these measurements on this host:

| Cells per axis | Steps | Maximum density error | Maximum gradient-component error | CPU orbit time |
|---|---:|---:|---:|---:|
| 4 | 12 | 7.014e-4 | 4.864e-3 | .096 s |
| 8 | 24 | 2.199e-5 | 2.236e-4 | .332 s |
| 16 | 48 | 6.798e-7 | 1.026e-5 | 4.110 s |

This is roughly fifth-order density and fourth-order gradient convergence on this smooth translation fixture, not a general theorem or a production timing claim.

| Gate | Required independent evidence | Current status |
|---|---|---|
| Tensor algebra | 1D reduction, mixed functional transfer, exact box moments, full C1 face nets | CPU passes |
| Smooth translation | Several resolutions, complete orbit, analytic value/gradient errors, same-field mass and bounds at every step | CPU 4³/8³/16³ passes |
| Rejection protocol | Admitted source → inadmissible candidate; accepted field/parity unchanged; no clip | CPU negative control; GPU implementation pending |
| GPU translation | Independent CPU comparison of every stored moment, queries, box amounts and bounds; f32 budget and resource counts | Pending |
| Saturation and curved primitives | Original clipped plane and sphere, t=0 fidelity, translated saturation boundaries, explicit branch/refinement admission | Unresolved; clipped-plane counterexample rejects |
| Non-affine deformation | Exact invertible three-shear map, current-field mixed-jet/partial-integral transfer, controlled integration errors and mass, reversal/convergence | Analytic map fixture only |
| Sharp branches | Moving box edge/corner, disjoint components, contact/merge semantics, winning-region integrals, branch overflow | Pending |
| Sparse production | Split/merge and clipped solids, signed support growth, accepted source parity, full-fine motion before coarse scenes, shipping native M0 from same field | Pending |

Only after these gates should the field be considered for the canonical production regression matrix and matched-camera fluid motion comparisons. The current result is a credible smooth-field experiment with explicit rejection, not a universal density solution.
