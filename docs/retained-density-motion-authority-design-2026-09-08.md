# Retained density must transport spatial information

Status: diagnosis and proposed transport contract, **not a production evolution acceptance**. The initial analytic field and zero-time native restrictions passed their gates. The evolved quarter-pool coarse/all-fine visual comparison failed in both configurations. No mesh adjustment can establish the missing density transport contract.

## What the present authority loses

`cm12AdvanceRetainedDensitySupportAt` in `webgpu-sparse-cm12-resident.wgsl.ts` retains

\[
q_K(x)=a_K q_{seed}(x)+b_K
\]

on each fixed finest support's current open geometry. The update observes old/new **native cell means**, with no departure coordinate, gradient or spatial moment. Within an unsaturated old ramp, its normal remains parallel to the original normal: `grad(q)=a grad(q_seed)`. Initially constant supports remain constant. This is a limitation of the physical density authority before rendering.

[The CPU contract tests](../tests/sparse-cm12-retained-motion-contract.test.ts) provide independent negative controls:

| Motion | Exact measure | Why the present representation fails |
| --- | --- | --- |
| Radius .1 m sphere moves .1 m inside one .4 m native cell | Both old/new amounts are .00432098738586867 m³ | Even ideal native target means are unchanged; the old shape stays pinned. Density error exceeds .9 at diagnostic points. |
| Plane moves into an initially dry .05 m fine support | New mean .3828125 | The exact interface is inside the support; any `a*q_seed+b` matching that mean is constant below .5 and has no interface there. |
| Radius .25 m sphere falls .1 m | New equator has horizontal normal | Old-ramp normal differs by 21.8014°. At the leading pole, only .0238613 m of motion reaches constant old air. |

Predictions: a falling sphere retains an old-centered cap and acquires block-like leading pieces; a dam front wets constant supports before an interface can appear; a collapsing pool cannot generate new inclined/curved interfaces from formerly saturated supports. Finer physics cells reduce the block size but do not restore missing spatial information. Asymmetry additionally depends on velocity/topology and is not implied by the basis argument alone.

There is a **second incompatibility**. With uniform velocity, uniform native width `H`, and `gamma=beta=1`, CM12 center interpolation translates means by the fractional offset `d/H`. In the contained-sphere example `d/H=.25`, this sends 25% of the mass into a neighboring cell which the exact translated diffuse sphere never reaches. Therefore exact transported geometry cannot also obey those unchanged native targets. Chentanez–Müller advects surface density itself (equation 3) and defines its conservative matrix from trilinear characteristic weights (§3.4); conservation does not make that low-order spatial approximation exact. [Original CM12 paper](https://matthias-research.github.io/pages/publications/masscon_sca.pdf).

## A separate published-zero contract failure

`cm12RetainedDensityPhiAtFine` also creates false zeros at the endpoints of the attainable density interval. For `a>0` and `s=q_seed` in `[0,1]`, the actual retained density is `q=a*s+b`:

| Coefficients | Exact points with `q=.5` | Present shortcut |
| --- | --- | --- |
| `b=.5` | Only `s=0` | Returns zero for every `s`, including points with `q>.5` |
| `a+b=.5` | Only `s=1` | Returns zero for every `s`, including points with `q<.5` |
| `b<.5<a+b` | `s=(.5-b)/a` | The inverse unsaturated ramp has the correct zero/sign |
| `a=0,b=.5` | Every point | A real volumetric half-density plateau |

This is observed in the **actual all-fine prescribed-flow capture**, not only an algebraic example. After the first half-cell translation, all 144 exactly-zero published GPU samples have retained density different from `.5` by more than `1e-4`; the largest discrepancy is `.5`. At dense index 10668, world point `(-.175,.675,-.275)`, the captured coefficients are exactly `a=.5,b=0`, point seed density and retained density are both zero, but published `phi=0`. Its seed mean is only `.00018469570204615593`; a tiny diffuse corner amount does not justify a zero over the support's whole volume. The independent CPU counterexample reconstructs that same geometry. Raw evidence: `artifacts/retained-imposed-flow/sphere-full-fine/published-zero-semantic-audit.json`; source hashes and capture time are in its neighboring `provenance.json` (WGSL SHA-256 `016ebcd1b74b3c1ab5d67a5c958950303ccb4116c6139c035623b5ab02b5ba4e`).

Removing a non-strict tie shortcut cannot restore the transported field. The mathematical companion `w*(.5-q)` would have exact point signs/zeros, but it inherits two intrinsic defects of the current authority:

- Real half-density plateaus occupy open volumes when a saturated seed is drained to `.5`, or empty seed is filled to `.5`. Any differentiable function with precisely those zeros has zero gradient there. A smooth function can have such a plateau, but it is not a regular two-dimensional interface and supplies no unique normal.
- Across a support jump from `q=.25` to `q=.75`, the half-open support definition has no `q=.5` point. Any continuous function with the correct opposite phase signs must nevertheless cross zero. It therefore cannot preserve both the signs and the density's zero set. Zero-set equivalence alone could be obtained with a constant nonzero function, but that would discard the liquid phase signs.

The representation must be replaced or rejected when it cannot carry the required transported interface; a valid native mean `.5` is not itself an invalid physical state. A surface interpolation adjustment would hide the failure rather than supply the missing density. These endpoint, plateau and discontinuity cases are retained as CPU negative controls. No publication or renderer patch is part of this diagnosis.

## Required authority and continuity

Let `X` be the accepted forward characteristic map from the previous generation. For fixed geometric supports `J,K`, the conservative measure is

\[
M_\alpha^{n+1}(K)=\sum_J\int_{J\cap X^{-1}(K)}(X(x)-x_K)^\alpha q^n(x)\,dx.
\]

For an incompressible map this accompanies `q^(n+1)(y)=q^n(X^-1(y))`. If the numerical map has a nonunit Jacobian, conservation requires accounting for it; scalar pullback alone is insufficient. Translations, rotations and unit-determinant shear provide exact first gates.

The new accepted field must yield both point queries and support/native integrals. Native pressure topology may change independently; native `rho` becomes a restriction of the transported measure. It must not be an independent center-interpolated target that a shape is subsequently forced to match. The same rule applies to sharpening, diffusion, injection and solid displacement: each changes the actual field/measure through an explicit operation, not only its native receipt.

Smooth interface branches need a single-valued field and first derivative across support faces near `q=.5`. Independently fitted quadratics do not ensure this. Globally coherent quadratic pullbacks do. Sharp branches intentionally have one-sided derivatives; their logical union/intersection and branch identity must survive transport. Averaging their normals would change the field. No global `C1` claim applies across actual sharp edges or the saturation boundaries of a clamped ramp.

## Representation choices

The following recommendations are engineering inferences; the cited methods do not supply a ready-made 3D Sparse CM12 cutover.

| Candidate | What it supplies | Limitation for this task |
| --- | --- | --- |
| Advected reference map | Carries the inverse motion, so evaluating a material field at transported coordinates preserves geometry under an exact map | Interpolation of a map does not automatically conserve the composed density's discrete integrals. Long-time distortion, injection and remapping require explicit treatment. |
| Persistent conservative semi-Lagrangian support pieces | Integrates the old field over departure overlaps and retains mapped geometry | Repeated clipping fragments can grow without bound; conservative compaction/refinement is necessary. |
| Spatial moments / DG polynomial | Carries mass, centroid and deformation information with bounded local storage | Moments do not uniquely identify an arbitrary interface; discontinuous local polynomials do not guarantee smooth cross-face geometry. |
| Shared Hermite/CIP multi-moment field | Transports point values/derivatives and conservative integrals as constraints of the same field; shared degrees of freedom can enforce smooth face traces | Positivity, saturation, sharp branches and multidimensional cost still need proof and tests. |
| Coherent local implicit quadrics/branches | Exact affine transport of plane, sphere and ellipsoid branches; compact curvature representation | General-flow refitting must preserve shared traces and provide an error bound. Finite sample agreement alone is not a certificate. |

Reference-map work explicitly uses an Eulerian inverse motion field; it is a useful kinematic model, not evidence that arbitrary composed liquid-density integrals are automatically conserved. [Kamrin, Rycroft and Nave (2012)](https://www.sciencedirect.com/science/article/pii/S0022509612001135).

Moment-of-fluid adds material centroids to volume fractions and uses their mismatch as a geometric refinement indicator. A conservative semi-Lagrangian DG method can instead remap polynomial moments through a characteristic weak formulation. These support carrying spatial information; neither establishes exact arbitrary curved/branched geometry from finitely many moments. Our Legendre counterexample has identical `M0,M1,M2` and different half-density sets. [Adaptive MOF](https://cnls.lanl.gov/~shashkov/papers/amr_mof_jcp.pdf), [Cai, Guo and Qiu](https://arxiv.org/abs/1612.06977).

CIP-CSL4 uses a quartic profile constrained by endpoint values, derivatives and a cell integral obtained from Lagrangian remapping. Thus integral conservation and characteristic shape transport belong to one numerical representation. Its published square-wave results also show overshoot: `C1` interpolation alone is not positivity. The multidimensional CIP/MM formulation transports boundary point values and volume averages using characteristics and fluxes. Neither method licenses constraining a new field to the old CM12 center-interpolation result. [Yabe et al. (2001)](https://journals.ametsoc.org/view/journals/mwre/129/2/1520-0493_2001_129_0332_aecsls_2.0.co_2.xml), [Ii and Xiao (2007)](https://www.sciencedirect.com/science/article/abs/pii/S0021999106004189).

## Bounded implementation to validate

First validate an isolated GPU **current-generation** quadratic pullback under prescribed affine flow. Use ping-pong banks and check all intersected donor supports for coherent coefficients; reject unsupported mixtures instead of truncating donors or selecting one winner. An originally dry destination must obtain arriving geometry from upstream. This exact restricted case tests the missing transport operation without a production scene oracle. Its coefficient pullback can be exact while numerical support quadrature still has an error; report both separately.

The [1D conservative Hermite oracle](../tools/implicit-density/conservative-hermite-oracle.ts) now transports shared values/derivatives and exact departure integrals of its current quartic. Its [three CPU tests](../tests/conservative-hermite-motion-oracle.test.ts) preserve face continuity and global mass through a periodic orbit. With 8/16/32 cells, maximum density errors are approximately `2.31e-5 / 7.12e-7 / 2.20e-8`. A necessary negative control gives endpoint values `1,0`, zero endpoint slopes and mean `.1`: the unique quartic has midpoint `-.25`, although a positive thin layer can realize those endpoint/mass data. Thus the promising smooth result does not establish positivity for unresolved interfaces. This oracle is not a full CIP solver or a production CPU path.

For general flow, prefer a shared smooth representation with bounded branch exceptions over unconstrained independent quadratic fits or indefinitely accumulating fragments. Carry current geometry through the same characteristic map used for integration. Use transport residuals and explicit derivative/range bounds to admit the representation; otherwise refine geometric supports independently of native pressure rungs. Do not hide residual mass in a subthreshold film, uniform offset or rescaled original seed. This remains an implementation decision pending the two probes, particularly positivity and branch tests.

Approximate storage budgets, excluding existing topology/masks/scratch:

- A 64-byte quadratic support record in two banks costs `128N` bytes: 24 MiB at `N=196608`. Two-to-four branch exceptions need a bounded sparse interface allocation.
- A possible shared tensor-Hermite layout stores eight derivative components per vertex (`q,qx,qy,qz,qxy,qxz,qyz,qxyz`) plus local integral data. At roughly 40 bytes per cell per bank this is about 15 MiB for two banks at that `N`, with extra boundary vertices. Caching 64 expanded cubic coefficients per cell would add 96 MiB for two banks; derive them locally instead. This is a budget sketch, not a selected 3D scheme.
- A translated axis-aligned support overlaps at most eight source supports, regardless of translation distance. A certified departure AABB of width at most `2h` per axis needs at most 27. More deformation requires subdivision or a larger admitted worklist, not missing donors. Large translation alone is not a reason to force a fine-cell CFL limit.

Reuse the existing fixed-support directory, allocation stamps, accepted-bank lifecycle, projected/extended velocity characteristic helper, and native integral compilation. Existing density-stencil weights are not geometric overlap integrals; the momentum channels are not spatial first moments. Rigid/static open geometry must clip the same source and transported measures, and full closure requires explicit conservative displacement. Zero-time partition changes only restrict the accepted field.

Acceptance must examine the **density authority before the mesh**: plane/sphere/box translation, support-boundary crossing, rotation, reversible deformation, mass, centroid, covariance, half-density location, normals and sharp edges. Use identical prescribed velocity for coarse/all-fine comparisons before coupling pressure. Then repeat live topology, terrain/rigid and the canonical Dawn regression gate. Initial-shape invariance and native-M0 agreement alone are not evolution acceptance.
