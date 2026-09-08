# Current density through a smooth departure map

Status: bounded research recommendation, not production adoption. The strongest next full-fine experiment is a short, explicitly capped map composition whose native amounts are integrated from the mapped current field. This avoids the demonstrated saturation infeasibility of the nonlinear tensor hierarchy. It does **not** yet solve indefinite evolution, map compression, solid contact or field-changing edits.

This assessment reads the [motion authority contract](retained-density-motion-authority-design-2026-09-08.md), [production dependency audit](retained-density-production-cutover-audit-2026-09-08.md), and [latent hierarchy counterexamples](latent-hierarchy-feasibility-review-2026-09-08.md). It introduces no production implementation or GPU result.

## A moving reference is a valid current field

At an accepted anchor epoch, retain a numeric potential/branch field `ψA`. Let `Xn` map current positions back to that epoch. Define

```text
ψn(x) = ψA(Xn(x)),       qn(x) = clamp(.5 − ψn(x)/w, 0, 1)
X(n+1) = Xn ◦ χn,        χn = accepted one-step departure map
M(n+1)(K) = integral over K of qA(X(n+1)(x)) dx.
```

The persistent reference is legitimate because its coordinate map evolves. The failed shipping lift `aK*qseed(x)+bK` keeps the geometry at the same world coordinates; it has no corresponding departure map. Under a translation, the proposed map evaluates upstream geometry in formerly dry supports and changes its normals correctly. Saturation does not erase the latent spatial information.

A C1 diffeomorphism composed with a C1 potential gives a C1 potential; regular zero-set normals remain continuous. C2 maps and potentials are needed for continuous curvature. Explicit sharp branches are composed individually with the same map, retaining their selectors and one-sided gradients. Invertible transport preserves component topology: injection, merging through diffusion/contact, sharpening, and solid displacement are separate physical operations, not consequences of changing the map alone.

If the departure map has unit determinant and the destination partition covers the transported source, change of variables conserves total density integral. Native means must be restrictions of this **same** field, not old CM12 center-interpolation targets. An imperfect determinant, missing coverage, or numerical integration error has its own mass effect. Multiplying density by a Jacobian or moving the interface afterwards is not an incompressible-map substitute.

Characteristic-mapping research evolves an inverse flow map and uses compositions of short-time submaps to represent long deformations. It supports the moving-coordinate formulation; its reported accuracy does not establish an exactly volume-preserving discrete map, finite-history production bound, or our cut-cell integral contract. [Yin, Schneider and Nave, 2023](https://www.math.mcgill.ca/jcnave/papers/CMM_Euler3D_2023.pdf)

## A practical full-fine velocity compiler exists, with conditions

Ordinary componentwise interpolation of discretely projected face data need not be pointwise divergence-free. Local polynomial MAC constructions can preserve divergence and C1 continuity in 3D. [Schroeder, Roy Chowdhury and Shinar, 2022](https://www.sciencedirect.com/science/article/am/pii/S0021999122005629)

Later work distinguishes face-center interpolation from **flux consistency**: the integral of reconstructed normal velocity over a face equals the stored face value times its area. This is the relevant contract if native compiled faces carry fluxes. It also shows that higher accuracy is tied to the discrete divergence stencil; fourth-order reconstruction cannot simply be substituted for a different pressure stencil. [Roy-Chowdhury, Shinar and Schroeder, 2024](https://www.sciencedirect.com/science/article/pii/S0021999124000809)

The following uniform, periodic C2 construction is our deduction from the cardinal spline-chain identity, not an assertion that the cited paper implements this particular prefilter. For centered cardinal B-splines,

```text
d B(p+1)(s)/ds = Bp(s+.5) − Bp(s−.5).
ux(x,y,z) = Σ cx(i+.5,j,k) B4(x/h−i−.5) B3(y/h−j) B3(z/h−k)
```

Use cyclic permutations for uy and uz. The divergence is the common `B3⊗B3⊗B3` interpolation of the discrete coefficient divergence. Thus discretely divergence-free coefficients give a pointwise divergence-free C2 velocity. Each component uses at most 5×4×4=80 coefficients, or 240 scalar fetches for a vector query before reuse. Directly using face fluxes as coefficients gives an approximation, not the supplied face integrals.

Face averaging integrates each transverse B3 into B4; evaluation on the normal face also gives B4. Consequently **every component's face averages are filtered by the same separable kernel**

```text
K = B4(integer offsets −2…2) = [1,76,230,76,1]/384.
U = Kx Ky Kz c.
```

On a periodic uniform grid, prefiltering `c=(Kx Ky Kz)^−1 U` commutes with the discrete divergence. It therefore preserves the divergence constraint while matching the supplied face integrals in exact arithmetic. The one-dimensional Fourier symbol has minimum 5/24 and maximum 1, so the filter is invertible; the worst three-dimensional inverse amplification is `(24/5)^3≈110.6`. This is a global separable banded/FFT solve, not a free local interpolation. Noise amplification, coefficient ranges and velocity approximation error need measurement.

The independently reviewed [periodic C2 MAC CPU oracle](../tools/implicit-density/periodic-c2-mac-oracle.ts) and its [five tests](../tests/periodic-c2-mac-oracle.test.ts) now pass; the owner's measured run took .326 seconds and targeted strict TypeScript checking passed. They verify arbitrary-coefficient divergence commuting, discrete-curl data, physical face averages through independent split Gauss integration, C2 jets, constants, and the Nyquist amplification counterexample. The oracle uses a tiny dense inverse solve with 3–16 cells per periodic axis. It is not a production prefilter, a velocity time integrator, a VEX conversion, or a GPU performance result. Its positive-face convention places cell centers at i+.5 and the x-positive face at i+1; this is the half-cell coordinate shift of the displayed formula.

This construction currently covers periodic, uniform, complete MAC flux data only. VEX stores effective **cell** velocity, so its buffer is not automatically compatible face data. The production audit requires freezing the source generation before gather rewrites effective velocity. Full-fine face coverage, the exact discrete divergence, and the relationship between projected faces and extended liquid/air transport velocity must be checked. Nonzero divergence cannot be silently removed by interpolation. An additional projection, if needed, is a separately specified velocity change with a measured residual. Coarse/fine interfaces and clipped rigid/terrain faces require another compatible construction.

## Smooth divergence-free velocity still needs the right time map

The exact flow of that velocity preserves volume, but generic RK2 does not. The existing saddle negative control gives departure determinant `1+(a dt)^4/4` even for exactly divergence-free velocity. Sampling a map at nodes and interpolating those samples also does not preserve its determinant.

A defensible general integrator decomposes the velocity into two-coordinate Hamiltonian fields and composes symplectic substeps. Such a composition preserves volume in exact arithmetic, while time accuracy and reversibility are separate properties. [Feng and Shang, 1995](https://link.springer.com/article/10.1007/s002110050153)

For periodic MAC coefficients, one possible implementation route is a periodic discrete vector potential for the mean-free velocity, plus its constant harmonic mean. Compatible potential components such as `Az` in `B4(x)B4(y)B3(z)` generate the C2 pair `(∂yAz,−∂xAz,0)`; cyclic components generate the other pairs. Each pair fixes one coordinate. An implicit midpoint pair solve is area-preserving when solved exactly; a symmetric composition of these pair maps and the constant translation supplies a volume-preserving time step. A vector-potential construction must handle its gauge and periodic harmonic part. Vector-potential interpolation has practical precedent, but that alone does not validate the time map. [Chang et al., Curl-Flow project](https://jumyung-jc-chang.com/research/CurlFlowSA22)

This is a candidate route, not implemented here. Bounded nonlinear iterations, uniqueness/conditioning, inverse residuals, determinant errors and derivatives of the **actual numerical map** require admission. Differentiating an ideal implicit equation is not automatically the derivative of an unconverged fixed-iteration solver. A canonical map query must be independent of which support requested it. C1 velocity only supports the corresponding C1 guarantee; the experiment must not call its curvature C2 by implication.

## Bounded history is the remaining representation problem

A two-step experiment can retain both immutable velocity/map descriptions and evaluate their composition directly. It preserves the reference field without density refitting. Retaining another description every step, however, grows memory and query cost. A fixed cap with explicit rejection is an honest bounded experiment, not a completed long-running solver.

The available continuation choices all need another gate:

| Choice | Benefit | Unresolved cost or invariant |
|---|---|---|
| Retain all submaps | Avoids repeatedly reconstructing density | History and evaluation depth grow |
| Interpolate the composed map into a fixed spline grid | Fixed storage and C2 interpolation are possible | Determinant, inverse consistency and thin-scale motion can be lost; map can fold |
| Rebase to a new current density anchor | Resets map depth | Reintroduces bounded conservative field reconstruction, saturation and branch feasibility |
| Compress into a fixed number of volume-preserving factors | Keeps determinant by construction if factors are valid | Approximation/inverse/derivative residuals and finite representational capacity remain; no universal fixed-depth result |

A three-component cubic B-spline map has only 3N f32 coefficients, or 24N bytes for two banks, but that storage figure says nothing about unit determinant after fitting. A C1 shared-Hermite map carries approximately 24 scalars per vertex; two f32 banks cost about 192N bytes. Keeping two generations of three-component spline velocity/potential coefficients costs 24N bytes before source geometry, integration worklists, potentials, solver scratch and receipts. At 196,608 supports that last core is 4.5 MiB. These are storage counts, not performance measurements.

Under arbitrary stretching, a fixed finite representation cannot indefinitely retain arbitrarily thin structures at unchanged error. The map approach postpones and changes the compression problem; it does not abolish it. Live field-changing operations need a valid new anchor or explicit bounded branches as well.

## Proposed next experiment and comparison

Prefer the following short full-fine ladder over expanding the currently infeasible nonlinear 27-DOF hierarchy. Each stage uses density and native amount before the mesh; no visual repair or mean target fitting is part of the experiment.

1. **Two-step translation baseline.** Reuse the actual imposed-flow sphere: 32³, h=.05 m, R=.25 m, velocity (.75,0,0) m/s and dt=1/30 s. Keep a numeric anchor and compose two current departures. Check exact displacement, all half-density crossings, gradients, latent and density values in former saturated supports, complete forward wet coverage, and all native means from mapped-field integrals. The all-fine rule must hold for every allocated owner. This gives the user a direct comparison with the measured failed shipping first update.
2. **Non-affine deformation without map fitting.** Use the independent three-shear map/inverse with the original sphere and six explicit box branches. A fixed sequence of shears supplies exact unit-determinant maps. Check reversal, shared C2 traces where applicable, branch identity and same-field mass, including complete coverage. This establishes more than coherent affine quadrics while keeping composition depth fixed.
3. **Velocity and map compiler gate.** Verify the periodic spline divergence/flux identities, then compare a volume-preserving integrator with an independently refined flow solution and Jacobian. Constants, rotations, a non-affine divergence-free field, face-crossing trajectories and the RK2 saddle negative control must all be represented. State the source-velocity generation and reject unsupported divergence or nonlinear solves.
4. **First actual face-velocity steps.** Only after the compiler gate, use immutable full-fine compiled face data for two steps and derive native density from the mapped field. Report any difference from the existing effective transport velocity, plus mass, centroid, symmetry and zero/gradient errors. Replaying captured face histories is only a field-transport probe. Live coupling additionally requires momentum transported with the same measure, or a separately validated velocity transport contract; dividing the old CM12 momentum numerator by the new density is incompatible. Gamma/sharpening and rigid displacement cannot remain unexplained native amount changes. No assertion of indefinite evolution or completed solid/edit support follows.

| Property | Short current-map experiment | Nonlinear latent 27-DOF hierarchy |
|---|---|---|
| Saturation | Retains mapped latent geometry without a local density fit | Pure face target can be infeasible despite a valid source |
| Continuity | Inherited from one canonical smooth map and anchor | Shared traces possible, but reconstruction jets can conflict |
| Sharp branches | Same map transports existing selectors/one-sided traces | Joint branch reconstruction and integral feasibility remain |
| Conservation | Same mapped field, admitted determinant, complete coverage and accurate integrals | Requires feasible nonlinear constraints and accurate donor integrals |
| Bounded cost | Clear fixed-horizon cap; indefinite compression unresolved | Fixed nominal DOFs, but bounded solve/refinement may reject |
| Closest useful result | Smooth full-fine motion before pressure/mesh complications | Valid 1D experiment plus concrete 3D counterexamples |

The recommendation is a bounded experiment, not a new production promise. Selecting a persistent map representation still requires a demonstrated cap-preserving continuation step; selecting the nonlinear hierarchy requires resolving its existing feasibility counterexample.
