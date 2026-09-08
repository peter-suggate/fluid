# Current-potential weak evolution: bounded one-dimensional evidence

Status: a six-test CPU oracle passes in 0.862 seconds, with targeted strict TypeScript checking. This is an intrinsic discretization of density advection in a current C2 potential. It is not a production change, a GPU result, a complete saturated carrier, or an indefinite map-reanchoring solution.

The [global moment-projection obstruction](galerkin-potential-feasibility-2026-09-08.md) prevents selecting arbitrary finite-step departure moments as always realizable by a smooth potential. The present experiment imposes a weak evolution equation instead. No destination cell means, old seed, surface positions, or renderer corrections enter its update.

## Formulation and admission

For periodic cubic B-splines and a fixed physical width,

```text
psi_n(x) = sum c_n,i B_i(x),       q_n = clamp(.5-psi_n/w,0,1)
psi_s = (1-s)*psi_n + s*psi_(n+1), s in [0,1]
q_bar(x) = integral_0^1 clamp(.5-psi_s(x)/w,0,1) ds
R_i = integral B_i*(q_(n+1)-q_n) - dt*integral B_i' * u * q_bar = 0.
```

The velocity is uniform in this oracle. Summing the equations uses `sum B_i=1` and `sum B_i'=0` and yields global amount conservation, including when saturation boundaries cross the coefficient segment. The implementation uses the integration-by-parts form above so the same quadrature locations preserve this partition identity up to floating arithmetic. Independent physical integration separately checks the actual field. A solved residual bounds the algebraic contribution by `N*max(abs(R_i))`; quadrature error is additional and its estimator is not a certified interval bound.

The segment average is analytic: clamp of an affine function of s is piecewise affine. Its derivative with respect to the destination potential is `-(U^2-L^2)/(2w)`, where [L,U] is the segment's unsaturated interval. This supplies the nonlinear endpoint Jacobian without finite differencing.

The [oracle](../tools/implicit-density/weak-segment-density-line-oracle.ts) accepts 8–16 periodic coefficients, at most half-cell displacement, and a numerically positive-definite ramp Gram matrix. It isolates every old/new cubic saturation root and every coefficient-difference root in each cell, then uses bounded adaptive Gauss4/Gauss8 integration on those pieces. A fixed stencil is never used to declare an unexamined cell dry. Default caps are eight Newton updates, eight line-search halvings, ten spatial refinement levels and 100,000 quadrature evaluations per assembly. The Newton and Gram condition estimates have explicit limits. Any failed integration or admission aborts the candidate, leaving the source unchanged.

Finite-step local existence is defensible where the ramp Gram matrix is nonsingular: at dt=0 the residual Jacobian is its negative divided by w. With regular saturation cuts, the implicit-function argument therefore supplies a nearby solution for sufficiently small steps. This is not a proof of finite-step solvability through changing nullspaces or large deformation. The tested weakly saturated wave was chosen to retain positive definiteness, which is measured rather than assumed.

## Why exact threshold plateaus are rejected

For zero-measure threshold sets, the formal semidiscrete equation is

```text
M_ij = integral_ramp B_i B_j
M*c_dot = -integral_ramp B_i*u*psi_x.
```

The right side lies in the range of M: a null vector represents a spline that vanishes on the ramp. With a regular solution, testing by one conserves amount; testing by psi conserves `integral q^2` through the primitive `G(psi)=w^2*(q-.5)^2/2`, whose derivative in the ramp is psi.

However, the strict-ramp derivative is not the true directional derivative on a threshold plateau. A concrete periodic example is `psi=w/2+A*B0`, A>0. Its density is identically zero and the strict-ramp matrix is zero. The full-latent L2 advection preference `P_V(-u*psi_x)` is a nonzero odd spline. It cannot remain supported on B0's four cells: the C2 cubic spline space with exactly that support is the one-dimensional even span of B0. It therefore has a negative lobe on a dry threshold plateau, producing positive first-order density there. Choosing that preference freely from the matrix nullspace would contradict the claimed zero amount derivative.

The finite-segment amount identity still holds, but solving its nonsmooth constraints with a suitable latent preference is separate work. The oracle explicitly rejects exact threshold plateaus and singular/ill-conditioned Gram matrices. It does not introduce a pseudoinverse cutoff, a one-sided tangent-cone solver, or a general saturated-nullspace rule.

## Measured tests and independent references

The [six tests](../tests/weak-segment-density-line-oracle.test.ts) use a global cardinal-basis evaluator independent of the update's local power-polynomial evaluator. Physical amount and squared density are integrated with a separate Gauss4 rule after independently bracketing the fixtures' simple saturation roots. Tests also inspect both half-density crossings, potential derivatives, every C2 knot trace, reversal, source immutability, physical-length scaling, and rejection of unsupported conditioning, plateau, quadrature and Newton cases.

The initial wave is a nodally interpolated sine potential on a unit periodic domain, with phase .13, w=.05 and u=1. Initial interpolation error is included when comparing to the analytic sine and separated when comparing to the translated current spline. This is a prescribed-flow mathematical test, not shipping freefall or a sphere capture.

| Gate | Measured result |
|---|---|
| Unsaturated amplitude .015, N=16, T=.125, 4/8/16 steps | Maximum potential error against translated current field: 3.7630e-5, 9.4483e-6, 2.3647e-6; approximately fourfold decrease |
| Weakly saturated amplitude .026, T=.125, 128 steps, N=8→16 | Analytic potential error 3.0800e-5→1.8830e-6 |
| Same spatial refinement, half-density positions | Maximum zero error 2.2457e-5→8.5275e-7 |
| Same runs, gradients against translated current field | Maximum derivative error 7.6469e-5→4.7720e-5; weaker improvement than potential/zero errors |
| Same runs, Gram/Newton admission | Maximum Gram condition 47.80 / 188.03; at most three Newton updates per fine step |
| Saturated temporal refinement, N=16 | Coefficient difference from a 128-step trajectory: 2.9280e-4, 7.4895e-5, 1.7968e-5 for 8/16/32 steps; this is a temporal reference for the same weak PDE, not an analytic transport oracle |
| Symmetric physical amount | Maximum endpoint drift 1.11e-16 |
| Asymmetric clipped wave, added potential offset .0007, 32 steps | Initial amount .48842322263178756; maximum independently integrated amount drift 9.124e-12 |
| Squared density | Unsaturated drift at roundoff; saturated N=16 fine-run change -2.2723e-11; the larger dt=1/64 single step changes it by -3.7303e-8 |

The asymmetric amount gate is necessary: the symmetric sine alone could conceal a conservation defect through half-period antisymmetry. The nonzero squared-density drift is deliberately tested and reported.

## What the finite-step scheme does not establish

The segment scheme is symmetric under endpoint exchange and dt reversal. At regular cuts it is consistent with the semidiscrete weak equation and shows second-order temporal convergence here. It does not automatically inherit the standard average-vector-field energy theorem, which assumes the appropriate constant symplectic structure. The ramp-weighted nonlinear system requires its own invariant analysis. [Celledoni et al., author paper](https://tur-www1.massey.ac.nz/~rmclachl/pdenew.pdf)

Planes under constant translation satisfy the segment equation pointwise. A translated quadratic generally does not when saturation crosses the segment: for Hessian H, the exact translated endpoints give

```text
delta_psi + dt*u dot grad(psi_s) = dt^2*(u^T H u)*(.5-s).
```

The unsaturated s interval need not be centered at .5, so its weighted integral need not vanish. No exact finite-step sphere-translation claim follows. General noninteger translations of fixed-knot splines require spatial refinement as well. The measured gradient improvement is modest and must not be described as demonstrated high-order normal transport.

Finite coefficients guarantee C2 **potential** traces. The clamped density remains only C0 at saturation thresholds; a regular half-density surface inherits C2 geometry only while its potential gradient stays nonzero. Intentional sharp branches, component merging, strong saturation, sparse support changes, curved/clipped physical integration, variable divergence-free flow, rigid interaction, momentum coupling, and live edits remain unimplemented. The current oracle contains dense matrices only because N≤16; its subsecond CPU result is not evidence of production GPU cost.

The next justified ladder is an independent review, then saturation/rank-boundary feasibility and accurate curved-feature temporal tests. Only after those gates should a bounded 3D implementation be considered. Preserve the accepted current field and explicitly reject unsupported cases; passing global amount alone is insufficient evidence of the smooth full-fine carrier the user requires.
