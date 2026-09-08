# Global spline-potential moments: a finite-solution obstruction

Status: independent mathematical negative control. Global moment fitting removes the fixed-trace constraints of the [local latent hierarchy](latent-hierarchy-feasibility-review-2026-09-08.md), but it still cannot guarantee a finite C2 potential for valid saturated input. Do not select it as the general map-reanchoring operation. An interior-moment experiment remains mathematically defensible with explicit rejection and geometric error checks.

## The proposed equations and what they conserve

Let the nonnegative C2 cubic B-splines form a partition of unity on a complete periodic domain. Coefficients may have either sign. With fixed physical width w,

```text
psi_c = sum_i c_i B_i,       q_c = clamp(.5-psi_c/w,0,1)
m_i = integral B_i q_depart
integral B_i q_c = m_i.
```

The departure is the current transported field. Summing the equations conserves its global amount. Native means are integrals of the resulting q_c; these equations do not preserve each transported native-cell amount. Truncating a sparse basis without a complete partition of unity invalidates the global argument. Numerical quadrature and incomplete departure coverage create separate errors.

Define F(s) to be -s below -w/2, `(w/2-s)^2/(2w)` in the ramp, and zero above w/2. Then F'=-q and

```text
D(c) = integral F(psi_c) + c dot m
gradient D = m - integral B q_c
H_ij = (1/w) integral_{0<q_c<1} B_i B_j.
```

D is convex and continuously differentiable. The displayed ordinary Hessian applies away from positive-measure threshold plateaus; those require a generalized derivative. A finite stationary point is equivalent to the bounded quadratic-entropy minimizer

```text
minimize integral (w/2)*(q-.5)^2
subject to 0<=q<=1 and integral B_i q=m_i.
```

The primal is feasible because q_depart is feasible. On a bounded domain it has a unique density minimizer, by weak compactness in L2 and strict convexity. This does **not** prove existence of finite dual coefficients. Entropy-moment theory explicitly separates primal existence, dual attainment, and recovery of the primal from finite multipliers. [Borwein and Lewis, 1993](https://people.orie.cornell.edu/aslewis/publications/93-partially.pdf)

For linearly independent basis functions, a useful sufficient condition is an interior moment vector. In direction v=sum d_i B_i the recession slope is

```text
D_infinity(d) = integral [max(v,0)*q_depart + max(-v,0)*(1-q_depart)].
```

If it is positive for every nonzero d, the finite-dimensional dual is coercive and attains a minimum. In particular a source bounded strictly away from 0 and 1 everywhere satisfies this condition. Pure supports can make the slope zero. Some boundary data still have finite solutions, such as the identically dry field; the counterexample below shows that other valid boundary data do not. This distinction is consistent with constraint qualifications in entropy-like minimization, rather than an application of results about unbounded kinetic domains. [Borwein and Lewis, 1991](https://epubs.siam.org/doi/10.1137/0329017)

## Exact counterexample, including a periodic embedding

First take C2 piecewise cubics on [-1,1] with a single knot at zero. This is a positive B-spline space spanning `P3 + span{x_+^3}`. Set w=1 and

```text
psi_depart(x) = .5-x+x^4
q_depart(x) = 0 for x<=0;  x-x^4 for 0<x<1.
```

The latent source is smooth. The nonnegative spline `(-x)_+^3` has zero target moment, so positivity forces every feasible q to vanish on (-1,0). On (0,1), the remaining constraints are the four P3 moments. Their unique quadratic-entropy minimizer is

```text
p(x) = 1/70 + (5/7)*x + (9/7)*x^2 - 2*x^3.
integral_0^1 x^k * [p-(x-x^4)] = 0,  k=0,1,2,3.
```

Its cubic Bernstein coefficients are `[1/70,53/210,193/210,1/70]`, all strictly between zero and one. Thus the bound constraints do not alter this polynomial minimizer. For any other feasible density, orthogonality gives `||q||^2=||p||^2+||q-p||^2` on this cell. The source has squared-norm excess exactly `1/44100`, and both have amount `3/10`.

The unique minimizer is zero on the left and tends to **1/70** on the right of the knot. It has a jump. Every finite C2 psi gives continuous clamped q, so no finite coefficient vector can solve all the proposed equations. A latent preference among exact solutions cannot repair the absence of an exact solution.

This is also a counterexample with an actual half-density component: multiply the source and p by **27/25**. The source at x=5/8 now exceeds .5, while it remains zero at both endpoints. All scaled source degree-four Bernstein coefficients remain below one (the maximum is 81/100), and the largest scaled coefficient of p is **5211/5250<1**. The moment argument and entropy-minimizer uniqueness are unchanged, with a nonzero jump of 27/1750. The smooth latent source becomes `.5-(27/25)*(x-x^4)` on the wet cell.

This is not an open-boundary artifact. Use eight periodic unit cells and put the same density in (0,1), zero elsewhere. The four active centered cardinal cubic B-splines have restrictions, multiplied by six,

```text
[1,-3,3,-1], [4,0,-6,3], [1,3,3,-3], [0,0,0,1]
```

in ascending powers of x. Their determinant is 108, so their restrictions span P3. Every other basis function has zero target moment; their positive supports cover all seven other open cells and force q=0 there. The same unique discontinuous minimizer follows.

An explicit periodic C2 source potential exists: let chi equal one on [0,1], taper to zero on [-.5,0] and [1,1.5] with quintic smoothstep, and be zero elsewhere in the period. Set `psi=.5-[chi*(x-x^4+1)-1]`. Its density is exactly the specified source; the latent value and first two derivatives agree at every join. Extruding in y,z gives the same obstruction in the tensor 3D space. Averaging a feasible density over y,z preserves the one-dimensional moments and cannot increase quadratic entropy; strict convexity forces the minimizer to be independent of y,z.

The [five independent CPU tests](../tests/galerkin-potential-feasibility.test.ts) check the moment equalities and norm gap with exact rational arithmetic, the periodic support/rank identities, the explicit C2 source, the incompatible minimizer traces, and the scaled half-density component. No candidate nonlinear solver, production code, GPU result, or tolerance relaxation is involved.

## Consequences for a bounded reanchor

Interior data remove this specific existence obstruction, but moment agreement alone does not imply geometric accuracy. On a uniform periodic grid, a smooth oscillation at one full cycle per cell has zero moments against every cardinal cubic B-spline. Consequently a bounded source `.5+epsilon*sin(2*pi*x/h)` has the same moments as constant .5; the entropy minimizer erases that unresolved shape. Exact recovery holds when the departed potential itself lies in the destination spline space. Arbitrary noninteger translations and nonlinear deformations do not preserve that fixed-knot space.

Density dimensional reduction is sound for a uniform tensor space with exact integration: uniqueness and averaging prevent invented transverse density in an x-only problem. Saturated **latent** coefficients are not unique; any preference must use consistent tensor functionals and be restricted to the exact solution set. Adding a small quadratic penalty to D generally changes the required moments. One smooth potential also cannot promise regular sharp edges; explicit branch selectors and branch-coupled mass remain separate work.

Storage is attractive: one cubic scalar coefficient per grid site, or 8N bytes for two f32 banks before all scratch and geometry. However each 3D sample touches up to 64 coefficients; an assembled ramp-weighted Hessian has up to 7^3 overlapping neighbors per row. At 196,608 sites, that dense stencil alone is about 257 MiB in f32. Matrix-free products save this storage but require repeated ramp integration and global iterations. Small ramp support causes ill-conditioning or exact nullspaces. Clipped tensor cubics and mapped departure fields require resolved integration; a nominal four-point Gauss rule is not exact through unknown saturation cuts. These are operation/storage counts, not benchmarks.

Conservative characteristic Galerkin remapping provides a useful weak conservation precedent. Its discontinuous polynomial space and additional bound-preserving filter do not supply a shared C2 potential or resolve this finite-multiplier obstruction. [Cai, Guo and Qiu](https://arxiv.org/abs/1612.06977)

The next decision should therefore remain with the bounded current-map experiment or a separately justified evolution formulation. Do not invest in a general 3D global moment reanchor merely because the scalar dual is convex. Any optional interior-only solver must reject nonattainment, retain the accepted field on failure, and report independent density, zero-set, gradient, coefficient-size and integration errors alongside the moment residual.
