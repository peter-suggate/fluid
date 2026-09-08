# Current latent potential: a bounded 1D conservative experiment

Status: six CPU research tests pass. This is not a production density representation, a 3D solution, or a surface correction. It addresses the saturation failure recorded in the [tensor density decision](current-density-tensor-decision-2026-09-08.md) by defining density from a transported current latent potential throughout dry, partial and full supports.

## Definition and scalar feasibility

Let `q(x)=clamp(.5−ψ(x)/w,0,1)`, with a fixed positive physical width w. On a support of width h, share endpoint ψ and physical derivative ψ′ with its neighbors. Let H(t) be their cubic Hermite interpolant and define

```text
ψδ(t) = H(t) + δ β(t),     β(t) = 30 t²(1−t)²,     0≤t≤1.
F(δ)  = ∫₀¹ clamp(.5−ψδ(t)/w,0,1) dt.
```

The bubble and its first derivative vanish on both faces, so any finite δ preserves the shared C1 **potential** traces. Density is bounded by its definition. This does not clip a previously mass-matched density polynomial: the nonlinear constraint integrates the already defined clamped field, and δ is chosen to match the departure integral of that same current field. At its half-density interface, `q=.5` is exactly equivalent to `ψ=0`; there is no endpoint inversion shortcut or separate published zero-set formula. Density itself is generally only C0 at saturation boundaries; the latent potential and regular half-density interface are the smooth objects.

The following feasibility facts are direct mathematical deductions, not borrowed claims about THINC:

* β is positive in the open support. F is continuous and nonincreasing, with limits F(−∞)=1 and F(+∞)=0 by dominated convergence. If 0<F<1, a nonempty interval is unsaturated, so `F′(δ)=−∫unsaturated β/w < 0`. Every strictly partial target therefore has a unique finite solution in exact arithmetic.
* A finite **zero** target requires H(0),H(1)≥w/2. If an endpoint equals w/2, its inward derivative must not point below the threshold: H′(0)≥0 and H′(1)≤0 at equal endpoints. These conditions are also sufficient for a finite bubble: away from the endpoints β>0, and near an equal endpoint the potentially adverse residual is at most quadratic and can be dominated by δβ. A negative inward linear term cannot be corrected by any finite quadratic-vanishing bubble.
* A finite **one** target has the reversed conditions: endpoint potentials≤−w/2; at equality H′(0)≤0 and H′(1)≥0. Incompatible endpoint values or slopes require rejection. Taking δ to infinity or rounding a near-pure target to pure would conceal this failure.
* Pure-cell mass does not identify latent geometry. Many δ values give exactly the same zero or unit density. The reference transports the **current** midpoint potential and uses `(ψdeparture(midpoint)−H(.5))/β(.5)` as the preferred δ. It retains that value if it already satisfies the mass constraint. Otherwise it finds a finite admissible solution and checks its geometric residual. This is additional transported latent information, not information inferred from M0.

The last condition matters even for a globally quartic potential. Its current fourth-order term remains meaningful in fully saturated supports and must survive until material reaches them. Shared endpoint jets plus pure-cell mass alone cannot recover it. No function in the transport path reads an initial primitive or initial coefficient field.

## Computation and what is rejected

The CPU reference recursively isolates polynomial derivative roots, partitions the quartic into monotone pieces, then bisects crossings of ψ=±w/2. On each resulting interval, density is either zero, one, or a polynomial with an analytic primitive. Thus no fixed sampling grid can miss a thin ramp merely because no quadrature node lands in it. Root isolation and primitive evaluation remain float64 numerical operations, not certified interval arithmetic. Near multiple roots, tiny widths, very large coefficients, and almost-pure targets require stronger conditioning/error analysis before GPU or production use.

A bounded bracket expansion and bisection solve the nonlinear mass equation. Partial-cell mean residual must be at most 2e-13 in these tests. A classified entirely full departure support uses its exact geometric unit measure, avoiding subtraction roundoff in `(departure+1)−departure`; mixed integrals are never clamped. Pure targets must produce exactly zero or one through the defined-field integration, rather than merely fall within the partial-cell residual tolerance.

Every translation evaluates the current endpoint jets and current midpoint, and integrates the current density over the complete departure interval. It then compares the reconstructed potential with **every old polynomial piece** intersecting that interval. Extrema of the polynomial difference and its derivative provide numerical maximum potential/derivative residuals. The caller must supply a potential-error budget. Exceeding it, exhausting the nonlinear solve, infeasible pure data, or missing departure support rejects the candidate while preserving the source arrays.

Shared jets and matched mass do not guarantee faithful geometry. The residual gate is therefore part of this experiment. Exact global planes, quadratics and quartics are retained under translation; general piecewise quartics have moving joins and are reconstructed approximately. The midpoint preference does not change this finite-space limitation.

## Evidence

Run `node --import tsx --test tests/latent-conservative-line-oracle.test.ts`. The six tests finish in about .5 seconds on this host. Independent analytic threshold crossings and primitives check the clamped plane, quadratic and quartic cases; the periodic sine integral uses analytic arcsine crossings, not the implementation's polynomial root finder.

* The plane crosses into formerly saturated support and preserves its zero, slope and departure-region mass to about 1e-12. Its bubble remains zero to roundoff.
* Global quadratic and quartic profiles preserve both zero crossings, derivatives and the mass of their analytically contained supports. The quartic bubble remains nonzero even inside pure supports, demonstrating why the transported latent preference is needed.
* Pure targets with incompatible endpoint data reject. Different preferred potentials can legitimately have identical pure-cell mass. A deliberately tiny geometric budget and absent departure supports both reject without modifying the source.
* The saturated nonquadratic fixture `ψ=.08 sin(2πx)`, w=.05, translates by h/3 for 3N steps, completing a periodic orbit. Shared potential value/derivative traces and mass are checked each step; q stays within [0,1] by definition.

| Cells | Orbit maximum ψ error | Maximum q error | Maximum ψ′ error | Zero-position error | Maximum mass error |
|---|---:|---:|---:|---:|---:|
| 8 | 5.108e-5 | 7.626e-4 | 8.331e-4 | 3.830e-6 | 1.89e-15 |
| 16 | 2.170e-6 | 3.707e-5 | 5.356e-5 | 4.664e-7 | 5.33e-14 |
| 32 | 8.497e-8 | 1.618e-6 | 4.032e-6 | 5.005e-9 | 1.93e-14 |

Maximum per-step potential reconstruction residuals are 8.78e-6, 2.97e-7 and 6.35e-9 respectively. These are measured convergence results for prescribed translation, not evidence of general fluid transport, pressure coupling or arbitrary deformation.

Only the periodic experiment is a closed-domain global conservation test. The finite plane/quadratic/quartic examples crop boundary cells so every requested departure interval exists; they are open/cropped analyses. For the contained polynomial examples, an independent analytic support bound explains why the crop discards no liquid. They do not establish a general complete transport transaction. A 3D implementation still needs forward coverage of every potentially wet source support before publication.

## Research context and unresolved 3D constraints

THINC/QQ represents a cell interface with a quadratic implicit surface and uses quadrature to constrain volume and evaluate fluxes. That is relevant precedent for nonlinear integration of an implicit reconstruction, but its cellwise construction is not a proof of shared C1 traces. [Xie and Xiao, 2017](https://www.sciencedirect.com/science/article/abs/pii/S0021999117305995)

THINC-scaling couples a polynomial level-set representation to a bounded hyperbolic-tangent field, using conservation information with transported geometric information. Its construction motivates examining the two aspects together. This experiment instead uses the existing finite-width clamp, canonical shared endpoint jets and a bubble that vanishes with its first derivative on faces. It does not adopt a cellwise constant offset as a smooth solution, nor claim the published multidimensional method is implemented here. [Kumar et al., 2021](https://arxiv.org/html/2103.09541)

One nonlinear volume constraint cannot be naively extended to 3D with `δβ(x)β(y)β(z)`. It would reproduce the earlier transverse-artifact problem. Even full tensor **linear latent-potential** edge/face moments plus only a nonlinear density-volume constraint can fail dimensional reduction: the edge quartic fixed by ∫ψ need not have the transported ∫q, forcing a volume-only correction that varies in transverse directions. Compatible nonlinear density constraints must extend through canonical shared edges and faces, along with the latent data needed where saturation makes those constraints non-identifying. Coupled feasibility, shared normal-derivative traces and robust integration are unproved. This note does not select a 27-scalar nonlinear 3D ABI.

Intentional sharp min/max branches still need explicit selectors and one-sided traces; a single C1 latent polynomial cannot exactly represent their cusps. General deformation additionally requires a controlled departure map and mixed-jet/partial-integral transport. Sparse support refinement, clipped solids, signed frontier coverage and GPU transactional admission remain later gates. The next justified step is independent 2D/3D dimensional-reduction and shared-edge/face feasibility analysis, followed by a bounded GPU scalar solve only after its numerical error contract is established.
