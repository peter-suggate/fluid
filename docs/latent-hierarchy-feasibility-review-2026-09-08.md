# Latent tensor hierarchy: feasibility review

This is a mathematical review of a proposed research representation, not a
production design approval. The independent executable counterexamples are in
`tests/latent-hierarchy-feasibility.test.ts`. No production density, surface,
mass target, or GPU kernel changes are part of this review.

## Proposed representation and the part that works

Let the current latent potential define density by
`q=clamp(1/2-psi/w,0,1)`, with constant positive `w`. Use the tensor product of
the four cubic Hermite endpoint bases and the normalized quartic bubble
`beta(t)=30 t²(1-t)²`. There are 125 coefficients in a local cell. Sharing them
by geometric entities gives approximately 27 scalar coefficients per cell in
a large regular grid: 8 per vertex, 4 on each of 3 edge orientations, 2 on
each of 3 face orientations, and 1 in each volume.

The proposed hierarchy sets vertex mixed jets, then edge coefficients, then
face coefficients, then the volume bubble. Scalar edge/face/volume constraints
match current-departure **density** integrals. Other edge and face coefficients
match current-departure **latent derivative** integrals. Shared traces and
shared normal-derivative traces can give C1 continuity of the latent field.

If both the departure field and the reconstruction stay entirely inside the
unsaturated ramp, every scalar density moment is equivalent to a linear latent
moment. With the same normalized functionals, all 125 local tensor moments
then reduce to the linear tensor Hermite-plus-integral reconstruction. This
does not hold merely because the mean lies strictly between zero and one:
there must be no saturated subregion in either field. Constant `w` is also
part of the reduction.

For a fixed continuous base polynomial `P` and a bubble `b` that is positive
in the entity interior, define

`F(delta) = integral clamp(1/2-(P+delta*b)/w,0,1)`.

This function is continuous and nonincreasing, approaches the entity's full
measure and zero at opposite infinities, and is strictly decreasing whenever
its value is strictly partial. Its derivative, where differentiable, is
`-integral_{0<q<1} b/w`. Therefore a strictly partial target has a unique finite
solution in exact arithmetic. A bounded implementation still needs a finite
bracket limit, quadrature and solve residuals, and a latent geometry-error
admission check: coefficients can grow without bound near pure targets.

## Blocking example: an exactly dry face has no finite solution

On the unit square, take the **current** field

`psi(x,y) = w/2 + y*f(x)`,

where `S(t)=(1-t)^3(1+3t)` and

`f(x)=S(2x)` for `x<=1/2`, otherwise `f(x)=S(2-2x)`.

`S` is nonnegative on the unit interval. The two quartic pieces of `f` meet
with matching value, first derivative, and second derivative, so `f` is C2.
It has endpoint values 1, endpoint derivatives 0, midpoint value 0, and
integral `2/5`. Consequently `psi>=w/2` and the current density is identically
zero on the entire face. A departure edge crossing a donor seam can see these
two polynomial pieces; the example does not require a nonpolynomial oracle.

The lower edge's scalar latent trace is exactly `w/2`, including its preferred
midpoint and density integral. Its normal derivative is `f(x)`. The proposed
normal-derivative line moment gives the unique quartic reconstruction

`N(x)=1-(3/5)*beta(x)`.

But `N(1/2)=-1/8`. At the upper edge a valid dry, midpoint-preferred scalar
trace is `w/2+T(x)`, where `T(x)=1-(8/15)*beta(x)>=0`. Its reconstructed normal
derivative is also `N`. The resulting tensor Hermite face, including an
arbitrary finite scalar bubble, has offset

`P_delta(x,y)-w/2 = y*N(x) + (3y²-2y³)*(T(x)-N(x)) + delta*beta(x)*beta(y)`.

At `x=1/2`, this is `-y/8+O(y²)`. The face bubble vanishes to second order at
the lower edge and cannot change the negative inward linear term. Thus no
finite `delta` gives zero density throughout the face or a zero face integral.

There is a positive-area lower bound, not just one wet ray. On
`x in [2/5,3/5]`, `N(x)<=-a`, where `a=23/625`. For `0<=y<=1`, the remaining
terms are bounded above by `C*y²`, with
`C=3/8+(225/4)*abs(delta)`. Set

`epsilon=min(1/2, a/(2C), w/a)>0`.

Then `q(x,y)>=a*y/(2w)` on that strip for `0<y<epsilon`, so its density integral
is at least `a*epsilon²/(20w)>0`. Increasing the bracket until the computed
integral rounds to zero would conceal the infeasibility; it would not solve it.

More generally a dry target requires a finite upper bound on
`sup_interior (w/2-P)/b`. The boundary values and inward derivatives alone can
already make this ratio unbounded. Higher-codimension corner and edge jets
also matter because a tensor bubble vanishes in several coordinates at once.
The analogous obstruction exists for full targets after reversing the sign.

This is an admission failure. It calls for rejection, a bounded geometric
refinement attempt, or a separately proved joint latent-trace construction.
It does **not** authorize changing the target density, clipping a matched
density polynomial, or repairing the extracted surface. There is no claim
that a finite number of dyadic refinements always cures such a conflict.

## Saturated preferences must respect dimensional reduction

A pure density target underdetermines latent information, but different
preferred linear functionals cannot be combined indiscriminately. For the
strictly dry field `psi(x,y)=2+f(x)`, `w=1`, a midpoint-preferred edge has
`E(x)=3-(8/15)*beta(x)` and mean `37/15`. The current latent mean is `12/5`.
A face-mean preference would add `-(1/15)*beta(x)*beta(y)`, changing the center
by `-15/64` while its density stays identically zero. The source had no
transverse dependence; the reconstruction has invented one.

The midpoint choice remains a legitimate explicitly limited 1D experiment.
A tensor extension needs compatible tensor-product latent functionals or an
explicit inactive-dimension invariant. Even choosing latent means everywhere
does not remove the pure-face feasibility obstruction above: when a preferred
mean conflicts with exact pure density, both constraints cannot be assumed
simultaneously satisfiable.

## Mixed jets and smoothness need stronger scope statements

A merely C1 source does not supply every mixed endpoint derivative required by
an arbitrary departure map. For example, let source `psi(X,Y)=X²` for `X>=0`
and `2X²` otherwise. It is C1 across `X=0`. Under the rotation
`X=(x+y)/sqrt(2)`, the two one-sided values of `d_y d_x psi` at the seam are 1
and 2. There is no unique transported mixed jet at that point. Selecting a
donor side is a reconstruction convention with error to assess, rather than
an exact derivative of the current field.

For a nonlinear departure map `chi`, even away from seams, mixed derivatives
need the full chain rule. For example,

`d_ij(psi o chi) = H_psi(chi_i,chi_j) + grad_psi dot chi_ij`.

The third mixed derivative also contains the third derivative of `psi`, three
Hessian/map-second-derivative terms, and `grad_psi dot chi_ijk`. Transporting
only a gradient or multiplying stored axis-mixed jets by a Jacobian does not
provide these quantities. A constant affine map removes map derivatives, but
still mixes source pure and mixed second/third derivatives.

Derivative line and face moments must be integrals of the **pullback
derivative in destination coordinates**, over destination parameters. A
rotated or sheared donor line is not a native axis line. Source arclength,
area factors, and transformed derivative directions cannot be omitted.

C1 latent continuity gives continuous normals only on regular zero sets where
`|grad psi|>0`. It does not guarantee continuous curvature: Hessians may jump
between cells. Nor is the clamped density globally C1 at its saturation
thresholds. The interface at `psi=0`, the saturation boundaries, and the
claimed order of smoothness need distinct, explicit acceptance conditions.

## Explicit branches do not remove the error contract

For a sharp union represented by `min_j psi_j`, a common bubble gives
`min_j(psi_j+delta*b)=min_j(psi_j)+delta*b`. It preserves every branch
difference and therefore the branch-selection boundary. It does not preserve
the union's zero set: the common shift moves it, and its intersection with a
branch-selection boundary can move as well. A mass constraint alone supplies
no bound on this displacement or on a change in interface topology.

Every branch needs shared traces and its own transported derivatives. A
derivative of the minimum is not defined at a sharp junction, so a smooth-jet
update cannot silently stand in for branch data. Any bound on a zero-set move
needs at least a latent residual and a lower bound on the relevant gradient
along the compared interface; normals additionally need a derivative-residual
bound. A bounded branch count also requires an explicit overflow policy and
transaction rejection or independently validated refinement. It must not
grow an unbounded tree of remap fragments across generations.

## Conservation and bounded GPU cost

Exact current-departure density integrals over a complete destination
partition conserve the source integral for a volume-preserving map. The
volume constraints can transfer that conservation to a feasible reconstructed
field. Edge and face moments establish reconstruction information; they do
not independently establish global conservation. All levels must sample the
same accepted current generation, and any failed level must reject the whole
candidate publication.

Numerically, source quadrature error, reconstruction solve residual, final
integration error, map/Jacobian error, and missing-coverage error have separate
effects. Matching a sampled target precisely does not prove that target is
accurate. Tiny wet components must not be discarded because quadrature
returns zero. The target comes from the current mapped field, not an older
coarse CM12 center-transfer amount that can conflict with the geometry.

Storage at 27 f32 coefficients per regular cell is 108 bytes per bank, or
216 bytes for two banks, before worklists, temporary moments, receipts, and
donor indices. This is 6.75 MiB for two banks at 32 cubed, 54 MiB at 64 cubed,
and 432 MiB at 128 cubed. A band of 100,000 cells uses about 20.6 MiB for the
two coefficient banks. Sparse band allocation is important at larger scales.

There are about 7 nonlinear scalar solves per cell: 3 edges, 3 faces, and 1
volume. The other 20 coefficients use linear latent functionals or endpoint
jets. The nonlinear dependencies require at least vertex, edge, face, and
volume stages. Per-evaluation 8-point tensor Gauss would cost 8 samples per
edge, 64 per face, and 512 per volume: 728 density evaluations per cell for
one simultaneous evaluation of those seven scalar constraints. Twenty trial
coefficients would make 14,560 evaluations per cell, before donor integration,
adaptive refinement, or derivative moments. At 32 cubed this is about
477 million evaluations, each involving a tensor polynomial or donor lookup.
These are operation counts, not measured GPU timings.

A candidate tensor polynomial restricted to a destination coordinate axis has
degree four. It allows exact piecewise integration after isolating up to eight
crossings with the two clamp thresholds. This can remove one quadrature
dimension from a candidate mass evaluation. However, a general affine
departure line through a source tensor polynomial can have degree twelve,
with up to 24 threshold roots per donor piece. Donor seams add pieces. It
cannot be treated as a source axis quartic unless the map actually preserves
that axis structure. Root isolation must remain bounded and report unresolved
cases. Each candidate coefficient changes its crossings; the previous
iteration's partition cannot simply be reused without validation. The full
tensor has 125 local coefficients, so loading it anew for every sample would
cost 500 bytes before any arithmetic. Workgroup reuse and factorized
evaluation are necessary considerations, not optional polish.

Useful accelerations require validation: analytic unsaturated solves; a
certified saturated range that accepts a compatible preferred coefficient;
range-based empty/full subtiles; safeguarded monotone Newton steps using the
partial-region bubble integral; and bounded root/subdivision work with an
error receipt. None removes the boundary feasibility or mixed-jet issues.

The next defensible scope is the 1D experiment with explicit shape residuals,
plus these negative controls for any later tensor extension. A general 3D
conservative, coherent and bounded-cost construction remains unproved.
