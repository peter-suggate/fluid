# 2D transport work-reduction exploration

Date: 2026-09-12.

This note asks a narrower question than the large-remap plan: now that the
visible surface is a shared reconstructed-distance field (RDF), what can give a
similar moving-liquid result with much less transport work in two dimensions?
It separates changes that retain a conservative liquid-volume authority from
approximations that retain only the visual effect.

## What the current method actually spends work on

Liquid amount is still the extensive cell volume `V`; open capacity is `K`, and
the accepted invariant is `0 <= V <= K`. The RDF is derived at publication time
from accepted VOF fractions and volume-matching PLIC planes. It supplies one
shared scalar value at a topology vertex and therefore a watertight C0 contour,
but it does not reproduce every cell volume. It is neither an advected mass
field nor the current transport authority.

Transport reconstructs a PLIC plane from the current `V/K` at the start of each
volume microstep. On a certified uniform 2D stencil it evaluates six
ELVIRA-style height candidates; elsewhere it uses the least-squares fallback.
Each physical subface then owns one low flux, one PLIC high flux, and one final
limited flux. The high flux is already an analytic planar calculation: the code
evaluates the exact fraction of a swept axis-aligned rectangle under the donor
line. The 2D volume-to-offset inverse is analytic as well. There is no general
polygon clipper in this hot face calculation.

This matters for prioritization. Replacing the planar integral with another
analytic planar formula cannot remove a different class of work; it is already
the implementation. An independent RDF-primary or algebraic high flux could
remove one PLIC reconstruction and face pass per microstep while leaving the
current low-flux feasibility iteration intact. Merely consuming today's RDF
does not remove PLIC: that RDF is itself derived from the frame-tail PLIC state.

That iteration is the larger structural target in one measured difficult case.
The ignored historical artifact
`artifacts/adaptive-volume-copy/dawn-volume-chunk8-mini64-debug.json` (modified
2026-09-11 21:43 NZ, dirty source `80f8615...`, three samples, no warmup, final
QA disabled) recorded 12 microsteps, 821 limiter passes, a maximum of 74 passes
in one microstep, and 180,007 compiled subfaces. Its `limitedFaceCount` of
767,502 is the count of faces whose low flux changed, accumulated across apply
passes; it is not a count of every executed face operation. A newer unqualified
chunk-size diagnostic at
`artifacts/dawn-regression/chunk8-vs-chunk32-mini64-2026-09-12.json` recorded
649 passes over ten microsteps. Neither is a current baseline. Together they
are structural evidence that repeated limiter work exists. PLIC reconstruction
and the high-flux face calculation run once per microstep; every additional
static limiter pass performs two dense cell scans plus scalar control work. The
historical transport timestamp includes continuation latency, so it cannot
attribute elapsed time to those kernels or support a speedup prediction.

Do not use the Advance Lab work strip as measured attribution. Its own header
marks it as an analytical model rather than a receipt. The current view passes
`maxVelocity` where the model names a CFL input and uses a compile-time limiter
pass count rather than the device's completed-pass receipts. Values such as nine
microsteps in a stage card, 256 packets, 71,900 workgroups, 4,365 encoded
dispatches or 96.5% of modeled work are useful for explaining the encoded
structure, but they are recomputed estimates rather than elapsed-time shares.
The live header can separately show the slice's actual microstep count. The
artifact limiter counters above are device receipts, but neither source
currently separates first-pass PLIC/face elapsed time from repeated limiter
elapsed time.

RDF cost is not free merely because publication already builds it. The CPU
measurements in `docs/ADVANCE_SLICE_SURFACE_RECONSTRUCTION.md` recorded about
10.1 ms on the native 64 by 48 scene and 29.5 ms on the 128 by 128 Figure 3
scene when selected. Its separate historical mini32 PLIC/RDF A/B under
`artifacts/sparse-cm12-rdf-presentation-performance-final/` recorded a 1.442 ms
presentation-stage median difference and a 0.983 ms whole-advance median
difference on that source snapshot. These report values use different CPU/GPU
implementations, stages and workloads; they cannot be compared with each other
or subtracted from transport. They establish no current RDF cost. Moving RDF
construction earlier or refreshing it inside microsteps needs a new matched
measurement. The accepted frame-tail RDF is fresh at the next frame start only
when topology and live edits have refreshed the same accepted generation; it
becomes lagged after the first volume microstep.

## The low-order feasibility equation

Let `q_f` be one signed swept open-volume amount on a shared face, and let
`s_if` be `+1` when its canonical orientation enters cell `i` and `-1` when it
leaves. The donor-upwind liquid flux is

```text
l_f = q_f alpha_upwind(f),       alpha_i = V_i / K_i.
```

A one-pass low update is bounded when two separate conditions hold:

```text
sum(outgoing |q_f|) <= K_i^n
K_i^(n+1) - K_i^n = dt S_i + sum_f s_if q_f.
```

The first is the transport CFL condition. The second is the discrete capacity,
source and geometric-conservation-law (GCL) balance. Positivity follows from
the donor CFL. Applying the same update to air amount `K - V` gives the upper
bound. Once this low state is bounded, the existing shared-face FCT operation
can limit `high - low` in one pass without changing global liquid volume.

The existing pressure RHS already contains face divergence, `-capacityRate`
and `+sourceRate`, so it approximately enforces this equation on pressure
members. It does not enforce it on the whole swept transport support. Membership
is selected from the half-full isovalue, previously submerged cells, predicted
moving-solid filling, and sources. A sub-isovalue cell that becomes wet during
the frozen sequence of transport microsteps can therefore receive flux without
having had its capacity equation projected. This is a membership and time-level
gap, even when the residual inside the pressure domain is small.

An auxiliary correction of `q_f` would be another pressure-like projection. If
it changes transport flux only, the interface follows a different velocity from
momentum. If it changes the stored face velocity, it changes the dynamics and
free-surface boundary condition. The earlier attempt to make surrounding air
divergence-free was removed after frozen liquid-adjacent rows left disconnected
air components with incompatible net flux. A new flux projection is therefore
not a cheap algebraic cleanup; it needs a compatible component boundary
condition and a velocity-coupling argument.

### Can a certified bulk be skipped safely?

Not from a static RDF distance test alone. In the static limiter, an oriented
edge `i -> j` is multiplied by the receiver factor `r_j`. If `r_j` decreases,
cell `i` loses outgoing volume and may then overfill; the change propagates
backward through the directed flow graph. A recirculating strongly connected
component can make the dependency component-wide. A narrow RDF band plus a
fixed halo is therefore not an exact closure in general.

There is, however, an exact active-frontier formulation of the same fixed
point. Scan all cells once, enqueue every invalid cell plus every cell whose
factor proposal changes, and when `r_j` changes revisit `j` plus donors on
positive low-flux edges entering `j`. Retaining unchanged-but-invalid cells and
the proposal/commit generation is necessary for the same Jacobi semantics.
Continue in the production accumulation order until the same zero-invalid
certificate is reached. The frontier is the dynamic reverse-flow closure, not
a geometric halo. Its result can be required to be byte-identical to the dense
iteration. Worst-case work is still the whole connected component for every
pass; the experiment must report frontier sizes rather than assume locality.
Moving-solid dual potentials couple both endpoints differently and should keep
the dense fallback in the first implementation.

## Ranked alternatives

### 1. Exact active-frontier execution of the current limiter

This changes scheduling rather than the numerical method, but it attacks the
largest observed repeated work with the least numerical risk. It retains PLIC,
shared physical subfaces, every factor update, FCT, volume authority, adaptive
seam behavior and solid fallback. Work changes from `cell count * pass count`
to one dense seed scan plus the sum of actual reverse-flow frontier sizes.

The risk is performance, not conservation: a long directed chain or circulation
can activate the whole component. Sparse queue construction and duplicate
suppression may also cost more than dense coherent scans when the frontier is
broad. This is still the first experiment to finish because exact output parity
makes the result easy to accept or reject.

The CPU reference proof now passes all eight targeted tests. In a 16,384-cell
synthetic case with one localized 64-cell directed chain, all algorithms
converged in 40 passes. Dense copy and dense ping-pong each evaluated 655,360
cell updates; dense copy also performed 638,976 bank-copy visits, while
ping-pong removed those copies but retained every update. The exact active
frontier evaluated 16,462 cells and committed 16,460 frontier entries, with
per-pass populations `[16384, then 39 x 2]`. That is about 40 times fewer update
visits and 39 times fewer combined update/commit visits than dense copy in this
deliberately localized fixture. In a production-derived B8:B4 topology with 40
cells, 94 faces and one real mixed seam, the dense algorithm evaluated 80 cells
over two passes and the frontier evaluated 42, with populations `[40, 2]`.
Factors, final low flux, low-state volume, and each pass's
invalid/first-invalid/changed receipts are byte-identical. The tests also cover
bounds, reflection, sources, exterior faces and partial static capacity; moving
solids explicitly retain the dense algorithm. These are CPU reference work
counts from one synthetic fixture and one manufactured state over an actual
adaptive topology, not a live full-transport run, GPU timing result or
production speedup claim. Dense ping-pong structurally removes the bank-copy
scan but leaves every repeated dense update; do not translate that removed scan
into a percentage of total transport time.

If the GPU replay succeeds, this becomes a conservative interface/bulk hybrid
without changing equations. Compile a VOF mixed-donor/swept-face list for PLIC
high-flux work, because exactly full and empty donors already take trivial
branches. A freshly rebuilt scalar/RDF support mask could help seed that list,
but the current frame-tail RDF is not current after the first microstep. Execute
low-flux feasibility on the dynamic reverse-flow frontier described above.
Certified untouched bulk remains implicit; a frontier that reaches it makes it
explicit again. These must be two independently measured lists: the narrow
geometry band does not certify the limiter dependency graph.

### 2. Directional conservative long-step remap with RDF endpoint geometry

For the first substantive numerical alternative, use a conservative
cell-integrated semi-Lagrangian remap along one coordinate at a time. Trace
shared destination faces backward, require their order to remain monotone, and
integrate the donor interval with row prefix sums. Whole crossed cells become a
range sum; only the two fractional endpoints need a local interface integral.
The RDF can supply the endpoint orientation, with the local line translated to
match the donor `V` before it is integrated. A symmetric X/Y composition reduces
directional bias.

For uniform translation, direct endpoint lookup makes work nearly independent
of the number of crossed cells. Conservation follows because shared departure
intervals partition a row. Positive weights and explicit transport of a
capacity/Jacobian measure can provide bounds without the receiver-factor fixed
point. A non-monotone departure map must split on deformation; large translation
alone need not split.

The main risks are multidimensional compression and split error, the need for
the transported capacity measure to return to the Eulerian capacities after an
X/Y composition, variable-width adaptive rows, and wall-separated row spans.
Static cut cells need actual open interval geometry, not a scalar aperture.
Moving solids need swept-capacity accounting. Frozen outer-step momentum and
pressure remain an accuracy issue even if volume transport is exact. This is
considerably simpler in 2D than the current 3D polyhedral remap experiment and
is the strongest route to removing work proportional to translation Courant
number.

### 3. Capacity-compatible subface flux followed by one-pass algebraic FCT

Widen or extend the projected face-flux field so every cell in the swept RDF
band satisfies the capacity/GCL equation, then use donor upwind plus the existing
one-pass FCT limiter. The high flux can remain the current exact planar flux or
become a fixed-stencil RDF-guided compressive flux. This removes the receiver
fixed point while retaining equal-and-opposite shared transfers.

It is attractive if compatibility can be folded into the existing projection.
A separate band solve duplicates pressure-like work, while expanding the liquid
pressure domain changes the atmospheric free-surface condition. Holding
liquid-adjacent rows fixed can make a component infeasible. Any prototype must
report the correction to physical face velocity and compare material and
momentum trajectories; bounds alone are insufficient.

### 4. RDF-guided algebraic VOF flux

Use `grad(RDF)` for interface direction and form a bounded high-order face value
from nearby `V/K`, for example a monotone reconstruction plus a compressive
antidiffusive term aligned with the RDF normal. Freeze one value on each shared
subface and pass `high - low` through the existing FCT limits. This removes the
ELVIRA candidate search and the swept-line fraction evaluation in favor of a
small fixed stencil.

Global volume conservation remains exact because the face transfer is shared.
Local bounds remain exact only if the low update is already feasible; this
alternative does not by itself remove the 821-pass bottleneck. It will usually
trade exact planar translation for some diffusion, compression, grid bias or
small wisps. A separately advected RDF would introduce a second interface
authority; the safer form derives RDF from each accepted volume state and uses
it only for direction. Using the frame-start RDF through all microsteps is
cheaper but lagged, and must be tested under rotation/deformation rather than
only translation.

### 5. Advected 2D contour with conservative rasterization

Treat the RDF zero contour as a global polyline, move its vertices through the
velocity field, remesh it, and rasterize the resulting closed polygons back to
cell volumes. Work can scale with interface length and touched cells rather than
the liquid bulk. It naturally gives a smooth visible boundary, and polygon area
supplies a strong global conservation ledger.

Independent vertex tracing does not guarantee an area-preserving map, so a
global area correction is visual bookkeeping rather than local physical
transport. Conservative swept-edge fluxes can strengthen it, but merging,
breakup, thin sheets, bubbles, contact with solids, coarse/fine crossings and
moving walls turn it into a substantial front-tracking system. It is a good 2D
research path when interface work is sparse, but a poor first production swap.

### 6. RDF/height-column hybrid

Where the RDF certifies a floor-connected, single-valued surface, store and
transport column liquid volume with a one-dimensional conservative flux. Use the
general cell method only for overturning, detached, multiply valued or
solid-intersecting patches. The repository already has a strict column-height
presentation certificate, so the classifier is not hypothetical.

This can reduce degrees of freedom dramatically for calm pools and long waves,
but it changes the represented dynamics unless vertical momentum and exchange
with general patches are derived carefully. Splash, bubbles, breaking waves and
arbitrary gravity directions defeat the height model. Treat it as a model
hybrid, not a cheaper implementation of the same equations.

### 7. Visual-only RDF advection

For the cheapest similar-looking result, semi-Lagrangian-advect the RDF,
reinitialize it near the interface, and shift its zero isovalue to match one
global target volume. This gives fixed local work, smooth normals and simple
topology changes. It is suitable for a display surface or a deliberately
nonphysical interactive mode.

It is not conservative physical transport. Global volume correction does not
restore local cell volume, momentum consistency, bounded capacity, or correct
solid contact. Coupling pressure to this field while retaining another mass
field creates disagreeing surface authorities. If used only for presentation it
does not reduce the current physical transport cost.

## Recommended bounded work

The exact static active-frontier CPU proof is positive enough to justify a GPU
replay experiment. Keep the high-flux bank frozen and compare the production
dense update/commit ping-pong with a compact reverse-flow frontier on the same
static microsteps. Require byte-identical factor banks, low fluxes, per-pass
invalid counts and first-invalid IDs, final volumes and density. Report dense
cell visits, frontier visits, compaction and queue overhead, and separate GPU
timestamps for the first PLIC/face pass, repeated limiter work, and host
continuation waits. Include a deliberately component-wide circulation as the
negative performance case. Moving solids remain on the dense path.

Then build the first numerical prototype as a 2D periodic all-fine directional
long-step remap, outside production. Exercise Courant numbers 0.5, 2, 8 and 25
for axis and diagonal motion; full and planar liquid; balanced variable face
flux; and forward/reverse deformation. Transport liquid and capacity/Jacobian
with identical positive weights, publish one candidate, and audit raw donor
coverage, receiver capacity, bounds, volume and RDF surface error without
clamping, normalization or redistribution. Count range queries and fractional
endpoint integrations while increasing only a uniform velocity boost. This
directly tests the desired work law before adaptivity, walls or pressure coupling
can obscure it.

Do not begin by replacing `gvHighFlux`. That is a useful later quality/cost A/B,
with a clean seam in `computeGeometricVolumeFluxes`, but it leaves the current
low-order iteration unchanged. Likewise, do not add an air-band projection until
the experiment states whether corrected flux is also the physical face velocity
and proves component feasibility.

## Repository evidence map

- `lib/methods/adaptive-volume/advance-slice/slice-stage-numerics.ts` contains
  the 2D ELVIRA reconstruction, analytic swept-rectangle high flux, dense static
  receiver-factor iteration and FCT commit.
- `lib/methods/adaptive-volume/resident-volume.wgsl.ts` contains the equivalent
  production GPU volume plan, PLIC/high-flux pass, low limiter and final FCT.
- `lib/methods/adaptive-volume/advance-slice/slice-presentation-publication.ts`
  constructs the presentation-only shared RDF from accepted VOF/PLIC state.
- `lib/core/geometric-low-flux-frontier/reference-2d.ts` and
  `tests/geometric-low-flux-frontier-2d.test.ts` contain the static CPU frontier
  proof and exact dense/ping-pong comparison.
- `docs/geometric-large-timestep-remap-plan.md` records the prescribed-map
  remap evidence, rejected arbitrary-velocity lift, and open production bridge.

## Primary references

- Scheufler and Roenby, *Accurate and efficient surface reconstruction from
  volume fraction data on general meshes*: <https://arxiv.org/abs/1801.05382>.
- Pilliod and Puckett, *Second-order accurate volume-of-fluid algorithms for
  tracking material interfaces*: <https://www.math.ucdavis.edu/~egp/PUBLICATIONS/JOURNAL_ARTICLES/APPEARED/2004/JEP-EGP-2004.pdf>.
- Lentine, Gretarsson and Fedkiw, *An unconditionally stable fully conservative
  semi-Lagrangian method*: <https://physbam.stanford.edu/~fedkiw/papers/stanford2010-01.pdf>.
- Chen et al., comparison of long-timestep flux-form transport schemes:
  <https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.3125>.

The repository's measured large-remap work and its rejected arbitrary-velocity
construction are recorded in `docs/geometric-large-timestep-remap-plan.md`.
That experiment establishes prescribed-map feasibility, not a production
velocity bridge or production speedup.
