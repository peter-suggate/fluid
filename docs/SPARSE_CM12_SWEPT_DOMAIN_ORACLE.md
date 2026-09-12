# Sparse CM12 swept-domain oracle

This oracle defines the minimum sparse page domain for one physical transport
step. It is independent of the current activity masks and host broad phase.

## Contract

Pages have width `B = 8` finest cells and are half-open:

`P(q) = [8 qx, 8(qx + 1)) × [8 qy, 8(qy + 1))`.

For donor material region `M_i` and its prescribed whole-frame displacement
`d_i = u_i Δt`, the exact swept material is

`S_i = M_i ⊕ {t d_i | 0 ≤ t ≤ 1}`.

The required material-page set is every page whose interior has
positive-measure intersection with `union(S_i)`. Disconnected donor components
are unioned separately; an AABB spanning two components is not part of the
domain. Page-boundary contact with zero overlap does not claim a page.

The geometric oracle above uses exact positive measure. A future numerical-set
oracle must separately apply the transport-volume roundoff rule:
`Δt A_face |u_face| > 2^-20 C_cell`. This rejects projected floating-point
residue, while a flux just above the solver's own accepted-volume bound still
claims its receiver. Every prescribed fixture displacement is far above that
bound, so its expected geometric and thresholded numerical sets are identical.
The generic thresholded-set oracle is still pending.

The oracle uses the complete physical frame `Δt`, even when transport divides
that frame into CFL microsteps. Every page in the full-frame set must be ready
before transport microstep 1. The velocity is the final projected face velocity
that transport will consume; body forces must not be applied to it a second
time.

Page membership is independent of rung. The minimum rung assignment is the
least dyadic closure satisfying face-neighbour 2:1 among pages that are already
active. An absent face or diagonal neighbour is not allocated solely for
grading. Any rule that raises a swept receiver for transport accuracy must be
specified and measured separately; the geometric oracle does not assume B8.

Velocity interpolation will be audited separately by replaying only accepted rows
that actually trace (`touchesExtendedVelocity`). For each characteristic
substep, its set contains owner cells at the initial/current and midpoint
velocity-support evaluations, respecting solid clipping and ignoring
zero-weight nodes. At the final departure it includes every positive-weight
source node: up to `4^d` for an eligible uniform-interior cubic sample, otherwise
the `2^d` linear nodes, plus the VEX fallback support used for a missing
staggered node. Missing sparse-air samples already have a defined one-sided/zero
value, so an interpolation point outside the swept material set is not, by
itself, an active material-page requirement. This executable interpolation-set
oracle is still pending. The resident implementation to replay is in
`webgpu-sparse-cm12-resident.wgsl.ts`, around
`traceFaceDepartureAtSpans` and `sampleSourceStaggeredVelocity`.

Surface reconstruction has its own topology-accuracy/presentation set `R_RDF`.
For each topology vertex used by a published interface or surface proof, its
vertex least-squares probes are owners of the fine-voxel octants
`v + {-1,0}^d`: four centres in 2-D and eight in 3-D. On a uniform finest-cell
stencil these lie at most `sqrt(d)/2` finest cells from the vertex. The separate
uniform cell-centre gradient stencil is `3^d` (eight other samples in 2-D, 26 in
3-D), at most `sqrt(d)` finest cells away. Mixed-rung fits use actual owner
centres and physical distances, which can be larger. Pagewise, `R_RDF` contains
only pages hit by those probes at actual interface seams; a generic interface
halo is not its definition.

In 3-D, missing sparse-air owners are omitted and the least-squares system is
rank-truncated. The current 2-D reference can synthesize missing in-domain point
neighbours without sparse authority. `R_RDF` may therefore be virtual
presentation support and does not inherently justify a resident air page. If a
page is materialized, face-2:1 grading sets its minimum rung (B4 beside B8); the
RDF math accepts mixed-rung centres and does not demand B8. Physical-solid
truncation remains intentional. The executable `R_RDF` oracle is pending while
that work proceeds in its owning change.

One parity choice remains unresolved: the 2-D virtual-air path currently places
missing samples at the minimum incident width (B8 beside a B8 interface), while
a materialized 3-D neighbour allowed by 2:1 would be B4 with width two. Those
are different least-squares sites. The shared contract must choose identical
virtual or materialized sample placement before RDF page counts can be compared;
until then, report sample coordinates and availability separately from resident
page membership.

The complete justified domain is therefore `T ∪ I ∪ R_RDF ∪ S ∪ G`: swept
material `T`, velocity interpolation `I`, RDF reconstruction `R_RDF`, authored
source/moving-geometry support `S`, and pages already active that need rung
grading `G`. Each set is reported separately so one consumer cannot hide excess
from another.

The strict 3-D parity fixture extrudes every 2-D donor through `[0, 8)` with
`uz = 0`. Its expected pages are exactly the 2-D coordinates with `z = 0`; any
page at another z is excess support. Separate all-axis 3-D cases test the same
Minkowski rule without dimensional reduction. A page-aligned cube translated
`(10,10,10)` crosses exactly 15 pages (`{1,2}³ ∪ {2,3}³`, after shifting the
fixture origin), and `(18,18,18)` crosses 22. A 3-D AABB would incorrectly
claim remote corners outside those diagonal bands.

## Shared fixture

The executable fixture is
[`tests/support/sparse-cm12-swept-domain-oracle.ts`](../tests/support/sparse-cm12-swept-domain-oracle.ts).
It uses `Δt = 1/60 s`, a 6×6 measurement plane, and a page-aligned 8×8 B8
material block at page `(1,1)`. Velocity is prescribed so displacement is
measured directly in finest cells. The analytic enumerator derives signed
candidate bounds from each donor sweep and is not clipped to that plane.

| Case | Displacement (cells) | Expected 2-D pages | Current 2-D pages | Result |
| --- | ---: | --- | --- | --- |
| static | `(0,0)` | `(1,1)` | `(1,1)` | exact |
| axis | `(2.5,0)` | `(1,1) (2,1)` | `(1,1) (2,1)` | exact |
| axis | `(10,0)` | `(1,1) (2,1) (3,1)` | `(1,1) (2,1)` | missing `(3,1)` |
| axis | `(18,0)` | `(1,1) (2,1) (3,1) (4,1)` | `(1,1) (2,1)` | missing `(3,1) (4,1)` |
| diagonal | `(2.5,2.5)` | `(1,1) (2,1) (1,2) (2,2)` | same | exact |
| diagonal | `(10,10)` | 7-page diagonal band | first 2×2 pages | missing 3 |
| diagonal | `(18,18)` | 10-page diagonal band | first 2×2 pages | missing 6 |
| disconnected | `(+10,0)` and `(-10,0)` | 6 local pages | 4 nearest pages | missing one downstream page per component; no bridge |

All current 2-D pages in this fixture are B8; that is current policy output, not
an oracle requirement. The confirmed membership failure is reach: the 3×3
support mask can encode only the first neighbouring page. A rectangular AABB
replacement would fix reach but would over-allocate the remote corners of
diagonal sweeps; for `(10,10)`, `(1,3)` and `(3,1)` are outside the exact
seven-page diagonal band.

A separate static-rung probe retained a zero-density B4 page beside wet B8
using an explicit boundary-support reason. The final coarse-first decision kept
it active at B4. This measured path therefore matches the least 2:1 rung; an
earlier generic `pageDemand` B8 branch is overwritten by the coarse-first
decision and is not evidence of an actual over-fine result.

The production half-pool slice is a separate integration observation, not an
oracle input. At the captured code state, with 64 pressure iterations, it has
28 material pages and zero dry pages at reset and after the first step. This
confirms that the prior left/right air beside the falling ball came from blanket
receiver-face backing rather than its measured downward sweep. RDF work may add
specific dry pages when they belong to the independently measured set `R_RDF`; this
observation is not a universal zero-dry-side requirement.

## Acceptance measurements

For every case and each physical step, record:

- `missing = expectedMaterialPages − activePages`
- `broadPhaseExcess = activePages − expectedMaterialPages` (observational until
  interpolation, RDF, source/geometry, and grading sets are available)
- `unjustifiedExcess = activePages − (T ∪ I ∪ R_RDF ∪ S ∪ G)`, once the pending
  interpolation, RDF, source/geometry, and rung oracles exist
- exact positive-measure pages versus pages after the numerical flux threshold
- over-fine active dry pages relative to the least 2:1 closure
- readiness of every expected page before transport microstep 1
- 2-D versus z-extruded 3-D XY difference, plus any 3-D z leakage

The implementation is acceptable only when missing and unjustified excess are
empty for all fixture displacements, including travel beyond one page.

## Current stage reality

The 2-D slice currently calls `planSliceResolution` after conservative
transport and scalar publication. It therefore plans the next topology from the
previous step's projected faces (plus an acceleration extrapolation), rather
than making the current step's final projected velocity available before its
first transport microstep. Direct policy measurements can compare page geometry
with the oracle, but an integration test must also prove pre-microstep-1
readiness after the lifecycle stage is moved or split. The 3-D stage is measured
against the same checkpoint; matching end-of-frame page counts alone is
insufficient.

The first joined 3-D measurement attempt constructed and warmed the resident in
about five seconds, then spent more than six minutes in
`waitForSimulationReady` compiling simulation pipeline chunks without reaching
a GPU dispatch. It was terminated. No 3-D page-set result is claimed from that
attempt. A second progress-instrumented attempt completed construction and
presentation in about 6.5 seconds, then spent more than 90 seconds compiling the
full simulation family without reaching a dispatch and was also stopped. The
GPU lease was released. The 2-D/3-D empirical columns remain incomplete until a
run passes that readiness checkpoint.

A CPU host reset of the standard static 3-D fixture does establish the
generation-zero distinction. Its 6×6×1 logical domain has nine allocated atlas
pages: one active B8 material donor, four inactive B4 face neighbours, and four
inactive B2 diagonal neighbours. The active solve domain is therefore exactly
the oracle's single material page (512 represented cells); the inactive
catalogue contributes 288 stored adaptive cells but no generation-zero solve
work. Integrated initial mass is 512 finest-cell volumes. This is host reset
evidence only; it does not substitute for the pending GPU pre-transport and
post-transport page-set measurements.

## CPU commands

```bash
node --import tsx --test tests/sparse-cm12-swept-domain-oracle.test.ts
node --import tsx --test lib/methods/adaptive-volume/advance-slice/slice-resolution-policy.test.ts
node --import tsx --test --test-name-pattern='coarse-first pool support follows motion|dynamic rigid and inflow reductions' lib/methods/adaptive-volume/advance-slice/production-scene-slice.test.ts
```

The analytic oracle passes 18/18. The 17 non-fingerprint slice-policy behaviours
pass, including below/above transport-roundoff cases. The focused production
slice tests pass 2/2. The resident-source fingerprint remains intentionally
pending until the concurrently edited 3-D shader is stable.
