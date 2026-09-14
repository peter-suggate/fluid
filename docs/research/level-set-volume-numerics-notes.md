# Level-set-plus-volume numerical notes

Date: 2026-09-14. Advisory notes for implementing
`level-set-volume-lab-plan.md` against the current 2-D Rust world.

Primary sources: [Chentanez and Müller 2012, *Mass-Conserving Eulerian
Liquid Simulation*](https://diglib.eg.org/items/d0199e87-1f0d-4230-8a0c-657d2fb9914e),
[Lentine, Aanjaneya, and Fedkiw 2011, *Mass and Momentum Conservation for
Fluid Simulation*](https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Lentine11.pdf),
and [Lentine, Grétarsson, and Fedkiw 2011, *An Unconditionally Stable Fully
Conservative Semi-Lagrangian Method*](https://physbam.stanford.edu/~fedkiw/papers/stanford2010-01.pdf).

## Adaptive conservative operator

`Fields::density` is a cell fill, not a physical amount. For cell `i`, define

```text
q_i = capacity[i] * cell[i].measure       open physical capacity
m_i = density[i] * cell[i].measure        liquid amount
f_i = m_i / q_i                           fill of open capacity, q_i > 0
```

On unequal cells, unweighted row and column sums are not the desired
invariants. A capacity-marginal transport coupling `T` should satisfy

```text
sum_j T_ij = q_i
sum_i T_ij = q_j
m'_i = sum_j T_ij f_j.
```

The second equality gives exact conservation. The first reproduces a constant
open-volume fill. Equivalently, donor fractions `B_ij = T_ij/q_j` have unit
column sums and satisfy `sum_j B_ij q_j = q_i`.

Build the initial support from each receiver's backward departure. Fine-lattice
bilinear corners can map to the same adaptive owner, so aggregate those weights
by owner before normalization. Zero-capacity cells are absent from the
transport marginals.

Alternating row and column scaling for a fixed number of passes does not make
both constraints exact. The final scaling determines which invariant is exact.
Finish with donor/column normalization to make mass exact, and report the
remaining capacity-row residual. A sparse stencil may not admit both marginals
at all; disconnected support and Hall-deficient subsets must be diagnosed.

The CM12 paper's “three scatter passes” are not three normalization iterations.
CM12 and Lentine use backward weights, clamp oversampled donors, and
forward-trace each donor's missing remainder. Their last donor constraint is
what proves conservation; destination row sums remain approximate and are
tracked by cumulative gamma plus diffusion. See CM12 section 3.4 and LAF11
section 2.1 in the primary sources linked above.

## Missing donors and large Courant numbers

A backward-only stencil cannot guarantee that every donor exports its amount.
At large Courant number, a donor can have `beta=0`, and column scaling cannot
create a nonzero entry. A positive-mass donor with zero backward support needs
a forward characteristic and a normalized receiver stencil, as in LGF11/CM12.
If the lab intentionally omits forward traces, it must name and count a
deterministic conservative fallback, such as retaining the donor in its own
valid cell; that fallback is stable but not characteristic-consistent.

Empty liquid donors still have capacity and therefore matter when measuring
constant-fill or row-sum preservation. They may be skipped only by a
mass-only operator that does not claim doubly stochastic behavior. Guard all
`q`, row sums, and column sums before division. Record zero-support donors,
fallback amount, invalid traces, maximum distance, and maximum Courant number.

The existing `sample_support` clamps requests to the domain and does not require
extended support. That is a boundary policy, not a trace-success certificate.
The level-set path should explicitly classify domain exits and unsupported
extension samples. RK2 removes some trajectory error but does not repair
missing support or make a one-step trace accurate at Courant 10 by itself.

## Phi, PLIC, and RDF lifecycle

The immutable source for a frame should be the previous accepted
`RdfSurface::vertex_phi_fine`. Bilinearly sample it at each receiver departure
to produce one cell-centered transported `phi`. Do not rebuild the RDF before
all phi samples are complete. Non-finite RDF vertices need a phase-consistent,
counted fallback; silently dropping and renormalizing corners shifts the zero
set.

Fit PLIC only in open, partial cells. Use a unit normal from a
resolution-aware gradient of transported phi and obtain the local plane offset
from `plane_from_fill(clamp(density/capacity, 0, 1), normal, cell.widths)`.
Then rebuild the shared RDF from those planes and resample cell-center phi for
the next accepted state. Volume above capacity remains in `density` and is
excluded only from the PLIC fill calculation.

`World::transition` currently transfers fields and reconstructs density-based
planes, but it knows nothing about a separate phi plane. After an accepted
topology transition, reseed cell phi from the newly rebuilt RDF before the next
frame, or extend generation transfer to carry phi explicitly. Injection and
other edits that call `refresh_surface` need the same reseed. A phi vector must
always carry the topology generation and cell count it belongs to.

Generation transfer also currently rejects `density * measure` above
`capacity * measure` at both source and target. That conflicts with this mode's
deliberate over-capacity state. Preserve the production transfer contract and
add an explicit level-set-volume transfer policy: distribute the representable
base amount with the existing capacity/PLIC rule, distribute excess
conservatively over overlapping open children while permitting target excess,
and report the donor residual and transferred excess. Deferring every topology
change until excess vanishes is a simpler experiment but can stall adaptivity.

The current RDF is a reconstructed local distance field and contains `NaN` for
unproved regions. It should not be treated as a globally finite exact signed
distance. Surface-gradient and phi-sampling diagnostics should expose missing
support separately from sub-cell liquid that the stated display rule hides.

## Seam metric

Enumerate `Graph::subfaces`, selecting interior pairs whose adjacent cells have
different widths. For each physical subface midpoint `p` and two valid,
unit-normal partial-cell PLICs, evaluate

```text
s_i(p) = n_i dot (p - center_i) - offset_i
seam_error = abs(s_left(p) - s_right(p)).
```

Report sample count, maximum, RMS, and invalid/skipped counts. Use each
subface's midpoint; a mixed-resolution row can own multiple fine subfaces, and
one row midpoint can hide disagreement. Compare baseline and level-set-volume
on identical accepted topology generations and frames.

## Pressure feedback and receipts

CM12 adds `min(0.5*(density/capacity - 1), 1)/dx` with the sign that produces
expansion. The current Rust RHS is an integrated face-rate balance, so the
source must be converted to that convention (including cell measure and the
solver's sign), then checked with a one-cell expanding-velocity test. Do not
infer the sign from the paper's word “add.”

Receipts should report physical mass `sum density[i]*measure[i]`, maximum and
total physical excess over `capacity`, donor residual, capacity-row residual,
zero-support/fallback counts, phi fallback counts, and the trace/gather/plane/
RDF timing split. Conservation tolerance should be based on an f32 summation
bound rather than asserted as literal equality.
