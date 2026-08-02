# Bet 4 Stage-1 — persistent-solver adoption contract

Status: shipping implementation complete; live aligned-scene work verdict and
final Dawn symmetry rerun pending. The persistent MGPCG consumes the accepted
GPU-published classes directly and its census is fail-closed.

## Paper basis

Aanjaneya et al. §8 is the narrow authorization for this hybrid. It proposes
the non-graded, lower-cost treatment deep inside the liquid and retains the
power-diagram discretization near the free surface. The authors report testing
that combination in a prototype and state that it still gives a second-order
pressure field. This contract does not generalize that permission: GFM,
cut-cell, solid/world-boundary, and level-transition rows remain in the power
band, and Stage 2 coarsening stays separately gated.

## Authority

The shipping solver must consume `WebGPUStructuredBoundaryCoefficients.worksets`
and its accepted `control`, not the earlier topology-only
`structured.section63.worksets` and `structured.control` pair.

The boundary publication is A/B banked. Its accepted control is the 64-byte
`C` record; the solver-relevant prefix is:

`[flags, firstError, rows, slots, epoch, bank, published, pad]`.

Acceptance requires `flags == 0`, `firstError == INVALID`, `rows > 0`,
`epoch != 0`, `bank <= 1`, and `published == epoch`. The selected bank contains
five row worksets. Each starts with the existing seven-word compact header:

`[epoch, count, capacity, state=3, dispatchX, dispatchY, dispatchZ=1]`,

followed by `count` row indices. Class counts must sum to `rows`, but that is
not sufficient: every row in `[0, rows)` must occur exactly once. A bad header,
out-of-range member, duplicate, omission, or generation mismatch rejects the
whole generation. It must never fall back by treating an unproved row as
regular.

Classes 0–3 retain their Section 6.3 meanings. Class 4 is a separate dry
identity proof used by Bet 1 accounting; it is never counted as regular liquid
interior or credited to the Bet 4 reduction:

| class | transition | power boundary | Stage-1 treatment |
|---:|:---:|:---:|---|
| 0 | no | no | proven regular interior |
| 1 | yes | no | power band |
| 2 | no | yes | power band |
| 3 | yes | yes | power band |
| 4 | n/a | dry | exact diag=1, offdiag=0, RHS=0 identity |

`rebuildStructuredBoundaryRows` marks a row power-boundary when it is air or
touches a missing/world/solid face, a non-unit aperture, a non-unit ghost-fluid
scale, or a liquid/air membership change. `dynamicRowClass` also retains the
existing diagonal-versus-offdiagonal boundary test. `rowTransition` keeps all
level-transition rows out of class 0. Thus class 0 is the conjunction of the
topological and same-epoch boundary proof; topology case zero by itself is not
authorization.

Class 4 is admitted only when the accepted liquid mask is dry, the diagonal
bits are exactly `1.0f`, and all eighteen offdiagonal coefficient bits are
zero. The persistent apply re-proves those coefficients, exact zero RHS, and a
finite input. Any failed proof falls through to the pre-authored power band or
terminates the solve. Liquid/dry membership mismatch still marks the liquid
endpoint as power, so this identity class cannot weaken a free-surface seam.

## Persistent apply

The persistent kernel binds the accepted control and A/B workset bank directly.
Its P0 five-way merge validates the header generation, dispatch record,
capacity, sorted membership, exact coverage, and disjointness before any solve
work. No host readback participates in scheduling.

For class 0 rows:

1. Visit only the six signed Cartesian axes; the twelve edge channels are
   structurally absent.
2. Resolve the neighbor through the already accepted SPGrid page adjacency,
   brick mask/rank, slot bounds, flags, and owner guards. Do not walk the power
   descriptor, catalog, catalog-slot chain, or general `pageSlot` path.
3. In the first shipping cut, continue loading the accepted dynamic diagonal
   and six face coefficients. This preserves GFM theta, cut apertures, and the
   seam coefficient without changing floating-point evaluation order.
4. A face is regular-coefficient eligible only when both endpoints are class 0.
   If either endpoint is class 1–3, both rows load the one boundary-authored
   power coefficient for that face. This unique face authority is the matrix
   symmetry rule.

For classes 1–3, the shipping path retains the complete power apply, including descriptor
and catalog interpretation, all eighteen channels, finer adjoint terms, GFM
theta, and cut-cell coefficients. Section 4.3 band construction and smoothing
must use these accepted dynamic classes as well; there is no global smoother
in this design.

The optional regular coefficient (`h`) substitution is a later micro-cut. It
may land only after the f64 differential harness proves the same operator on
every regular/regular face and the seam continues to load the dynamic value.
It is not needed for the Stage-1 work-minimization verdict.

## Required measurement

`decodeOctreePersistentMGPCGHybridCensus` accepts only the census marker written
after the kernel's exact five-class partition validation. It reports:

- regular rows = class-0 count;
- power rows = class-1 + class-2 + class-3 counts;
- dry identity rows = class-4 count, separately attributed to Bet 1;
- liquid rows = regular rows + power rows;
- descriptor/catalog machinery: `liquid rows -> power rows`;
- general page-slot chains: `18 * liquid rows -> 18 * power rows`;
- Bet 4 machinery reduction: `liquid rows / power rows`.

Bet 4 Stage 1 passes its own ≥2× work requirement only when an actual accepted
shipping generation reports `liquid rows / power rows >= 2`. Adding any number
of dry identity rows cannot improve that score. Synthetic Cartesian fixtures
are supporting evidence, not the live verdict. Wall-clock and GPU
instruction tuning remain deliberately outside this gate.

## Correctness gates

The adoption is releasable only when all of these pass on the shipping path:

1. the f64 full-power versus hybrid differential operator harness;
2. exact bilinear symmetry and energy equality at regular/power seams;
3. all 48 cube/D4 transforms;
4. the Dawn `symmetric-expansion` bitwise oracle through its accepted window;
5. dam-break volume and energy brackets;
6. forced rejection for stale headers, duplicate/missing members, invalid row
   indices, and mismatched banks/epochs.

Stage 2 interior coarsening is not authorized by this contract. It begins only
after this machinery-only path meets the gates above.
