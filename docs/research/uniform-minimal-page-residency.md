# Uniform Geometric: minimum necessary page residency

Design proposal, 2026-09-21. This document specifies proposed behavior; it does not
claim the production cutover or the policies below are implemented. It extends
[the page-domain architecture](uniform-page-domain-design.md) and
[the support producer](uniform-page-support.md). Performance tuning follows the
production cutover; aggressive sparsity is the intended default architecture for
all scenes, not an optional alternate solver.

## Implementation receipt — 2026-09-21

The standalone GPU support producer now implements occupied bounds, 4³ masks,
physical-volume adapters, local signed motion intervals, forward destination roles,
and bounded closure over neighboring air velocities and backward reads. Classification
writes accepted slots only. Request/coordinate/field/closure failures prevent allocator
publication; focused Dawn generation/support/publication tests pass (13 tests).
See [the support receipt](uniform-page-support.md) for the precise scope and limits.

The production root solver now audits actual texture-load taps against a GPU-built
accepted-membership map. SIM exposes missing-read count and field-binding mask.
Reduced/reordered/empty/restored catalogue tests verify the fault counts and ensure
that the diagnostic leaves field values unchanged. This is observability, not a
sample substitution or rollback mechanism; extension and multigrid need their own
coverage, and metric-phi validity is distinct from page membership.

This is partial implementation of steps 1–2 below. It does **not** complete production
minimal residency: production still retains the authored catalogue. Finite-band
validity and actual operator sample certification, sparse pressure/extension and
presentation consumers, whole-fluid commit/rollback, cleanup ledger, and residency
role diagnostics in the UI remain outstanding. Cleanup is not enabled. The rejected
phi-cap experiment is not being promoted to bypass those dependencies.

## Decision

Keep the smallest conservative support for the fluid we choose to retain. Make the
choice to discard visually negligible fluid explicit, bounded and observable.
After that choice, operator support and pressure coupling remain correctness
requirements. Do not trade missing samples or artificial page walls for speed.

Use 32³ storage pages initially, with 4³ occupancy/compute tiles. Retain the existing
pressure storage edge where required by its implementation; translate between
storage layouts using world cell coordinates. A page is an allocation unit, not an
instruction to run every operator on all its cells. Tune storage size later using
thin streams, broad ponds and small dense scenes.

The desired set is:

```
retained liquid + swept source footprints + predicted transport destinations
                + the actual read/write support of scheduled operators
```

Deep stationary liquid remains represented and pressure-coupled. Support pages
must never recursively seed more support merely because they are resident.
Distance between disconnected ponds and authored empty-world extent must not
determine allocations or fine-grid work.

## Why the current support policy overallocates

The original `lib/methods/uniform/uniform-page-support.ts` policy had the following
defects. The receipt above records which have since been addressed:

- Every `V != 0` retains a page, including indefinitely small numerical remnants.
- Every `phi < positiveBand` seeds support, including stale negative phi with no V.
- Velocity travel is reduced to a maximum over all resident pages. A fast jet can
  expand the support of a distant quiet pond.
- Expansion starts from the full page and uses a symmetric integer page radius.
  Even a tiny positive reach can request all 27 neighboring pages, although fluid
  may be well inside one page and moving in one direction.
- Duplicate requests are emitted before merging, and emission/planning are serial.
  Request capacity and planning cost can grow much faster than unique residency.
- The default read reach is not yet a complete contract for all fluid operators.

Changing these policies alone is insufficient while consumers still require dense
fields or evolving far-air phi. The finite-band experiments already showed that
simply clamping phi can materially change the simulation. Implement the field
contract as well as the allocation policy.

## 1. Classify actual occupancy on the GPU

Reduce canonical cell/face owners into compact summaries over accepted pages:

- Retained physical liquid volume, maximum liquid fraction and occupied-cell bounds.
- 4³ tile occupancy masks and interface bounds; support validity separate from phi.
- Per-axis velocity intervals over the samples the transport integrator can use.
- Source/edit protection, solid-contact status, pending flux and dependency roles.
- Cleanup eligibility age and a stable world-coordinate identity for selection.

Define physical volume from the solver's actual V convention. If V is a fraction of
an open cell, use `V * openCellVolume`; if it already stores physical volume, do not
multiply it again. This conversion must have a tested adapter.

Build summaries as a byproduct of final field writes where possible. Do not add a
second complete scan of every field for every operator. Use compact GPU counts for
summary and dispatch work; reserve buffers by capacity but do not clear or scan the
whole field pool each frame. Metadata clearing should touch used ranges, or use
generation tags with an explicit wrap/reset policy.

Phi/V disagreement is not permission to delete fluid. Resolve it through the
defined interface/volume reconstruction or flag it. A stale negative phi must not
keep an otherwise cleaned page alive forever, but a valid interface cannot be
discarded solely because one volume sample is zero.

## 2. Predict a local swept footprint for each simulation substep

Predict what fluid *may* reach, not only what current velocity says it will reach.
Recompute per simulation substep, rather than reserving for a variable render frame.

For occupied bounds `[lo, hi]` in cell coordinates and certified velocity intervals
`[uMin, uMax]` over the trajectory neighborhood, a conservative swept box per axis is:

```
travelLo = min(0, dt * uMin / h) - uncertaintyCells
travelHi = max(0, dt * uMax / h) + uncertaintyCells
swept   = [lo + travelLo, hi + travelHi]
```

Include cell extents when forming occupied bounds. Compute page coordinates by
signed floor division of the expanded cell bounds, with explicit endpoint rules.
Do not expand a whole page by `ceil(reach / pageEdge)` when its occupied bounds are
available. Start with per-page boxes; use occupied tile boxes where a thin stream
or separated patches make the page box wasteful.

The uncertainty term must be derived from the actual integration stages: force
increments, interpolation footprint, midpoint/backtrace queries, source velocity,
and any bounded correction. `0.5 * aMax * dt² / h` is valid only for an integrator
whose positional error is bounded that way; it is not a universal allowance.
Pressure can change velocity nonlocally, so a previous-frame velocity maximum is
not a certified bound on post-projection transport.

Use two support phases where the solver ordering permits:

1. Acquire support needed for current-liquid pressure/forces and their stencils.
2. After the actual advecting velocity is known, validate or expand transport support
   before transport writes. Include the source state used by that transport stage.

Local velocity bounds must cover the whole characteristic neighborhood, including
air velocities used by extension/interpolation. A liquid-only reduction is unsafe.
Compute a bounded closure over the candidate trajectory neighborhood; if closure
cannot be certified within the encoded pass budget, reject the candidate step.
Forward swept occupancy alone also does not certify backward sampling: separately
request the reads of all destination cells the transport scheme will evaluate.

Use GPU-generated indirect expansion/retry passes with a fixed encoded maximum.
No CPU readback chooses the next pass. A failed support check leaves accepted state,
clock, source consumption and cleanup ledger unchanged. Never silently clamp a
characteristic to the allocated region.

Remote or moving emitters seed their swept footprint directly from source commands,
including interpolation/interface support. They do not rely on last frame's fluid
being nearby. Teleport and continuous source motion must have explicit semantics.

## 3. Derive support per operator, not one universal halo

All lists derive from one candidate generation and canonical field ownership.

| Operator | Work and support rule |
| --- | --- |
| Volume transport | Retained liquid, possible flux destinations, and all reconstruction/backtrace samples. |
| Phi/interface | Valid finite interface band plus composed advection, redistance and correction reach. |
| Velocity extension | Only velocities needed by upcoming transport/interface queries, plus extension dependencies. |
| Pressure | All retained liquid rows and true coefficient neighbors; one coupled graph across page seams. |
| Multigrid | Ancestors of active rows with required transfer/operator support; no scene-wide dense coarse rectangle. |
| Solids | Geometry/open fractions required by these operators; allocated edges never become solid walls. |
| Surface output | Interface tiles and meshing samples; deep liquid need not execute surface extraction. |

Sequential stencil reaches compose: an intermediate result can require more support
than the maximum radius of either individual kernel. Record that composition in an
operator support contract, including iteration counts. A global interface correction
must have a bounded displacement or trigger support revalidation. Pressure has global
coupling through liquid; a finite halo does not replace that solve.

Absent space outside certified support represents ambient air under the explicit
finite-band contract. An absent required sample is a fault. Rendering consumes the
accepted generation and cannot independently keep simulation pages active forever.

## 4. Remove tiny remnants under a physical-volume budget

Treat `1e-2` as a proposed threshold in **full-cell equivalents**, not 1% of page
capacity and not a fraction threshold applied independently to every cell.
At 32³, a 1%-of-page threshold permits removing 327.68 full cells, which is far too
large for this purpose. Define `cellVolume = hx * hy * hz` and start with:

```
pageRemovalCandidate: sumPhysicalVolume(page) <= 0.01 * cellVolume
```

These are initial research settings, not measured perceptual tolerances. Test
`0`, `1e-4`, `1e-3`, and `1e-2` cell equivalents before choosing production defaults.
Keeping the unit tied to a cell makes the physical tolerance resolution-dependent;
report that explicitly in comparisons.

Eligibility also requires:

- No source/edit protection or incoming fluid predicted in the upcoming substep.
- No nontrivial wet connection to retained fluid across page faces. A low-volume
  tail attached to a pond is initially retained; do not sever a thin fluid bridge.
- No significant reconstructed interface feature inconsistent with the tiny-volume
  classification. Repair or reject inconsistent field state before deletion.
- Continuous eligibility for a short simulation-time dwell (propose 0.05 seconds),
  reset by replenishment. Aging follows accepted simulation time, not render frames.

Start with conservative page-local candidates plus cross-page wet-neighbor checks.
Later, bounded connected-remnant labeling can remove tiny droplets straddling seams
without page-alignment bias. Accept only components whose complete membership is
known; incomplete labeling is not evidence of a small component. Sum a component's
volume once, never permit the threshold separately for each of its pages.

Use a second, global deletion budget so many individually tiny candidates cannot
drain a scene. Proposed starting limits:

- Lifetime deletion <= 0.1% of initial volume plus cumulative committed inflow.
- Token refill rate <= 0.01% of current retained volume per simulated second.
- Burst capacity <= 0.01 full-cell equivalents, also limited by lifetime budget.

The lifetime denominator excludes numerical corrections and recycled residuals.
The token bucket and ledger advance only on commit. Initialize the bucket up to its
burst allowance subject to the lifetime cap. Sort eligible candidates by stable
world key and use a deterministic prefix volume sum to admit a fitting prefix.
This can underuse the budget; it is preferable to race-dependent deletion. Later
selection may prioritize age without making slot allocation order affect physics.

If a remnant exceeds the budget, retain it. Do not increase tolerances under memory
pressure. A budget reset must be an explicit scene reset, not allocator churn.

Deletion is an intentional sink: clear candidate V, reconstruct the affected phi
band and update validity masks. Remove obsolete velocity/history only where no
retained fluid depends on its canonical faces. Rebuild pressure membership and
coefficients before solving the cleaned state. Update the volume-correction target
by exactly the removed volume, so correction cannot recreate the discarded fluid.
Track a momentum diagnostic too; tiny high-speed remnants can carry meaningful
momentum even when their volume is small. Add a momentum eligibility limit if
high-speed splash tests show unacceptable changes.

Do not spread discarded volume back over a pond by default: that changes another
body and can undermine cleanup. Report the intentional sink separately from transport
error, inflow/outflow and numerical correction in the mass balance.

## 5. Retire residency independently from retaining a physical slot

After cleanup, rebuild dependency masks from retained fluid, sources and the operator
contracts. Retire a page only when it has neither retained fluid nor any required
support role. Removing a tiny remnant does not guarantee its page can be removed:
an adjacent pond may still need its samples.

Drop unused pages from compute lists immediately. Avoid using a long active halo as
hysteresis. Optionally retain recently unused slots in a small bounded cache, excluded
from simulation membership and compute. Reuse requires full field initialization
and validity checks; cached data never implies valid fluid. Cache slots are reclaimed
first under pressure and cannot be reused until old consumers are ordered complete.

Topology, candidate fields, source consumption, cleanup accounting and presentation
publish atomically. An allocator metadata transaction alone cannot roll back fluid
already modified in place. Reserve transition capacity for new support while the old
generation remains live; report peak transition demand as well as accepted pages.

## 6. Keep the residency machinery cheaper than the work it removes

Replace global velocity dilation first. Then parallelize request construction using
per-tile counts, prefix allocation and early coordinate deduplication. Merge role
masks by unique world coordinate before expensive field initialization. Maintain
compact resident and per-operator lists with GPU indirect dispatches.

Bound request capacity independently from resident capacity. Record unique demand,
duplicate demand, high-water marks and retry counts. Neither a full-pool metadata
scan nor hash-table clearing should grow with empty world extent; measure their
capacity cost explicitly. Stable slots and cached adjacency can reduce rebuild work,
but no optimization may make a stale generation visible.

For small dense scenes, use contiguous dispatch where it wins, with the same page
membership and boundary rules. Aggressive page removal is a means to lower total
frame time; it does not justify expensive cleanup or component labeling every step.
Run optional cleanup classification at a GPU simulation-time cadence, while motion
support validation remains per substep.

## Diagnostics and visual layer

Show distinct colors for retained liquid, interface support, predicted destinations,
stencil-only support and cleanup candidates. Show retired/cache slots separately as
optional historical outlines. Predicted pages should expose their requesting operator
and motion bound on inspection.

The simulation panel should report:

- Resident pages by role, occupied tiles and dispatched tiles per operator.
- Newly allocated/retired pages, cached slots and transition peak capacity.
- Predicted support versus destinations actually touched, and support faults/retries.
- Discarded volume this step, cumulative volume, budget remaining and blocked candidates.
- Timings for classification, request/dedup, initialization, topology, each fluid stage,
  cleanup and publication; distinguish GPU time from CPU/queue gaps.

For a garden hose entering a pond, support should move with the jet, expand at the
advancing pond edge and retreat after the source moves. A high-speed jet should not
inflate a disconnected quiet pond's support. A thin residual trail should disappear
only when the recorded cleanup policy permits it.

## Implementation sequence and acceptance

1. Establish finite-band field semantics and end-to-end page authority as prerequisites
   for actual retirement. Add summaries and diagnostics while this cutover proceeds.
2. Replace symmetric global dilation with occupied-bounds/local swept support and
   explicit operator contracts. Validate with cleanup disabled to isolate correctness.
3. Derive all heavy dispatches from occupied/operator tiles; remove capacity-shaped
   clears and dense publication work that conceal the benefit of fewer pages.
4. Add transactional residual cleanup and immediate compute-list retirement. Start
   conservatively; component-based cleanup is a later improvement.
5. Optimize deduplication, topology reuse, slot caching and dense-scene dispatch.

Required comparisons use identical dt, sources and numerical settings, with a
cleanup-disabled page solver as the behavioral control. Long chaotic trajectories
need physical metrics rather than bitwise identity; also retain short deterministic
seam and source fixtures.

- Motion: fast diagonal jets, acceleration, projection-induced velocity changes,
  moved/remote emitters and variable substeps. No required sample may be missing.
- Retention: quiet deep ponds, thin connected sheets, wet solid contacts and separate
  ponds. Page seams and translated page alignment must not introduce walls or leaks.
- Cleanup: isolated and seam-straddling remnants, repeated tiny source additions and
  many simultaneous candidates. Verify ledger, dwell, budget, correction target and
  absence of frame-rate-dependent loss or remnant resurrection.
- Transaction: insufficient resident/request/transition capacity and failed retries.
  Accepted fields, clock, inflow and deletion ledger must remain unchanged.
- Scale: keep fluid/support fixed while moving bodies far apart and enlarging empty
  world extent. Fine allocation/work stays fixed, apart from documented sparse
  hierarchy costs; no full-world census is permitted.
- Plausibility: measure retained volume after accounting for intentional sinks,
  interface displacement in cells, pond level, front arrival, pressure residual,
  divergence and momentum changes. Start with a 0.1-cell quiescent-interface tolerance
  outside the deleted feature; establish dynamic tolerances before promotion.
- Performance: compare full-frame and per-stage medians/p95 for garden hose and
  mini64, including growth/retirement frames. The historical 95% mini64 throughput
  target remains a later optimization goal; it does not delay the requested cutover.

The major risks are an incomplete finite-phi contract, underestimated characteristic
support, excessive topology churn, resolution-dependent cleanup, cumulative erosion
and hidden capacity-shaped consumers. Conservative support validation, explicit sink
accounting and separate operator work lists address these directly. Merely lowering
the resident-page counter is not evidence of either correctness or a faster frame.
