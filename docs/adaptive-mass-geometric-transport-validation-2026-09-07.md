# Geometric transport and physical-face repair

Implementation of the two mechanisms in the
[repair plan](adaptive-mass-mechanism-plan-2026-09-07.md). The comparison remains
`coarse-first-pool-impact-quarter`, A = max1, B = min1/max2, with both initial
topologies frozen. These are general solver changes; no scene-specific solver
branches, reflected averaging, or additional symmetry passes were added.

## Actual donor geometry

The transport stencil now uses the eight incident **cell centres** around a
primal vertex. Inverting their trilinear geometric map supplies nonnegative
weights. Repeated coarse donors turn this dual cell into a wedge or pyramid.
The regular-grid case retains ordinary trilinear interpolation. This removes
the coarse-cell dead zone and the incorrect fine-side slope demonstrated by
the production-shader affine probe.

The same geometry serves backward mass gathering, forward deficit transport,
effective-velocity sampling, the frozen face-support cache, and sharpening.
The eight stored donor slots and CM12 beta/deficit passes remain unchanged.
Native staggered face sampling still preserves a zero-length characteristic.

The locator walks actual cell boundaries in the direction of an exit. Taking
the smallest width anywhere in the stencil could instead step to a coordinate
that was not a primal vertex. Newton's iterates remain inside the parameter
cube to avoid folded extrapolated roots and singular wedge edges; the final
weights use the solved parameter and must satisfy a geometric residual check.
A failure still halts the simulation instead of substituting a donor return.

Exact cell-centre queries use the cardinal nodal weights algebraically,
while retaining the surrounding donor IDs for the existing clearance
certificate. A geometric residual tolerance alone does not guarantee exact
zero-characteristic identity at a wedge apex. All 43,600 actual cell centres
in the mixed/clipped fixtures must have exactly one aggregated donor of
weight one; no neighbour-weight tolerance is used.

A later 4/2/1-corner fixture exposed cancellation in the Jacobian: summing
signed node positions could erase a small nonzero derivative beside repeated
nodes. Forming opposite-node edge differences before weighting them avoids
that cancellation. The reported failing point and 8,192 surrounding queries
pass without increasing the iteration count or relaxing the residual bound.

Physical boundary ghost centres mirror the interior geometry while retaining
its values, implementing a Neumann continuation. Clipped owner queries reject
coordinates outside an axis or beyond the actual final cell; an in-range
flattened index alone could alias the next row. These checks preserve dynamic
SparseWorld expansion beyond the original authored domain.

## Physical adjacency and exact receipts

A multi-term pressure row represents two sides of a face. Its membership is
not a clique: fine cells on the same side do not exchange mass or velocity
through that row. Opposite-sign gradient coefficients identify physical pairs.
Their area fractions recover each pair's physical subface area.

Velocity extension uses subface area divided by row distance. Capacity repair
splits its excess according to open physical area. Four fine children across
one coarse face collectively receive that face's share, instead of four votes
against one vote from an equally large unsplit face.

Capacity repair credits integer receipts and debits their exact sum. Equal
area portions use integer division, avoiding an upward-rounded f32 quotient
on large receipts. Remaining integer residue stays at its source; no final
catalogue entry receives a privileged remainder. The existing eight repair
rounds remain unchanged, and the gather QA variants use the same receipts.

## Sparse support exposed by the integration runs

After the geometry fixes, mini32 initially stopped at frame 2 with
`EMPTY_DEFICIT_STENCIL`. A diagnostic capture showed nonzero density near
0.002 travelling from a finest cell centred at x=23.5 to x approximately
24.6, into an inactive page. Both direct and staged owner queries agreed that
the page was inactive. The old three-step run passed, so this was a regression
to address, not a failure to dismiss as an empty donor.

The residency census predicted travel from visible surface cells and thin
features, whereas CM12 deficit transport includes **every nonzero donor**.
The existing census now includes nonzero donors' adjacent interpolation
support and predicted sweep. These requests remain valid below the visual
feature threshold. Keeping only a swept endpoint was insufficient: a later
capture found a single fixed-point density quantum crossing into an inactive
page at frame 11. The adjacent support closure handles that case too.

The adjacent mass halo stays inside the resident world: interpolation outside
that boundary uses its existing ghost values. Only actual swept motion can
request outward growth. Treating stationary interpolation support as a growth
request opened new air pages beside deep liquid and broke the coarsening and
world-page-budget checks; that is a distinct demand-semantics error.

The wider support also exposed a packed-word error: extracting flags by
shifting the second word by 22 allowed its support bits 22–26 to become
surface/occupancy flags. The decoder now extracts only the high flag bits.
The regression covers all 256 flag combinations and every support direction.

Incoming-surface anticipation now applies to a surface or an empty/dilute
receiver. Residency demand alone cannot apply that refinement floor inside
fully flooded bulk. A control using the earlier support census isolated this
coupling; the full two-second deep-water coarsening test and the incoming
relative-motion prediction test pass after separating those meanings.

Surface classification still controls the visible-feature refinement score.
The exact-zero exception in the deficit validator is unchanged; nonzero mass
without recipients continues to halt. No additional GPU dispatch was added.

## Validation methodology

Focused Dawn tests execute the production interpolation and receipt functions.
They cover affine reproduction, positive partition, aggregated donor-weight
covariance under reflections and coordinate permutations, graded 1/2/4/8
junctions, clipped geometry, narrow wall wedges, zero-time staggered identity,
finite wave translation, physical face-area shares, and exact integer receipt
conservation. Affine reproduction is tested between physical sample centres;
normal affine continuation beyond the last centre is not a Neumann boundary
condition.

Frozen A/B captures use 240 steps at 1/30 s, pressure iteration ceiling 128,
and relative pressure tolerance 0.001, matching the earlier baseline. Source
hashes for every adaptive-mass TypeScript module and the shared numerical
constants are recorded and checked before and after capture, and between A
and B. The analysis verifies identical initial fields and unchanged accepted
leaf/rung rosters. Quantities are integrated from simulation density/velocity,
not measured from rendered surfaces.

The retained symmetry limits are density maximum 0.01, density mean 0.001,
velocity maximum 0.02 m/s, and velocity mean 0.001 m/s. The reference max1 arm
is a discretization for comparison, not physical ground truth. Lower kinetic
energy alone does not establish improved physical accuracy.

## Frozen A/B results

Both final runs completed all 240 steps without a solver fault, with identical
initial fields, matching source hashes, and unchanged topologies. The final
captures are in `artifacts/pool-impact-symmetry/mechanism-final-max1` and
`mechanism-final-max2`; the analysis and before/after figure are in
`mechanism-final-analysis` and `mechanism-final-comparison`.

| Time | Before KE A / B (J) | After KE A / B (J) | Before column RMS gap (mm) | After gap (mm) |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 s | 9.470 / 13.370 | 9.470 / 8.264 | 20.63 | 14.95 |
| 1 s | 6.665 / 8.088 | 6.665 / 1.972 | 28.21 | 20.75 |
| 2 s | 5.059 / 4.190 | 5.071 / 1.876 | 25.60 | 25.71 |
| 4 s | 2.368 / 5.292 | 2.343 / 2.719 | 23.87 | 20.72 |
| 8 s | 2.182 / 10.548 | 2.141 / 4.177 | 41.90 | 31.74 |

The later A/B kinetic-energy disagreement improves substantially: its mean
absolute value over steps 121–240 falls from 7.378 J to 1.269 J. At 8 s the
column-depth RMS gap falls 24.2%, and the discrepancy in mass distribution
restricted to width-two blocks falls from 8.206% to 6.935%.

This is **not uniformly better agreement**. B loses much more kinetic energy
than A near 1 s: the mean absolute energy disagreement over steps 9–30 grows
from 2.918 J to 4.167 J. The width-two mass-distribution discrepancy at 4 s
also grows, from 4.252% to 4.790%. A's early energy is almost unchanged.
The mechanisms are corrected, but additional damping in B cannot be declared
physically accurate from this comparison alone.

B's symmetry improves markedly. Its first maximum-velocity limit violation
moves from step 41 (1.37 s) to step 83 (2.77 s); its first maximum-density
violation moves from step 42 to step 84. Mean density error averaged over all
240 steps falls from 0.007530 to 0.001233. At 8 s, B's maximum density error
falls from 0.3200 to 0.06359 and maximum velocity error from 0.6841 to
0.2140 m/s. A does not show a comparable symmetry improvement.

Neither arm satisfies all retained symmetry limits through 4 s. At 4 s,
maximum density errors are 0.1654 / 0.1293 and maximum velocity errors are
0.4456 / 0.5373 m/s for A / B. A's first density-maximum violation occurs at
step 50 and velocity-maximum violation at step 49. These failures remain
visible; no threshold was raised.

Both arms also retain the previously observed upward density redistribution.
Initial mechanical energy above the discrete resting minimum is 41.39 J;
at 8 s it is 311.77 J / 333.02 J, despite the absence of external work. That
is still a physical defect. Final mass errors are -0.02810% / +0.00651%,
so total mass conservation alone does not explain or excuse the energy drift.
The next physics investigation should separate sharpening/capacity movement
from pressure's excess-volume feedback using the existing stage captures,
and reconcile pressure membership with velocity-extension seeding.

![Before and after frozen A/B](../artifacts/pool-impact-symmetry/mechanism-final-comparison/before-after.png)

## Regression status at the end of this implementation

The integrated geometry, exact nodal identity, physical-subface receipt,
activity-mask, face-remap, incoming-front prediction and four-second mini32
conservation tests pass. The full two-second hydrostatic/deep-bottom test
passes when run without the suite's shorter process timeout. The frozen
coarse quarter scene passes its 52-step short regression; the changing-topology
18-step lane fails velocity symmetry starting at step 6. Its pre-change
attribution has not been established.

The canonical suite remains red (`mechanism-final-gate.json`): symmetric
expansion fails density symmetry, several lanes exceed their original process
timeouts, and the 180-second total budget prevents the tail from running.
Individual tail runs pass mini64 min8 surface, live rigid coupling, live liquid
injection and outside-tank collapse. The long dam reports missing deficit
support at frame 13; Tall Cells Hills exceeds its time limit. These remain
unresolved and are not covered by the successful frozen pool A/B.

The subsequent investigation moved to
[minidam32 frozen topology](minidam32-frozen-correctness-2026-09-07.md), per the
user's direction. It includes the first surface-boundary repair and records
the remaining freeze-residency and pressure-interface problems.
