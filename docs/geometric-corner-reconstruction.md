# Detached-box reconstruction concern

The all-fine constant-velocity plug passes the analytic translation criteria.
Adding detached corners preserves velocity, energy, volume and centroid, but
produces a 3.125% volume-field L1 error. The following calculation predicts the
observed error from the production reconstruction algorithm without fitting a
parameter to the result.

## Exact prediction

Use finest-cell units. The initial rectangle occupies x∈[2,6], y∈[2,6],
extruded over eight z cells. Each internal step translates it by 0.5 cell.
After one step, its front corner cell (x=6,y=2) is half full. Its face-neighbour
fractions are x−=1, x+=0, y−=0, y+=0.5; the z neighbours are also half full.
The uniform-grid least-squares volume gradient is therefore (−0.5,0.25,0),
and the reconstructed outward normal is (2,−1,0)/√5. The plane offset is zero.
The exact interface inside that cell is vertical, with outward normal (1,0,0).

During the next half-cell sweep, the reconstructed liquid intersects the right
half of the cell in a triangle of area 1/16. The geometric flux routine thus
sends 1/16 cell volume through a face that should carry no liquid yet. The rear
corner has the complementary error: it sends 7/16 instead of 1/2. The FCT
bounds allow both fluxes, because neither causes a negative or overfull cell.

| Outgoing +x face | Low flux | Predicted geometric high flux | Exact flux |
| --- | ---: | ---: | ---: |
| Rear corner, cell (2,2) | 1/4 | 7/16 | 1/2 |
| Front corner, cell (6,2) | 1/4 | 1/16 | 0 |
| Rear interior, cell (2,3) | 1/4 | 1/2 | 1/2 |
| Front interior, cell (6,3) | 1/4 | 0 | 0 |

All entries are normalized by a finest cell's volume. After two half-steps,
edge-column fills at x=2,3,6,7 become [1/16,15/16,15/16,1/16] instead of
[0,1,1,0]. There are two y edges, eight z layers, and four affected cells per
column. Therefore the relative volume-field L1 error is

```
2 × 8 × 4 × (1/16) / (4 × 4 × 8) = 0.03125.
```

The next half-step reverses these paired errors, explaining the measured
alignment-dependent oscillation without momentum or energy loss.

![Exact and reconstructed corner geometry](../artifacts/analytic-motion/corner-reconstruction.png)

## Why this warrants a method review

The shared flux calculation and limiter can be internally correct while the
plane they transport is wrong. A gradient of cell-average volume fractions is
not generally the geometric surface normal. This is a reconstruction limitation,
not a demonstrated face-index, sign, conservation or pressure-solver defect.
Snapping the normal to the flow direction would fit this example but would not
establish correctness for oblique interfaces or general flows.

A useful next design criterion is exact reconstruction of planar interfaces
from cell-integrated volumes, followed by explicit treatment of corner and
thin-sheet stencils. Volume-matching approaches such as LVIRA/ELVIRA provide
relevant precedent: they fit reconstructed volumes under the central-cell
volume constraint. Their smooth-interface accuracy does not by itself promise
exact preservation of this detached corner. See [Pilliod and Puckett (2004),
sections 2.5–2.6](https://www.math.ucdavis.edu/~egp/PUBLICATIONS/JOURNAL_ARTICLES/APPEARED/2004/JEP-EGP-2004.pdf).

This concern applies to the interface-normal reconstruction, not to the choice
of sparse bricks or the algebraic conservation of shared volume fluxes. The
steady-vortex damping remains a separate, already isolated velocity-advection
and pressure-coupling issue.

## The concern also exists for a straight interface

A separate exact calculation applies the same uniform-grid stencil to a single
plane, so the limitation cannot be attributed solely to a nonsmooth corner.
For liquid `2x+y <= 0` in centred unit cells, the centre fill is 1/2 and the
neighbour fills are `(x−,x+,y−,y+)=(1,0,15/16,1/16)`. The production LS stencil
therefore returns outward direction `(1/2,7/16,0)`, whose y/x ratio is 7/8.
The true normal is `(2,1,0)`, with ratio 1/2. The angular error is 14.62 degrees.
These fractions follow from exact plane–square intersections. This is a direct
calculation of the code's stencil, not an additional GPU scene measurement.
At this relative cell alignment the error is independent of the physical cell
size. It strengthens the case for replacing the normal estimator rather than
tuning a tolerance around the detached-box result.

## Dawn confirmation

The two-frame `--detached --dt=0.004166666666666667 --duration=0.008333333333333333 --transport-detail`
run confirms the calculation. Both frames execute exactly one volume microstep.
The published half-filled front normal is
`(0.8944271803, -0.4472135901, 0)`, and the rear normal has the opposite x sign;
both offsets are zero. In the next frame the outgoing front and rear high
fluxes are exactly `0.0625` and `0.4375` finest-cell volumes. Their limited
fluxes are identical, and the sweep is exactly `0.5`. The four measured edge
fills are exactly `[0.0625, 0.9375, 0.9375, 0.0625]`.

The analytic shape criterion correctly remains failed. Source fingerprints
are unchanged and the physical bounds checks pass. Full source receipt:
`artifacts/analytic-motion/corner-transport-detail.json`. A compact comparison
against the independently predicted values is
`artifacts/analytic-motion/corner-reconstruction-confirmation.json`.

That run marked the requested stopping point for the diagnostic increment. No
normal snapping or tolerance change was introduced to make it pass. The result
and its receipts remain the before-change explanation of the defect.

## Volume-consistent 2D reconstruction

The production reconstruction now recognizes certified open, uniform,
z-extruded 3×3 stencils and evaluates six ELVIRA-style height-difference
candidates. Every candidate fits the central cell's stored volume. A candidate
replaces the displacement-aware least-squares fallback only when it improves
the full stencil fit; a one-sided corner fit is accepted only at geometric
evaluation roundoff. Mixed-resolution, cut-cell and general 3D stencils retain
the existing fallback.

An independent Dawn probe seeds exact CPU-integrated plane fractions into both
production scalar banks, executes the real interface-cache refresh through the
`pressure-topology` stage, and reads the resulting planes. All eight axis,
oblique, offset and signed 2D cases pass. Across those cases the maximum normal
angle error is 1.71e−6 degrees, the maximum central-volume mismatch is
3.07e−8, and the maximum 3×3 neighbour-volume prediction error is 8.86e−8.
The predeclared limits are 0.1 degrees and 1e−5 central-volume mismatch.
Receipt: `artifacts/analytic-motion/plane-fit-2d.json`.

The original detached-box case now passes without changing its analytic
criteria. At the accepted clock, the two-half-step shape L1 error is
4.58e−16; evaluated at the cumulative executed f32 physical clock it is
2.61e−8. Volume, centroid, velocity, kinetic energy, accepted bounds, all-fine
topology and source fingerprints also pass. A CFL-2 frame with four geometric
substeps finishes at 5.62e−16 accepted-clock shape L1. Receipts:
`artifacts/analytic-motion/corner-fit-detail.json` and
`artifacts/analytic-motion/corner-fit-cfl2.json`.

This resolves the demonstrated 2D planar and detached-corner failure. It does
not establish a 3D reconstruction result: the separately captured pre-change
3D cases have 12.4–15.3 degree normal errors and remain the baseline for the
3D work. Receipt: `artifacts/analytic-motion/plane-fit-3d-before.json`.
