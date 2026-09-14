# Level-set-plus-volume stalled front

Date: 2026-09-14.

## Finding

The gentle moving blob stalls at fine-to-coarse brick boundaries because the
level-set-plus-volume gather samples four finest-lattice points around each
receiver centre, then maps those points to adaptive owner cells. It does not
sample or integrate the receiver footprint.

For the width-2 cell immediately beyond the observed `x = 24` fine-coordinate
boundary, the receiver centre is `x = 25`. The measured velocity is
`1.5999999` fine cells/s and `dt = 1/30` s, so its backward departure is

```text
25 - 1.5999999 / 30 = 24.9467.
```

The gather's x samples are therefore `24.5` and `25.5`. Both lie inside the
same width-2 cell `[24, 26]`, so owner coalescing reduces the row to a self
edge. An empty coarse receiver consequently gathers only from itself and
cannot acquire liquid. Repeating the step does not accumulate subcell travel:
each frame traces again from the same receiver centre.

The same dead zone applies on both axes. With diagonal motion whose per-frame
x and y displacements are each below half a finest cell, all four bilinear
samples remain inside a width-2 or wider receiver. This explains the observed
coarse diagonal bricks that never gain volume.

## Historical pre-fix evidence

The final-state capture contains adjacent cells with the same transported
velocity but radically different volume:

| Cell | Minimum | Width | Density | x velocity |
| --- | --- | ---: | ---: | ---: |
| 215 | `[23, 12]` | 1 | `3.400489` | `1.5999999` |
| 248 | `[24, 12]` | 2 | `1.73e-10` | `1.5999999` |
| 249 | `[26, 12]` | 2 | `0` | `1.5999999` |

All 12 over-capacity cells have minimum x coordinate 15 or 23: immediately
before the coarse boundaries at x = 16 and x = 24. The largest fill is about
`7.064 K`. These locations are the expected accumulation sites when donor
mass is conserved but the gather graph has no receiving edge across the next
coarse boundary.

The redistanced run retains 111 finest-cell units of authoritative liquid, but
its final RDF receives only 20.637 units after the existing phi mask. The raw
RDF leading x position moves from 0.041198 m to 0.080728 m, only 39.53 mm,
while ideal translation is 480 mm. The UI can appear to reach the x = 0.4 m
boundary because `drawSlice` paints accepted full and overfull fine cells
directly, bypassing RDF geometry. Authoritative volume, raw RDF geometry, and
the displayed full-cell fallback are therefore three distinct fronts in this
case.

The physical-volume centroid after 180 frames is `0.1388179645` m with
redistancing and was `0.1388179758` m before it. Redistancing changed the
distance magnitude and repaired the internal RDF hole, but did not alter the
gather stencil or the stalled material motion. The transport dead zone
predates redistancing.

The supporting captures are
`artifacts/level-set-volume/gentle-moving-blob-redistanced.json` and
`artifacts/level-set-volume/gentle-moving-blob-redistanced-final-state.json`.

## Remedy scope

A numerical remedy must give an adaptive receiver nonzero upstream support
whenever its backward-shifted footprint overlaps an upstream donor. The
coherent scope is a receiver-footprint overlap or integrated conservative
kernel in finest-coordinate space, with coarse/fine measure weighting and
the existing exact donor-mass requirement. It must cover axis-aligned and
diagonal motion and remain valid at every 2:1 boundary.

Scaling the point stencil by receiver width or forcing the swept destination
band to finest resolution can remove this particular dead zone, but neither
makes a centre sample a coarse/fine-consistent cell-average transport. More
normalisation passes also cannot create an edge absent from the raw support.
Redistancing, phi sharpening, the RDF mask, and pressure draining are separate
concerns and cannot move material through this missing connection.

## Implementation addendum — 2026-09-14

The replacement support construction backward-translates each receiver's full
axis-aligned cell box by the displacement from its centre to the existing RK2
landing. A compact spatial-bin index supplies candidate donor leaves. Each raw
edge is the exact `f64` rectangle-intersection area between that translated box
and a candidate leaf, after clipping to the physical domain. Thus the example
width-2 receiver has positive overlap with the upstream width-1 leaf even when
both of the old centre samples belonged to the receiver itself. The same area
construction supplies edge and corner donors for diagonal translation.

This follows the repository's conservative transfer geometry: `plan_transfer`
in `rust/crates/fluid-core/src/transfer.rs` bins source cells, deduplicates the
candidate set, and computes exact axis-aligned overlaps.

The production adaptive-mass correspondence is in
`lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts`.
`traceGammaAndBeta` calls `effectiveTransportStencilAtSpans` with the receiver's
`cellWidths`. That sampler holds the receiver-owned lattice spacing fixed for
the whole trace, specifically to avoid the hard-coded finest-lattice half-cell
dead zone. For the width-2 example at departure `24.9467`, its x probes are 23
and 25, so it includes the upstream fine owner immediately. It filters donors
through `cellTransportActive`; `cm12VolumeWeightedBetaContribution` then applies
receiver/donor physical cell volumes. The CPU analogue `sampleWeights` in
`lib/methods/adaptive-mass/sparse-atlas-cm12-transport.ts` likewise chooses an
adaptive sampling span.

The level-set-volume change has the same goals of adaptive-scale support and
measure-aware conservation, but uses exact translated-box overlap instead of
adaptive-mass's span-aware point interpolation. Its `capacity[i] * cell.measure`
marginals provide the measure authority. Only the older fine-lattice helper in
`lib/methods/adaptive-mass/sparse-brick-translation.ts` exhibits the specific
hard-coded point-stencil limitation diagnosed above; it is not the production
resident transport and is not the geometry copied here.

Raw overlap is geometric area. It is not multiplied by either cell's open
fraction: the existing capacity-marginal normalisation already scales rows and
columns to physical open capacity. A zero-capacity donor is not useful support,
and a zero-capacity receiver has a zero marginal. Fractional capacity remains a
cell-average constraint; the available fields do not identify which subregion
of an overlap is open, so this change does not claim exact cut-cell overlap or
swept solid geometry.

The transported box is a rigid local translation based on the receiver-centre
trace. It does not deform under velocity gradients, trace corners separately,
or add swept collision handling for internal solids. Domain clipping and the
subsequent row scaling retain the current closed-domain boundary convention.
The three alternating row/column normalisation passes, final exact donor
normalisation, self-edge fallback for uncovered donors, and their known
receiver-marginal residual and first-moment limits remain unchanged.

## Post-change validation — 2026-09-14

The 180-frame native and SIMD arms in
`artifacts/level-set-volume/gentle-moving-blob-receiver-support.json` both
completed with a null failure. The targeted native suite passed 23 tests: 11
library tests, 9 level-set-volume tests, and 3 baseline tests. All four Wasm
flow tests also passed, covering scalar and SIMD resolution edits, the 30-frame
Figure 7 lane, and the 10-frame pool lane.

The original boundary regression transferred zero volume into the empty
width-2 receiver, compared with an overlap oracle of `0.8`. With translated-box
support the axis result is `0.79999924`. In the full moving-blob run, liquid in
width-2 cells increased from `0.286665` to `25.18211995` finest-cell volume
units. This directly confirms that coarse receivers now acquire material across
the boundary that previously had no gather edge.

| Quantity after 180 frames | Historical point support | Translated-box support |
| --- | ---: | ---: |
| Physical centroid error | 91.182 mm behind | 6.476 mm ahead |
| Total over-capacity volume | 34.15254 | 0 |
| Raw RDF maximum x | 0.08072815 m | 0.34536734 m |
| Raw RDF represented area | 18.78718 | 50.19709 fine-area units |
| Final accepted cells | 352 | 480 |
| Median volume-gather time | 0.027166 ms | 0.18925 ms |

Initial authoritative volume was 111 finest-cell units and the new final total
was `110.99999863829`, a relative drift of `-1.2268e-8`. The timing values are
measurements of two runs whose evolved topologies differ, so they are not a
controlled microbenchmark of support construction alone.

The transport improvement does not establish that the reconstructed shape is
correct. The selected-section ideal leading x is `0.52895652` m, while the new
raw RDF reaches `0.34536734` m. Its represented area is `50.19709` fine-area
units against 111 units of authoritative volume. Translated-box support removes
the measured adaptive transfer dead zone and the resulting excess in this run;
RDF under-representation and the remaining phase/geometry discrepancy remain
separate unresolved observations.
