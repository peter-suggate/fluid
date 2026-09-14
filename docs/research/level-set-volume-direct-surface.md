# Level-set-plus-volume direct surface

Date: 2026-09-14.

The level-set-plus-volume path now keeps two independent authorities. Cell
volume `V` is transported conservatively and answers mass questions. The
advected signed-distance field `phi` supplies the visible surface through its
zero set. Every advance traces both fields with the same velocity sampler and
RK2 rule: `V` traces adaptive cell centres, while `phi` traces fine-grid
vertices. The `phi` trace queries exact signed distance to the accepted source
contour, preserving that source contour as the distance authority, then
publishes the sampled vertex field and its centre-fan zero contour directly.
Vertex resampling still introduces the usual level-set discretization and area
drift; the published discrete contour is not claimed to preserve the advected
continuous zero set exactly. The stored vertices are therefore not guaranteed
to be a global signed-distance field. The next consumer queries exact distance
to this accepted discrete contour, which provides reinitialization before use
without adding a second resampling pass that could move the accepted zero set.

The initial `phi` is seeded once from the accepted occupancy at the `V/K = 0.5`
boundary. This bootstrap does not continue during motion. After initialization,
`V/K` does not fit a PLIC plane, mask cells for surface reconstruction, adjust
the zero set, or correct level-set area. Consequently the method conserves
`V` while permitting the level-set area to drift; that difference is an
explicit diagnostic rather than a reconstruction error to force to zero.

The implementation boundary is `levelset_surface::{initialize_from_volume,
publish, refresh, cell_phi}`. `levelset_volume::advance` continues to return an
`RdfSurface`-shaped publication for ABI compatibility, but its
`vertex_phi_fine` is the authoritative direct scalar field and
`segments_fine` is its zero contour. The receipt's `exactAreaFine` is conserved
volume used as a reference, `representedAreaFine` is direct level-set area, and
`signedAreaErrorFine` is level-set area minus volume.

Pressure compatibility planes use local tangents derived only from the
cell-centred `phi`. They do not feed back into the published surface.

The lab therefore draws only `vertexPhiFine` and `segmentsFine` in level-set
surface views. It never replaces full cells or nonfinite scalar regions with
volume fill or PLIC geometry. The conservative `V/K` opacity view remains a
separate transport diagnostic with the direct zero contour drawn as a
reference. PLIC selection, interface-normal overlays, plane probes, plane
timings, and plane-seam diagnostics are unavailable for this method because
they are not part of its surface pipeline.

This change removes the measured feedback loop in which diffuse volume
fractions generated many local planes, a phi cutoff discarded positive volume
from geometry, and the resulting RDF zero set became the next phi source. It
does not sharpen the level set, reconcile level-set area to conserved volume,
or make the two fields describe an identical shape.

## Validation

The 180-frame, 6 s `gentle-moving-blob` result is recorded in
`artifacts/level-set-volume/gentle-moving-blob-direct-surface.json`. The direct
contour retained a nearly circular shape: circularity was `0.9850` at frame
180, and its centroid was `1.256 mm` from the translated reference. At the same
frame in the preceding receiver-support artifact, the reconstructed surface
had circularity `0.4074` and centroid error `80.158 mm`. The corresponding
frame-171 comparison was `0.9834` and `1.124 mm` for the direct surface versus
`0.4233` and `59.266 mm` previously. The conservative-volume centroid error at
frame 180 also decreased from `6.476 mm` to `4.749 mm`.

This does not resolve level-set area loss. Direct represented area fell from
`109.0964` fine-area units initially to `63.6367` at frame 180, a `41.7%`
shrink, while conservative `V` ended at `111.00000047` with relative drift
`4.25e-9`. The result isolates scalar transport and vertex/contour resampling
as the remaining surface-loss mechanism; no area correction or extra
redistancing pass was added.

Validation passed 31 targeted native tests, 12 UI lens/playback tests, all four
Wasm flow tests, and the scalar, SIMD, and threaded production builds. The
served Wasm hashes matched the build outputs and the recorded Rust source
fingerprint.
