# Advance-lab reflection audit

`coarse-first-pool-impact-half` is centred on the vertical axis. Reflecting x
about half the domain width should preserve density, pressure, vertical
velocity and topology, and negate horizontal velocity.

Run the CPU stage audit with:

```bash
node --import tsx tools/probe-advance-slice-reflection.ts --frames=6 --pressure=64
```

The JSON lines report the first stage at which reflected fields disagree.
Compare several pressure iteration budgets: small rounding errors can cross
later classification thresholds, so a short run at one budget is insufficient.

## Sources found

- **Topology admission order.** A key-order budget can admit one requested
  demotion before its reflected equivalent. Both 2D and bounded 3D admission
  now group equivalent reflected requested transitions. Unspent selection
  credits let an orbit progress even when one frame's increment is too small.
  These are selection credits; the independent generation-preparation time
  limit is unchanged. Signed-coordinate 3D domains retain their existing
  admission path.
- **Fixed display diagonals.** Splitting every scalar square along the same
  diagonal is not invariant under reflection. Canvas fill, contour extraction
  and represented-area integration now share a four-triangle centre fan.
  The centre uses the bilinear centre value. This preserves affine fields and
  does not alter the accepted liquid volume.
- **Inconsistent reconstruction certification.** Transport accepts bounded
  float32 volume roundoff, but the ELVIRA stencil required every fill to be
  strictly in `[0, 1]`. A value of `-1.862645149e-9` versus its mirrored zero
  disabled ELVIRA on just one side. Certification now uses the existing
  transport roundoff ratio, `9.5367431640625e-7`, and clamps only reconstruction
  samples. Conserved density is unchanged. In controlled runs at 28 and 64
  pressure iterations, frame-four maximum density mirror error drops from
  `0.005383` to `1.1920929e-7`.

The 3D uniform-extrusion and full-volume reconstruction paths use the same
observation policy. Out-of-tolerance values, NaN and infinities still fail
certification. The production WGSL helper has a direct Dawn boundary test.

## Remaining transport asymmetry

At 64 pressure iterations, maximum density mirror error is `3.576e-7` after
five frames, but reaches `0.0217323` during frame six, while the topology still
has zero mirror mismatches. Microstep zero leaves one cell at `0.9999976158`
versus its mirrored `1`; subsequent full/partial reconstruction and flux
branches amplify the difference. These values exceed the existing roundoff
margin, so this work does not discard them using a wider clamp. The following
one-sided topology demand is downstream of the transport error.

## Regression scope

The admission tests exercise equivalent and asymmetric requests and small
budgets. The RDF tests reflect nonlinear fields and check matching geometry,
area and scalar samples. The stage probe measures the actual production scene,
rather than imposing symmetry on its outputs.

Before this audit's edits, the advance-slice and lens CPU suite had 95 passes
and five failures: automatic water-box rerung, unresolved published partial
planes, retained pressure generations, automatic pressure mappings and a
policy source fingerprint. These failures are not relaxed by this work.

The final focused CPU set has 114 passes and four existing failures. It adds
the admission, reconstruction, display and contour regressions. The impact
RDF fixture advances five frames to retain its original coverage of at least
50 transported near-full cells; its phase and area assertions are unchanged.

The final full Dawn gate has 3 passing and 14 failing/timed-out lanes, taking
442.2 seconds within its unchanged 480-second suite budget. Before the certificate change,
a controlled rerun with the old admission code reproduced the hydrostatic halt
at frame 4, generation 5, owner 892 (`computeGeometricVolumeFluxes`, operands
`9,1,0,1`). The final certificate-enabled run also halts in that kernel at
frame 4, generation 5, now owner 3160. Other failing lanes and timeouts have
not all been independently attributed. No lane or timing ceiling was relaxed.
Repository type checking also reports existing errors outside these changes.
