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

## Continuing the audit

The ordinary 2D pressure operator and velocity projection now reduce mixed
seam terms by physical side inside their existing row traversal. Previously,
mirroring a three-term seam changed the addition association. A focused test
with mirrored pressures produces jumps of `-0.00390625` versus zero under the
old reduction and reflected velocities under the corrected reduction. This
change adds no solver pass or field averaging.
The 3D counterpart preserves the existing term traversal and uses the graded
fine-side pairing order. Explicit `fma(value, 1, other)` pair sums prevent Metal
from reassociating the intended reduction; no sorting or additional dispatch
is used. The extracted production helper passed its isolated Dawn reflection
test. Its exact saved version was restored after cleanup; a repeat run was
blocked by another task's live GPU lease.

ELVIRA also skips a candidate integration axis when its orientation component
is exactly zero. The former positive-axis default could reconstruct an exactly
X-invariant near-full 3×3 stencil with `nx = +1`. Skipping that unsupported
candidate reduces work and restores `nx = 0` in the regression fixture.

## Remaining transport asymmetry

After the pressure and candidate-axis fixes, maximum density mirror error at
frame five is `1.192e-7` with 28 pressure iterations and `8.941e-8` with 64.
Frame six still reaches `0.0205731` and `0.00448412`, respectively, while the
topology has zero mirror mismatches. The latter improves from the earlier
`0.0217323`, but the default 28-iteration scene remains materially asymmetric.

The residual passes through full/partial reconstruction and flux branches,
but its root cause was not isolated reliably. Wider clamps, weak-gradient
gates, phase-bracketing gates, reordered transport sums and removal of the
corner fallback either failed correctness checks or worsened other checkpoints;
they were reverted. No averaging, additional solve iterations or simulation
passes were introduced. Investigation stopped at the user's request.

## Regression scope

The admission tests exercise equivalent and asymmetric requests and small
budgets. The RDF tests reflect nonlinear fields and check matching geometry,
area and scalar samples. The stage probe measures the actual production scene,
rather than imposing symmetry on its outputs.

Before this audit's edits, the advance-slice and lens CPU suite had 95 passes
and five failures: automatic water-box rerung, unresolved published partial
planes, retained pressure generations, automatic pressure mappings and a
policy source fingerprint. These failures are not relaxed by this work.

The broad CPU set before the pressure follow-up had 114 passes and four
existing failures. It adds
the admission, reconstruction, display and contour regressions. The impact
RDF fixture advances five frames to retain its original coverage of at least
50 transported near-full cells; its phase and area assertions are unchanged.
The final six focused stage tests pass, including the reflected mixed-seam
pressure and exact-zero candidate-axis regressions.

The final full Dawn gate has 3 passing and 14 failing/timed-out lanes, taking
442.2 seconds within its unchanged 480-second suite budget. Before the certificate change,
a controlled rerun with the old admission code reproduced the hydrostatic halt
at frame 4, generation 5, owner 892 (`computeGeometricVolumeFluxes`, operands
`9,1,0,1`). The final certificate-enabled run also halts in that kernel at
frame 4, generation 5, now owner 3160. Other failing lanes and timeouts have
not all been independently attributed. No lane or timing ceiling was relaxed.
Repository type checking also reports existing errors outside these changes.
