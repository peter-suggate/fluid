# Sparse geometric remap performance work: paused notes

## Status

Performance work stopped after S0 measurement and before any S1-S5 change was
accepted. The production remap source is restored to the S0 timer-only state;
post-restoration production artifact validation was still pending when these
notes were finalized. The S1-S5 code described below was an experimental
prototype; it was not validated for shipping, was not used to publish
production artifacts, and must not be read as the current algorithm.

The accepted S0 measurements, source hashes, binary hash, and reproduction
commands remain in
[`sparse-geometric-remap-performance-results.md`](./sparse-geometric-remap-performance-results.md).
The original review and staged optimization request remain in
[`sparse-geometric-remap-review-handoff.md`](./sparse-geometric-remap-review-handoff.md).

The experimental working-tree patch is preserved at [`docs/research/sparse-geometric-remap-s1-s5-paused.patch`](./sparse-geometric-remap-s1-s5-paused.patch) (SHA-256 `23068f6184651afd4f68d6bd0cf6a2f8aaf943c18835feb7c7683d5389bf447b`). The patch contains only the paused remap experiment; concurrent level-set work was removed from the recovery artifact.

## Measured S1 and S2 result

S1 removed harmonic continuation and made a trace outside represented
streamfunction support fail before material mutation. S2 restricted tracing
to faces incident to the existing receiver band and removed several fixed
diagnostic costs. Neither stage passed its go/no-go scenes:

- the half-pool rejected frame 8 with 3,171 actual RK support exits;
- Figure 7 rejected frame 3 with 64 actual RK support exits.

The exact coordinates, source hashes, compact artifact paths, and timing
through rejection are recorded in the S1/S2 section of the performance-results
document. These failures disproved the assumption that the old zero fallback
counter meant harmonic continuation was unused: the old full-domain active
mask had hidden the unsupported traces.

## Material-causal trace-band prototype

A later unaccepted prototype kept the broad closure band but selected tracing
work with a forward material-reach enclosure. It converted each represented
unit bicubic streamfunction patch to Bernstein derivative control bounds for
`u = psi_y` and `v = -psi_x`, seeded old donor PLIC support, carried clipped
sub-boxes between quarter-cell time slices, and traced the complete physical
face star of reached receiver owners.

Measured on half-pool frame 8 after the gauge repair below, selection took
about 3.09 ms. It used six time slices and about 156,000 patch visits. The
initial trace band contained 663 cells, versus 1,010 in the broad closure band,
and its exact selected RK traces had zero streamfunction support exits.

The experiment did not establish a complete conservative selector:

- interval AABBs contacted SparseAir 4,352 times even though exact selected
  traces had no support exit, so an enclosure contact is uncertainty rather
  than proof that a characteristic left support;
- skipping an unknown patch can under-approximate reach if a true path could
  leave and later re-enter represented support;
- corrected edge chains can move beyond the uncorrected flow enclosure;
- geometry identified four material receivers outside the selected trace band
  and rejected the frame.

The strict actual-trace, receiver, fold, and donor-conservation gates remained
useful fail-closed checks, but they do not turn this selector into a proof of
global support completeness. The prototype and its geometry-band behavior
need a fresh mathematical review before reuse.

## Two-component streamfunction gauge defect

The half-pool frame-8 pressure graph had exactly two disconnected pressure
components: 160 pool cells spanning `y = 0..16`, and 226 falling-droplet cells
spanning `y = 21..40`. The extension code seeded each hard streamfunction
component independently at zero. That fixed an arbitrary relative additive
gauge and forced adjustable air-gap edges to bridge the resulting potential
offset.

At unit patch `[39,21]`, the uncorrected prototype reported a maximum velocity
Jacobian Frobenius norm of `384.3313094`. Its corner data were:

```text
psi = [494.8138130272, 481.5911015692, 627.6255774077, 570.6854277060]
dx  = [  1.6936422997, -12.9515625400,  -2.3190090244, -40.0179224766]
dy  = [117.8954106183,  89.3654750476,  66.4058820966,  72.1720989217]
```

The corresponding face-mean rates were approximately `u = 132.812/89.094`
and `v = 13.223/56.940`, while the projected local physical field was nearly
uniform `(0, -52.302)`. This was a remap-private extension artifact, not liquid
strain from the primary velocity field.

An unaccepted weighted-DSU quotient prototype retained every hard pressure and
`ClosedWorld` potential difference, solved one additive offset per hard
component through the adjustable-edge quotient graph, and pinned one gauge per
connected quotient component. Unit fixtures for disconnected hard chains and
uniform translation across a wet mixed seam passed. On the same native frame
it reduced maximum Courant from about `5.47` to `3.63` and the maximum Jacobian
norm from `384.33` to `236.22`, without joining the two physical pressure
components.

That result confirms the independent-zero gauges were a real defect. It does
not accept the quotient prototype or prove that the remaining gradient is a
streamfunction seam.

## Remaining S3 question

After the quotient experiment, the worst patch still had a finite,
divergence-free Jacobian and one shared global `psi/dx/dy` representation, so
no C0 velocity jump was demonstrated. The stationary pool and falling droplet
impose different boundary motion across a narrowing represented air gap; a
global divergence-free extension can therefore contain genuine return shear.
Changing extension weights or limiting Hermite derivatives from this one
sample would be a model change without a derived objective.

Quarter-cell travel alone is not a sufficient tracing error condition. The
field `u = lambda*x`, `v = -lambda*y` is a counterexample: near its stagnation
point, point travel can be arbitrarily small while material-line amplification
is `exp(lambda*dt)`. A future S3 review should consider a local RK-stage
Jacobian condition or an embedded/step-doubling error estimate in addition to
local travel. This is a proposal for review. It was not implemented or
validated, and it should not restore a global gradient census or a global
maximum-work multiplier.

## Resume conditions

Before resuming S1-S5:

1. reproduce S0 from the recorded timer-only source hashes;
2. review represented-support completeness and corrected-chain reach;
3. independently validate the quotient operator on the two native scenes;
4. define a local tracing acceptance rule that covers high strain near
   stagnation without imposing maximum work on every point;
5. rerun both native go/no-go lanes and the unchanged repository regression
   gates before building or serving production artifacts.
