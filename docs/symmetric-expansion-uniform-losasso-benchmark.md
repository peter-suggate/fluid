# Symmetric-expansion uniform/Losasso benchmark

This is the native Dawn comparison lane for the paper-aligned dense uniform
solver and the adaptive Losasso backend. It is an integration benchmark, not a
unit test. Each arm runs in a fresh isolated process under the existing WebGPU
exclusive lock.

```sh
npm run benchmark:symmetric-expansion:uniform-vs-losasso
```

The default run advances both methods for exactly 250 steps at `dt = 0.004 s`
on the same `32 x 16 x 32` finest lattice. The script refuses mismatched grids,
accepted/encoded step counts, submitted/completed times, validation state, or
checkpoint counts.

Every ten steps it reads the conservative represented-volume receipt and
reconstructs the presentation field plus collocated velocity on the common
lattice. Keeping those volume quantities separate is important for adaptive
Losasso: a renderer-phi reconstruction is a spatial estimator, not the
integral mass authority. The gate checks:

- finite volume and velocity fields;
- initial conservative mass within 1% of the authored body;
- at most 1% temporal conservative mass drift from each method's actual t0 receipt;
- bounded reconstructed occupancy, with its separate presentation drift reported;
- horizontal D4 volume error at most `1e-3`;
- horizontal D4 velocity error at most `1e-4 m/s`;
- one dominant finite liquid state throughout the matched run.

The report also records construction time, simulation wall time per step,
resident allocation, conservative and reconstructed volume drift, radial front
shape, and mechanical energy. Radial circularity is meaningful only while the
front is freely propagating: the gate uses checkpoints strictly before the
first occupied side-wall cell. The contact checkpoint and everything after it
are reported as tank-clipping telemetry instead of being judged against a
circle. D4 field-symmetry limits continue to apply at every checkpoint.

For the uniform arm, `uniformPaperInvariants` is a checkpoint receipt from the
same low-level audit textures used to localize a failed field gate. Its
`stageMassLedger` keeps the raw-density sums at `previousRawRho`,
`densityAdvection`, `densityDiffusion`, `densitySharpening`, and `finalRawRho`,
then reports the absolute and relative delta for each transition. Each stage's
scalar summary names its integral `sum_cells`. Those are
within-step conservation receipts; they do not replace the temporal t0-to-tn
mass authority above. Its `gamma` record publishes minimum, maximum, sum, and
D4 diagnostics after advection and after diffusion: the scalar summaries are
`gamma.postAdvection` and `gamma.postDiffusion`, while their symmetry receipts
are `gamma.d4.postAdvection` and `gamma.d4.postDiffusion`. The top-level
`betaPostAdvection.maximumAbsoluteDeviationFromOne` records the beta scatter.
The sibling `d4` record
localizes symmetry through the raw-density, advection, diffusion, sharpening,
sharpening-delta, and gamma stages. `rhoPrime` records the Sec. 3.3 extension
authority invariants. D4 and rho-prime records are forensic telemetry; the
canonical D4 gates remain the common reconstructed-field checks described
above. The summary retains rho-prime excess at every checkpoint plus its
initial, final, maximum, net delta, and non-increasing transition count. That
trend is not a decay gate: the calibration legitimately rose from zero to
about 1.168 cells during early extension.

Each checkpoint reconstructs both methods onto the same
cell-centred velocity/volume lattice before computing the common potential,
kinetic, and total-energy retention values. In addition,
`lateToMiddleKineticEnvelopeRatio` and
`normalizedLateMechanicalEnergySlopePerSecond` are the dissipation indicators;
they are reported as an A/B rather than hidden inside the stability verdict.
`initialMassRelativeError` measures initialization fidelity against the authored
body; `relativeVolumeDrift` measures temporal conservation against the method's
own accepted t0 mass. Neither substitutes for the other.

The same checkpoints also publish method-neutral artifact diagnostics for the
acceptance observations visible in long liquid runs:

- the complete `densityBands` and `connectivity` receipts retained as
  step/time checkpoint series, rather than only terminal or peak reductions;
- positive conservative-volume gain and positive rendered/isosurface-volume
  gain are reported separately;
- liquid occupancy and contact-cell counts in the one-cell ceiling and four
  side-wall boundary layers;
- peak, final, and minimum-post-peak boundary mass, including the final residue
  fraction after peak contact;
- mass and component counts outside the largest component, outside every
  floor-connected component, and both disconnected from the main body and
  suspended above the floor.

Connectivity and contact use occupancy `>= 0.01` for both methods. Component
mass is accumulated from the underlying fractional occupancy, while occupancy
below that threshold is retained as a separate diffuse-mass quantity.
`densityBands.paperBands` separately retains raw count and raw rho mass for
`0 < rho < epsilon`, `epsilon <= rho < 0.5`, `0.5 <= rho < 0.95`, and
`rho >= 0.95`, with `epsilon = 0.01`. When the reconstructed interface-face
area is nonzero, each band also reports count and mass per interface face.
These band and connectivity series are trend telemetry, not new gates.

Symmetric expansion contains neither embedded nor terrain solids, so its
Sec. 3.3 authority has `rhoPrime = rho / V = rho`. The receipt verifies that
equality and reports the maximum rho-prime plus
`sum(max(rhoPrime - 1, 0))`. Solid-cell rho and `rho <= V` gates are explicitly
inapplicable here rather than synthesized from unavailable solid state.

For a short wiring smoke or a retained machine-readable artifact:

```sh
node --import tsx tools/benchmark-symmetric-expansion-comparison.ts \
  --steps=20 --checkpoint-steps=5 \
  --out=artifacts/symmetric-expansion-comparison.json
```

Thresholds can be changed explicitly with `--max-volume-drift`,
`--max-volume-symmetry`, `--max-velocity-symmetry`, and
`--min-dominant-component`. The uniform paper-invariant bounds are explicit as
`--max-stage-mass-absolute-delta` (default `0.002` cells),
`--max-stage-mass-relative-delta` (default `1e-6`), `--min-gamma` (default
`0`), and `--max-gamma` (default `2.5`). A looser exploratory
run must therefore say so in its command and artifact rather than silently
changing the canonical package script.

## Opt-in large-dt invariant stress

The comparison runner can append a third, uniform-only stress arm:

```sh
node --import tsx tools/benchmark-symmetric-expansion-comparison.ts \
  --large-dt-factor=4 --large-dt-steps=20 \
  --large-dt-out=artifacts/symmetric-expansion-large-dt.json
```

`--large-dt-factor` is the opt-in switch and accepts only `4` through `8`.
`--large-dt-steps` defaults to 20. The arm therefore advances for
`largeDtSteps * dt * factor`; it does not preserve the canonical duration and
is explicitly not an A/B physics comparison with either 250-step arm.
`--large-dt-out` writes a focused stress artifact in addition to any artifact
requested with `--out`.

The stress arm still requires its process, exact-step/time contract, validation
state, and requested checkpoints to be complete. Its physics acceptance is
deliberately narrow: finite fields; finite stage and gamma summaries; absolute
and relative mass change across every adjacent ledger stage; and the gamma
range after advection and diffusion. It does not apply the canonical temporal
drift, D4, connected-component, circularity, energy, or cross-method
thresholds. The same ledger and gamma checks also apply to the canonical
uniform arm; gamma D4, beta, and rho-prime remain telemetry-only.

The retained Dawn calibration is
`artifacts/symmetric-expansion-paper-invariants-calibration-8.json`. Its worst
stage transition was `0.0002477383` cells (`1.2096599e-7` relative), so the
default ledger bounds are approximately eight times that measured numerical
floor. Gamma measured `[0.9804648, 1.0149598]`, with maximum gamma D4
`4.7683716e-7`. The accepted `[0, 2.5]` gamma envelope is deliberately broader:
the paper's Table 1 reports the method spanning approximately `0.627` to
`2.403`. It is an empirical benchmark envelope, not a theorem or a clamp on
the solver.
