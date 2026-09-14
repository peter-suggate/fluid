# Level-set-volume energy and solid-wall investigation

Date: 2026-09-14. Scope: the 2-D Rust/Wasm Advance Lab experiment.
Original reference checkout: `cc525139`.

## Requirements and measurement plan

Keep the existing 1/30-second frame. No smaller timesteps or substepping.
Improve the dam's wall-impact response by correcting the numerical cause,
without global velocity scaling, artificial rebound forces, or an expensive
new series of passes. Preserve conservative liquid volume and stability at
terrain cut cells, adaptive seams, and hydrostatic rest.

Use the exact `minimal-power-dam-break-32` scene from the catalog and the native
production `World`, with 256 pressure iterations and relative tolerance 1e-6.
Measure kinetic and gravitational potential energy separately, plus excess
volume, mass drift, peak speed, and frame cost. Check mini64, Figure 7,
coarse-first half-pool impact, and Tall Cells as well. Energy conservation is
not implied by mass conservation, and a larger splash alone is not evidence
of a better discretization.

## What the supplied CM12 paper does

Primary reference: [Mass-Conserving Eulerian Liquid Simulation](../papers/massConservingLiquids.txt).

- Algorithm 1 (lines 175–188) advances and sharpens density before advecting
  velocity, adding forces, and enforcing incompressibility. The existing lab
  projects velocity before transporting its volume and level set.
- Section 3.4 conserves mass by controlling donor weights, forwarding missing
  donor contributions, and diffusing cumulative sampling weights and density.
  These are mass-transport mechanisms, not a kinetic-energy constraint.
- Section 3.6 (lines 398–429, with the continuation at 370–375) separately
  scatters excess density out of partially solid cells along the solid-distance
  gradient. Its displacement is one grid spacing in the reported examples.
- Section 3.7 uses open-volume-normalized density, a capped artificial
  divergence with lambda 0.5 and eta 1, and the CM11a pressure solver's
  separating solid boundary conditions. The authors explicitly warn that
  uncapped expansion can become unstable when fast liquid encounters solids.
- The reported examples use a fixed 1/30-second timestep (lines 440–445).

The paper therefore motivates checking solid handling and projection timing,
not simply increasing the lab's existing excess-release coefficient. Its
algorithm does not guarantee exact total mechanical-energy conservation.

The printed divergence correction has a `Delta x` denominator. The lab uses
`min(0.5 max(V-C, 0), C) / dt` in integrated flux units: a per-step adaptation,
not a literal transcription of that expression. Comparing their strengths
requires consistent physical length and velocity units; replacing `dt` with a
cell width measured in lattice units is not a justified correction. Likewise,
an existing post-transfer projection must enforce the same expansion target:
reimposing that constraint does not double it, whereas projecting to zero
divergence would cancel it.

## Original mini32 baseline

The original release binary was preserved before edits. Native compact
receipts give liquid-volume-weighted velocity means, velocity covariances,
and liquid centroids. The initial diagnostic uses

```text
K = 0.5 M (mean(u)^2 + mean(v)^2 + var(u) + var(v))
P = M (9.80665 / 0.025) mean(y)
E = K + P
```

These are lattice-unit diagnostics with unit material density and potential
zero at lattice y=0, not joules or energy above the hydrostatic resting state.
The original moment collector omits cells with density at most 1e-5; this
approximation must be held consistent in before/after comparisons.

| Frame | K | P | E | Transport excess volume |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0 | 3,400,789 | 3,400,789 | 0 |
| 5 | 731,646 | 2,605,570 | 3,337,216 | 3.076 |
| 10 | 234,024 | 2,110,672 | 2,344,695 | 20.040 |
| 15 | 27,862 | 2,180,042 | 2,207,903 | 2.355 |
| 30 | 31,555 | 2,170,410 | 2,201,965 | 0.742 |
| 90 | 21,282 | 2,185,199 | 2,206,481 | 0.228 |

The large loss occurs around the first far-wall impact, frames 6–12. Excess
volume subsequently drains while the lost motion does not return. The stage
budget shows substantial losses at the velocity-publication and topology
transfer boundaries. Face preparation and forces update face velocities, but
the observed cell velocities are refreshed only after projection. That budget
therefore combines velocity advection, forces, and projection; it cannot
identify pressure alone as the sink. Transport's reweighting of the unchanged
velocity field can increase the cell-based kinetic-energy diagnostic. Such
stage differences are diagnostic, not a complete conservative momentum budget.

## Controlled ordering trial

Moving LSV transport ahead of velocity preparation and pressure, while retaining
the same timestep and capped pressure source, raised frame-90 energy from
2,206,480 to 2,225,643. This recovers only 0.56% of the initial energy. The peak
excess/capacity ratio worsened from 3.27 to 6.44. That trial was removed: paper
ordering alone did not adequately address the observed loss.

## Face-advection source aliasing

`prepare_faces_impl` evaluates `sample_source` and immediately stores each
result into `fields.face_velocity`. Both the uniform staggered sampler and
the adaptive staggered sampler read that same array. Later rows therefore
consume a mixture of old and already-advected velocities. This is not the
single old-time-level source field required by semi-Lagrangian advection.

A fixed-1/30-second regression reverses the row numbering of the same physical
graph, remaps its incidences and face fields, and compares the physical output.
The original implementation differs by 0.0020463616, above the 2e-6 tolerance.
Constant wall-tangential flow and a tiny divergence-free vortex pass their
independent checks. Approaching and rebounding flows obey zero normal velocity
at every closed wall after the production boundary-enforcement stage.

The minimal correction is a separate destination face bank for LSV. The same
row loop reads the immutable old bank and writes its results into the new bank,
which is published once the loop finishes. There is no additional advection,
pressure, or transport pass. This targets an order-dependent numerical error
without adding an energy source or changing the expansion coefficient.

## Measured result at the unchanged timestep

The original frame order is retained. Native release runs use the same live
catalog scene, dt, pressure budget, and compact diagnostic on both sides.

| Mini32 metric | Original | Immutable face source |
| --- | ---: | ---: |
| Frame 10 kinetic energy | 234,023 | 434,438 |
| Frames 10–20 mean kinetic energy | 57,117 | 118,276 |
| Frame 30 kinetic energy | 31,555 | 70,130 |
| Frame 10 total energy | 2,344,695 | 2,569,855 |
| Frame 30 total energy | 2,201,965 | 2,285,012 |
| Frame 90 total energy | 2,206,480 | 2,223,940 |
| Peak speed, finest cells/s | 282.941 | 263.750 |
| Maximum relative mass drift, 90 frames | 9.92e-9 | 8.47e-9 |
| Peak total energy / initial energy | 1 | 1 |

Mean post-impact kinetic energy more than doubles. This is a reduction in
avoidable numerical dissipation, not an exact energy-conservation scheme:
semi-Lagrangian interpolation, pressure projection, and adaptive averaging
remain dissipative. At three seconds both simulations have lost substantial
mechanical energy. No velocity rescaling or restitution impulse is applied.

The immutable-source run also completes mini64 for 90 frames and Figure 7,
half-pool impact, and Tall Cells for 30 frames without faults. Each run's peak
total-energy diagnostic stays at or below its initial value. Mini32 completes
300 frames (10 seconds) with relative mass drift 2.47e-8; Tall Cells completes
120 frames (4 seconds) with drift 3.11e-9. Excess volume is still possible: this
change fixes velocity advection, not the representation of compressed volume.

### Stability-test correction

The older Tall Cells Wasm check treated a Courant number above 5 or a late/early
peak-speed ratio above 1.25 as a velocity burst. The corrected advection gives
more sustained motion: its first-40-frame Courant maximum is 6.024 and its
late/early speed ratio is 1.395. Across 120 native frames, total energy never
exceeds its initial value and eventually decreases to 0.242 of that value.
The original native control itself has a late/early ratio of 1.260.

These speed heuristics are not appropriate energy-stability requirements for
a method explicitly intended to take large-Courant steps. The replacement
Wasm check uses published conservative cell volumes and cell velocities to
check total mechanical energy, retains finite-state and mass checks, and runs
three times as long. The numerical pressure-release cap and the fixed timestep
are unchanged; the test does not impose additional damping.

### Cost

Three alternating 90-frame native release runs per arm, with stage observers
disabled, give median mean frame costs of 21.731 ms originally and 23.965 ms
with the fix: +10.3%, or 2.234 ms. Field building changes from approximately
0.787 to 0.856 ms; pressure work changes from approximately 2.69 to 3.48 ms.
These are whole-simulation costs for the changed trajectory, not an isolated
memory-copy microbenchmark. The algorithm adds one temporary face array and
its copy, no additional traces or solver passes. The more energetic dynamics
also change pressure iteration and topology work.

### Regression compatibility

Figure 7's Wasm test previously required trailing brick 2246 to reach rung 1
at frame 16. Both original and corrected native controls still request rung 2
then, and schedule rung 1 at frame 17. The test is tied to the actual scheduled
coarsening event, with the final retirement/coarsening check retained.

The broader Rust run has three existing failures, reproduced with test
executables built before the immutable-source change, including the same
assertion values:

- `cellwise_remap_lab::edge_samples_are_intervals_per_finest_unit`;
- `levelset_volume_regression::zero_velocity_is_identity_on_mixed_adaptive_cells`;
- `levelset_volume_regression::small_translations_cross_fine_coarse_seam_in_both_directions`.

Those tests exercise unchanged remap/gather/sharpening paths and were not
weakened or altered. The repository-wide TypeScript check also reports 97
errors in existing, unrelated files; none are in the new energy reporting code.

## Validation and reproduction

- Six new Rust regressions pass: row-order invariance, tiny-vortex energy,
  tangential free slip, approaching/rebounding wall enforcement, and the
  production mini32 impact-energy window.
- The 187 core Rust unit tests pass (one existing ignored test). The selected
  integration run has 36 passing tests and the two pre-existing LSV failures
  listed above; frozen baseline World frames and cutover checks pass.
- Seven reporting tests pass, including the repaired existing receipt fixture.
- All seven scalar/SIMD Wasm flow tests pass, including 120-frame Tall Cells,
  hydrostatic rest, liquid insertion, Figure 7, and half-pool impact.
- The full production build, all three Wasm artifact checks, focused ESLint,
  and diff checks pass. The local server on port 4173 serves matching rebuilt
  scalar/SIMD/threaded Wasm hashes and passes all nine isolation-header checks.

```bash
cargo build --manifest-path rust/Cargo.toml -p fluid-core --release --example verify_world
node --import tsx tools/verify-level-set-volume-energy-native.ts \
  --output=artifacts/level-set-volume/mini32-energy-after.json
cargo test --manifest-path rust/Cargo.toml -p fluid-core \
  --test levelset_volume_wall_velocity --test levelset_volume_mini32_energy
node --import tsx --test tools/wasm/levelset-volume-flow.test.ts
```

The energy CLI also accepts `--input=<archived-native-output.json>` for an
apples-to-apples comparison without rerunning a control binary. Local measured
reports are in `artifacts/level-set-volume/mini32-energy-before.json`,
`mini32-energy-after.json`, and `mini32-energy-performance.json`.
