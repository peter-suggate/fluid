# Half-size pool: surface artifacts, fixed-topology dissipation and topology transfer

## Current investigation status

The next-test investigation is recorded in
[controlled waves and coarse-grid physics](sparse-cm12-gravity-wave-investigation-2026-09-06.md).
It includes a fresh three-second max1/frozen/adaptive comparison using current
transport from reset, matched fine-state coarsening, real free-surface
oscillation, and causal pressure tests. Adaptive's first wall crest is 20.8%
below max1 but its later wall oscillation is 34.3% larger. The priority is close
physical behavior relative to max1 and Uniform while preserving coarse cells.
The new direct width-1 to width-8 reduction fix prevents numerical conservation
receipt failures; pressure/interface discrepancies remain unresolved.

The sections below “Initial presentation investigation” record the first
presentation-only A/B. Their capture names and validation results are historical,
not measurements of the current combined changes. In particular, stronger waves
from the subsequent transport fixes make absolute surface roughness unsuitable
as a publication correctness test. The current max 1 regression compares the
published waterline against the continuous reconstruction of the same accepted
physical field, within 1/1024 of a finest cell.

### Frozen topology isolates a numerical face filter

The pool advances with the previous face transport until step 9 (0.15 s), then
freezes all accepted leaf membership, coordinates, spans and rungs. Every later
checkpoint asserts the same topology and generation. The legacy, regular-face
and mixed-port arms have all nine checkpoint-9 field files byte-identical and
identical frozen topology. Each keeps 24,396 active scalar cells through step 197;
max 1 keeps 104,448. No extra refinement was used to obtain the improvements.

The old face preparation reconstructs collocated velocity from opposite
staggered faces, then interpolates those cell means back to faces. At zero time
this applies a normal-axis `[1/4, 1/2, 1/4]` filter. For a velocity wave of
wavelength 32 finest cells, a width 8 face loses half its amplitude even when the
characteristic has zero length. That is numerical loss independent of topology
changes or an actual advection distance.

Face preparation now samples the accepted staggered component on its native
face lattice. Equal-width brick faces are included (every B1 face is such a
face). Mixed-width ports use the coarser patch lattice. Unsupported donor
corners individually use the seam-safe collocated interpolant; one unsupported
corner no longer causes the entire characteristic to use the filtered field.
Dry receivers still acquire extended velocity. Production-WGSL fixtures check
zero-time identity, newly wetted fronts and finite translations across widths
1, 2, 4, 8. These changes preserve coarse cells and the existing selector.

| Frozen arm | First wall crest | Later wall peak-to-trough | Final cell kinetic energy / density |
| --- | ---: | ---: | ---: |
| Previous face transport | 21.25 mm | 15.59 mm | 0.002756 m⁵/s² |
| Native regular faces | 26.29 mm | 32.39 mm | 0.014835 m⁵/s² |
| Native regular faces and mixed ports | 30.32 mm | 65.90 mm | 0.024790 m⁵/s² |
| Max 1 reference | 41.24 mm | 101.40 mm | 0.052435 m⁵/s² |

The gauges integrate accepted column mass over the same physical wall patches;
the first crest window is 0.5–1.4 s and the later window 1.8–3.283333 s. The corrected
frozen coarse arm has 4.23 times the previous later oscillation amplitude while
keeping the same cells. It still has lower amplitude than max 1; this is not a
claim of complete resolution independence. Raw captures are `frozen-control`,
`frozen-regular`, `frozen-seam` and `frozen-fine`; matched plots and receipts are
under `artifacts/pool-impact-ab/frozen-matched-report/`.

Read-only copies at existing GPU stage boundaries identify the offending
operation directly. At step 180, face preparation alone reduces staggered face
quadrature kinetic energy by 57.63% with previous transport,21.53% with native
regular faces,10.38% including mixed ports, and 6.72% for max 1. These are separate
trajectories after a common coarse checkpoint, not equal-input operator ratios.
The zero-time fixtures provide the controlled equal-input proof. Later pressure
projection cannot be labeled entirely as dissipation: it must cancel the
hydrostatic velocity introduced by gravity. Potential energy based on diffuse
density also changes with vertical spreading, so it is not used as an exact
free-surface wave-energy balance.

An earlier experiment transported only the increment of collocated velocity
and retained the residual face detail in place. It increased amplitude but left
small-scale detail stationary. It was rejected in favor of the native staggered
characteristic. Its `*-face-increment` captures are ablations, not final results.

### Topology changes had a second, independent filter

The oscillating-topology subtask demonstrated another cell-to-face averaging
filter during candidate publication. A near-zero-time repeated round trip
retained only 0.65% of fixed-grid kinetic energy before the fix and 99.79% after
conservative staggered face remapping, with exactly conserved mass and density
symmetry. This test returns to the same final grid to avoid comparing different
face quadratures. See
[sparse-cm12-topology-oscillation-2026-09-06.md](sparse-cm12-topology-oscillation-2026-09-06.md).
The frozen pool results above predate that transfer fix and deliberately exclude
subsequent topology activity. The combined validation below covers the final
implementation; the full canonical performance gate remains unresolved.

### Freeze control

The UI checkbox now persists through per-frame `applyRuntimeValues` uploads;
previously those uploads erased the freeze flag. Freezing also cancels pending
background topology preparation, and stale asynchronous candidates cannot commit.
Physics continues. CPU renderer/solver integration tests cover repeated normal
draws, changing live settings, freeze and unfreeze; the Dawn pool probe checks
unchanged accepted topology through 197 steps.

## Validation before the controlled-wave investigation

The focused `npm run test:dawn:sparse-cm12:pool-impact` suite passes all 13
production and scene tests, including the 197-step max1 surface reference,
actual topology oscillation (99.788% energy retention), reset waterline,
flat/refined/detached reconstruction, mixed native face identity and D4
aggregation/conservation. The twelve CPU freeze, surface-proof and incremental
activity checks pass. Type checking still has pre-existing repository errors;
there are no errors in the newly added probes/regressions after the local fixes.

The unfiltered canonical gate was run without Fluid in the browser. Mini32
performance passes at 35.8482 ms against 40 ms. Mini64 fails at 107.6101 ms against 50 ms;
the full suite then exhausts its 180-second wall budget. All correctness lanes
that completed before the budget expired passed. This is a remaining performance
regression, not a clean canonical acceptance. No timing limits or correctness
bounds were increased. Earlier pre-change mini64 captures also exceeded 50 ms
(65.798 ms), but the current 107.6101 ms is worse and is not excused by that baseline.
Budget-skipped lanes are checked separately and recorded below; those diagnostic
runs do not substitute for a passing unfiltered gate.

The four correctness lanes interrupted or skipped by that budget all pass when
run separately: tall-cells-hills-far-wall (12.766 s), live-rigid-body-coupling
(9.646 s), live-liquid-injection (8.203 s), and outside-tank-symmetric-collapse
(3.171 s). Thus every canonical correctness lane has passed on the combined
implementation, while mini64 performance and total suite duration remain open.

## Initial presentation investigation (historical captures)

The late max1 square ridges are reproducible in native Dawn and originate in
surface publication. Removing the inconsistent representation switch fixes
them without changing the max1 simulation fields. A separate matched A/B
confirms much weaker wall oscillations with unrestricted coarse-first adaptation.

## Reproduction and causal controls

The fixture is `coarse-first-pool-impact-half`, balanced quality, B8 ladder,
coarse-first selection, 0.05 m finest cells and the scene's 1/60 s timestep.
The max1 arm authors a whole-domain enforcement region with minimum and maximum
cell widths of one finest cell. Every sampled active leaf is checked to be B8.
The run reaches the screenshot's **197 encoded steps / 3.283333 s**. Deferred
topology preparation is awaited and every encoded step is asserted.

`tools/probe-pool-impact-ab-dawn.ts` writes the scene, method settings, shader
hash, accepted density, gamma, velocity, pressure, divergence, activity and
integrated column mass every five steps and at the final step. Publication
payloads and decoded phi are captured at steps 60, 120, 180 and 197. Ordinary
surface pages are decoded; macro interiors remain NaN in the convenience phi
array, with their original payload retained separately.

The capture directories under `artifacts/pool-impact-ab/` are:

- `max1-before-complete`: pre-change shader, complete physical/publication capture.
- `max1-no-sharpen`: surface sharpening disabled; artifacts persist.
- `max1-no-conditioning`: both gamma diffusion and sharpening disabled; artifacts persist.
- `max1-no-height`: column-height substitution disabled as a causal control.
  The added ridges disappear and accepted fields remain unchanged.
- `max1-corrected`: final continuous local reconstruction.
- `adaptive`: final code, same settings and timestep, without an enforcement region.

The first baseline attempt exited with a native crash before impact. The probe
now retains the Dawn GPU instance through teardown, as required by the existing
long-running native probes. Only complete successful captures are compared.

## Cause and correction

The previous resolved publication path selected between two different surfaces:

1. An integrated density-column height, if a column passed nearly-full endpoint
   and monotonicity checks with 1% tolerances.
2. The density-derived scalar `4h (0.5 - rho)` otherwise.

Moving water contains diffuse interfaces and underfilled submerged cells. These
two surface definitions need not agree, even when a column barely crosses one
of the proof thresholds. The validity change can move the zero by more than a
finest cell. Adjacent columns and pages then produce conspicuous ridges.
This is not evidence that the max1 velocity field has those same square waves.
In particular, integrated column mass is not interchangeable with the 0.5
surface when the submerged density is below capacity.

Resolved deep pages now use a single continuous, local scalar. For five
vertical cell samples `r[-2] ... r[2]`, the helper computes:

```text
I = 0.5 (r[-2] + r[2]) + r[-1] + r[0] + r[1]
w = r[-2] (1 - r[2])
phi = h * mix(4 (0.5 - r[0]), 2 - I, w)
```

The four-cell trapezoidal integral has the same scalar units as the density
path. It reproduces affine density exactly and reconstructs the volume of a
sharp horizontal waterline, including the tested B4-to-B8 prolonged profile.
Endpoint support varies continuously: an unbacked thin sheet retains the
density scalar. Solid intersections retain the existing solid-aware fallback.
There is no horizontal averaging, fluid-state edit, scene-name special case,
or velocity/time threshold. The coarse-column and floor-continuation contracts
remain separate and unchanged.

The first candidate used only a cell-local fractional-fill remapping. The
existing B8→B4→B8 test rejected its 0.1-cell shift after prolongation. That
candidate was replaced; its captures (`max1-local`) are not the final result.

At 3.283333 s:

| Published surface diagnostic | Before | Final |
| --- | ---: | ---: |
| RMS unscaled discrete height Laplacian | 30.22 mm | 3.76 mm |
| Largest axial neighbour height jump | 88.25 mm | 16.39 mm |
| Largest displacement from density's 0.5 crossing | 84.84 mm | 25.45 mm |

The last row is a representation comparison, not an error against an exact
fluid solution. The final surface preserves a local volume interpretation;
it is not constrained to equal the density isosurface in diffuse water.
The broad four-lobed wave after wall reflections remains. Exact circularity
is not an appropriate late-time requirement in this square tank.

All **205 saved max1 physical-field files** (density, gamma, velocity, pressure
and divergence at 41 checkpoints) are byte-for-byte identical before and after.

![Dawn surface comparison](../artifacts/pool-impact-ab/surface-before-after.png)

## Matched adaptive A/B

Both arms use **coarse-first**. Here “adaptive” means removal of the max1
enforcement region, not switching to the separate surface-distance selector.
The initial density arrays are identical, with 8.257625 m³ of liquid in each.

The primary gauge integrates accepted column mass over four equal patches:
the outermost 0.4 m at the middle of each wall, each 0.8 m wide. This averages
complete coarse footprints and gives both arms the same physical measurement.
The initial pool height is 0.8 m. The first crest window is 0.5–1.4 s; the later
peak-to-trough window is 1.8–3.283333 s. Samples are 1/12 s apart.

| Measurement | Max1 | Adaptive coarse-first |
| --- | ---: | ---: |
| First wall-gauge crest above initial level | 38.90 mm | 27.95 mm |
| First crest time | 1.0833 s | 1.0833 s |
| Later wall-gauge peak-to-trough range | 88.10 mm | 11.51 mm |
| Final kinetic energy / water density | 0.02197 m⁵/s² | 0.003920 m⁵/s² |
| Final relative mass drift | −0.02140% | −0.000252% |
| Final active scalar cells | 98,304 | 3,552 |

The adaptive first crest is **28.2% smaller**, later oscillation range is
**86.9% smaller**, and final kinetic energy is **17.8% of max1's**. Kinetic
energy alone is not a total-energy dissipation balance because it exchanges
with gravitational potential energy. The wall amplitudes independently show
the strong attenuation. Density-isosurface wall gauges agree on the qualitative
loss but differ quantitatively from column-mass gauges, as expected for coarse,
diffuse representations.

The wave does reach the outer walls. Its persistence is the larger discrepancy.
The two kinetic histories are close through impact (98.7% at 0.5 s); the ratio
falls to 51.1% at 1.5 s and 26.3% at 2 s as the adaptive cell count drops. This
locates the divergence after impact and supports investigating wave resolution,
restriction and coarse transport next. It does not isolate which of those
operations causes the attenuation, and no adaptation threshold was tuned here.

![Accepted-field gauges](../artifacts/pool-impact-ab/wave-gauges.png)

![Radial propagation](../artifacts/pool-impact-ab/wave-propagation.png)

## Re-run

Unload Fluid browser tabs first. Every Dawn run must be sequential and owns
the repository WebGPU lease.

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_WEBGPU_BACKEND=metal \
POOL_MAX_CELL=1 POOL_OUTPUT=artifacts/pool-impact-ab/max1-corrected \
node --max-old-space-size=12288 --import tsx tools/probe-pool-impact-ab-dawn.ts

WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_WEBGPU_BACKEND=metal \
POOL_OUTPUT=artifacts/pool-impact-ab/adaptive \
node --max-old-space-size=12288 --import tsx tools/probe-pool-impact-ab-dawn.ts

uv run --with numpy --with matplotlib python tools/analyze-pool-impact-ab.py
npm run test:dawn:sparse-cm12:pool-impact
npm run test:dawn:sparse-cm12
```

## Validation

- The focused pool gate passes all three tests: production WGSL waterline
  fixtures, the existing B8→B4→B8 transition, and the 197-step max1 scene.
  The late-scene test bounds RMS discrete height Laplacian to 0.1 finest cell;
  the original capture measures 0.6045 cells and the fix 0.0752 cells.
- The WGSL fixtures cover 1,025 sharp waterlines, affine diffuse profiles,
  liquid/air reflection, unbacked thin sheets, and continuity across the former
  99% threshold. Coarse and floor eligibility are checked separately.
- CPU density-column and coarse-first tests: 12/12 pass. Lint passes for the
  new TypeScript files. Type checking still reports errors outside these files.
- The canonical gate passes all 14 correctness lanes and mini32 performance
  (30.54 ms / 40 ms). Mini64 fails its unchanged 50 ms ceiling at 61.47 ms.
  A pre-change comparison and the broader pool fixture audit are recorded below
  when complete.
