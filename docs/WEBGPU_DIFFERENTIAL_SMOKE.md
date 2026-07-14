# WebGPU differential smoke matrix

`tools/run-webgpu-smoke.ts` runs the repository's Dawn/WebGPU solvers as peers.
Neither `uniform` nor `tall-cell` is treated as the expected answer. The runner
records their discrepancy and, on tractable grids, compares both with the
binary64 CPU MAC solver after the same initial time steps.

## Scenarios

| ID | Primary behaviour |
| --- | --- |
| `settled-tank` | Hydrostatic preservation and spurious currents |
| `dam-break-boxes` | Moving interface and immersed boxes |
| `hose-tank` | Ramped boundary inflow into a shallow pool |
| `sphere-jet` | Directed inlet flow around fixed rigid geometry |
| `deep-water` | Extreme vertical aspect ratio and tall-cell compression |

The matched CPU comparison uses the exact GPU `nx × ny × nz` dimensions. It
does not silently reduce resolution. The smoke default uses one marker sample
per cell axis to keep the exact-grid run practical; set
`FLUID_CPU_MARKERS_PER_AXIS=2` for the normal CPU reference quadrature.

The CPU solver is currently fluid-only in this runner. In the two rigid-body
scenarios it is therefore a useful free-fluid reference, not a complete oracle
for immersed-boundary coupling.

## Running

Point `WEBGPU_NODE_MODULE` at Dawn's installed WebGPU package entry point, then:

```sh
npm run test:webgpu
```

Useful focused runs:

```sh
FLUID_SCENE=settled-tank npm run test:webgpu
FLUID_SCENE=dam-break-boxes,hose-tank npm run test:webgpu
FLUID_METHOD=tall-cell FLUID_SCENE=deep-water npm run test:webgpu
npm run test:webgpu:dam-conservation
```

`test:webgpu:dam-conservation` runs the Figure 4 dam break for 5 seconds with
only 12 regular surface layers. It requires exact reconstructed volume drift
below 0.1%, zero base-zero columns, and zero missing tall cells beneath
threshold-liquid regular samples.

Environment controls:

| Variable | Default | Meaning |
| --- | --- | --- |
| `FLUID_SCENE` | `hose-tank` in the direct tool; `all` in the npm script | Scenario ID, comma-separated IDs, or `all` |
| `FLUID_METHOD` | both | `uniform` or `tall-cell` |
| `FLUID_QUALITY` | `balanced` | GPU resolution preset |
| `FLUID_TARGET_S` | scenario-specific | GPU observation duration |
| `FLUID_ORACLE_STEPS` | scenario-specific | Initial matched CPU/GPU step count |
| `FLUID_CPU_ORACLE` | `1` | Set to `0` to disable the CPU run |
| `FLUID_CPU_MAX_CELLS` | `250000` | Exact-grid CPU safety budget; `0` is unlimited |
| `FLUID_CPU_MARKERS_PER_AXIS` | `1` | CPU marker quadrature, from 1 to 4 |
| `FLUID_FIELD_STATS` | `1` | Set to `0` to omit a second final field readback |
| `FLUID_REPORT_EVERY` | `0` | Emit intermediate GPU diagnostics every N steps |
| `FLUID_REMESH_INTERVAL` | method preset | Override the Tall remesh interval for diagnostic A/B runs |
| `FLUID_REGULAR_LAYERS` | method preset | Override the Tall moving surface-band depth |

The extreme deep-water grid commonly exceeds the default CPU safety budget.
The output then contains an explicit `oracle-skipped` record with the exact
cell count and the override needed to run it; no lower-resolution result is
substituted.

## Reading the output

Each line is JSON. `result` records contain solver diagnostics and reconstructed
cubic volume-field statistics. `discrepancy` records contain:

- mean absolute and RMS volume-fraction error;
- relative represented-volume difference;
- wet-cell intersection-over-union; and
- volume-centroid separation in grid cells.

Each A/B scenario also emits a `performance-comparison` record. Uniform is the
named baseline and Tall is the candidate; a speedup greater than one favors
Tall. The record includes wall-runtime and construction ratios, timestamped
GPU-stage ratios, active-sample reduction, both represented-volume drifts,
their delta, finite-state counts, and available pressure residuals. Timing is
reported but is not a pass/fail gate because short GPU samples are noisy.

The `tall-cell-activity` interrogation is emitted before GPU construction and
again in each restricted Tall result. It reports how many columns actually
have a tall base, the height distribution, the maximum adjacent split delta,
and whether the band leaves enough vertical space for later remeshing. The
runner fails when the observed delta exceeds the configured `D`. The selected
Tall method uses the ordinary-grid limit only when the initial range of
vertical crossings needs so many regular layers that a height-two tall cell
cannot fit. A vertical dam face remains representable across tall columns.

The Figure 4 dam invariant additionally requires the selected Tall method to
construct the restricted backend, retain a tall cell in every packed column,
keep exact reconstructed volume drift below `0.1%`, and report zero dry tall
endpoint pairs underneath threshold-liquid regular cells in the final packed
field readback. The exact reconstruction is authoritative for this gate; the
compact GPU reduction floors each weighted sample to `1/256` and remains a
cheap live diagnostic.

The runner fails on method-neutral invariants: WebGPU validation errors,
non-finite state, closed-scene represented-volume drift above 1%, materially
unbounded volume fractions, mismatched comparison grids, remesh deltas above
`D`, or failure of deep-water tall-cell compression. Equilibrium scenarios
also require one connected liquid component and less than `0.1%` exact volume
drift. Inflow scenarios instead require that neither method loses more than 1%
of its initial represented volume; admitted volume, speed, and shape
differences are reported without declaring either method the oracle.
