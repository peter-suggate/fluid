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
```

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

The `tall-cell-activity` interrogation is emitted before GPU construction and
again in each tall-cell `result`. It reports how many columns actually have a
tall base, the height distribution, whether the layout is entirely ordinary,
and whether the allocated regular band leaves enough vertical space for later
remeshing to create tall cells. In particular, `classification: "none"` plus
`canRemeshToTall: false` means the selected Tall method is permanently running
at its ordinary-grid limit for that scene.

The runner fails on method-neutral invariants: WebGPU validation errors,
non-finite state, materially unbounded volume fractions, mismatched comparison
grids, or failure of deep-water tall-cell compression. Equilibrium scenarios
also require one connected liquid component and less than `0.1%` exact volume
drift. The hose A/B case gates admitted-volume and jet-speed parity while still
reporting shape differences rather than declaring either method the oracle.
