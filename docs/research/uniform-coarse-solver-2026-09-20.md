# Single-path Uniform Geometric coarse pressure solve

The wet hero garden's 72×48×48 hierarchy ends at 9×3×3 physical cells.
Its halo makes 275 rows, exceeding the old one-row-per-lane kernel's 256-row
capacity. The same odd terminal dimensions remain when resolution doubles.

The implementation now uses one storage-backed, strided kernel at every size.
A single 256-thread workgroup visits all rows of each red/black phase, then
synchronizes before the next phase. Compensated pressure arithmetic, projected
lower bounds, residual tolerance (1e-4 s^-1), sweep cap (4096), and the finest
pressure acceptance/recovery gate are retained. No larger-grid alternative
entry point or runtime storage selector remains.

Coarse scratch occupies 48 bytes per haloed row after a 112-byte header in the
existing convergence buffer. Sharing that binding avoids increasing the device
storage-buffer slot requirement. Allocation is checked against the device's
buffer limits before hierarchy textures are created. Diagnostics now reserve
30 bits for the worst-row index; coarse texture captures use a width-dependent
row stride instead of assuming every topology row fits in 256 bytes.

The planner retains its existing hierarchy preferences, including the preferred
256-cell coarse target, but that target no longer rejects an otherwise valid
hierarchy. Existing supported hierarchies and window preferences remain intact.
The garden geometry and voxel resolution are unchanged. Dense finest-field
memory limits remain; this change does not implement sparse simulation storage.

## Measurement

Apple M1 Max, Dawn Metal. Compare only the coarse shader with the version in
commit `1fb1b93c2c7efa9383fc84a5061e1c6164a4b40b`; all other code and solver options
come from the same working tree. The historical kernel is injected only by the
benchmark, not retained as an application path. Each arm gets a new device,
excludes compilation, runs 40 frames, and discards five warmup frames. Order is
original/strided/strided/original. GPU timestamp quantization is disabled.

The table averages the two per-run medians for each arm:

| Scene | Original frame | Strided frame | Original coarse passes | Strided coarse passes |
| --- | ---: | ---: | ---: | ---: |
| mini32 | 24.59 ms | 24.08 ms | 0.772 ms | 1.097 ms |
| mini64 | 59.52 ms | 62.29 ms | 2.105 ms | 2.447 ms |

The coarse portion adds roughly 0.3 ms in both cases. Total mini64 frame time
is about 4.6% higher; mini32 is about 2.1% lower, which should not be interpreted
as a speedup. Run-to-run variability is material: mini64's strided coarse
medians were 2.865 and 2.029 ms. This is a short tradeoff measurement, not a
performance ceiling or a statistically established regression bound.

Raw samples, stats and source hashes are in
[uniform-coarse-solver-2026-09-20.json](uniform-coarse-solver-2026-09-20.json).

```sh
node --import tsx tools/benchmark-uniform-coarse-solver-dawn.ts \
  --baseline=1fb1b93c2c7efa9383fc84a5061e1c6164a4b40b --frames=40 \
  --out=/tmp/uniform-coarse-comparison.json
```

## Verification

- CPU hierarchy tests retain supported lockstep/semi-coarsened plans and accept
  the garden and larger odd terminal grids.
- Manufactured constrained-pressure tests cover haloed grids of 64, 125, 275,
  1323 and 2025 rows, including multiple thread batches and widths above 64.
  They compare against prescribed pressure, enforce nonzero lower bounds,
  and verify convergence plus explicit one-sweep nonconvergence diagnostics.
- The authored wet garden constructs with default geometric options, advances
  three frames and returns finite pressure/capture data.
- Pressure safety, hydrostatic, boundary and volume tests: 23 checks passed.
- `npm run test:dawn:uniform-pressure`: all 13 checks passed, including the
  new coarse tests, the six-second Figure 9 run and pressure fault injection.
- Repository-wide TypeScript checking still reports existing errors in unrelated
  tests/tools; none are reported in the changed coarse-solver files.
