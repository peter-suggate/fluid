# Settled Pool Party surface update

The small particle footprint does not bound the work in the timing panel's
level-set/particle row. It includes particle transport, the coarse level set
across the basin, reconstruction, air retirement and redistance.

## Causes and changes

- Reconstruction queried incident solid owners before checking particle
  support. Most coarse vertices have no particle support. Check support first,
  preserving the same transported value; reject COLD vertices before geometry
  work whose result the store would discard anyway.
- Reconstruction cleared both nearest-distance banks across allocated vertex
  capacity. Clear mapped pages instead, including every vertex of their 33³
  halo. Newly mapped and recycled pages are cleared on every reconstruction;
  free pages consume no memory bandwidth. In the sampled layout this reduces
  the two-bank clear from roughly 139 MB to 14 MB per reconstruction.
- Coarse advection mixed basin-wide enumeration with expensive transport
  proofs and characteristic tracing. Enumerate and hold COLD vertices in a
  small kernel, and put other vertices in the existing compact queue. This
  preserves canonical ownership and the original numerical predicates.
- Timing wrapped logical compute passes before the scheduler. Its explicit
  timestamps disabled pass sharing, and each timestamped pass forced a native
  Metal encoder break. Instrument physical passes after scheduling and close
  them at semantic stage boundaries. Timing still measures real GPU work;
  no estimated overhead is subtracted.

Split-kernel diagnostics reduced coarse reconstruction from approximately
1.46 ms to 0.13 ms, and coarse enumeration from 1.44 ms to 0.07 ms. Enumeration's
transport work now runs in the compact queue, so its reduction alone is not the
total advection speedup.

## Reproduction

Measurements use an Apple M1 Max, the production balanced defaults, the authored
Pool Party scene and moving rigid body, and its 1/60 s step. Run 912 frames and
sample frames 895–912, after the ball has landed. Renderer work is excluded.

The retained implementation's production stage measurements are:

| Level-set/particle stage | Before | After |
| --- | ---: | ---: |
| Median | 9.699 ms | 6.128 ms |
| Mean | 9.885 ms | 6.095 ms |
| Range across 18 frames | 8.913–11.403 ms | 5.177–7.012 ms |

This is a 37% median reduction, including both less simulation work and less
instrumentation overhead. It does **not** establish a 2–3 ms update. The remaining
largest individual kernel is the general coarse-advection queue (about 1.64 ms
in split diagnostics), alongside crossing search, redistance and scheduling
overhead. The particle-tile count alone cannot predict this whole-row cost.

```sh
/usr/bin/lockf /private/tmp/claude-501/fluid-gpu.lock \
  env WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" \
  TIMESTAMPS=1 STEPS=912 FROM=895 CURRENT_VOLUME=1 \
  OUT=artifacts/pool-surface-update/native.json \
  node --import tsx tools/investigate-pool-work.ts
```

Sum both `Vertex phi transport and redistance` entries within each trace. The
two entries span separate command submissions. `TIMESTAMPS=1` uses the same
hardware stage recorder as the UI. `PROFILE=1 SPLIT=1 NB_SPLIT_ALL=1` instead
isolates kernels and adds diagnostic overhead; do not compare its sum with
the UI row. Hardware timestamps are quantized to 65.536 µs on this setup.

Source snapshots and raw measurements are retained locally under
`artifacts/pool-surface-update/`. The baseline is the working tree at the start
of the investigation, including existing uncommitted work. Independent runs
can differ in particle counts because atomic ordering changes the splash;
these are end-to-end workload comparisons, not bitwise trajectory comparisons.

An additional alternating two-solver trial showed roughly 3 ms lower untimed
frame latency before synchronization. Its attempt to transplant GPU state
between independently evolved solvers failed the pressure convergence guard
at frame 899. That experiment is excluded from accepted results: it did not
establish a valid comparison with identical complete solver state.

## Verification

`npm run check:types` passes. `npm run test:unit` passes with 979 passed and
87 skipped. The final targeted native-GPU run passes all six reconstruction,
clear and schedule checks, plus the 120-frame hydrostatic-pool regression.

GPU checks cover reconstruction ownership across mixed seams and sparse page
corners, equivalent reconstruction values with fewer solid queries, complete
scratch clearing across page reuse and launch widths, and physical-pass timing
while alternating timed and untimed frames. The hydrostatic pool remains at
less than 0.0001 m/s, and stationary curved liquid retains spatial detail.

The existing falling-drop assertion in `uniform-pool-rest.test.ts` fails on
the untouched baseline too: the drop has no particles at frame 10. It is not
used as a passing regression claim for this change.


## Follow-up: proof ordering

Coarse transport still covers the basin even with a small particle footprint.
A settled-frame diagnostic counted 24,395 coarse tiles and 181 fine tiles.
The whole-transport plane-fit shortcut admitted zero evaluations; the Euler
shortcut admitted 67 against 18,340 full fallbacks. Regional phi residuals
near geometry prevent these shortcuts from treating small motion as cheap.
Evaluation counts include repeated pipeline attempts, not unique vertices.

The retained follow-up checks the numerical transport bound before querying
eight solid footprints. A failed bound cannot admit the shortcut regardless
of those geometry results. This preserves the numerical calculation, error
budget, solid tests for accepted candidates and full fallback. Only diagnostic
rejection-reason attribution can change.

An alternating-frame production comparison (895–954, 30 frames per arm) measured:

| Transport and redistance | Previous order | Early bound |
| --- | ---: | ---: |
| Mean | 5.046 ms | 4.957 ms |
| Median | 5.014 ms | 4.882 ms |

This is a small observed improvement, not achievement of a 2–3 ms target.
Both arms ran in the same continuously advancing simulation; unlike the
older table above, the benchmark did not idle for 360 ms between measurements.
These tables therefore must not be compared as an overall speedup.

```sh
/usr/bin/lockf /private/tmp/claude-501/fluid-gpu.lock \
  env WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" \
  TIMESTAMPS=1 CONTINUOUS_TIMING=1 SURFACE_ALTERNATE=1 \
  STEPS=954 FROM=895 OUT=artifacts/pool-advection/native-reorder.json \
  node --import tsx tools/investigate-pool-work.ts
```

The native GPU regression compares every surface output bit against the old
ordering at zero, 0.0004, 0.4 and 2 cells/step, with constant/varying velocity
and embedded solids. All comparisons passed. Type checking and unit tests also
passed (981 passed, 91 skipped). Cache, kernel-splitting, cubic-loop and wider
admission experiments were not retained. Pending launch-size and runtime-loop
experiments were stopped when the user requested wrap-up.
