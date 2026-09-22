# Uniform Geometric 256³: memory reduction

The current-default `cm12-figure-7-256` scene fits the **3,000,000,000-byte GPU resource budget**. Peak live requested resource bytes fell from **8,163,023,308 to 2,969,977,312** (2.970 GB / 2.766 GiB), a **63.6% reduction**. Mean GPU advance time improved from **283.03 to 273.82 ms** (3.25% faster).

These are WebGPU buffer/texture descriptor bytes, audited at the device boundary with destruction tracked. They include profiling/readback resources, but do not measure Metal heap alignment, driver/compiler caches, CPU arrays, or total process RSS. The production allocation counter was also corrected for omitted hierarchy/origin and column resources; the independent audit is the budget authority.

## Unchanged simulation, revised storage

The 256³ grid, physical scene, numerical algorithms, timestep, iteration/sweep settings, FP32 floating fields, and exact six-limb donor accumulation are retained. The new scene keeps the 128³ Figure 7 geometry and halves cell size to 0.025 m. See [the original scene/profile report](report.md).

- One scratch arena serves velocity-extension temporaries, conservative-transport stencils and donor sums, pressure temporaries, and surface-correction bands/partial sums at different stages.
- Finest pressure topology and level-set temporaries become cycle workspace after coefficient baking. Unused duplicate volume/residual textures are removed; finest published pressure remains a native texture.
- A stencil row stores one donor-base index and nine FP32 weights: 40 bytes instead of 80. Sharpening reuses the same row storage for its flags and limits.
- Decoding donor sums overwrites each sum's own low limb with the rounded result once exact limbs are no longer needed. The next accumulation clears all limbs. This removes a full-grid scalar allocation without changing summation or rounding.
- Main velocity, volume, and vertex fields retain native textures and direct shader accesses. Small work maps reserve their actual tile counts. The shared arena binds atomic donor storage as a disjoint range, leaving ordinary field reads/writes non-atomic.
- The default semi-Lagrangian mode omits MacCormack-only fields. Selecting MacCormack in the UI rebuilds the solver. Geometric pressure uses its own CM11a hierarchy, so legacy pressure slots are tiny views and unused seed writes are omitted.
- Full FIM-state diagnostics explicitly request `retainStageDiagnosticsForQA` during construction. That diagnostic mode retains additional storage and is outside this default-scene budget. The harness requests it when collecting those fields. `scratchStorageForQA: "separate"` retains an independent-storage control.

## Dawn/Metal measurements

Apple M1 Max, macOS 26.6.2. The fresh baseline uses repository commit `407a11f372a2fce54e713bbf8f8b41c2a51bf529` in an isolated checkout, plus the new scene and profiling tool. Runs were sequential under the exclusive WebGPU lease. Both used the same resolved current defaults, 60 advances / two simulated seconds, and hardware timestamps. First four advances are excluded. Rendering, stats readback, and trace-cadence waits are outside the queue-fenced wall interval. Stage values are solver seam intervals, not isolated kernel durations.

| Window | Baseline GPU ms | Optimized GPU ms | Baseline wall ms | Optimized wall ms |
|---|---:|---:|---:|---:|
| Free fall, 5–24 | 200.91 | 200.07 | 209.52 | 207.10 |
| Impact / spread, 25–60 | 328.65 | 314.80 | 339.61 | 324.14 |
| All measured, 5–60 | 283.03 | 273.82 | 293.15 | 282.34 |

GPU p90 across the evolving sequence fell from 380.50 to 359.53 ms. This is one fresh baseline and one final optimized trajectory; percentiles describe frame costs along that trajectory, not statistical confidence intervals. Earlier development runs also reached the memory target but had performance regressions; their storage/access choices were revised before this result.

| GPU stage, frames 5–60 | Baseline ms | Optimized ms |
|---|---:|---:|
| CM11a Full-Cycles | 65.44 | 66.80 |
| Dense geometric volume coupling | 46.25 | 36.56 |
| Dense vertex phi transport and redistance | 40.01 | 37.38 |
| Dense conservative volume gather | 35.87 | 35.85 |
| Sec. 3.3 interface authority | 24.17 | 23.86 |
| CM11a topology + RHS pyramid | 17.32 | 20.84 |
| Velocity advection + body forces | 11.79 | 11.46 |
| Sec. 3.3 hierarchy fill + transport shell | 10.14 | 9.30 |
| Dense conservative volume sharpening | 9.38 | 6.35 |
| Sec. 3.3 narrow-band FIM front | 9.04 | 12.12 |
| CM11a parity copy + fine residual | 6.24 | 6.19 |
| Pressure projection + surface publication | 4.83 | 4.55 |
| Diagnostics reduction | 2.55 | 2.55 |

## Numerical evidence and limits

The storage change is **not bitwise identical** across compiled shader variants. The 64³ same-input regression exercises 40 advances through impact, with a separate-storage control and both default/diagnostic shared-storage variants. It compares volume, velocity, pressure, extended velocity, vertex phi, presentation, and retained diagnostics. It supplies the control's physical fields as the next input to each variant, isolating operator differences from accumulation of FP32 roundoff. Bounds are 1e-5 relative L2 or a small absolute floor; near-zero free-fall pressure uses a 1 mPa floor. [Raw same-input comparisons](storage-equivalence-64.json).

An additional independent 64³ trajectory comparison showed late amplification in conservative volume: at step 40, relative L2 was 3.96e-4 and the largest cell-volume difference was 0.0126. Velocity relative L2 was 9.36e-6; maximum phi difference was 7.83e-5 m (0.000783 cells). The initial strict independent-trajectory volume threshold failed. The automated test uses identical inputs to assess storage equivalence; these free-running differences remain a separate reported limitation. [Raw independent trajectories](independent-trajectory-roundoff-64.json).

The uninterrupted 256³ runs ended at 267723.157227 baseline versus 267730.046875 optimized cell-volume units. Drift from frame 1 was -0.127849% versus -0.125279%. The endpoint difference is 0.002570% of initial volume. No WebGPU validation errors occurred in either profile.

## Validation

- Passed: the explicit 3 GB peak-resource budget and all 60 hardware-timestamp samples.
- Passed: same-input shared-storage regression at 64³, including full retained diagnostics.
- Passed: native/paged/logical pressure-layout equivalence, including live liquid insertion.
- Passed: all six surface-volume cases (planes, curved surfaces, solid capacity, empty surface).
- Passed: 10 focused pressure/page/texture unit tests.
- Broader uniform Dawn run: 19 passing tests; one existing mini32 separating-far-wall conservation subtest fails (and marks its parent failed). The unchanged baseline reproduces the same represented-volume drift `0.05509824338166414` and maximum volume `17.105260848999023`. No threshold was raised.
- Three existing `uniform-volume-initial` UI-label expectations also fail on the unchanged baseline (`dense finest lattice` versus `page domain`).
- Repository-wide TypeScript checking still reports existing errors in unrelated sparse tests/tools; no errors were reported in the changed files.

## Reproduce

```sh
node --import tsx tools/profile-uniform-geometric-dawn.ts \
  --scene=cm12-figure-7-256 --frames=60 \
  --allocation-audit --max-gpu-bytes=3000000000 \
  --out=/tmp/uniform-geometric-256.json

WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
node --import tsx --test --test-concurrency=1 \
  tests/uniform-scratch-storage-dawn.test.ts \
  tests/uniform-pressure-layout-dawn.test.ts \
  tests/uniform-surface-volume-dawn.test.ts
```

Do not run these alongside another Dawn or browser GPU workload. Raw 256³ timing/allocation records: [fresh baseline](memory-baseline-256.json), [optimized](memory-optimized-256.json).
