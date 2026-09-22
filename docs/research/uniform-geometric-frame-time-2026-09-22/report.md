# Uniform Geometric 256³: pressure work proportional to liquid tiles

Memory changes were committed as `cad6ba32`. This follow-up targets frame time for the same `cm12-figure-7-256` scene and resolved defaults.

Mean GPU advance time fell from **275.72 to 248.74 ms (9.8% faster)** in a fresh sequential Dawn/Metal A/B. Pressure Full-Cycles fell from **67.28 to 37.29 ms (44.6%)**. Peak requested GPU allocations remain below the decimal 3 GB budget: **2,971,243,028 bytes**, including profiling resources. The new work lists and parameters add approximately 1.27 MB.

## Capacity diagnosis

The finest pressure level has **274,625 potential 4³ tiles** including the halo. Its actual liquid-tile counts are:

| Frame | Liquid tiles | Fraction of tile capacity |
|---|---:|---:|
| 5 | 4,930 | 1.80% |
| 24 | 4,933 | 1.80% |
| 40 | 8,412 | 3.06% |
| 60 | 10,231 | 3.73% |

Previously, every pressure-smoothing visit performed six red-black sweeps over the whole level. The new GPU-only list is built from baked liquid masks once per level per solve and reused by subsequent smoothing visits. Each listed tile launches 32 lanes per colour; the colours, stencil, FP32 arithmetic, sweep counts, convergence policy, and multigrid schedule are preserved. No CPU readback controls execution.

The first sweep of each visit remains dense. Prolongation can leave non-liquid pressure below its minimum constraint, so omitting that first sweep would be incorrect. Once it projects every row, non-liquid rows cannot change during the remaining sweeps: their minimum is fixed, and they do not receive stencil updates. The remaining five sweeps visit only tiles containing liquid. Empty lists launch no work; partial boundary tiles retain bounds checks. Lists rebuild every frame, including after live edits. Small fused visits and the coarsest solve retain their existing execution; their list counters are unpopulated, not evidence of empty coarse levels.

A separate indirect-argument buffer avoids binding the same writable work-list buffer as indirect arguments. The three new kernels use only the bindings they need, keeping pressure compilation within the storage-buffer binding limit. The dense control is selectable with `pressureSmoothingForQA: "dense"` or `FLUID_UNIFORM_AB_OFF=pressuretiles`.

## Measurements

Apple M1 Max, Dawn/Metal, 60 advances at 1/30 s. First four advances excluded. Both runs use the same scene/defaults and exclusive WebGPU lease. Rendering, stats readbacks, tile-count readbacks, and the recorder cadence gap are excluded from wall timing. GPU measurements are hardware timestamp seam intervals. Memory is requested live WebGPU resource sizes, not process RSS or driver heap sizes.

| Window | Dense GPU ms | Tile GPU ms | Dense wall ms | Tile wall ms |
|---|---:|---:|---:|---:|
| Free fall, frames 5–24 | 200.39 | 178.60 | 213.27 | 187.21 |
| Impact/spread, frames 25–60 | 317.57 | 287.70 | 327.30 | 297.90 |
| All measured, frames 5–60 | 275.72 | 248.74 | 286.57 | 258.37 |

GPU p90 fell from 365.23 to 329.84 ms. An earlier optimized run measured 246.82 ms, with the same end volume. These are evolving-trajectory measurements, not confidence intervals from repeated stationary frames.

| Stage, frames 5–60 | Dense ms | Tile ms |
|---|---:|---:|
| Dense vertex phi transport and redistance | 37.52 | 37.74 |
| CM11a Full-Cycles | 67.28 | 37.29 |
| Dense geometric volume coupling | 36.80 | 37.12 |
| Dense conservative volume gather | 35.82 | 36.29 |
| Sec. 3.3 interface authority | 24.65 | 24.76 |
| CM11a topology + RHS pyramid | 20.93 | 22.24 |
| Sec. 3.3 narrow-band FIM front | 12.22 | 12.32 |
| Velocity advection + body forces | 11.45 | 11.59 |
| Sec. 3.3 hierarchy fill + transport shell | 9.30 | 9.39 |
| Dense conservative volume sharpening | 6.40 | 6.50 |
| CM11a parity copy + fine residual | 6.26 | 6.29 |
| Pressure projection + surface publication | 4.55 | 4.61 |
| Diagnostics reduction | 2.56 | 2.61 |

List construction adds about 1.31 ms to pressure setup and saves about 30.00 ms in Full-Cycles. Other stages remain essentially unchanged. The pressure setup, transfers, copies, and final residual still contain full-capacity work. The next largest remaining costs are vertex phi transport/redistance (37.7 ms), conservative coupling (37.1 ms), and gather plus global surface-volume correction (36.3 ms). Phi already has a conservative bounding-box dispatch; that is not equivalent to compact tile traversal. Surface correction still makes dense band and reduction passes, and donor normalization clears the full six-limb accumulation range four times per advance. These are further candidates, not changes in this patch.

## Correctness and checks

- **64³ ball drop, 40 independent advances through impact:** dense and compact scheduling produced bit-exact pressure, velocity, volume, and vertex-phi fields at frames 1, 2, 5, 24, 30, and 40. This uses the same shared-storage implementation in both arms, without resetting trajectories between frames.
- **Partial-page and long-dam fixtures:** native, logical-atlas, and tiled-atlas pressure outputs remain bit-exact through 12 advances, including live liquid insertion at frame 7. The schedule comparison excludes only smoothing and its new list construction; all other pressure-operator launches still match.
- **256³ uninterrupted profiles:** all 60 samples exactly match for liquid volume, maximum speed, initial/accepted pressure residuals, and executed cycle counts. This is telemetry equality, not a full 256³ field comparison. Both end at 267730.046875 cell-volume units; the existing conservation drift is unchanged.
- Zero WebGPU validation errors; 3 GB peak-allocation assertion passes.
- Five pressure page/plan unit tests pass; `git diff --check` passes.
- Repository-wide TypeScript checking still reports existing errors in unrelated sparse tests/tools; no errors in changed files.

## Reproduce

Run sequentially, never alongside browser GPU work or another Dawn process:

```sh
FLUID_UNIFORM_AB_OFF=pressuretiles node --import tsx tools/profile-uniform-geometric-dawn.ts \
  --frames=60 --max-gpu-bytes=3000000000 --out=/tmp/uniform-frame-baseline-256.json
node --import tsx tools/profile-uniform-geometric-dawn.ts \
  --frames=60 --max-gpu-bytes=3000000000 --out=/tmp/uniform-frame-optimized-256.json

WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
node --import tsx --test --test-concurrency=1 \
  tests/uniform-pressure-work-dawn.test.ts \
  tests/uniform-pressure-layout-dawn.test.ts
```

Raw data: [dense control](baseline-256.json), [compact pressure smoothing](pressure-tiles-256.json), [Dawn regression output](pressure-regressions.txt), [bit-exact field regression](bit-exact-regression.txt). Earlier memory work: [memory report](../uniform-geometric-256-2026-09-22/memory-report.md).
