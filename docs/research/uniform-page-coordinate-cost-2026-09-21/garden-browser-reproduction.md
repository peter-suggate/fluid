# Garden browser / Dawn reproduction

Scene `hero-garden-hose`, **Uniform Geometric** (`uniform-volume`), balanced,
144 × 96 × 96, dt 1/30 s, production options, five static cylindrical stepping
stones, dust floor 1e-3. Browser trace enabled; Dawn uses hardware stage traces
and queue-fenced whole advances. Browser scene was closed before Dawn ran.

The earlier benchmark called `advanceTo(time)` with its default empty body list.
The UI passes all five scene bodies. Initialization alone does not retain them:
each advance synchronizes the supplied list. This made the old approximately
47 ms result unrepresentative of the UI, although its before/after transport
comparison used the same omission in both arms.

The benchmark now initializes and supplies scene bodies, records their count
and IDs, and rejects dynamic scenes because this standalone loop does not advance
body dynamics. `OMIT_BODIES_FOR_QA=1` explicitly reproduces the old diagnostic.
The garden transport oracle now supplies the scene body list too (five in the
original reproduction, zero after the voxel-stone correction).

| Measurement | Browser | Dawn |
|---|---:|---:|
| Early advance | 106.43 ms (first step) | 97.06 ms (median, frames 5–44) |
| Later advance | 141.23 ms (12-sample window at 10.233 s) | 129.93 ms (last 12 advances through 10.333 s) |
| Later phi transport/redistance | 47.6 ms | 33.64 ms |
| Later conservative coupling | 17.2 ms | 15.75 ms |
| Later pressure topology | 17.3 ms | 16.41 ms |
| Later multigrid cycles | 6.95 ms | 6.66 ms |
| Later projection | 16.1 ms | 15.44 ms |

The 310-advance Dawn run has an overall median of 123.86 ms. These are comparable
workloads, not identical timestamps or identical measurement boundaries. Browser
trace seams include scheduling gaps and rendering competes for the GPU; Dawn
excludes rendering. The residual difference, especially phi, remains unresolved.
Browser FPS is not the reciprocal of its simulation-only trace cost.

## Identified repeated work

`cellSolidFraction` loops over all bodies and evaluates eight shape samples per
body. `cellOpenFraction` invokes it. Phi's `uvEmbeddedContact` checks the eight
adjacent cells and neighboring solid cells; `uvEmbeddedAir` repeats openness
queries along characteristics. `pressureFaceData` uses eight samples per face,
each querying body occupancy, plus nearest-body velocity extrapolation. These
queries recur in pressure construction, projection, extension and phi, even
though all five garden bodies are static. The approximately 2 ms rigid-coupling
stage does not include this distributed geometry cost.

## Correction: stones are static voxel solids

The scene now bakes the existing inscribed stone collider shapes into static
SolidWorld voxel patches at document creation. It has zero rigid bodies. The
production lattice retains five separate obstacles, stored in 21 X-run patches.
This is binary cell-centre voxelization, so its boundary differs from the previous
eight-sample analytic body fractions. The detailed scenery remains in place.
No runtime rigid geometry cache was added.

The corrected 310-advance Dawn run measures **53.65 ms median**, versus
**123.86 ms** with analytic stones (56.7% lower). Stage medians:

| Stage | Analytic stones | Voxel stones |
|---|---:|---:|
| Phi transport/redistance | 32.11 ms | 6.03 ms |
| Pressure topology | 16.12 ms | 4.78 ms |
| Projection | 14.94 ms | 3.01 ms (includes surface publication) |
| Multigrid cycles | 6.55 ms | 6.55 ms |

Browser verification after rebuilding the scene: first advance **50.07 ms**;
live trace at 1.300 s **51.12 ms**, with phi 8.09 ms, topology 5.24 ms, cycles
6.75 ms and projection 3.24 ms. The panel reports no rigid bodies.

The static-solid component test passes. The 12-frame garden transport oracle
passes with voxel stones; tiled and dense transport agree. These verify geometry
presence and transport equivalence, not equivalence to the old fractional body
boundary. Demand-only page allocation remains unfinished.

Reproduce (with no browser GPU scene or other Dawn process running):

```bash
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
UNIFORM_BENCH_SCENE=hero-garden-hose FRAMES=310 ASYNC_DEMAND=1 \
UNIFORM_BENCH_OUTPUT=/tmp/garden-with-bodies.json \
node --import tsx tools/benchmark-uniform-long-dam-paging-dawn.ts
```

Captures: [garden-browser-dawn-measurements.json](garden-browser-dawn-measurements.json).

The voxel-stone capture is retained in [garden-voxel-stones-measurements.json](garden-voxel-stones-measurements.json). Type checking reports 15 pre-existing errors outside these changes; `git diff --check` passes.

The post-change canonical Sparse CM12 gate completed with 4/17 lanes passing. The mini64-min8-surface lane now fails where the preceding run passed; all other pass/fail statuses match. This additional failure has not been diagnosed or attributed to the garden change. No thresholds were changed. See [garden-voxel-stones-sparse-regression.json](garden-voxel-stones-sparse-regression.json). This separate method gate remains failing.
