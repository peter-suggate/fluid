# Long-dam extraction and presentation investigation

The reported URL uses `sparse-cm12-ladder-long-dam` but omits `method`.
The default method is `uniform-volume`; the preset's “Sparse Geometric” title
is not the active solver. This matters: uniform and sparse water take different
extraction paths.

## Browser bottleneck and verified fix

The dominant paused-scene bottleneck was **zero-duration physics submission**.
`submitNextPreparedGPUAdvance` can call an asynchronous solver at its current
time to let it settle. `planGPUAdvance` rejected backward time but accepted equal
time. With scene time stepping enabled, Uniform therefore submitted the full
physics chain with dt=0 on every presentation, incremented `encodedSteps`, and
invalidated the water mesh, while the transport still displayed 0.0000 s.
This also runs non-time-scaled field conditioning while nominally paused.

The planner now rejects equal target/current times. It is shared by Uniform
and the octree Eulerian solver. Asynchronous settlement calls remain available;
only the spurious physics step is rejected.

Browser observations in the original scene, full render resolution, Live timing
and surface extraction enabled:

| Metric | Before zero-step fix | After |
|---|---:|---:|
| Paused presentation rate | approximately 25–28 FPS | 59.9 FPS |
| Submission-to-completion wall time | approximately 50–80 ms | 8.7–11.5 ms |
| Host encoding time | approximately 0.5 ms settled | approximately 0.5 ms |
| Water mesh builds while paused | increased every frame (over 3,000 observed) | stable at 2 |
| Surface extraction on retained frames | repeatedly encoded | 0 ms / no encoded work |

An isolated manual 17 ms step after reset advanced the clock to 0.0170 s
and the cumulative mesh counter from 7 to 8 with revision 0 → 1; presentation
returned to 60.0 FPS. The scene was then reset to paused t=0. This is a live observational comparison, not a
controlled statistical benchmark. The GPU shader total alone hid the queued
physics submission preceding the presentation.

Before this fix, disabling extraction left throughput around 26 FPS and queue
completion around 67 ms. At 35% render scale, deferred shading dropped from
about 5 ms to 0.74 ms, but throughput only reached 29–31 FPS. Those ablations
pointed to work outside pixel shading and extraction; the mesh revision counter
identified its source. Original scale 100% was restored.

## Changes

- Sparse/compact extraction now counts contours once per cube across 64-thread
  workgroups, scans integer block totals, and adds the block prefix to the
  public offsets. The contour and emission geometry is unchanged.
- Classification uses a GPU-authored indirect launch based on active pages,
  bounded by allocation capacity. One group still visits an empty publication
  to retire its old mesh. The classifier retains publication validation.
- A completed uniform presentation reuses its completion promise for the
  health receipt. Previously, `assertSimulationHealthy()` fenced the queue
  again after the first fence resolved. Newer frames submitted in between
  could hold the completed frame's throughput slot.
- The live render panel now retains and displays queue-wall time alongside
  CPU time, plus water mesh build count and the latest rebuild reason. The GPU stage total excludes queue backlog and callback delivery;
  queue-wall time overlaps GPU execution and must not be added to it.
- The surface tooltip no longer claims a 250 ms extraction throttle or that
  extraction is universally the largest compute block. Extraction uses the
  presentation cadence; diagnostics use the 250 ms interval.

## Native measurements

Browser scene unloaded; Dawn runs sequentially under the exclusive GPU lease.
Measurements exclude shader compilation and physics advance. Native water
captures use the production long-dam scene at t=0 and a 640×360 render target;
these are **not** full browser/SVO frame measurements.

| Work | Before | After |
|---|---:|---:|
| Scan of 32,769 mixed surface cubes, queue-fenced sustained mean | 13.073 ms | 0.333 ms |

The GPU equivalence test compares every offset, indirect draw/dispatch count,
and publication generation against the old scan. Cases include 0/1/63/64/65
cubes, partial blocks, >256 blocks, capacity overflow, invalid generations,
all cube sign patterns, transitions, walls, heightfields and adaptive triangles.
It also checks active-page dispatch sizes, including empty and 2D launches.

| Production water pipeline | Uniform Geometric | Sparse Geometric |
|---|---:|---:|
| Fresh mesh CPU encoding, median | 0.268 ms | 0.597 ms |
| Fresh mesh queue completion, median | 1.018 ms | 11.236 ms |
| Fresh mesh total, median | 1.293 ms | 11.959 ms |
| Retained mesh total, median | 1.016 ms | 0.878 ms |

Individual timestamped captures show uniform classification at 0.197 ms and
polygonization at 0.066 ms. Sparse classification is 4.588 ms, counting is
0.066 ms, and emission is 5.636 ms. Block prefix/add passes are below the
65.536 μs timer granularity in this capture. Remaining sparse extraction cost
is classification and emission, not the scan.

The screenshot's 20 FPS therefore cannot be explained by uniform extraction's
isolated execution cost. Initial browser ablation also stayed around 17–20 FPS
with extraction disabled. Completion-frontier stage attribution alone does
not isolate a kernel: sampled cost moved between extraction and primary
rasterization. The second queue fence was a concrete scheduling inefficiency;
its individual end-to-end effect was not isolated from browser variance. The
zero-duration physics fix above accounts for the decisive paused-frame gain.

## Validation

- GPU scan equivalence test: passed, twice.
- Startup/cadence/mesh invalidation, completion-fence, zero-step regression and
  host clock tests: 27 passed. The zero-step test exercises the production
  Uniform advance method over 120 paused repaints at each of three times.
- Native production water extraction: both methods passed, nonempty current
  meshes, no uncaptured GPU validation errors.
- `npm run test:dawn:sparse-cm12`: **failed**, 5/17 lanes passed, 398.9 seconds.
  This gate ran after the sparse scan changes and before the final shared
  zero-step planner guard (which Sparse CM12 does not use). The exact results
  are in `regression.json`. Failures include generation-zero
  topology counts, rung commits, timeouts, mini64 simulation timing, terrain
  volume transport and outside-drop compiled connectivity. No assertion or
  timing ceiling was changed. The working tree already contained substantial
  solver/editor changes; this run is not a clean-baseline comparison.
- The old mini64 presentation-capture tool, invoked with long-dam and zero
  steps, failed its surface-brick activity assertion before water rendering.
  That assertion was left intact; the new benchmark checks the actual output
  mesh and validation errors without assuming an evolved activity census.
- Repository-wide TypeScript checking reports errors outside the files changed
  for this task; no reported errors remain in this task's changes.

## Reproduce

Keep the browser scene unloaded while Dawn runs.

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx --test tests/water-surface-parallel-scan-dawn.test.ts
node --import tsx --test tests/uniform-presentation-completion.test.ts tests/water-surface-startup-readiness.test.ts tests/webgpu-water-surface-cadence.test.ts tests/sparse-cm12-adaptive-mesh.test.ts
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/benchmark-water-extraction-dawn.ts
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/benchmark-water-extraction-dawn.ts --method=adaptive-volume
npm run test:dawn:sparse-cm12
```
