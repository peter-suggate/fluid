# Depth-3 renderer scalability

The requested refinement is render detail. Fluid can be inserted into the scene
at runtime and retains its coarser simulation lattice. Renderer-only resources
must not duplicate the solver's evolving fields; solver-owned resources must
retain their dynamic lanes.

## Findings

A CPU-only run of the production `hero-garden-hose-x10` depth-3 planner requested
5,719,349,080 bytes of buffers before textures and mesh storage. Of these,
4,186,589,184 bytes were unused initial dynamic source fields:

- Geometry: 1,860,706,304 bytes.
- Velocity: 1,860,706,304 bytes.
- Material owners: 465,176,576 bytes.

The renderer-only publication path already skips copying these fields. They
were allocated as if it were a dynamic solver publication anyway. In addition,
the topology-only publication dispatched by voxel capacity despite copying
only nodes and leaves. Mesh extraction then processed all scene bricks in a
single dispatch. These are independently avoidable memory/work amplifiers;
the original browser hang was not deliberately reproduced.

## Design

1. Allocate minimal placeholders for absent dynamic source lanes, using the
   resolved payload layout. Full solver publications retain full buffers.
2. Dispatch topology copying by node/leaf count and dynamic copying by voxel
   count. Validate structural and payload arena sizes before allocating either.
3. Extract at most 128 mesh bricks per frame. A GPU cursor tracks progress;
   topology/geometry revision changes restart the unpublished build.
4. Publish only a complete mesh. During construction use the existing ray
   fallback. An incomplete count is a lower bound, never an exact requirement.
5. On capacity overflow, roll back only the current batch to its complete-brick
   checkpoint and pause. Double storage within device limits, copy the existing
   arena in queue order, then resume from that checkpoint. GPU binding size
   acknowledges growth, so delayed diagnostics never overwrite current revision
   state. Retain old storage until submitted commands finish; never display a
   partial mesh.

## Validation tools

`tools/probe-svo-refinement-allocations.ts` runs the production planner with a
non-allocating device facade. It never imports Dawn or submits GPU commands.
The census covers buffers; it does not estimate texture storage.

`tests/svo-surface-mesh-scheduler-dawn.test.ts` runs the production mesh
build kernels over a two-brick synthetic octree with tiny buffers. It verifies
the first build, cache reuse, an incremental re-extraction from the
maintenance dirty list, overflow rollback and growth, and a replacement build's
flip to the second arena. It is a scheduler and range check, not an image
comparison.

`tools/probe-svo-depth3-dawn.ts` uses the full scene and renderer at 64×64. Its
safety limits reject buffers above 1 GiB or cumulative buffer requests above
3 GiB, fence every batch, and stop after a submission takes 500 ms wall time.
Run with `FLUID_SVO_VOXELIZATION_BRICK_BUDGET=64` and the repository WebGPU lease.
A successful result requires the requested depth (no silent downgrade), a
complete non-overflowing mesh, nonzero drawn quads, and no validation errors.
The wall-time stop is a safety guard, not a performance acceptance ceiling.

## Probe results so far

The first full-scene Dawn run on Apple M1 Max allocated and compiled depth 3
without downgrade. CPU preparation took 215.7 seconds. The initial fenced
maintenance batch took 287.8 ms. Extraction reached at least 205,824 bricks
and 3,443,689 required quads before a 508.5 ms frame tripped the 500 ms safety
stop; no GPU hang was observed. That run used the preliminary 1,024-brick mesh
batch. The implementation now uses 128 bricks and the probe records GPU pass
timestamps for slow frames. The smaller-batch full-scene rerun stopped at 812.1 ms wall time after
processing at least 51,328 bricks. Its timestamps showed a 22.3 ms GPU span
and 7.1 ms extraction pass, so that stop does not establish a long GPU kernel.
A subsequent browser inventory found Fluid Lab open; the run cannot be treated
as an isolated performance result. The probe now separates command finishing,
submission and queue-wait times, observes host GC, and supports `--expose-gc`
to collect temporary planner objects before measurement. The 500 ms guard is
unchanged. The clean rerun result follows below.

The 128-brick synthetic scheduler passed Dawn on M1 Max: cursors 128, 256,
320 with publication only at completion, cache reuse and restart assertions
passing, and no validation errors.

A separate winding audit found that negative-axis strips had inward winding.
Their corner coordinates are now transposed, preserving the diagonal and
rectangle while making both triangles outward-facing. Hardware culling stays
disabled; compute culling uses explicit outward face directions. This does not
establish winding as the cause of the original browser failure.

## Isolated depth-3 result

With Fluid Lab browser tabs closed, the 128-brick full-scene Dawn probe passed
on Apple M1 Max / Metal with no validation errors or refinement downgrade:

- CPU preparation: 201.2 seconds.
- Longest fenced submission: 275.6 ms, below the unchanged 500 ms safety stop.
- 227,137 bricks; 3,812,972 complete quads; 1,720,651 camera-visible quads.
- 3,575 encoded frames, including a complete count and replacement build.
- 1,796,343,140 cumulative requested buffer bytes, including probe instrumentation
  and the old and enlarged mesh arenas. This is not peak resident memory and
  excludes textures.

The probe used 64×64 output with GI, shadows and AO disabled to isolate geometry
publication. It confirms bounded construction and successful raster publication,
not full-resolution frame performance. Temporary CPU planner objects were
collected before measurements with Node's `--expose-gc` option.

Refinement 3 is now the requested default. Explicit saved/URL tuning still wins,
and existing device-limit fallback reports the actual built depth. CPU startup
is still slow and remains a separate optimization target. Fluid simulation
resolution is unchanged.

## Regression gate

`npm run test:dawn:sparse-cm12` was run after changing the default. Nine lanes
passed, including mixed-ratio topology, topology transfers/storage, hydrostatic
adaptivity, mini32 correctness, min8 reconstruction and mini32 performance
(31.3 ms against the unchanged 40 ms ceiling). Mini64 performance failed at
65.2 ms against its unchanged 50 ms ceiling. A separate `adaptive-pool-interior`
browser audit acquired the GPU lease between lanes and blocked the remaining
six checks. This is not a passing full gate and does not validate live insertion
for the final worktree. No assertion or performance ceiling was weakened.

The 17 targeted CPU renderer/publication checks and focused ESLint passed.
The workspace-wide TypeScript check still reports errors in unrelated adaptive
simulation/test files; none were reported in the depth-3 files checked here.

## Mesh preparation and fallback investigation

A frame-panel capture showing 25,856 processed bricks and 86.5 ms of planes/ray
fallback was an incomplete build: only 11.4% of this scene's 227,137 bricks.
At 128 bricks per presentation, one publication needs 1,775 frames. At the
captured 91.5 ms primary time alone that is at least 162 seconds. The prior
capacity policy discarded the first complete build and repeated it in a larger
arena, accounting for the isolated probe's 3,575 presentations. The new prefix
retention removes that compulsory second full extraction; the overflowing batch
alone is repeated. This change does not make extraction independent of frame
rate. Measurements of the new lifecycle belong in subsequent probe results.

`Mesh builds: 3` alone does not establish an erroneous restart. GPU extraction
restarts when topology or authored geometry publication revisions change, or
when its source storage changes. Camera movement does not invalidate the mesh;
per-frame culling still uses the current camera. Diagnostics now expose the
completed/total brick count, capacity pause and latest restart reason.

Continuous authored geometry changes can still prevent a whole-world build
from completing. The next architectural step is independently published brick
or chunk meshes with dirty-neighbour invalidation and exact fallback restricted
to missing or dirty regions. Keeping a stale whole-world mesh would violate
current-frame scene correctness. The current improvement makes no stationary
camera or stationary scene assumption and does not claim to solve that broader
incremental-publication requirement.

Mesh extraction now assigns one face/layer mask to each invocation, instead of
serially processing all 48 masks of an 8³ brick in one invocation. A bounded
128-brick batch consequently supplies 96 workgroups instead of two. The greedy
mask and mixed-resolution boundary extraction body is unchanged. A CPU audit
verified its byte-for-byte preservation and the bijection from dispatched jobs
to `(brick, face, layer)`, including rounded final workgroups. Only atomic quad
append order may change. Production WGSL passed Naga at full and half lighting
resolution. GPU timings and scheduler validation remain to be recorded for
this revision; extra workgroups alone do not prove a speedup.

The subsequent isolated full-scene run with prefix-preserving growth and
face/layer-parallel extraction passed: 1,790 encoded frames, one build generation,
3,812,972 quads, no overflow or validation errors, maximum fenced submission
313.9 ms. The synthetic Dawn growth/scheduler assertions passed as well.
