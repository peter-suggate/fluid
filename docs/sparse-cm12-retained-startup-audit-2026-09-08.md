# Sparse CM12 retained startup audit — 2026-09-08

The original census below used source inspection and CPU-only resource recording. It did not
open a GPU device, run Dawn, or measure driver compilation. The reported
19–22 second setup times motivated the audit; the measurements below do not
attribute that elapsed time to a particular GPU kernel.

The current production acceptance record is
[retained-density-production-progress-2026-09-08.md](retained-density-production-progress-2026-09-08.md).

## Subsequent measured work

Commit `559a6acb` reuses the WGSL parse index within one source construction,
including journal and allocator families. All 60 four-entry families covering
238 entries from a 789,916-code-unit generated B8/P8 fixture emit exactly the
same bytes as the previous parser. Nine bounded CPU tests pass. Three
alternating CPU benchmark runs measured median total pruning time, including
index creation, of **1,394.31 ms before and 73.25 ms after**. This is a parser
benchmark, not a solver construction or canonical timing receipt. The cache
is construction-local and does not globally retain source generations.

An explicit child-only profiling launcher (`56fff3f9`) replaces the unusable
inherited `NODE_OPTIONS` preload attempt. It registers TypeScript before
loading the observer and preserves target arguments. The tests verify the
observer loads once in the direct target and never in its worker/fork. No
timings were obtained from the earlier hung preload attempt.

The first valid instrumented mini32 profile was stopped at its normal
20-second external deadline (20.040 seconds including termination). Logs:
`/tmp/fluid-mini32-compilation-diagnostic.log` and
`/tmp/fluid-cm12-compilation-43377.jsonl`. `createConfigured` began at 2.873
seconds and returned at 7.454 seconds relative to observer startup. At 15
seconds, 26 module calls and 114 completed direct pipeline jobs had been
observed; background simulation compilation was still running. Full solver
construction had not completed by the deadline, so no frame benchmark result
exists for this run.

The two longest completed first-frame pipeline requests were
`executeSparseCM12FramePlanPresentationPacket` (3.478 seconds) and
`publishSparseCM12SurfaceRepresentabilityReceipts` (3.171 seconds). Several
transport, sharpening and dynamic-face pipeline requests took 0.8–1.15
seconds. Synchronous module creation accounted for 131 ms cumulatively at
the 15-second snapshot. Pipeline request durations include event-loop stalls
and can overlap; they are not measured hardware compiler times and must not
be summed as elapsed wall time. The observer covers direct compilation, not
the manager's separate manifest route.

The unchanged plain canonical run before this parser improvement still
failed: five lanes passed, six timed out, six were unrun when the 180-second
suite budget expired. Mini64 received only 13.212 seconds of remaining suite
time, rather than its nominal 30-second lane deadline. Timeout lane callbacks
suppress initialization progress, so their TAP-only logs cannot place each
timeout at a specific phase. The valid profile supplies that evidence for
mini32 only; the final plain gate remains required.

## CPU construction census

The probe calls `WebGPUSparseCM12Resident.recordPreparedGeneration` with an
`8 × 8 × 8` lattice, one brick, `brickFineResolution = 8`, accepted resolution
1, zero reserved dynamic pages, no rigid bodies, no pressure journal, and no
source-generation transfer. The retained variant uses a horizontal plane at
`y = 0.2 m`, physical cell size and transition width `0.05 m`, and domain
`[-0.2, 0, -0.2]` to `[0.2, 0.4, 0.2]`. The recorder creates symbolic resources
and returns a construction recipe; it does not compile GPU programs.

| Recorded construction | Without retained field | With retained field |
| --- | ---: | ---: |
| Compute pipeline operations | 188 | 197 |
| Shader module operations | 50 | 51 |
| Sum of WGSL `source.length` | 3,896,864 | 4,056,789 |
| CPU recording time | 1,245.7 ms | 1,207.9 ms |

Source lengths are JavaScript string code units, including repeated helpers
across modules, rather than distinct shader bytes. These sequential single
measurements are a census, not evidence that retained construction is faster.
The nine added retained pipelines comprise three initialization kernels and
six simulation kernels. Rigid bodies additionally select eight displacement
kernels in the resident and three retained voxelization kernels in rigid
coupling; ordinary liquid-only scenes do not select them.

The new catalog expansion adds one shader module and two pipelines per
construction. The isolated GPU generation-transfer implementation adds one
module and four pipelines per transfer preparation. Transfer was absent from
this census, and its production integration was still awaiting validation at
the time of the audit.

## Exact shader identity across generations

A second CPU probe records three retained recipes and hashes each pipeline's
grouped module, then hashes the exact result of
`sparseCM12WGSLForEntryPoints(moduleSource, [entryPoint])` separately. Pipeline
counts remain 197. Recording takes 1,185.6–1,293.4 ms per recipe; the additional
197 singleton slices take 519.4–545.0 ms per recipe.

| Comparison | Identical grouped module sources | Identical singleton sources |
| --- | ---: | ---: |
| One brick, accepted resolution 1 → 2 | 172 / 197 | 196 / 197 |
| One brick → two adjacent bricks, accepted resolution 1 | 0 / 197 | 17 / 197 |

The first comparison keeps the domain and catalog capacity fixed. Accepted
resolution 1 → 2 changes native cell width from eight to four finest cells.
Only `validateSparseCM12InternedBoundaryImmutable` changes in its singleton
source, through the immutable certificate literal. The second comparison
extends the domain to two bricks along X and changes capacities and offsets.
The other 180 singleton sources differ only in numeric literals under a
diagnostic comparison. **Removing numeric literals is not a valid cache key:**
these values address different live resources and must remain authoritative.

This is evidence of possible reuse, not a measured cache speedup. Exact
per-entry keys can recover much of the work for a fixed-capacity replacement;
capacity growth requires moving changing offsets/counts into explicit runtime
data to make most programs stable. Cache identity must also include the
device, compatible pipeline/binding layout, entry point, and override values.

## Current reuse boundaries

- `webgpu-sparse-cm12-resident.ts` keys both presentation and simulation
  pipeline families by the entire generated resident shader. A changed
  certificate or offset invalidates both families. It also slices and creates
  the presentation shader module before checking the family cache.
- Simulation compilation uses bounded chunks of four roots, with selected
  heavy roots isolated. Inserting retained roots changes chunk membership.
  Exact per-entry identity can avoid this incidental cache invalidation while
  keeping bounded compiler memory. Increasing compiler fanout is not proposed.
- At the time of the census, `sparseCM12WGSLForEntryPoints` reparsed declarations
  and dependencies on each call. Commit `559a6acb` now reuses this parse within
  each source and avoids the duplicate journal/allocator slices, as measured
  above.
- `gpu-compilation-manager.ts` caches `acquire(manifest)` bundles. Its direct
  `compileComputePipeline` route schedules a fresh job and does not deduplicate
  programs. The manager's `snapshot.cached` counts manifest bundles, excluding
  the resident's separate caches; zero is not a complete cache-miss metric.
- `recordPreparedGeneration` uses a fresh recorder device. Its full pipeline
  recipe is replayed by `realizeCM12ResourceRecipe`, which awaits direct
  pipeline compilation one operation at a time. Hydration currently does not
  consult the real device's resident source cache. Reuse must therefore cover
  recipe hydration as well as direct construction, with bounded cache lifetime.
- Catalog expansion, isolated GPU transfer, and rigid coupling build their
  own layouts/modules/pipelines on each invocation. The small expansion and
  transfer kernels are candidates for stable runtime parameter blocks.

Canonical regression lanes use isolated processes/devices. Same-device cache
improvements do not eliminate their cold compilation requirement. First-frame
presentation and complete simulation readiness also differ:
`waitForSimulationPipelines` starts and awaits the entire selected simulation
family, whereas initial presentation awaits a smaller critical family.

## Duplicate initial integration

At inspection time, solver atlas construction independently compiled the
retained field, unrestricted finest moments, and open-domain moments. The
adapter then compiled a new field and resident construction compiled a
preparation cache without that initial moment snapshot. There is no implicit
memo in `compileRetainedSceneFineMeans`; reuse requires the explicit immutable
preparation cache. Parent coordination assigned the solver → adapter → resident
cache handoff to the production agent. This document records the finding and
work in progress, not a validated fix or measured startup reduction.

## Probe points for the serialized GPU gate

1. Time atlas retained integration and resident preparation separately; record
   cache hit/miss, integrated support count, and reused support count.
2. Record full source length/hash, source generation time, aggregate slicing
   time, and presentation/simulation family or per-entry cache hits. Split
   `createShaderModule` time from pipeline compilation.
3. Timestamp direct pipeline enqueue, manager execution start, and underlying
   `createComputePipelineAsync` resolution per entry point and source hash.
   Report queue latency separately from driver latency and failures.
4. Separate CPU recipe recording from real-device hydration. Count and time
   module creation, pipeline operations, and uploaded bytes during hydration;
   retain its cooperative yielding and temporary-resource release behavior.
5. Measure first presentation readiness and `waitForSimulationPipelines`
   independently. Run a cold device and an actual second generation on the
   same device as distinct probes; do not infer one result from the other.

No lane, correctness threshold, compiler concurrency, or timing ceiling was
changed by this audit.

## Unused pipeline removal and unchanged gate

The resident no longer warms seven entry points with no dispatch consumers:
`advanceRetainedDensityDynamicSupportAccepted`, `buildShadowCellWorklist`,
`buildShadowRowWorklist`, `transferCandidateCells`,
`prepareCandidateFaceReceipts`, `transferCandidateFaces`, and
`publishCandidateTopologyDelta`. Their shader definitions remain available;
the active direct, topology-delta and worklist variants are unchanged.
Tracer pipelines remain because the same resident can enable that view after
construction. Both pressure implementations and packed transport also remain.

The CPU resource-recorder test exercises actual construction, ordinary frames,
initial presentation, paused region editing and tracer off/on/off/on changes.
Every dispatched handle must have compiled and its pruned shader must contain
the entry point. It and the resource-recipe suite pass seven tests
(`/tmp/fluid-unused-pipeline-cpu-1.log`). The actual two-step native VEX capture
also passes with these removals in place.

The unchanged canonical gate still fails:
`/tmp/fluid-current-field-canonical-1.log`, 180.043 s, five passed, six timed out,
six unrun. The mini64 performance lane had only 12.300 s of remaining suite
budget. No numerical assertion failed in the completed lanes, but no result
exists for the timed-out or unrun work. Removing unused compile requests is
sound independently of a timing benefit; this run does not establish a cold
startup speedup or resolve the regression gate.
