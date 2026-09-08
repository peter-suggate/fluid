# Sparse CM12 retained startup audit — 2026-09-08

This audit used source inspection and CPU-only resource recording. It did not
open a GPU device, run Dawn, or measure driver compilation. The reported
19–22 second setup times motivated the audit; the measurements below do not
attribute that elapsed time to a particular GPU kernel.

The current production acceptance record is
[retained-density-production-progress-2026-09-08.md](retained-density-production-progress-2026-09-08.md).

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
- `sparseCM12WGSLForEntryPoints` reparses declarations and dependencies on each
  call. Parsed graphs could be reused within one source. Journal and allocator
  construction also compute some identical slices twice.
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
