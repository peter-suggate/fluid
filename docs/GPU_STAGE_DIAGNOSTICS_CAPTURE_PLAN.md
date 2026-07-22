# GPU stage diagnostics capture plan

Status: D1/D2 vertical slice implemented, later milestones remain planned,
2026-07-19. This is a cross-cutting companion to
`docs/SPARSE_VOXEL_OCTREE_RENDERER_MIGRATION.md`.

## Current implementation

The shipped vertical slice now provides:

- a one-shot, stale-fenced capture coordinator with explicit armed, encoding,
  submitted, reading, ready, and failed phases, plus cancellation back to idle;
- typed capture targets for advection, pressure, projection, surface update,
  topology ownership, dry-scene HDR, SVO temporal output, interface normals,
  and final composite output;
- full-domain finite/min/max/sign/near-zero reductions, a histogram, and one
  bounded centre-slice preview produced in queue order and mapped only after
  submission;
- zero diagnostic GPU allocations and commands while capture mode is off;
- Performance-tab arming, progress, preview, field interpretation, clean-stage
  baseline, coverage, staging bytes, and instrumented readback latency; and
- coordinator/registry tests, an optional real-WebGPU fixture, production
  build verification, and an in-browser end-to-end capture check.

The next work is deliberately evidence-driven: add dispatch/work-domain
counters and a representative scene matrix before broadening the resource
catalogue. Mean/RMS/percentiles, input/output deltas, pooled multi-resource
staging, capture comparison/export, deeper SVO traversal views, and imported
hardware-counter correlation remain D3-D7 work rather than being implied by
the current UI.

## Verdict on the hypothesis

The optimization premise is directionally right, but the claim that the GPU is
generally underutilized across scenes is not yet proved.

There is already strong evidence that efficiency varies by stage and workload:

- The measured ocean-seiche profile spends 42.5% of a step in advection over a
  nearly full wet domain, while pressure is already area-scaled. A domain and
  activity visualization should expose that mismatch directly.
- The former octree pressure path was dominated by 128 short global dispatch
  boundaries. Replacing it with 32 row-parallel Chebyshev passes reduced the
  pressure stage by 78%. That was a dispatch/dependency problem, not evidence
  that all GPU cores were uniformly idle.
- The Performance tab already attributes simulation and presentation time with
  hardware timestamp queries. Its `GPU busy` value estimates timestamped queue
  work over queue-confirmed wall intervals; it is not a shader-core occupancy,
  cache, bandwidth, or SIMD-lane measurement.
- The octree solver already materializes topology, pressure, and divergence
  fields for live GPU overlays. The SVO path already has G-buffer, generation,
  traversal-status, residency, and timestamp seams that can seed a common
  capture contract.

Stage inputs and outputs will reveal oversized work domains, inactive data,
bad locality proxies, divergent traversal work, redundant materialization,
unexpected state transitions, and numerical pathologies. They cannot alone
prove low hardware occupancy or distinguish an ALU limit from a bandwidth
limit. Portable WebGPU exposes timestamps but not the vendor performance
counters needed for that conclusion. The plan therefore combines four kinds of
evidence:

1. clean stage timestamps and queue cadence;
2. dispatch/workload/synchronization counters;
3. visual resource snapshots and input/output deltas;
4. an external Metal/browser GPU capture when hardware counters are required.

The proposed CPU readback of every input and output should not be the default.
It would duplicate shared stage boundaries, copy large 3D fields, alter the
queue being measured, and can turn a bandwidth or latency investigation into a
readback benchmark. Readback remains useful after the GPU has reduced or
visualized the selected data, and for an explicit bounded raw export.

## Capture contract

Capture is an opt-in, one-shot diagnostic transaction with these states:

```text
off -> armed -> encoding -> submitted -> reading -> ready
                                      \-> failed / stale
```

- `armed` selects one simulation advance and/or one presentation frame. It does
  not continuously capture while the simulation runs.
- The normal uncaptured rolling sample immediately before capture is the timing
  baseline. Captured timestamps are retained separately and marked
  `instrumented`; they never enter realtime demand, scheduling, or rolling
  performance averages.
- Stage-boundary copies are encoded in queue order. The CPU never maps, waits,
  or makes a production decision while the captured advance/frame is encoding.
- `mapAsync` starts only after submission and uses a bounded staging buffer. A
  capture may delay GPU work behind that diagnostic transaction, and
  that cost is reported explicitly, but the CPU scheduler must not wait on the
  map or mutate simulation/render ownership.
- Capture identity includes method, scene, quality, solver generation,
  renderer mode, render timing epoch, simulation time/step, camera, output and
  internal resolution, resource revisions, and adapter identity. A mismatch
  fails stale rather than displaying mixed generations.
- Current limits are one capture in flight, a preview no larger than 256x256
  `vec4f` samples (about 1 MiB plus its summary), and no raw 3D readback. A
  reusable staging pool and explicit raw-export budget belong to D3/D6.

Capturing every logical stage input would copy the same resource repeatedly:
stage N's output is commonly stage N+1's input. The capture unit is therefore a
**resource version**, identified by producer stage, logical resource ID,
generation, subresource, and write version. Immutable inputs are captured once;
each produced version is captured once before it is overwritten. The UI still
presents those versions under every consuming stage.

## Single source of stage truth

Extend the existing `PerformanceStage` model into a stage registry rather than
building an unrelated diagnostic list. Each executable stage descriptor owns:

- stable key, lane (`physics`, `presentation`, or `async`) and queue order;
- timestamp range and active/idle predicate;
- dispatch count, workgroup dimensions, indirect-count source, and known
  global dependency boundaries;
- typed reads and writes using stable resource IDs and access/version rules;
- optional counters and capture providers;
- deterministic interpretation rules and links to producer/consumer stages.

Resource descriptors record kind, shape, format/stride, units, semantic range,
invalid sentinel, visualization, reduction, and raw-copy capability. They must
describe actual bindings or copy sources, not only friendly names such as
`pressure p`. A registry validation test fails duplicate stage keys, missing
producers, impossible copy usages, ambiguous versions, and unsupported
visualizers.

This registry should continue to drive the existing stage cards, dependency
labels, and timing lanes. Diagnostics enrich that model instead of drifting
away from it.

## What is captured

### Always-on lightweight evidence

Keep this small enough for production timing runs:

- timestamp duration, active/idle status, dispatch and global-boundary count;
- logical versus launched work items/workgroups where an indirect count already
  exists;
- active rows/bricks/pages/rays and overflow/exhaustion counts already produced
  by a stage;
- bytes allocated and an analytical byte-traffic lower bound from formats and
  pass counts, clearly labelled as estimates rather than hardware counters;
- queue completion/presentation cadence and CPU encode/submit regions.

Do not add a full-domain reduction to every normal frame merely to populate the
UI. New counters that require scans belong to capture mode.

### Capture-only GPU reductions

For each selected resource version, a diagnostic compute pass writes a fixed
summary record:

- finite/NaN/Inf/invalid counts, min, max, mean, RMS and selected percentiles;
- zero/near-zero, active, clamped and changed fractions;
- a fixed-bin histogram;
- maximum location and, for stage pairs, absolute/relative delta summaries;
- semantic counters such as divergence above tolerance, pressure residual,
  active/halo/retired bricks, traversal exhaustion, temporal rejection reason,
  or material/owner frequency.

Atomics and float ordering can make summaries nondeterministic at the last bit.
Tests use documented tolerances and do not treat capture reductions as solver
state.

### Capture-only visual products

The GPU converts selected fields to compact 2D preview textures before
readback. Default views are:

| Resource | Preview and interpretation |
| --- | --- |
| scalar 3D field | orthogonal/swept slice atlas, histogram, min/max; semantic contour such as `phi=0` |
| velocity | magnitude plus signed components, divergence/vorticity panels, optional downsampled glyphs |
| pressure/residual | signed diverging map, log-magnitude residual, liquid/solid mask |
| topology/owners | leaf level, owner ID, 2:1 transitions, active/dirty classification |
| sparse residency/worklists | core/halo/new/retired atlas, occupancy ratio, indirect work density |
| sparse matrix/rows | row degree, diagonal strength, residual by row/leaf, compact sparsity thumbnail |
| SVO traversal | node visits, DDA/root steps, hit/miss/invalid/exhausted status and divergence heat maps |
| G-buffer/media | HDR color, depth, normal, material/owner, medium transitions and generation validity |
| temporal stage | accepted history weight and rejection-reason heat map |

The first useful view is often output-versus-input change, not two unrelated
textures. Each stage therefore gets `Inputs`, `Outputs`, `Delta`, `Work`, and
`Timing` views where its descriptors support them.

Interpretation text is deterministic and evidence-backed. For example,
`82% of launched advection cells had speed below the configured activity
threshold` is useful; `the GPU looks underutilized` is not available without a
supporting counter. Every interpretation shows the threshold and denominator.

### Raw export

Raw export is selected-resource only. It writes a versioned bundle containing
the manifest, summaries, preview images, timestamps, registry schema version,
and optional packed resource bytes. Formats and row padding are explicit so a
capture is reproducible outside the app. No capture is silently persisted.

## Performance-tab experience

Add a `CAPTURE DIAGNOSTICS` control beside the existing timestamp capture card.
The flow is:

1. choose `selected stage`, `all physics`, `all SVO presentation`, or
   `physics + presentation boundary`;
2. show estimated extra GPU bytes, staging bytes, and expected products;
3. arm the next complete advance/frame and show state without stopping normal
   UI updates;
4. pin the resulting capture to its baseline timing sample;
5. select a stage to inspect timing, work, inputs, outputs, and delta;
6. optionally compare with another capture from the same registry schema.

The stage inspector should show two performance columns:

- `clean`: the authoritative preceding baseline;
- `captured`: the instrumented transaction and its absolute/percentage delta.

A large delta is itself useful evidence that copying or diagnostic reductions
perturbed the stage, but it is never presented as production performance. A
capture from a different scene, method, quality, resolution, renderer mode, or
registry version may be visually compared only after an explicit warning; its
timing ratios are disabled by default.

## Optimization decision matrix

Use the combined evidence to classify a slow stage before changing it:

| Evidence | Likely class | First experiment |
| --- | --- | --- |
| large launched domain, mostly inactive/unchanged | excess work | activity/worklist gating or coarser representation |
| many short dispatches/global scalar boundaries | dispatch/dependency latency | fuse or change the algorithm to do more useful work per boundary |
| high estimated bytes, low state change, few dependencies | bandwidth/copy pressure | reduce formats, ping-pongs, materialization, or passes |
| spatially concentrated high traversal counts | divergence/acceleration structure | summaries, ordering, candidate rejection, or local refinement |
| CPU/queue gaps with short GPU stages | scheduling/host dependency | remove fences/readbacks or batch independent work |
| high useful-work ratio and long arithmetic stage | genuine compute demand | numerical method, approximation, quality scaling, then shader tuning |
| captured-only regression | instrumentation artifact | reduce capture scope; do not optimize production from it |

This classification must precede workgroup-size tuning. Vendor-counter evidence
from Xcode/Metal or a browser GPU trace can then confirm occupancy, bandwidth,
cache, or stall hypotheses for the few stages that remain ambiguous.

## Implementation milestones

### D0 — Freeze the evidence contract

- Document that `GPU busy` is queue-time utilization, not core occupancy.
- Add dispatch/global-boundary/work-domain fields to stage descriptors without
  adding GPU work.
- Record a clean scene matrix for calm tank, dam break, ocean seiche, moving
  rigid bodies, default SVO dry scene, terrain-heavy SVO, and water/glass.
- Gate: the existing timing history and scheduling arithmetic are unchanged.

### D1 — Capture state, identity, and registry

- Add the capture state machine and context/generation identity to the
  diagnostics store.
- Create typed stage/resource registries and validate producer/consumer closure.
- Add capture hooks that are zero-work branches while off.
- Gate: no new resource allocation, command, map, or measurable timing change
  while capture is off; stale captures cannot publish.

### D2 — One vertical slice

- Support one octree physics chain (`advection -> pressure -> projection`) and
  the SVO dry/temporal presentation boundary.
- Capture fixed summaries and one preview per selected stage through a pooled
  asynchronous readback.
- Show clean versus captured timing and capture cost in the Performance tab.
- Gate: no synchronous wait, no scheduling input from capture, bounded memory,
  and exact simulation state parity with capture off/on after the tagged step.

### D3 — Shared visualizers and deltas

- Land scalar/vector/ID/depth/normal/histogram visualizers and version-deduped
  input/output/delta navigation.
- Reuse existing topology, pressure, divergence, and G-buffer materialization
  where semantics match; do not duplicate production overlays.
- Gate: CPU fixtures and small real-GPU fixtures agree on summaries, sentinels,
  slices, signs, units, and change fractions.

### D4 — Complete simulation coverage

- Register octree, quadtree tall-cell, restricted tall-cell, uniform, rigid,
  spray, residency, sparse publication, and asynchronous topology stages.
- Add matrix/row, indirect-worklist, activity, conservation, and residual views.
- Gate: every displayed physics stage has typed resources or an explicit
  `timing/counters only` reason; no phantom input/output names remain.

### D5 — Complete SVO presentation coverage

- Split SVO primary visibility, direct light, shadow, media, temporal, and
  upscale timestamps where pass structure permits meaningful attribution.
- Add visits/steps/rays/status/history counters and heat maps, plus G-buffer and
  media transition previews.
- Gate: unchanged/changed topology and raster/SVO captures retain identical
  renderer-independent simulation state; exhausted/invalid work is visible.

### D6 — Comparison and export

- Compare captures across scenes with strict compatibility rules and export the
  versioned evidence bundle.
- Add a scene matrix view for stage time, useful-work ratio, domain size,
  dispatch count, estimated bytes, and capture interpretations.
- Gate: raw artifacts round-trip, schema mismatches fail closed, and benchmark
  aggregation never consumes instrumented timings as clean samples.

### D7 — Hardware-counter correlation

- Add metadata slots and import notes for external Metal/browser captures; do
  not invent portable WebGPU occupancy.
- Correlate external counters to stable stage labels and capture identity.
- Gate: occupancy/bandwidth/cache claims cite a matching external capture;
  timestamp/work-domain conclusions remain useful when counters are absent.

## Acceptance rules

- Production simulation and rendering remain GPU-resident; diagnostic readback
  never controls work, publication, fallback, quality, or scheduling.
- Capture-off code submits no diagnostic commands and allocates no capture
  resources. The registry and UI metadata are the only resident overhead.
- All capture buffers/textures are pooled, bounded, epoch-fenced, and released
  on solver replacement, renderer-mode change, device loss, or teardown.
- Every preview declares field, units, component/slice, transform, range,
  invalid color, generation, producer stage, and whether values are raw,
  reduced, normalized, or estimated.
- Every claimed optimization is re-measured with capture off using interleaved
  A/B and raw timestamp distributions. Capture explains a result; it does not
  constitute the performance result.
- Native/browser hardware tools remain the authority for actual occupancy,
  bandwidth, cache, and stall counters.

## Initial code map

- `lib/performance-stage-model.ts`: stage registry and executable resource
  descriptors.
- `lib/stores/diagnostics-store.ts`: capture state, identity, manifests, and
  bounded completed captures.
- `components/PerformancePanel.tsx`: capture controls, clean/captured timing,
  previews, interpretations, and comparison.
- `lib/webgpu-eulerian.ts`, `lib/webgpu-uniform-eulerian.ts`,
  `lib/webgpu-quadtree-tall-cell.ts`: physics boundary hooks and counters.
- `lib/webgpu-renderer.ts`, `lib/webgpu-svo-dry-scene.ts`,
  `lib/webgpu-svo-temporal-accumulator.ts`: presentation hooks, G-buffer and SVO
  counter products, and timing epoch fencing.
- New `lib/gpu-stage-capture.ts`: state machine, resource-version deduplication,
  budgets, pooled staging, mapping, and manifest assembly.
- New `lib/webgpu-diagnostic-visualizers.ts`: GPU summary and preview kernels.
- New focused CPU/mock/real-GPU tests plus a capture-bundle schema fixture.
