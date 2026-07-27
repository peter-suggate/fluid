# Frame profiler architecture

Status: proposed design for the central wide-column profiler. The right-side
range inspector, capture history, comparisons, and export are intentionally out
of scope for the first implementation.

## 1. Decisions

1. The initial profiler captures one coherent presentation frame, including the
   CPU work and GPU submissions associated with the physics publication that
   frame consumes.
2. GPU pass timing uses WebGPU timestamp queries. It does not require shader
   changes or pipeline recompilation.
3. Compile-time WGSL variants are reserved for deep shader telemetry: logical
   workgroup participation and algorithm-specific counters. Enabling that mode
   asynchronously compiles instrumented pipelines and atomically swaps them in.
4. The UI distinguishes measured, shader-reported, derived, and unknown data.
   It never presents inferred utilization as a hardware measurement.
5. CPU lanes represent software execution contexts (main thread and named
   workers), not physical CPU cores. GPU lanes represent the queue, passes,
   dispatches, and logical workgroups, not physical GPU compute units.
6. The existing `PerformanceTrace` remains as a derived/legacy exact-accounting
   summary. A new event model preserves absolute positions, nesting, repeated
   stages, resource identity, and capture lineage.
7. UI and capture work proceed in parallel against the immutable
   `ProfilerCapture` contract in section 5.

## 2. Capability boundary

WebGPU timestamp queries measure the beginning and end of compute/render passes.
They live on pass descriptors, outside WGSL. The existing
`GPUStageTimestampRecorder` in `lib/performance-trace.ts` already demonstrates
this path, and the pass timestamp audit in `tools/run-webgpu-smoke.ts` provides a
useful prototype for exhaustive pass interception.

WGSL exposes logical `workgroup_id`, `local_invocation_id`,
`local_invocation_index`, and feature-dependent subgroup identifiers. It does
not expose a shader clock or the physical core/CU executing an invocation.
Consequently, a shader can report:

- scheduled, reached, and completed logical workgroups;
- logical workgroup IDs or sampled workgroup records;
- active-item, branch, loop, rejection, and algorithm-specific counters;
- direct dispatch dimensions and instrumented indirect-dispatch participation.

It cannot report portable per-workgroup start/end times or physical-core IDs.
Writing an atomic event order is not a clock and must not be drawn as one.

Likewise, browser JavaScript can time work in the main context and Workers with
`performance.timeOrigin + performance.now()`, but cannot identify the physical
CPU core that ran it.

GPU timestamps share a queue-relative time domain. WebGPU does not expose a
calibrated mapping from that clock to the CPU monotonic clock. The combined view
therefore keeps separate exact CPU and GPU accounting ledgers. A common ruler
may align GPU work to the CPU `queue.submit` bracket, but that offset is marked
estimated and carries an uncertainty range.

Relevant standards:

- [WebGPU timestamp queries](https://gpuweb.github.io/gpuweb/#timestamp-query)
- [WGSL built-in values](https://www.w3.org/TR/WGSL/#builtin-values)
- [High Resolution Time across Window and Worker contexts](https://www.w3.org/TR/hr-time-3/)

## 3. Capture profiles

| Profile | Shader variant | Captures | Perturbation |
| --- | --- | --- | --- |
| `off` | production | nothing | none |
| `timeline` | production | nested CPU spans, submissions, every real GPU pass, dispatch metadata | timestamp query/resolve overhead only |
| `counters` | instrumented | timeline plus bounded workgroup and algorithm counters | shader atomics, extra binding, buffer traffic |
| `workgroup-log` | instrumented debug variant | sampled record per logical workgroup | high; debug/short captures only |
| `dispatch-detail` | production or counters | pass splitting to time individual dispatches | high; changes pass fusion and scheduling |

Detailed activity plus timeline capture is the product default; the panel has
one Capture selector for `Detailed`, `Timeline only`, and `Off`. `counters` is the first shader-recompile
mode. `workgroup-log` and `dispatch-detail` are explicit diagnostic modes, not
the default profiler.

The requested/effective lifecycle is:

```text
off
  -> rebuilding-instrumented-pipelines  (counters only)
  -> armed
  -> capturing
  -> draining
  -> ready
```

Error and cancellation can occur from rebuilding, capturing, or draining. The
toolbar shows both requested and effective profiles plus the effective pipeline
generation/hash. The last finalized capture stays visible while a new variant
is compiling and is marked historical.

## 4. What is a frame?

The current renderer can submit physics before or after presentation and can
queue multiple advances. A profiler frame must not assume one physics submit is
one render submit.

Each `requestAnimationFrame` draw receives a monotonic `frameId`. That lineage is
propagated through:

- controller tick and renderer draw;
- every admitted physics advance;
- command encoders and queue submissions;
- solver/publication generation;
- the presentation submission that consumes a publication;
- worker requests/responses and asynchronous readback/completion callbacks.

The captured GPU frame is defined as one presentation frame plus the physics
submission or submissions whose publication generation it consumes. Physics
queued after that presentation remains visible in its real queue order but is
associated with the later publication/frame. We record reality rather than
forcing the queue into a conceptual frame boundary.

The first implementation is a one-shot capture: arm the next eligible complete
presentation frame, drain its timestamp/counter data asynchronously, then
freeze it for exploration.

## 5. Shared capture contract

Time is stored as integer nanoseconds relative to each clock-domain origin. Raw
GPU timestamp values are retained until capture finalization.

```ts
type EvidenceSource =
  | "cpu-wall"
  | "gpu-timestamp"
  | "shader-counter"
  | "queue-wall"
  | "derived"
  | "unknown";

type EvidenceQuality = "exact" | "estimated" | "truncated" | "unknown";

interface ProfilerCapture {
  schemaVersion: 1;
  captureId: string;
  frameId: number;
  context: string;
  profile: "timeline" | "counters" | "workgroup-log" | "dispatch-detail";
  pipelineGeneration: string;
  duration_ns: bigint;
  clocks: ClockDomain[];
  lanes: ProfilerLane[];
  spans: ProfilerSpan[];
  submissions: GpuSubmission[];
  counters: CounterSeries[];
  occupancy?: LogicalOccupancyBuffers;
  accounting: AccountingLedger[];
  capabilities: CaptureCapabilities;
  truncation: CaptureTruncation;
}

interface ProfilerLane {
  id: string;
  parentId?: string;
  kind: "group" | "cpu-thread" | "gpu-queue" | "gpu-pass" |
        "logical-workgroups" | "counter" | "summary";
  label: string;
  order: number;
  height: number;
}

interface ProfilerSpan {
  id: number;
  laneId: string;
  parentId?: number;
  stableStageId: string;
  label: string;
  start_ns: bigint;
  end_ns: bigint;
  depth: number;
  source: EvidenceSource;
  quality: EvidenceQuality;
  frameId: number;
  submissionId?: number;
  passId?: number;
  dispatchId?: number;
  attributes?: Record<string, string | number | boolean>;
}
```

Large occupancy and counter data uses typed-array/structure-of-arrays storage,
not one JavaScript object per workgroup. Immutable finalized captures live in a
byte-bounded repository (initial target: 64 MiB or 16 frames). Zustand stores
only lifecycle, capture/version IDs, viewport transform, collapsed lanes, and
selection.

Stable stage IDs are namespaced and never contain dynamic values, for example:

```text
cpu.main.simulation.tick
cpu.worker.quadtree.system.pack
gpu.physics.octree.fine.transport.classify
gpu.physics.octree.pressure.mgpcg.iteration
gpu.presentation.water.surface.polygonise
gpu.presentation.svo.primary
gpu.presentation.present.upscale
```

Substep, iteration, method, shader variant, workgroup geometry, and current
pipeline label are span attributes. Display labels come from a registry. The
existing `PaperPhaseId` values remain roll-up categories rather than unique IDs.

## 6. Capture service

Add a device-scoped `PerformanceTraceHub`, owned alongside the renderer/device.
Its production implementation is a frozen no-op object, so call sites do not
branch repeatedly.

Core API:

```ts
hub.beginFrame(frameContext)
hub.beginSpan(descriptor) / hub.endSpan(token)
hub.withSpan(descriptor, fn)
hub.beginSubmission(descriptor)
hub.instrumentCommandEncoder(encoder, submission)
hub.endFrame(frameId)
hub.cancelGeneration(deviceGeneration)
```

The instrumented command encoder records stable IDs for clears, copies, compute
passes, render passes, dispatches, draws, and `finish`. Compute/render pass
proxies record pipeline changes and direct/indirect dispatch metadata. Each real
pass receives beginning/end timestamp indices. Multi-dispatch passes remain one
measured GPU span in `timeline` mode; their child dispatches have exact command
order and geometry but unknown individual duration.

Use a preallocated ring of 4-8 capture slots. Each slot owns a fixed query set,
resolve buffer, MAP_READ buffer, and optional telemetry buffers, with states:

```text
free -> encoding -> submitted -> mapping -> free
```

Resolve and copy commands are encoded into the measured submission. Mapping is
asynchronous and never blocks the render loop. Query/counter capacity is fixed;
overflow drops detail, sets truncation metadata, and preserves the accounted
outer interval. Queue-wall fallback remains a clearly marked coarse source.

The current boundary recorder's Dawn/Metal 4-byte encoder-break copies remain a
portability fallback, not the default exhaustive pass path, because they change
the command stream.

## 7. CPU instrumentation

Replace flat phase transitions with nested, allocation-light spans backed by
preallocated typed arrays. Instrument:

- controller clock, admission, CPU-reference solve, coupling, and publication;
- renderer readiness/path selection, physics admission, uploads, command
  encoding, submit, completion scheduling, and diagnostics publication;
- top-level `encode*` calls and command encoder/pass/dispatch/draw CPU work;
- quadtree worker request transfer, worker-side preparation stages, response
  transfer, and receive/install work;
- async readback, timestamp decode, capture finalization, and error/device-loss
  paths.

Workers emit epoch-relative timestamps using
`performance.timeOrigin + performance.now()` and include their stable logical
worker ID and frame/capture lineage. Gaps are explicit idle/browser/GC/JIT/driver
unknown time; the profiler does not assign them to a CPU core.

## 8. GPU timeline instrumentation

Promote the pass timestamp interception pattern from
`tools/run-webgpu-smoke.ts` into the device-scoped capture hub. Every real pass
gets two queries where supported. The encoder/pass proxy also records:

- stable pass and pipeline IDs;
- pass label and semantic parent stage;
- direct dispatch dimensions;
- indirect dispatch buffer identity/offset;
- draw geometry;
- command order, submission ordinal, and solver/publication generation.

`PassBroker` requests gain a stable stage descriptor. Semantic fences end the
current child stage. A named dispatch helper eventually replaces bare
`setPipeline + dispatch*` pairs so exhaustive coverage can be asserted.

The current top-level physics taxonomy is retained:

- adaptive coarse-grid topology;
- power topology and physical faces;
- velocity transport and body forces;
- pressure operator/RHS and Section 4.3 MGPCG solve;
- projection and extrapolation;
- fine SDF advection, redistance, restriction, and publication;
- rigid exchange, sparse render publication, and diagnostics.

Presentation retains extraction, caustics, SVO/dry-scene stages, front/back
water interfaces, optical composite, overlays, and final present.

## 9. Compile-time shader telemetry

The user's proposed shader option becomes an explicit cache-key dimension:

```ts
type InstrumentationProfile = "off" | "counters" | "workgroup-log";
```

Every shader/pipeline cache key includes profile, telemetry ABI version, module
ID, entrypoint ID, and relevant feature set. A centralized shader/pipeline
factory builds the WGSL variant; raw booleans are not scattered across shader
files. The current source audit finds 73 shader-module creation sites in 48
files and 110 compute-pipeline creation sites, so a manifest and coverage audit
are required.

Instrumented variants inject a telemetry ABI only when enabled. The preferred
layout is a dedicated high-numbered bind group (currently group 3) containing:

- a bounded read-write telemetry buffer;
- a dispatch descriptor selected by dynamic offset;
- header atomics for capacity and overflow;
- per-dispatch aggregate records;
- optional sampled logical-workgroup records.

This is conditional on adapter limits and each pipeline's binding/storage
budget. A capability manifest records support per entrypoint. Where a dedicated
group is illegal, use an existing diagnostics buffer only if its ABI permits;
otherwise mark the entrypoint counter-unsupported rather than silently changing
the measurement.

Before each profiled dispatch, the pass proxy assigns a `dispatchId` and binds
its dynamic telemetry slot. A workgroup leader (`local_invocation_index == 0`)
increments reached/completed counts. Algorithm-specific helpers aggregate
within the workgroup and perform bounded atomic updates for branch, loop,
active-item, rejection, or memory-operation counters. `workgroup-log` sampling
records logical `workgroup_id`; it still records no clock or physical core ID.

Instrumented pipelines compile asynchronously while production pipelines keep
running. Swap only when every required pipeline for the chosen capture is
ready. Turning instrumentation off swaps immediately to cached production
pipelines and drains old telemetry resources. Solver state is not rebuilt.

The central view may derive logical utilization from pass duration, scheduled
versus reached workgroups, workgroup size, and device limits. Such rows are
named `Logical workgroup occupancy` or `Derived utilization` and use the
`derived/estimated` evidence style.

### Minimal atomic-clock feasibility result

`tools/experiment-webgpu-atomic-clock.ts` is a standalone, bounded proof of
concept. One workgroup increments a storage atomic while 2,048 synthetic
payload workgroups record the counter and a separate ordering ticket before and
after deterministic ALU work. A hardware timestamp query measures the complete
pass, and a paired non-instrumented pipeline estimates perturbation.

On the local Metal adapter, five 16,384-iteration trials produced:

- 100% of payload workgroups observed the clock running at entry;
- 100% observed the clock advance during their payload;
- all 2,048 completions, 4,096 ordering tickets, and sampled checksums closed;
- median hardware pass time 5.156 ms instrumented versus 5.210 ms baseline,
  which is inside run-to-run noise rather than evidence of negative overhead;
- clock-rate coefficient of variation 12.6%, so a single affine tick-to-time
  conversion is reconstruction evidence, not precision timing.

A shorter 1,024-iteration pass also retained 100% clock visibility/advance over
three trials at a 1.049 ms median pass time, with 3.1% clock-rate variation.
The same longer workload at workgroup size 64 retained 100% visibility and
advance, had 6.4% clock-rate variation, and measured 4.981 ms for both baseline
and instrumented medians. It passed the experimental gate.

The size-64 short workload exposed the practical floor. One three-trial run
doubled a 0.328 ms baseline and produced 40.8% clock-rate variation. A separate
ten-trial run was better but still showed 18.9% variation and 15.3% median
overhead. Every workgroup observed the clock in both runs, but the inferred time
scale was not reproducible enough. The experiment therefore uses a 15%
clock-rate-variation gate, and the initial integration should attempt
atomic-clock reconstruction only for multi-millisecond compute passes. Short
passes remain hardware-timestamp-only. Memory-heavy real solver kernels and
indirect dispatches still require separate calibration.

Run the experiment with:

```text
node --import tsx tools/experiment-webgpu-atomic-clock.ts
```

### Low-bloat adoption audit

The implemented adoption seam is deliberately small:

- CPU workloads receive one optional `CPUPerformanceActivityProfiler`; the
  disabled default is a frozen no-op that does not read a clock, allocate, or
  touch Zustand. A workload is one `activity.measure(task, () => work())`
  expression. The CPU reference solver now covers timestep selection, inflow,
  forces, velocity transport, viscosity, both divergence checks, pressure,
  marker transport, and diagnostics. Its controller also covers fluid loads,
  fluid reactions, rigid integration/contact work, and the nested fluid step.
  Quadtree CPU preparation uses the same optional argument around unpacking,
  pressure-grid construction, topology identity, solid fields, variational
  assembly, and packing; browser/Node worker transport can carry its output.
- GPU modules receive one immutable `GPULogicalActivityAdoptionContext` while
  compiling. Disabled and timestamp-only contexts return byte-identical WGSL,
  an empty checkpoint fragment, a stable disabled-variant cache key, and no recorder.
  Enabled contexts add the generation to the cache key and inject the bounded
  heartbeat ABI after WGSL `enable`/`requires` directives.
- A frame-scoped encoder proxy observes `setPipeline` and binds group 3 only for
  registered instrumented pipelines. Dispatch call sites therefore do not gain
  a telemetry bind or a store branch. Auto-layout pipelines need only be
  registered after construction; explicit layouts must append group 3.
- The dedicated activity store retains eight finalized frames by default,
  rejects late generations, and keeps raw spans/events authoritative. Its 1 ms
  rows are derived indices and distinguish measured, reconstructed, idle, and
  unknown evidence.
- `gpuLogicalActivityMatrixAddition` converts decoded heartbeat records into
  one vertical row per observed logical workgroup/subgroup and one horizontal
  cell per millisecond. It requires an external measured or reconstructed time
  projection; append order is never treated as time. Asynchronous readback is
  merged into the already-retained frame through `mergeEvidence`.

The source has 116 existing pass-broker compute calls and 81 direct
`beginComputePass` references (including profiler infrastructure). Those are
the right exhaustive timing seam: pass timestamps require no shader changes
and apply to every shader, including render shaders and storage-saturated
compute shaders.

Deep heartbeat coverage cannot honestly be universal through an extra storage
binding. The production device contract is ten storage buffers per shader
stage, and several current entrypoints already use all ten. Known saturated
families include structured dynamics/advection, structured boundary work, air
support, SPGrid/Section 4.3 pressure work, fine summary/refinement, the global
fine water classifier, and an SVO dry-fragment path. Those modules must declare
`support: "timestamp-only"` until either an existing storage arena receives a
telemetry region or the shader layout is reduced. The facade then fails closed:
activity mode leaves their WGSL byte-identical and the UI reports detailed
activity as unknown while retaining exact pass duration.

Group 3 binding 0 is currently unused by literal production WGSL (production
shaders use groups 0 and 1). Dynamic SVO group remapping and every explicit
pipeline layout still require validation during adoption. The important
coverage invariant is therefore two-tiered:

1. every CPU physics workload and every GPU pass has timing coverage;
2. every compute entrypoint has an explicit heartbeat capability of
   `dedicated-group`, future `existing-arena`, or `timestamp-only`.

This avoids pretending that a timestamp interval proves shader lanes were
continuously active. A heartbeat proves progress at a point; it marks the 1 ms
cell as measured but does not invent a full millisecond of occupancy.
Aggregate CPU/GPU phase envelopes are displayed in a separate section and are
never labeled as logical workgroup rows. Until heartbeat readback arrives, the
GPU logical matrix explicitly says that its rows are awaiting data.

### Octree shader rollout and binding policy

Heartbeat adoption is entrypoint-reachability based, not module-declaration
based. `auditWGSLComputeBindingReachability` follows the transitive WGSL call
graph and counts only resources reachable by a specific compute entrypoint.
`assertWGSLActivityBindingEligibility` then enforces the hard rule:

```text
reachable production storage bindings + 1 activity binding <= 10
```

For explicit layouts, the same assertion is made against the compute-visible
layout entries. A module-wide count is insufficient because unrelated kernels
often share a large source string while using disjoint buffer sets.

The rollout order is:

1. Adopt activity-safe recurring families first: power-volume publication,
   fine-volume/JFA work, structured velocity publication, coarse-phi bootstrap,
   fine-to-coarse restriction, and all pipelined MGPCG stages.
2. Split broad explicit layouts by reachability. The projection/topology shader
   now keeps its common family at nine storage bindings and isolates frontier
   merge-sort scratch in a four-storage layout, leaving one recorder slot in
   both families without changing dispatch order or WGSL algorithms.
3. Continue through fine bricks/transport/topology, owner and power-topology
   publication, structured dynamics/boundary/air support, and the
   SPGrid/Section 4.3 preconditioner families. Each entrypoint receives a stable
   task descriptor beside its shader and an explicit capability test.
4. Keep a ten-storage entrypoint timestamp-only until one of these safe changes
   is proven: remove a genuinely unreachable legacy layout entry, split a
   mixed-purpose layout, alias buffers whose lifetimes and access modes are
   mutually exclusive, or reserve a telemetry region inside an already-owned
   diagnostics arena. Never add an eleventh storage binding or silently omit
   the entrypoint from the capability audit.

Repeated solver kernels use bounded logical-workgroup sampling; cold singleton
publication kernels emit one progress checkpoint. The shared recorder reports
overflow and the number of dropped appends while retaining its complete,
sequence-validated prefix. Truncation therefore leaves the unrecorded tail
unknown instead of erasing the entire matrix or presenting it as idle.

## 10. Central timeline UI

The current `PerformancePanel` lays additive duration segments and cannot
represent a positioned nested timeline. Keep it as the legacy observatory while
adding a new central workspace.

Suggested components:

```text
components/performance/
  PerformanceWorkspace
  ProfilerToolbar
  ProfilerTimeline
  TimelineRulerCanvas
  TimelineCanvas
  VirtualLaneGutter
  TimelineTooltip
```

`PerformanceWorkspace` overlays the still-mounted WebGPU viewport so capture
ownership and renderer state survive switching views. Add a center workspace
state such as `scene | profiler`; do not continue overloading the right-panel
state.

Use a hybrid DOM + Canvas 2D renderer:

- DOM for toolbar, virtualized lane gutter, tooltip, and accessibility;
- DPR-aware Canvas 2D for grid, spans, flame nesting, occupancy bins, and
  counter tracks;
- draw only on capture/view/interaction changes through `requestAnimationFrame`;
- vertically flatten expanded lanes and render the viewport plus overscan;
- binary-search start-sorted spans per visible lane;
- bin events by pixel/category when their projected width is below 0.5 px;
- draw labels only when the bar is wide enough; never inflate narrow bars.

Viewport state is `viewStart_ns` plus `nsPerPx`. Wheel zoom anchors at the
pointer; horizontal pan uses Shift+wheel or drag; `F` fits selection/full frame;
hover and selection use indexed hit testing. A 1/2/5 time-ruler scale spans ns,
us, and ms.

The first UI phase contains the toolbar, ruler, frame flame rows, CPU software
thread rows, GPU queue/pass rows, and lower counter tracks. It omits the right
inspector. Physical-core rows are absent unless a future native backend supplies
real hardware evidence.

## 11. Parallel implementation plan

### Workstream A: central timeline UI

Can start against a synthetic `ProfilerCapture` immediately:

1. workspace state/shell and mock capture;
2. pure timeline model, transform, lane flattening, indexing, and LOD;
3. Canvas renderer plus virtual lane gutter;
4. zoom, pan, hover, selection, collapse, fit, and partial-capture states;
5. adapter for current independent `PerformanceTrace` samples, clearly marked
   as independent lanes for temporary UI bring-up.

### Workstream B: capture and instrumentation

1. new capture schema, stable stage registry, lifecycle store, and capture hub;
2. coherent frame/submission/publication lineage;
3. nested CPU spans and Worker clock translation;
4. exhaustive real-pass timestamp interception and ring-buffer readback;
5. migrate pass brokers/dispatch helpers and add coverage assertions;
6. compile-time counter variants, telemetry ABI, and capability matrix.

### Join point

Both streams join only through a finalized immutable `ProfilerCapture`. The UI
does not import renderer/solver classes, and instrumentation does not know about
Canvas geometry or view state.

## 12. Delivery phases

### Phase 1: truthful one-frame timeline

- central workspace and canvas timeline;
- one-shot coherent frame capture;
- main-thread and quadtree-worker nested CPU spans;
- timestamps for every real GPU compute/render pass;
- stable hierarchy for existing physics and presentation stages;
- exact per-domain accounting and explicit unknown intervals;
- capability/evidence styling and overflow reporting.

### Phase 2: exhaustive command attribution

- stable registry for all pass/pipeline/entrypoint IDs;
- named dispatch helpers throughout physics and presentation;
- direct/indirect dispatch metadata;
- automated coverage audit for unregistered shader/pipeline/pass creation;
- richer worker, async, readback, and device-loss spans.

### Phase 3: deep shader counters

- central shader/pipeline variant factory and cache key;
- asynchronous instrumented pipeline compilation/swap;
- special telemetry buffer and dynamic dispatch slots;
- per-entrypoint capability matrix;
- bounded workgroup/algorithm counters and derived utilization rows.

### Phase 4: optional native hardware backend

True per-core/CU utilization, hardware counter sampling, and calibrated CPU/GPU
clocks require a native backend/tooling path (for example Metal/Vulkan/Dawn
integration). That data can enter the same capture contract with a new evidence
source without redesigning the central UI.

## 13. Verification and budgets

- exact accounting: child interval unions plus explicit unknown gaps close to
  each parent within one timestamp quantum;
- query/counter overflow and unsupported capability are visible, never dropped;
- capture never waits synchronously for GPU completion or buffer mapping;
- no span DOM nodes; fewer than 200 visible lane DOM nodes;
- target Canvas interaction render below 8 ms;
- synthetic million-event index test with fewer than 10,000 visible draw
  commands after LOD;
- unit tests for clock translation, transform, lane flattening, visible-range
  query, hit testing, nesting, exclusive-time accounting, ring reuse, overflow,
  device loss, and variant swap;
- source/manifest audit fails when new shader modules, pipelines, passes, or
  dispatches bypass the approved profiling seams.
