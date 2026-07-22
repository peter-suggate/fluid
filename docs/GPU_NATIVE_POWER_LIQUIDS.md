# GPU-native Power Liquids

This implementation preserves the numerical method in Aanjaneya et al. 2017,
not the scheduling assumptions of its CPU/SPGrid implementation. The target is
a bandwidth-oriented GPU pipeline with deterministic ownership, immutable
topology publications, bounded gathers, and no atomics.

## Semantics that remain invariant

- The pressure operator is the symmetric power-diagram discretization of
  Section 4.1 and Equation (3), with the same primal/dual orthogonality.
- The Section 4.3 preconditioner remains linear and SPD: `k` power-operator
  boundary sweeps, one first-order sparse-pyramid V-cycle, then matching `k`
  sweeps. Convergence and residual acceptance are unchanged.
- The fine level set remains a separate factor-4 or factor-8 narrow band. One
  coarse step traces one piecewise characteristic in `m` velocity samples and
  performs one final phi interpolation, as required by Section 5.
- Cell-centred least-squares velocity reconstruction, cube/tetrahedral
  interpolation, signed-distance ordering, redistance, and fine-to-coarse
  correction retain their existing mathematical definitions.
- Every generation remains fail-closed. Capacity, non-finite data, incomplete
  topology, an invalid tetrahedron, or a failed residual rejects publication.

Those requirements constrain results, not buffer layout, work ownership,
kernel fusion, ordering of independent records, or how topology is indexed.

## Required pipeline

### 1. Build topology as sorted immutable data

1. Emit one or a fixed number of Morton/key records per source item. No append
   counters are permitted.
2. Stable radix-sort records by `(generation, level, Morton key, relation)`.
3. Mark run boundaries and use an exclusive prefix sum to allocate unique
   cells, faces, adjacency ranges, and error records.
4. Materialize structure-of-arrays publications with one deterministic writer
   per output record.
5. Construct reverse/parent adjacency by the same sort/scan process. Consumers
   gather owned child contributions; they never scatter with atomic adds.
6. Reduce validation flags and counts through workgroup and hierarchical
   reductions. Diagnostic counters do not justify atomics in production code.

The published generation contains direct row indices, compact descriptor and
catalog IDs, cube-neighbour indices, tetrahedron vertex rows, parent/child
ranges, and bounded worklists. Hash tables may be used during bring-up, but are
not part of the target recurring or construction pipeline.

### Current atomic debt and removal order

The zero-atomic sub-goal is not yet complete. Characteristic transport, its
direct power-velocity prepass, and the recurring air sampler are atomic-free.
The remaining recurring production atomics are removed in this order:

1. Split surface-page lifecycle metadata from phi payload. Phi samples have
   exclusive owners and must use ordinary storage loads/stores; only page
   publication needs a separate construction path.
2. Replace fine desired-brick hash claims and counters with Morton-key emission,
   radix sort, boundary marking, and scan compaction.
3. Replace fine summary hash/CAS merges with sorted `(parent, child)` records
   and deterministic segmented reductions.
4. Replace JFA-CPT accepted/residual and face-transport CFL atomics with local
   workgroup reductions followed by one hierarchical reduction.
5. Apply the same sorted-record construction to adaptive leaves, power
   descriptors, generalized faces, transfer records, and generation errors.

These are all recurring per outer advance in the current implementation; none
may be dismissed as startup-only work.

### 2. Reuse topology by generation

Geometry/topology is built once for a generation and remains read-only while
pressure, extrapolation, characteristic tracing, redistance, restriction, and
rendering consume it. A consumer never rediscovers a one-ring, probes a row
hash, or scans every row for each sample. Geometry changes create a new A/B
publication; they do not mutate the live generation.

### 3. Make transport a bounded gather

Each query resolves its owner from the fine-brick directory, then performs a
bounded indexed gather from the immutable cube/tetrahedron publication. Direct
regular interpolation and catalog-Delaunay interpolation are separate compact
worklists. There is no per-query full-row fallback in the target path. A query
that lacks a published bounded stencil rejects the generation instead of
launching an unbounded search.

Trajectory segments should be fused when register pressure permits. If they
remain separate, their input and output are dense arrays with deterministic
indices; classification, evaluation, and finalization do not communicate via
atomics.

### 4. Make pressure device-driven

For small compact solves, one persistent workgroup owns the solve. Threads
stride over rows and sparse-pyramid slots; former dispatch boundaries become
workgroup barriers. Dot products use deterministic workgroup reductions and
convergence terminates the device loop. Parent-owned gathers replace
restriction/accumulation atomics.

For domains too large for one persistent workgroup, use a small fixed number
of fused kernels per level and hierarchical reductions. The host may select
the small or large path from immutable capacities, but it does not schedule
iterations or read convergence during a solve.

### 5. Keep the queue dense

A normal advance has no diagnostic maps, host topology packing, adaptive host
batching, or queue fences. Intrusive phase fences exist only in the opt-in
profiler. Pipelines and bind groups are retained. Work is encoded in a small
number of coarse command sections so Metal/WebGPU pass-transition cost cannot
dominate a small scene.

## Rejected architectures

- Fixed tails whose kernels merely check a convergence atomic still pay command
  processing for every encoded dispatch. The current pressure solve converges
  in 7 iterations but encodes thousands of commands.
- GPU-written indirect gates split by WebGPU usage scopes are not viable here.
  On the exact 24x18x16 trace they changed host pressure encode from 22.3 ms to
  202.3 ms and pressure wall time from 139.8 ms to 179.0 ms.
- Caching a small direct interpolation path while retaining the per-query
  full-row/surrounding search does not remove the evaluator bottleneck. A cache
  prototype increased the four-segment evaluator sum from 66.0 ms to 76.1 ms
  and was reverted.
- A first persistent-pressure prototype compiled and removed recurring atomics,
  but was rejected before production integration: it did not preserve SPGrid
  ghost-row smoothing/residual semantics, its boundary restriction and
  prolongation were not the validated adjoint pair, and invalid sparse rows
  could reach arithmetic before a uniform fail-closed gate. A replacement must
  prove ghost behavior, low/high boundary transfer adjointness, invalid-row
  rejection, and numerical Dawn parity before wall-time measurement.
- Replacing atomics with many extra passes is not success. Atomics, dispatches,
  pass transitions, bytes moved, and wall time are joint constraints.

## Acceptance gates

A performance change is accepted only when all of these hold:

1. `npm run test:webgpu:dam-ui-performance` reproduces the exact browser scene,
   resolved default octree profile, 24x18x16 grid, 0.008 s GPU advance cap,
   timestamp/readback mode, and evolved third 30-advance profiler sample.
2. The changed phase improves its queue-boundary wall time on repeated Metal
   runs. Total advance wall time must not regress or merely move the cost to an
   adjacent phase.
3. The two-step exact Dawn smoke has zero validation errors and identical
   publication/authority acceptance.
4. The 500-step minimal dam-break gate completes exactly, remains finite and
   connected, and preserves the existing volume and motion checks.
5. Recurring shaders contain zero atomic operations. New topology and pressure
   paths also contain zero atomics before replacing their existing fallbacks.
6. Normal production has no new readback, fence, per-iteration host decision,
   or simulation-sized host work.

The UI is used once to calibrate the Dawn reproduction. Subsequent diagnosis,
optimization, and regression measurement use Dawn exclusively.

## Implementation references

- [Aanjaneya et al. 2017](papers/aanjaneya-2017-power-liquids.txt)
- [Work-efficient parallel prefix sum](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)
- [GPU radix sorting](https://research.nvidia.com/publication/2009-05_designing-efficient-sorting-algorithms-manycore-gpus)
- [Morton-key hierarchy construction](https://research.nvidia.com/publication/2012-06_maximizing-parallelism-construction-bvhs-octrees-and-k-d-trees)
