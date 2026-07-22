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

### Measured baseline and topology spine

There are two separate Dawn authorities. `test:webgpu:dam-ui-performance` is
the exact browser parity lane: the canonical `water-box-dam-break` preset,
24x18x16 grid, 0.008 s GPU outer step, factor-4 fine level set, Metal adapter,
and the evolved third 30-advance profiler sample. The 16x16x16
`minimal-power-dam-break` remains the longer numerical regression lane; it is
not a substitute geometry for UI performance. `test:webgpu:dam-ui-throughput`
differs from the parity lane only by disabling intrusive profiler readbacks and
is the optimization throughput authority.

The one-time browser calibration measured an ordinary single-submission
production advance at 86.6 ms. The matching profiler-free Dawn run measured
5.559 s for 62 advances, or 89.7 ms/advance: a 3.5% difference. The exact Dawn
phase sample measured 85.51 ms and attributed 47.25 ms to pressure/project and
35.67 ms to fine surface. Rendering was outside the production physics fence.
The browser's boundary-split replay instead took 3704.6 ms (2050.5 ms charged
to fine surface and 1483.1 ms to pressure) because it submitted and fenced at
every semantic boundary. Fine transport has the most boundaries, so that
replay inverted the true ordering. The UI now rejects a split sample whose
total exceeds the ordinary production fence by more than 25%; rejected phase
ratios are not performance evidence.

The accepted live-prefix topology trace now completes 62 exact advances in
4.240 s, or 68.39 ms/advance, a 23.73% improvement over the calibrated
89.66 ms/advance baseline. It encodes 114,575 dispatches, 63,736 indirect
dispatches, and 15,499 compute passes: about 1,848 dispatches and 250 pass
transitions per advance. The pass count includes explicit publication fences
between Section 5 candidate emission, owner resolution, and row insertion.
Those fences fixed a measured schedule-sensitive generation-95 row-publication
failure while removing an invalid cross-workgroup polling loop. Even with the
additional correctness boundaries, wall time fell.

The same trace clears 273,720,576 bytes and copies 551,117,504 bytes over the
run, or 4.41 MB of clears and 8.89 MB of copies per advance. Counted face-slot
retirement removed the former 5.31 MB/advance Section 5 face-arena clear; the
remaining dominant clear is now the old interpolation mesh at 2.10 MB/advance.
The dominant copy remains the canonical face publication into that retained
mesh at 5.31 MB/advance, followed by SPGrid's captured L1 entries at
1.99 MB/advance. Recurring clear traffic is down 82.4% from the original exact
trace. Copy traffic is unchanged apart from four bytes per advance carrying
the previous-face generation into a new fail-closed compact publication.

The original 16x16x16 62-step warm baseline was 14.414 s. Removing a dead
old-mesh snapshot reduced that to 11.322 s. The current profiler-free run is
4.015 s, or 64.8 ms/advance, with zero validation errors. It remains useful for
fast iteration but is intentionally reported separately from the exact UI
scene.

One deliberately incomplete eight-owner local-search experiment reduced a
comparable 11.35 s warm run to 4.195 s. It was rejected after a checked-in
catalog counterexample proved that the eight owners do not form a complete
candidate set. The production replacement enumerates the complete 5x5x5
dyadic origin neighbourhood at each legal leaf size, validates exact row
identity through `rowHash`, and preserves lowest-row selection. It reduced the
same 62-step Dawn contract to 4.797 s with no validation errors. Its four
air-evaluator segments totalled 5.46 ms instead of the former 81.4 ms. This is
the first accepted structural result: capacity-sized row search, not the
catalog interpolation arithmetic, was the evaluator bottleneck.

Replacing SPGrid's direct full-capacity correction schedule with live indirect
work reduced the 62-step contract again to 4.562 s, about 4.9%. The modest wall
change despite a roughly 17x workgroup-count reduction is important evidence:
dispatch count alone was not the dominant remaining limit. The corrected
queue-boundary trace attributed 53.0 ms to pressure/projection, only 7.36 ms of
which was the isolated pressure solve, while Section 5 fine transport used
12.86 ms and its exact evaluator used 5.46 ms. Subsequent work therefore
instruments and removes capacity traffic inside the mixed pressure/topology
bucket rather than tuning the already bounded evaluator.

The pressure command tail is currently capped at 12 iterations only for the
measured launch-bound envelope up to 8192 finest cells. Exact audits observed a
maximum of 7 iterations over all 62 UI generations and 8 over the 500-step
minimal regression. This is an interim small-solve schedule, not the ocean
architecture: larger domains retain the correctness-preserving 128-iteration
fallback until the residual-driven persistent/hierarchical solver below
replaces both fixed tails.

At six hierarchy levels and the 12-iteration recorded cap, MGPCG alone encodes
1,252 dispatches per advance—67.8% of every dispatch in the exact UI trace.
The GPU convergence word prevents arithmetic after convergence but cannot
remove already encoded commands. Fusing restriction with ghost accumulation
and prolongation with ghost propagation removed 130 commands per advance;
prolongation is now one deterministic fine-owned sum/store with no floating
point atomics. The small-domain cooperative replacement is
therefore accepted only if it preserves the Section 4.3 hybrid operator while
collapsing the whole solve to a small fixed kernel sequence. Moving the
remaining 1,252 calls behind indirect early-outs is not a throughput result.

Exact key dimensions now bound face-transfer radix work. The 24x18x16 scene
requires eight nibble passes rather than an unconditional 32; larger domains
derive as many digits as their coordinates require. This removed 4,464 passes,
2,976 indirect dispatches, and 97.5 MB of clears over 62 advances without
assuming a maximum scene size. Stopped MGPCG and SPGrid tails also no longer
rewrite hybrid/correction vectors after device-published convergence.

The current 500-step numerical gate passes with zero validation errors,
maximum pressure relative residual 9.99e-5, maximum exact volume drift 0.3012%,
one connected component, and no non-finite velocities. It also confirms that
energy dissipation remains unresolved: mechanical-energy retention is 93.07%
at 0.2 s, 44.94% at 0.5 s, 40.83% at 1 s, and 39.46% at 2 s. Performance work
must not hide that numerical debt behind a shorter or different scene.

An opt-in fixed-ring GPU energy ledger now measures authoritative boundaries
without adding any default-path allocation, command, readback, or fence. Its
first moving UI-equivalent step measured face kinetic proxy changes of
0.00014777 after gravity, 0.00012349 after projection (-16.4%), and 0.00012206
after face-band publication (a further -1.15%). Fine transport changed the
resident potential proxy by only -0.034% in that step. Topology/redistance
measurements currently change resident support, so their raw delta is recorded
but is not yet labeled dissipation; the next ledger revision must compare a
common immutable sample set.

The target topology is one generic radix/scan/compact spine. Every topology
product is a counted, sorted A/B publication: a small header containing its
live count, generation, validity, and indirect dispatch dimensions followed by
a payload whose only readable region is `[0, count)`. The inactive tail is
stale and is neither cleared nor searched. Construction emits fixed-fanout
candidate records, stable-radix-sorts their exact keys, marks runs, scans run
heads, compacts unique records, and builds adjacency as sorted ranges or CSR.
Morton order is a locality aid; exact cell, level, and relation words remain
part of identity.

The old publication remains immutable while the next one is built. A sorted
old/new merge carries forward unchanged rows and their numerical state, emits
additions, retires removals, and publishes the new header only after validation.
The same warm scratch arena supplies ping-pong keys and values, flags, offsets,
block totals, radix histograms, segment heads, and indirect arguments. Active
workgroups overwrite the scratch prefix they consume, so recurring execution
has no capacity-sized clears, capacity-sized searches, global append counters,
CAS hash insertion, or other recurring atomics.

The first reusable primitive should be a counted radix-set builder; the
canonical sorted `(cell, size) -> row` leaf publication then becomes the spine
for owner and site lookup, face-key generation, fine-brick seeding, support
closure, parent reduction, and warm generation transfer. Downstream consumers
use bounded gathers, binary search, or sorted merge joins against this
publication rather than rebuilding separate hashes. Each transition must be
measured first on the exact 24x18x16 UI Dawn throughput lane, then preserve the
longer 16x16x16 minimal dam-break regression gate.

The primitive has two size classes. A live set that fits one workgroup uses a
fused local histogram/scan/scatter path so a small scene does not pay a long
dispatch floor. Larger sets use explicit hierarchical block scan, block-total
scan, and carry/scatter passes. Merrill and Garland's single-pass decoupled
look-back scan approaches copy bandwidth on CUDA, but it requires relative
progress between workgroups. Portable WGSL supplies barriers only at workgroup
scope, and published progress testing reports that Apple and ARM GPUs do not
support the commonly assumed linear occupancy-bound model. Therefore the
portable WebGPU path must not spin on another workgroup's publication. Its
cross-workgroup dependencies are command-ordered passes; its atomics, if any
exist during migration, cannot be synchronization dependencies.

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
2. `npm run test:webgpu:dam-ui-throughput` improves the single-submission wall
   time on repeated Metal runs. A split phase sample is accepted for attribution
   only when its total is no more than 25% above the ordinary production fence;
   otherwise it is rejected rather than used to claim a phase improvement.
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
- [Single-pass parallel prefix scan with decoupled look-back](https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf)
- [GPU workgroup progress models](https://arxiv.org/abs/2109.06132)
- [WGSL memory and synchronization model](https://gpuweb.github.io/gpuweb/wgsl/#memory-model)
- [GPU radix sorting](https://research.nvidia.com/publication/2009-05_designing-efficient-sorting-algorithms-manycore-gpus)
- [Morton-key hierarchy construction](https://research.nvidia.com/publication/2012-06_maximizing-parallelism-construction-bvhs-octrees-and-k-d-trees)
