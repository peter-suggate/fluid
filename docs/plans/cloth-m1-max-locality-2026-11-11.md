# Cloth GPU locality and compiled-data optimization

Date: 2026-11-11

Status: proposed; no optimization implemented or speedup established by this plan.

Target: `bridson-2002-rotating-sphere` on the Apple M1 Max, through the existing WebGPU implementation.

## Objective and constraints

Accelerate the existing simulation and collision algorithm by improving data layout, locality, compilation of topology, and GPU execution. Pay for indirection when preparing a workload, then stream through compiled records in hot loops.

Preserve the current integrator, Newton/PCG method, preconditioner mathematics, collision predicates, DAT/OGC behavior, friction, timestep, tolerances, and termination rules. Do not introduce multilevel solves, adaptive simulation resolution, a replacement broad phase, weaker collision tests, or candidate reuse across detection rounds as part of this work.

Reordering arithmetic may change floating-point rounding. Require numerical equivalence against the baseline and verify trajectories and collision safety; do not assume bitwise equality or accept materially different convergence as a performance improvement.

Keep authored geometry and stable IDs intact. Internal permutations and compiled topology must be rebuilt or invalidated correctly on edits, load, and undo. Preserve the repository geometry/editing contract.

## Baseline and evidence

The earlier profile used 22,500 vertices, 44,402 triangles, and 66,901 edges. At the later sampled state, the uninterrupted GPU step took approximately 827 ms. The separately instrumented breakdown was approximately:

| Work | Share | Volume per step |
| --- | ---: | --- |
| Collision queries | 56% | 753 million scanned records; about 30 detection passes |
| PCG | 38% | 1,577 iterations across eight Newton iterations |
| Other stages | 6% | Preparation, assembly, projection, trials, etc. |

Instrumented stage timings are not directly additive to uninterrupted timings. Counts include repeated work, not unique contacts. The estimated 15 GB of logical matrix reads per step is not a measurement of physical DRAM traffic.

Profile artifacts: `/Users/petersuggate/code/me/fluid-scratch/sphere-profile-2026-11-11/`, including `simulation.json`, `workload.json`, and `summary.csv`. Re-establish the baseline on the implementation commit before drawing comparisons; code may have changed since the original capture.

Existing optimizations to retain and build on:

- Morton-ordered PCG patches and sliced-ELL matrix storage.
- Compiled local stencil columns and workgroup staging of rows and halo values.
- Grid entries already carry bounds, avoiding reconstruction for every rejection.
- Gather-based contact matrix products avoid scatter atomics.

The current scene has 88 PCG workgroups of 256 threads. Timing alone does not prove cache misses, bandwidth saturation, or occupancy limits. Obtain hardware-counter evidence where the WebGPU/Metal tooling permits it; label inferred bottlenecks otherwise.

## Experiment 1: compile contact linearizations

Primary location: `lib/cloth/contact/ogc-wgsl.ts`, `ogcVertexContactHvLanes` and `ogcEntryLin`; integration in `lib/cloth/gpu-newton-wgsl.ts`.

Currently, each contact matrix-vector product reconstructs IDs, weights, and normal/friction matrices through `ogcEntryLin`. These coefficients should remain fixed within a given linear solve while the search vector changes.

1. Audit every input to the linearization, including lagged friction, projection, solid transforms, and contact generation. Establish its exact lifetime and invalidation dependencies.
2. Build coefficients once at the same Newton iterate used by the existing matrix product.
3. Store pretranslated partner IDs and compact weights/matrix factors. Compare factorized records with expanded blocks; do not blindly materialize a large dense representation.
4. Retain the same per-row contribution order initially to isolate layout and preparation effects.
5. Refresh whenever a dependency changes, including retries/redetection when applicable.

Expected mechanism: replace repeated geometry interpretation with coefficient loads and vector multiplication, amortized over up to 200 PCG iterations per Newton solve.

Measure preparation cost, record bytes, product time per iteration, full solve time, and full step time. Verify the compiled operator on identical vectors against the original operator before trajectory comparisons.

## Experiment 2: compile collision topology and round geometry

Primary location: `lib/cloth/contact/ogc-detector-wgsl.ts`, `edgeEdge`, `hullEdge`, and associated topology packing.

Current dependent chain:

```text
grid entry -> face -> owned edge IDs -> endpoints -> positions
```

Compile a face record with owned edge IDs, endpoint IDs, ownership masks, and required immutable metadata. Prepare compact per-primitive geometry records once per detection round for positions, predictions, and envelopes that queries repeatedly gather.

Keep the same grid, traversal coverage, canonical pair ordering, duplicate-ownership rule, and exact geometric tests. Preserve both directed edge visits where they supply per-edge minima or contact entries; do not remove them merely because DAT pairs have canonical ownership.

Compare static-topology compilation alone with additional dynamic geometry packing. Avoid duplicating large geometry records in every grid cell entry: primitives can occupy multiple cells.

Also test materializing final bucket offsets once. `bucketStart` currently sums three prefix-table entries per lookup; a final contiguous offset table could reduce repeated loads at modest preparation cost.

Measure query time by role, preparation cost, scanned/candidate/pair counts, extra memory, and end-to-end time. Counts and coverage must agree with the baseline.

## Experiment 3: store PCG state in compiled patch order

Primary locations: `lib/cloth/gpu-solver.ts`, `compileClothPCGPatches`; `lib/cloth/gpu-newton-wgsl.ts`, `pcgStep`, `pcgUGlobal`, and `pcgDir`.

Matrix rows are patch-ordered, but solver vectors and preconditioners are addressed through original vertex IDs. Move internal residuals, directions, iterates where appropriate, and preconditioner records into physical patch-slot order.

- Translate static stencil/halo addresses at topology compilation.
- Translate dynamic contact partner IDs during contact preparation.
- Keep stable document IDs and explicit mappings at geometry/publication boundaries.
- Include conversion costs in measurements; avoid adding gathers to another equally hot stage.

Expected mechanism: adjacent lanes access adjacent state records and eliminate repeated row-to-vertex translation in PCG.

Verify pins, contact projection, halo reads, partially filled patches, disconnected components, edits, load, and undo.

## Experiment 4: transpose compiled contact records

Current per-vertex contact addressing is effectively:

```text
base + vertex * capacity + contactIndex
```

Adjacent vertex threads therefore access widely separated records at the same loop iteration. Test patch-local sliced layouts:

```text
patchBase + contactIndex * patchWidth + lane
```

Use bounded padding or compact groups based on contact counts. Do not pad all vertices to the global maximum. Test a transposed compiled linearization buffer before changing the detector's append representation; a compact preparation pass can leave collision generation untouched.

Measure padding, packing time, coefficient traffic, contact-loop divergence, and total PCG time. Retain contribution ordering where practical.

## Experiment 5: spatial query ordering and cooperative bucket tiles

First dispatch through a spatially coherent permutation of query IDs while retaining stable semantic IDs. Compare a cheap static ordering with any dynamic ordering including its preparation cost.

Then investigate groups of queries sharing a cell cooperatively loading bucket records into workgroup memory. Reuse those records across queries while preserving all cell visits and the existing canonical ownership tests.

This is an execution-scheduling experiment, not a new collision algorithm. It is higher risk than record compilation because multi-cell queries, variable bucket lengths, and barriers complicate scheduling.

Measure total traversal time, staging overhead, occupancy, and redundant reads. Compare direct cached reads against explicit staging; workgroup memory is not automatically faster. Reject a variant that saves loads but loses more to synchronization or reduced occupancy.

## Experiment 6: patch size and shader specialization

Benchmark 64-, 128-, and 256-thread PCG patches after adapting all layout and reduction assumptions. Smaller patches expose more workgroups but duplicate halos and increase partial-reduction work.

Generate bounded variants for common stencil sizes and contact types. Keep a general path for uncommon topology. Compare ordinary loops, selective unrolling, and compile-time constants. Avoid a separate enormous shader per mesh or excessive register growth.

Record:

- Time per actual PCG iteration and total step time.
- Workgroups, halo duplication, staged bytes, and global fallback loads.
- Register/threadgroup-memory pressure and occupancy where observable.
- Pipeline compilation time and variant count.

Use capabilities actually exposed by the M1 Max WebGPU adapter. Do not assume newer Apple GPU features or native Metal features are available through WGSL.

## Measurement and correctness protocol

1. Record commit, adapter/backend, solver settings, scene parameters, and active implementation flags.
2. Reuse the saved states near ticks 2, 35, and 120, plus a continuous trajectory run. Verify state caches match the implementation/model before reuse.
3. Use the repository GPU timing harness to avoid overlapping GPU benchmarks. Warm pipelines before recording steady-state timings; report cold compilation separately.
4. Change one representation at a time. Retain an A/B path during evaluation.
5. Report preparation plus execution time, not just a faster inner kernel. Include medians, spread, memory footprint, iteration counts, detections, candidates, and contacts.
6. Compare identical-input operator outputs and collision results before full trajectories. Check penetration, residuals, strain, frictional behavior, pins, and failure/overflow flags. Investigate changes in iteration counts instead of crediting them automatically as a kernel speedup.
7. Run targeted regressions for the changed layout, including edits/load/undo when mappings change. Run `npm run check:types` and `npm run test:unit`; identify unrelated baseline failures explicitly.
8. Accept an optimization only when its end-to-end gain is repeatable, correctness is retained, and memory/preparation costs are documented. Keep failed experiments as measured findings rather than layering more changes on an unverified premise.

## Delivery order and completion criteria

Implement experiments 1 and 2 first, then physical PCG ordering, transposed contact records, query scheduling, and tuning. Re-profile after each accepted change; priorities may shift.

Completion means a measured comparison on the rotating-sphere scene, preserved numerical/contact behavior, mapping/edit regressions where applicable, and a documented explanation of which loads or repeated computations were removed. No numerical speedup target is asserted before measurement.

## Hardware reference

[Apple: Finding your Metal app's GPU occupancy](https://developer.apple.com/documentation/xcode/finding-your-metal-apps-gpu-occupancy) describes register/threadgroup-memory constraints and cache contention. Use hardware observations to guide staging and group size rather than treating maximum occupancy or maximum shared-memory use as objectives by themselves.
