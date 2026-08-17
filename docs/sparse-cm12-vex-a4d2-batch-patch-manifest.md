# Sparse CM12 VEX1 + A4D2 resident batch

This is the mechanical production cutover for exact eight-edge velocity extension and
deterministic tile activity reduction. It appends to the resident's final accepted
activity and state tails; it does not reconstruct or renumber FCA/FPL/FPP, pressure,
or scalar-authority allocations.

## Frozen ABI

- Construct `createSparseCM12VexActivityBatchLayout` with the final
  `activityTailWords` and `stateTailFloats` after every previously accepted arena.
- Upload `createSparseCM12VexActivityBatchInitialWords(layout)` at
  `layout.activityBaseWords * 4`. The activity buffer must include `COPY_SRC`; no
  runtime header is read by the CPU.
- Extend the state buffer to `layout.totalStateFloats * 4`. VEX1 owns one persistent
  `vec4f` per cell beginning at `velocityState.acceptedVelocityFloatBase`. This range
  must be after the pressure journal and every candidate/transient range.
- Allocate one 60-byte `INDIRECT | COPY_DST` buffer. Its five packets are fixed:
  A4D2 classify `[0,3)`, scan `[3,6)`, rebuild `[6,9)`, census `[9,12)`, and serial
  VEX `[12,15)`. The VEX packet is reused only after the prior indirect dispatch.
- B16/P16 is the optimized profile (64 lane-owned 4³ tiles per brick). B4 and B8 use
  the same ABI with one and eight valid lanes respectively.

The batch files are:

- `sparse-cm12-vex-activity-batch.ts`: arena composition, initialization, indirect
  copy descriptors, manager pipeline descriptors, encoder schedule, roots, receipts,
  and QA contract.
- `sparse-cm12-vex-activity-batch.wgsl.ts`: binding-free A4D2/VEX1 shader composition
  and the exact resident hook ABI.
- `tools/check-sparse-cm12-vex-activity-batch.ts`: overlap, parity, packet, lifecycle,
  receipt, root, and construction-only-oracle gate.

## Resident patch map

Apply only after the pressure batch releases `webgpu-sparse-cm12-resident.ts` and
`webgpu-sparse-cm12-resident.wgsl.ts`.

1. In the resident TypeScript imports, import the batch layout, initializer, indirect
   copy list, pipeline descriptors, schedule, and combined WGSL generator. Do not
   change the existing pressure/scalar generator arguments.
2. Immediately after the last current activity allocation, create the batch with
   that allocation's `totalWords`. Immediately after the last persistent state
   allocation, pass its `floatCount`. Replace only the final activity/state allocation
   sizes with the batch totals; preserve every preceding base.
3. Append the returned initialization slice to `initialActivity`. The slice origin is
   `activityBaseWords`, so no accepted header or historical record is overwritten.
4. Create the dedicated 60-byte indirect buffer. Encode the adapter's 12-byte copies
   with `copyBufferToBuffer(activity, sourceWord*4, batchIndirect,
   destinationWord*4, 12)`. Never use `mapAsync`, a CPU count, or a CPU parity to
   choose a dispatch.
5. Append `createSparseCM12VexActivityBatchWGSL(...)` to the resident shader after all
   resident hook definitions. Use `arenaName: "activity"`, `stateName: "state"`, FCA's
   accepted/candidate generations, FCA's ready predicate, and provenance prefix
   `cm12Batch`.
6. Acquire every descriptor returned by
   `createSparseCM12VexActivityBatchPipelineDescriptors()` through the existing
   `GPUCompilationManager`. Do not call either WebGPU pipeline constructor directly.
   Frontier depths 1–8 and recurrence depths 1–8 are distinct constant-specialized
   pipelines.
7. Encode `createSparseCM12VexActivityBatchSchedule()` at two existing timing seams:
   VEX steps replace the eight global velocity-extension sweeps; A4D2 steps replace
   full `measureBrickActivity`. Preserve all intervening physics stage order.

## Producer hooks

Call `cm12ExtensionRecordRoot` (through the resident bridge) in the existing producer
that has final authority for each mutation. The call is fused with the producer; no
cell-domain discovery scan is allowed.

| Root | Required coverage |
| --- | --- |
| Construction bootstrap | Every accepted cell, construction-only |
| Final projected/collocated velocity | Bit-changed liquid cell incident to accepted air |
| Density/liquid classification | Changed cell and incident endpoints |
| Topology/edge ownership | Entering active cell, incident active endpoints, and local retired identity invalidation |
| Injection | Every written cell and incident endpoints |
| Moving solid/open fraction | Affected cell and both row endpoints |

`cm12BatchVelocityExtensionRoot/Closure/Scheduled` map the physical cell owner to FPL
stage 0 direct, closure-depth, and executed coverage. Owner lookup must be generation
checked. `cm12BatchVelocityExtensionFault` increments the local FPL fault and uncovered
write receipt; any missing owner, generation, or tile coverage prevents acceptance.

A4D2's candidate list comes from producer-owned dirty brick roots. Each B16 brick is
one workgroup and each of 64 lanes owns one 4³ summary. Exact tile re-evaluation writes
private lane records, deterministic prefix/scatter builds the brick list, and only the
dirty brick is re-reduced into the persistent census. Heavy consumers do not append to
a journal and do not perform global activity atomics.

## Grid/FPL truth contract

Binding `framePlan` is allowed only when all six logical stages share one accepted
generation and each valid tile has executed or skipped evidence:

| Stage | Authority | Required receipt |
| --- | --- | --- |
| 0 face/VEX | VEX1 | direct roots, closure depth, scheduled/executed or skipped |
| 1 mass | SAW1 | direct/closure and executed/skipped |
| 2 gamma | SAW1 | direct/closure and executed/skipped |
| 3 surface | SAW1 | direct/closure and executed/skipped |
| 4 pressure coefficients | PCF1 | direct/closure and executed/skipped |
| 5 presentation | FPP1 | dirty page publication or skipped |

A4D2 publishes future FPL activity roots during exact tile rebuild; it does not claim
that a physics stage executed. Absent receipts, generation mismatch, uncovered writes,
or local faults render the corresponding Grid selector tile magenta. Packet/packing
faults remain red. No stage may infer reuse from a zero-initialized record.

## QA oracle and acceptance

The legacy eight-sweep extension and full activity measurement are immutable
construction options for separate QA solvers. They are not runtime-selectable, cannot
publish the production output, and are never a fallback after a local fault.

The batch gate is:

1. TypeScript, lint, standalone A4D2 Naga, integrated resident WGSL, water shaders,
   and production build pass.
2. Five-step dam: density, velocity, pressure, divergence, gamma/internal transport,
   and accepted presentation payload are byte-exact against the construction oracle;
   VEX/A4D2/FPL fault and omission counts are zero. Topology/workset hashes remain
   diagnostic when physical hashes match.
3. Ocean B16/P16: record a 24-sample artifact; VEX1 and A4D2 each have p95 below
   0.5 ms, clean indirect packets dispatch zero work, and Grid receipts are accepted.
4. The combined production milestone, not this individual batch, runs the canonical
   60-step dam paired and weakened-symmetry equivalence gates.

