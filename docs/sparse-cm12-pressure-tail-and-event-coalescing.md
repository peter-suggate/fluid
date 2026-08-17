# PSA tail authority and pressure producer coalescing

This is the serial integration contract for `PTL1/v1` and `PCE1/v1`. Both are
GPU-only production paths. Full paths exist only as immutable construction QA
specializations and are never selectable after construction.

## PTL1 convergence tail

PSA1 already owns two storage-resident banks of four indirect triplets: PCM
cells, wet bricks, active hierarchy nodes, and scalar singleton work. PTL1
makes their synchronization explicit:

1. Encode `publishSparseCM12PressureTailA` or `...B` at each existing
   convergence-gate seam. The publisher writes all four triplets, including
   zero triplets when PSA is faulted or the exact arithmetic-active bit is
   false, then publishes `tailPublishedGeneration`.
2. Close the compute pass. Never bind the PSA storage arena as an indirect
   source in that pass.
3. Copy exactly 48 bytes from that bank into its own 48-byte
   `COPY_DST|INDIRECT` buffer.
4. Dispatch eligible guarded arithmetic through that buffer. Journal,
   snapshot, final residual, unconditional clears, gate reductions, and fault
   receipts remain direct.
5. Run the unchanged gate reduction, then repeat with the opposite bank.

Ordinary and curvature-recovery seams have distinct exact GPU predicates.
The ordinary publisher uses `pipelinedPressureActive()`. The recovery publisher
uses the same condition as `applyPipelinedRecovery`: pressure enabled and the
curvature-loss flag set. A false predicate publishes four zero triplets; a true
predicate publishes the exact PSA cell/wet-brick/hierarchy/scalar tuple. The
telemetry acceptance helper rejects any other tuple, including a wide recovery
dispatch after the curvature flag is clear. Predicate selection is fixed by
the encoded seam and is never a runtime host decision.

The command stream and A/B alternation are construction-fixed. No CPU count,
readback, convergence boolean, or host-maintained parity selects work. A PSA
fault produces four zero triplets; a missing publisher/copy is rejected by the
static manifest checker rather than consuming a stale bank.

Construction QA runs the fully encoded guarded kernels and PTL1 through the
same PCM rank order. It gates on pressure/vector/scalar output bytes and on
every published generation/triplet. Topology and worklist layout hashes are
diagnostic only.

Integration uses
`WebGPUSparseCM12PressureTailAuthority`, calls `encodeCopy` after the publisher
pass, and dispatches each family through `dispatch`. The telemetry readback
must show `tailPublishedGeneration == acceptedGeneration` for every successful
frame.

## PCE1 producer coalescing

The ocean receipt shows the remaining problem precisely:

- FPA preparation: about 1.43 million rows executed;
- FPA projection: about 814 thousand rows executed;
- PCF: about 1.55 million direct producer receipts, despite only 1,701 repaired
  leaves;
- PCA: only 11 dirty leaves and roughly 168--178 executions.

PCE1 removes repeated producer atomics before changing numerical work. It has
five stable packet domains: post-scalar 4³ cell tiles, pressure 4³ cell tiles,
theta row leaves, PCM membership 4³ cell tiles, and topology bricks. Each
packet stores generation, ORed cause, maximum closure depth, receipt state,
and provenance. Capacity is fixed from HTP1/PCM construction counts.
The tile key is exactly CMD1/SCA1's dense
`logicalBrick * 64 + local4Tile`; it is not `stableCellId / 64`. A coarse cell
publishes every 4³ tile intersecting its exact HTP1 bounds. This keeps the
blast spatially bounded without assuming a particular current rung.

Final writers publish a packet only when their exact output bits differ from
accepted authority. Consumers expand packets through immutable HTP1 CSR,
deduplicate stable row/cell leaves, and rank-select retained IDs in ascending
order. This is exact under nonzero solenoidal motion: it observes bit changes,
not velocity magnitude, scene class, or an approximate hash.

The first cut is deliberately split:

1. **PCF producer cut.** `pcfStoreThetaAndRecord` performs its exact exchange
   first and publishes a row-leaf packet only when theta bits change. PCM and
   PTR publish membership/topology packets. One packet receipt replaces every
   unchanged classify-row receipt. PCF cell/edge/diagonal arithmetic and order
   do not change.
2. **FPA preparation cut.** Post-scalar final writers publish changed 4³ tiles.
   A tile expands to its immutable incidence/one-ring row leaves. Remove the
   current activity-brick walk as an authority; activity remains telemetry.
3. **FPA projection cut.** Final pressure writes publish changed pressure
   tiles. Expansion marks incidence row leaves once. This removes duplicate
   atomics, but rows whose projected output genuinely changes still execute.
   A later fusion may reduce dispatch overhead; it cannot omit changed rows.

Every packet generation has expected/covered receipts and a first fault
domain/id. Invalid generation, packet capacity, HTP provenance, or incomplete
expansion publishes zero consumer indirects. There is no dirty-ratio switch or
runtime global rebuild.

## Serial resident touchpoints

After explicit ownership handoff:

1. In `webgpu-sparse-cm12-resident.ts`, allocate
   `WebGPUSparseCM12PressureTailAuthority` beside PSA1. Keep the PSA storage
   arena as its source; the adapter allocates two dedicated 48-byte
   `COPY_DST|INDIRECT` buffers. Destroy both with the resident.
2. Replace each eligible cell/brick/hierarchy/scalar convergence-tail dispatch
   by the construction-fixed seam sequence: publisher A/B, close pass, copy
   that bank, reopen pass, then `tail.dispatch`. Do not move journal, snapshot,
   final-residual, convergence reduction, or unconditional clear kernels.
   Use ordinary publishers after initialize/ordinary iteration gates and
   recovery publishers only between `reduceCurvatureRecovery` and
   `reducePipelinedRecovery`. The recovery predicate must be
   `scalars[5] > 0.5 && scalars[14] > 0.5`, matching the numerical kernel.
3. Allocate PCE1 after the current FPA arena tail and initialize its fixed
   packet headers. Add a copy-isolated indirect snapshot for each expansion
   consumer; no PCE1 storage buffer is also an INDIRECT source.
4. In `webgpu-sparse-cm12-resident.wgsl.ts`, call
   `pcePublishFacePreparationCellTile` only after the post-sharpen destination
   bits become final and differ from the accepted face-input authority.
5. Call `pcePublishFaceProjectionPressureTile` only after the final pressure
   bits differ, `pcePublishPCFThetaRowLeaf` only when the exact theta bits
   change, `pcePublishPCFMembershipCellTile` only on a PCM transition, and
   `pcePublishTopologyBrick` once per PTR changed brick.
6. Expand packets through immutable HTP1 incidence in
   `expandSparseCM12PressureProducerPackets`, sort/deduplicate with the existing
   stable-ID bitsets/count trees, then publish coverage and copied indirects in
   `finalizeSparseCM12PressureProducerPackets`.
7. Encode begin, fixed producers, finalize, repair, copy, and consumers inside
   their existing named stages so packet construction remains timestamped.
8. Add construction-only direct-producer arms and compare expanded stable-ID
   sets plus active coefficient, face, and physical bytes. The production
   specialization contains no full producer arm.

### Required source-token removals

The live integration checker rejects these old authority/scheduling tokens:

- `dispatchActivity("markSparseCM12FacePreparationFromActivity")`;
- `dispatchPressureCell("markSparseCM12FaceProjectionFromPressure")`;
- direct `dispatchPressureCell` calls for `updatePipelinedState`,
  `applyPipelinedImage`, and `applyPipelinedRecovery`.

The WGSL entry points may remain as construction-oracle-only specializations,
but they must not be reachable from the production command stream. Remaining
brick/hierarchy tail calls are migrated at the same serial seam; their names
are intentionally not banned because the bootstrap and unconditional recovery
setup use some of the same kernels outside the convergence tail.

Static gates:

```text
npx tsx tools/check-sparse-cm12-pressure-tail-authority.ts
npx tsx tools/check-sparse-cm12-pressure-tail-event-integration.ts
```

The integration checker is deliberately red before resident integration. It is the
serial handoff completion gate, not a standalone-module gate.

Scene gates after integration are paired dam5 exactness, dynamic changed-brick
timing, then ocean8 selection/timing. Ocean24 remains reserved for the combined
optimized milestone.
