# PAB1 serial resident integration map

PAB1 is a construction-only comparison of two address functions inside the
same Sparse CM12 pressure kernel source. It is not a solver option, URL state,
GPU-selected mode, or production fallback.

## Frozen A/B contract

- `canonicalRankSelect`: `pabPressureCellAddress(rank)` calls
  `pcmCellRankSelect(rank)`.
- `materializedList`: the same helper performs one load from the PAB1 list.
  A missing, stale, or faulted receipt returns `INVALID`; it never calls rank
  select as recovery.
- Both arms retain the resident's existing post-address validation:
  `cell != INVALID && pcmCellContains(cell) && cellActive(cell)`.
- Kernel bodies, dispatch counts, pressure seed, coefficients, reductions,
  iteration cap/cadence, and output buffers are otherwise identical.
- The list is authored from accepted PCM rank order, not from the historical
  `isLiquid` compactors. This preserves the exact pressure domain and stable-ID
  arithmetic order.

## Resident anchors remaining after the disjoint scaffold

1. In `webgpu-adaptive-mass-solver.ts`, beside the four private QA symbols near
   `PRESSURE_REFRESH_ORACLE_QA_TOKEN`, add two private symbols and two explicitly
   named factories:
   `createPressureAddressingRankSelectForQA` and
   `createPressureAddressingMaterializedListForQA`. Extend only the private
   `createAsync` token union and its construction routing. Do not add a method
   value, normalized value, preset field, or runtime setter.
2. Forward `readPressureAddressingABQA()` beside the existing QA read methods on
   `WebGPUAdaptiveMassSolver`.
3. In `webgpu-sparse-cm12-resident.ts`, beside the QA factories around
   `createPressureRefreshOracleForQA`, add the two corresponding constructors.
   Replace further boolean growth with one private construction-mode value if
   practical; production must remain `undefined`.
4. Immediately after `createSparseCM12VexActivityBatchLayout`, append
   `createSparseCM12PressureAddressingABLayout` only for either PAB1 QA mode,
   with `baseWords = vexActivityBatchLayout.totalActivityWords` and
   `cellCapacity = templates.cellCount`. Size `initialActivity` to the PAB1
   total and upload its initial words after the VEX/A4D2 slice. Production keeps
   the current VEX total exactly.
5. Extend `createWebgpuSparseCM12ResidentWGSL` after its current
   `vexActivityBatchLayout` argument with an optional PAB1 layout. Generate
   `createSparseCM12PressureAddressingABWGSL` into the existing atomic
   `activity` arena. When absent, provide only
   `fn pabPressureCellAddress(rank:u32)->u32{return pcmCellRankSelect(rank);}`.
6. At `pressureCellInvocation`, replace only the address expression with
   `pabPressureCellAddress(invocation)`. Keep its `pcmCellContains` and
   `cellActive` predicates byte-identical.
7. Compile QA pressure-cell pipelines through `GPUCompilationManager` with the
   construction-fixed constant from
   `sparseCM12PressureAddressingABPipelineConstants`. The two solver instances
   compile from the same module/kernel entry points. No pipeline may be chosen
   from frame state.
8. Immediately after `finalizeCanonicalPressureCells`, encode the list arm's
   QA-only sequence: begin; storage-to-indirect copy of PAB1 words 21..23;
   materialize; verify; finalize. Use a dedicated 12-byte indirect snapshot.
   The rank-select arm does not schedule this sequence. A PAB1 fault zeros its
   indirect and gates the list solve; it never schedules the other arm.
9. Add QA-only hardware intervals for raw materialization and verification.
   Expose those values plus the parsed PAB1 receipt from
   `readPressureAddressingABQA`. Do not merge verification into materialization
   or pressure-solve time.
10. Destroy the QA-only PAB1 indirect/timestamp resources with the resident.
    Production allocation and recurring schedule remain unchanged.

## Required serial gate

```sh
node --import tsx tools/check-sparse-cm12-pressure-addressing-ab.ts
npm exec tsc -- --noEmit
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
FLUID_WEBGPU_BACKEND=metal \
node --import tsx tools/probe-sparse-cm12-pressure-addressing-ab.ts \
  --scene=ocean-seiche --brick-fine=16 --presentation-page=16 \
  --warmup=8 --frames=24 --capture-gap-ms=110 \
  --require-bit-exact=1 --enforce-pressure-receipts=1 \
  --out=artifacts/sparse-cm12-pressure-addressing-ab-ocean24.json
```

The probe runs the two ocean arms sequentially to bound memory, compares every
measured physical frame bit-for-bit, requires hardware timestamps, and reports
solve-only and solve-plus-materialization deltas. Before the resident hooks are
installed it fails before acquiring WebGPU and names both missing factories.
