#!/usr/bin/env node
/** CPU-only fail-closed gate for pressure cutover receipts and attribution. */
import assert from "node:assert/strict";
import type { GPUAdaptivePressureLocalStageReceipt } from
  "../lib/core/webgpu-eulerian";
import { adaptiveMassPressureTopologyChip } from
  "../lib/methods/adaptive-mass/adaptive-mass-frame-pipeline";
import {
  assertSparseCM12PressureCutoverLocalSources,
  formatSparseCM12PressureCutoverAuthorities,
  inspectSparseCM12PressureCutoverAuthorities,
  type SparseCM12PressureCutoverAuthorities,
} from "../lib/methods/adaptive-mass/sparse-cm12-pressure-cutover-observability";
import { sparseCM12PressureTopologyAttribution } from
  "../lib/methods/adaptive-mass/sparse-cm12-pressure-topology-attribution";

const stage = (overrides: Partial<GPUAdaptivePressureLocalStageReceipt> = {}):
GPUAdaptivePressureLocalStageReceipt => Object.freeze({
  acceptedGeneration: 9, candidateGeneration: 9, topologyGeneration: 17,
  directCount: 3, closureCount: 4, dirtyCount: 2, workCount: 7,
  executedCount: 5, skippedCount: 2, expectedProducerReceipts: 4,
  coveredProducerReceipts: 4, causeMask: 3, fault: 0,
  firstFaultId: 0xffff_ffff, ...overrides,
});

const authorities = (overrides: Partial<SparseCM12PressureCutoverAuthorities> = {}):
SparseCM12PressureCutoverAuthorities => Object.freeze({
  status: "matched", inputTopologyGeneration: 17,
  fpa: { projection: stage() },
  pcf: stage(),
  pca: { ...stage({ dirtyCount: 5, executedCount: 5 }),
    familyDirtyCount: [1, 1, 2, 1] as const,
    familyExecutedCount: [1, 1, 2, 1] as const },
  pressureAddressing: { ready: true, phase: 2, fault: 0,
    firstFaultRank: 0xffff_ffff, expectedPCMGeneration: 9,
    materializedPCMGeneration: 9, expectedCount: 20, materializedCount: 20 },
  ...overrides,
});

assert.deepEqual(inspectSparseCM12PressureCutoverAuthorities(undefined).complete, false);
assert.equal(inspectSparseCM12PressureCutoverAuthorities(authorities(), 17).complete, true);
assert.match(formatSparseCM12PressureCutoverAuthorities(undefined), /UNAVAILABLE/);
assert.match(formatSparseCM12PressureCutoverAuthorities(authorities(), 17),
  /Face prepare: brick-owned persistent rows \(no FPA stage\)/);

const wrongInput = authorities({ inputTopologyGeneration: 18 });
assert.equal(inspectSparseCM12PressureCutoverAuthorities(wrongInput, 17).complete, false);
assert.match(inspectSparseCM12PressureCutoverAuthorities(wrongInput, 17).issues.join(" "),
  /does not match prior-frame pressure input/);
const faulted = authorities({ pcf: stage({ fault: 8, firstFaultId: 41 }) });
assert.equal(inspectSparseCM12PressureCutoverAuthorities(faulted, 17).complete, false);
assert.match(formatSparseCM12PressureCutoverAuthorities(faulted, 17), /FAULT 8@41/);

const localSource = "fn localInvocation(id:u32)->u32{return id;} dispatchWorkgroupsIndirect";
assert.doesNotThrow(() => assertSparseCM12PressureCutoverLocalSources({
  fpaProjection: localSource, pcf: localSource,
  pca: localSource,
}));
assert.throws(() => assertSparseCM12PressureCutoverLocalSources({
  fpaProjection: `${localSource} acceptedTemplateRowInvocation`,
  pcf: localSource, pca: localSource,
}), /forbidden global token acceptedTemplateRowInvocation/);
assert.throws(() => assertSparseCM12PressureCutoverLocalSources({
  fpaProjection: localSource, pcf: localSource,
  pca: `${localSource} bakeBrickAggregateEdges`,
}), /forbidden global token bakeBrickAggregateEdges/);

const pcm = { cell: { phase: 1, fault: 0, firstFault: 0xffff_ffff, dirtyCount: 1,
  totalCount: 20, candidateGeneration: 9, acceptedGeneration: 9 },
row: { phase: 1, fault: 0, firstFault: 0xffff_ffff, dirtyCount: 1,
  totalCount: 40, candidateGeneration: 9, acceptedGeneration: 9 } };
const work = { acceptedCellCount: 20, acceptedRowCount: 40,
  temporalScalarCellCount: 2, temporalScalarRowCount: 3,
  pressureCellCount: 18, pressureActiveRowCount: 35, pcm, authorities: authorities() };
const attributed = sparseCM12PressureTopologyAttribution({
  prior: { encodedStep: 9, topologyGeneration: 17, committedBrickCount: 2 },
  current: { encodedStep: 10, topologyGeneration: 18, committedBrickCount: 1 }, work,
});
assert.equal(attributed.authorities?.status, "matched");
assert.match(adaptiveMassPressureTopologyChip({
  adaptivePressureTopologyAttribution: attributed,
}), /Pressure local authorities: MATCHED/);

const wronglyCurrent = sparseCM12PressureTopologyAttribution({
  prior: { encodedStep: 9, topologyGeneration: 17, committedBrickCount: 2 },
  current: { encodedStep: 10, topologyGeneration: 18, committedBrickCount: 1 },
  work: { ...work, authorities: authorities({ inputTopologyGeneration: 18 }) },
});
assert.equal(wronglyCurrent.authorities?.status, "unavailable",
  "current end-frame topology must never be relabelled as pressure input");

console.log("Sparse CM12 pressure cutover observability: PASS");
