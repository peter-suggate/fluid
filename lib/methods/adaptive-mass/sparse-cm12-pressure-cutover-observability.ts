import type { GPUEulerianInfo, GPUAdaptivePressureLocalStageReceipt } from
  "../../core/webgpu-eulerian";

export type SparseCM12PressureCutoverAuthorities = NonNullable<NonNullable<
  GPUEulerianInfo["adaptivePressureTopologyAttribution"]>["authorities"]>;

export const SPARSE_CM12_PRESSURE_CUTOVER_FORBIDDEN_RUNTIME_TOKENS = Object.freeze([
  "acceptedTemplateCellInvocation",
  "acceptedTemplateRowInvocation",
  "acceptedTemplateCellCount",
  "acceptedTemplateRowCount",
  "acceptedTemplateCellWorkgroups",
  "acceptedTemplateRowWorkgroups",
  "dispatchAccepted",
  "acceptedIndirectArguments",
  "bakeBrickAggregateEdges",
  "bakeBrickAggregateDiagonal",
  "globalAggregateBake",
  "globalHierarchyBake",
] as const);

const localStages = (receipt: SparseCM12PressureCutoverAuthorities): readonly [
  string, GPUAdaptivePressureLocalStageReceipt,
][] => [
  ["PCA aggregate/hierarchy", receipt.pca],
];

const validCount = (value: number): boolean => Number.isSafeInteger(value) && value >= 0;
const stageFields: readonly (keyof GPUAdaptivePressureLocalStageReceipt)[] = Object.freeze([
  "acceptedGeneration", "candidateGeneration", "topologyGeneration", "directCount",
  "closureCount", "dirtyCount", "workCount", "executedCount", "skippedCount",
  "expectedProducerReceipts", "coveredProducerReceipts", "causeMask", "fault",
  "firstFaultId",
]);

/**
 * Fail-closed diagnostics validation. A missing or malformed receipt is never
 * interpreted as a quiescent frame, and the pressure input generation must be
 * the topology accepted before this frame's end-frame commit.
 */
export function inspectSparseCM12PressureCutoverAuthorities(
  receipt: SparseCM12PressureCutoverAuthorities | undefined,
  expectedInputTopologyGeneration?: number,
): { readonly complete: boolean; readonly issues: readonly string[] } {
  if (!receipt) return Object.freeze({ complete: false,
    issues: Object.freeze(["PCA receipt is unavailable"]) });
  const issues: string[] = [];
  if (!validCount(receipt.inputTopologyGeneration)) {
    issues.push("authority input topology generation is invalid");
  }
  if (expectedInputTopologyGeneration !== undefined
    && receipt.inputTopologyGeneration !== expectedInputTopologyGeneration) {
    issues.push(`authority input topology generation ${receipt.inputTopologyGeneration}`
      + ` does not match prior-frame pressure input ${expectedInputTopologyGeneration}`);
  }
  for (const [label, stage] of localStages(receipt)) {
    for (const field of stageFields) {
      if (!validCount(stage[field])) issues.push(`${label} ${field} is invalid`);
    }
    if (stage.fault !== 0) issues.push(`${label} fault ${stage.fault} at ${stage.firstFaultId}`);
    if (stage.acceptedGeneration !== stage.candidateGeneration) {
      issues.push(`${label} candidate generation is not accepted`);
    }
    if (stage.topologyGeneration !== receipt.inputTopologyGeneration) {
      issues.push(`${label} topology generation does not match the prior-frame input`);
    }
    if (stage.expectedProducerReceipts !== stage.coveredProducerReceipts) {
      issues.push(`${label} producer coverage is incomplete`);
    }
    if (stage.executedCount > stage.workCount) {
      issues.push(`${label} executed count exceeds local work count`);
    }
  }
  const familyDirty = receipt.pca.familyDirtyCount.reduce((sum, value) => sum + value, 0);
  const familyExecuted = receipt.pca.familyExecutedCount.reduce((sum, value) => sum + value, 0);
  if (familyDirty !== receipt.pca.dirtyCount) issues.push("PCA family dirty census mismatch");
  if (familyExecuted !== receipt.pca.executedCount) {
    issues.push("PCA family execution census mismatch");
  }
  const complete = receipt.status === "matched" && issues.length === 0;
  if (receipt.status === "fault" && issues.length === 0) {
    issues.push("authority status is fault without a decoded fault");
  } else if (receipt.status === "unavailable" && issues.length === 0) {
    issues.push("authority status is unavailable");
  }
  return Object.freeze({ complete, issues: Object.freeze(issues) });
}

export function formatSparseCM12PressureCutoverAuthorities(
  receipt: SparseCM12PressureCutoverAuthorities | undefined,
  expectedInputTopologyGeneration?: number,
): string {
  const inspection = inspectSparseCM12PressureCutoverAuthorities(
    receipt, expectedInputTopologyGeneration,
  );
  if (!receipt) {
    return "PCA receipt: UNAVAILABLE (no accepted GPU authority receipt)";
  }
  const stage = (label: string, value: GPUAdaptivePressureLocalStageReceipt): string =>
    `${label} g${value.acceptedGeneration} · dirty ${value.dirtyCount}`
    + ` · work ${value.workCount} · executed/skipped ${value.executedCount}/${value.skippedCount}`
    + ` · direct/closure ${value.directCount}/${value.closureCount}`
    + ` · ${value.fault === 0 ? "fault 0" : `FAULT ${value.fault}@${value.firstFaultId}`}`;
  const status = inspection.complete ? "Pressure local authorities: MATCHED"
    : `Pressure local authorities: UNAVAILABLE/FAULT — ${inspection.issues.join("; ")}`;
  return [status,
    "Face project: direct compiled dirty/pressure row masks",
    `${stage("PCA", receipt.pca)} · family dirty ${receipt.pca.familyDirtyCount.join("/")}`,
  ].join("\n");
}

/** Static cutover check over isolated production entrypoint source slices. */
export function assertSparseCM12PressureCutoverLocalSources(sources: Readonly<Record<
  "pca", string
>>): void {
  for (const [authority, source] of Object.entries(sources)) {
    if (source.trim().length === 0) throw new Error(`${authority} source slice is empty`);
    assertSparseCM12PressureCutoverNoGlobalTokens(source, authority);
    if (!/dispatchWorkgroupsIndirect|Invocation\(|activeLeaf|dirtyLeaf|workIndirect/.test(source)) {
      throw new Error(`${authority} source lacks a local GPU work authority`);
    }
  }
}

export function assertSparseCM12PressureCutoverNoGlobalTokens(
  source: string,
  label = "pressure cutover",
): void {
  for (const token of SPARSE_CM12_PRESSURE_CUTOVER_FORBIDDEN_RUNTIME_TOKENS) {
    if (source.includes(token)) {
      throw new Error(`${label} production source uses forbidden global token ${token}`);
    }
  }
}
