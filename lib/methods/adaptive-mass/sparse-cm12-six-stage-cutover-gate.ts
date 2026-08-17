import {
  SPARSE_CM12_FRAME_PLAN_STAGE_COUNT,
  type SparseCM12FramePlanLayout,
} from "../../core/sparse-cm12-frame-plan";

export const SPARSE_CM12_SIX_STAGE_AUTHORITY_EXPECTATIONS = Object.freeze([
  { id: "VEX1", stages: [0], role: "physical-consumer" },
  { id: "FPA1.preparation", stages: [0], role: "physical-consumer" },
  { id: "FPA1.projection", stages: [0], role: "physical-consumer" },
  { id: "SAW1.mass", stages: [1], role: "physical-consumer" },
  { id: "SRR1.mass", stages: [1], role: "result-authority" },
  { id: "SAW1.gamma", stages: [2], role: "physical-consumer" },
  { id: "SRR1.gamma", stages: [2], role: "result-authority" },
  { id: "SAW1.surface", stages: [3], role: "physical-consumer" },
  { id: "SRR1.surface", stages: [3], role: "result-authority" },
  { id: "PCF1", stages: [4], role: "physical-consumer" },
  { id: "PCA1", stages: [4], role: "physical-consumer" },
  { id: "PSA1", stages: [4], role: "execution-authority" },
  { id: "FPP1", stages: [5], role: "physical-consumer" },
  { id: "A4D2", stages: [0, 1, 2, 3, 4, 5], role: "next-plan-producer" },
] as const);

export type SparseCM12SixStageAuthorityId =
  typeof SPARSE_CM12_SIX_STAGE_AUTHORITY_EXPECTATIONS[number]["id"];

export const SPARSE_CM12_SIX_STAGE_REQUIRED_RECEIPT_FIELDS = Object.freeze([
  "directCount", "closureCount", "executedCount", "skippedCount", "unknownCount",
  "uncoveredWriteCount", "expectedCoverageCount", "coveredCoverageCount",
  "producerGeneration", "consumerGeneration", "causeMask", "maximumClosureDepth",
  "faultCount", "firstFaultId",
] as const);

export const SPARSE_CM12_SIX_STAGE_FORBIDDEN_RUNTIME_TOKENS = Object.freeze([
  "acceptedTemplateCellInvocation", "acceptedTemplateRowInvocation",
  "acceptedTemplateCellCount", "acceptedTemplateRowCount",
  "acceptedTemplateCellWorkgroups", "acceptedTemplateRowWorkgroups",
  "acceptedCellInvocation", "acceptedRowInvocation",
  "pressureCellInvocation", "pressureRowInvocation",
  "dispatchAccepted", "dispatchAcceptedCells", "dispatchAcceptedRows",
  "acceptedIndirectArguments", "globalAcceptedCells", "globalAcceptedRows",
  "globalCellInvocation", "globalRowInvocation", "globalAggregateBake",
  "globalHierarchyBake", "bakeBrickAggregateEdges", "bakeBrickAggregateDiagonal",
] as const);

export interface SparseCM12SixStageReceipt {
  readonly acceptedGeneration: number;
  readonly topologyGeneration: number;
  readonly stages: readonly SparseCM12LogicalStageReceipt[];
}

export interface SparseCM12LogicalStageReceipt {
  readonly stage: 0 | 1 | 2 | 3 | 4 | 5;
  readonly authorityIds: readonly SparseCM12SixStageAuthorityId[];
  readonly validTileCount: number;
  readonly directCount: number;
  readonly closureCount: number;
  readonly executedCount: number;
  readonly skippedCount: number;
  readonly unknownCount: number;
  readonly uncoveredWriteCount: number;
  readonly expectedCoverageCount: number;
  readonly coveredCoverageCount: number;
  readonly producerGeneration: number;
  readonly consumerGeneration: number;
  readonly causeMask: number;
  readonly maximumClosureDepth: number;
  readonly faultCount: number;
  readonly firstFaultId: number;
}

const nonnegative = (value: number): boolean => Number.isSafeInteger(value) && value >= 0;
const expectedAuthorities = (stage: number): readonly SparseCM12SixStageAuthorityId[] =>
  SPARSE_CM12_SIX_STAGE_AUTHORITY_EXPECTATIONS.filter((entry) =>
    (entry.stages as readonly number[]).includes(stage)).map((entry) => entry.id);

export function inspectSparseCM12SixStageReceipts(
  receipt: SparseCM12SixStageReceipt | undefined,
): { readonly state: "matched" | "unknown" | "fault"; readonly issues: readonly string[] } {
  if (!receipt) return Object.freeze({ state: "unknown" as const,
    issues: Object.freeze(["six-stage FPL receipt is absent"]) });
  const issues: string[] = [];
  if (!nonnegative(receipt.acceptedGeneration) || !nonnegative(receipt.topologyGeneration)) {
    issues.push("accepted frame/topology generation is invalid");
  }
  if (receipt.stages.length !== SPARSE_CM12_FRAME_PLAN_STAGE_COUNT) {
    issues.push(`expected six logical stages, received ${receipt.stages.length}`);
  }
  const seen = new Set<number>();
  for (const stage of receipt.stages) {
    if (seen.has(stage.stage)) issues.push(`stage ${stage.stage} is duplicated`);
    seen.add(stage.stage);
    for (const field of SPARSE_CM12_SIX_STAGE_REQUIRED_RECEIPT_FIELDS) {
      if (!nonnegative(stage[field])) issues.push(`stage ${stage.stage} ${field} is invalid`);
    }
    const required = expectedAuthorities(stage.stage);
    const observed = new Set(stage.authorityIds);
    for (const authority of required) {
      if (!observed.has(authority)) issues.push(`stage ${stage.stage} omits ${authority}`);
    }
    if (observed.size !== stage.authorityIds.length) {
      issues.push(`stage ${stage.stage} repeats an authority receipt`);
    }
    if (stage.producerGeneration !== receipt.acceptedGeneration
      || stage.consumerGeneration !== receipt.acceptedGeneration) {
      issues.push(`stage ${stage.stage} generation is not accepted`);
    }
    if (stage.directCount + stage.closureCount !== stage.executedCount) {
      issues.push(`stage ${stage.stage} dirty/executed census mismatch`);
    }
    if (stage.executedCount + stage.skippedCount !== stage.validTileCount) {
      issues.push(`stage ${stage.stage} executed/skipped partition is incomplete`);
    }
    if (stage.expectedCoverageCount !== stage.coveredCoverageCount) {
      issues.push(`stage ${stage.stage} producer coverage is incomplete`);
    }
    if (stage.unknownCount !== 0) issues.push(`stage ${stage.stage} has unknown tiles`);
    if (stage.uncoveredWriteCount !== 0) {
      issues.push(`stage ${stage.stage} has uncovered writes`);
    }
    if (stage.faultCount !== 0) {
      issues.push(`stage ${stage.stage} fault ${stage.faultCount}@${stage.firstFaultId}`);
    }
    if (stage.directCount > 0 && stage.maximumClosureDepth !== 0
      && stage.closureCount === 0) {
      issues.push(`stage ${stage.stage} reports closure depth without closure work`);
    }
  }
  for (let stage = 0; stage < SPARSE_CM12_FRAME_PLAN_STAGE_COUNT; stage += 1) {
    if (!seen.has(stage)) issues.push(`stage ${stage} receipt is absent`);
  }
  return Object.freeze({ state: issues.length === 0 ? "matched" as const : "fault" as const,
    issues: Object.freeze(issues) });
}

export function assertSparseCM12SixStageProductionSources(sources: Readonly<Record<
  SparseCM12SixStageAuthorityId, string
>>): void {
  for (const expectation of SPARSE_CM12_SIX_STAGE_AUTHORITY_EXPECTATIONS) {
    const source = sources[expectation.id];
    if (!source || source.trim().length === 0) {
      throw new Error(`${expectation.id} production source slice is absent`);
    }
    for (const token of SPARSE_CM12_SIX_STAGE_FORBIDDEN_RUNTIME_TOKENS) {
      if (source.includes(token)) {
        throw new Error(`${expectation.id} uses forbidden accepted/global token ${token}`);
      }
    }
    if (!/dispatchWorkgroupsIndirect|Invocation\(|dirty|workList|activeLeaf|pageList/.test(source)) {
      throw new Error(`${expectation.id} lacks a local GPU authority`);
    }
  }
}

/** Static guard for the Grid shader's missing/invalid-publication behavior. */
export function assertSparseCM12GridOverlayFailClosedSource(source: string): void {
  const required = [
    "if(!sparseFramePlanTileValid(address)){return vec4f(1.0,0.035,0.63,0.96);}",
    "unknown||(direct&&closure)||((direct||closure)!=processed)",
    "(!(direct||closure)&&!skipped)||(processed&&skipped)",
    "if(!sparseDirtyPublicationComplete()||sparseDirtyStageUnknown(at,stage))",
  ];
  for (const token of required) if (!source.includes(token)) {
    throw new Error(`Grid dirty overlay lost fail-closed clause: ${token}`);
  }
}

export function assertSparseCM12SixStageFramePlanLayout(
  layout: Pick<SparseCM12FramePlanLayout, "packetCount">,
): void {
  if (layout.packetCount !== SPARSE_CM12_FRAME_PLAN_STAGE_COUNT) {
    throw new Error("production FramePlan must publish all six logical stage packets");
  }
}
