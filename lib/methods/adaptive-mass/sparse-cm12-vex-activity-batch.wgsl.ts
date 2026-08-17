/** Binding-free WGSL composition for the A4D2 + VEX1 production batch. */
import { createSparseCM12ProductionActivityWGSL } from
  "./sparse-cm12-production-activity.wgsl";
import {
  createSparseCM12VelocityExtensionWGSL,
  type SparseCM12VelocityExtensionWGSLOptions,
} from "./sparse-cm12-velocity-extension.wgsl";
import type { SparseCM12VexActivityBatchLayout } from
  "./sparse-cm12-vex-activity-batch";

export const SPARSE_CM12_VEX_ACTIVITY_BATCH_WGSL_HOOK_ABI = Object.freeze({
  activity: Object.freeze([
    "cm12ActivityCandidateGeneration()->u32",
    "cm12ActivityCandidateBrickCount()->u32",
    "cm12ActivityCandidateListGeneration()->u32",
    "cm12ActivityCandidateBrickInvocation(u32)->u32",
    "cm12ActivityTopologyGeneration()->u32",
    "cm12ActivityFramePlanGeneration()->u32",
    "cm12ActivityBrickTopologySignature(u32)->u32",
    "cm12ActivityBrickTopologyChanged(u32)->bool",
    "cm12ActivityBuildTileTrigger(u32,u32)->vec4u",
    "cm12ActivityRebuildExactTile(u32,u32)->A4D2TileSummary",
    "cm12ActivityExpectedTileContributionCount(u32,u32)->u32",
    "cm12ActivityExpectedTileCheck(u32,u32)->u32",
    "cm12ActivityPublishFramePlanRoot(u32,u32,u32,u32,u32,u32,u32)->bool",
    "cm12ActivityPublishExactBrick(u32,vec4i,vec4f,vec4u,vec4u)->vec4u",
  ]),
  provenance: Object.freeze([
    "cm12BatchVelocityExtensionRoot(u32,u32,u32)->bool",
    "cm12BatchVelocityExtensionClosure(u32,u32,u32)->bool",
    "cm12BatchVelocityExtensionScheduled(u32,u32,u32)->bool",
    "cm12BatchVelocityExtensionOwner(u32)->u32",
    "cm12BatchVelocityExtensionFault(u32,u32)",
  ]),
} as const);

export interface SparseCM12VexActivityBatchWGSLOptions {
  readonly layout: SparseCM12VexActivityBatchLayout;
  readonly arenaName?: string;
  readonly stateName?: string;
  readonly topologyGenerationExpression?: string;
  readonly sourceFrameGenerationExpression?: string;
  readonly nextFrameGenerationExpression?: string;
  readonly frameAuthorityReadyExpression?: string;
  readonly provenanceHookPrefix?: string;
}

/** Append after resident helper declarations and before shader compilation. */
export function createSparseCM12VexActivityBatchWGSL(
  options: SparseCM12VexActivityBatchWGSLOptions,
): string {
  const velocityOptions: SparseCM12VelocityExtensionWGSLOptions = {
    layout: options.layout.velocityExtension,
    arenaName: options.arenaName,
    stateName: options.stateName,
    acceptedVelocityFloatBase: options.layout.velocityState.acceptedVelocityFloatBase,
    topologyGenerationExpression: options.topologyGenerationExpression,
    sourceFrameGenerationExpression: options.sourceFrameGenerationExpression,
    nextFrameGenerationExpression: options.nextFrameGenerationExpression,
    frameAuthorityReadyExpression: options.frameAuthorityReadyExpression,
    provenanceHookPrefix: options.provenanceHookPrefix ?? "cm12Batch",
  };
  return `${createSparseCM12ProductionActivityWGSL(options.layout.productionActivity)}\n`
    + createSparseCM12VelocityExtensionWGSL(velocityOptions);
}
