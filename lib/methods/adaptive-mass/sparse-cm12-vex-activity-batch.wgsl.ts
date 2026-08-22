/** Binding-free WGSL composition for the VEX1 production batch. */
import {
  createSparseCM12VelocityExtensionWGSL,
  type SparseCM12VelocityExtensionWGSLOptions,
} from "./sparse-cm12-velocity-extension.wgsl";
import type { SparseCM12VexActivityBatchLayout } from
  "./sparse-cm12-vex-activity-batch";

export const SPARSE_CM12_VEX_ACTIVITY_BATCH_WGSL_HOOK_ABI = Object.freeze({
  provenance: Object.freeze([
    "cm12BatchVelocityExtensionRoot(u32,u32,u32)->bool",
    "cm12BatchVelocityExtensionClosure(u32,u32,u32)->bool",
    "cm12BatchVelocityExtensionScheduled(u32,u32,u32)->bool",
    "cm12BatchVelocityExtensionOwner(u32)->u32",
    "cm12BatchVelocityExtensionFault(u32,u32)",
  ]),
  effectiveVelocity: Object.freeze([
    "cm12BatchPublishVexAcceptedEffectiveVelocity(u32,vec4f)",
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
  readonly injectionReopenReadyExpression?: string;
  readonly provenanceHookPrefix?: string;
  readonly effectiveVelocityHookPrefix?: string;
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
    injectionReopenReadyExpression: options.injectionReopenReadyExpression,
    provenanceHookPrefix: options.provenanceHookPrefix ?? "cm12Batch",
    effectiveVelocityHookPrefix: options.effectiveVelocityHookPrefix,
  };
  return createSparseCM12VelocityExtensionWGSL(velocityOptions);
}
