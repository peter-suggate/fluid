/** Binding-free WGSL composition for the VEX2 packet transform. */
import {
  createSparseCM12VelocityExtensionWGSL,
  type SparseCM12VelocityExtensionWGSLOptions,
} from "./sparse-cm12-velocity-extension.wgsl";
import type { SparseCM12VexActivityBatchLayout } from
  "./sparse-cm12-vex-activity-batch";

export const SPARSE_CM12_VEX_ACTIVITY_BATCH_WGSL_HOOK_ABI = Object.freeze({
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
    topologyGenerationExpression: options.topologyGenerationExpression,
    sourceFrameGenerationExpression: options.sourceFrameGenerationExpression,
    effectiveVelocityHookPrefix: options.effectiveVelocityHookPrefix,
  };
  return createSparseCM12VelocityExtensionWGSL(velocityOptions);
}
