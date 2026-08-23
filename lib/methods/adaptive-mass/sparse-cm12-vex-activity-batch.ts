/** Thin resident composition for the VEX2 packet masks and output plane. */
import {
  SPARSE_CM12_VELOCITY_EXTENSION_DEPTH,
  createSparseCM12VelocityExtensionInitialWords,
  createSparseCM12VelocityExtensionResidentLayouts,
  type SparseCM12VelocityExtensionLayout,
  type SparseCM12VelocityExtensionStateLayout,
} from "./sparse-cm12-velocity-extension";

export const SPARSE_CM12_VEX_ACTIVITY_BATCH_MAGIC = 0x5641_4232; // VAB2
export const SPARSE_CM12_VEX_ACTIVITY_BATCH_VERSION = 2;

const align64 = (value: number): number => Math.ceil(value / 64) * 64;

export interface SparseCM12VexActivityBatchLayout {
  readonly activityBaseWords: number;
  readonly velocityExtension: SparseCM12VelocityExtensionLayout;
  readonly velocityState: SparseCM12VelocityExtensionStateLayout;
  readonly totalActivityWords: number;
  readonly totalStateFloats: number;
}

export function createSparseCM12VexActivityBatchLayout(options: {
  readonly activityTailWords: number;
  readonly stateTailFloats: number;
  readonly cellCapacity: number;
  readonly packetCapacity?: number;
}): SparseCM12VexActivityBatchLayout {
  const activityBaseWords = align64(options.activityTailWords);
  const velocity = createSparseCM12VelocityExtensionResidentLayouts({
    activityTailWords: activityBaseWords,
    stateTailFloats: options.stateTailFloats,
    cellCapacity: options.cellCapacity,
    packetCapacity: options.packetCapacity,
  });
  return Object.freeze({ activityBaseWords, velocityExtension: velocity.activity,
    velocityState: velocity.state, totalActivityWords: velocity.activity.totalWords,
    totalStateFloats: velocity.state.floatCount });
}

export function createSparseCM12VexActivityBatchInitialWords(
  layout: SparseCM12VexActivityBatchLayout,
): Uint32Array {
  const result = new Uint32Array(layout.totalActivityWords - layout.activityBaseWords);
  result.set(createSparseCM12VelocityExtensionInitialWords(layout.velocityExtension),
    layout.velocityExtension.headerBaseWords - layout.activityBaseWords);
  return result;
}

export type SparseCM12VexActivityDispatch = Readonly<{
  kind: "direct-packet-grid";
  x: "min(packet-capacity,65535)";
  y: "ceil(packet-capacity/x)";
  z: 1;
}>;
export interface SparseCM12VexActivityPipelineDescriptor {
  readonly key: string;
  readonly entryPoint: string;
  readonly dispatch: SparseCM12VexActivityDispatch;
  readonly constants?: Readonly<Record<string, number>>;
}

const direct = (): SparseCM12VexActivityDispatch => Object.freeze({
  kind: "direct-packet-grid", x: "min(packet-capacity,65535)",
  y: "ceil(packet-capacity/x)", z: 1,
});

export function createSparseCM12VexActivityBatchPipelineDescriptors():
readonly SparseCM12VexActivityPipelineDescriptor[] {
  const descriptors: SparseCM12VexActivityPipelineDescriptor[] = [
    { key: "initializeVelocityExtensionPackets",
      entryPoint: "initializeVelocityExtensionPackets", dispatch: direct() },
  ];
  for (let depth = 1; depth <= SPARSE_CM12_VELOCITY_EXTENSION_DEPTH; depth += 1) {
    descriptors.push({ key: `advanceVelocityExtensionPackets${depth}`,
      entryPoint: "advanceVelocityExtensionPackets", dispatch: direct(),
      constants: Object.freeze({ EXTENSION_RECURRENCE_DEPTH: depth }) });
  }
  return Object.freeze(descriptors);
}

export interface SparseCM12VexActivityBatchScheduleStep {
  readonly id: string;
  readonly operation: "pipeline" | "acceptance-seam";
  readonly pipeline?: string;
  readonly after: readonly string[];
}

export function createSparseCM12VexActivityBatchSchedule():
readonly SparseCM12VexActivityBatchScheduleStep[] {
  const result: SparseCM12VexActivityBatchScheduleStep[] = [{
    id: "vex-mask-initialize", operation: "pipeline",
    pipeline: "initializeVelocityExtensionPackets", after: [],
  }];
  let prior = result[0]!.id;
  for (let depth = 1; depth <= SPARSE_CM12_VELOCITY_EXTENSION_DEPTH; depth += 1) {
    const id = `vex-sweep-${depth}`;
    result.push({ id, operation: "pipeline",
      pipeline: `advanceVelocityExtensionPackets${depth}`, after: [prior] });
    prior = id;
  }
  result.push({ id: "face-stage-receipt", operation: "acceptance-seam",
    after: [prior] });
  return Object.freeze(result);
}

export const SPARSE_CM12_VEX_ACTIVITY_BATCH_PRODUCER_HOOKS = Object.freeze({
  vexRoots: Object.freeze([]), provenance: Object.freeze([]),
});
export const SPARSE_CM12_VEX_ACTIVITY_BATCH_QA = Object.freeze([]);
export const SPARSE_CM12_VEX_ACTIVITY_BATCH_GRID_REQUIRED_FIELDS = Object.freeze([
  "validityMaskLow", "validityMaskHigh", "depth", "velocity",
] as const);
export const SPARSE_CM12_VEX_ACTIVITY_BATCH_GRID_RECEIPTS = Object.freeze([]);
