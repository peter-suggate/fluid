import {
  performanceActivityTaskColor,
  type ActivityFrameIdentity,
  type ActivityClockDescriptor,
  type ActivityClockDomain,
  type ActivityEvent,
  type ActivityEvidence,
  type ActivityLane,
  type ActivityResource,
  type ActivitySpan,
  type ActivityTask,
  type ActivityWorkIdentity,
  type PerformanceActivityFrameAddition,
  type PerformanceActivityEvidenceIngestResult,
} from "./performance-activity";
import type {
  GPULogicalActivityCapture,
  GPULogicalActivityEvent,
  GPULogicalActivityTimeLocator,
} from "./gpu-logical-activity";

export { gpuPhysicsPerformanceActivityFrameId } from "./performance-activity";

export interface GPUMatrixTaskDescriptor {
  id: string;
  label: string;
  color?: string;
  parentId?: string;
  /** Optional semantic pairing; only explicit enter/exit roles create occupied spans. */
  checkpoints?: Readonly<{ enter: number; exit: number }>;
}

export interface GPULogicalActivityMatrixOptions {
  capture: GPULogicalActivityCapture;
  identity: ActivityWorkIdentity;
  lane: Extract<ActivityLane, "gpu-physics" | "gpu-presentation">;
  clockDomain: ActivityClockDomain;
  windowStart_ms: number;
  windowEnd_ms: number;
  locateTime: GPULogicalActivityTimeLocator;
  /** Subgroup rows are used when subgroup identity exists; otherwise one row per workgroup. */
  granularity?: "workgroup" | "subgroup";
  tasks?: Readonly<Record<number, GPUMatrixTaskDescriptor>>;
  clock?: ActivityClockDescriptor;
  maximumRows?: number;
}

export interface GPULogicalActivityMatrixAddition extends PerformanceActivityFrameAddition {
  rowCount: number;
  droppedRowCount: number;
  unknownTimeEventCount: number;
}

export interface PerformanceActivityEvidenceSink {
  ingestEvidence(
    identity: ActivityFrameIdentity,
    addition: PerformanceActivityFrameAddition,
  ): PerformanceActivityEvidenceIngestResult;
}

export interface PublishDecodedGPULogicalActivityOptions extends GPULogicalActivityMatrixOptions {
  sink: PerformanceActivityEvidenceSink;
}

export interface PublishedGPULogicalActivity {
  result: PerformanceActivityEvidenceIngestResult;
  addition: GPULogicalActivityMatrixAddition;
}

const workgroupKey = (event: GPULogicalActivityEvent) => event.workgroupId.join(".");

function rowKey(event: GPULogicalActivityEvent, granularity: "workgroup" | "subgroup") {
  const workgroup = workgroupKey(event);
  return granularity === "subgroup" && event.subgroupId !== undefined
    ? `${workgroup}.sg.${event.subgroupId}`
    : workgroup;
}

function rowLabel(event: GPULogicalActivityEvent, granularity: "workgroup" | "subgroup") {
  const workgroup = `WG ${event.workgroupId.join(",")}`;
  return granularity === "subgroup" && event.subgroupId !== undefined
    ? `${workgroup} · SG ${event.subgroupId}`
    : workgroup;
}

/**
 * Convert raw shader heartbeat readback into the vertical resource × time matrix.
 * Append order is never treated as time: events without an external projection
 * create their logical row but leave every bucket unknown.
 */
export function gpuLogicalActivityMatrixAddition(
  options: GPULogicalActivityMatrixOptions,
): GPULogicalActivityMatrixAddition {
  const granularity = options.granularity ?? "subgroup";
  const maximumRows = Math.max(1, Math.floor(options.maximumRows ?? 512));
  const prefix = options.lane === "gpu-physics" ? "gpu.physics" : "gpu.presentation";
  const rows = new Map<string, { resource: ActivityResource; first: GPULogicalActivityEvent }>();
  const tasks = new Map<string, ActivityTask>();
  const events: ActivityEvent[] = [];
  const spans: ActivitySpan[] = [];
  const openIntervals = new Map<string, {
    event: GPULogicalActivityEvent;
    time_ms: number;
    evidence: Exclude<ActivityEvidence, "idle" | "unknown">;
  }>();
  let droppedRowCount = 0;
  let unknownTimeEventCount = 0;

  for (const heartbeat of [...options.capture.events].sort((left, right) => left.sequence - right.sequence)) {
    const key = rowKey(heartbeat, granularity);
    let row = rows.get(key);
    if (!row) {
      if (rows.size >= maximumRows) {
        droppedRowCount += 1;
        continue;
      }
      const resource: ActivityResource = {
        id: `${prefix}.logical.${key}`,
        label: rowLabel(heartbeat, granularity),
        kind: "gpu-logical-capacity",
        lane: options.lane,
        clockDomain: options.clockDomain,
        capacitySlot: rows.size,
      };
      row = { resource, first: heartbeat };
      rows.set(key, row);
    }
    const descriptor = options.tasks?.[heartbeat.taskId] ?? {
      id: `${prefix}.task.${heartbeat.taskId.toString(16).padStart(8, "0")}`,
      label: `GPU task 0x${heartbeat.taskId.toString(16).padStart(8, "0")}`,
    };
    if (!tasks.has(descriptor.id)) tasks.set(descriptor.id, {
      id: descriptor.id,
      label: descriptor.label,
      color: descriptor.color ?? performanceActivityTaskColor(descriptor.id),
      lane: options.lane,
      ...(descriptor.parentId ? { parentId: descriptor.parentId } : {}),
    });
    const location = options.locateTime(heartbeat);
    if (!location || !Number.isFinite(location.time_ms)) {
      unknownTimeEventCount += 1;
      continue;
    }
    events.push({
      id: `${options.identity.frameId}.${prefix}.heartbeat.${options.capture.captureId}.${heartbeat.sequence}`,
      kind: "instant",
      taskId: descriptor.id,
      resourceId: row.resource.id,
      clockDomain: options.clockDomain,
      at_ms: location.time_ms,
      evidence: location.evidence,
      identity: options.identity,
      metadata: {
        checkpointId: heartbeat.checkpointId,
        workgroup: heartbeat.workgroupId.join(","),
        ...(heartbeat.subgroupId !== undefined ? { subgroupId: heartbeat.subgroupId } : {}),
        ...(heartbeat.logicalLaneCount !== undefined ? { logicalLaneCount: heartbeat.logicalLaneCount } : {}),
        ...(heartbeat.activeLaneCount !== undefined ? { activeLaneCount: heartbeat.activeLaneCount } : {}),
        ...(heartbeat.tick !== undefined ? { logicalTick: heartbeat.tick } : {}),
      },
    });
    const checkpointRole = descriptor.checkpoints?.enter === heartbeat.checkpointId
      ? "enter"
      : descriptor.checkpoints?.exit === heartbeat.checkpointId ? "exit" : undefined;
    const intervalKey = `${row.resource.id}\0${descriptor.id}`;
    if (checkpointRole === "enter") {
      openIntervals.set(intervalKey, {
        event: heartbeat,
        time_ms: location.time_ms,
        evidence: location.evidence,
      });
    } else if (checkpointRole === "exit") {
      const opened = openIntervals.get(intervalKey);
      openIntervals.delete(intervalKey);
      if (opened && location.time_ms > opened.time_ms) {
        const evidence = opened.evidence === "reconstructed" || location.evidence === "reconstructed"
          ? "reconstructed" : "measured";
        spans.push({
          id: `${options.identity.frameId}.${prefix}.interval.${options.capture.captureId}.${opened.event.sequence}.${heartbeat.sequence}`,
          kind: "task",
          taskId: descriptor.id,
          resourceId: row.resource.id,
          clockDomain: options.clockDomain,
          start_ms: opened.time_ms,
          end_ms: location.time_ms,
          evidence,
          identity: options.identity,
          metadata: {
            enterCheckpointId: opened.event.checkpointId,
            exitCheckpointId: heartbeat.checkpointId,
            workgroup: heartbeat.workgroupId.join(","),
            timeProjection: evidence,
          },
        });
      }
    }
  }

  const resources = [...rows.values()].map((row) => row.resource);
  return {
    resources,
    clocks: [options.clock ?? {
      id: options.clockDomain,
      label: options.lane === "gpu-physics" ? "GPU physics timestamp" : "GPU presentation timestamp",
      status: { state: "unsynchronized" },
    }],
    tasks: [...tasks.values()],
    spans,
    events,
    windows: resources.map((resource) => ({
      resourceId: resource.id,
      start_ms: options.windowStart_ms,
      end_ms: options.windowEnd_ms,
    })),
    rowCount: resources.length,
    droppedRowCount,
    unknownTimeEventCount,
  };
}

/**
 * Solver-facing ingestion seam: convert one decoded recorder capture and join
 * it to its retained frame whether readback completes before or after the base
 * frame is published.
 */
export function publishDecodedGPULogicalActivity(
  options: PublishDecodedGPULogicalActivityOptions,
): PublishedGPULogicalActivity {
  const { sink, ...matrixOptions } = options;
  const addition = gpuLogicalActivityMatrixAddition(matrixOptions);
  const identity: ActivityFrameIdentity = {
    frameId: options.identity.frameId,
    generation: options.identity.generation,
  };
  return { result: sink.ingestEvidence(identity, addition), addition };
}
