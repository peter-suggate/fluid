"use client";

import { useMemo, useState, type CSSProperties } from "react";
import type {
  ActivityEvidence,
  ActivityEvent,
  ActivityLane,
  ActivityRow,
  PerformanceActivityCaptureDiagnostics,
  PerformanceActivityCaptureReason,
  PerformanceActivityFrame,
} from "@/lib/performance-activity";
import {
  performanceTraceAccounting,
  type PaperPhaseId,
  type PerformanceTrace,
} from "@/lib/performance-trace";

export type PerformanceActivityEvidence =
  | "measured-progress"
  | "reconstructed"
  | "idle"
  | "unknown";

export interface PerformanceActivityTask {
  id: string;
  label: string;
  stageId?: string;
  start_ms: number;
  end_ms: number;
  evidence: PerformanceActivityEvidence;
  detail?: string;
  color?: string;
}

export interface PerformanceActivityLane {
  id: string;
  label: string;
  group: "cpu" | "gpu";
  /** `frame` means start/end share the capture's clock; `local` does not. */
  clockOrigin: "frame" | "local";
  tasks: readonly PerformanceActivityTask[];
}

export interface PerformanceActivitySegment {
  start_ms: number;
  end_ms: number;
  evidence: PerformanceActivityEvidence;
  /** Registered task identity retained so sampled rows can join timestamped stage envelopes. */
  taskId?: string;
  /** Stable semantic stage shared by aggregate timestamps and shader checkpoints. */
  stageId?: string;
  /** Present only when point evidence has been extended by an independent stage envelope. */
  projection?: "stage-envelope";
  color?: string;
  /** Temporal coverage for a raw row, or observed member occupancy for a coalesced display bin. */
  activeFraction?: number;
  /** Logical resources reporting progress in a coalesced display bin. */
  activeResourceCount?: number;
  /** Logical resources represented by a coalesced display bin. */
  resourceCount?: number;
  /** Measured active shader lanes across ballot-bearing heartbeat events. */
  activeLaneCount?: number;
  /** Measured logical shader lanes represented by those ballots. */
  logicalLaneCount?: number;
  label?: string;
  detail?: string;
}

export interface PerformanceActivityResource {
  id: string;
  label: string;
  group: "cpu" | "gpu";
  matrixKind: "cpu-context" | "gpu-logical" | "aggregate";
  /** Number of raw logical resources represented by this display row. */
  sourceResourceCount?: number;
  segments: readonly PerformanceActivitySegment[];
}

export interface PerformanceActivityGridProps {
  cpu?: PerformanceTrace;
  physics?: PerformanceTrace;
  presentation?: PerformanceTrace;
  frame?: PerformanceActivityFrame;
  /** Store-owned label for the selected retained capture. */
  captureLabel?: string;
  captureOptions?: readonly { id: string; label: string }[];
  selectedCaptureId?: string;
  onSelectCapture?: (frameId: string) => void;
  /** A retained store frame used for the compact duration comparison. */
  referenceFrame?: PerformanceActivityFrame;
  referenceLabel?: string;
  onSetReference?: () => void;
  onClearReference?: () => void;
  statusLabel?: string;
  className?: string;
}

export interface PerformanceActivityView {
  duration_ms: number;
  synchronized: boolean;
  lanes: readonly PerformanceActivityLane[];
  resources: readonly PerformanceActivityResource[];
  /** Raw logical GPU rows retained by the source frame, before display projection. */
  logicalGpuResourceCount: number;
  captureState: "complete" | "incomplete" | "verifying";
  captureDiagnostics?: PerformanceActivityCaptureDiagnostics;
}

export interface PerformanceActivityTaskLegendEntry {
  id: string;
  label: string;
  color: string;
  lane: ActivityLane;
  stageId: string;
  observed: boolean;
}

export interface PerformanceActivityTaskLegendGroup {
  id: string;
  label: string;
  entries: readonly PerformanceActivityTaskLegendEntry[];
}

/** React output is bounded while the retained frame keeps every captured workgroup row. */
export const GPU_ACTIVITY_DISPLAY_BAND_LIMIT = 16 as const;
export const DEFAULT_ACTIVITY_MILLISECONDS_WIDTH_PX = 2 as const;
export const MIN_ACTIVITY_MILLISECONDS_WIDTH_PX = 1 as const;
export const MAX_ACTIVITY_MILLISECONDS_WIDTH_PX = 8 as const;

const LANE_META: Readonly<Record<PerformanceTrace["lane"], {
  id: string;
  label: string;
  resourceLabel: string;
  group: "cpu" | "gpu";
}>> = {
  "main-thread": {
    id: "cpu-main-thread",
    label: "CPU · main thread",
    resourceLabel: "Main-thread envelope",
    group: "cpu",
  },
  physics: {
    id: "gpu-physics",
    label: "GPU · physics queue",
    resourceLabel: "Physics queue envelope",
    group: "gpu",
  },
  presentation: {
    id: "gpu-presentation",
    label: "GPU · presentation queue",
    resourceLabel: "Presentation queue envelope",
    group: "gpu",
  },
};

const PHASE_COLORS: Partial<Record<PaperPhaseId, string>> = {
  "frame-control": "#728f88",
  "scene-upload": "#4f94a4",
  "command-encoding": "#789a91",
  "coarse-grid": "#9b74c2",
  "power-topology": "#b478bd",
  "velocity-advection": "#32a9b8",
  "pressure-system": "#df9960",
  "pressure-solve": "#e8bf5e",
  "velocity-projection": "#d36972",
  "velocity-extrapolation": "#67b682",
  "fine-sdf-advection": "#39a8b1",
  "fine-sdf-redistance": "#66b684",
  "adaptive-publication": "#4d91a3",
  "surface-extraction": "#45a6cf",
  "water-caustics": "#52c4b0",
  "svo-cone-lighting": "#d2a55a",
  "svo-environment-gi": "#c68f55",
  "svo-voxel-light": "#bd9a63",
  "svo-primary": "#667dd0",
  "svo-rigid": "#5d91be",
  "svo-glass": "#6c9fd2",
  "dry-scene": "#657dcd",
  "water-front-interface": "#48a8ce",
  "water-back-interface": "#348fb9",
  "water-interfaces": "#45a6cf",
  "optical-composite": "#b56cc1",
  "inspection-overlay": "#ca8499",
  present: "#8ba5b0",
  other: "#9a7b55",
};

const FALLBACK_COLORS = ["#46a6c8", "#a079be", "#d79a61", "#62b68b", "#cf7180", "#7389cf"] as const;

const CANONICAL_LANE_META: Readonly<Record<ActivityLane, { label: string; group: "cpu" | "gpu" }>> = {
  "cpu-main": { label: "CPU · main thread", group: "cpu" },
  "cpu-worker": { label: "CPU · workers", group: "cpu" },
  "gpu-physics": { label: "GPU · physics", group: "gpu" },
  "gpu-presentation": { label: "GPU · presentation", group: "gpu" },
};

const MATRIX_GROUPS = [
  { kind: "cpu-context", label: "CPU SOFTWARE RESOURCES", group: "cpu" },
  { kind: "aggregate", label: "TIMED TASK / STAGE ENVELOPES", group: "gpu" },
  { kind: "gpu-logical", label: "GPU LOGICAL WORKGROUP BINS", group: "gpu" },
] as const satisfies readonly {
  kind: PerformanceActivityResource["matrixKind"];
  label: string;
  group: "cpu" | "gpu" | "mixed";
}[];

const finiteNonNegative = (value: number) => Number.isFinite(value) ? Math.max(0, value) : 0;

const phaseEvidence = (trace: PerformanceTrace): PerformanceActivityEvidence =>
  trace.measurementSource === "gpu-queue-wall" ? "unknown" : "reconstructed";

const canonicalEvidence = (evidence: ActivityEvidence): PerformanceActivityEvidence =>
  evidence === "measured" ? "measured-progress" : evidence;

function traceLane(trace: PerformanceTrace): PerformanceActivityLane {
  const meta = LANE_META[trace.lane];
  let cursor_ms = 0;
  const evidence = phaseEvidence(trace);
  const tasks = trace.phases.flatMap((phase, index) => {
    const duration_ms = finiteNonNegative(phase.duration_ms);
    const start_ms = cursor_ms;
    cursor_ms += duration_ms;
    if (duration_ms === 0) return [];
    return [{
      id: `${meta.id}-${trace.sampleId}-${index}`,
      label: phase.label,
      stageId: phase.id,
      start_ms,
      end_ms: cursor_ms,
      evidence,
      detail: trace.measurementSource === "gpu-queue-wall"
        ? "Queue-completion duration; semantic phase timing is unavailable"
        : "Measured duration reconstructed on this lane's local origin",
    }];
  });
  return { id: meta.id, label: meta.label, group: meta.group, clockOrigin: "local", tasks };
}

function traceResource(trace: PerformanceTrace, lane: PerformanceActivityLane): PerformanceActivityResource {
  const meta = LANE_META[trace.lane];
  const duration_ms = finiteNonNegative(trace.total_ms);
  const segments = Array.from({ length: Math.max(1, Math.ceil(duration_ms)) }, (_, index) => {
    const start_ms = index;
    const end_ms = Math.min(duration_ms, index + 1);
    const candidates = lane.tasks.map((task) => ({
      task,
      overlap_ms: Math.max(0, Math.min(end_ms, task.end_ms) - Math.max(start_ms, task.start_ms)),
    })).filter((candidate) => candidate.overlap_ms > 0)
      .sort((left, right) => right.overlap_ms - left.overlap_ms);
    const dominant = candidates[0]?.task;
    const active_ms = Math.min(end_ms - start_ms,
      candidates.reduce((total, candidate) => total + candidate.overlap_ms, 0));
    return {
      start_ms,
      end_ms,
      evidence: dominant?.evidence === "unknown" ? "unknown" : "reconstructed",
      taskId: dominant?.id,
      stageId: dominant?.stageId,
      color: dominant
        ? PHASE_COLORS[dominant.stageId as PaperPhaseId]
          ?? FALLBACK_COLORS[index % FALLBACK_COLORS.length]
        : undefined,
      activeFraction: end_ms > start_ms ? active_ms / (end_ms - start_ms) : 0,
      label: dominant?.label ?? "Unknown activity",
      detail: dominant?.evidence === "unknown"
        ? "No resource activity attribution is available"
        : "Reconstructed from adjacent phase boundaries; not a physical core measurement",
    } satisfies PerformanceActivitySegment;
  });
  return {
    id: `${lane.id}-envelope`,
    label: meta.resourceLabel,
    group: meta.group,
    matrixKind: "aggregate",
    segments,
  };
}

/** Pure adapter used by the UI and its contract tests. */
export function buildPerformanceActivityView(input: Pick<PerformanceActivityGridProps, "cpu" | "physics" | "presentation" | "frame">): PerformanceActivityView {
  if (input.frame) return canonicalFrameView(input.frame);
  const traces = [input.cpu, input.physics, input.presentation]
    .filter((trace): trace is PerformanceTrace => trace !== undefined);
  const lanes = traces.map(traceLane);
  const timestampFallback = traces.some((trace) => trace.domain === "gpu"
    && trace.measurementSource === "gpu-queue-wall");
  const captureDiagnostics = timestampFallback
    ? { reasons: ["timestamp-fallback" as const] }
    : undefined;
  return {
    duration_ms: Math.max(0.000001, ...traces.map((trace) => finiteNonNegative(trace.total_ms))),
    synchronized: false,
    lanes,
    resources: traces.map((trace, index) => traceResource(trace, lanes[index])),
    logicalGpuResourceCount: 0,
    captureState: captureDiagnostics ? "incomplete" : "verifying",
    ...(captureDiagnostics ? { captureDiagnostics } : {}),
  };
}

type CanonicalTask = PerformanceActivityFrame["tasks"][number];

function canonicalRowResource(input: {
  row: ActivityRow;
  taskById: ReadonlyMap<string, CanonicalTask>;
  relativeTime: (time_ms: number, resourceId: string, clockDomain: string) => number;
  trustIdle: boolean;
}): PerformanceActivityResource {
  const { row, taskById, relativeTime, trustIdle } = input;
  return {
    id: row.resource.id,
    label: row.resource.label,
    group: row.resource.lane.startsWith("cpu") ? "cpu" : "gpu",
    matrixKind: row.resource.kind === "gpu-logical-capacity"
      ? "gpu-logical"
      : row.resource.kind === "gpu-aggregate" ? "aggregate" : "cpu-context",
    segments: row.slices.map((slice) => {
      const task = slice.taskId ? taskById.get(slice.taskId) : undefined;
      return {
        start_ms: relativeTime(slice.start_ms, row.resource.id, row.resource.clockDomain),
        end_ms: relativeTime(slice.end_ms, row.resource.id, row.resource.clockDomain),
        evidence: slice.evidence === "idle" && !trustIdle ? "unknown" : canonicalEvidence(slice.evidence),
        taskId: slice.taskId,
        stageId: task ? task.stageId ?? task.parentId ?? task.id : undefined,
        color: task?.color,
        activeFraction: slice.end_ms > slice.start_ms
          ? Math.min(1, slice.active_ms / (slice.end_ms - slice.start_ms))
          : 0,
        label: task?.label
          ?? (slice.evidence === "idle" && trustIdle ? "Measured idle" : "Unknown activity"),
        detail: `${slice.active_ms.toFixed(3)} ms active · ${slice.spanIds.length} span${slice.spanIds.length === 1 ? "" : "s"}`,
      };
    }),
  };
}

/**
 * Project exhaustive logical GPU rows into a bounded number of display bins.
 * This never mutates or replaces the raw rows retained by PerformanceActivityFrame.
 */
export function coalesceGpuLogicalRows(input: {
  rows: readonly ActivityRow[];
  taskById: ReadonlyMap<string, CanonicalTask>;
  relativeTime: (time_ms: number, resourceId: string, clockDomain: string) => number;
  eventById?: ReadonlyMap<string, ActivityEvent>;
  limit?: number;
  trustIdle?: boolean;
}): PerformanceActivityResource[] {
  const { taskById, relativeTime } = input;
  const limit = Math.max(1, Math.floor(input.limit ?? GPU_ACTIVITY_DISPLAY_BAND_LIMIT));
  const ordered = input.rows.map((row, originalIndex) => ({ row, originalIndex })).sort((left, right) => {
    const leftSlot = left.row.resource.capacitySlot;
    const rightSlot = right.row.resource.capacitySlot;
    if (leftSlot !== undefined && rightSlot !== undefined && leftSlot !== rightSlot) return leftSlot - rightSlot;
    if (leftSlot !== undefined && rightSlot === undefined) return -1;
    if (leftSlot === undefined && rightSlot !== undefined) return 1;
    return left.row.resource.id.localeCompare(right.row.resource.id) || left.originalIndex - right.originalIndex;
  });
  const bandCount = Math.min(limit, ordered.length);
  if (bandCount === 0) return [];

  return Array.from({ length: bandCount }, (_, bandIndex): PerformanceActivityResource => {
    const firstIndex = Math.floor(bandIndex * ordered.length / bandCount);
    const lastIndex = Math.floor((bandIndex + 1) * ordered.length / bandCount);
    const members = ordered.slice(firstIndex, lastIndex).map(({ row }) => row);
    const sliceCount = members.reduce((maximum, row) => Math.max(maximum, row.slices.length), 0);
    const slots = members.map((row) => row.resource.capacitySlot).filter((slot): slot is number => slot !== undefined);
    const sampleCounts = members.map((row) => row.resource.sampleCount)
      .filter((count): count is number => count !== undefined);
    const dispatchCounts = [...new Set(members.flatMap((row) => row.resource.dispatchWorkgroupCounts ?? []))]
      .sort((left, right) => left - right);
    const dispatchRange = dispatchCounts.length === 0 ? undefined
      : dispatchCounts.length === 1 ? `${dispatchCounts[0]}`
      : `${dispatchCounts[0]}–${dispatchCounts.at(-1)}`;
    const memberRange = sampleCounts.length === members.length && dispatchRange
      ? `${Math.max(...sampleCounts)} / ${dispatchRange} workgroups sampled`
      : slots.length === members.length
        ? `strata ${Math.min(...slots) + 1}–${Math.max(...slots) + 1}`
        : `${members.length} sampled strata`;
    const firstMemberId = members[0]?.resource.id ?? bandIndex.toString();
    const lastMemberId = members.at(-1)?.resource.id ?? bandIndex.toString();
    const segments = Array.from({ length: sliceCount }, (_, sliceIndex): PerformanceActivitySegment => {
      const slices = members.flatMap((row) => {
        const slice = row.slices[sliceIndex];
        return slice ? [{ row, slice }] : [];
      });
      const active = slices.filter(({ slice }) => slice.evidence === "measured" || slice.evidence === "reconstructed");
      const taskScores = new Map<string, { count: number; active_ms: number }>();
      for (const { slice } of active) {
        if (!slice.taskId) continue;
        const current = taskScores.get(slice.taskId) ?? { count: 0, active_ms: 0 };
        current.count += 1;
        current.active_ms += slice.active_ms;
        taskScores.set(slice.taskId, current);
      }
      const dominantTaskId = [...taskScores.entries()].sort((left, right) =>
        right[1].count - left[1].count
        || right[1].active_ms - left[1].active_ms
        || left[0].localeCompare(right[0]))[0]?.[0];
      const start_ms = Math.min(...slices.map(({ row, slice }) =>
        relativeTime(slice.start_ms, row.resource.id, row.resource.clockDomain)));
      const end_ms = Math.max(...slices.map(({ row, slice }) =>
        relativeTime(slice.end_ms, row.resource.id, row.resource.clockDomain)));
      const activeResourceCount = active.length;
      const occupancy = activeResourceCount / members.length;
      const laneObservations = slices.flatMap(({ slice }) => slice.eventIds.flatMap((eventId) => {
        const metadata = input.eventById?.get(eventId)?.metadata;
        const activeLaneCount = metadata?.activeLaneCount;
        const logicalLaneCount = metadata?.logicalLaneCount;
        return typeof activeLaneCount === "number" && typeof logicalLaneCount === "number"
          && logicalLaneCount > 0 && activeLaneCount >= 0 && activeLaneCount <= logicalLaneCount
          ? [{ activeLaneCount, logicalLaneCount }]
          : [];
      }));
      const activeLaneCount = laneObservations.reduce((sum, sample) => sum + sample.activeLaneCount, 0);
      const logicalLaneCount = laneObservations.reduce((sum, sample) => sum + sample.logicalLaneCount, 0);
      const sampledLaneUtilization = logicalLaneCount > 0 ? activeLaneCount / logicalLaneCount : undefined;
      const evidence: PerformanceActivityEvidence = activeResourceCount > 0
        ? active.some(({ slice }) => slice.evidence === "reconstructed") ? "reconstructed" : "measured-progress"
        : input.trustIdle && slices.length === members.length && slices.every(({ slice }) => slice.evidence === "idle") ? "idle" : "unknown";
      const dominantTask = dominantTaskId ? taskById.get(dominantTaskId) : undefined;
      return {
        start_ms: Number.isFinite(start_ms) ? start_ms : sliceIndex,
        end_ms: Number.isFinite(end_ms) ? end_ms : sliceIndex + 1,
        evidence,
        taskId: dominantTaskId,
        stageId: dominantTask
          ? dominantTask.stageId ?? dominantTask.parentId ?? dominantTask.id
          : undefined,
        color: dominantTask?.color,
        activeFraction: sampledLaneUtilization ?? occupancy,
        activeResourceCount,
        resourceCount: members.length,
        ...(sampledLaneUtilization !== undefined ? { activeLaneCount, logicalLaneCount } : {}),
        label: dominantTask?.label ?? (evidence === "idle" ? "Measured idle" : evidence === "unknown" ? "Unknown activity" : "Observed GPU progress"),
        detail: sampledLaneUtilization === undefined
          ? `${activeResourceCount}/${members.length} sampled strata reported progress · ${(occupancy * 100).toFixed(1)}% sample coverage · ${memberRange} · no active-lane ballot · not a physical execution unit`
          : `${activeLaneCount}/${logicalLaneCount} active shader lanes · ${(sampledLaneUtilization * 100).toFixed(1)}% sampled logical utilization · ${activeResourceCount}/${members.length} strata reported · ${memberRange} · not hardware occupancy`,
      };
    });
    return {
      id: `gpu-logical-bin-${bandIndex}-${firstMemberId}-${lastMemberId}`,
      label: `Stratified WG bin ${String(bandIndex + 1).padStart(2, "0")} · ${memberRange}`,
      group: "gpu",
      matrixKind: "gpu-logical",
      sourceResourceCount: members.length,
      segments,
    };
  });
}

interface LogicalStageProfile {
  taskIds: Set<string>;
  activeLaneCount: number;
  logicalLaneCount: number;
  weightedActiveFraction: number;
  weight: number;
}

function mergeProjectedSegments(
  segments: readonly PerformanceActivitySegment[],
): PerformanceActivitySegment[] {
  const merged: PerformanceActivitySegment[] = [];
  for (const segment of segments) {
    const previous = merged.at(-1);
    const canMerge = segment.projection === "stage-envelope"
      && previous?.projection === "stage-envelope"
      && Math.abs(previous.end_ms - segment.start_ms) < 0.000001
      && previous.taskId === segment.taskId
      && previous.stageId === segment.stageId
      && previous.activeFraction === segment.activeFraction
      && previous.color === segment.color
      && previous.detail === segment.detail;
    if (canMerge && previous) previous.end_ms = segment.end_ms;
    else merged.push({ ...segment });
  }
  return merged;
}

/**
 * Horizontally bridge shader point samples with independently timestamped GPU
 * stage envelopes. Projection is deliberately row-local: an envelope is only
 * painted on a display band which sampled that semantic stage. It therefore
 * adds temporal context without inventing hardware-core occupancy.
 */
export function projectGpuStageEnvelopes(input: {
  resources: readonly PerformanceActivityResource[];
  taskById: ReadonlyMap<string, CanonicalTask>;
}): PerformanceActivityResource[] {
  const envelopes = input.resources
    .filter((resource) => resource.group === "gpu" && resource.matrixKind === "aggregate")
    .flatMap((resource) => resource.segments)
    .filter((segment) => segment.evidence !== "unknown" && segment.evidence !== "idle" && segment.stageId);
  if (envelopes.length === 0) return [...input.resources];
  // Index the one-millisecond aggregate slices once. This keeps opening the
  // panel proportional to painted evidence instead of bands × slices².
  const envelopesByBucket = new Map<number, PerformanceActivitySegment[]>();
  for (const envelope of envelopes) {
    const firstBucket = Math.floor(envelope.start_ms);
    const lastBucket = Math.max(firstBucket, Math.ceil(envelope.end_ms) - 1);
    for (let bucket = firstBucket; bucket <= lastBucket; bucket += 1) {
      const candidates = envelopesByBucket.get(bucket) ?? [];
      candidates.push(envelope);
      envelopesByBucket.set(bucket, candidates);
    }
  }

  return input.resources.map((resource) => {
    if (resource.matrixKind !== "gpu-logical") return resource;
    const profiles = new Map<string, LogicalStageProfile>();
    for (const segment of resource.segments) {
      if (!segment.taskId || !segment.stageId
        || (segment.evidence !== "measured-progress" && segment.evidence !== "reconstructed")) continue;
      const profile = profiles.get(segment.stageId) ?? {
        taskIds: new Set<string>(),
        activeLaneCount: 0,
        logicalLaneCount: 0,
        weightedActiveFraction: 0,
        weight: 0,
      };
      profile.taskIds.add(segment.taskId);
      if (segment.logicalLaneCount && segment.activeLaneCount !== undefined) {
        profile.activeLaneCount += segment.activeLaneCount;
        profile.logicalLaneCount += segment.logicalLaneCount;
      }
      const weight = Math.max(Number.EPSILON, segment.end_ms - segment.start_ms);
      profile.weightedActiveFraction += (segment.activeFraction ?? 1) * weight;
      profile.weight += weight;
      profiles.set(segment.stageId, profile);
    }
    if (profiles.size === 0) return resource;

    const segments = resource.segments.map((segment): PerformanceActivitySegment => {
      if (segment.evidence !== "unknown") return segment;
      let envelope: PerformanceActivitySegment | undefined;
      let envelopeOverlap_ms = 0;
      const seen = new Set<PerformanceActivitySegment>();
      for (let bucket = Math.floor(segment.start_ms);
        bucket <= Math.max(Math.floor(segment.start_ms), Math.ceil(segment.end_ms) - 1);
        bucket += 1) {
        for (const candidate of envelopesByBucket.get(bucket) ?? []) {
          if (seen.has(candidate) || !candidate.stageId || !profiles.has(candidate.stageId)) continue;
          seen.add(candidate);
          const overlap_ms = Math.max(0,
            Math.min(segment.end_ms, candidate.end_ms) - Math.max(segment.start_ms, candidate.start_ms));
          if (overlap_ms > envelopeOverlap_ms) {
            envelope = candidate;
            envelopeOverlap_ms = overlap_ms;
          }
        }
      }
      if (!envelope?.stageId) return segment;
      const profile = profiles.get(envelope.stageId);
      if (!profile) return segment;
      const ballotFraction = profile.logicalLaneCount > 0
        ? profile.activeLaneCount / profile.logicalLaneCount
        : profile.weight > 0 ? profile.weightedActiveFraction / profile.weight : 0;
      const anchorLabels = [...profile.taskIds]
        .map((taskId) => input.taskById.get(taskId)?.label ?? taskId)
        .sort()
        .join(", ");
      return {
        ...segment,
        evidence: "reconstructed",
        taskId: envelope.taskId,
        stageId: envelope.stageId,
        projection: "stage-envelope",
        color: envelope.color,
        activeFraction: ballotFraction,
        label: `${envelope.label ?? envelope.stageId} · projected context`,
        detail: `Timestamped stage envelope, anchored by sampled shader checkpoint${profile.taskIds.size === 1 ? "" : "s"}: ${anchorLabels} · intensity held from ${profile.logicalLaneCount > 0 ? `${profile.activeLaneCount}/${profile.logicalLaneCount} active-lane ballots` : "sample coverage"} · inferred time, excluded from utilization, not hardware occupancy`,
      };
    });
    return { ...resource, segments: mergeProjectedSegments(segments) };
  });
}

function canonicalFrameView(frame: PerformanceActivityFrame): PerformanceActivityView {
  const captureState = frame.captureDiagnostics
    ? frame.captureDiagnostics.reasons.length === 0 ? "complete" : "incomplete"
    : "verifying";
  const trustIdle = captureState === "complete";
  const clockById = new Map(frame.clocks.map((clock) => [clock.id, clock]));
  const synchronized = frame.clocks.length > 0
    && frame.clocks.every((clock) => clock.status.state !== "unsynchronized");
  const offsetFor = (clockDomain: string) => {
    const status = clockById.get(clockDomain)?.status;
    return status?.state === "synchronized" ? status.offset_ms : 0;
  };
  const domainStarts = new Map<string, number>();
  for (const row of frame.rows) domainStarts.set(
    row.resource.clockDomain,
    Math.min(domainStarts.get(row.resource.clockDomain) ?? Infinity, row.windowStart_ms),
  );
  const alignedStarts = frame.rows.map((row) => row.windowStart_ms + offsetFor(row.resource.clockDomain));
  const commonStart_ms = synchronized && alignedStarts.length > 0 ? Math.min(...alignedStarts) : 0;
  const relativeTime = (time_ms: number, resourceId: string, clockDomain: string) => synchronized
    ? time_ms + offsetFor(clockDomain) - commonStart_ms
    : time_ms - (domainStarts.get(clockDomain) ?? frame.rows.find((row) => row.resource.id === resourceId)?.windowStart_ms ?? 0);
  const rowDurations = frame.rows.map((row) => synchronized
    ? row.windowEnd_ms + offsetFor(row.resource.clockDomain) - commonStart_ms
    : row.windowEnd_ms - (domainStarts.get(row.resource.clockDomain) ?? row.windowStart_ms));
  const duration_ms = Math.max(0.000001, ...rowDurations.map(finiteNonNegative));
  const taskById = new Map(frame.tasks.map((task) => [task.id, task]));
  const laneTasks = new Map<ActivityLane, PerformanceActivityTask[]>();
  for (const span of frame.spans) {
    if (span.kind !== "task" || !span.taskId) continue;
    const task = taskById.get(span.taskId);
    if (!task) continue;
    const tasks = laneTasks.get(task.lane) ?? [];
    tasks.push({
      id: span.id,
      label: task.label,
      stageId: task.id,
      start_ms: relativeTime(span.start_ms, span.resourceId, span.clockDomain),
      end_ms: relativeTime(span.end_ms, span.resourceId, span.clockDomain),
      evidence: canonicalEvidence(span.evidence),
      detail: `${span.evidence === "measured" ? "Measured" : "Reconstructed"} activity · ${span.clockDomain}`,
      color: task.color,
    });
    laneTasks.set(task.lane, tasks);
  }
  const laneOrder: readonly ActivityLane[] = ["cpu-main", "cpu-worker", "gpu-physics", "gpu-presentation"];
  const lanes = laneOrder.flatMap((lane): PerformanceActivityLane[] => {
    const tasks = laneTasks.get(lane);
    if (!tasks?.length) return [];
    const meta = CANONICAL_LANE_META[lane];
    return [{ id: lane, label: meta.label, group: meta.group, clockOrigin: synchronized ? "frame" : "local", tasks }];
  });
  const logicalRows = frame.rows.filter((row) => row.resource.kind === "gpu-logical-capacity");
  const eventById = new Map(frame.events.map((event) => [event.id, event]));
  const resources = projectGpuStageEnvelopes({ resources: [
    ...frame.rows.filter((row) => row.resource.kind !== "gpu-logical-capacity")
      .map((row) => canonicalRowResource({ row, taskById, relativeTime, trustIdle })),
    ...coalesceGpuLogicalRows({ rows: logicalRows, taskById, relativeTime, eventById, trustIdle }),
  ], taskById });
  return {
    duration_ms,
    synchronized,
    lanes,
    resources,
    logicalGpuResourceCount: logicalRows.length,
    captureState,
    ...(frame.captureDiagnostics ? { captureDiagnostics: frame.captureDiagnostics } : {}),
  };
}

const CAPTURE_REASON_LABELS: Readonly<Partial<Record<PerformanceActivityCaptureReason, string>>> = {
  "recorder-overflow": "RECORDER OVERFLOW",
  "missing-frame-begin": "MISSING FRAME BEGIN",
  "missing-frame-end": "MISSING FRAME END",
  "phase-marker-mismatch": "PHASE MARKER MISMATCH",
  "unprofiled-dispatch": "UNPROFILED DISPATCH",
  "row-limit": "ROW LIMIT",
  "unprojected-event": "UNPROJECTED EVENT",
  "timestamp-fallback": "TIMESTAMP FALLBACK",
};

function captureStatusLabel(view: PerformanceActivityView): { short: string; detail: string } {
  if (view.captureState === "verifying") return {
    short: "CAPTURE VERIFYING",
    detail: "Completeness verdict is awaiting recorder readback",
  };
  if (view.captureState === "complete") return {
    short: "CAPTURE COMPLETE",
    detail: "Frame boundaries and retained recorder evidence passed validation",
  };
  const diagnostics = view.captureDiagnostics;
  const reasons = diagnostics?.reasons.map((reason) => CAPTURE_REASON_LABELS[reason] ?? reason.toUpperCase()) ?? [];
  const dropped = (diagnostics?.droppedEventCount ?? 0) + (diagnostics?.droppedRowCount ?? 0);
  const unprofiled = diagnostics?.unprofiledDispatchCount ?? 0;
  const unprofiledLabels = diagnostics?.unprofiledPipelineLabels?.join(", ");
  return {
    short: `CAPTURE INCOMPLETE${dropped > 0 ? ` · ${dropped} DROPPED` : ""}`,
    detail: [
      ...reasons,
      ...(unprofiled > 0 ? [`${unprofiled} unprofiled dispatch${unprofiled === 1 ? "" : "es"}`] : []),
      ...(unprofiledLabels ? [unprofiledLabels] : []),
    ].join(" · ") || "Capture validation failed",
  };
}

const humanizeStage = (stageId: string) => stageId
  .replace(/[._]/g, "-")
  .split("-")
  .filter(Boolean)
  .map((part) => part[0]?.toUpperCase() + part.slice(1))
  .join(" ");

/** Dynamic generation-scoped catalog, with observation state from raw frame evidence. */
export function performanceActivityTaskLegend(
  frame: PerformanceActivityFrame | undefined,
): PerformanceActivityTaskLegendGroup[] {
  if (!frame) return [];
  const observed = new Set<string>();
  for (const span of frame.spans) if (span.taskId) observed.add(span.taskId);
  for (const event of frame.events) if (event.taskId) observed.add(event.taskId);
  for (const row of frame.rows) for (const slice of row.slices) {
    if (slice.taskId) observed.add(slice.taskId);
  }
  const groups = new Map<string, PerformanceActivityTaskLegendEntry[]>();
  for (const task of frame.tasks) {
    const stageId = task.stageId ?? task.parentId ?? task.lane;
    const entries = groups.get(stageId) ?? [];
    entries.push({
      id: task.id,
      label: task.label,
      color: task.color,
      lane: task.lane,
      stageId,
      observed: observed.has(task.id),
    });
    groups.set(stageId, entries);
  }
  return [...groups.entries()].map(([id, entries]) => ({
    id,
    label: humanizeStage(id),
    entries: entries.sort((left, right) => Number(right.observed) - Number(left.observed)
      || left.label.localeCompare(right.label)),
  })).sort((left, right) => {
    const leftObserved = left.entries.some((entry) => entry.observed);
    const rightObserved = right.entries.some((entry) => entry.observed);
    return Number(rightObserved) - Number(leftObserved) || left.label.localeCompare(right.label);
  });
}

function niceTickStep(duration_ms: number): number {
  const rough = Math.max(duration_ms / 8, Number.EPSILON);
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const normalized = rough / magnitude;
  const unit = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
  return unit * magnitude;
}

export function performanceActivityTicks(duration_ms: number): number[] {
  const safeDuration = Math.max(0.000001, finiteNonNegative(duration_ms));
  const step = niceTickStep(safeDuration);
  const ticks: number[] = [];
  for (let value = 0; value <= safeDuration + step * 0.0001; value += step) ticks.push(value);
  if (ticks.length === 1) ticks.push(safeDuration);
  return ticks;
}

const formatMs = (value: number) => {
  if (value === 0) return "0 ms";
  if (value < 0.01) return `${value.toFixed(4)} ms`;
  if (value < 1) return `${value.toFixed(3)} ms`;
  return `${value.toFixed(value < 10 ? 2 : 1)} ms`;
};

const geometry = (start_ms: number, end_ms: number, duration_ms: number): CSSProperties => {
  const start = Math.min(duration_ms, finiteNonNegative(start_ms));
  const end = Math.min(duration_ms, Math.max(start, finiteNonNegative(end_ms)));
  return { left: `${start / duration_ms * 100}%`, width: `${(end - start) / duration_ms * 100}%` };
};

const cellStyle = (
  segment: PerformanceActivitySegment,
  index: number,
  duration_ms: number,
): CSSProperties => ({
  ...geometry(segment.start_ms, segment.end_ms, duration_ms),
  "--activity-cell-color": segment.color ?? FALLBACK_COLORS[index % FALLBACK_COLORS.length],
  // A measured zero-lane ballot is progress evidence but zero sampled logical
  // utilization; keep it barely visible rather than painting it as busy.
  "--activity-cell-opacity": 0.06
    + 0.94 * Math.max(0, Math.min(1, segment.activeFraction ?? 1)),
} as CSSProperties);

function matrixStats(view: PerformanceActivityView) {
  const detailed = view.resources.filter((resource) => resource.matrixKind !== "aggregate");
  const resources = detailed.length > 0 ? detailed : view.resources;
  let activeCells = 0;
  let observedCells = 0;
  for (const resource of resources) for (const segment of resource.segments) {
    // Projected bars explain when an observed stage could have occupied this
    // sampled stratum. They are visual context, never utilization evidence.
    if (segment.evidence === "unknown" || segment.projection) continue;
    observedCells += 1;
    if (segment.evidence === "measured-progress" || segment.evidence === "reconstructed") {
      activeCells += Math.max(0, Math.min(1, segment.activeFraction ?? 1));
    }
  }
  return {
    cpuRows: resources.filter((resource) => resource.group === "cpu").length,
    gpuRows: resources.filter((resource) => resource.group === "gpu").length,
    logicalGpuBands: resources.filter((resource) => resource.matrixKind === "gpu-logical").length,
    logicalGpuRows: view.logicalGpuResourceCount,
    activeCellRatio: observedCells > 0 ? activeCells / observedCells : undefined,
    taskCount: new Set(view.lanes.flatMap((lane) => lane.tasks.map((task) => task.stageId ?? task.id))).size,
  };
}

function Ruler({ duration_ms, ticks }: { duration_ms: number; ticks: readonly number[] }) {
  return <div className="activity-ruler" aria-label={`Shared time ruler, ${formatMs(duration_ms)}`}>
    <div className="activity-axis-label">TIME</div>
    <div className="activity-ruler-plot">
      {ticks.map((tick) => <span key={tick} style={{ left: `${tick / duration_ms * 100}%` }}><i />{formatMs(tick)}</span>)}
    </div>
  </div>;
}

function GridLines({ duration_ms, ticks }: { duration_ms: number; ticks: readonly number[] }) {
  return <>{ticks.map((tick) => <i
    className="activity-grid-line"
    key={tick}
    style={{ left: `${tick / duration_ms * 100}%` }}
    aria-hidden="true"
  />)}</>;
}

export function PerformanceActivityGrid({
  cpu,
  physics,
  presentation,
  frame,
  captureLabel,
  captureOptions = [],
  selectedCaptureId,
  onSelectCapture,
  referenceFrame,
  referenceLabel,
  onSetReference,
  onClearReference,
  statusLabel,
  className = "",
}: PerformanceActivityGridProps) {
  const [millisecondsWidth_px, setMillisecondsWidth_px] = useState<number>(
    DEFAULT_ACTIVITY_MILLISECONDS_WIDTH_PX,
  );
  const view = useMemo(
    () => buildPerformanceActivityView({ cpu, physics, presentation, frame }),
    [cpu, physics, presentation, frame],
  );
  const referenceView = useMemo(
    () => referenceFrame ? buildPerformanceActivityView({ frame: referenceFrame }) : undefined,
    [referenceFrame],
  );
  const ticks = useMemo(() => performanceActivityTicks(view.duration_ms), [view.duration_ms]);
  const sliceCount = Math.max(1, Math.ceil(view.duration_ms));
  // The aggregate lane and logical samples answer different questions and
  // must remain visible together. The envelope is continuous task context
  // from hardware-timestamped stage boundaries; logical rows are sparse,
  // stratified shader progress observations. Hiding the envelope when samples
  // exist makes honest point evidence look like an almost-empty GPU timeline.
  const matrixResources = view.resources;
  const hasData = matrixResources.length > 0;
  const stats = useMemo(() => matrixStats(view), [view]);
  const ledgers = [cpu, physics, presentation].filter(
    (trace): trace is PerformanceTrace => trace !== undefined,
  );
  const retainedLedgerResourceIds = new Set(frame?.spans
    .filter((span) => span.kind === "observation")
    .map((span) => span.resourceId) ?? []);
  const retainedLedgerCount = retainedLedgerResourceIds.size;
  const ledgerCount = ledgers.length > 0 ? ledgers.length : retainedLedgerCount;
  // Canonical retained frames are produced only after each source trace has
  // closed; asynchronous shader evidence is merged into that immutable base.
  const closedLedgers = ledgers.length > 0
    ? ledgers.filter((trace) => performanceTraceAccounting(trace).exact).length
    : retainedLedgerCount;
  const queueWallFallback = ledgers.some((trace) => trace.measurementSource === "gpu-queue-wall");
  const selectedCaptureLabel = captureLabel ?? frame?.identity.frameId;
  const resolvedReferenceLabel = referenceLabel ?? referenceFrame?.identity.frameId;
  const taskLegend = useMemo(() => performanceActivityTaskLegend(frame), [frame]);
  const observedLegendTasks = taskLegend.reduce((count, group) =>
    count + group.entries.filter((entry) => entry.observed).length, 0);
  const registeredLegendTasks = taskLegend.reduce((count, group) => count + group.entries.length, 0);
  const captureStatus = captureStatusLabel(view);
  const completeUtilization = view.captureState === "complete";

  return <section
    className={`performance-activity-grid ${className}`.trim()}
    data-clock-alignment={view.synchronized ? "synchronized" : "independent"}
    data-accounting-ledgers="cpu-gpu-independent"
    data-queue-wall-fallback={queueWallFallback}
    data-capture-completeness={view.captureState}
    aria-label="Frame resource activity matrix"
  >
    <header className="activity-grid-header">
      <div><small>UNIFIED FRAME ACCOUNTING</small><h3>Resource activity matrix</h3></div>
      <div className="activity-header-meta">
        <label className="activity-density-control">
          <span>TIME SCALE</span>
          <input
            type="range"
            min={MIN_ACTIVITY_MILLISECONDS_WIDTH_PX}
            max={MAX_ACTIVITY_MILLISECONDS_WIDTH_PX}
            step={1}
            value={millisecondsWidth_px}
            onChange={(event) => setMillisecondsWidth_px(Number(event.currentTarget.value))}
            aria-label="Matrix horizontal scale in pixels per millisecond"
          />
          <output>{millisecondsWidth_px} PX/MS</output>
        </label>
        {statusLabel && <span>{statusLabel}</span>}
        <span className="activity-capture-status" data-state={view.captureState} title={captureStatus.detail}>
          {captureStatus.short}
        </span>
        {queueWallFallback && <span className="activity-queue-fallback" title="The duration waits for the whole submitted GPU queue and can include work queued before this lane">
          QUEUE-WALL FALLBACK · INCLUDES BACKLOG
        </span>}
        <span className={view.synchronized ? "synchronized" : "independent"}>
          {view.synchronized ? "CORRELATED FRAME CLOCK" : "INDEPENDENT LANE ORIGINS"}
        </span>
      </div>
    </header>
    <div className="activity-matrix-summary" aria-label="Matrix summary">
      <span><small>{queueWallFallback ? "QUEUE ENVELOPE" : "FRAME WINDOW"}</small><strong>{formatMs(view.duration_ms)}</strong></span>
      <span><small>CPU ROWS</small><strong>{stats.cpuRows}</strong></span>
      <span><small>GPU WG / DISPLAY BANDS</small><strong>{stats.logicalGpuRows > 0 ? `${stats.logicalGpuRows} / ${stats.logicalGpuBands}` : stats.gpuRows}</strong></span>
      <span><small>{completeUtilization ? "ACTIVE OBSERVED CELLS" : "UTILIZATION"}</small><strong>{completeUtilization
        ? stats.activeCellRatio === undefined ? "—" : `${(stats.activeCellRatio * 100).toFixed(1)}%`
        : "WITHHELD"}</strong></span>
      <span><small>TASKS OBS / REG</small><strong>{registeredLegendTasks > 0 ? `${observedLegendTasks} / ${registeredLegendTasks}` : stats.taskCount}</strong></span>
      <span><small>ACCOUNTING LEDGERS</small><strong>{ledgerCount === 0 ? "—" : `${closedLedgers}/${ledgerCount} CLOSED`}</strong></span>
    </div>
    {(captureOptions.length > 0 || referenceView) && <div className="activity-matrix-capture" aria-label="Retained performance captures">
      <label>
        <span>FRAME</span>
        <select
          value={selectedCaptureId ?? ""}
          disabled={captureOptions.length === 0}
          onChange={(event) => onSelectCapture?.(event.currentTarget.value)}
          aria-label="Selected performance frame"
        >
          {captureOptions.length === 0 && <option value="">Waiting for capture</option>}
          {captureOptions.map((capture) => <option key={capture.id} value={capture.id}>{capture.label}</option>)}
        </select>
      </label>
      {(selectedCaptureLabel || referenceView) && <div className="activity-capture-comparison" aria-label="Capture comparison">
        <span title={selectedCaptureLabel ?? "Selected capture"}>{selectedCaptureLabel ?? "SELECTED"}</span>
        {referenceView ? <><b>VS</b><span title={resolvedReferenceLabel}>{resolvedReferenceLabel}</span><output className={view.duration_ms <= referenceView.duration_ms ? "faster" : "slower"}>
          {view.duration_ms === referenceView.duration_ms ? "±0.00 ms" : `${view.duration_ms > referenceView.duration_ms ? "+" : "−"}${formatMs(Math.abs(view.duration_ms - referenceView.duration_ms))}`}
        </output>{onClearReference && <button type="button" onClick={onClearReference} aria-label="Clear reference capture">×</button>}</> : onSetReference && <button className="activity-set-reference" type="button" onClick={onSetReference}>SET REFERENCE</button>}
      </div>}
    </div>}
    <section
      className="activity-task-legend"
      data-empty={taskLegend.length === 0}
      aria-label="Dynamically registered task and stage legend"
    >
      <header><span>TASK / STAGE LEGEND</span><small>{observedLegendTasks} OBSERVED · {registeredLegendTasks} REGISTERED</small></header>
      <div className="activity-task-legend-groups">
        {taskLegend.length === 0 ? <div className="activity-task-legend-empty">TASK CATALOG AWAITING CAPTURE</div>
          : taskLegend.map((group) => <div className="activity-task-legend-group" key={group.id}>
          <strong>{group.label}</strong>
          <div>{group.entries.map((entry) => <span
            data-observed={entry.observed}
            data-lane={entry.lane}
            key={entry.id}
            title={`${entry.label} · ${entry.stageId} · ${entry.observed ? "observed in this capture" : "registered, not observed in this capture"}`}
          ><i style={{ background: entry.color }} />{entry.label}</span>)}</div>
        </div>)}
      </div>
    </section>
    {!hasData ? <div className="activity-grid-empty">Waiting for a complete measured activity sample.</div> : <>
      <div className="activity-scrollport">
        <div className="activity-timeline" style={{
          "--activity-slice-count": sliceCount,
          "--activity-ms-width": `${millisecondsWidth_px}px`,
        } as CSSProperties}>
          <Ruler duration_ms={view.duration_ms} ticks={ticks} />
          <div className="activity-section-label"><span>Resource × 1 ms activity matrix</span><small>{completeUtilization
            ? "TASK COLOR · ONE CELL PER RESOURCE PER TIME SLICE"
            : "PARTIAL EVIDENCE · IDLE AND UTILIZATION NOT ASSERTED"}</small></div>
          <div className="activity-resource-lanes">
            {MATRIX_GROUPS.map(({ kind, label, group }) => {
              const resources = matrixResources.filter((resource) => resource.matrixKind === kind);
              if (resources.length === 0) return null;
              const hasBallotUtilization = kind === "gpu-logical" && resources.some(
                (resource) => resource.segments.some((segment) => segment.logicalLaneCount !== undefined),
              );
              const hasStageProjection = kind === "gpu-logical" && resources.some(
                (resource) => resource.segments.some((segment) => segment.projection === "stage-envelope"),
              );
              return <div className="activity-resource-group" data-group={group} data-kind={kind} key={kind}>
                <div className="activity-resource-group-label">
                  <span>{label}</span>
                  <small>{kind === "gpu-logical"
                    ? `${resources.length} BAND${resources.length === 1 ? "" : "S"} · ${stats.logicalGpuRows} STRATIFIED SAMPLE${stats.logicalGpuRows === 1 ? "" : "S"} · ${hasBallotUtilization ? "ACTIVE-LANE INTENSITY" : "SAMPLE COVERAGE"}${hasStageProjection ? " · TIMESTAMP-PROJECTED CONTEXT" : ""}`
                    : kind === "aggregate"
                      ? `${resources.length} ${queueWallFallback ? "QUEUE-COMPLETION ENVELOPE" : "TIMESTAMPED QUEUE LANE"}${resources.length === 1 ? "" : "S"}`
                      : `${resources.length} ROW${resources.length === 1 ? "" : "S"}`}</small>
                </div>
                {resources.map((resource, resourceIndex) => <div className="activity-resource-row" key={resource.id} data-group={resource.group}>
                  <div className="activity-resource-label"><b>{String(resourceIndex).padStart(2, "0")}</b><span>{resource.label}</span><small>{formatMs(Math.max(0, ...resource.segments.map((segment) => segment.end_ms)))}</small></div>
                  <div className="activity-plot activity-resource-plot">
                    <GridLines duration_ms={view.duration_ms} ticks={ticks} />
                    {resource.segments.flatMap((segment, index) => segment.evidence === "unknown" ? [] : [<span
                      className="activity-cell"
                      data-evidence={segment.evidence}
                      data-projection={segment.projection}
                      data-active-resources={segment.activeResourceCount}
                      data-resource-count={segment.resourceCount}
                      data-active-lanes={segment.activeLaneCount}
                      data-logical-lanes={segment.logicalLaneCount}
                      key={`${segment.start_ms}-${segment.end_ms}-${index}`}
                      style={cellStyle(segment, index, view.duration_ms)}
                      title={`${segment.label ?? segment.evidence} · ${formatMs(segment.end_ms - segment.start_ms)} · ${segment.detail ?? segment.evidence}`}
                    />])}
                  </div>
                </div>)}
              </div>;
            })}
            {!view.resources.some((resource) => resource.matrixKind === "gpu-logical") &&
              view.resources.some((resource) => resource.group === "gpu") &&
              <div className="activity-matrix-missing">GPU LOGICAL ROWS AWAITING SHADER HEARTBEAT READBACK</div>}
          </div>
        </div>
      </div>
      <footer className="activity-evidence-legend" aria-label="Activity evidence legend">
        <span data-evidence="measured-progress"><i />Measured progress</span>
        <span data-evidence="reconstructed"><i />Reconstructed</span>
        <span data-evidence="idle"><i />Measured idle · complete capture only</span>
        <span data-evidence="unknown"><i />Unknown</span>
      </footer>
    </>}
  </section>;
}
