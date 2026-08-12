/**
 * Reduce an exported Metal System Trace into a per-frame CPU+GPU breakdown of
 * one simulation advance, with throughput and occupancy attributed to
 * individual tasks and shaders.
 *
 * Three things make that attribution possible, and each has a catch:
 *
 *  - GPU *intervals* time the work but their label text is not a usable
 *    grouping key: Instruments concatenates the labels of every encoder it drew
 *    as one interval, so the same task yields a different string whenever the
 *    merge grouping shifts, and 128 real tasks explode into thousands of
 *    one-off keys. The CPU-side encoder list is stable, so `encoder-id` is the
 *    join key and its label is the authority. Dawn packs several WebGPU compute
 *    passes into one Metal encoder, so an encoder is the finest unit with a
 *    real GPU timestamp; its constituent passes are listed, not discarded.
 *
 *  - GPU *counters* (occupancy, ALU, bandwidth) are device-wide. They cannot
 *    be filtered by process, so they are only sampled inside windows where no
 *    other process had GPU work in flight. The share of our GPU time that
 *    qualifies is reported as `exclusiveCoverage` -- occupancy numbers are
 *    only as trustworthy as that number is large.
 *
 *  - The trace has no notion of a simulation step, so frame boundaries are
 *    recovered by finding a pass that fires exactly once per advance.
 */
import { xctraceSafeComputeLabel } from "../lib/webgpu-pass-broker";
import { readTraceRows, durationMicroseconds, timestampMicroseconds } from "./xctrace-trace-tables";

export interface Interval {
  /** Stable Metal encoder identity. One encoder may appear as several GPU
   * execution slices when it is suspended and resumed. */
  readonly encoderId?: string;
  readonly start: number;
  readonly duration: number;
  readonly label: string;
  readonly encoders: readonly string[];
  readonly channel: string;
  readonly merged: boolean;
  occupancy?: number | null;
  alu?: number | null;
}

export interface PassCost {
  readonly label: string;
  /**
   * True only when label isolation proves this encoder contains that stage and
   * nothing else. When false, `label` is merely the FIRST label of the Metal
   * encoder: `gpuMsPerFrame` is the whole encoder, including every stage
   * encoded after it until the next pass boundary.
   */
  readonly exactAttribution: boolean;
  /**
   * Why this bucket is not a stage cost. Present exactly when
   * `exactAttribution` is false, so a renderer can never show the number
   * without the caveat.
   */
  readonly compositeReason?: string;
  readonly callsPerFrame: number;
  readonly gpuMsPerFrame: number;
  readonly meanMicroseconds: number;
  readonly share: number;
  readonly merged: boolean;
  readonly exclusiveShare: number;
  readonly counterSamples: number;
  readonly occupancy?: number;
  readonly alu?: number;
  readonly readGBs?: number;
  readonly writeGBs?: number;
  readonly limiter?: string;
  /** Measured occupancy on each GPU partition, 0..1, index = hardware stream. */
  readonly partitions?: readonly number[];
  /** Every counter measured inside this task, percent or native units. */
  readonly counters?: Readonly<Record<string, number>>;
  /** Resident SIMD groups implied by the measured occupancy. */
  readonly residentSlots?: number;
  readonly residentThreads?: number;
  /** Peak partition occupancy divided by the mean: 1 is balanced. */
  readonly imbalance?: number;
  readonly shaders?: readonly { name: string; samples: number }[];
}

export interface FlameNode {
  readonly name: string;
  value: number;
  children: FlameNode[];
}

/**
 * One advance, measured on its own. The averages elsewhere in the report are
 * these reduced; keeping them separate is what lets a reader check that every
 * advance has the same shape -- and compare an early one against a late one.
 */
export interface FrameSample {
  /** Position in the analysed window, 0-based. */
  readonly index: number;
  /** Milliseconds from the start of the analysed window. */
  readonly startMs: number;
  readonly durationMs: number;
  readonly busyMs: number;
  readonly gapMs: number;
  readonly encoders: number;
  readonly passes: number;
  readonly occupancy?: number;
  readonly counterSamples: number;
  /** GPU ms in this advance, aligned to `frames.taskLabels`. */
  readonly tasks: readonly number[];
}

/** A frame retained in full, so the timeline can be drawn for it. */
export interface FrameCapture {
  readonly index: number;
  readonly durationUs: number;
  readonly intervals: readonly Interval[];
}

export interface FrameReport {
  /** Display vocabulary for this report. Older summaries default to advance. */
  readonly workUnit?: "advance" | "frame";
  readonly title?: string;
  readonly lane: string;
  readonly scene: string;
  readonly grid: string;
  readonly capturedAt: string;
  readonly wall: {
    readonly baselineMsPerAdvance?: number;
    readonly tracedMsPerAdvance?: number;
    readonly distortion?: number;
    readonly steps?: number;
  };
  readonly frames: {
    readonly count: number; readonly anchor: string;
    readonly medianMs: number; readonly p10Ms: number; readonly p90Ms: number;
    /** Task labels the per-frame `tasks` arrays are aligned to, costliest first. */
    readonly taskLabels: readonly string[];
    /** Every analysed advance, in capture order. */
    readonly samples: readonly FrameSample[];
    /**
     * Frames whose interval list is retained in full. Capped and evenly spread
     * -- always including the first, the last and the representative -- so a
     * 500-advance capture stays a file you can open.
     */
    readonly captures: readonly FrameCapture[];
    /** Index into `captures` of the frame the timeline shows by default. */
    readonly representative: number;
    /** Advance number in the run of `samples[0]`, when it can be recovered. */
    readonly firstAdvance?: number;
  };
  readonly gpu: {
    readonly busyMsPerFrame: number; readonly wallMsPerFrame: number;
    /** Sum of attributed interval durations. May exceed busy time because
     * Instruments can emit overlapping parent/child Metal intervals. */
    readonly intervalMsPerFrame: number;
    readonly overlapMsPerFrame: number;
    readonly exactMsPerFrame: number;
    readonly compositeMsPerFrame: number;
    readonly exactIntervalCoverage: number;
    readonly occupancy: number; readonly intervalsPerFrame: number;
    readonly encodersPerFrame: number; readonly passesPerFrame: number;
    readonly gapMsPerFrame: number;
    readonly largestGapsUs: readonly number[];
    readonly mergedShare: number;
    /** Metal blit time inserted by diagnostic encoder isolation, not solver
     * stage work. It remains part of busy/wall time but not the stage table. */
    readonly diagnosticBlitMsPerFrame: number;
    readonly diagnosticBlitsPerFrame: number;
  };
  readonly counters: {
    /** The pipeline-specific occupancy stream used throughout this report. */
    readonly occupancyCounterName?: "Compute Occupancy" | "Fragment Occupancy";
    readonly available: boolean;
    readonly partitionCount: number;
    readonly partitionOccupancy: readonly number[];
    /** Scheduler slots per partition, recovered from counter quantisation. */
    readonly slotsPerPartition: number;
    readonly slotConfidence: number;
    /** Threads per slot: an Apple GPU SIMD group. */
    readonly threadsPerSlot: number;
    readonly totalSlots: number;
    readonly totalThreads: number;
    readonly exclusiveCoverage: number;
    readonly meanOccupancy?: number;
    readonly meanAlu?: number;
    readonly meanReadGBs?: number;
    readonly meanWriteGBs?: number;
    /** Source-vs-retained extraction policy. GPU stage intervals are always
     * full-resolution; these fields apply only to the counter overlay. */
    readonly sourceCounterCount?: number;
    readonly retainedCounterCount?: number;
    readonly timestampStride?: number;
    readonly retainedSampleIntervalUs?: number;
    readonly frameSeries: readonly { name: string; points: { t: number; v: number }[] }[];
    /**
     * Every occupancy sample inside the representative advance: microseconds
     * from frame start, occupancy 0..1 on each partition, and the task that
     * owned the GPU. This is the finest true occupancy-over-time the hardware
     * reports -- one SIMD group out of 768 per partition, at the retained
     * counter cadence recorded alongside the report.
     * Slots are anonymous: the counter says how many were resident, never
     * which, because Apple exposes no per-wave thread trace.
     */
    readonly occupancyTrace: readonly {
      t: number; p: readonly number[]; label: string | null;
    }[];
  };
  /**
   * What the pass table is allowed to claim.
   *
   * Every capture so far has been *partially* scoped: label isolation on, but
   * restricted to one prefix, so a handful of buckets are true stages and the
   * rest are named after whichever stage happened to open their Metal encoder.
   * That distinction decided a 3.5 ms attribution question on the mini lane
   * once already, so it is a first-class field rather than a per-row flag a
   * renderer may forget to read.
   */
  readonly attribution: {
    readonly mode: "off" | "full" | "scoped";
    readonly isolatedPrefixes: readonly string[];
    readonly exactBuckets: number;
    readonly compositeBuckets: number;
    readonly exactMsPerFrame: number;
    readonly compositeMsPerFrame: number;
    /** Composite share of attributed interval time, 0..1. */
    readonly compositeShare: number;
    /** Costliest composite buckets -- the rows most likely to be misread. */
    readonly largestComposites: readonly {
      label: string; gpuMsPerFrame: number; reason: string;
    }[];
  };
  readonly passes: readonly PassCost[];
  readonly occupancyGrid: Record<string, (number | null)[]>;
  readonly shaders: readonly { name: string; samples: number; share: number; pipelines: number }[];
  readonly channels: readonly { channel: string; msPerFrame: number; count: number }[];
  readonly contention: readonly { process: string; gpuMs: number; intervals: number }[];
  readonly cpu: {
    readonly samples: number; readonly runningSamples: number;
    readonly threads: readonly { thread: string; samples: number; running: number }[];
    readonly flame: FlameNode;
    readonly hotLeaves: readonly { symbol: string; samples: number; share: number }[];
  };
  readonly timeline: {
    readonly frameStart: number; readonly frameDuration: number;
    readonly intervals: readonly Interval[];
  };
  readonly console: readonly string[];
}

// ---- Labels ----------------------------------------------------------------

const TRAILER = /\s{2,}\(\s*[^)]*\)\s*0x[0-9a-f]+\s*$/i;

/**
 * Split a GPU interval label into its encoder names. Instruments formats these
 * as `<command buffer>:<encoder>[ & <encoder>…]` plus a process/handle trailer,
 * and joins encoders it merged into one interval with " & ".
 */
export const parseEncoderLabel = (raw: string | null | undefined): {
  encoders: string[]; merged: boolean;
} => {
  if (!raw) return { encoders: ["(unlabelled)"], merged: false };
  const withoutTrailer = raw.replace(TRAILER, "").trim();
  const colon = withoutTrailer.indexOf(":");
  const body = colon === -1 ? withoutTrailer : withoutTrailer.slice(colon + 1);
  const encoders = body.split(" & ")
    .map((part) => part.trim().replace(/^Dawn_(?:Compute|Render|Blit)PassEncoder_/, "").trim())
    .filter((part) => part.length > 0);
  if (encoders.length === 0) return { encoders: ["(unlabelled)"], merged: false };
  return { encoders, merged: encoders.length > 1 };
};

const quantile = (sorted: readonly number[], q: number): number => {
  if (sorted.length === 0) return 0;
  const position = (sorted.length - 1) * q;
  const low = Math.floor(position);
  const high = Math.ceil(position);
  return low === high ? sorted[low] : sorted[low] + (sorted[high] - sorted[low]) * (position - low);
};

const mergeSpans = (spans: { start: number; end: number }[]): { start: number; end: number }[] => {
  spans.sort((left, right) => left.start - right.start);
  const merged: { start: number; end: number }[] = [];
  for (const span of spans) {
    const last = merged[merged.length - 1];
    if (last && span.start <= last.end) last.end = Math.max(last.end, span.end);
    else merged.push({ ...span });
  }
  return merged;
};

/**
 * Instruments may split one Metal encoder into several execution records when
 * the GPU suspends and resumes it. Keep the slices for cost/counter accounting,
 * but draw one logical stage spanning its first-to-last execution so the
 * timeline never presents continuations as duplicate stage invocations.
 */
export const coalesceEncoderSlices = (slices: readonly Interval[]): Interval[] => {
  const groups = new Map<string, Interval[]>();
  slices.forEach((slice, index) => {
    const key = slice.encoderId ?? `slice:${index}`;
    const group = groups.get(key);
    if (group) group.push(slice); else groups.set(key, [slice]);
  });
  return [...groups.values()].map((group) => {
    if (group.length === 1) return { ...group[0] };
    const first = group.reduce((best, value) => value.start < best.start ? value : best);
    const start = Math.min(...group.map((value) => value.start));
    const end = Math.max(...group.map((value) => value.start + value.duration));
    return { ...first, start, duration: end - start };
  }).sort((left, right) => left.start - right.start);
};

/** Index of the last span starting at or before `time`, or -1. */
const findSpan = (spans: readonly { start: number; end: number }[], time: number): number => {
  let low = 0;
  let high = spans.length - 1;
  let hit = -1;
  while (low <= high) {
    const mid = (low + high) >> 1;
    if (spans[mid].start <= time) { hit = mid; low = mid + 1; } else high = mid - 1;
  }
  return hit;
};

// ---- Frame segmentation ----------------------------------------------------

/** Existing, real work that is contractually first in every recurring solver
 * advance. Prefer it over a statistically tighter interior loop label so a
 * frame timeline starts at the frame's work rather than merely having the
 * correct period. */
export const GPU_FRAME_START_LABELS = [
  // The dense reference clears its step telemetry, then this is its first
  // compute pass. It fires once for the current-velocity extension and the
  // predicted extension uses a distinct "Uniform predicted" prefix.
  "Uniform Sec. 3.3 rho-prime and face authority",
  // Losasso has its own reduced ready-commit path and therefore never emits
  // the shared Power topology gate below. This validation is the first real
  // recurring dispatch in that backend's command buffer.
  "Losasso - validate ready row commit",
  "Open coupled topology ready-commit gate",
  // This recurring publication is submitted immediately before the topology
  // gate. Depending on where the attached encoder metadata stream warms up,
  // it can be the only once-per-advance start candidate with the modal count.
  "Publish deterministic fine-seed brick residency",
] as const;

/**
 * Recover advance boundaries from the interval stream. A pass that fires once
 * per advance has tight inter-arrival times; many passes qualify and they all
 * agree on the count, so the modal count across low-variance labels IS the
 * frame count, and the tightest label at that count is the anchor.
 */
export const detectFrames = (
  intervals: readonly Interval[],
  preferredLabels: readonly string[] = GPU_FRAME_START_LABELS,
): {
  boundaries: number[]; anchor: string;
} => {
  const occurrences = new Map<string, number[]>();
  const seenInvocations = new Set<string>();
  for (const interval of intervals) {
    for (const encoder of interval.encoders) {
      const invocation = interval.encoderId === undefined
        ? `${encoder}\0${interval.start}` : `${encoder}\0${interval.encoderId}`;
      if (seenInvocations.has(invocation)) continue;
      seenInvocations.add(invocation);
      const list = occurrences.get(encoder);
      if (list) list.push(interval.start); else occurrences.set(encoder, [interval.start]);
    }
  }
  interface Candidate { label: string; count: number; cv: number; starts: number[] }
  const candidates: Candidate[] = [];
  for (const [label, raw] of occurrences) {
    if (raw.length < 5) continue;
    const starts = [...raw].sort((left, right) => left - right);
    const deltas: number[] = [];
    for (let index = 1; index < starts.length; index += 1) {
      deltas.push(starts[index] - starts[index - 1]);
    }
    // One WindowServer preemption or shader-service hiccup must not make an
    // otherwise periodic frame anchor disappear. Estimate cadence from the
    // central 90% once the window is long enough, while retaining the full
    // occurrence list as the actual frame boundaries.
    const orderedDeltas = [...deltas].sort((left, right) => left - right);
    const trim = orderedDeltas.length >= 20 ? Math.floor(orderedDeltas.length * 0.05) : 0;
    const cadenceDeltas = trim > 0
      ? orderedDeltas.slice(trim, orderedDeltas.length - trim) : orderedDeltas;
    const mean = cadenceDeltas.reduce((sum, value) => sum + value, 0) / cadenceDeltas.length;
    if (!(mean > 0)) continue;
    const variance = cadenceDeltas.reduce(
      (sum, value) => sum + (value - mean) ** 2, 0,
    ) / cadenceDeltas.length;
    const cv = Math.sqrt(variance) / mean;
    // A semantic marker can legitimately occur in two distinct phases of one
    // advance (legacy uniform current/predicted extension labelling), making
    // the unsplit inter-arrival series alternate between a short and long
    // delta. Keep preferred markers long enough to phase-split them below.
    if (cv > 0.35 && !preferredLabels.includes(label)) continue;
    candidates.push({ label, count: starts.length, cv, starts });
  }
  if (candidates.length === 0) return { boundaries: [], anchor: "(no repeating pass found)" };
  const tally = new Map<number, number>();
  for (const candidate of candidates) tally.set(candidate.count, (tally.get(candidate.count) ?? 0) + 1);
  const frameCount = [...tally.entries()]
    .sort((left, right) => right[1] - left[1] || right[0] - left[0])[0][0];
  const atModalCount = candidates.filter((candidate) => candidate.count === frameCount);
  // A bounded metadata stream commonly clips one edge occurrence from most
  // interior labels while retaining the semantic start (or vice versa). Do
  // not reject the authored frame boundary merely because its count differs
  // from the statistical mode by one; clipping the first/last boundaries
  // below is explicitly designed for that edge condition.
  const anchor = preferredLabels
    .map((label) => {
      const candidate = candidates.find((value) => value.label === label);
      if (candidate === undefined) return undefined;
      if (Math.abs(candidate.count - frameCount) <= 1) return candidate;
      // Older uniform traces used the same authority label for the current and
      // predicted velocity-extension invocations. It is still the first work
      // in an advance, but occurs exactly twice per advance. Recover the first
      // phase of that repeated marker instead of falling back to an arbitrary
      // (and usually interior) once-per-frame kernel. New captures give the
      // predicted invocation its own prefix, but retained traces must remain
      // reducible.
      const repeats = Math.round(candidate.count / frameCount);
      if (repeats < 2 || repeats > 4
        || Math.abs(candidate.count - repeats * frameCount) > 1) return undefined;
      return { ...candidate, count: Math.ceil(candidate.count / repeats),
        starts: candidate.starts.filter((_start, index) => index % repeats === 0) };
    })
    .find((candidate) => candidate !== undefined)
    ?? atModalCount.sort((left, right) => left.cv - right.cv)[0];
  return { boundaries: anchor.starts, anchor: anchor.label };
};

// ---- Flame graph -----------------------------------------------------------

const foldStacks = (stacks: readonly (readonly string[])[]): FlameNode => {
  const root: FlameNode = { name: "all", value: 0, children: [] };
  for (const leafFirst of stacks) {
    let node = root;
    node.value += 1;
    for (let index = leafFirst.length - 1; index >= 0; index -= 1) {
      const name = leafFirst[index];
      let child = node.children.find((candidate) => candidate.name === name);
      if (!child) { child = { name, value: 0, children: [] }; node.children.push(child); }
      child.value += 1;
      node = child;
    }
  }
  const prune = (node: FlameNode): void => {
    node.children = node.children.filter((child) => child.value >= 2)
      .sort((left, right) => right.value - left.value);
    for (const child of node.children) prune(child);
  };
  prune(root);
  return root;
};


/**
 * Recover the GPU's scheduler-slot count per partition from the quantisation of
 * the occupancy counter. Readings land on exact multiples of 100/slots, so the
 * divisor that explains the most readings is the hardware's slot count.
 * Verified on an M1 Max: 768 slots per partition explains 98.3% of the
 * commonest readings, and 4 partitions x 768 slots x 32 threads per SIMD group
 * reproduces Apple's published 98,304 concurrent threads exactly.
 */
export const detectPartitionSlots = (histogram: ReadonlyMap<number, number>): {
  slots: number; confidence: number;
} => {
  // Weighted accumulation leaves a long tail of one-off fractional readings
  // that fit no quantum. Score only the readings that actually recur, exactly
  // as a human would eyeballing the histogram.
  const common = [...histogram.entries()]
    .sort((left, right) => right[1] - left[1])
    .slice(0, 60);
  const total = common.reduce((sum, [, count]) => sum + count, 0);
  if (total === 0) return { slots: 0, confidence: 0 };

  const candidates = [64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072];
  const scored = candidates.map((slots) => {
    const quantum = 100 / slots;
    let hit = 0;
    for (const [value, count] of common) {
      const units = value / quantum;
      if (Math.abs(units - Math.round(units)) < 0.02) hit += count;
    }
    return { slots, confidence: hit / total };
  });
  // Every multiple of a true quantum is also a multiple of its subdivisions, so
  // the coarsest divisor that still explains the readings is the real one.
  const bestConfidence = Math.max(...scored.map((entry) => entry.confidence));
  return scored.find((entry) => entry.confidence >= bestConfidence - 0.02)
    ?? { slots: 0, confidence: 0 };
};

// ---- Report ----------------------------------------------------------------

export interface BuildFrameReportInput {
  readonly tables: Record<string, string>;
  readonly lane: string;
  readonly environment: Record<string, string>;
  readonly workUnit?: "advance" | "frame";
  readonly title?: string;
  readonly traced?: { simulationWall_ms?: number; steps?: number };
  readonly baseline?: { simulationWall_ms?: number; steps?: number };
  /**
   * The traced process, when the profiler attached to one it spawned itself.
   * Identifying us by pid beats inferring it from whoever encoded the most:
   * an attached recording only sees the encoders created inside its window, so
   * a busy neighbour can out-encode the solver.
   */
  readonly tracedPid?: number;
  /** Preferred once-per-frame Metal labels for non-simulation workloads. */
  readonly frameStartLabels?: readonly string[];
  /** Occupancy counter for the pipeline being measured. Compute workloads use
   * Compute Occupancy; render passes should use Fragment Occupancy. */
  readonly occupancyCounterName?: "Compute Occupancy" | "Fragment Occupancy";
  /** Reduce the labelled counter window to one complete representative
   * advance. The recorder still needs a multi-second window for Metal encoder
   * metadata and hardware counters to warm up. */
  readonly singleFrame?: boolean;
  /** Retain this many adjacent representative advances. Supersedes
   * `singleFrame`; useful when comparing frame-to-frame stage stability rather
   * than only the median-shaped advance. */
  readonly frameLimit?: number;
  /** Select advance 1, bounded by the semantic start of advance 2. Unlike
   * `singleFrame`, this is a literal cold-frame capture, not a representative
   * steady-state sample. */
  readonly firstFrame?: boolean;
  readonly counterExtraction?: {
    readonly sourceCounterCount: number;
    readonly retainedCounterCount: number;
    readonly timestampStride: number;
  };
}

const GRID_BINS = 90;
/** Apple GPU SIMD group width. */
const SIMD_WIDTH = 32;

export const buildFrameReport = async (input: BuildFrameReportInput): Promise<FrameReport> => {
  const labelIsolationEnabled = input.environment.FLUID_GPU_ISOLATE_PASS_LABELS === "1";
  // The broker normalises prefixes with exactly this function before matching,
  // so the report must use the same one or a prefix with punctuation would
  // isolate on the GPU and fail to match here -- silently demoting real stages
  // to composite buckets.
  const isolatedPrefixes = (input.environment.FLUID_GPU_ISOLATE_PASS_LABEL_PREFIXES ?? "")
    .split(",").map((prefix) => xctraceSafeComputeLabel(prefix.trim()))
    .filter((prefix) => prefix.length > 0);
  const attributionMode: "off" | "full" | "scoped" = !labelIsolationEnabled ? "off"
    : isolatedPrefixes.length === 0 ? "full" : "scoped";
  /** Why a bucket cannot be read as its named stage, or undefined when it can. */
  const compositeReason = (label: string, merged: boolean): string | undefined => {
    if (merged) {
      return "Instruments merged several Metal encoders into this interval;"
        + " the cost covers all of them";
    }
    if (attributionMode === "off") {
      return "label isolation was off: this is the first label of its Metal encoder"
        + " and the cost includes every stage encoded until the next pass boundary";
    }
    if (attributionMode === "scoped"
      && !isolatedPrefixes.some((prefix) => label.startsWith(prefix))) {
      return `outside the isolated prefix scope (${isolatedPrefixes.join(", ")}):`
        + " this is the first label of its Metal encoder and the cost includes"
        + " every stage encoded until the next pass boundary";
    }
    return undefined;
  };
  // ---- Encoder identity ----
  // The GPU track's own label text is unusable as a grouping key: Instruments
  // concatenates the labels of every encoder it drew as one interval, so the
  // same work yields a different string whenever the merge grouping shifts,
  // and 128 real tasks explode into thousands of one-off keys. The CPU-side
  // encoder list is stable, so encoder-id is the join key and its label is the
  // authority. Every GPU interval of ours resolves through it.
  const encoderLabels = new Map<string, string>();
  const encoderProcess = new Map<string, string>();
  const processEncoderTally = new Map<string, number>();
  if (input.tables.encoders) {
    for await (const row of readTraceRows(input.tables.encoders)) {
      const id = String(row["encoder-id"] ?? "");
      if (!id) continue;
      encoderLabels.set(id, String(row["encoder-label"] ?? "(unlabelled)"));
      const process = String(row.process ?? "(unknown)");
      encoderProcess.set(id, process);
      processEncoderTally.set(process, (processEncoderTally.get(process) ?? 0) + 1);
    }
  }
  const busiestEncoder = [...processEncoderTally.entries()]
    .sort((left, right) => right[1] - left[1])[0]?.[0] ?? "(unknown)";
  // Instruments spells a process "node (69273)", so the pid identifies us even
  // when the encoder metadata stream never carried a row of ours.
  const pidSuffix = input.tracedPid === undefined ? undefined : `(${input.tracedPid})`;
  const isOurs = (process: string): boolean => (pidSuffix === undefined
    ? process === busiestEncoder : process.endsWith(pidSuffix));
  let ourProcess = pidSuffix === undefined
    ? busiestEncoder : [...processEncoderTally.keys()].find(isOurs) ?? "(unknown)";

  const ours: Interval[] = [];
  const diagnosticBlits: Interval[] = [];
  const byProcess = new Map<string, Interval[]>();
  const intervalTally = new Map<string, number>();
  // Ours, but arriving before the encoder metadata stream warmed up, so with no
  // usable grouping key. Reported rather than dropped in silence -- and kept out
  // of the contention set, since contending with ourselves is not contention.
  let unlabelledCount = 0;
  let unlabelledMicroseconds = 0;
  for await (const row of readTraceRows(input.tables["gpu-intervals"])) {
    const id = String(row["encoder-id"] ?? "");
    const canonical = encoderLabels.get(id);
    const owner = canonical !== undefined
      ? encoderProcess.get(id) ?? "(unknown)" : String(row.process ?? "(unknown)");
    intervalTally.set(owner, (intervalTally.get(owner) ?? 0) + 1);
    const { encoders, merged } = parseEncoderLabel(
      canonical !== undefined ? canonical : (row["event-label"] as string),
    );
    const interval: Interval = {
      ...(id ? { encoderId: id } : {}),
      start: timestampMicroseconds(row.start as string),
      duration: durationMicroseconds(row.duration as string),
      label: encoders.join(" · "),
      encoders,
      channel: String(row["channel-name"] ?? "?"),
      merged,
    };
    if (isOurs(owner)) {
      if (ourProcess === "(unknown)") ourProcess = owner;
      if (canonical !== undefined) {
        const encoderIsolationBlit = input.environment.FLUID_GPU_ISOLATE_PASS_ENCODERS === "1"
          && interval.encoders.every((label) => /^Blit Command \d+$/.test(label));
        if (encoderIsolationBlit) diagnosticBlits.push(interval); else ours.push(interval);
      }
      else { unlabelledCount += 1; unlabelledMicroseconds += interval.duration; }
    } else {
      const list = byProcess.get(owner);
      if (list) list.push(interval); else byProcess.set(owner, [interval]);
    }
  }
  ours.sort((left, right) => left.start - right.start);
  diagnosticBlits.sort((left, right) => left.start - right.start);
  if (ours.length === 0) {
    const seen = [...intervalTally.entries()].sort((left, right) => right[1] - left[1])
      .slice(0, 5).map(([process, count]) => `${process} ${count}`).join(", ");
    throw new Error("no GPU intervals resolved to the traced process"
      + `${pidSuffix === undefined ? "" : ` node ${input.tracedPid}`}.`
      + ` The trace holds ${[...intervalTally.values()].reduce((sum, n) => sum + n, 0)}`
      + ` intervals from ${seen || "nobody"}`
      + (unlabelledCount > 0
        ? `, plus ${unlabelledCount} of ours that the encoder metadata stream never labelled.`
          + " The GPU work was captured, but the Metal application encoder stream was absent;"
          + " retain this trace and retry with a longer --counter-seconds window"
        : ". The counter window missed the stepping phase; raise --steps or move"
          + " --counter-warmup earlier"));
  }

  let literalFrameUsesCommandBufferCompletion = false;
  const detected = input.firstFrame ? await (async () => {
    // Backends can emit different semantic starts in the same advance. Select
    // the first preferred label that is actually present; combining every
    // preferred label would mistake the next fallback marker for advance 2.
    const preferredLabel = GPU_FRAME_START_LABELS.find((label) => ours.some(
      (interval) => interval.encoders.includes(label),
    ));
    const starts: number[] = [];
    const seen = new Set<string>();
    for (const interval of ours) {
      if (preferredLabel === undefined || !interval.encoders.includes(preferredLabel)) continue;
      const invocation = interval.encoderId ?? `${interval.label}\0${interval.start}`;
      if (seen.has(invocation)) continue;
      seen.add(invocation);
      starts.push(interval.start);
    }
    starts.sort((left, right) => left - right);
    if (starts.length >= 2) {
      return { boundaries: starts.slice(0, 2), anchor: preferredLabel! };
    }
    if (starts.length === 1 && input.tables["command-buffer-submissions"]
      && input.tables["command-buffer-completed"]) {
      const completionByCommandBuffer = new Map<string, number>();
      for await (const row of readTraceRows(input.tables["command-buffer-completed"])) {
        const id = String(row["cmdbuffer-id"] ?? "");
        if (id) completionByCommandBuffer.set(id,
          timestampMicroseconds(row.timestamp as string));
      }
      const candidates: { start: number; end: number; encoders: number }[] = [];
      for await (const row of readTraceRows(input.tables["command-buffer-submissions"])) {
        if (!isOurs(String(row.process ?? "(unknown)"))) continue;
        const id = String(row["cmdbuffer-id"] ?? "");
        const start = timestampMicroseconds(row.start as string);
        const end = completionByCommandBuffer.get(id);
        const encoders = Number(row["num-encoders"] ?? 0);
        if (end !== undefined && start <= starts[0] && end > starts[0]) {
          candidates.push({ start, end, encoders });
        }
      }
      const frameCommandBuffer = candidates.sort((left, right) =>
        right.encoders - left.encoders || right.start - left.start)[0];
      if (frameCommandBuffer && frameCommandBuffer.encoders > 1) {
        literalFrameUsesCommandBufferCompletion = true;
        return { boundaries: [starts[0], frameCommandBuffer.end],
          anchor: preferredLabel! };
      }
    }
    throw new Error(`literal first-frame capture found ${starts.length} semantic frame starts and`
      + " no enclosing labelled Metal command-buffer completion");
  })() : detectFrames(ours, input.frameStartLabels ?? GPU_FRAME_START_LABELS);
  const anchor = detected.anchor;
  // The first recurring octree command is a buffer clear immediately followed
  // by the ready-commit compute gate. Encoder isolation puts both that real
  // setup clear and its diagnostic separator in the preceding Metal blit.
  // Shift each semantic gate boundary back to that nearest blit so time zero is
  // the first observed GPU activity of the advance, not the first compute row.
  const semanticFrameStart = GPU_FRAME_START_LABELS.some((label) => label === anchor);
  const boundaries = semanticFrameStart ? detected.boundaries.map((gateStart, index) => {
    if (literalFrameUsesCommandBufferCompletion && index === 1) return gateStart;
    let low = 0;
    let high = diagnosticBlits.length - 1;
    let preceding = gateStart;
    while (low <= high) {
      const middle = (low + high) >> 1;
      if (diagnosticBlits[middle].start < gateStart) {
        preceding = diagnosticBlits[middle].start;
        low = middle + 1;
      } else high = middle - 1;
    }
    return preceding;
  }) : detected.boundaries;
  // Clip the first and last anchors: those frames are cut by the recording and
  // would drag every per-frame average down.
  const completeBoundaries = boundaries.length >= 4 ? boundaries.slice(1, -1) : boundaries;
  const completeDurations: number[] = [];
  for (let index = 1; index < completeBoundaries.length; index += 1) {
    completeDurations.push(completeBoundaries[index] - completeBoundaries[index - 1]);
  }
  const representativeBoundary = (): number => {
    if (completeDurations.length === 0) return 0;
    const sorted = [...completeDurations].sort((left, right) => left - right);
    const median = quantile(sorted, 0.5);
    const middle = (completeDurations.length - 1) / 2;
    // Counter streams start and stop independently. Frames near either edge
    // can be perfectly complete in the encoder stream while missing a hardware
    // partition (the 2026-07-29 trace lost ring 3 in its final quarter). Pick
    // only from the middle third, then choose the most duration-representative
    // member of that safe interior.
    const firstInterior = Math.floor(completeDurations.length / 3);
    const afterInterior = Math.max(firstInterior + 1,
      Math.ceil(2 * completeDurations.length / 3));
    return completeDurations.map((duration, index) => ({
      index,
      distance: Math.abs(duration - median),
      middleDistance: Math.abs(index - middle),
    })).filter((entry) => entry.index >= firstInterior && entry.index < afterInterior)
      .sort((left, right) => left.distance - right.distance
      || left.middleDistance - right.middleDistance)[0].index;
  };
  const selectedBoundary = input.firstFrame ? 0 : representativeBoundary();
  const requestedFrames = input.firstFrame ? 1
    : input.frameLimit !== undefined
      ? Math.max(1, Math.floor(input.frameLimit))
      : input.singleFrame ? 1 : undefined;
  const usable = requestedFrames !== undefined && completeBoundaries.length >= 2
    ? (() => {
      const availableFrames = completeBoundaries.length - 1;
      const retainedFrames = Math.min(requestedFrames, availableFrames);
      // Keep the representative inside the requested window, biased one frame
      // earlier for an even-sized window so a two-frame report shows the lead-in
      // and the representative rather than the representative and an arbitrary
      // tail neighbour.
      const wantedStart = selectedBoundary - Math.floor(retainedFrames / 2);
      const start = Math.max(0, Math.min(wantedStart, availableFrames - retainedFrames));
      return completeBoundaries.slice(start, start + retainedFrames + 1);
    })()
    : completeBoundaries;
  const windowStart = usable[0] ?? ours[0].start;
  const windowEnd = usable[usable.length - 1]
    ?? ours[ours.length - 1].start + ours[ours.length - 1].duration;
  const frameCount = Math.max(usable.length - 1, 1);
  const frameDurations: number[] = [];
  for (let index = 1; index < usable.length; index += 1) {
    frameDurations.push(usable[index] - usable[index - 1]);
  }
  frameDurations.sort((left, right) => left - right);

  const windowed = ours.filter((interval) => interval.start >= windowStart
    && interval.start < windowEnd);
  const windowedDiagnosticBlits = diagnosticBlits.filter((interval) =>
    interval.start >= windowStart && interval.start < windowEnd);
  const allWindowed = [...windowed, ...windowedDiagnosticBlits]
    .sort((left, right) => left.start - right.start);
  const diagnosticBlitMicroseconds = windowedDiagnosticBlits.reduce((sum, interval) =>
    sum + interval.duration, 0);
  const logicalWindow = coalesceEncoderSlices(windowed);
  const logicalEncoderCount = logicalWindow.length;
  const logicalPassCount = logicalWindow.reduce((sum, interval) =>
    sum + interval.encoders.length, 0);
  const continuationSlices = Math.max(0, windowed.length - logicalEncoderCount);

  // ---- Per-frame series ----
  // Averages cannot answer "is this a whole advance" or "does advance 480 look
  // like advance 20". Every frame is therefore accumulated separately: same
  // boundaries as the averages, so the two can never disagree.
  interface FrameAccumulator {
    start: number; end: number; busy: number; encoders: number; passes: number;
    tasks: Map<string, number>; occupancySum: number; occupancySamples: number;
    invocations: Set<string>;
  }
  const frameAccumulators: FrameAccumulator[] = [];
  for (let index = 1; index < usable.length; index += 1) {
    frameAccumulators.push({
      start: usable[index - 1], end: usable[index], busy: 0, encoders: 0, passes: 0,
      tasks: new Map(), occupancySum: 0, occupancySamples: 0, invocations: new Set(),
    });
  }
  /** Index of the frame containing `time`, or -1 outside the analysed window. */
  const frameAt = (time: number): number => {
    const index = findSpan(frameAccumulators, time);
    return index >= 0 && time < frameAccumulators[index].end ? index : -1;
  };
  for (const interval of windowed) {
    const index = frameAt(interval.start);
    if (index < 0) continue;
    const frame = frameAccumulators[index];
    const invocation = interval.encoderId ?? `${interval.label}\0${interval.start}`;
    if (!frame.invocations.has(invocation)) {
      frame.invocations.add(invocation);
      frame.encoders += 1;
      frame.passes += interval.encoders.length;
    }
    frame.tasks.set(interval.label, (frame.tasks.get(interval.label) ?? 0) + interval.duration);
  }

  // ---- Contention and exclusive windows ----
  const foreignSpans: { start: number; end: number }[] = [];
  const contentionTally = new Map<string, { gpuMs: number; intervals: number }>();
  for (const [process, list] of byProcess) {
    for (const interval of list) {
      if (interval.start + interval.duration < windowStart || interval.start >= windowEnd) continue;
      foreignSpans.push({ start: interval.start, end: interval.start + interval.duration });
      const entry = contentionTally.get(process) ?? { gpuMs: 0, intervals: 0 };
      entry.gpuMs += interval.duration / 1e3;
      entry.intervals += 1;
      contentionTally.set(process, entry);
    }
  }
  const foreign = mergeSpans(foreignSpans);
  const contendedMicroseconds = (start: number, end: number): number => {
    let total = 0;
    let index = Math.max(findSpan(foreign, start), 0);
    for (; index < foreign.length && foreign[index].start < end; index += 1) {
      total += Math.max(0, Math.min(end, foreign[index].end) - Math.max(start, foreign[index].start));
    }
    return total;
  };

  // ---- Per-pass GPU cost ----
  interface PassAccumulator {
    us: number; invocations: Set<string>; merged: boolean; exclusiveUs: number;
    counters: Map<string, { sum: number; n: number }>;
    partitionSum: number[]; partitionCount: number[];
  }
  const passes = new Map<string, PassAccumulator>();
  const intervalMicroseconds = windowed.reduce((sum, interval) => sum + interval.duration, 0);
  const stageBusySpans = mergeSpans(windowed.map((interval) => ({
    start: Math.max(windowStart, interval.start),
    end: Math.min(windowEnd, interval.start + interval.duration),
  })));
  const busySpans = mergeSpans(allWindowed.map((interval) => ({
    start: Math.max(windowStart, interval.start),
    end: Math.min(windowEnd, interval.start + interval.duration),
  })));
  const stageBusyMicroseconds = stageBusySpans.reduce((sum, span) =>
    sum + span.end - span.start, 0);
  const busyMicroseconds = busySpans.reduce((sum, span) => sum + span.end - span.start, 0);
  for (const frame of frameAccumulators) {
    frame.busy = busySpans.reduce((sum, span) => sum
      + Math.max(0, Math.min(frame.end, span.end) - Math.max(frame.start, span.start)), 0);
  }
  let mergedMicroseconds = 0;
  let exclusiveMicroseconds = 0;
  for (const interval of windowed) {
    if (interval.merged) mergedMicroseconds += interval.duration;
    const end = interval.start + interval.duration;
    const exclusive = interval.duration - contendedMicroseconds(interval.start, end);
    exclusiveMicroseconds += Math.max(0, exclusive);
    const entry = passes.get(interval.label)
      ?? {
        us: 0, invocations: new Set(), merged: interval.merged, exclusiveUs: 0,
        counters: new Map(),
        partitionSum: [], partitionCount: [],
      };
    entry.us += interval.duration;
    entry.invocations.add(interval.encoderId ?? `${interval.label}\0${interval.start}`);
    entry.exclusiveUs += Math.max(0, exclusive);
    passes.set(interval.label, entry);
  }

  // ---- GPU gaps ----
  const gaps: number[] = [];
  let cursor = windowStart;
  for (const interval of allWindowed) {
    if (interval.start > cursor) gaps.push(interval.start - cursor);
    cursor = Math.max(cursor, interval.start + interval.duration);
  }
  if (windowEnd > cursor) gaps.push(windowEnd - cursor);
  const gapMicroseconds = gaps.reduce((sum, value) => sum + value, 0);
  const wallMicroseconds = Math.max(windowEnd - windowStart, 1);

  const channelTotals = new Map<string, { us: number; count: number }>();
  for (const interval of allWindowed) {
    const entry = channelTotals.get(interval.channel) ?? { us: 0, count: 0 };
    entry.us += interval.duration;
    entry.count += 1;
    channelTotals.set(interval.channel, entry);
  }

  // ---- Representative frame ----
  let timelineStart = windowStart;
  let representativeIndex = 0;
  let timelineDuration = frameDurations.length > 0
    ? quantile(frameDurations, 0.5) : wallMicroseconds;
  if (usable.length >= 2) {
    const median = quantile(frameDurations, 0.5);
    let best = Number.POSITIVE_INFINITY;
    for (let index = 1; index < usable.length; index += 1) {
      const duration = usable[index] - usable[index - 1];
      if (Math.abs(duration - median) < best) {
        best = Math.abs(duration - median);
        timelineStart = usable[index - 1];
        timelineDuration = duration;
        representativeIndex = index - 1;
      }
    }
  }
  const timelineSlices = ours.filter((interval) => interval.start >= timelineStart
    && interval.start < timelineStart + timelineDuration);

  // ---- GPU counters -> per-pass occupancy ----
  const counterNames = new Map<string, { name: string; type: string; interval: number }>();
  if (input.tables["counter-info"]) {
    try {
      for await (const row of readTraceRows(input.tables["counter-info"])) {
        counterNames.set(String(row["counter-id"]), {
          name: String(row.name ?? ""), type: String(row.type ?? ""),
          interval: Number(String(row["sample-interval"] ?? "10000").replace(/,/g, "")),
        });
      }
    } catch { /* optional */ }
  }
  // Every counter is retained: the export already paid for them, and which one
  // explains a task is not knowable in advance.
  const OCCUPANCY = input.occupancyCounterName ?? "Compute Occupancy";
  const globalCounters = new Map<string, { sum: number; n: number }>();
  const devicePartitionSum: number[] = [];
  const devicePartitionCount: number[] = [];
  /**
   * Occupancy readings are quantised: on an M1 Max 98.3% of the commonest
   * values are exact integer multiples of 100/768, i.e. the counter resolves
   * one scheduler slot out of 768 per partition. Detecting the divisor from
   * the data rather than hardcoding it keeps this correct on other parts
   * (M1 Pro has half the partitions, Ultra twice).
   */
  const occupancyHistogram = new Map<number, number>();
  const frameSeriesRaw = new Map<string, { t: number; v: number }[]>();
  const occupancyTrace: { t: number; p: number[]; label: string | null }[] = [];
  let counterSampleCount = 0;
  let usableSampleCount = 0;

  // Interval lookup for attribution: our windowed intervals, sorted.
  const spans = windowed.map((interval) => ({
    start: interval.start, end: interval.start + interval.duration, label: interval.label,
  }));

  if (input.tables["counter-value"] && counterNames.size > 0) {
    // Ring buffers report the same counter from several accelerator slices;
    // average them per (timestamp, counter).
    // Instruments splits every counter across several hardware streams
    // (`ring-buffer-index`). They are NOT redundant copies: inside a
    // wide dispatch all four read within 1-2% of each other, while inside a
    // single-workgroup dispatch stream 0 sustains 1.00% against 0.12-0.17% on
    // the rest for milliseconds at a time -- far too long to be a sampling
    // lag. They behave as the four GPU partitions, so their spread is the
    // placement of work across the machine, and their mean is the device
    // figure.
    let pendingTime = -1;
    const pending = new Map<string, { sum: number; n: number; rings: number[] }>();
    const flush = (): void => {
      if (pendingTime < 0 || pending.size === 0) return;
      const time = pendingTime;
      if (time >= windowStart && time < windowEnd) {
        counterSampleCount += 1;
        const index = findSpan(spans, time);
        const span = index >= 0 ? spans[index] : undefined;
        const inOurWork = span !== undefined && time < span.end;
        const foreignIndex = findSpan(foreign, time);
        const contended = foreignIndex >= 0 && time < foreign[foreignIndex].end;
        if (inOurWork && !contended) {
          usableSampleCount += 1;
          const accumulator = passes.get(span.label);
          const frame = frameAccumulators[frameAt(time)];
          for (const [name, value] of pending) {
            const mean = value.sum / value.n;
            const global = globalCounters.get(name) ?? { sum: 0, n: 0 };
            global.sum += mean; global.n += 1;
            globalCounters.set(name, global);
            if (frame && name === OCCUPANCY) {
              frame.occupancySum += mean; frame.occupancySamples += 1;
            }
            if (accumulator) {
              const bucket = accumulator.counters.get(name) ?? { sum: 0, n: 0 };
              bucket.sum += mean; bucket.n += 1;
              accumulator.counters.set(name, bucket);
            }
            if (name === OCCUPANCY) {
              value.rings.forEach((reading) => {
                if (reading === undefined || reading <= 0) return;
                occupancyHistogram.set(reading, (occupancyHistogram.get(reading) ?? 0) + 1);
              });
              value.rings.forEach((reading, ring) => {
                if (reading === undefined) return;
                devicePartitionSum[ring] = (devicePartitionSum[ring] ?? 0) + reading;
                devicePartitionCount[ring] = (devicePartitionCount[ring] ?? 0) + 1;
                if (!accumulator) return;
                accumulator.partitionSum[ring] = (accumulator.partitionSum[ring] ?? 0) + reading;
                accumulator.partitionCount[ring] = (accumulator.partitionCount[ring] ?? 0) + 1;
              });
            }
          }
        }
        if (inOurWork && !contended
          && time >= timelineStart && time < timelineStart + timelineDuration) {
          const occupancy = pending.get(OCCUPANCY);
          if (occupancy && occupancy.rings.length > 0) {
            const index = findSpan(spans, time);
            occupancyTrace.push({
              t: time - timelineStart,
              p: occupancy.rings.map((reading) => (reading ?? 0) / 100),
              label: index >= 0 && time < spans[index].end ? spans[index].label : null,
            });
          }
          for (const [name, value] of pending) {
            if (name !== OCCUPANCY && name !== "ALU Utilization"
              && name !== "GPU Last Level Cache Utilization") continue;
            const series = frameSeriesRaw.get(name) ?? [];
            series.push({ t: time - timelineStart, v: value.sum / value.n });
            frameSeriesRaw.set(name, series);
          }
        }
      }
      pending.clear();
    };
    for await (const row of readTraceRows(input.tables["counter-value"])) {
      const time = timestampMicroseconds(row.timestamp as string);
      if (time !== pendingTime) { flush(); pendingTime = time; }
      const meta = counterNames.get(String(row["counter-id"]));
      if (!meta) continue;
      const value = Number(String(row.value ?? "0").replace(/,/g, ""));
      if (!Number.isFinite(value)) continue;
      const bucket = pending.get(meta.name) ?? { sum: 0, n: 0, rings: [] };
      bucket.sum += value; bucket.n += 1;
      const ring = Number(String(row["ring-buffer-index"] ?? "0").replace(/,/g, ""));
      if (Number.isInteger(ring) && ring >= 0 && ring < 16) bucket.rings[ring] = value;
      pending.set(meta.name, bucket);
    }
    flush();
  }

  // ---- Shader profiler ----
  const shaderTally = new Map<string, { samples: number; pipelines: Set<string> }>();
  const shadersByPass = new Map<string, Map<string, number>>();
  if (input.tables["shader-profile"]) {
    try {
      for await (const row of readTraceRows(input.tables["shader-profile"])) {
        const label = row.label ? String(row.label).replace(/^Dawn_ShaderModule_/, "") : undefined;
        if (!label) continue;
        const time = timestampMicroseconds(row.start as string);
        const entry = shaderTally.get(label) ?? { samples: 0, pipelines: new Set<string>() };
        entry.samples += 1;
        entry.pipelines.add(String(row["pso-label"] ?? "?").replace(/^Dawn_ComputePipeline_/, ""));
        shaderTally.set(label, entry);
        const index = findSpan(spans, time);
        if (index >= 0 && time < spans[index].end) {
          const perPass = shadersByPass.get(spans[index].label) ?? new Map<string, number>();
          perPass.set(label, (perPass.get(label) ?? 0) + 1);
          shadersByPass.set(spans[index].label, perPass);
        }
      }
    } catch { /* optional */ }
  }

  // ---- CPU sampling ----
  const stacks: string[][] = [];
  const threadTally = new Map<string, { samples: number; running: number }>();
  const leafTally = new Map<string, number>();
  let samples = 0;
  let runningSamples = 0;
  try {
    for await (const row of readTraceRows(input.tables["time-profile"])) {
      if (String(row.process ?? "") !== ourProcess) continue;
      const time = timestampMicroseconds(row.time as string);
      if (time < windowStart || time >= windowEnd) continue;
      samples += 1;
      const thread = String(row.thread ?? "?").replace(/\s*\(node, pid: \d+\)/, "");
      const entry = threadTally.get(thread) ?? { samples: 0, running: 0 };
      entry.samples += 1;
      if (String(row["thread-state"] ?? "") === "Running") { entry.running += 1; runningSamples += 1; }
      threadTally.set(thread, entry);
      const frames = row.frames as string[] | undefined;
      if (frames && frames.length > 0) {
        stacks.push(frames);
        leafTally.set(frames[0], (leafTally.get(frames[0]) ?? 0) + 1);
      }
    }
  } catch { /* optional */ }

  // ---- Assemble ----
  const counterMean = (map: Map<string, { sum: number; n: number }>, name: string):
  number | undefined => {
    const entry = map.get(name);
    return entry && entry.n > 0 ? entry.sum / entry.n : undefined;
  };
  /** Percentage counters arrive as 0..100; the report speaks in 0..1. */
  const asFraction = (value: number | undefined): number | undefined => (value === undefined
    ? undefined : value / 100);
  // Instruments already reports the bandwidth counters as a rate in GB/s --
  // observed peak ~78 on a part with ~400 GB/s of memory bandwidth -- so these
  // pass through rather than being divided by the sample interval.
  const toGigabytesPerSecond = (value: number | undefined): number | undefined => value;

  const slotModel = detectPartitionSlots(occupancyHistogram);
  const passList: PassCost[] = [...passes.entries()].map(([label, entry]) => {
    const limiterCandidates = [...entry.counters.entries()]
      .filter(([name]) => name.endsWith("Limiter"))
      .sort((left, right) => (right[1].sum / right[1].n) - (left[1].sum / left[1].n));
    const shaders = [...(shadersByPass.get(label) ?? new Map<string, number>()).entries()]
      .map(([name, count]) => ({ name, samples: count }))
      .sort((left, right) => right.samples - left.samples).slice(0, 12);
    const counterSamples = entry.counters.get(OCCUPANCY)?.n ?? 0;
    const reason = compositeReason(label, entry.merged);
    return {
      label,
      exactAttribution: reason === undefined,
      ...(reason === undefined ? {} : { compositeReason: reason }),
      callsPerFrame: entry.invocations.size / frameCount,
      gpuMsPerFrame: entry.us / 1e3 / frameCount,
      meanMicroseconds: entry.us / Math.max(entry.invocations.size, 1),
      share: intervalMicroseconds > 0 ? entry.us / intervalMicroseconds : 0,
      merged: entry.merged,
      exclusiveShare: entry.us > 0 ? entry.exclusiveUs / entry.us : 0,
      counterSamples,
      ...(() => {
        const partitions = entry.partitionSum
          .map((sum, ring) => sum / Math.max(entry.partitionCount[ring] ?? 1, 1) / 100);
        if (partitions.length === 0) return {};
        const mean = partitions.reduce((sum, value) => sum + value, 0) / partitions.length;
        return {
          partitions,
          imbalance: mean > 0 ? Math.max(...partitions) / mean : 1,
        };
      })(),
      counters: Object.fromEntries([...entry.counters.entries()]
        .map(([name, bucket]) => [name, bucket.sum / bucket.n])),
      ...(() => {
        const occ = counterMean(entry.counters, OCCUPANCY);
        if (occ === undefined || slotModel.slots === 0 || slotModel.confidence < 0.9) return {};
        const slots = (occ / 100) * slotModel.slots * devicePartitionCount.length;
        return { residentSlots: slots, residentThreads: slots * SIMD_WIDTH };
      })(),
      occupancy: asFraction(counterMean(entry.counters, OCCUPANCY)),
      alu: asFraction(counterMean(entry.counters, "ALU Utilization")),
      readGBs: toGigabytesPerSecond(counterMean(entry.counters, "GPU Read Bandwidth")),
      writeGBs: toGigabytesPerSecond(counterMean(entry.counters, "GPU Write Bandwidth")),
      limiter: limiterCandidates.length > 0 && limiterCandidates[0][1].n > 0
        ? `${limiterCandidates[0][0].replace(/ Limiter$/, "")}`
          + ` ${(limiterCandidates[0][1].sum / limiterCandidates[0][1].n).toFixed(0)}%`
        : undefined,
      shaders,
    };
  }).sort((left, right) => right.gpuMsPerFrame - left.gpuMsPerFrame);
  const exactPassList = passList.filter((pass) => pass.exactAttribution);
  const compositePassList = passList.filter((pass) => !pass.exactAttribution);
  const exactMicroseconds = exactPassList.reduce((sum, pass) =>
    sum + pass.gpuMsPerFrame * 1e3 * frameCount, 0);
  const compositeMicroseconds = Math.max(0, intervalMicroseconds - exactMicroseconds);
  const attribution: FrameReport["attribution"] = {
    mode: attributionMode,
    isolatedPrefixes,
    exactBuckets: exactPassList.length,
    compositeBuckets: compositePassList.length,
    exactMsPerFrame: exactMicroseconds / 1e3 / frameCount,
    compositeMsPerFrame: compositeMicroseconds / 1e3 / frameCount,
    compositeShare: compositeMicroseconds / Math.max(intervalMicroseconds, 1),
    largestComposites: compositePassList.slice(0, 12).map((pass) => ({
      label: pass.label,
      gpuMsPerFrame: pass.gpuMsPerFrame,
      reason: pass.compositeReason ?? "composite",
    })),
  };

  // ---- Per-frame samples and retained frames ----
  const taskLabels = (exactPassList.length > 0 ? exactPassList : passList)
    .slice(0, 24).map((pass) => pass.label);
  const frameSamples: FrameSample[] = frameAccumulators.map((frame, index) => {
    const duration = frame.end - frame.start;
    return {
      index,
      startMs: (frame.start - windowStart) / 1e3,
      durationMs: duration / 1e3,
      busyMs: frame.busy / 1e3,
      gapMs: Math.max(0, duration - frame.busy) / 1e3,
      encoders: frame.encoders,
      passes: frame.passes,
      occupancy: frame.occupancySamples > 0
        ? frame.occupancySum / frame.occupancySamples / 100 : undefined,
      counterSamples: frame.occupancySamples,
      tasks: taskLabels.map((label) => (frame.tasks.get(label) ?? 0) / 1e3),
    };
  });
  // Retaining every frame's intervals would make a 500-advance capture a file
  // no browser will open, so keep an evenly spread subset -- with the two ends
  // and the representative always in it, since those are what get compared.
  const CAPTURE_LIMIT = 96;
  const wanted = new Set<number>();
  if (frameAccumulators.length > 0) {
    wanted.add(0);
    wanted.add(frameAccumulators.length - 1);
    wanted.add(Math.min(representativeIndex, frameAccumulators.length - 1));
    const stride = Math.max(1, Math.ceil(frameAccumulators.length / CAPTURE_LIMIT));
    for (let index = 0; index < frameAccumulators.length; index += stride) wanted.add(index);
  }
  const captures: FrameCapture[] = [...wanted].sort((left, right) => left - right)
    .map((index) => {
      const frame = frameAccumulators[index];
      return {
        index,
        durationUs: frame.end - frame.start,
        intervals: coalesceEncoderSlices(windowed
          .filter((interval) => interval.start >= frame.start && interval.start < frame.end))
          .map((interval) => ({ ...interval, start: interval.start - frame.start })),
      };
    });
  const capturedRepresentative = Math.max(0,
    captures.findIndex((capture) => capture.index === representativeIndex));
  // Advance numbering is only knowable when the trace spans the whole run; the
  // analysed window drops the first anchor, so frame 0 is the second advance.
  const coversWholeRun = input.traced?.steps !== undefined
    && boundaries.length >= input.traced.steps * 0.9;
  const firstAdvance = input.firstFrame ? 1 : coversWholeRun
    ? 2 + (input.singleFrame ? selectedBoundary : 0) : undefined;

  // Per-task occupancy across the representative frame, binned for the grid.
  const occupancyGrid: Record<string, (number | null)[]> = {};
  const binWidth = timelineDuration / GRID_BINS;
  for (const pass of passList.slice(0, 18)) {
    const bins: (number | null)[] = new Array(GRID_BINS).fill(null);
    for (const interval of timelineSlices) {
      if (interval.label !== pass.label) continue;
      const from = Math.max(0, Math.floor((interval.start - timelineStart) / binWidth));
      const to = Math.min(GRID_BINS - 1,
        Math.floor((interval.start - timelineStart + interval.duration) / binWidth));
      // A task with no counter samples has unknown occupancy; leaving it null
      // renders it as idle-grey rather than inventing a mid-scale green.
      for (let bin = from; bin <= to; bin += 1) bins[bin] = pass.occupancy ?? null;
    }
    occupancyGrid[pass.label] = bins;
  }
  // Annotate every drawable interval with its pass occupancy -- the
  // representative frame and each retained frame alike, so switching frames in
  // the report does not switch colouring rules.
  const occupancyByLabel = new Map(passList.map((pass) => [pass.label, pass]));
  const annotate = (interval: Interval): void => {
    const pass = occupancyByLabel.get(interval.label);
    interval.occupancy = pass?.occupancy ?? null;
    interval.alu = pass?.alu ?? null;
  };
  const timelineIntervals = coalesceEncoderSlices(timelineSlices);
  for (const interval of timelineIntervals) annotate(interval);
  for (const capture of captures) for (const interval of capture.intervals) annotate(interval);

  const baselineMsPerAdvance = input.baseline?.simulationWall_ms && input.baseline.steps
    ? input.baseline.simulationWall_ms / input.baseline.steps : undefined;
  const tracedMsPerAdvance = input.traced?.simulationWall_ms && input.traced.steps
    ? input.traced.simulationWall_ms / input.traced.steps : undefined;
  const busyMsPerFrame = busyMicroseconds / 1e3 / frameCount;
  const wallMsPerFrame = wallMicroseconds / 1e3 / frameCount;
  const contention = [...contentionTally.entries()]
    .map(([process, entry]) => ({ process, ...entry }))
    .sort((left, right) => right.gpuMs - left.gpuMs);
  const exclusiveCoverage = intervalMicroseconds > 0
    ? exclusiveMicroseconds / intervalMicroseconds : 0;

  const lines: string[] = [];
  lines.push(`frames analysed: ${frameCount} (anchor "${anchor}")`
    + (input.firstFrame ? " -- literal advance 1 only"
      : input.singleFrame ? " -- one complete representative advance only" : ""));
  if (literalFrameUsesCommandBufferCompletion) {
    lines.push("frame boundary:  advance 1 ends at its labelled Metal command-buffer completion"
      + " (advance 2's semantic start was outside the counter window)");
  }
  // First, because it decides what every number below is allowed to mean.
  // Metal names an encoder once, when it begins; the broker deliberately shares
  // one pass across consecutive stages, so an un-isolated bucket carries its
  // named stage AND everything encoded after it until the next pass boundary.
  // A 3.55 ms row once read as a 24-workgroup classify on this lane and was in
  // fact the whole SPGrid candidate hierarchy rebuild sharing its pass.
  {
    const composite = `${attribution.compositeBuckets} composite`
      + ` (${attribution.compositeMsPerFrame.toFixed(2)} ms/advance,`
      + ` ${(100 * attribution.compositeShare).toFixed(1)}%)`;
    lines.push(`attribution:     ${attribution.mode === "off"
      ? `label isolation OFF -- ALL ${passList.length} buckets are composite`
      : attribution.mode === "full"
        ? `full label isolation -- ${attribution.exactBuckets} exact stages, ${composite}`
        : `PARTIAL label isolation, scoped to "${isolatedPrefixes.join(", ")}"`
          + ` -- only ${attribution.exactBuckets} of ${passList.length} buckets`
          + ` (${attribution.exactMsPerFrame.toFixed(2)} ms/advance) are exact stages; ${composite}`}`);
    if (attribution.compositeBuckets > 0) {
      lines.push("                 A COMPOSITE ROW IS NOT ITS NAMED KERNEL'S COST: it is the first"
        + " label of a Metal encoder");
      lines.push("                 and carries every stage encoded until the next pass boundary."
        + " Re-run with the label");
      lines.push("                 prefix that covers the row you care about before quoting its ms.");
    }
  }
  // Whether a "frame" here is really one advance is checkable rather than
  // assumable, and it is a separate question from whether every advance does
  // the same work. The boundary is validated against the step count the
  // harness reports; the shape census then describes the run, and a varying
  // shape is a finding about the solver, not a fault in the segmentation.
  {
    const anchors = literalFrameUsesCommandBufferCompletion
      ? Math.max(0, boundaries.length - 1) : boundaries.length;
    const steps = input.traced?.steps;
    const perAdvance = steps === undefined || steps === 0 ? undefined : anchors / steps;
    lines.push(`frame boundary:  "${anchor}" fired ${anchors}x`
      + (perAdvance === undefined ? " (no step count to check it against)"
        : perAdvance > 1.15 || perAdvance < 0.85
          ? ` for the ${steps} advances the harness ran = ${perAdvance.toFixed(2)} per advance`
            + (perAdvance < 0.85
              ? `, so this capture covers ${(100 * perAdvance).toFixed(0)}% of the run`
              : " -- A FRAME HERE IS NOT AN ADVANCE")
          : ` for the ${steps} advances the harness ran -- one per advance`));
    if (semanticFrameStart) {
      const firstStage = timelineIntervals[0];
      const semanticStage = timelineIntervals.find((interval) => interval.label === anchor);
      lines.push(`frame origin:    preceding GPU setup activity is t=0; "${anchor}" starts`
        + ` ${Math.max(0, (semanticStage?.start ?? timelineStart) - timelineStart).toFixed(0)}`
        + " µs later"
        + (firstStage?.label === anchor ? " and is the first compute stage"
          : ` after recurring "${firstStage?.label ?? "unknown"}" work`));
    }
    const shapeTally = new Map<string, number>();
    for (const frame of frameSamples) {
      const shape = `${frame.encoders}/${frame.passes}`;
      shapeTally.set(shape, (shapeTally.get(shape) ?? 0) + 1);
    }
    const [modal, modalCount] = [...shapeTally.entries()]
      .sort((left, right) => right[1] - left[1])[0] ?? ["0/0", 0];
    lines.push(`frame shape:     ${shapeTally.size === 1
      ? `all ${frameCount} advances carry ${modal.replace("/", " encoders / ")} passes`
      : `${modalCount} of ${frameCount} advances carry ${modal.replace("/", " encoders / ")} passes;`
        + ` ${frameCount - modalCount} do more or less work`
        + ` (${Math.min(...frameSamples.map((frame) => frame.encoders))}-`
        + `${Math.max(...frameSamples.map((frame) => frame.encoders))} encoders,`
        + ` ${Math.min(...frameSamples.map((frame) => frame.passes))}-`
        + `${Math.max(...frameSamples.map((frame) => frame.passes))} passes),`
        + " so the per-advance figures describe the modal advance"}`);
  }
  lines.push(`frame wall:      ${wallMsPerFrame.toFixed(2)} ms/advance`
    + ` (p10 ${(quantile(frameDurations, 0.1) / 1e3).toFixed(2)},`
    + ` p90 ${(quantile(frameDurations, 0.9) / 1e3).toFixed(2)})`);
  if (baselineMsPerAdvance !== undefined) {
    lines.push(`untraced wall:   ${baselineMsPerAdvance.toFixed(2)} ms/advance`
      + (tracedMsPerAdvance !== undefined
        ? ` -> traced ${tracedMsPerAdvance.toFixed(2)} ms`
        + ` (${(tracedMsPerAdvance / baselineMsPerAdvance).toFixed(2)}x)` : ""));
  }
  lines.push(`GPU busy:        ${busyMsPerFrame.toFixed(2)} ms/advance`
    + ` = ${(100 * busyMsPerFrame / wallMsPerFrame).toFixed(1)}% of frame wall`);
  const overlapMicroseconds = Math.max(0, intervalMicroseconds - stageBusyMicroseconds);
  if (overlapMicroseconds > 0) {
    lines.push(`GPU intervals:   ${(intervalMicroseconds / 1e3 / frameCount).toFixed(2)}`
      + ` ms/advance attributed, including ${(overlapMicroseconds / 1e3 / frameCount).toFixed(2)}`
      + " ms of overlapping Metal interval records");
  }
  lines.push(`GPU idle gaps:   ${(gapMicroseconds / 1e3 / frameCount).toFixed(2)} ms/advance`
    + ` across ${Math.round(gaps.length / frameCount)} gaps`);
  if (windowedDiagnosticBlits.length > 0) {
    lines.push(`isolation blits: ${(diagnosticBlitMicroseconds / 1e3 / frameCount).toFixed(2)}`
      + ` ms/advance across ${Math.round(windowedDiagnosticBlits.length / frameCount)}`
      + " diagnostic encoder-boundary clears; excluded from solver stages");
  }
  lines.push(`GPU encoders:    ${Math.round(logicalEncoderCount / frameCount)}/advance`
    + ` carrying ${Math.round(logicalPassCount / frameCount)} labelled `
    + `${input.workUnit === "frame" ? "render" : "compute"} passes`
    + (continuationSlices > 0
      ? ` (${Math.round(continuationSlices / frameCount)} resumed execution slices coalesced)` : ""));
  if (unlabelledCount > 0) {
    // These sit before the analysis window, which opens at the first labelled
    // anchor, so they cost coverage rather than accuracy. Worth seeing: a large
    // share means the window is mostly encoder-stream warm-up.
    lines.push(`unlabelled:      ${unlabelledCount} of our intervals`
      + ` (${(unlabelledMicroseconds / 1e3).toFixed(0)} ms) arrived before the encoder`
      + " metadata stream started and are outside the analysed window");
  }
  const occupancy = counterMean(globalCounters, OCCUPANCY);
  if (occupancy !== undefined) {
    lines.push(`${OCCUPANCY.toLowerCase()}: ${occupancy.toFixed(1)}% mean`
      + ` (ALU ${(counterMean(globalCounters, "ALU Utilization") ?? 0).toFixed(1)}%,`
      + ` ${usableSampleCount}/${counterSampleCount} counter samples uncontended)`);
    if (input.counterExtraction && input.counterExtraction.timestampStride > 1) {
      lines.push(`counter overlay: ${input.counterExtraction.retainedCounterCount}/`
        + `${input.counterExtraction.sourceCounterCount} series, every `
        + `${input.counterExtraction.timestampStride}th hardware timestamp;`
        + " exact GPU stage timings remain full resolution");
    }
  } else {
    lines.push(`${OCCUPANCY.toLowerCase()}: not captured (rerun with --counters)`);
  }
  if (contention.length > 0) {
    lines.push(`contention:      ${contention[0].process} used`
      + ` ${contention[0].gpuMs.toFixed(0)} ms of GPU in the window;`
      + ` ${(100 * exclusiveCoverage).toFixed(1)}% of our GPU time was uncontended`);
  }
  if (exactPassList.length > 0) {
    lines.push(`exact targets:   ${(exactMicroseconds / 1e3 / frameCount).toFixed(2)}`
      + ` ms/advance = ${(100 * exactMicroseconds / Math.max(intervalMicroseconds, 1)).toFixed(1)}%`
      + ` of attributed interval time; ${(compositeMicroseconds / 1e3 / frameCount).toFixed(2)}`
      + " ms remains composite/outside this targeted capture");
  }
  lines.push("");
  // The header must describe the list that is actually printed. A scoped
  // capture whose prefix matched nothing falls back to the composite list, and
  // calling that "exactly isolated" is how a composite bucket gets quoted.
  const rankedPasses = exactPassList.length > 0 ? exactPassList : passList;
  lines.push(exactPassList.length === 0
    ? "top COMPOSITE GPU pass groups -- no stage was exactly isolated in this capture:"
    : attributionMode === "scoped"
      ? "top exactly isolated GPU tasks (non-target composite buckets omitted):"
      : "top GPU tasks (ms per advance):");
  for (const pass of rankedPasses.slice(0, 15)) {
    lines.push(`  ${pass.gpuMsPerFrame.toFixed(3).padStart(8)} ms`
      + ` ${(100 * pass.share).toFixed(1).padStart(5)}%`
      + ` ${pass.callsPerFrame.toFixed(1).padStart(6)}x`
      + ` ${pass.meanMicroseconds.toFixed(0).padStart(6)} µs`
      + ` occ ${pass.occupancy === undefined ? "  n/a" : `${(100 * pass.occupancy).toFixed(0).padStart(3)}%`}`
      + `  ${pass.exactAttribution ? "" : "[composite] "}${pass.merged ? "[merged] " : ""}`
      + `${pass.label.slice(0, 70)}`);
  }
  // Printed even when exact stages exist: these are the rows a reader is most
  // likely to lift out of the HTML table and quote as a kernel cost.
  if (attribution.largestComposites.length > 0) {
    lines.push("");
    lines.push("largest COMPOSITE buckets -- each is \"this label plus every stage encoded"
      + " until the next pass boundary\", NOT a kernel cost:");
    for (const composite of attribution.largestComposites.slice(0, 10)) {
      lines.push(`  ${composite.gpuMsPerFrame.toFixed(3).padStart(8)} ms`
        + `  ${composite.label.slice(0, 78)}`);
    }
  }
  const pressureMicroStages = passList.filter((pass) =>
    pass.exactAttribution && (pass.label.startsWith("SPGrid accurate A2 -")
    || pass.label.startsWith("SPGrid Section 6.3 -")));
  if (pressureMicroStages.length > 0) {
    lines.push("");
    lines.push("pressure micro-stages (label isolation required for exact attribution):");
    for (const pass of pressureMicroStages) {
      lines.push(`  ${pass.gpuMsPerFrame.toFixed(3).padStart(8)} ms`
        + ` ${pass.callsPerFrame.toFixed(1).padStart(6)}x`
        + ` occ ${pass.occupancy === undefined ? "  n/a" : `${(100 * pass.occupancy).toFixed(1).padStart(5)}%`}`
        + ` ALU ${pass.alu === undefined ? "  n/a" : `${(100 * pass.alu).toFixed(1).padStart(5)}%`}`
        + ` resident ${pass.residentThreads === undefined ? "?" : pass.residentThreads.toFixed(0)} threads`
        + `  ${pass.label}`);
    }
  }
  const fineJFAMicroStages = passList.filter((pass) =>
    pass.exactAttribution && pass.label.startsWith("Fine JFA -"));
  if (fineJFAMicroStages.length > 0) {
    lines.push("");
    lines.push("fine JFA micro-stages (B4 floods use one voxel per lane):");
    for (const pass of fineJFAMicroStages) {
      lines.push(`  ${pass.gpuMsPerFrame.toFixed(3).padStart(8)} ms`
        + ` ${pass.callsPerFrame.toFixed(1).padStart(6)}x`
        + ` occ ${pass.occupancy === undefined ? "  n/a" : `${(100 * pass.occupancy).toFixed(1).padStart(5)}%`}`
        + ` ALU ${pass.alu === undefined ? "  n/a" : `${(100 * pass.alu).toFixed(1).padStart(5)}%`}`
        + ` resident ${pass.residentThreads === undefined ? "?" : pass.residentThreads.toFixed(0)} threads`
        + `  ${pass.label}`);
    }
  }

  return {
    workUnit: input.workUnit,
    title: input.title,
    lane: input.lane,
    scene: input.environment.FLUID_SCENE ?? "?",
    grid: input.environment.FLUID_EXPECT_GRID ?? "?",
    capturedAt: new Date().toISOString(),
    wall: {
      baselineMsPerAdvance,
      tracedMsPerAdvance,
      distortion: baselineMsPerAdvance && tracedMsPerAdvance
        ? tracedMsPerAdvance / baselineMsPerAdvance : undefined,
      steps: input.traced?.steps,
    },
    frames: {
      count: frameCount, anchor,
      medianMs: quantile(frameDurations, 0.5) / 1e3,
      p10Ms: quantile(frameDurations, 0.1) / 1e3,
      p90Ms: quantile(frameDurations, 0.9) / 1e3,
      taskLabels,
      samples: frameSamples,
      captures,
      representative: capturedRepresentative,
      firstAdvance,
    },
    gpu: {
      busyMsPerFrame, wallMsPerFrame,
      intervalMsPerFrame: intervalMicroseconds / 1e3 / frameCount,
      overlapMsPerFrame: Math.max(0, intervalMicroseconds - stageBusyMicroseconds)
        / 1e3 / frameCount,
      exactMsPerFrame: exactMicroseconds / 1e3 / frameCount,
      compositeMsPerFrame: compositeMicroseconds / 1e3 / frameCount,
      exactIntervalCoverage: exactMicroseconds / Math.max(intervalMicroseconds, 1),
      occupancy: busyMsPerFrame / wallMsPerFrame,
      intervalsPerFrame: windowed.length / frameCount,
      encodersPerFrame: logicalEncoderCount / frameCount,
      passesPerFrame: logicalPassCount / frameCount,
      gapMsPerFrame: gapMicroseconds / 1e3 / frameCount,
      largestGapsUs: [...gaps].sort((left, right) => right - left).slice(0, 10),
      mergedShare: intervalMicroseconds > 0 ? mergedMicroseconds / intervalMicroseconds : 0,
      diagnosticBlitMsPerFrame: diagnosticBlitMicroseconds / 1e3 / frameCount,
      diagnosticBlitsPerFrame: windowedDiagnosticBlits.length / frameCount,
    },
    counters: {
      occupancyCounterName: OCCUPANCY,
      available: globalCounters.size > 0,
      partitionCount: devicePartitionCount.length,
      partitionOccupancy: devicePartitionSum
        .map((sum, ring) => sum / Math.max(devicePartitionCount[ring] ?? 1, 1) / 100),
      slotsPerPartition: slotModel.slots,
      slotConfidence: slotModel.confidence,
      threadsPerSlot: SIMD_WIDTH,
      totalSlots: slotModel.confidence >= 0.9
        ? slotModel.slots * devicePartitionCount.length : 0,
      totalThreads: slotModel.confidence >= 0.9
        ? slotModel.slots * devicePartitionCount.length * SIMD_WIDTH : 0,
      exclusiveCoverage,
      meanOccupancy: occupancy === undefined ? undefined : occupancy / 100,
      meanAlu: counterMean(globalCounters, "ALU Utilization") === undefined
        ? undefined : (counterMean(globalCounters, "ALU Utilization") as number) / 100,
      meanReadGBs: toGigabytesPerSecond(counterMean(globalCounters, "GPU Read Bandwidth")),
      meanWriteGBs: toGigabytesPerSecond(counterMean(globalCounters, "GPU Write Bandwidth")),
      sourceCounterCount: input.counterExtraction?.sourceCounterCount,
      retainedCounterCount: input.counterExtraction?.retainedCounterCount,
      timestampStride: input.counterExtraction?.timestampStride,
      retainedSampleIntervalUs: input.counterExtraction === undefined ? undefined
        : ([...counterNames.values()][0]?.interval ?? 0)
          * input.counterExtraction.timestampStride / 1e3,
      occupancyTrace: occupancyTrace.sort((left, right) => left.t - right.t),
      frameSeries: [...frameSeriesRaw.entries()].map(([name, points]) => ({
        name, points: points.sort((left, right) => left.t - right.t),
      })),
    },
    attribution,
    passes: passList,
    occupancyGrid,
    shaders: [...shaderTally.entries()]
      .map(([name, entry]) => ({
        name, samples: entry.samples, pipelines: entry.pipelines.size,
        share: entry.samples / Math.max([...shaderTally.values()]
          .reduce((sum, value) => sum + value.samples, 0), 1),
      }))
      .sort((left, right) => right.samples - left.samples),
    channels: [...channelTotals.entries()].map(([channel, entry]) => ({
      channel, msPerFrame: entry.us / 1e3 / frameCount, count: entry.count,
    })).sort((left, right) => right.msPerFrame - left.msPerFrame),
    contention,
    cpu: {
      samples, runningSamples,
      threads: [...threadTally.entries()].map(([thread, entry]) => ({ thread, ...entry }))
        .sort((left, right) => right.samples - left.samples),
      flame: foldStacks(stacks),
      hotLeaves: [...leafTally.entries()]
        .map(([symbol, count]) => ({ symbol, samples: count, share: count / Math.max(samples, 1) }))
        .sort((left, right) => right.samples - left.samples).slice(0, 30),
    },
    timeline: {
      frameStart: timelineStart,
      frameDuration: timelineDuration,
      intervals: timelineIntervals.map((interval) => ({
        ...interval, start: interval.start - timelineStart,
      })),
    },
    console: lines,
  };
};

export { renderFrameReportHtml } from "./xctrace-frame-report-html";
