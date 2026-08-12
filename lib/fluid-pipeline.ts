import type { MethodParamValues } from "./methods/types";
import type { GPUEulerianInfo } from "./webgpu-eulerian";
import type { PerformanceTrace } from "./performance-trace";

/**
 * The simulation-side sibling of `render-pipeline-graph`: one frozen graph per
 * method describing every stage its `advanceTo` encodes, in encode order.
 *
 * The contract that keeps the diagram honest is label identity: a stage's
 * `phaseLabels` are the exact strings the solver's trace seams emit through
 * `GPUStageTimestampRecorder.completePhase`, so the figure on a node is the
 * hardware time of the passes between that stage's seam and the previous one —
 * an exact partition of the advance, never a reconstruction. To make drift
 * structurally hard, a method's graph is declared in the same file as the
 * encoder that emits the seams, built from the same phase-map constants.
 *
 * Only stages that declare a `toggle` are ablatable. The solver owns the
 * corresponding identity/fallback handoff, so closing one of those gates can
 * never leave a downstream texture stale. Structural stages keep read-only
 * lamps.
 */

export interface FluidPipelineBand {
  readonly id: string;
  readonly label: string;
}

/**
 * What a stage is doing this advance.
 *
 * `off` is a stage the method's configuration withholds (Sec. 3.8
 * post-processing with the param off); `unavailable` is a stage whose gate the
 * scene cannot currently satisfy (rigid coupling with no bodies). Both encode
 * nothing; the distinction is whether the user could turn it on from here.
 */
export type FluidPipelineStageState = "on" | "off" | "unavailable";

/** Everything a stage consults to describe itself. Read once per render. */
export interface FluidPipelineContext {
  /** Resolved method parameter values: defaults ← quality ← overrides. */
  readonly values: MethodParamValues;
  /** The live solver's published info record, when one is attached. */
  readonly info: GPUEulerianInfo | null;
  readonly sceneId: string;
  readonly bodyCount: number;
  readonly hasTerrain: boolean;
  readonly hasInflow: boolean;
  readonly running: boolean;
}

export interface FluidPipelineTip {
  readonly summary: string;
  readonly reads?: string;
  readonly writes?: string;
  readonly feeds?: string;
  /** The condition that decides whether this stage encodes at all. */
  readonly gate?: string;
}

/**
 * A control a stage declares for itself, colocated with the stage rather than
 * hand-wired in the panel. `param-*` controls bind to a method parameter by
 * key — the panel routes changes through the method store, so a runtime-unsafe
 * parameter rebuilds the solver exactly as it would from the method panel —
 * and `readout` rows surface a live fact with no input at all.
 */
export type FluidStageControl =
  | {
    readonly kind: "param-choice";
    readonly param: string;
    readonly label: string;
    readonly hint?: string;
    readonly options: ReadonlyArray<{ readonly value: string; readonly label: string; readonly hint?: string }>;
  }
  | {
    readonly kind: "param-range";
    readonly param: string;
    readonly label: string;
    readonly unit?: string;
    readonly min: number;
    readonly max: number;
    readonly step: number;
    readonly digits?: number;
    readonly hint?: string;
    /** Hide the input semantics (while retaining its readout) when a parent
     * stage gate makes the value irrelevant. */
    readonly enabled?: (context: FluidPipelineContext) => boolean;
  }
  | {
    readonly kind: "readout";
    readonly label: string;
    readonly hint?: string;
    readonly value: (context: FluidPipelineContext) => string;
  };

export interface FluidPipelineStage {
  readonly id: string;
  readonly band: string;
  readonly side: "left" | "right";
  readonly label: string;
  /** Trace seam labels this stage owns. Summed for its measured cost. */
  readonly phaseLabels: readonly string[];
  /**
   * Set when this stage's work is a term inside another stage's passes rather
   * than a seam of its own, naming the stage that owns the passes.
   */
  readonly costInsideStage?: string;
  /** A decision rather than a dispatch: its cost is a true zero. */
  readonly spendsNoFrameTime?: boolean;
  readonly tip: FluidPipelineTip;
  /** Controls this stage owns, rendered under its head. */
  readonly controls?: readonly FluidStageControl[];
  /** Optional user gate. Values are method parameters and therefore follow
   * the same rebuild/runtime contract as every other stage control. */
  readonly toggle?: {
    readonly param: string;
    readonly on: string | number | boolean;
    readonly off: string | number | boolean;
    readonly hint?: string;
  };
  readonly state: (context: FluidPipelineContext) => FluidPipelineStageState;
  /** The short factual chip under the label. Never a description. */
  readonly chip: (context: FluidPipelineContext) => string;
}

export interface FluidPipelineGraph {
  readonly methodId: string;
  readonly bands: readonly FluidPipelineBand[];
  readonly stages: readonly FluidPipelineStage[];
}

/**
 * Why a stage shows the number it shows.
 *
 * - `measured`   summed from seams this stage owns in the averaged trace.
 * - `withheld`   the stage was not encoded. A true zero, not a gap.
 * - `shared`     a term inside another stage's passes; the figure is the host's.
 * - `idle`       the partition arrived, but this gated stage did not encode.
 * - `structural` a decision that never spends advance time either way.
 * - `unmeasured` no trace yet, or a partition too coarse to name this stage.
 */
export type FluidPipelineCostKind = "measured" | "withheld" | "shared" | "idle" | "structural" | "unmeasured";

export interface FluidPipelineMeasurement {
  readonly kind: FluidPipelineCostKind;
  /** Undefined only when `kind` is `unmeasured`. */
  readonly duration_ms?: number;
  /** Fraction of the advance, for the stage's figure. */
  readonly share: number;
  /** For `shared`, the stage whose passes carry this one's work. */
  readonly insideStage?: string;
  /**
   * Fraction of sampled advances that encoded this stage, when below one.
   * `duration_ms` is already the expected per-advance cost — the per-encode
   * mean diluted by this fraction — so a stage that ran in 3 of 12 samples
   * reads as what it costs a typical advance, badged rather than silent.
   */
  readonly encodedFraction?: number;
}

const UNMEASURED: FluidPipelineMeasurement = Object.freeze({ kind: "unmeasured", share: 0 });

export interface FluidPhaseCost {
  /** Expected per-advance milliseconds: per-encode mean × encoded fraction. */
  readonly expected_ms: number;
  readonly encodedFraction: number;
}

/**
 * Expected per-advance milliseconds per seam label, from an averaged trace.
 *
 * `averagePerformanceTraces` reports each phase as its per-encode mean plus the
 * fraction of samples that encoded it. The panel's shares must add up to the
 * advance, so the map holds the expected value and keeps the fraction for the
 * intermittency badge.
 */
export function fluidPipelinePhaseCosts(trace: PerformanceTrace | undefined): ReadonlyMap<string, FluidPhaseCost> {
  const costs = new Map<string, FluidPhaseCost>();
  if (!trace) return costs;
  for (const phase of trace.phases) {
    const encodedFraction = phase.encodedFraction ?? 1;
    const current = costs.get(phase.label);
    costs.set(phase.label, {
      expected_ms: (current?.expected_ms ?? 0) + phase.duration_ms * encodedFraction,
      encodedFraction: Math.max(current?.encodedFraction ?? 0, encodedFraction),
    });
  }
  return costs;
}

function measureLabels(
  phaseLabels: readonly string[],
  costs: ReadonlyMap<string, FluidPhaseCost>,
  total_ms: number,
): FluidPipelineMeasurement {
  if (phaseLabels.length === 0) return UNMEASURED;
  let measured = false;
  let duration_ms = 0;
  let encodedFraction = 1;
  for (const label of phaseLabels) {
    const cost = costs.get(label);
    if (cost === undefined) continue;
    measured = true;
    duration_ms += cost.expected_ms;
    encodedFraction = Math.min(encodedFraction, cost.encodedFraction);
  }
  if (!measured) return UNMEASURED;
  return {
    kind: "measured",
    duration_ms,
    share: total_ms > 0 ? Math.min(1, duration_ms / total_ms) : 0,
    ...(encodedFraction < 1 ? { encodedFraction } : {}),
  };
}

/**
 * What this stage cost the advance, and on what basis.
 *
 * The withheld check runs before the label sum on purpose: an averaged window
 * can still carry a label from samples taken just before a stage's gate
 * closed, and reporting that stale mean as a live cost was the render panel's
 * single largest switch/measurement mismatch. State speaks for now; the trace
 * speaks for the window; now wins.
 */
export function measureFluidPipelineStage(
  stage: FluidPipelineStage,
  stages: readonly FluidPipelineStage[],
  costs: ReadonlyMap<string, FluidPhaseCost>,
  total_ms: number,
  state: FluidPipelineStageState,
): FluidPipelineMeasurement {
  if (stage.spendsNoFrameTime) return { kind: "structural", duration_ms: 0, share: 0 };
  if ((state === "off" || state === "unavailable") && costs.size > 0) {
    return { kind: "withheld", duration_ms: 0, share: 0 };
  }
  const own = measureLabels(stage.phaseLabels, costs, total_ms);
  if (own.kind === "measured") return own;
  if (stage.costInsideStage) {
    const host = stages.find((candidate) => candidate.id === stage.costInsideStage);
    const hostCost = host ? measureLabels(host.phaseLabels, costs, total_ms) : UNMEASURED;
    if (hostCost.kind === "measured") {
      return { ...hostCost, kind: "shared", insideStage: stage.costInsideStage };
    }
  }
  // The stage recorder's partition is exhaustive: every encoded seam is in the
  // trace. Once any seam has arrived, a missing label is a stage that encoded
  // nothing in the sampled advances, not a missing measurement.
  if (costs.size > 0) return { kind: "idle", duration_ms: 0, share: 0 };
  return UNMEASURED;
}

export function measureFluidPipelineBand(
  bandId: string,
  graph: FluidPipelineGraph,
  costs: ReadonlyMap<string, FluidPhaseCost>,
  total_ms: number,
): FluidPipelineMeasurement {
  const labels = graph.stages
    .filter((stage) => stage.band === bandId)
    .flatMap((stage) => stage.phaseLabels);
  return measureLabels(labels, costs, total_ms);
}

/** The hover tip as one string, in the tip's declared order. */
export function fluidPipelineTipText(stage: FluidPipelineStage, chip: string): string {
  const lines = [stage.label, chip ? `· ${chip}` : "", "", stage.tip.summary];
  const detail = [
    stage.tip.reads ? `Reads: ${stage.tip.reads}` : "",
    stage.tip.writes ? `Writes: ${stage.tip.writes}` : "",
    stage.tip.feeds ? `Feeds: ${stage.tip.feeds}` : "",
    stage.tip.gate ? `Runs when: ${stage.tip.gate}` : "",
  ].filter(Boolean);
  if (detail.length > 0) lines.push("", ...detail);
  return lines.filter((line, index) => line !== "" || lines[index - 1] !== "").join("\n");
}
