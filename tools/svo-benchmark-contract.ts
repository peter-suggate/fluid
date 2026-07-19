import {
  SVO_BASELINE_CASES,
  canonicalSVOBaselineCase,
  type SVOBaselineCase,
  type SVOBaselineQuality,
  type SVOBaselineRenderer,
} from "./svo-baseline-cases";
import {
  SVO_BASELINE_TOLERANCES,
  validateSVOBaselineAdapterLimits,
  type SVOBaselineAdapterObservation,
} from "./svo-baseline-contract";

export const SVO_BENCHMARK_SCHEMA_VERSION = 2;

export interface SVOBenchmarkResolution {
  readonly width: number;
  readonly height: number;
}

export interface SVOBenchmarkRunPlan {
  readonly id: string;
  readonly sequenceIndex: number;
  readonly pairIndex: number;
  readonly baselineId: string;
  readonly baselineCanonical: string;
  readonly renderer: SVOBaselineRenderer;
  readonly requestedMode: SVOBaselineRenderer;
  readonly quality: SVOBaselineQuality;
  readonly outputResolution: SVOBenchmarkResolution;
  readonly internalResolution: SVOBenchmarkResolution;
  readonly revision: string;
  readonly adapterId: string;
  readonly resetToken: string;
  readonly captureNotBeforeUnixMs: number;
  readonly warmupFrames: number;
  readonly measuredFrames: number;
}

export interface SVOBenchmarkPlan {
  readonly schemaVersion: typeof SVO_BENCHMARK_SCHEMA_VERSION;
  readonly purpose: "external-observation-raster-svo-ab";
  readonly revision: string;
  readonly adapterId: string;
  readonly resetToken: string;
  readonly captureNotBeforeUnixMs: number;
  readonly pairCount: number;
  readonly runs: readonly SVOBenchmarkRunPlan[];
  readonly captureInstructions: readonly string[];
}

export interface SVOBenchmarkEquivalenceObservation {
  /** Canonical scene state as captured, or an immutable content hash. */
  readonly sceneStateIdentity: string;
  /** Canonical camera state as captured, or an immutable content hash. */
  readonly cameraStateIdentity: string;
  /** Renderer-independent solver state hash at the requested checkpoint. */
  readonly simulationStateIdentity: string;
  readonly simulatedTime_s: number;
  readonly stepCount: number;
}

export interface SVOBenchmarkFrameObservation {
  readonly frameIndex: number;
  readonly sampledAtUnixMs: number;
  readonly resetToken: string;
  readonly requestedMode: SVOBaselineRenderer;
  readonly effectiveMode: SVOBaselineRenderer;
  readonly fallbackReason: string | null;
  readonly renderTimingContext: string;
  readonly renderTimingEpoch: number;
  /** Accepted timestamp-query identity; collectors must not duplicate cached values. */
  readonly renderTimingSampleId: number | null;
  readonly gpuRenderTimingAvailable: boolean;
  readonly cpuFrame_ms: number;
  readonly gpuRender_ms: number | null;
  readonly gpuDryScene_ms: number | null;
  readonly gpuSvoTemporal_ms: number | null;
  readonly rendererOwnedBytes: number;
}

export interface SVOBenchmarkRunObservation {
  readonly runId: string;
  readonly sequenceIndex: number;
  readonly revision: string;
  readonly adapterId: string;
  readonly baselineCanonical: string;
  readonly quality: SVOBaselineQuality;
  readonly outputResolution: SVOBenchmarkResolution;
  readonly internalResolution: SVOBenchmarkResolution;
  readonly resetToken: string;
  readonly adapter: SVOBaselineAdapterObservation;
  readonly equivalence: SVOBenchmarkEquivalenceObservation;
  readonly frames: readonly SVOBenchmarkFrameObservation[];
}

export interface SVOBenchmarkObservationBundle {
  readonly schemaVersion: typeof SVO_BENCHMARK_SCHEMA_VERSION;
  readonly runs: readonly SVOBenchmarkRunObservation[];
}

export interface SVOBenchmarkDistribution {
  readonly p50: number;
  readonly p95: number;
  readonly maximum: number;
}

export interface SVOBenchmarkRunResult {
  readonly plan: SVOBenchmarkRunPlan;
  readonly adapter: SVOBaselineAdapterObservation;
  readonly equivalence: SVOBenchmarkEquivalenceObservation;
  readonly effectiveMode: SVOBaselineRenderer;
  readonly fallbackReason: null;
  readonly renderTimingContext: string;
  readonly renderTimingEpoch: number;
  readonly timestampQueriesAvailable: boolean;
  readonly cpuFrame_ms: SVOBenchmarkDistribution;
  readonly gpuRender_ms: SVOBenchmarkDistribution | null;
  readonly gpuDryScene_ms: SVOBenchmarkDistribution | null;
  readonly gpuSvoTemporal_ms: SVOBenchmarkDistribution | null;
  readonly rendererOwnedBytes: SVOBenchmarkDistribution;
  /** Warmup and measured observations are retained verbatim for re-analysis. */
  readonly rawFrames: readonly SVOBenchmarkFrameObservation[];
}

export interface SVOBenchmarkPairResult {
  readonly baselineId: string;
  readonly pairIndex: number;
  readonly order: readonly [SVOBaselineRenderer, SVOBaselineRenderer];
  readonly rasterRunId: string;
  readonly svoRunId: string;
  readonly equivalenceValidated: true;
  readonly cpuFrameP95RatioSvoToRaster: number;
  readonly gpuRenderP95RatioSvoToRaster: number | null;
  readonly peakMemoryRatioSvoToRaster: number;
}

export interface SVOBenchmarkAggregateResult {
  readonly baselineId: string;
  readonly renderer: SVOBaselineRenderer;
  readonly adapterId: string;
  readonly adapter: SVOBaselineAdapterObservation;
  readonly runIds: readonly string[];
  readonly quality: SVOBaselineQuality;
  readonly outputResolution: SVOBenchmarkResolution;
  readonly internalResolution: SVOBenchmarkResolution;
  readonly timestampQueriesAvailable: boolean;
  readonly cpuFrame_ms: SVOBenchmarkDistribution;
  readonly gpuRender_ms: SVOBenchmarkDistribution | null;
  readonly gpuDryScene_ms: SVOBenchmarkDistribution | null;
  readonly gpuSvoTemporal_ms: SVOBenchmarkDistribution | null;
  readonly rendererOwnedBytes: SVOBenchmarkDistribution;
}

export interface SVOBenchmarkReport {
  readonly schemaVersion: typeof SVO_BENCHMARK_SCHEMA_VERSION;
  readonly plan: SVOBenchmarkPlan;
  readonly runs: readonly SVOBenchmarkRunResult[];
  readonly pairs: readonly SVOBenchmarkPairResult[];
  readonly aggregates: readonly SVOBenchmarkAggregateResult[];
}

function nonEmpty(value: string, label: string): string {
  if (typeof value !== "string" || value.trim().length === 0) throw new Error(`${label} must be non-empty`);
  return value;
}

function integer(value: number, minimum: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < minimum) throw new RangeError(`${label} must be an integer of at least ${minimum}`);
  return value;
}

function finiteNonNegative(value: number, label: string): number {
  if (!Number.isFinite(value) || value < 0) throw new RangeError(`${label} must be finite and non-negative`);
  return value;
}

function resolution(value: SVOBenchmarkResolution, label: string): SVOBenchmarkResolution {
  return Object.freeze({
    width: integer(value.width, 1, `${label} width`),
    height: integer(value.height, 1, `${label} height`),
  });
}

function stableJson(value: unknown): string {
  const stable = (item: unknown): unknown => {
    if (Array.isArray(item)) return item.map(stable);
    if (item && typeof item === "object") return Object.fromEntries(
      Object.entries(item).sort(([left], [right]) => left.localeCompare(right))
        .map(([key, child]) => [key, stable(child)]),
    );
    return item;
  };
  return JSON.stringify(stable(value));
}

function distribution(values: readonly number[], label: string): SVOBenchmarkDistribution {
  if (values.length === 0) throw new Error(`${label} needs at least one measured sample`);
  const sorted = values.map((value) => finiteNonNegative(value, label)).sort((left, right) => left - right);
  const quantile = (fraction: number) => {
    const position = (sorted.length - 1) * fraction;
    const lower = Math.floor(position), upper = Math.ceil(position);
    return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
  };
  return Object.freeze({ p50: quantile(.5), p95: quantile(.95), maximum: sorted.at(-1)! });
}

function runOrder(pairIndex: number): readonly [SVOBaselineRenderer, SVOBaselineRenderer] {
  return pairIndex % 2 === 0 ? ["raster", "svo"] : ["svo", "raster"];
}

export function buildSVOBenchmarkPlan(options: {
  readonly revision: string;
  readonly adapterId: string;
  readonly resetToken: string;
  readonly captureNotBeforeUnixMs: number;
  readonly pairCount?: number;
  readonly cases?: readonly SVOBaselineCase[];
  readonly quality?: SVOBaselineQuality;
  readonly internalResolution: Readonly<Record<SVOBaselineRenderer, SVOBenchmarkResolution>>;
}): SVOBenchmarkPlan {
  const revision = nonEmpty(options.revision, "Benchmark revision");
  const adapterId = nonEmpty(options.adapterId, "Benchmark adapter ID");
  const resetToken = nonEmpty(options.resetToken, "Benchmark reset token");
  const captureNotBeforeUnixMs = integer(options.captureNotBeforeUnixMs, 0, "Benchmark not-before timestamp");
  const pairCount = integer(options.pairCount ?? 4, 1, "Benchmark pair count");
  const warmupFrames = SVO_BASELINE_TOLERANCES.performance.warmupFrames;
  const measuredFrames = SVO_BASELINE_TOLERANCES.performance.measuredFrames;
  const internalResolution = {
    raster: resolution(options.internalResolution.raster, "Raster internal resolution"),
    svo: resolution(options.internalResolution.svo, "SVO internal resolution"),
  };
  const runs: SVOBenchmarkRunPlan[] = [];
  for (const baseline of options.cases ?? SVO_BASELINE_CASES) for (let pairIndex = 0; pairIndex < pairCount; pairIndex += 1) {
    const quality = options.quality ?? baseline.quality;
    const baselineCanonical = canonicalSVOBaselineCase({ ...baseline, quality });
    for (const renderer of runOrder(pairIndex)) {
      const sequenceIndex = runs.length;
      runs.push(Object.freeze({
        id: `${baseline.id}--pair-${pairIndex}--${renderer}`,
        sequenceIndex,
        pairIndex,
        baselineId: baseline.id,
        baselineCanonical,
        renderer,
        requestedMode: renderer,
        quality,
        outputResolution: resolution(baseline.outputResolution, "Benchmark output resolution"),
        internalResolution: internalResolution[renderer],
        revision,
        adapterId,
        resetToken: `${resetToken}:${sequenceIndex}`,
        captureNotBeforeUnixMs,
        warmupFrames,
        measuredFrames,
      }));
    }
  }
  return Object.freeze({
    schemaVersion: SVO_BENCHMARK_SCHEMA_VERSION,
    purpose: "external-observation-raster-svo-ab",
    revision,
    adapterId,
    resetToken,
    captureNotBeforeUnixMs,
    pairCount,
    runs: Object.freeze(runs),
    captureInstructions: Object.freeze([
      "Capture is external: this plan does not claim browser automation.",
      "Reset timing history before every run and publish the exact per-run resetToken.",
      "Retain every warmup and measured frame in sequence; do not pre-average samples.",
      "Record requested/effective renderer and fallback identity on every frame.",
      "Append a timing frame only when renderTimingSampleId advances; cached 250 ms telemetry is not an independent sample.",
      "Record scene, SVO-temporal (zero only when explicitly idle), total, mode context, epoch, adapter, and exact output/internal resolution.",
      "Use the canonical checkpoint and camera identity unchanged for both members of each raster/SVO pair.",
    ]),
  });
}

function validateRunObservation(plan: SVOBenchmarkRunPlan, observation: SVOBenchmarkRunObservation): SVOBenchmarkRunResult {
  const mismatch = (actual: unknown, expected: unknown, label: string) => {
    if (stableJson(actual) !== stableJson(expected)) throw new Error(`${plan.id} ${label} mismatch`);
  };
  if (observation.runId !== plan.id || observation.sequenceIndex !== plan.sequenceIndex) throw new Error(`${plan.id} run identity/order mismatch`);
  if (observation.revision !== plan.revision) throw new Error(`${plan.id} stale revision ${observation.revision}; expected ${plan.revision}`);
  if (observation.adapterId !== plan.adapterId) throw new Error(`${plan.id} adapter mismatch`);
  if (observation.baselineCanonical !== plan.baselineCanonical) throw new Error(`${plan.id} baseline mismatch`);
  if (observation.quality !== plan.quality) throw new Error(`${plan.id} quality mismatch`);
  mismatch(observation.outputResolution, plan.outputResolution, "output resolution");
  mismatch(observation.internalResolution, plan.internalResolution, "internal resolution");
  if (observation.resetToken !== plan.resetToken) throw new Error(`${plan.id} stale reset token`);
  const limitFailures = validateSVOBaselineAdapterLimits(observation.adapter);
  if (limitFailures.length > 0) throw new Error(`${plan.id} adapter limit failure: ${limitFailures.join(", ")}`);
  const expectedFrames = plan.warmupFrames + plan.measuredFrames;
  if (observation.frames.length !== expectedFrames) throw new Error(`${plan.id} has ${observation.frames.length} frames; exactly ${expectedFrames} are required`);
  let previousTimestamp = plan.captureNotBeforeUnixMs;
  let previousTimingSampleId = -1;
  const renderTimingContext = nonEmpty(observation.frames[0]?.renderTimingContext ?? "", `${plan.id} render timing context`);
  const renderTimingEpoch = integer(observation.frames[0]?.renderTimingEpoch ?? -1, 1, `${plan.id} render timing epoch`);
  if (!renderTimingContext.endsWith(`:${plan.requestedMode}:epoch-${renderTimingEpoch}`)) {
    throw new Error(`${plan.id} render timing context does not identify ${plan.requestedMode} epoch ${renderTimingEpoch}`);
  }
  for (let index = 0; index < observation.frames.length; index += 1) {
    const frame = observation.frames[index];
    if (frame.frameIndex !== index) throw new Error(`${plan.id} frame sequence is stale or discontinuous at ${index}`);
    if (frame.resetToken !== plan.resetToken) throw new Error(`${plan.id} frame ${index} has a stale reset token`);
    if (!Number.isSafeInteger(frame.sampledAtUnixMs) || frame.sampledAtUnixMs < previousTimestamp) {
      throw new Error(`${plan.id} frame ${index} predates the reset or is out of order`);
    }
    previousTimestamp = frame.sampledAtUnixMs;
    if (frame.requestedMode !== plan.requestedMode) throw new Error(`${plan.id} frame ${index} requested-renderer mismatch`);
    if (frame.effectiveMode !== plan.requestedMode || frame.fallbackReason !== null) {
      throw new Error(`${plan.id} frame ${index} effective renderer/fallback mismatch: ${frame.effectiveMode}/${frame.fallbackReason ?? "none"}`);
    }
    if (frame.renderTimingContext !== renderTimingContext || frame.renderTimingEpoch !== renderTimingEpoch) {
      throw new Error(`${plan.id} frame ${index} timing mode/epoch changed during the run`);
    }
    finiteNonNegative(frame.cpuFrame_ms, `${plan.id} CPU frame`);
    finiteNonNegative(frame.rendererOwnedBytes, `${plan.id} renderer memory`);
    if (frame.gpuRenderTimingAvailable) {
      const sampleId = integer(frame.renderTimingSampleId ?? -1, 1, `${plan.id} timestamp sample ID`);
      if (sampleId <= previousTimingSampleId) throw new Error(`${plan.id} frame ${index} repeats a cached timestamp sample`);
      previousTimingSampleId = sampleId;
      finiteNonNegative(frame.gpuRender_ms ?? Number.NaN, `${plan.id} GPU render`);
      finiteNonNegative(frame.gpuDryScene_ms ?? Number.NaN, `${plan.id} GPU dry scene`);
      finiteNonNegative(frame.gpuSvoTemporal_ms ?? Number.NaN, `${plan.id} GPU SVO temporal`);
      if (plan.requestedMode === "raster" && frame.gpuSvoTemporal_ms !== 0) {
        throw new Error(`${plan.id} frame ${index} raster mode must report temporal as explicitly idle`);
      }
    } else if (frame.renderTimingSampleId !== null || frame.gpuRender_ms !== null || frame.gpuDryScene_ms !== null || frame.gpuSvoTemporal_ms !== null) {
      throw new Error(`${plan.id} frame ${index} reports unavailable timestamps with timing values`);
    }
  }
  const measured = observation.frames.slice(plan.warmupFrames);
  const timestampQueriesAvailable = measured.every(({ gpuRenderTimingAvailable }) => gpuRenderTimingAvailable);
  if (!timestampQueriesAvailable && measured.some(({ gpuRenderTimingAvailable }) => gpuRenderTimingAvailable)) {
    throw new Error(`${plan.id} timestamp availability changed during measured frames`);
  }
  return Object.freeze({
    plan,
    adapter: observation.adapter,
    equivalence: observation.equivalence,
    effectiveMode: plan.requestedMode,
    fallbackReason: null,
    renderTimingContext,
    renderTimingEpoch,
    timestampQueriesAvailable,
    cpuFrame_ms: distribution(measured.map(({ cpuFrame_ms }) => cpuFrame_ms), `${plan.id} CPU frame`),
    gpuRender_ms: timestampQueriesAvailable ? distribution(measured.map(({ gpuRender_ms }) => gpuRender_ms!), `${plan.id} GPU render`) : null,
    gpuDryScene_ms: timestampQueriesAvailable ? distribution(measured.map(({ gpuDryScene_ms }) => gpuDryScene_ms!), `${plan.id} GPU dry scene`) : null,
    gpuSvoTemporal_ms: timestampQueriesAvailable ? distribution(measured.map(({ gpuSvoTemporal_ms }) => gpuSvoTemporal_ms!), `${plan.id} GPU SVO temporal`) : null,
    rendererOwnedBytes: distribution(measured.map(({ rendererOwnedBytes }) => rendererOwnedBytes), `${plan.id} renderer memory`),
    rawFrames: Object.freeze([...observation.frames]),
  });
}

function validatePair(
  baseline: SVOBaselineCase,
  pairIndex: number,
  left: SVOBenchmarkRunResult,
  right: SVOBenchmarkRunResult,
): SVOBenchmarkPairResult {
  const byRenderer = new Map([[left.plan.renderer, left], [right.plan.renderer, right]]);
  const raster = byRenderer.get("raster"), svo = byRenderer.get("svo");
  if (!raster || !svo) throw new Error(`${baseline.id} pair ${pairIndex} does not contain raster and SVO`);
  if (stableJson(raster.adapter) !== stableJson(svo.adapter)) throw new Error(`${baseline.id} pair ${pairIndex} adapter observation mismatch`);
  if (stableJson(raster.equivalence) !== stableJson(svo.equivalence)) throw new Error(`${baseline.id} pair ${pairIndex} renderer-equivalence mismatch`);
  const equivalence = raster.equivalence;
  if (equivalence.stepCount !== baseline.checkpoint.stepCount
    || Math.abs(equivalence.simulatedTime_s - baseline.checkpoint.simulatedTime_s) > 1e-12) {
    throw new Error(`${baseline.id} pair ${pairIndex} checkpoint mismatch`);
  }
  for (const [value, label] of [
    [equivalence.sceneStateIdentity, "scene state"],
    [equivalence.cameraStateIdentity, "camera state"],
    [equivalence.simulationStateIdentity, "simulation state"],
  ] as const) nonEmpty(value, `${baseline.id} ${label} identity`);
  return Object.freeze({
    baselineId: baseline.id,
    pairIndex,
    order: runOrder(pairIndex),
    rasterRunId: raster.plan.id,
    svoRunId: svo.plan.id,
    equivalenceValidated: true,
    cpuFrameP95RatioSvoToRaster: svo.cpuFrame_ms.p95 / Math.max(raster.cpuFrame_ms.p95, Number.EPSILON),
    gpuRenderP95RatioSvoToRaster: raster.gpuRender_ms && svo.gpuRender_ms
      ? svo.gpuRender_ms.p95 / Math.max(raster.gpuRender_ms.p95, Number.EPSILON)
      : null,
    peakMemoryRatioSvoToRaster: svo.rendererOwnedBytes.maximum / Math.max(raster.rendererOwnedBytes.maximum, Number.EPSILON),
  });
}

function aggregateRuns(results: readonly SVOBenchmarkRunResult[]): readonly SVOBenchmarkAggregateResult[] {
  const keys = [...new Set(results.map(({ plan }) => `${plan.baselineId}\0${plan.renderer}`))];
  return Object.freeze(keys.map((key) => {
    const [baselineId, renderer] = key.split("\0") as [string, SVOBaselineRenderer];
    const grouped = results.filter(({ plan }) => plan.baselineId === baselineId && plan.renderer === renderer);
    const first = grouped[0];
    const measured = grouped.flatMap((run) => run.rawFrames.slice(run.plan.warmupFrames));
    const timestampQueriesAvailable = grouped.every(({ timestampQueriesAvailable }) => timestampQueriesAvailable);
    return Object.freeze({
      baselineId,
      renderer,
      adapterId: first.plan.adapterId,
      adapter: first.adapter,
      runIds: Object.freeze(grouped.map(({ plan }) => plan.id)),
      quality: first.plan.quality,
      outputResolution: first.plan.outputResolution,
      internalResolution: first.plan.internalResolution,
      timestampQueriesAvailable,
      cpuFrame_ms: distribution(measured.map(({ cpuFrame_ms }) => cpuFrame_ms), `${baselineId} ${renderer} aggregate CPU frame`),
      gpuRender_ms: timestampQueriesAvailable
        ? distribution(measured.map(({ gpuRender_ms }) => gpuRender_ms!), `${baselineId} ${renderer} aggregate GPU render`)
        : null,
      gpuDryScene_ms: timestampQueriesAvailable
        ? distribution(measured.map(({ gpuDryScene_ms }) => gpuDryScene_ms!), `${baselineId} ${renderer} aggregate GPU dry scene`)
        : null,
      gpuSvoTemporal_ms: timestampQueriesAvailable
        ? distribution(measured.map(({ gpuSvoTemporal_ms }) => gpuSvoTemporal_ms!), `${baselineId} ${renderer} aggregate GPU SVO temporal`)
        : null,
      rendererOwnedBytes: distribution(measured.map(({ rendererOwnedBytes }) => rendererOwnedBytes), `${baselineId} ${renderer} aggregate memory`),
    });
  }));
}

export function aggregateSVOBenchmarkObservations(
  plan: SVOBenchmarkPlan,
  observations: SVOBenchmarkObservationBundle,
  cases: readonly SVOBaselineCase[] = SVO_BASELINE_CASES,
): SVOBenchmarkReport {
  if (observations.schemaVersion !== SVO_BENCHMARK_SCHEMA_VERSION) throw new Error("SVO benchmark observation schema mismatch");
  if (observations.runs.length !== plan.runs.length) throw new Error(`SVO benchmark observation has ${observations.runs.length} runs; ${plan.runs.length} are required`);
  const results = plan.runs.map((run, index) => validateRunObservation(run, observations.runs[index]));
  const baselineById = new Map(cases.map((baseline) => [baseline.id, baseline]));
  const pairs: SVOBenchmarkPairResult[] = [];
  for (const baselineId of [...new Set(plan.runs.map(({ baselineId }) => baselineId))]) {
    const baseline = baselineById.get(baselineId);
    if (!baseline) throw new Error(`Unknown SVO benchmark baseline ${baselineId}`);
    for (let pairIndex = 0; pairIndex < plan.pairCount; pairIndex += 1) {
      const pair = results.filter((result) => result.plan.baselineId === baselineId && result.plan.pairIndex === pairIndex);
      if (pair.length !== 2) throw new Error(`${baselineId} pair ${pairIndex} is incomplete`);
      pairs.push(validatePair(baseline, pairIndex, pair[0], pair[1]));
    }
  }
  return Object.freeze({
    schemaVersion: SVO_BENCHMARK_SCHEMA_VERSION,
    plan,
    runs: Object.freeze(results),
    pairs: Object.freeze(pairs),
    aggregates: aggregateRuns(results),
  });
}
