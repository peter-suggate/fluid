/**
 * What does one UNIFORM advance actually spend, stage by stage, on hardware?
 *
 * The sibling of `probe-sparse-cm12-stage-trace.ts` for the dense reference
 * method: it drives `minimal-power-dam-break-32` under Dawn/Metal with
 * instrumentation in "timeline" mode, collects every accepted
 * `GPUStageTimestampRecorder` sample (one per advance, gated by
 * `UNIFORM_PHYSICS_TRACE_CADENCE_MS`), and reports per-seam medians with p10/p90
 * so a single bimodal capture cannot be mistaken for a measurement.
 *
 * Three things it adds over the sparse probe:
 *  - it aggregates MANY samples rather than reporting the last one, because the
 *    Dawn timestamp tick is 65.5 us and small stages quantise hard;
 *  - it counts the compute passes each seam owns, by wrapping the recorder's
 *    encoder proxy in the probe (the solver is never modified);
 *  - it captures the work counts that give the timings a denominator: lattice,
 *    liquid cells, active region, FIM sweeps, CM11a levels and pass schedule.
 *
 * Both `densityPostProcessing` variants are captured, so the Sec. 3.8
 * reconstruction can be priced against the wall-film-only lane the Dawn
 * benchmarks run.
 *
 * Run:
 *   WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
 *   node --import tsx tools/probe-uniform-stage-trace.ts
 *
 * Optional env: FLUID_PROBE_ADVANCES (default 40), FLUID_PROBE_VARIANTS
 * ("off", "on", or "off,on"), FLUID_PROBE_OUT (JSON output path).
 */
import assert from "node:assert/strict";
import { writeFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import {
  fluidPipelinePhaseCosts,
  measureFluidPipelineStage,
} from "../lib/core/fluid-pipeline";
import { resolveMethodValues } from "../lib/core/method-contract";
import {
  GPUStageTimestampRecorder,
  type GPUTimestampPhase,
  type PerformanceTrace,
} from "../lib/core/performance-trace";
import { createMinimalPowerDamBreak32Scene } from "../lib/core/scenes";
import { usePerformanceInstrumentationStore } from "../lib/core/stores/performance-instrumentation-store";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import { uniformMethod } from "../lib/methods/uniform/method";
import {
  UNIFORM_ADVANCE_PHASE,
  UNIFORM_FLUID_PIPELINE,
} from "../lib/methods/uniform/webgpu-uniform-reference";

const SAMPLED_ADVANCES = Number(process.env.FLUID_PROBE_ADVANCES ?? 40);
/** UNIFORM_PHYSICS_TRACE_CADENCE_MS is 100 ms; without a gap every advance declines. */
const TRACE_GAP_MS = 115;
/** The first accepted sample still pays lazily-created resources. */
const WARMUP_SAMPLES = 1;
const VARIANTS = (process.env.FLUID_PROBE_VARIANTS ?? "off,on").split(",");
/** A median over fewer than two dozen samples cannot see this lane's bimodality. */
const MIN_SAMPLES = Number(process.env.FLUID_PROBE_MIN_SAMPLES ?? 24);

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

function percentile(sorted: readonly number[], q: number): number {
  if (sorted.length === 0) return 0;
  const rank = Math.min(sorted.length - 1, Math.max(0, Math.ceil(q * sorted.length) - 1));
  return sorted[rank]!;
}
function stats(values: readonly number[]) {
  const sorted = [...values].sort((a, b) => a - b);
  return {
    n: sorted.length,
    p10: Number(percentile(sorted, 0.1).toFixed(4)),
    median: Number(percentile(sorted, 0.5).toFixed(4)),
    p90: Number(percentile(sorted, 0.9).toFixed(4)),
    min: Number((sorted[0] ?? 0).toFixed(4)),
    max: Number((sorted[sorted.length - 1] ?? 0).toFixed(4)),
  };
}

// --- Per-seam compute-pass counting -----------------------------------------
// The recorder wraps the frame encoder to splice timestampWrites into passes;
// wrapping that proxy once more counts the passes between two seam calls, which
// is exactly the set of passes a seam's interval measures.
interface PassLedger {
  open: number;
  readonly phases: { label: string; passes: number; final?: boolean }[];
}
const ledgers = new WeakMap<GPUStageTimestampRecorder, PassLedger>();
const passCounts = new Map<string, number[]>();
let activeLedger: PassLedger | undefined;

const recorderPrototype = GPUStageTimestampRecorder.prototype;
const baseInstrument = recorderPrototype.instrument;
recorderPrototype.instrument = function instrumentCounting(encoder: GPUCommandEncoder) {
  const instrumented = baseInstrument.call(this, encoder);
  const ledger: PassLedger = { open: 0, phases: [] };
  ledgers.set(this, ledger);
  activeLedger = ledger;
  return new Proxy(instrumented, {
    get(target, property) {
      if (property === "beginComputePass" || property === "beginRenderPass") {
        const begin = Reflect.get(target, property) as (...args: unknown[]) => unknown;
        return (...args: unknown[]) => {
          ledger.open += 1;
          return begin(...args);
        };
      }
      return Reflect.get(target, property);
    },
  }) as GPUCommandEncoder;
};
const baseCompletePhase = recorderPrototype.completePhase;
recorderPrototype.completePhase = function completePhaseCounting(
  encoder: GPUCommandEncoder,
  phase: GPUTimestampPhase,
) {
  const ledger = ledgers.get(this);
  if (ledger) {
    ledger.phases.push({ label: phase.label, passes: ledger.open });
    ledger.open = 0;
  }
  return baseCompletePhase.call(this, encoder, phase);
};
const baseCompleteFinal = recorderPrototype.completeFinalPhaseOnNextPass;
recorderPrototype.completeFinalPhaseOnNextPass = function completeFinalCounting(
  phase: GPUTimestampPhase,
) {
  const ledger = ledgers.get(this);
  if (ledger) {
    // The final seam closes on the pass encoded *after* this call.
    ledger.phases.push({ label: phase.label, passes: ledger.open + 1, final: true });
    ledger.open = 0;
  }
  return baseCompleteFinal.call(this, phase);
};
function harvestPassCounts(): void {
  if (!activeLedger) return;
  for (const phase of activeLedger.phases) {
    const bucket = passCounts.get(phase.label) ?? [];
    bucket.push(phase.passes);
    passCounts.set(phase.label, bucket);
  }
  activeLedger = undefined;
}

interface CapturedSample {
  readonly sampleId: number;
  readonly total_ms: number;
  readonly source: string;
  readonly phases: ReadonlyMap<string, number>;
  readonly trace: PerformanceTrace;
}

await acquireWebGPUExclusiveLock("dawn-probe", "tools/probe-uniform-stage-trace.ts");
try {
  usePerformanceInstrumentationStore.getState().setMode("timeline");
  const modulePath = process.env.WEBGPU_NODE_MODULE
    ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));
  const { create, globals } = await import(pathToFileURL(modulePath).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, globals);
  const gpu = create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter, "WebGPU did not expose an adapter");
  const features: GPUFeatureName[] = [];
  if (adapter.features.has("timestamp-query")) features.push("timestamp-query");
  const device = await adapter.requestDevice({
    requiredFeatures: features,
    requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    validationErrors.push(event.error.message);
  });
  // Without the compiled closing marker the first sample decodes as unsampled
  // and hardware tracing is retired for the run.
  await GPUStageTimestampRecorder.prepare(device);

  const report: Record<string, unknown>[] = [];
  for (const variant of VARIANTS) {
    passCounts.clear();
    const scene = createMinimalPowerDamBreak32Scene();
    const values = resolveMethodValues(uniformMethod, "balanced", {
      timeStep: "scene",
      densityPostProcessing: variant,
    });
    const solver = await uniformMethod.createSolverAsync!(
      device, scene, "balanced", values, undefined, () => {});
    const dt_s = scene.numerics.maxDt_s;
    const samples = new Map<number, CapturedSample>();
    const cpuSamples = new Map<number, PerformanceTrace>();
    const sources = new Set<string>();
    const liquidCells: number[] = [];
    const activeCells: number[] = [];
    const fimPasses: number[] = [];
    const fimTerminalFaces: number[] = [];
    const cm11aCoarseIterations: number[] = [];
    let lastInfo: Record<string, unknown> = {};

    const collect = (info: Record<string, unknown>) => {
      const trace = info.physicsTrace as PerformanceTrace | undefined;
      if (trace && !samples.has(trace.sampleId)) {
        sources.add(trace.measurementSource ?? "unknown");
        const phases = new Map<string, number>();
        for (const phase of trace.phases) {
          phases.set(phase.label, (phases.get(phase.label) ?? 0) + phase.duration_ms);
        }
        samples.set(trace.sampleId, {
          sampleId: trace.sampleId,
          total_ms: trace.total_ms,
          source: trace.measurementSource ?? "unknown",
          phases,
          trace,
        });
      }
      const cpu = info.physicsCPUTrace as PerformanceTrace | undefined;
      if (cpu && !cpuSamples.has(cpu.sampleId)) cpuSamples.set(cpu.sampleId, cpu);
    };

    for (let frame = 1; frame <= SAMPLED_ADVANCES; frame += 1) {
      while (!solver.advanceTo(frame * dt_s, [])) await new Promise(setImmediate);
      harvestPassCounts();
      const info = await solver.readStats() as unknown as Record<string, unknown>;
      lastInfo = info;
      collect(info);
      liquidCells.push(Number(info.volumeCellSum ?? 0));
      activeCells.push(Number(info.uniformActiveRegionCellCount ?? 0));
      fimPasses.push(Number(info.uniformFIMExecutedPasses ?? 0));
      fimTerminalFaces.push(Number(info.uniformFIMTerminalActiveFaces ?? 0));
      cm11aCoarseIterations.push(Number(info.uniformCM11aCoarseIterations ?? 0));
      await sleep(TRACE_GAP_MS);
    }
    // Trailing traces resolve asynchronously; drain them before aggregating.
    for (let attempt = 0; attempt < 10 && samples.size < SAMPLED_ADVANCES; attempt += 1) {
      await sleep(25);
      collect(await solver.readStats() as unknown as Record<string, unknown>);
    }

    const ordered = [...samples.values()].sort((a, b) => a.sampleId - b.sampleId);
    const accepted = ordered
      .filter((sample) => sample.source === "gpu-hardware-timestamp")
      .slice(WARMUP_SAMPLES);
    assert.ok(accepted.length >= MIN_SAMPLES,
      `only ${accepted.length} hardware samples were accepted (of ${ordered.length})`);
    const totals = stats(accepted.map((sample) => sample.total_ms));
    // The seam table is the encode order, so report in it — including seams the
    // scene never encodes, which are a fact about the frame rather than a gap.
    const seamKeys = Object.entries(UNIFORM_ADVANCE_PHASE) as [string, GPUTimestampPhase][];
    const known = new Set(seamKeys.map(([, phase]) => phase.label));
    const extra: string[] = [];
    for (const sample of accepted) {
      for (const label of sample.phases.keys()) {
        if (!known.has(label) && !extra.includes(label)) extra.push(label);
      }
    }
    const rowFor = (key: string | undefined, id: string | undefined, label: string) => {
      // A phase missing from a sample is a stage whose two boundaries landed on
      // the same 65.5 us tick: `partitionPerformanceTrace` drops zero-length
      // intervals, so absence means sub-tick, not unmeasured.
      const values_ms = accepted.map((sample) => sample.phases.get(label) ?? 0);
      const encoded = accepted.filter((sample) => sample.phases.has(label)).length;
      const passes = passCounts.get(label) ?? [];
      const summary = stats(values_ms);
      return {
        key,
        id: id ?? accepted.find((sample) => sample.trace.phases
          .some((phase) => phase.label === label))?.trace.phases
          .find((phase) => phase.label === label)?.id,
        label,
        ...summary,
        aboveTickFraction: Number((encoded / accepted.length).toFixed(3)),
        share: Number((summary.median / Math.max(1e-9, totals.median)).toFixed(4)),
        passes: passes.length > 0 ? stats(passes).median : 0,
        encoded: passes.length > 0,
      };
    };
    const phaseRows = [
      ...seamKeys.map(([key, phase]) => rowFor(key, phase.id, phase.label)),
      ...extra.map((label) => rowFor(undefined, undefined, label)),
    ];
    // The pipeline panel's stage rollup, measured the way the panel measures it.
    const stageRows = UNIFORM_FLUID_PIPELINE.stages.map((stage) => {
      const perSample = accepted.map((sample) => {
        const costs = fluidPipelinePhaseCosts(sample.trace);
        const measurement = measureFluidPipelineStage(
          stage, UNIFORM_FLUID_PIPELINE.stages, costs, sample.total_ms, "on");
        return measurement.duration_ms ?? 0;
      });
      const summary = stats(perSample);
      return {
        stage: stage.id,
        band: stage.band,
        label: stage.label,
        ...summary,
        share: Number((summary.median / Math.max(1e-9, totals.median)).toFixed(4)),
      };
    });
    const cpuAccepted = [...cpuSamples.values()]
      .sort((a, b) => a.sampleId - b.sampleId)
      .slice(WARMUP_SAMPLES);
    const cpuLabels: string[] = [];
    for (const trace of cpuAccepted) {
      for (const phase of trace.phases) if (!cpuLabels.includes(phase.label)) cpuLabels.push(phase.label);
    }
    const cpuRows = cpuLabels.map((label) => ({
      label,
      ...stats(cpuAccepted.map((trace) => trace.phases
        .filter((phase) => phase.label === label)
        .reduce((sum, phase) => sum + phase.duration_ms, 0))),
    }));

    const closure = accepted.map((sample) => sample.total_ms
      - sample.trace.phases.reduce((sum, phase) => sum + phase.duration_ms, 0));
    // A dam break is not a steady state: the narrow band widens as the front
    // runs, so the raw per-sample series is published beside the medians.
    const series = accepted.map((sample) => ({
      sampleId: sample.sampleId,
      total_ms: Number(sample.total_ms.toFixed(4)),
      ...Object.fromEntries(seamKeys
        .map(([key, phase]) => [key, Number((sample.phases.get(phase.label) ?? 0).toFixed(4))])
        .filter(([, value]) => (value as number) > 0)),
    }));
    report.push({
      variant: `densityPostProcessing=${variant}`,
      sceneId: scene.sceneId,
      timestampQuery: features.length > 0,
      measurementSources: [...sources],
      advances: SAMPLED_ADVANCES,
      hardwareSamples: accepted.length,
      warmupSamplesDropped: WARMUP_SAMPLES,
      dt_s,
      total_ms: totals,
      cpuTotal_ms: stats(cpuAccepted.map((trace) => trace.total_ms)),
      closureError_ms: stats(closure),
      phases: phaseRows,
      stages: stageRows,
      cpuPhases: cpuRows,
      work: {
        lattice: { nx: lastInfo.nx, ny: lastInfo.ny, nz: lastInfo.nz, cells: lastInfo.cellCount },
        cellSize_m: lastInfo.cellSize_m,
        dt_s,
        substepsPerAdvance: lastInfo.lastSubsteps,
        encodedSteps: lastInfo.encodedSteps,
        allocatedBytes: lastInfo.allocatedBytes,
        pressureSolver: lastInfo.pressureSolver,
        pipelineFacts: lastInfo.uniformPipelineFacts,
        liquidCells_volumeCellSum: stats(liquidCells),
        activeRegionCells: stats(activeCells),
        activeRegionFraction: lastInfo.uniformActiveRegionFraction,
        activeRegionMinimum: lastInfo.uniformActiveRegionMinimum,
        activeRegionMaximum: lastInfo.uniformActiveRegionMaximum,
        fimExecutedPasses: stats(fimPasses),
        fimTerminalActiveFaces: stats(fimTerminalFaces),
        cm11aCoarseIterations: stats(cm11aCoarseIterations),
        cm11aConverged: lastInfo.uniformCM11aConverged,
        cm11aResidualInfinity: lastInfo.uniformCM11aResidualInfinity,
        cm11aFineResidualInfinity: lastInfo.uniformCM11aFineResidualInfinity,
        cm11aCoarseActiveRows: lastInfo.uniformCM11aCoarseActiveRows,
        cm11aCoarseFreeRows: lastInfo.uniformCM11aCoarseFreeRows,
        maxSpeed_m_s: lastInfo.maxSpeed_m_s,
        gammaDiffusionIterations: values.gammaDiffusionIterations,
        velocityTransport: values.velocityTransport,
      },
      passCountsPerAdvance: Object.fromEntries(
        [...passCounts].map(([label, counts]) => [label, stats(counts).median])),
      series,
      perFrame: {
        liquidCells: liquidCells.map((value) => Number(value.toFixed(2))),
        activeRegionCells: activeCells,
        fimExecutedPasses: fimPasses,
        fimTerminalActiveFaces: fimTerminalFaces,
        cm11aCoarseIterations: cm11aCoarseIterations,
      },
    });
    solver.destroy();
  }

  const output = {
    phase: "uniform-advance-stage-trace",
    capturedAt: new Date().toISOString(),
    adapter: (adapter as unknown as { info?: unknown }).info,
    variants: report,
    validationErrors,
  };
  console.log(JSON.stringify(output, null, 2));
  const out = process.env.FLUID_PROBE_OUT;
  if (out) writeFileSync(out, JSON.stringify(output, null, 2));
  assert.deepEqual(validationErrors, []);
} finally {
  releaseWebGPUExclusiveLock();
}
