/**
 * Tall-air A/B for Uniform Geometric: what does empty air above the liquid cost?
 *
 * Arms are the SAME scene at the SAME cell size with the SAME absolute reservoir
 * box, floor footprint and dt; only the container height (and therefore the
 * empty air above the liquid) changes. See `cpu-census.mts` for the fixture.
 *
 * All arms are built up front and advanced in LOCKSTEP, one frame each per round
 * with the arm order rotating, because this machine's uniform Dawn lane is
 * bimodal (identical code runs 48-86 ms). The deliverable is the paired,
 * same-round comparison, not two medians from two captures.
 *
 * Per stage it reports the solver's own hardware-timestamp seams (spliced into
 * real passes, so an instrumented advance encodes the same passes as a plain
 * one) plus the compute-pass count each seam owns, and the host encode time.
 *
 * Run (under the repository WebGPU lease, one Dawn process at a time):
 *   node --import tsx docs/research/uniform-geometric-tall-air-2026-09-19/probe-tall-air-dawn.mts
 *
 * Env: FLUID_TALL_MULTIPLES (default "1,4"), FLUID_TALL_FRAMES (default 70),
 *      FLUID_TALL_OUT (JSON path), FLUID_TALL_STUB (see `stub-*` arms below).
 */
import assert from "node:assert/strict";
import { writeFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { resolveMethodValues } from "../../../lib/core/method-contract";
import { GPUStageTimestampRecorder, type GPUTimestampPhase, type PerformanceTrace } from "../../../lib/core/performance-trace";
import { usePerformanceInstrumentationStore } from "../../../lib/core/stores/performance-instrumentation-store";
import { requiredFluidDeviceLimits } from "../../../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../../../lib/harness/webgpu-smoke-isolation";
import { uniformVolumeMethod } from "../../../lib/methods/uniform/uniform-volume-method";
import type { GPUEulerianInfo } from "../../../lib/core/webgpu-eulerian";
import { tallAirScene } from "./tall-air-scene.mjs";

/**
 * Arm specs. "4" is the shipped Uniform Geometric default (two-level sampler,
 * extension shell tiles, fine-tile advection/projection, live-set transport,
 * 4h sharpening map). "4d" is the SAME height with every one of those gates
 * turned off -- the dense control the method retains, unchanged.
 */
const ARM_SPECS = (process.env.FLUID_TALL_MULTIPLES ?? "1,4").split(",").map((spec) => {
  // Suffixes may be combined in any order: "d" = the dense control (every
  // shipped work-map gate off), "w" = Uniform Geometric's solve window on
  // (`activeRegion: "on"`), "p" = the CM11a hierarchy planned on the window
  // instead of the domain (`pressureWindow: "window"`, needs "w").
  // "4" is the shipped default, "4w" is the same arm with the window and the
  // whole-domain hierarchy -- which is exactly what the solve-window capture
  // measured -- and "4wp" adds the window-local hierarchy on top.
  let rest = spec.trim();
  let dense = false;
  let windowed = false;
  let pressureWindow = false;
  for (;;) {
    if (rest.endsWith("p")) { pressureWindow = true; rest = rest.slice(0, -1); continue; }
    if (rest.endsWith("w")) { windowed = true; rest = rest.slice(0, -1); continue; }
    if (rest.endsWith("d")) { dense = true; rest = rest.slice(0, -1); continue; }
    break;
  }
  return { multiple: Number(rest), dense, windowed, pressureWindow };
});
const MULTIPLES = ARM_SPECS.map((a) => a.multiple);
const FRAMES = Number(process.env.FLUID_TALL_FRAMES ?? 70);
const WARMUP = Number(process.env.FLUID_TALL_WARMUP ?? 4);

// ---------------------------------------------------------------- pass ledger
interface PassLedger { open: number; direct: number; indirect: number; phases: { label: string; passes: number }[] }
const ledgers = new WeakMap<GPUStageTimestampRecorder, PassLedger>();
let activeLedger: PassLedger | undefined;
const prototype = GPUStageTimestampRecorder.prototype;
const baseInstrument = prototype.instrument;
prototype.instrument = function instrumentCounting(encoder: GPUCommandEncoder) {
  const instrumented = baseInstrument.call(this, encoder);
  const ledger: PassLedger = { open: 0, direct: 0, indirect: 0, phases: [] };
  ledgers.set(this, ledger); activeLedger = ledger;
  return new Proxy(instrumented, { get(target, property) {
    if (property === "beginComputePass" || property === "beginRenderPass") {
      const begin = (Reflect.get(target, property) as (...a: unknown[]) => unknown).bind(target);
      return (...a: unknown[]) => {
        ledger.open += 1;
        const pass = begin(...a) as object;
        if (property !== "beginComputePass") return pass;
        // Count how many of the step's dispatches are indirect: on Dawn/Metal an
        // indirect launch costs ~15-25 us against ~3-6 us for a direct one, so
        // the launch mix is the price of the solve window.
        return new Proxy(pass, { get(pTarget, pProperty) {
          const value = Reflect.get(pTarget, pProperty);
          if (typeof value !== "function") return value;
          const bound = (value as (...a: unknown[]) => unknown).bind(pTarget);
          if (pProperty === "dispatchWorkgroups") {
            return (...a: unknown[]) => { ledger.direct += 1; return bound(...a); };
          }
          if (pProperty === "dispatchWorkgroupsIndirect") {
            return (...a: unknown[]) => { ledger.indirect += 1; return bound(...a); };
          }
          return bound;
        } });
      };
    }
    return Reflect.get(target, property);
  } }) as GPUCommandEncoder;
};
const baseComplete = prototype.completePhase;
prototype.completePhase = function completePhaseCounting(encoder: GPUCommandEncoder, phase: GPUTimestampPhase) {
  const ledger = ledgers.get(this);
  if (ledger) { ledger.phases.push({ label: phase.label, passes: ledger.open }); ledger.open = 0; }
  return baseComplete.call(this, encoder, phase);
};
const baseFinal = prototype.completeFinalPhaseOnNextPass;
prototype.completeFinalPhaseOnNextPass = function completeFinalCounting(phase: GPUTimestampPhase) {
  const ledger = ledgers.get(this);
  if (ledger) { ledger.phases.push({ label: phase.label, passes: ledger.open + 1 }); ledger.open = 0; }
  return baseFinal.call(this, phase);
};

// ------------------------------------------------------------------ statistics
const percentile = (sorted: readonly number[], q: number) =>
  sorted.length === 0 ? 0 : sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil(q * sorted.length) - 1))]!;
function stats(values: readonly number[]) {
  const sorted = [...values].sort((a, b) => a - b);
  const round = (v: number) => Number(v.toFixed(4));
  return { n: sorted.length, min: round(sorted[0] ?? 0), p25: round(percentile(sorted, 0.25)),
    median: round(percentile(sorted, 0.5)), p75: round(percentile(sorted, 0.75)),
    p90: round(percentile(sorted, 0.9)), max: round(sorted[sorted.length - 1] ?? 0) };
}

interface Arm {
  readonly multiple: number;
  readonly label: string;
  readonly solver: { advanceTo(t: number): boolean; readStats(): Promise<GPUEulerianInfo>; readonly info: GPUEulerianInfo; dispose?(): void };
  readonly structure: Record<string, unknown>;
  readonly wall_ms: number[];
  readonly cpu_ms: number[];
  readonly gpuTotal_ms: number[];
  readonly phase_ms: Map<string, number[]>;
  readonly phasePasses: Map<string, number[]>;
  readonly series: Record<string, unknown>[];
}

await acquireWebGPUExclusiveLock("dawn-probe", "uniform geometric tall-air A/B");
let device: GPUDevice | undefined;
try {
  usePerformanceInstrumentationStore.getState().setMode("timeline");
  const modulePath = process.env.WEBGPU_NODE_MODULE
    ?? fileURLToPath(new URL("../../../node_modules/webgpu/index.js", import.meta.url));
  const dawn = await import(pathToFileURL(modulePath).href) as { create(o: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter, "no WebGPU adapter");
  const features: GPUFeatureName[] = [];
  if (adapter.features.has("timestamp-query")) features.push("timestamp-query");
  device = await adapter.requestDevice({ requiredFeatures: features, requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => { validationErrors.push((event as GPUUncapturedErrorEvent).error.message); });
  await GPUStageTimestampRecorder.prepare(device);

  const tiledValues = resolveMethodValues(uniformVolumeMethod, "balanced", {});
  const denseValues = resolveMethodValues(uniformVolumeMethod, "balanced", {
    twoLevelVelocity: "off", twoLevelExtension: "dense", twoLevelAdvection: "dense",
    transportWorkMap: "dense", sharpeningWorkMap: "off",
  });
  const arms: Arm[] = [];
  for (const spec of ARM_SPECS) {
    const { multiple, dense, windowed, pressureWindow } = spec;
    const values = windowed
      ? resolveMethodValues(uniformVolumeMethod, "balanced", {
          ...(dense
            ? { twoLevelVelocity: "off", twoLevelExtension: "dense", twoLevelAdvection: "dense",
                transportWorkMap: "dense", sharpeningWorkMap: "off" }
            : {}),
          activeRegion: "on",
          pressureWindow: pressureWindow ? "window" : "domain",
          // Diagnostic lever: FLUID_TALL_OVERRIDES is a JSON object of extra
          // param values, applied to the windowed arms only. It is how the
          // window's padding floor is moved without editing a default.
          ...JSON.parse(process.env.FLUID_TALL_OVERRIDES ?? "{}") as Record<string, string>,
        })
      : dense ? denseValues : tiledValues;
    const scene = tallAirScene(multiple);
    const solver = await uniformVolumeMethod.createSolverAsync!(device, scene, "balanced", values, undefined, () => {}) as unknown as Arm["solver"];
    // Structural facts: the pressure plan is walked here, on the host, so the
    // per-level pass census costs no GPU time at all.
    const multigrid = (solver as unknown as { pressureMultigrid: {
      plan?: readonly { entryPoint: string; stage: string; activeLevel: number; workgroups: readonly [number, number, number] }[];
      levels: readonly { dimensions: readonly [number, number, number] }[];
      cycleBoundaries?: readonly number[]; levelCount: number; planPassCount?: number;
    } }).pressureMultigrid;
    const extrapolator = (solver as unknown as { velocityExtrapolator: {
      hierarchyLevelCount: number; frontPasses: number; encodedPassCount: number; allocatedBytes?: number } }).velocityExtrapolator;
    const plan = multigrid.plan ?? [];
    const byEntry: Record<string, { passes: number; levels: Record<number, number>; workgroups: number }> = {};
    const byLevel: Record<number, { passes: number; workgroups: number }> = {};
    const byStage: Record<string, number> = {};
    for (const d of plan) {
      const wg = d.workgroups[0] * d.workgroups[1] * d.workgroups[2];
      (byEntry[d.entryPoint] ??= { passes: 0, levels: {}, workgroups: 0 });
      byEntry[d.entryPoint]!.passes += 1;
      byEntry[d.entryPoint]!.workgroups += wg;
      byEntry[d.entryPoint]!.levels[d.activeLevel] = (byEntry[d.entryPoint]!.levels[d.activeLevel] ?? 0) + 1;
      (byLevel[d.activeLevel] ??= { passes: 0, workgroups: 0 });
      byLevel[d.activeLevel]!.passes += 1; byLevel[d.activeLevel]!.workgroups += wg;
      byStage[d.stage] = (byStage[d.stage] ?? 0) + 1;
    }
    arms.push({ multiple,
      label: `${multiple}x${dense ? "-dense" : ""}${windowed ? "-window" : ""}${pressureWindow ? "-plattice" : ""}`,
      solver,
      structure: {
        dense, windowed, pressureWindow, activeRegion: values.activeRegion,
        pressureWindowValue: values.pressureWindow,
        sceneId: scene.sceneId, height_m: scene.container.height_m,
        dims: [solver.info.nx, solver.info.ny, solver.info.nz], cellCount: solver.info.cellCount,
        pressureLevels: multigrid.levelCount,
        pressureLevelDimensions: multigrid.levels.map((l) => l.dimensions),
        pressurePlanPasses: multigrid.planPassCount,
        pressurePlanByEntry: byEntry, pressurePlanByLevel: byLevel, pressurePlanByStage: byStage,
        pressureCycleBoundaries: multigrid.cycleBoundaries,
        extensionHierarchyLevels: extrapolator.hierarchyLevelCount,
        extensionFrontPasses: extrapolator.frontPasses,
        extensionPassesPerInvocation: extrapolator.encodedPassCount,
        pipelineFacts: solver.info.uniformPipelineFacts,
        allocatedBytes: solver.info.allocatedBytes,
      },
      wall_ms: [], cpu_ms: [], gpuTotal_ms: [], phase_ms: new Map(), phasePasses: new Map(), series: [] });
  }

  const seen = new Map<Arm, Set<number>>(arms.map((a) => [a, new Set<number>()]));
  for (let frame = 1; frame <= FRAMES; frame += 1) {
    const order = arms.map((_, i) => arms[(i + frame) % arms.length]!);
    for (const arm of order) {
      activeLedger = undefined;
      const started = performance.now();
      assert.ok(arm.solver.advanceTo(frame / 30), `${arm.label}: advanceTo(${frame}/30) declined`);
      const cpu_ms = performance.now() - started;
      await device.queue.onSubmittedWorkDone();
      const wall_ms = performance.now() - started;
      const ledger = activeLedger as PassLedger | undefined; activeLedger = undefined;
      const dispatchMix = ledger ? { direct: ledger.direct, indirect: ledger.indirect } : undefined;
      const info = await arm.solver.readStats();
      const any = info as unknown as Record<string, any>;
      const trace = info.physicsTrace as PerformanceTrace | undefined;
      if (frame > WARMUP) {
        arm.wall_ms.push(wall_ms); arm.cpu_ms.push(cpu_ms);
        if (trace && !seen.get(arm)!.has(trace.sampleId)) {
          seen.get(arm)!.add(trace.sampleId);
          arm.gpuTotal_ms.push(trace.total_ms);
          const perPhase = new Map<string, number>();
          for (const phase of trace.phases) perPhase.set(phase.label, (perPhase.get(phase.label) ?? 0) + phase.duration_ms);
          for (const [label, ms] of perPhase) (arm.phase_ms.get(label) ?? arm.phase_ms.set(label, []).get(label)!).push(ms);
          if (ledger) for (const p of ledger.phases) (arm.phasePasses.get(p.label) ?? arm.phasePasses.set(p.label, []).get(p.label)!).push(p.passes);
        }
      }
      arm.series.push({ frame, wall_ms: Number(wall_ms.toFixed(3)), cpu_ms: Number(cpu_ms.toFixed(3)),
        traced: trace !== undefined, traceSource: trace?.measurementSource,
        volumeCellSum: any.volumeCellSum, maxSpeed: any.maxSpeed_m_s, rawVolumeDrift: any.rawVolumeDrift,
        cyclesEncoded: any.uniformPressureCyclesEncoded, passesEncoded: any.uniformPressurePassesEncoded,
        cyclesExecuted: any.uniformPressureCyclesExecuted, converged: any.uniformCM11aCycleConverged,
        fullCyclesExecuted: any.uniformCM11aFullCyclesExecuted, vCyclesExecuted: any.uniformCM11aVCyclesExecuted,
        fineResidual: any.uniformCM11aFineResidualInfinity,
        fineTiles: any.uniformTwoLevelFineTiles, shellTiles: any.uniformTwoLevelShellTiles,
        transportTiles: any.uniformTransportTiles, sharpenTiles: any.uniformSharpenTilesActive,
        tilesTotal: any.uniformTwoLevelTilesTotal, sharpenTilesTotal: any.uniformSharpenTilesTotal,
        maxDisplacement_cells: any.uniformTransportMaxDisplacement_cells,
        fimPasses: any.uniformFIMExecutedPasses, dustCells: any.uniformVolumeDustCells,
        windowFraction: any.uniformActiveRegionFraction, windowCells: any.uniformActiveRegionCellCount,
        windowMin: any.uniformActiveRegionMinimum, windowMax: any.uniformActiveRegionMaximum,
        directDispatches: dispatchMix?.direct, indirectDispatches: dispatchMix?.indirect,
        windowDispatch: any.uniformSolveWindowDispatch,
        windowClippedSteps: any.uniformSolveWindowClippedSteps,
        windowDenseSteps: any.uniformSolveWindowDenseSteps,
        windowMaxLagSteps: any.uniformSolveWindowMaxLagSteps,
        // The live CM11a instance: which lattice this step solved on, the
        // level count of its plan, and what re-planning has cost so far.
        pressureLattice: any.uniformPressureLattice,
        pressureLatticeWindowed: any.uniformPressureLatticeWindowed,
        pressureReplans: any.uniformPressureLatticeReplans,
        pressureReplanMs: any.uniformPressureLatticeReplanMs,
        pressureLevels: any.uniformPipelineFacts?.multigridLevels,
        front_m: any.front_m,
      });
    }
  }

  const report = {
    generated: new Date().toISOString(), frames: FRAMES, warmup: WARMUP,
    multiples: MULTIPLES, validationErrors,
    params: { tiled: tiledValues, dense: denseValues },
    arms: arms.map((arm) => ({
      label: arm.label, structure: arm.structure,
      wall_ms: stats(arm.wall_ms), cpu_ms: stats(arm.cpu_ms), gpuTotal_ms: stats(arm.gpuTotal_ms),
      phases: Object.fromEntries([...arm.phase_ms].map(([label, xs]) => [label, {
        ...stats(xs), passes: arm.phasePasses.get(label)?.at(-1),
        passesMedian: arm.phasePasses.has(label) ? stats(arm.phasePasses.get(label)!).median : undefined }])),
      finalStats: arm.series.at(-1), series: arm.series,
    })),
  };
  console.log(JSON.stringify(report, null, 2));
  if (process.env.FLUID_TALL_OUT) writeFileSync(process.env.FLUID_TALL_OUT, JSON.stringify(report, null, 2));
  for (const arm of report.arms) {
    console.error(`\n=== ${arm.label} ${JSON.stringify(arm.structure.dims)} cells=${arm.structure.cellCount} ===`);
    console.error(`wall med ${arm.wall_ms.median} ms | cpu med ${arm.cpu_ms.median} ms | gpu total med ${arm.gpuTotal_ms.median} ms (n=${arm.gpuTotal_ms.n})`);
    for (const [label, s] of Object.entries(arm.phases)) {
      console.error(`  ${label.padEnd(46)} ${String((s as {median:number}).median).padStart(9)} ms  passes=${(s as {passesMedian?:number}).passesMedian ?? "?"}`);
    }
  }
  for (const arm of arms) arm.solver.dispose?.();
} finally {
  device?.destroy?.();
  releaseWebGPUExclusiveLock();
}
