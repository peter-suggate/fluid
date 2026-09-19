/**
 * P1: what does the lagged CM11a cycle budget cost and buy?
 *
 * Two arms of the SAME solver — `pressureCycleBudget: "lagged"` and `"fixed"` —
 * advanced in LOCKSTEP, one frame each per round in rotating order. This
 * machine's stage lane is strongly order- and thermally dependent (two
 * sequential runs of identical code differed by ~30 ms of whole-frame wall, see
 * docs/benchmarks/uniform-pressure-pass-floor-2026-09-19.md), so the deliverable
 * is the PAIRED per-frame delta on the same frame index, never two medians from
 * two captures.
 *
 * Per frame and arm it records:
 *   cpu_ms              `advanceTo` wall before any fence — the CPU encode
 *   wall_ms             the same plus `onSubmittedWorkDone` — the whole step
 *   pressure_ms         two timestamps around the first and last ENCODED
 *                       pressure pass — the pressure stage's GPU span
 *   passesEncoded       compute passes the multigrid actually encoded
 *   cyclesEncoded       the host-side budget this step used
 *   full/vCycles        cycles the GPU executed before its residual gate tripped
 *   converged           whether that gate tripped at all
 *   fineResidual        mgConvergence[10]: the finish-pass projected residual
 *                       infinity norm in s^-1. Written only when the solve did
 *                       NOT converge, so zero means "met tolerance".
 *   volumeCellSum       the diagnostics reduction's liquid volume, for drift
 *
 * Frames alternate between `clean` (no encoder proxy at all, so cpu_ms is the
 * true encode cost) and `bracket` (two timestamp writes, ~0.5 ms of a 40-140 ms
 * frame). Nothing under lib/ is modified; the probe wraps the multigrid's
 * `encode` to splice the two bracket stamps in and to read the budget the host
 * passed it.
 *
 * Run (one Dawn process at a time, foreground):
 *   FLUID_P1_SCENE=minimal-power-dam-break-64 FLUID_P1_FRAMES=60 \
 *   FLUID_P1_OUT=docs/research/uniform-pressure-granularity-2026-09-19/p1-mini64.json \
 *   node --import tsx docs/research/uniform-pressure-granularity-2026-09-19/probe-cycle-budget-dawn.mts
 */
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { createProcessRetainedDawnGPU } from "/Users/petersuggate/code/me/fluid/lib/harness/node-dawn-provider";
import { managedGPUDevice } from "/Users/petersuggate/code/me/fluid/lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "/Users/petersuggate/code/me/fluid/lib/core/webgpu-device-limits";
import {
  acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock, readWebGPUExclusiveLockHolder,
} from "/Users/petersuggate/code/me/fluid/lib/harness/webgpu-smoke-isolation";
import { sceneDocument } from "/Users/petersuggate/code/me/fluid/lib/core/scene-definition";
import { getSceneDefinition } from "/Users/petersuggate/code/me/fluid/lib/core/scenes";
import { uniformVolumeMethod } from "/Users/petersuggate/code/me/fluid/lib/methods/uniform/uniform-volume-method";
import { resolveMethodValues } from "/Users/petersuggate/code/me/fluid/lib/core/method-contract";
import type { WebGPUUniformReferenceSolver } from "/Users/petersuggate/code/me/fluid/lib/methods/uniform/webgpu-uniform-reference";

type Arm = "lagged" | "fixed";
type Mode = "clean" | "bracket";

const FRAMES = Number(process.env.FLUID_P1_FRAMES ?? 60);
const WARMUP = 3;
const SCENE = process.env.FLUID_P1_SCENE ?? "minimal-power-dam-break-64";
const OUT = process.env.FLUID_P1_OUT;
const ARMS: Arm[] = (process.env.FLUID_P1_ARMS?.split(",") as Arm[] | undefined) ?? ["lagged", "fixed"];

const sortNumbers = (xs: number[]) => [...xs].sort((a, b) => a - b);
const median = (xs: number[]) => {
  if (xs.length === 0) return Number.NaN;
  const s = sortNumbers(xs);
  return (s[Math.floor((s.length - 1) / 2)]! + s[Math.floor(s.length / 2)]!) / 2;
};
const quantile = (xs: number[], q: number) => {
  if (xs.length === 0) return Number.NaN;
  const s = sortNumbers(xs);
  return s[Math.min(s.length - 1, Math.max(0, Math.round(q * (s.length - 1))))]!;
};
const spread = (xs: number[]) => ({
  n: xs.length, min: xs.length ? sortNumbers(xs)[0]! : Number.NaN,
  p25: quantile(xs, 0.25), median: median(xs), p75: quantile(xs, 0.75),
  max: xs.length ? sortNumbers(xs).at(-1)! : Number.NaN,
  mean: xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : Number.NaN,
});
const delay = (ms: number) => new Promise((done) => setTimeout(done, ms));

interface MultigridAccess {
  readonly diagnostics: GPUBuffer;
  encode(encoder: GPUCommandEncoder, group: GPUBindGroup,
    boundary?: (stage: string) => void, cycleBudget?: number): void;
  encodedPassCount(cycleBudget?: number): number | undefined;
  readonly planPassCount?: number;
  readonly cycleCount: number;
}
interface SolverAccess {
  readonly pressureMultigrid: MultigridAccess;
  readonly reductions: GPUBuffer;
}

async function acquireWithWait(): Promise<void> {
  for (let attempt = 0; ; attempt += 1) {
    const holder = await readWebGPUExclusiveLockHolder();
    if (holder) {
      if (!holder.alive) {
        throw new Error(`GPU lock held by a dead owner (${holder.description}); clear it by hand and rerun`);
      }
      if (attempt % 6 === 0) console.error(`waiting for GPU lock: ${holder.description}`);
      await delay(10_000);
      continue;
    }
    try { await acquireWebGPUExclusiveLock("dawn-probe", "uniform P1 cycle budget"); return; }
    catch { await delay(5_000); }
  }
}

interface Sample {
  frame: number; mode: Mode; arm: Arm;
  cpu_ms: number; wall_ms: number; pressure_ms?: number;
  passesEncoded: number; cyclesEncoded: number;
  fullCycles: number; vCycles: number; cyclesExecuted: number; converged: boolean;
  fineResidual: number; cycleResidual: number;
  volumeCellSum: number; maxSpeed_m_s: number; advanced: boolean;
}

interface ArmRun {
  arm: Arm;
  solver: WebGPUUniformReferenceSolver;
  mg: MultigridAccess;
  reductions: GPUBuffer;
  bracketSet: GPUQuerySet; bracketResolve: GPUBuffer; bracketStaging: GPUBuffer;
  diagnosticsStaging: GPUBuffer; reductionStaging: GPUBuffer;
  setMode(mode: Mode): void;
  lastEncoded(): { passes: number; cycles: number };
  samples: Sample[];
}

await acquireWithWait();
let device: GPUDevice | undefined;
const report: Record<string, unknown> = {
  tool: "docs/research/uniform-pressure-granularity-2026-09-19/probe-cycle-budget-dawn.mts",
  generatedAt: new Date().toISOString(), scene: SCENE, frames: FRAMES,
  warmupFramesExcluded: WARMUP, arms: ARMS,
  protocol: "arms advanced in lockstep, one frame each per round, rotating order; paired per-frame deltas",
};
try {
  const dawn = await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);
  Object.assign(globalThis, dawn.globals);
  const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]);
  const adapter = await gpu.requestAdapter();
  assert.ok(adapter); assert.ok(adapter.features.has("timestamp-query"));
  device = managedGPUDevice(await adapter.requestDevice({ requiredFeatures: ["timestamp-query"],
    requiredLimits: requiredFluidDeviceLimits(adapter.limits) }), { requireWorkerRealm: false });
  const errors: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    const typed = event as GPUUncapturedErrorEvent & { preventDefault(): void };
    typed.preventDefault(); errors.push(typed.error.message); console.error(typed.error.message);
  });

  const runs: ArmRun[] = [];
  for (const arm of ARMS) {
    const scene = structuredClone(sceneDocument(getSceneDefinition(SCENE)));
    const values = resolveMethodValues(uniformVolumeMethod, "balanced", { pressureCycleBudget: arm });
    const solver = await uniformVolumeMethod.createSolverAsync!(device, scene, "balanced", values,
      undefined, () => {}) as WebGPUUniformReferenceSolver;
    const access = solver as unknown as SolverAccess;
    const mg = access.pressureMultigrid;

    const bracketSet = device.createQuerySet({ label: `p1 bracket ${arm}`, type: "timestamp", count: 4 });
    const bracketResolve = device.createBuffer({ size: 256,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const bracketStaging = device.createBuffer({ size: 32,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    // mgConvergence words 10..18: final fine residual, gap, ... cycle residual,
    // stopped, Full-Cycles executed, V-Cycles executed.
    const diagnosticsStaging = device.createBuffer({ size: 36,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const reductionStaging = device.createBuffer({ size: 32,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    let mode: Mode = "clean";
    let encodedPasses = mg.planPassCount ?? 0;
    let encodedCycles = mg.cycleCount;
    const originalEncode = mg.encode.bind(mg);
    mg.encode = (encoder, group, boundary, cycleBudget) => {
      // The host decides the budget and hands it straight to us, so the probe
      // knows the encoded length BEFORE the stream exists — which is what the
      // closing bracket stamp needs.
      encodedCycles = cycleBudget ?? mg.cycleCount;
      encodedPasses = mg.encodedPassCount(cycleBudget) ?? (mg.planPassCount ?? 0);
      if (mode === "clean") { originalEncode(encoder, group, boundary, cycleBudget); return; }
      const last = encodedPasses - 1;
      let index = 0;
      const proxy = new Proxy(encoder, {
        get(target, key) {
          if (key === "beginComputePass") {
            return (descriptor?: GPUComputePassDescriptor) => {
              const i = index; index += 1;
              let writes: GPUComputePassTimestampWrites | undefined;
              if (i === 0) {
                writes = { querySet: bracketSet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 };
              } else if (i === last) {
                writes = { querySet: bracketSet, beginningOfPassWriteIndex: 2, endOfPassWriteIndex: 3 };
              }
              return target.beginComputePass(writes ? { ...descriptor, timestampWrites: writes } : descriptor);
            };
          }
          const value = Reflect.get(target, key, target);
          return typeof value === "function" ? value.bind(target) : value;
        },
      });
      originalEncode(proxy as GPUCommandEncoder, group, boundary, cycleBudget);
      assert.equal(index, encodedPasses,
        `encoded ${index} pressure passes but encodedPassCount said ${encodedPasses}`);
    };

    runs.push({ arm, solver, mg, reductions: access.reductions,
      bracketSet, bracketResolve, bracketStaging, diagnosticsStaging, reductionStaging,
      setMode: (next) => { mode = next; },
      lastEncoded: () => ({ passes: encodedPasses, cycles: encodedCycles }),
      samples: [] });
  }

  for (let frame = 1; frame <= FRAMES; frame += 1) {
    const mode: Mode = frame % 2 === 0 ? "bracket" : "clean";
    const order = runs.map((_, index) => runs[(index + frame) % runs.length]!);
    for (const run of order) {
      run.setMode(mode);
      const start = performance.now();
      const advanced = run.solver.advanceTo(frame / 30);
      const cpu_ms = performance.now() - start;
      await device.queue.onSubmittedWorkDone();
      const wall_ms = performance.now() - start;

      const encoder = device.createCommandEncoder();
      if (mode === "bracket") {
        encoder.resolveQuerySet(run.bracketSet, 0, 4, run.bracketResolve, 0);
        encoder.copyBufferToBuffer(run.bracketResolve, 0, run.bracketStaging, 0, 32);
      }
      encoder.copyBufferToBuffer(run.mg.diagnostics, 40, run.diagnosticsStaging, 0, 36);
      encoder.copyBufferToBuffer(run.reductions, 0, run.reductionStaging, 0, 32);
      device.queue.submit([encoder.finish()]);
      await run.diagnosticsStaging.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(run.diagnosticsStaging.getMappedRange().slice(0));
      run.diagnosticsStaging.unmap();
      await run.reductionStaging.mapAsync(GPUMapMode.READ);
      const reduction = new Uint32Array(run.reductionStaging.getMappedRange().slice(0));
      run.reductionStaging.unmap();
      const bits = (value: number) => new Float32Array(new Uint32Array([value]).buffer)[0]!;
      const encoded = run.lastEncoded();
      const sample: Sample = {
        frame, mode, arm: run.arm, cpu_ms, wall_ms, advanced,
        passesEncoded: encoded.passes, cyclesEncoded: encoded.cycles,
        fineResidual: bits(words[0]!), cycleResidual: bits(words[5]!),
        converged: words[6] === 1, fullCycles: words[7]!, vCycles: words[8]!,
        cyclesExecuted: words[7]! + words[8]!,
        volumeCellSum: reduction[3]! / 2048, maxSpeed_m_s: bits(reduction[2]!),
      };
      if (mode === "bracket") {
        await run.bracketStaging.mapAsync(GPUMapMode.READ);
        const stamps = new BigUint64Array(run.bracketStaging.getMappedRange().slice(0), 0, 4);
        run.bracketStaging.unmap();
        sample.pressure_ms = stamps[0]! === 0n || stamps[3]! < stamps[0]!
          ? Number.NaN : Number(stamps[3]! - stamps[0]!) / 1e6;
      }
      run.samples.push(sample);
    }
  }

  const armsOut: Record<string, unknown> = {};
  const reference = runs.find((r) => r.arm === "fixed");
  for (const run of runs) {
    const measured = run.samples.filter((s) => s.frame > WARMUP);
    const clean = measured.filter((s) => s.mode === "clean");
    const bracket = measured.filter((s) => s.mode === "bracket");
    const unconverged = measured.filter((s) => !s.converged);
    let paired: Record<string, unknown> | undefined;
    if (reference && run.arm !== "fixed") {
      const byFrame = new Map(reference.samples.map((s) => [s.frame, s]));
      const pressureDelta: number[] = [], cpuDelta: number[] = [], wallDelta: number[] = [];
      const pressureRatio: number[] = [];
      for (const s of bracket) {
        const other = byFrame.get(s.frame);
        if (!other || !Number.isFinite(other.pressure_ms!) || !Number.isFinite(s.pressure_ms!)) continue;
        pressureDelta.push(s.pressure_ms! - other.pressure_ms!);
        pressureRatio.push(s.pressure_ms! / other.pressure_ms!);
      }
      for (const s of clean) {
        const other = byFrame.get(s.frame);
        if (!other) continue;
        cpuDelta.push(s.cpu_ms - other.cpu_ms);
        wallDelta.push(s.wall_ms - other.wall_ms);
      }
      paired = { pressureDelta_ms: spread(pressureDelta), pressureRatio: spread(pressureRatio),
        cpuEncodeDelta_ms: spread(cpuDelta), wallDelta_ms: spread(wallDelta) };
    }
    armsOut[run.arm] = {
      cpuEncode_ms: spread(clean.map((s) => s.cpu_ms)),
      wall_ms: spread(clean.map((s) => s.wall_ms)),
      pressure_ms: spread(bracket.map((s) => s.pressure_ms!).filter(Number.isFinite)),
      passesEncoded: spread(measured.map((s) => s.passesEncoded)),
      cyclesEncoded: spread(measured.map((s) => s.cyclesEncoded)),
      cyclesExecuted: spread(measured.map((s) => s.cyclesExecuted)),
      convergedFraction: measured.filter((s) => s.converged).length / Math.max(1, measured.length),
      unconvergedSteps: unconverged.length,
      worstFineResidual: Math.max(0, ...measured.map((s) => s.fineResidual)),
      fineResidualUnconverged: spread(unconverged.map((s) => s.fineResidual)),
      volumeCellSum: spread(measured.map((s) => s.volumeCellSum)),
      finalVolumeCellSum: measured.at(-1)?.volumeCellSum,
      maxSpeed_m_s: spread(measured.map((s) => s.maxSpeed_m_s)),
      paired,
      series: measured.map((s) => ({ frame: s.frame, mode: s.mode,
        cpu_ms: Number(s.cpu_ms.toFixed(3)), wall_ms: Number(s.wall_ms.toFixed(3)),
        pressure_ms: s.pressure_ms === undefined ? undefined : Number(s.pressure_ms.toFixed(3)),
        passesEncoded: s.passesEncoded, cyclesEncoded: s.cyclesEncoded,
        cyclesExecuted: s.cyclesExecuted, converged: s.converged,
        fineResidual: s.fineResidual, volumeCellSum: Number(s.volumeCellSum.toFixed(3)),
        maxSpeed_m_s: Number(s.maxSpeed_m_s.toFixed(4)) })),
    };
  }
  report.armResults = armsOut;
  report.planPassCount = runs[0]?.mg.planPassCount;
  report.cycleCount = runs[0]?.mg.cycleCount;
  report.validationErrors = errors;
  for (const run of runs) run.solver.destroy();
  if (OUT) { mkdirSync(dirname(OUT), { recursive: true }); writeFileSync(OUT, JSON.stringify(report, null, 2)); }
  const line = (arm: string) => {
    const a = armsOut[arm] as Record<string, { median: number }> & Record<string, unknown>;
    return `${arm}: pressure ${(a.pressure_ms as { median: number }).median.toFixed(2)} ms`
      + ` · cpu ${(a.cpuEncode_ms as { median: number }).median.toFixed(2)} ms`
      + ` · passes ${(a.passesEncoded as { median: number }).median}`
      + ` · cycles enc ${(a.cyclesEncoded as { median: number }).median}`
      + ` exec ${(a.cyclesExecuted as { median: number }).median}`
      + ` · converged ${(100 * (a.convergedFraction as unknown as number)).toFixed(0)}%`
      + ` · worst residual ${(a.worstFineResidual as unknown as number).toExponential(2)}`;
  };
  console.log(`\n${SCENE}`);
  for (const arm of ARMS) console.log(line(arm));
  const laggedPaired = (armsOut.lagged as { paired?: Record<string, { median: number }> } | undefined)?.paired;
  if (laggedPaired) {
    console.log(`paired lagged-fixed: pressure ${laggedPaired.pressureDelta_ms!.median.toFixed(2)} ms`
      + ` · cpu ${laggedPaired.cpuEncodeDelta_ms!.median.toFixed(2)} ms`
      + ` · wall ${laggedPaired.wallDelta_ms!.median.toFixed(2)} ms`);
  }
  assert.deepEqual(errors, []);
} finally {
  // Release BEFORE touching the device: a throwing destroy must not orphan the
  // repository-wide lock for the next process.
  releaseWebGPUExclusiveLock();
  try { device?.destroy?.(); } catch { /* the process is exiting anyway */ }
}
