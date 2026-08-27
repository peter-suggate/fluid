import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { resolveMethodValues } from "../lib/core/method-contract";
import {
  createSparseCM12LongDamBreakScene,
  SPARSE_CM12_LONG_DAM_METHOD_PROFILE,
} from "../lib/core/scenes";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import type {
  SparseCM12DiagnosticFields,
} from "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

type ResolutionMode = "adaptive" | "all-fine" | "all-coarse";
type TimeStepMode = "paper" | "scene";

interface Lane {
  readonly id: string;
  readonly resolutionMode: ResolutionMode;
  readonly timeStep: TimeStepMode;
  readonly gammaDiffusion: "on" | "off";
  readonly surfaceSharpening: "on" | "off";
  readonly sharpeningStrength?: number;
  /** Optional whole-domain floor on cell edge, in finest-cell units. */
  readonly minimumCellSize_cells?: 2 | 4;
}

const LANES: readonly Lane[] = Object.freeze([
  { id: "baseline-paper-adaptive", resolutionMode: "adaptive", timeStep: "paper",
    gammaDiffusion: "on", surfaceSharpening: "on" },
  { id: "no-gamma-paper-adaptive", resolutionMode: "adaptive", timeStep: "paper",
    gammaDiffusion: "off", surfaceSharpening: "on" },
  { id: "no-sharpening-paper-adaptive", resolutionMode: "adaptive", timeStep: "paper",
    gammaDiffusion: "on", surfaceSharpening: "off" },
  { id: "no-conditioning-paper-adaptive", resolutionMode: "adaptive", timeStep: "paper",
    gammaDiffusion: "off", surfaceSharpening: "off" },
  { id: "sharpening-25-paper-adaptive", resolutionMode: "adaptive", timeStep: "paper",
    gammaDiffusion: "on", surfaceSharpening: "on", sharpeningStrength: 0.25 },
  { id: "sharpening-50-paper-adaptive", resolutionMode: "adaptive", timeStep: "paper",
    gammaDiffusion: "on", surfaceSharpening: "on", sharpeningStrength: 0.5 },
  { id: "sharpening-75-paper-adaptive", resolutionMode: "adaptive", timeStep: "paper",
    gammaDiffusion: "on", surfaceSharpening: "on", sharpeningStrength: 0.75 },
  { id: "baseline-paper-all-fine", resolutionMode: "all-fine", timeStep: "paper",
    gammaDiffusion: "on", surfaceSharpening: "on" },
  { id: "baseline-paper-all-coarse", resolutionMode: "all-coarse", timeStep: "paper",
    gammaDiffusion: "on", surfaceSharpening: "on" },
  { id: "baseline-paper-min-cell-2", resolutionMode: "adaptive", timeStep: "paper",
    gammaDiffusion: "on", surfaceSharpening: "on", minimumCellSize_cells: 2 },
  { id: "baseline-scene-adaptive", resolutionMode: "adaptive", timeStep: "scene",
    gammaDiffusion: "on", surfaceSharpening: "on" },
] as const);

function argument(name: string): string | undefined {
  const prefix = `--${name}=`;
  return process.argv.find((value) => value.startsWith(prefix))?.slice(prefix.length);
}

const paperSteps = Number(argument("steps") ?? 24);
if (!Number.isSafeInteger(paperSteps) || paperSteps < 1) {
  throw new RangeError("--steps must be a positive integer");
}
const checkpointEvery = Number(argument("checkpoint-every") ?? 8);
if (!Number.isSafeInteger(checkpointEvery) || checkpointEvery < 1) {
  throw new RangeError("--checkpoint-every must be a positive integer");
}
const selectedIds = new Set((argument("lanes") ?? LANES.map(({ id }) => id).join(","))
  .split(",").filter(Boolean));
const lanes = LANES.filter(({ id }) => selectedIds.has(id));
const unknownLanes = [...selectedIds].filter((id) => !LANES.some((lane) => lane.id === id));
if (unknownLanes.length > 0) throw new RangeError(`unknown lanes: ${unknownLanes.join(", ")}`);
if (lanes.length === 0) throw new RangeError("no lanes selected");

const targetTime_s = paperSteps * CM12_PAPER_DT_S;
const outputPath = resolve(argument("out")
  ?? `artifacts/sparse-cm12-long-dam-conditioning-${paperSteps}.json`);
const dawnModule = process.env.WEBGPU_NODE_MODULE;
if (!dawnModule) throw new Error("set WEBGPU_NODE_MODULE to Dawn's webgpu module");

function finiteSummary(values: readonly number[]) {
  let count = 0, sum = 0, squareSum = 0;
  let minimum = Number.POSITIVE_INFINITY, maximum = Number.NEGATIVE_INFINITY;
  for (const value of values) {
    if (!Number.isFinite(value)) continue;
    count += 1; sum += value; squareSum += value * value;
    minimum = Math.min(minimum, value); maximum = Math.max(maximum, value);
  }
  const mean = count > 0 ? sum / count : Number.NaN;
  return Object.freeze({ count, minimum, maximum, mean,
    standardDeviation: count > 0 ? Math.sqrt(Math.max(0, squareSum / count - mean * mean))
      : Number.NaN,
    rms: count > 0 ? Math.sqrt(squareSum / count) : Number.NaN });
}

function roughness(values: readonly number[]) {
  const finite = values.filter(Number.isFinite);
  let totalVariation = 0, curvatureSquareSum = 0, curvatureCount = 0;
  for (let at = 1; at < finite.length; at += 1) {
    totalVariation += Math.abs(finite[at]! - finite[at - 1]!);
  }
  for (let at = 1; at + 1 < finite.length; at += 1) {
    const second = finite[at - 1]! - 2 * finite[at]! + finite[at + 1]!;
    curvatureSquareSum += second * second; curvatureCount += 1;
  }
  return Object.freeze({ ...finiteSummary(finite),
    span: finite.length > 0 ? Math.max(...finite) - Math.min(...finite) : Number.NaN,
    meanAbsoluteStep: finite.length > 1 ? totalVariation / (finite.length - 1) : 0,
    curvatureRms: curvatureCount > 0 ? Math.sqrt(curvatureSquareSum / curvatureCount) : 0 });
}

function interpolateTopCrossing(density: Float32Array, nx: number, ny: number,
  x: number, z: number): number {
  const at = (y: number) => density[x + nx * (y + ny * z)]!;
  for (let y = ny - 2; y >= 0; y -= 1) {
    const below = at(y), above = at(y + 1);
    if (below >= 0.5 && above < 0.5) {
      const fraction = (0.5 - below) / Math.min(-1e-9, above - below);
      return y + 0.5 + Math.max(0, Math.min(1, fraction));
    }
  }
  return Number.NaN;
}

function fieldMetrics(fields: SparseCM12DiagnosticFields,
  dimensions: readonly [number, number, number], dt_s: number, cellSize_m: number) {
  const [nx, ny, nz] = dimensions;
  const { density, gamma, velocity } = fields;
  assert.equal(density.length, nx * ny * nz);
  const xMass = new Float64Array(nx);
  const xMeanDepth = new Float64Array(nx);
  const frontByZ = new Float64Array(nz).fill(Number.NaN);
  const centerSurfaceByX = new Float64Array(nx).fill(Number.NaN);
  const interfaceGammaError: number[] = [];
  let mass = 0, firstMomentX = 0, kineticEnergy = 0, maximumDensity = 0;
  let maximumCfl = 0, densityOverOneMass = 0, wetCellCount = 0;
  let densitySymmetrySquare = 0, densitySymmetryMaximum = 0, symmetryCount = 0;
  let interfaceVariation = 0;
  const index = (x: number, y: number, z: number) => x + nx * (y + ny * z);

  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
    for (let x = 0; x < nx; x += 1) {
      const cell = index(x, y, z), rho = density[cell]!;
      mass += rho; firstMomentX += rho * (x + 0.5); xMass[x] += rho;
      maximumDensity = Math.max(maximumDensity, rho);
      densityOverOneMass += Math.max(0, rho - 1);
      wetCellCount += rho >= 0.5 ? 1 : 0;
      const vx = velocity[4 * cell]!, vy = velocity[4 * cell + 1]!, vz = velocity[4 * cell + 2]!;
      const speed2 = vx * vx + vy * vy + vz * vz;
      kineticEnergy += 0.5 * Math.max(0, rho) * speed2;
      if (rho > 0.005) maximumCfl = Math.max(maximumCfl,
        Math.sqrt(speed2) * dt_s / cellSize_m);
      let interfaceCell = rho > 0.05 && rho < 0.95;
      if (x + 1 < nx) {
        const next = density[index(x + 1, y, z)]!;
        interfaceVariation += Math.abs(next - rho);
        interfaceCell ||= (rho >= 0.5) !== (next >= 0.5);
      }
      if (y + 1 < ny) {
        const next = density[index(x, y + 1, z)]!;
        interfaceVariation += Math.abs(next - rho);
        interfaceCell ||= (rho >= 0.5) !== (next >= 0.5);
      }
      if (z + 1 < nz) {
        const next = density[index(x, y, z + 1)]!;
        interfaceVariation += Math.abs(next - rho);
        interfaceCell ||= (rho >= 0.5) !== (next >= 0.5);
      }
      if (interfaceCell) interfaceGammaError.push(gamma[cell]! - 1);
      const reflected = density[index(x, y, nz - 1 - z)]!;
      const symmetryError = Math.abs(rho - reflected);
      densitySymmetrySquare += symmetryError * symmetryError;
      densitySymmetryMaximum = Math.max(densitySymmetryMaximum, symmetryError);
      symmetryCount += 1;
    }
  }
  for (let x = 0; x < nx; x += 1) xMeanDepth[x] = xMass[x]! / nz;

  // The leading edge is the farthest longitudinal column holding at least half
  // a finest-cell equivalent of vertical mass. Measuring it independently for
  // every transverse slice exposes the large front corrugations seen in the scene.
  for (let z = 1; z + 1 < nz; z += 1) {
    const depth = new Float64Array(nx);
    for (let x = 0; x < nx; x += 1) {
      for (let y = 0; y < ny; y += 1) depth[x] += density[index(x, y, z)]!;
    }
    for (let x = nx - 1; x >= 0; x -= 1) {
      if (depth[x]! < 0.5) continue;
      if (x + 1 >= nx) { frontByZ[z] = x + 0.5; break; }
      const next = depth[x + 1]!;
      const fraction = Math.max(0, Math.min(1,
        (depth[x]! - 0.5) / Math.max(1e-9, depth[x]! - next)));
      frontByZ[z] = x + 0.5 + fraction;
      break;
    }
  }
  const validFront = Array.from(frontByZ.slice(1, nz - 1)).filter(Number.isFinite);
  const meanFront = finiteSummary(validFront).mean;
  const leadingSurface: number[] = [];
  for (let x = 0; x < nx; x += 1) {
    centerSurfaceByX[x] = interpolateTopCrossing(density, nx, ny, x, Math.floor(nz / 2));
    if (Number.isFinite(meanFront) && x + 0.5 >= meanFront - 24 && x + 0.5 <= meanFront) {
      const value = centerSurfaceByX[x]!;
      if (Number.isFinite(value)) leadingSurface.push(value);
    }
  }

  const cumulativeQuantile = (fraction: number) => {
    const target = fraction * mass; let cumulative = 0;
    for (let x = 0; x < nx; x += 1) {
      cumulative += xMass[x]!;
      if (cumulative >= target) return x + 0.5;
    }
    return nx - 0.5;
  };
  const firstLeadingX = Math.max(0, Math.floor(cumulativeQuantile(0.9)));
  const leadingDepth = Array.from(xMeanDepth.slice(firstLeadingX));
  const depthVariation = leadingDepth.slice(1).reduce((sum, value, at) =>
    sum + Math.abs(value - leadingDepth[at]!), 0);
  const endpointVariation = leadingDepth.length > 1
    ? Math.abs(leadingDepth.at(-1)! - leadingDepth[0]!) : 0;

  return Object.freeze({
    massFineCells: mass,
    materialCenterXFineCells: firstMomentX / mass,
    massQuantileXFineCells: { q90: cumulativeQuantile(0.9), q95: cumulativeQuantile(0.95),
      q99: cumulativeQuantile(0.99), q999: cumulativeQuantile(0.999) },
    maximumDensity,
    densityOverOneMassFineCells: densityOverOneMass,
    wetCellCount,
    kineticEnergyProxy: kineticEnergy,
    maximumAdvectiveCfl: maximumCfl,
    interfaceVariation,
    interfaceGammaError: finiteSummary(interfaceGammaError),
    transverseFront: { ...roughness(validFront), profileXFineCells: validFront },
    leadingSurfaceHeight: { ...roughness(leadingSurface), profileYFineCells: leadingSurface },
    longitudinalDepth: { ...roughness(leadingDepth),
      firstXFineCell: firstLeadingX,
      profileMeanDepthFineCells: leadingDepth,
      excessTotalVariation: Math.max(0, depthVariation - endpointVariation) },
    densityReflectionZ: { rms: Math.sqrt(densitySymmetrySquare / symmetryCount),
      maximum: densitySymmetryMaximum },
  });
}

for (;;) {
  try {
    await acquireWebGPUExclusiveLock("dawn-probe", "sparse-cm12-long-dam-conditioning");
    break;
  } catch (error) {
    if (!(error instanceof Error)
      || !error.message.startsWith("Refusing concurrent GPU execution")) throw error;
    process.stderr.write("[conditioning-probe] GPU busy; waiting for the exclusive lane\n");
    await new Promise((resolveDelay) => setTimeout(resolveDelay, 1_000));
  }
}
let device: GPUDevice | undefined;
try {
  const dawn = await import(pathToFileURL(dawnModule).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([
    `backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`,
    "enable-dawn-features=disable_blob_cache",
  ]);
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter, "Dawn must expose a WebGPU adapter");
  device = managedGPUDevice(await adapter.requestDevice({
    requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  }), { requireWorkerRealm: false, maximumConcurrentBundles: 1 });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    event.preventDefault(); validationErrors.push(event.error.message);
  });

  const results: unknown[] = [];
  for (const lane of lanes) {
    process.stderr.write(`[conditioning-probe] starting ${lane.id}\n`);
    const started = performance.now();
    const scene = createSparseCM12LongDamBreakScene();
    if (lane.minimumCellSize_cells !== undefined) {
      scene.fluid.refinementRegions = [{
        id: `conditioning-probe-min-cell-${lane.minimumCellSize_cells}`,
        rule: "minimum-cell-size",
        minimumCellSize_cells: lane.minimumCellSize_cells,
        min_m: { x: -0.5 * scene.container.width_m, y: 0,
          z: -0.5 * scene.container.depth_m },
        max_m: { x: 0.5 * scene.container.width_m, y: scene.container.height_m,
          z: 0.5 * scene.container.depth_m },
      }];
    }
    const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
      ...SPARSE_CM12_LONG_DAM_METHOD_PROFILE.overrides,
      resolutionMode: lane.resolutionMode,
      timeStep: lane.timeStep,
      gammaDiffusion: lane.gammaDiffusion,
      surfaceSharpening: lane.surfaceSharpening,
      sharpeningStrength: lane.sharpeningStrength ?? 1,
    });
    let solver: WebGPUAdaptiveMassSolver | undefined;
    try {
      solver = await adaptiveMassMethod.createSolverAsync!(
        device, scene, "balanced", values, undefined, () => {},
      ) as WebGPUAdaptiveMassSolver;
      await solver.waitForSimulationReady();
      const dimensions = [solver.info.nx, solver.info.ny, solver.info.nz] as const;
      const checkpoints: unknown[] = [];
      let acceptedSteps = 0, simulatedTime_s = 0;
      while (simulatedTime_s < targetTime_s - 1e-10) {
        const next = lane.timeStep === "paper"
          ? Math.min(targetTime_s, simulatedTime_s + CM12_PAPER_DT_S)
          : Math.min(targetTime_s, simulatedTime_s + scene.numerics.maxDt_s);
        assert.equal(solver.advanceTo(next, []), true,
          `${lane.id} rejected advance ${acceptedSteps + 1} to ${next}`);
        simulatedTime_s = next; acceptedSteps += 1;
        const paperEquivalent = simulatedTime_s / CM12_PAPER_DT_S;
        const checkpoint = simulatedTime_s >= targetTime_s - 1e-10
          || (lane.timeStep === "paper" && acceptedSteps % checkpointEvery === 0
            && lane.resolutionMode === "adaptive");
        if (!checkpoint) continue;
        await device.queue.onSubmittedWorkDone();
        const fields = await solver.readDiagnosticFields(true);
        checkpoints.push(Object.freeze({ acceptedSteps, simulatedTime_s,
          paperEquivalentSteps: paperEquivalent,
          metrics: fieldMetrics(fields, dimensions,
            lane.timeStep === "paper" ? CM12_PAPER_DT_S : scene.numerics.maxDt_s,
            scene.voxelDomain.finestCellSize_m) }));
        process.stderr.write(`[conditioning-probe] ${lane.id} t=${simulatedTime_s.toFixed(4)} `
          + `steps=${acceptedSteps}\n`);
      }
      const [activity, stats] = await Promise.all([
        solver.readGPUActivityPolicy(), solver.readStats(),
      ]);
      const active = activity.bricks.filter((brick) => brick.active);
      results.push(Object.freeze({ lane, dimensions, targetTime_s, acceptedSteps,
        wallTime_ms: performance.now() - started, checkpoints,
        resolutionCensus: Object.fromEntries([1, 2, 4, 8].map((resolution) =>
          [resolution, active.filter((brick) => brick.acceptedResolution === resolution).length])),
        activeBrickCount: active.length,
        pressure: {
          relativeResidual: stats.pressureRelativeResidual,
          trueResidualMaximum: stats.pressureTrueResidualMaximum,
          iterationsExecuted: stats.pressureIterationsExecuted,
          iterationsEncoded: stats.pressureIterationsEncoded,
        },
      }));
    } finally {
      solver?.destroy();
    }
  }
  await device.queue.onSubmittedWorkDone();
  assert.deepEqual(validationErrors, []);
  const report = Object.freeze({
    schema: "sparse-cm12-long-dam-conditioning/v1",
    generatedAt: new Date().toISOString(),
    backend: process.env.FLUID_WEBGPU_BACKEND ?? "metal",
    paperSteps,
    targetTime_s,
    checkpointEvery,
    lanes: results,
    validationErrors,
  });
  await mkdir(dirname(outputPath), { recursive: true });
  await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify({ outputPath, report }, null, 2)}\n`);
} finally {
  device?.destroy();
  await releaseWebGPUExclusiveLock();
}
