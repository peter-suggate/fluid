import { pathToFileURL } from "node:url";
import { EulerianFluidSolver } from "../lib/eulerian-solver";
import { tallCellMethod } from "../lib/methods/tall-cell";
import type { GPUSolverInstance, SimulationMethod } from "../lib/methods/types";
import { uniformMethod } from "../lib/methods/uniform";
import { initializeRigidBodies } from "../lib/rigid-body";
import { createTallCellLayout, tallCellSettings } from "../lib/tall-cell-grid";
import type { GPUEulerianInfo, GPUQuality } from "../lib/webgpu-eulerian";
import {
  compareScalarFields,
  createSmokeScenario,
  isSmokeScenarioId,
  smokeScenarioIds,
  summarizeScalarField,
  summarizeTallCellActivity,
  type ScalarFieldSummary,
  type TallCellActivitySummary,
  type SmokeScenarioId
} from "./webgpu-smoke-scenarios";

const modulePath = process.env.WEBGPU_NODE_MODULE;
if (!modulePath) throw new Error("Set WEBGPU_NODE_MODULE to the installed webgpu package index.js");
const { create, globals } = await import(pathToFileURL(modulePath).href) as {
  create(options: string[]): GPU;
  globals: Record<string, unknown>;
};
Object.assign(globalThis, globals);
const gpu = create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });

const availableMethods = [tallCellMethod, uniformMethod];
const methodFilter = process.env.FLUID_METHOD;
const methods = availableMethods.filter((method) => !methodFilter || method.id === methodFilter);
if (methods.length === 0) throw new Error(`Unknown FLUID_METHOD=${methodFilter}; expected tall-cell or uniform`);

const qualityValue = process.env.FLUID_QUALITY ?? "balanced";
if (!["balanced", "high", "ultra"].includes(qualityValue)) throw new Error(`Unknown FLUID_QUALITY=${qualityValue}`);
const quality = qualityValue as GPUQuality;
const targetOverride = process.env.FLUID_TARGET_S === undefined ? undefined : Number(process.env.FLUID_TARGET_S);
const reportEvery = Number(process.env.FLUID_REPORT_EVERY ?? 0);
const includeFinalFieldStats = process.env.FLUID_FIELD_STATS !== "0";
const runCPUOracle = process.env.FLUID_CPU_ORACLE !== "0";
const cpuMaximumCells = Number(process.env.FLUID_CPU_MAX_CELLS ?? 250_000);
const cpuMarkerSamplesPerAxis = Number(process.env.FLUID_CPU_MARKERS_PER_AXIS ?? 1);
const oracleStepsOverride = process.env.FLUID_ORACLE_STEPS === undefined ? undefined : Number(process.env.FLUID_ORACLE_STEPS);

function selectedScenarios(): SmokeScenarioId[] {
  const selection = process.env.FLUID_SCENE ?? "hose-tank";
  if (selection === "all") return [...smokeScenarioIds];
  const ids = selection.split(",").map((value) => value.trim()).filter(Boolean);
  for (const id of ids) if (!isSmokeScenarioId(id)) throw new Error(`Unknown FLUID_SCENE=${id}; expected all or ${smokeScenarioIds.join(", ")}`);
  return ids as SmokeScenarioId[];
}

async function readFloatTexture3D(device: GPUDevice, texture: GPUTexture, width: number, height: number, depth: number) {
  const bytesPerRow = Math.ceil(width * 4 / 256) * 256;
  const buffer = device.createBuffer({ size: bytesPerRow * height * depth, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow, rowsPerImage: height }, { width, height, depthOrArrayLayers: depth });
  device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const mapped = new Uint8Array(buffer.getMappedRange());
  const output = new Float32Array(width * height * depth);
  for (let z = 0; z < depth; z += 1) for (let y = 0; y < height; y += 1) {
    const row = new Float32Array(mapped.buffer, mapped.byteOffset + bytesPerRow * (y + height * z), width);
    output.set(row, width * (y + height * z));
  }
  buffer.unmap(); buffer.destroy();
  return output;
}

async function readFloatTexture2D(device: GPUDevice, texture: GPUTexture, width: number, height: number) {
  const bytesPerRow = Math.ceil(width * 4 / 256) * 256;
  const buffer = device.createBuffer({ size: bytesPerRow * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow, rowsPerImage: height }, { width, height });
  device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const mapped = new Uint8Array(buffer.getMappedRange());
  const output = new Float32Array(width * height);
  for (let y = 0; y < height; y += 1) output.set(new Float32Array(mapped.buffer, mapped.byteOffset + bytesPerRow * y, width), width * y);
  buffer.unmap(); buffer.destroy();
  return output;
}

async function readCubicVolumeField(device: GPUDevice, solver: GPUSolverInstance) {
  const { nx, ny, nz, storedNy, gridKind } = solver.info;
  const packed = await readFloatTexture3D(device, solver.volumeTexture, nx, storedNy, nz);
  let bases = new Float32Array(nx * nz);
  if (gridKind !== "uniform") bases = await readFloatTexture2D(device, solver.columnBaseTexture, nx, nz);
  const field = new Float32Array(nx * ny * nz);
  const index = (x: number, y: number, z: number) => x + nx * (y + ny * z);
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
    if (gridKind === "uniform") field[index(x, y, z)] = packed[index(x, y, z)];
    else {
      const base = Math.round(bases[x + nx * z]);
      const packedY = y < base && base > 0 ? 0 : 2 + y - base;
      field[index(x, y, z)] = packedY >= 0 && packedY < storedNy ? packed[x + nx * (packedY + storedNy * z)] : 0;
    }
  }
  return {
    field,
    summary: summarizeScalarField(field, nx, ny, nz),
    tallCellActivity: gridKind === "uniform" ? undefined : summarizeTallCellActivity(bases, ny, solver.info.regularLayers)
  };
}

interface GPUSmokeResult {
  method: string;
  info: GPUEulerianInfo;
  grid: [number, number, number];
  matchedField: Float32Array;
  matchedSummary: ScalarFieldSummary;
  matchedTallCellActivity?: TallCellActivitySummary;
  finalSummary?: ScalarFieldSummary;
  finalTallCellActivity?: TallCellActivitySummary;
  validationErrors: string[];
  runtime_ms: number;
  steps: number;
}

function reportResult(scenario: SmokeScenarioId, result: GPUSmokeResult) {
  const info = result.info;
  console.log(JSON.stringify({
    scenario, method: result.method, phase: "result", runtime_ms: Math.round(result.runtime_ms), steps: result.steps,
    simulatedTime_s: info.simulatedTime_s, grid: [info.nx, info.storedNy, info.nz], cubicGrid: result.grid,
    encodedSteps: info.encodedSteps, compressionRatio: info.compressionRatio,
    initialVolumeCellSum: info.initialVolumeCellSum, volumeCellSum: info.volumeCellSum,
    representedVolumeCellSum: info.representedVolumeCellSum, volumeDrift: info.volumeDrift,
    representedVolumeDrift: info.representedVolumeDrift, front_m: info.front_m,
    maxSpeed_m_s: info.maxSpeed_m_s, maxDivergenceBefore_s: info.maxDivergenceBefore_s,
    maxDivergenceAfter_s: info.maxDivergenceAfter_s, pressureRelativeResidual: info.pressureRelativeResidual,
    nonFiniteCount: info.nonFiniteCount, stabilityFlags: info.stabilityFlags,
    matchedFieldStats: result.matchedSummary, volumeFieldStats: result.finalSummary,
    matchedTallCellActivity: result.matchedTallCellActivity, finalTallCellActivity: result.finalTallCellActivity,
    validationErrors: result.validationErrors
  }));
}

async function runGPU(
  scenarioId: SmokeScenarioId,
  method: SimulationMethod,
  target_s: number,
  oracleSteps: number
): Promise<GPUSmokeResult> {
  const scenario = createSmokeScenario(scenarioId), scene = scenario.scene;
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("Dawn did not expose a WebGPU adapter");
  const device = await adapter.requestDevice();
  let lost: GPUDeviceLostInfo | undefined;
  void device.lost.then((info) => { lost = info; });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => validationErrors.push(event.error.message));
  const instrumentedDevice = new Proxy(device, {
    get(target, property) {
      if (property === "createComputePipeline") return (descriptor: GPUComputePipelineDescriptor) => {
        const started = performance.now(), result = target.createComputePipeline(descriptor);
        console.log(JSON.stringify({ scenario: scenarioId, method: method.id, phase: "pipeline", entryPoint: descriptor.compute.entryPoint, elapsed_ms: Math.round(performance.now() - started) }));
        return result;
      };
      const value = Reflect.get(target, property, target);
      return typeof value === "function" ? value.bind(target) : value;
    }
  }) as GPUDevice;
  const bodies = initializeRigidBodies(scene.rigidBodies);
  const constructionStarted = performance.now();
  const solver = method.createSolver!(instrumentedDevice, scene, quality, method.presetFor(quality));
  const construction_ms = performance.now() - constructionStarted;
  console.log(JSON.stringify({ scenario: scenarioId, method: method.id, phase: "constructed", construction_ms: Math.round(construction_ms), grid: [solver.info.nx, solver.info.storedNy, solver.info.nz], cubicGrid: [solver.info.nx, solver.info.ny, solver.info.nz] }));
  const runStarted = performance.now();
  let steps = 0, matched: Awaited<ReturnType<typeof readCubicVolumeField>> | undefined;
  while ((solver.info.submittedTime_s ?? 0) + 1e-9 < target_s) {
    const requestedTime = Math.min(target_s, (steps + 1) * scene.numerics.maxDt_s);
    solver.advanceTo(requestedTime, bodies);
    steps += 1;
    if (steps === oracleSteps) {
      await device.queue.onSubmittedWorkDone();
      matched = await readCubicVolumeField(device, solver);
    }
    if (steps % 30 === 0) await device.queue.onSubmittedWorkDone();
    if (reportEvery > 0 && steps % reportEvery === 0) {
      await device.queue.onSubmittedWorkDone();
      solver.info.simulatedTime_s = solver.info.submittedTime_s;
      const sample = await solver.readStats();
      console.log(JSON.stringify({ scenario: scenarioId, method: method.id, phase: "running", steps, simulatedTime_s: sample.simulatedTime_s, maxSpeed_m_s: sample.maxSpeed_m_s, maxAirSpeed_m_s: sample.maxAirSpeed_m_s, maxDivergenceBefore_s: sample.maxDivergenceBefore_s, maxDivergenceAfter_s: sample.maxDivergenceAfter_s, pressureRelativeResidual: sample.pressureRelativeResidual, maxComponentCfl: sample.maxComponentCfl, representedVolumeDrift: sample.representedVolumeDrift, stabilityFlags: sample.stabilityFlags }));
    }
    if (lost) throw new Error(`${method.id} device lost: ${lost.message || lost.reason}`);
  }
  await device.queue.onSubmittedWorkDone();
  solver.info.simulatedTime_s = solver.info.submittedTime_s;
  const info = { ...await solver.readStats() };
  matched ??= await readCubicVolumeField(device, solver);
  const final = includeFinalFieldStats && steps !== oracleSteps ? await readCubicVolumeField(device, solver) : matched;
  await device.queue.onSubmittedWorkDone();
  const result: GPUSmokeResult = {
    method: method.id, info, grid: [info.nx, info.ny, info.nz], matchedField: matched.field,
    matchedSummary: matched.summary, matchedTallCellActivity: matched.tallCellActivity,
    finalSummary: final?.summary, finalTallCellActivity: final?.tallCellActivity, validationErrors,
    runtime_ms: performance.now() - runStarted, steps
  };
  reportResult(scenarioId, result);
  solver.destroy(); device.destroy();
  return result;
}

function cpuField(solver: EulerianFluidSolver) {
  const field = new Float32Array(solver.fluid.length);
  for (let index = 0; index < field.length; index += 1) field[index] = solver.fluid[index] ? 1 : 0;
  return field;
}

function runMatchedCPUOracle(scenarioId: SmokeScenarioId, grid: [number, number, number], oracleSteps: number) {
  const cellCount = grid[0] * grid[1] * grid[2];
  if (cpuMaximumCells > 0 && cellCount > cpuMaximumCells) {
    console.log(JSON.stringify({ scenario: scenarioId, method: "cpu-reference", phase: "oracle-skipped", cubicGrid: grid, cellCount, reason: `exact grid exceeds FLUID_CPU_MAX_CELLS=${cpuMaximumCells}; set 0 for unlimited` }));
    return undefined;
  }
  const scene = createSmokeScenario(scenarioId).scene, started = performance.now();
  const solver = new EulerianFluidSolver(scene, { dimensions: { nx: grid[0], ny: grid[1], nz: grid[2] }, markerSamplesPerAxis: cpuMarkerSamplesPerAxis });
  for (let step = 0; step < oracleSteps; step += 1) solver.step(scene.numerics.maxDt_s);
  const field = cpuField(solver), summary = summarizeScalarField(field, ...grid);
  console.log(JSON.stringify({
    scenario: scenarioId, method: "cpu-reference", phase: "oracle", precision: "binary64", cubicGrid: grid,
    markerSamplesPerAxis: cpuMarkerSamplesPerAxis, oracleSteps, runtime_ms: Math.round(performance.now() - started),
    diagnostics: solver.diagnostics, fieldStats: summary
  }));
  return { field, summary, diagnostics: solver.diagnostics };
}

function invariantFailures(scenarioId: SmokeScenarioId, results: GPUSmokeResult[]) {
  const failures: string[] = [];
  const fail = (condition: boolean, message: string) => { if (!condition) failures.push(`${scenarioId}: ${message}`); };
  for (const result of results) {
    fail(result.validationErrors.length === 0, `${result.method} WebGPU validation errors: ${result.validationErrors.join("; ")}`);
    fail((result.info.nonFiniteCount ?? 0) === 0, `${result.method} reported ${result.info.nonFiniteCount} non-finite values`);
    fail(Number.isFinite(result.info.maxSpeed_m_s ?? NaN), `${result.method} max speed is not finite`);
    fail(result.matchedSummary.minimum >= -0.01, `${result.method} volume minimum ${result.matchedSummary.minimum} is below -0.01`);
    fail(result.matchedSummary.maximum <= 1.01, `${result.method} volume maximum ${result.matchedSummary.maximum} exceeds 1.01`);
    if (result.finalSummary) {
      fail(result.finalSummary.minimum >= -0.01, `${result.method} final volume minimum ${result.finalSummary.minimum} is below -0.01`);
      fail(result.finalSummary.maximum <= 1.01, `${result.method} final volume maximum ${result.finalSummary.maximum} exceeds 1.01`);
    }
  }
  if (results.length > 1) {
    const [first, ...rest] = results;
    for (const result of rest) fail(result.grid.every((value, axis) => value === first.grid[axis]), `${result.method} cubic grid ${result.grid} differs from ${first.method} ${first.grid}`);
  }
  const tall = results.find((result) => result.method === "tall-cell");
  if ((scenarioId === "settled-tank" || scenarioId === "deep-water") && tall) {
    fail((tall.info.stabilityFlags?.length ?? 0) === 0, `tall-cell equilibrium flags: ${tall.info.stabilityFlags?.join(", ")}`);
    const exactVolumeDrift = tall.finalSummary
      ? (tall.finalSummary.cellSum - (tall.info.initialVolumeCellSum ?? 0)) / Math.max(1, Math.abs(tall.info.initialVolumeCellSum ?? 0))
      : tall.info.representedVolumeDrift ?? Infinity;
    fail(Math.abs(exactVolumeDrift) <= 1e-3, `tall-cell equilibrium exact volume drift ${exactVolumeDrift} exceeds 0.1%`);
    fail((tall.finalSummary?.componentCount ?? 1) === 1, `tall-cell equilibrium split into ${tall.finalSummary?.componentCount} components`);
  }
  if (scenarioId === "deep-water" && tall) fail((tall.info.compressionRatio ?? 1) < 0.5, `tall-cell compression ratio ${tall.info.compressionRatio} is not below 0.5`);
  if (scenarioId === "hose-tank" && results.length > 1) {
    const uniform = results.find((result) => result.method === "uniform");
    if (tall && uniform) {
      const tallAdded = (tall.info.volumeCellSum ?? 0) - (tall.info.initialVolumeCellSum ?? 0);
      const uniformAdded = (uniform.info.volumeCellSum ?? 0) - (uniform.info.initialVolumeCellSum ?? 0);
      fail(Math.abs(tallAdded - uniformAdded) / Math.max(1, Math.abs(uniformAdded)) <= 0.1, `tall/uniform admitted-volume difference exceeds 10% (${tallAdded} vs ${uniformAdded})`);
      fail(Math.abs((tall.info.maxSpeed_m_s ?? 0) - (uniform.info.maxSpeed_m_s ?? 0)) / Math.max(0.1, Math.abs(uniform.info.maxSpeed_m_s ?? 0)) <= 0.15, `tall/uniform jet-speed difference exceeds 15% (${tall.info.maxSpeed_m_s} vs ${uniform.info.maxSpeed_m_s})`);
    }
  }
  return failures;
}

const failures: string[] = [];
try {
  for (const scenarioId of selectedScenarios()) {
    const scenario = createSmokeScenario(scenarioId);
    const oracleSteps = Math.max(1, Math.round(oracleStepsOverride ?? scenario.oracleSteps));
    const target_s = Math.max(targetOverride ?? scenario.target_s, oracleSteps * scenario.scene.numerics.maxDt_s);
    console.log(JSON.stringify({ scenario: scenarioId, phase: "scenario", description: scenario.description, target_s, oracleSteps, quality, methods: methods.map((method) => method.id), cpuOracle: runCPUOracle }));
    if (methods.some((method) => method.id === "tall-cell")) {
      const layout = createTallCellLayout(scenario.scene, quality);
      console.log(JSON.stringify({
        scenario: scenarioId, phase: "interrogation", interrogation: "tall-cell-activity", stage: "planned",
        cubicGrid: [layout.nx, layout.fineNy, layout.nz], storedNy: layout.packedNy,
        requestedRegularLayers: tallCellSettings[quality].regularLayers,
        effectiveRegularLayers: layout.settings.regularLayers, compressionRatio: layout.compressionRatio,
        planning: layout.planning,
        activity: summarizeTallCellActivity(layout.columnBases, layout.fineNy, layout.settings.regularLayers)
      }));
    }
    const results: GPUSmokeResult[] = [];
    for (const method of methods) results.push(await runGPU(scenarioId, method, target_s, oracleSteps));
    failures.push(...invariantFailures(scenarioId, results));

    const grid = results[0].grid;
    const cpu = runCPUOracle ? runMatchedCPUOracle(scenarioId, grid, oracleSteps) : undefined;
    for (let left = 0; left < results.length; left += 1) for (let right = left + 1; right < results.length; right += 1) {
      console.log(JSON.stringify({ scenario: scenarioId, phase: "discrepancy", left: results[left].method, right: results[right].method, oracleSteps, metrics: compareScalarFields(results[left].matchedField, results[right].matchedField, ...grid) }));
    }
    if (cpu) for (const result of results) {
      console.log(JSON.stringify({ scenario: scenarioId, phase: "discrepancy", left: result.method, right: "cpu-reference", oracleSteps, metrics: compareScalarFields(result.matchedField, cpu.field, ...grid) }));
    }
    console.log(JSON.stringify({ scenario: scenarioId, phase: "scenario-complete", passedInvariants: invariantFailures(scenarioId, results).length === 0 }));
  }
} finally {
  Reflect.deleteProperty(globalThis, "navigator");
}

if (failures.length > 0) throw new Error(`WebGPU smoke invariant failures:\n${failures.map((failure) => `- ${failure}`).join("\n")}`);
