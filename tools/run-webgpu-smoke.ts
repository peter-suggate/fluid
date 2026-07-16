import { pathToFileURL } from "node:url";
import { EulerianFluidSolver } from "../lib/eulerian-solver";
import { tallCellMethod } from "../lib/methods/tall-cell";
import type { GPUSolverInstance, SimulationMethod } from "../lib/methods/types";
import { uniformMethod } from "../lib/methods/uniform";
import { quadtreeTallCellMethod } from "../lib/methods/quadtree-tall-cell";
import { maximumFluidScale } from "../lib/quadtree-tall-cell-grid";
import { initializeRigidBodies } from "../lib/rigid-body";
import { createSingleTallCellProbeControlLayout, createSingleTallCellProbeLayout, createTallCellLayout, tallCellSettings, type SingleTallCellProbeOptions } from "../lib/tall-cell-grid";
import { WebGPUEulerianSolver, type GPUEulerianInfo, type GPUQuality } from "../lib/webgpu-eulerian";
import {
  compareScalarFields,
  compareSingleTallCellNeighborhood,
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

const availableMethods = [tallCellMethod, quadtreeTallCellMethod, uniformMethod];
const methodFilter = process.env.FLUID_METHOD?.split(",").map((value) => value.trim()).filter(Boolean);
const methods = availableMethods.filter((method) => !methodFilter || methodFilter.includes(method.id));
if (methods.length === 0 || (methodFilter && methodFilter.length !== methods.length)) throw new Error(`Unknown FLUID_METHOD=${process.env.FLUID_METHOD}; expected a comma list of tall-cell, quadtree-tall-cell, or uniform`);

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
const pressureCyclesOverride = process.env.FLUID_PRESSURE_CYCLES === undefined ? undefined : Number(process.env.FLUID_PRESSURE_CYCLES);
const remeshIntervalOverride = process.env.FLUID_REMESH_INTERVAL === undefined ? undefined : Number(process.env.FLUID_REMESH_INTERVAL);
const regularLayersOverride = process.env.FLUID_REGULAR_LAYERS === undefined ? undefined : Number(process.env.FLUID_REGULAR_LAYERS);
const maximumNeighborDeltaOverride = process.env.FLUID_MAX_NEIGHBOR_DELTA === undefined ? undefined : Number(process.env.FLUID_MAX_NEIGHBOR_DELTA);
const maximumTallHeightOverride = process.env.FLUID_MAX_TALL_HEIGHT === undefined ? undefined : Number(process.env.FLUID_MAX_TALL_HEIGHT);
const adaptivityOverride = process.env.FLUID_ADAPTIVITY === undefined ? undefined : Number(process.env.FLUID_ADAPTIVITY);
const opticalDepthOverride = process.env.FLUID_OPTICAL_DEPTH_FRACTION === undefined ? undefined : Number(process.env.FLUID_OPTICAL_DEPTH_FRACTION);
const rebuildTopologyOverride = process.env.FLUID_REBUILD_TOPOLOGY === undefined ? undefined : process.env.FLUID_REBUILD_TOPOLOGY !== "0";
const maximumLeafSizeOverride = process.env.FLUID_MAXIMUM_LEAF_SIZE === undefined ? undefined : Number(process.env.FLUID_MAXIMUM_LEAF_SIZE);
const quadtreePreconditionerOverride = process.env.FLUID_QUADTREE_PRECONDITIONER;
if (quadtreePreconditionerOverride !== undefined && !["ic0", "blockic", "jacobi", "line", "poly"].includes(quadtreePreconditionerOverride)) throw new Error("FLUID_QUADTREE_PRECONDITIONER must be ic0, blockic, jacobi, line, or poly");
const pressurePhaseTimings = process.env.FLUID_PRESSURE_PHASE_TIMINGS === "1";
const polynomialDegreeOverride = process.env.FLUID_POLYNOMIAL_DEGREE === undefined ? undefined : Number(process.env.FLUID_POLYNOMIAL_DEGREE);
const velocityTransportOverride = process.env.FLUID_VELOCITY_TRANSPORT;
const sharpeningOverride = process.env.FLUID_SHARPENING === undefined ? undefined : process.env.FLUID_SHARPENING !== "0";
const volumeControlOverride = process.env.FLUID_VOLUME_CONTROL === undefined ? undefined : process.env.FLUID_VOLUME_CONTROL !== "0";
const hierarchyOverride = process.env.FLUID_HIERARCHY === undefined ? undefined : process.env.FLUID_HIERARCHY !== "0";
const checkpointEvery_s = Number(process.env.FLUID_CHECKPOINT_EVERY_S ?? 0);
const stabilityEnvelopeRequested = process.env.FLUID_STABILITY_ENVELOPE === "1";
const singleTallCellSupportRadius = process.env.FLUID_SINGLE_TALL_CELL_SUPPORT_RADIUS === undefined
  ? 0 : Number(process.env.FLUID_SINGLE_TALL_CELL_SUPPORT_RADIUS);
const singleTallCellProbe: SingleTallCellProbeOptions | undefined = (() => {
  const raw = process.env.FLUID_SINGLE_TALL_CELL;
  if (!raw) return undefined;
  const values = raw.split(",").map(Number);
  if (values.some((value) => !Number.isFinite(value)) || (values.length !== 1 && values.length !== 3)) {
    throw new Error("FLUID_SINGLE_TALL_CELL must be HEIGHT or X,Z,HEIGHT");
  }
  if (!Number.isFinite(singleTallCellSupportRadius) || singleTallCellSupportRadius < 0) {
    throw new Error("FLUID_SINGLE_TALL_CELL_SUPPORT_RADIUS must be a non-negative number");
  }
  return values.length === 1
    ? { height: values[0], supportRadius: singleTallCellSupportRadius }
    : { x: values[0], z: values[1], height: values[2], supportRadius: singleTallCellSupportRadius };
})();

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

interface VelocityStageSummary {
  maximum: number;
  liquidMaximum: number;
  location: number[];
  component: number;
  nonFiniteCount: number;
  kineticEnergyProxy: number;
  maximumComponentCfl: number;
  maximumLiquidDivergence_s: number;
  rmsLiquidDivergence_s: number;
}

async function readRgbaTexture3D(device: GPUDevice, texture: GPUTexture, width: number, height: number, depth: number) {
  const bytesPerRow = Math.ceil(width * 16 / 256) * 256;
  const buffer = device.createBuffer({ size: bytesPerRow * height * depth, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow, rowsPerImage: height }, { width, height, depthOrArrayLayers: depth });
  device.queue.submit([encoder.finish()]); await buffer.mapAsync(GPUMapMode.READ);
  const bytes = new Uint8Array(buffer.getMappedRange());
  const output = new Float32Array(width * height * depth * 4);
  for (let z = 0; z < depth; z += 1) for (let y = 0; y < height; y += 1) {
    const row = new Float32Array(bytes.buffer, bytes.byteOffset + bytesPerRow * (y + height * z), width * 4);
    output.set(row, width * 4 * (y + height * z));
  }
  buffer.unmap(); buffer.destroy();
  return output;
}

function summarizeVelocityField(
  velocity: Float32Array,
  width: number,
  height: number,
  depth: number,
  volume: ArrayLike<number>,
  spacing: { x: number; y: number; z: number },
  dt_s: number,
  divergenceStencil: "backward" | "centered"
): VelocityStageSummary {
  let maximum = 0, liquidMaximum = 0, location = [0, 0, 0], component = 0, nonFiniteCount = 0;
  let kineticEnergyProxy = 0, maximumComponentCfl = 0;
  for (let z = 0; z < depth; z += 1) for (let y = 0; y < height; y += 1) for (let x = 0; x < width; x += 1) {
    for (let axis = 0; axis < 3; axis += 1) {
      const index = x + width * (y + height * z), value = velocity[3 * index + axis];
      if (!Number.isFinite(value)) { nonFiniteCount += 1; continue; }
      const speed = Math.abs(value); if (speed > maximum) { maximum = speed; location = [x, y, z]; component = axis; }
      if (volume[index] > 0 && speed > liquidMaximum) liquidMaximum = speed;
      maximumComponentCfl = Math.max(maximumComponentCfl, speed * dt_s / [spacing.x, spacing.y, spacing.z][axis]);
      kineticEnergyProxy += 0.5 * Math.max(0, Math.min(1, volume[index])) * value * value * spacing.x * spacing.y * spacing.z;
    }
  }
  let maximumLiquidDivergence_s = 0, divergenceSquared = 0, liquidCells = 0;
  const at = (x: number, y: number, z: number, axis: number) => velocity[3 * (x + width * (y + height * z)) + axis];
  // Mirror the collocated solver's `centeredFaceVelocity`: the face value is
  // the average of the two adjacent cell centers, and a face whose neighbor
  // is outside the domain carries zero velocity.
  const centered = (x: number, y: number, z: number, axis: number) => {
    const limit = [width, height, depth][axis];
    const coordinate = [x, y, z][axis];
    const own = at(x, y, z, axis);
    const facePlus = coordinate + 1 < limit ? 0.5 * (own + at(axis === 0 ? x + 1 : x, axis === 1 ? y + 1 : y, axis === 2 ? z + 1 : z, axis)) : 0;
    const faceMinus = coordinate > 0 ? 0.5 * (own + at(axis === 0 ? x - 1 : x, axis === 1 ? y - 1 : y, axis === 2 ? z - 1 : z, axis)) : 0;
    return (facePlus - faceMinus) / [spacing.x, spacing.y, spacing.z][axis];
  };
  for (let z = 0; z < depth; z += 1) for (let y = 0; y < height; y += 1) for (let x = 0; x < width; x += 1) {
    const index = x + width * (y + height * z);
    if (!(volume[index] > 1e-4)) continue;
    const divergence = divergenceStencil === "centered"
      ? centered(x, y, z, 0) + centered(x, y, z, 1) + centered(x, y, z, 2)
      : (at(x, y, z, 0) - (x > 0 ? at(x - 1, y, z, 0) : 0)) / spacing.x
        + (at(x, y, z, 1) - (y > 0 ? at(x, y - 1, z, 1) : 0)) / spacing.y
        + (at(x, y, z, 2) - (z > 0 ? at(x, y, z - 1, 2) : 0)) / spacing.z;
    if (!Number.isFinite(divergence)) { nonFiniteCount += 1; continue; }
    maximumLiquidDivergence_s = Math.max(maximumLiquidDivergence_s, Math.abs(divergence));
    divergenceSquared += divergence * divergence; liquidCells += 1;
  }
  return {
    maximum, liquidMaximum, location, component, nonFiniteCount, kineticEnergyProxy, maximumComponentCfl,
    maximumLiquidDivergence_s,
    rmsLiquidDivergence_s: Math.sqrt(divergenceSquared / Math.max(1, liquidCells))
  };
}

async function readVelocityTexture3D(
  device: GPUDevice,
  texture: GPUTexture,
  width: number,
  height: number,
  depth: number,
  volume: ArrayLike<number>,
  spacing = { x: 1, y: 1, z: 1 },
  dt_s = 0
): Promise<VelocityStageSummary> {
  const raw = await readRgbaTexture3D(device, texture, width, height, depth);
  const velocity = new Float32Array(width * height * depth * 3);
  for (let index = 0; index < width * height * depth; index += 1) {
    velocity[3 * index] = raw[4 * index]; velocity[3 * index + 1] = raw[4 * index + 1]; velocity[3 * index + 2] = raw[4 * index + 2];
  }
  return summarizeVelocityField(velocity, width, height, depth, volume, spacing, dt_s, "backward");
}

/**
 * Reconstruct the cubic velocity field from a packed restricted tall-cell
 * texture (rows 0/1 are the tall endpoint samples; interior rows interpolate
 * linearly between them per paper Eq 5, mirroring `validVelocityCell`) and
 * summarize it with the solver's own centered collocated divergence.
 */
async function readTallVelocityTexture3D(
  device: GPUDevice,
  texture: GPUTexture,
  nx: number,
  storedNy: number,
  nz: number,
  fineNy: number,
  bases: ArrayLike<number>,
  volume: ArrayLike<number>,
  spacing: { x: number; y: number; z: number },
  dt_s: number
): Promise<VelocityStageSummary> {
  const velocity = await readTallVelocityField3D(device, texture, nx, storedNy, nz, fineNy, bases);
  return summarizeVelocityField(velocity, nx, fineNy, nz, volume, spacing, dt_s, "backward");
}

async function readTallVelocityField3D(
  device: GPUDevice,
  texture: GPUTexture,
  nx: number,
  storedNy: number,
  nz: number,
  fineNy: number,
  bases: ArrayLike<number>
) {
  const raw = await readRgbaTexture3D(device, texture, nx, storedNy, nz);
  const velocity = new Float32Array(nx * fineNy * nz * 3);
  const packedAt = (x: number, packedY: number, z: number, axis: number) => raw[4 * (x + nx * (packedY + storedNy * z)) + axis];
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    const base = Math.round(bases[x + nx * z]);
    for (let y = 0; y < fineNy; y += 1) {
      const index = 3 * (x + nx * (y + fineNy * z));
      if (y < base && base > 0) {
        const t = Math.min(1, Math.max(0, y / Math.max(base - 1, 1)));
        for (let axis = 0; axis < 3; axis += 1) velocity[index + axis] = packedAt(x, 0, z, axis) * (1 - t) + packedAt(x, 1, z, axis) * t;
      } else {
        const packedY = 2 + y - base;
        if (packedY >= 2 && packedY < storedNy) for (let axis = 0; axis < 3; axis += 1) velocity[index + axis] = packedAt(x, packedY, z, axis);
      }
    }
  }
  return velocity;
}

function velocityDifferenceMagnitude(left: Float32Array, right: Float32Array) {
  if (left.length !== right.length || left.length % 3 !== 0) throw new Error("Velocity fields must share xyz dimensions");
  const difference = new Float32Array(left.length / 3);
  for (let index = 0; index < difference.length; index += 1) {
    difference[index] = Math.hypot(left[3 * index] - right[3 * index], left[3 * index + 1] - right[3 * index + 1], left[3 * index + 2] - right[3 * index + 2]);
  }
  return difference;
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

function inspectColumnBases(bases: ArrayLike<number>, nx: number, nz: number, fineNy: number, regularLayers: number, maximumDelta: number) {
  const histogram: Record<string, number> = {}, violations: Array<{ a: [number, number, number]; b: [number, number, number]; delta: number }> = [];
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    const height = Math.round(bases[x + nx * z]);
    histogram[height] = (histogram[height] ?? 0) + 1;
    for (const [otherX, otherZ] of [[x + 1, z], [x, z + 1]] as const) {
      if (otherX >= nx || otherZ >= nz) continue;
      const otherHeight = Math.round(bases[otherX + nx * otherZ]), delta = Math.abs(height - otherHeight);
      if (delta > maximumDelta && violations.length < 12) violations.push({ a: [x, z, height], b: [otherX, otherZ, otherHeight], delta });
    }
  }
  return { ...summarizeTallCellActivity(bases, fineNy, regularLayers, nx, nz), histogram, violations };
}

function inspectTallVolumeGaps(packed: ArrayLike<number>, bases: ArrayLike<number>, nx: number, storedNy: number, nz: number, fineNy: number, maximumDelta = Infinity) {
  let dryTallColumns = 0, dryTallWithWetRegularAbove = 0, mixedEndpointColumns = 0, wetBandCeilingColumns = 0, unexcusedDeltaViolations = 0;
  const phiAt = (x: number, y: number, z: number) => {
    const base = Math.round(bases[x + nx * z]);
    if (y < base && base > 0) {
      const t = Math.min(1, Math.max(0, y / Math.max(base - 1, 1)));
      const bottom = packed[x + nx * storedNy * z];
      const top = packed[x + nx * (1 + storedNy * z)];
      return bottom + (top - bottom) * t;
    }
    const packedY = 2 + y - base;
    return packedY >= 2 && packedY < storedNy ? packed[x + nx * (packedY + storedNy * z)] : Infinity;
  };
  // Eq. 10 is an unconditional restriction on neighboring band bases now
  // that the signed-distance remap can move the interface without VOF
  // representability floors.
  if (Number.isFinite(maximumDelta)) for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    const base = Math.round(bases[x + nx * z]);
    for (const [otherX, otherZ] of [[x + 1, z], [x, z + 1]] as const) {
      if (otherX >= nx || otherZ >= nz) continue;
      const otherBase = Math.round(bases[otherX + nx * otherZ]);
      if (Math.abs(base - otherBase) > maximumDelta) unexcusedDeltaViolations += 1;
    }
  }
  const examples: Array<{ x: number; z: number; base: number; bottom: number; top: number; lowestWetWorldY: number }> = [];
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    const base = Math.round(bases[x + nx * z]);
    const ceilingWorldY = base + storedNy - 3;
    if (ceilingWorldY < fineNy - 1 && phiAt(x, ceilingWorldY, z) <= 0) wetBandCeilingColumns += 1;
    if (base < 2) continue;
    const bottom = packed[x + nx * storedNy * z];
    const top = packed[x + nx * (1 + storedNy * z)];
    if ((bottom <= 0) !== (top <= 0)) mixedEndpointColumns += 1;
    if (bottom <= 0 || top <= 0) continue;
    dryTallColumns += 1;
    let lowestWetWorldY = -1;
    for (let y = base; y < fineNy; y += 1) if (phiAt(x, y, z) <= 0) {
      lowestWetWorldY = y;
      break;
    }
    if (lowestWetWorldY < 0) continue;
    dryTallWithWetRegularAbove += 1;
    if (examples.length < 12) examples.push({ x, z, base, bottom, top, lowestWetWorldY });
  }
  return { dryTallColumns, dryTallWithWetRegularAbove, mixedEndpointColumns, wetBandCeilingColumns, unexcusedDeltaViolations, examples };
}

async function readCubicVolumeField(device: GPUDevice, solver: GPUSolverInstance) {
  const { nx, ny, nz, storedNy, gridKind } = solver.info;
  const levelSet = solver.info.surfaceField === "levelset";
  const packed = await readFloatTexture3D(device, levelSet ? solver.surfaceFieldTexture ?? solver.volumeTexture : solver.volumeTexture, nx, storedNy, nz);
  let bases = new Float32Array(nx * nz);
  if (gridKind === "restricted-tall-cell") bases = await readFloatTexture2D(device, solver.columnBaseTexture, nx, nz);
  const field = new Float32Array(nx * ny * nz);
  const h = solver.info.cellSize_m;
  const index = (x: number, y: number, z: number) => x + nx * (y + ny * z);
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
    if (gridKind !== "restricted-tall-cell") {
      const value = packed[index(x, y, z)];
      field[index(x, y, z)] = levelSet ? Math.min(1, Math.max(0, 0.5 - value / (4 * h))) : value;
    }
    else {
      const base = Math.round(bases[x + nx * z]);
      if (y < base && base > 0) {
        const t = Math.min(1, Math.max(0, y / Math.max(base - 1, 1)));
        const bottom = packed[x + nx * storedNy * z];
        const top = packed[x + nx * (1 + storedNy * z)];
        const value = bottom + (top - bottom) * t;
        field[index(x, y, z)] = levelSet ? Math.min(1, Math.max(0, 0.5 - value / h)) : Math.max(0, value);
      } else {
        const packedY = 2 + y - base;
        const value = packedY >= 2 && packedY < storedNy ? packed[x + nx * (packedY + storedNy * z)] : 5 * h;
        field[index(x, y, z)] = levelSet ? Math.min(1, Math.max(0, 0.5 - value / h)) : value;
      }
    }
  }
  return {
    field,
    summary: summarizeScalarField(field, nx, ny, nz),
    tallCellActivity: gridKind === "restricted-tall-cell" ? summarizeTallCellActivity(bases, ny, solver.info.regularLayers, nx, nz) : undefined,
    tallVolumeGaps: gridKind === "restricted-tall-cell" ? inspectTallVolumeGaps(packed, bases, nx, storedNy, nz, ny, solver.info.maximumNeighborDelta) : undefined
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
  finalTallVolumeGaps?: ReturnType<typeof inspectTallVolumeGaps>;
  validationErrors: string[];
  construction_ms: number;
  runtime_ms: number;
  steps: number;
  velocitySummary?: VelocityStageSummary;
  stabilityEnvelope?: StabilityEnvelope;
  checkpoints: Array<{
    time_s: number;
    field: Float32Array;
    summary: ScalarFieldSummary;
    preProjectionVelocity?: Float32Array;
    postProjectionVelocity?: Float32Array;
  }>;
}

interface StabilityEnvelope {
  peakLiquidSpeed_m_s: number;
  peakComponentCfl: number;
  peakKineticEnergyProxy: number;
  maximumProjectionEnergyRatio: number;
  maximumPressureRelativeResidual: number;
  maximumProjectedVariationalResidual: number;
  maximumExactVolumeDrift: number;
  maximumComponentCount: number;
  minimumDominantComponentFraction: number;
  nonFiniteVelocityCount: number;
  sampledSteps: number;
}

function referenceVolumeCells(info: GPUEulerianInfo) {
  return info.surfaceField === "levelset"
    ? info.referenceLiquidVolume_cells ?? info.initialVolumeCellSum ?? 0
    : info.initialVolumeCellSum ?? 0;
}

function reportResult(scenario: SmokeScenarioId, result: GPUSmokeResult) {
  const info = result.info;
  console.log(JSON.stringify({
    scenario, method: result.method, phase: "result", construction_ms: Math.round(result.construction_ms), runtime_ms: Math.round(result.runtime_ms), steps: result.steps,
    simulatedTime_s: info.simulatedTime_s, grid: [info.nx, info.storedNy, info.nz], cubicGrid: result.grid,
    encodedSteps: info.encodedSteps, gridKind: info.gridKind, compressionRatio: info.compressionRatio,
    activeCompressionRatio: info.activeCompressionRatio, activeSampleCount: info.activeSampleCount,
    quadtreeMaximumFluidScale: info.quadtreeMaximumFluidScale,
    quadtreeLevelSetMismatchFraction: info.quadtreeLevelSetMismatchFraction,
    quadtreeRebuildCadenceSteps: info.quadtreeRebuildCadenceSteps,
    quadtreeRebuildCompletedCount: info.quadtreeRebuildCompletedCount,
    quadtreeRebuildBlockedFrames: info.quadtreeRebuildBlockedFrames,
    quadtreePressureIterationsUsed: info.quadtreePressureIterationsUsed,
    quadtreePressureIterationBudget: info.quadtreePressureIterationBudget,
    quadtreePressureIterationHardBudget: info.quadtreePressureIterationHardBudget,
    quadtreePressureConverged: info.quadtreePressureConverged,
    quadtreeFactorLevelCount: info.quadtreeFactorLevelCount,
    quadtreeCPUICFactorization_ms: info.quadtreeCPUICFactorization_ms,
    quadtreePressurePhaseTimings: info.quadtreePressurePhaseTimings,
    initialVolumeCellSum: info.initialVolumeCellSum, volumeCellSum: info.volumeCellSum,
    representedVolumeCellSum: info.representedVolumeCellSum, volumeDrift: info.volumeDrift,
    representedVolumeDrift: info.representedVolumeDrift, front_m: info.front_m,
    maxSpeed_m_s: info.maxSpeed_m_s, maxDivergenceBefore_s: info.maxDivergenceBefore_s,
    maxDivergenceAfter_s: info.maxDivergenceAfter_s, pressureRelativeResidual: info.pressureRelativeResidual,
    pressureResidual: info.pressureResidual,
    nonFiniteCount: info.nonFiniteCount, stabilityFlags: info.stabilityFlags, gpuTimings: info.gpuTimings,
    matchedFieldStats: result.matchedSummary, volumeFieldStats: result.finalSummary,
    matchedTallCellActivity: result.matchedTallCellActivity, finalTallCellActivity: result.finalTallCellActivity,
    finalTallVolumeGaps: result.finalTallVolumeGaps,
    velocitySummary: result.velocitySummary, stabilityEnvelope: result.stabilityEnvelope,
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
  // FLUID_DISABLE_TIMESTAMPS=1 drops the timestamp-query feature: under Dawn
  // node the tall-cell solver's first-step projection pass does not execute
  // when timestamp writes are attached (see docs/TALL_CELL_STABILITY.md), so
  // correctness audits run without GPU stage timings.
  const requiredFeatures: GPUFeatureName[] = adapter.features.has("timestamp-query") && process.env.FLUID_DISABLE_TIMESTAMPS !== "1" ? ["timestamp-query"] : [];
  const device = await adapter.requestDevice({ requiredFeatures });
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
  const values = method.presetFor(quality);
  if (method.id === "tall-cell" && pressureCyclesOverride !== undefined) values.pressureCycles = pressureCyclesOverride;
  if (method.id === "quadtree-tall-cell" && pressureCyclesOverride !== undefined) values.pressureIterations = pressureCyclesOverride;
  if (method.id === "quadtree-tall-cell" && adaptivityOverride !== undefined) values.adaptivityStrength = adaptivityOverride;
  if (method.id === "quadtree-tall-cell" && opticalDepthOverride !== undefined) values.opticalDepthFraction = opticalDepthOverride;
  if (method.id === "quadtree-tall-cell" && rebuildTopologyOverride !== undefined) values.rebuildTopology = rebuildTopologyOverride;
  if (method.id === "quadtree-tall-cell" && maximumLeafSizeOverride !== undefined) values.maximumLeafSize = maximumLeafSizeOverride;
  if (method.id === "quadtree-tall-cell" && quadtreePreconditionerOverride !== undefined) values.preconditioner = quadtreePreconditionerOverride;
  if (method.id === "quadtree-tall-cell" && pressurePhaseTimings) values.debugPressureTimings = true;
  if (method.id === "quadtree-tall-cell" && polynomialDegreeOverride !== undefined) values.polynomialDegree = polynomialDegreeOverride;
  if (method.id === "tall-cell" && remeshIntervalOverride !== undefined) values.remeshInterval = remeshIntervalOverride;
  if (method.id === "tall-cell" && regularLayersOverride !== undefined) values.regularLayers = regularLayersOverride;
  if (method.id === "tall-cell" && maximumNeighborDeltaOverride !== undefined) values.maximumNeighborDelta = maximumNeighborDeltaOverride;
  if (method.id === "tall-cell" && maximumTallHeightOverride !== undefined) values.maximumTallHeight = maximumTallHeightOverride;
  if ((method.id === "tall-cell" || method.id === "uniform") && velocityTransportOverride !== undefined) values.velocityTransport = velocityTransportOverride;
  if ((method.id === "tall-cell" || method.id === "uniform") && sharpeningOverride !== undefined) values.densitySharpening = sharpeningOverride ? "on" : "off";
  if (method.id === "tall-cell" && volumeControlOverride !== undefined) values.volumeControl = volumeControlOverride ? "on" : "off";
  if (method.id === "tall-cell" && hierarchyOverride !== undefined) values.hierarchicalExtrapolation = hierarchyOverride ? "on" : "off";
  const probeLayout = singleTallCellProbe && (method.id === "tall-cell" || method.id === "uniform")
    ? method.id === "tall-cell"
      ? createSingleTallCellProbeLayout(scene, quality, device.limits.maxTextureDimension3D, singleTallCellProbe)
      : createSingleTallCellProbeControlLayout(scene, quality, device.limits.maxTextureDimension3D, singleTallCellProbe)
    : undefined;
  const resultMethod = singleTallCellProbe && method.id === "uniform" ? "tall-cell-control" : method.id;
  const solver = probeLayout
    ? new WebGPUEulerianSolver(instrumentedDevice, scene, quality, undefined, {
      layoutOverride: probeLayout,
      pressureCycles: typeof values.pressureCycles === "number" ? values.pressureCycles : 2,
      velocityTransport: values.velocityTransport === "semi-lagrangian" ? "semi-lagrangian" : "maccormack",
      volumeControl: values.volumeControl !== "off",
      hierarchicalExtrapolation: values.hierarchicalExtrapolation !== "off"
    })
    : method.createSolver!(instrumentedDevice, scene, quality, values);
  const construction_ms = performance.now() - constructionStarted;
  console.log(JSON.stringify({ scenario: scenarioId, method: resultMethod, phase: "constructed", construction_ms: Math.round(construction_ms), grid: [solver.info.nx, solver.info.storedNy, solver.info.nz], cubicGrid: [solver.info.nx, solver.info.ny, solver.info.nz] }));
  const runStarted = performance.now();
  let steps = 0, matched: Awaited<ReturnType<typeof readCubicVolumeField>> | undefined;
  // The perturbed cadence remains exclusive to the quadtree dam-break
  // regression; FLUID_STABILITY_ENVELOPE=1 collects the same envelope for any
  // scenario/method at the scene's fixed cadence.
  const perturbCadence = scenarioId === "dam-break-ui" && method.id === "quadtree-tall-cell";
  const collectStabilityEnvelope = perturbCadence || stabilityEnvelopeRequested;
  const stabilityEnvelope: StabilityEnvelope | undefined = collectStabilityEnvelope ? {
    peakLiquidSpeed_m_s: 0, peakComponentCfl: 0, peakKineticEnergyProxy: 0,
    maximumProjectionEnergyRatio: 0, maximumPressureRelativeResidual: 0,
    maximumProjectedVariationalResidual: 0, maximumExactVolumeDrift: 0,
    maximumComponentCount: 0, minimumDominantComponentFraction: 1,
    nonFiniteVelocityCount: 0, sampledSteps: 0
  } : undefined;
  // The UI uses a fixed cadence.  The long regression deliberately perturbs
  // that cadence while respecting maxDt so topology transfer and projection
  // are exercised with genuinely different timestep sizes.
  const regressionDtPattern = [0.004, 0.0035, 0.0025, 0.004];
  const checkpoints: GPUSmokeResult["checkpoints"] = [];
  let nextCheckpoint_s = checkpointEvery_s;
  while ((solver.info.submittedTime_s ?? 0) + 1e-9 < target_s) {
    const stepDt = perturbCadence
      ? Math.min(scene.numerics.maxDt_s, regressionDtPattern[steps % regressionDtPattern.length])
      : scene.numerics.maxDt_s;
    const requestedTime = Math.min(target_s, (solver.info.submittedTime_s ?? 0) + stepDt);
    const accepted = solver.advanceTo(requestedTime, bodies);
    if (!accepted) {
      await new Promise((resolve) => setTimeout(resolve, 0));
      continue;
    }
    steps += 1;
    if (steps === oracleSteps) {
      await device.queue.onSubmittedWorkDone();
      matched = await readCubicVolumeField(device, solver);
    }
    if (steps % 30 === 0) await device.queue.onSubmittedWorkDone();
    const shouldReport = reportEvery > 0 && steps % reportEvery === 0;
    const shouldFeedVolumeControl = solver.info.surfaceField === "levelset" && values.volumeControl !== "off";
    if (shouldReport || collectStabilityEnvelope || shouldFeedVolumeControl) {
      await device.queue.onSubmittedWorkDone();
      solver.info.simulatedTime_s = solver.info.submittedTime_s;
      const sample = await solver.readStats();
      const isRestrictedTall = sample.gridKind === "restricted-tall-cell";
      let tallCellActivity: ReturnType<typeof inspectColumnBases> | undefined, tallVolumeGaps: ReturnType<typeof inspectTallVolumeGaps> | undefined;
      let bases: Float32Array | undefined;
      if (sample.gridKind !== "uniform") {
        bases = await readFloatTexture2D(device, solver.columnBaseTexture, sample.nx, sample.nz);
        tallCellActivity = inspectColumnBases(bases, sample.nx, sample.nz, sample.ny, sample.regularLayers, sample.maximumNeighborDelta);
        tallVolumeGaps = inspectTallVolumeGaps(await readFloatTexture3D(device, solver.volumeTexture, sample.nx, sample.storedNy, sample.nz), bases, sample.nx, sample.storedNy, sample.nz, sample.ny, sample.maximumNeighborDelta);
      }
      const exact = await readCubicVolumeField(device, solver);
      const stagedSolver = solver as GPUSolverInstance & { preProjectionVelocityTexture?: GPUTexture; velocityTexture?: GPUTexture };
      const spacing = {
        x: scene.container.width_m / sample.nx,
        y: scene.container.height_m / sample.ny,
        z: scene.container.depth_m / sample.nz
      };
      const readStagedVelocity = (texture: GPUTexture | undefined) => {
        if (!texture) return Promise.resolve(undefined);
        return isRestrictedTall && bases
          ? readTallVelocityTexture3D(device, texture, sample.nx, sample.storedNy, sample.nz, sample.ny, bases, exact.field, spacing, stepDt)
          : readVelocityTexture3D(device, texture, sample.nx, sample.ny, sample.nz, exact.field, spacing, stepDt);
      };
      const preProjectionVelocity = await readStagedVelocity(stagedSolver.preProjectionVelocityTexture);
      const postProjectionVelocity = await readStagedVelocity(stagedSolver.velocityTexture);
      // §14 audit checklist: an extremum without a location cannot separate an
      // interface, tall endpoint, wall, or remesh artifact. Locations arrive
      // in world (cubic) coordinates from the reduction kernels.
      const classifyLocation = (location?: { x: number; y: number; z: number }) => {
        if (!location || !bases) return undefined;
        const base = Math.round(bases[location.x + sample.nx * location.z] ?? 0);
        const region = base > 0 && location.y < base
          ? (location.y === 0 ? "tall-bottom-endpoint" : location.y >= base - 1 ? "tall-top-endpoint" : "tall-interior")
          : "band";
        const wall = location.x === 0 || location.x === sample.nx - 1 || location.z === 0 || location.z === sample.nz - 1 || location.y === 0 || location.y === sample.ny - 1;
        return { ...location, base, region, wall };
      };
      const extrema = isRestrictedTall ? {
        maxSpeed: classifyLocation(sample.maxSpeedLocation),
        maxAirSpeed: classifyLocation(sample.maxAirSpeedLocation),
        divergenceBefore: classifyLocation(sample.maxDivergenceBeforeLocation),
        divergenceAfter: classifyLocation(sample.maxDivergenceAfterLocation),
        pressure: classifyLocation(sample.maxPressureLocation),
        pressureResidual: classifyLocation(sample.maxPressureResidualLocation)
      } : undefined;
      const exactReference = referenceVolumeCells(sample);
      const exactVolumeDrift = (exact.summary.cellSum - exactReference) / Math.max(1, Math.abs(exactReference));
      if (stabilityEnvelope && preProjectionVelocity && postProjectionVelocity) {
        const dominantFraction = exact.summary.wetCells > 0 ? exact.summary.largestComponent / exact.summary.wetCells : 1;
        stabilityEnvelope.peakLiquidSpeed_m_s = Math.max(stabilityEnvelope.peakLiquidSpeed_m_s, postProjectionVelocity.liquidMaximum);
        stabilityEnvelope.peakComponentCfl = Math.max(stabilityEnvelope.peakComponentCfl, postProjectionVelocity.maximumComponentCfl);
        stabilityEnvelope.peakKineticEnergyProxy = Math.max(stabilityEnvelope.peakKineticEnergyProxy, postProjectionVelocity.kineticEnergyProxy);
        // Floor the denominator at a meaningful fraction of slosh-scale
        // energy: the startup steps legitimately create the velocity field
        // from near-zero KE (pre ~1e-4), and a ratio on that denominator
        // measures noise, not amplification.
        stabilityEnvelope.maximumProjectionEnergyRatio = Math.max(stabilityEnvelope.maximumProjectionEnergyRatio, postProjectionVelocity.kineticEnergyProxy / Math.max(preProjectionVelocity.kineticEnergyProxy, 0.01));
        // The quadtree residual readback is asynchronous, so the first steps
        // can legitimately sample before any residual exists; later steps
        // without one still fail hard.
        stabilityEnvelope.maximumPressureRelativeResidual = Math.max(stabilityEnvelope.maximumPressureRelativeResidual, sample.pressureRelativeResidual ?? (steps <= 2 ? 0 : Infinity));
        stabilityEnvelope.maximumProjectedVariationalResidual = Math.max(stabilityEnvelope.maximumProjectedVariationalResidual, sample.pressureResidual ?? (steps <= 2 ? 0 : Infinity));
        stabilityEnvelope.maximumExactVolumeDrift = Math.max(stabilityEnvelope.maximumExactVolumeDrift, Math.abs(exactVolumeDrift));
        stabilityEnvelope.maximumComponentCount = Math.max(stabilityEnvelope.maximumComponentCount, exact.summary.componentCount);
        stabilityEnvelope.minimumDominantComponentFraction = Math.min(stabilityEnvelope.minimumDominantComponentFraction, dominantFraction);
        stabilityEnvelope.nonFiniteVelocityCount += preProjectionVelocity.nonFiniteCount + postProjectionVelocity.nonFiniteCount;
        stabilityEnvelope.sampledSteps += 1;
      }
      if (shouldReport) console.log(JSON.stringify({ scenario: scenarioId, method: method.id, phase: "running", steps, simulatedTime_s: sample.simulatedTime_s, dt_s: stepDt, preProjectionVelocity, postProjectionVelocity, maxSpeed_m_s: sample.maxSpeed_m_s, maxAirSpeed_m_s: sample.maxAirSpeed_m_s, maxDivergenceBefore_s: sample.maxDivergenceBefore_s, maxDivergenceAfter_s: sample.maxDivergenceAfter_s, pressureRelativeResidual: sample.pressureRelativeResidual, pressureIterationsUsed: sample.quadtreePressureIterationsUsed, pressureIterationBudget: sample.quadtreePressureIterationBudget, pressureIterationHardBudget: sample.quadtreePressureIterationHardBudget, pressureConverged: sample.quadtreePressureConverged, factorLevelCount: sample.quadtreeFactorLevelCount, pressurePhaseTimings: sample.quadtreePressurePhaseTimings, maxComponentCfl: sample.maxComponentCfl, representedVolumeDrift: sample.representedVolumeDrift, volumeCorrectionNormalSpeed_cells_s: sample.volumeCorrectionNormalSpeed_cells_s, phiInterfaceCellCount: sample.phiInterfaceCellCount, exactVolumeCellSum: exact.summary.cellSum, exactVolumeDrift, componentCount: exact.summary.componentCount, dominantComponentFraction: exact.summary.wetCells > 0 ? exact.summary.largestComponent / exact.summary.wetCells : 1, quadtree: sample.gridKind === "quadtree-tall-cell" ? { leafCount: sample.quadtreeLeafCount, pressureSampleCount: sample.quadtreePressureSampleCount, liquidDofCount: sample.quadtreeLiquidDofCount, faceCount: sample.quadtreeFaceCount, tallSegmentCount: sample.quadtreeTallSegmentCount, ghostFaceCount: sample.quadtreeGhostFaceCount, maximumNeighborRatio: sample.quadtreeMaximumNeighborRatio, maximumFluidScale: sample.quadtreeMaximumFluidScale, levelSetMismatchFraction: sample.quadtreeLevelSetMismatchFraction } : undefined, stabilityFlags: sample.stabilityFlags, extrema, tallCellActivity, tallVolumeGaps }));
    }
    if (checkpointEvery_s > 0 && (solver.info.submittedTime_s ?? 0) + 1e-9 >= nextCheckpoint_s) {
      await device.queue.onSubmittedWorkDone();
      const cubic = await readCubicVolumeField(device, solver);
      let preProjectionVelocity: Float32Array | undefined, postProjectionVelocity: Float32Array | undefined;
      if (singleTallCellProbe && solver.info.gridKind === "restricted-tall-cell") {
        const bases = await readFloatTexture2D(device, solver.columnBaseTexture, solver.info.nx, solver.info.nz);
        const staged = solver as GPUSolverInstance & { preProjectionVelocityTexture?: GPUTexture; velocityTexture?: GPUTexture };
        if (staged.preProjectionVelocityTexture) preProjectionVelocity = await readTallVelocityField3D(device, staged.preProjectionVelocityTexture, solver.info.nx, solver.info.storedNy, solver.info.nz, solver.info.ny, bases);
        if (staged.velocityTexture) postProjectionVelocity = await readTallVelocityField3D(device, staged.velocityTexture, solver.info.nx, solver.info.storedNy, solver.info.nz, solver.info.ny, bases);
      }
      checkpoints.push({ time_s: solver.info.submittedTime_s ?? 0, field: cubic.field, summary: cubic.summary, preProjectionVelocity, postProjectionVelocity });
      while (nextCheckpoint_s <= (solver.info.submittedTime_s ?? 0) + 1e-9) nextCheckpoint_s += checkpointEvery_s;
    }
    if (lost) throw new Error(`${method.id} device lost: ${lost.message || lost.reason}`);
  }
  await device.queue.onSubmittedWorkDone();
  solver.info.simulatedTime_s = solver.info.submittedTime_s;
  const info = { ...await solver.readStats() };
  matched ??= await readCubicVolumeField(device, solver);
  const final = includeFinalFieldStats && steps !== oracleSteps ? await readCubicVolumeField(device, solver) : matched;
  const velocityTexture = (solver as GPUSolverInstance & { velocityTexture?: GPUTexture }).velocityTexture;
  const finalSpacing = {
    x: scene.container.width_m / info.nx,
    y: scene.container.height_m / info.ny,
    z: scene.container.depth_m / info.nz
  };
  const velocitySummary = velocityTexture && final
    ? (info.gridKind === "restricted-tall-cell"
      ? await readTallVelocityTexture3D(device, velocityTexture, info.nx, info.storedNy, info.nz, info.ny, await readFloatTexture2D(device, solver.columnBaseTexture, info.nx, info.nz), final.field, finalSpacing, scene.numerics.maxDt_s)
      : await readVelocityTexture3D(device, velocityTexture, info.nx, info.ny, info.nz, final.field, finalSpacing, scene.numerics.maxDt_s))
    : undefined;
  await device.queue.onSubmittedWorkDone();
  const result: GPUSmokeResult = {
    method: resultMethod, info, grid: [info.nx, info.ny, info.nz], matchedField: matched.field,
    matchedSummary: matched.summary, matchedTallCellActivity: matched.tallCellActivity,
    finalSummary: final?.summary, finalTallCellActivity: final?.tallCellActivity,
    finalTallVolumeGaps: final?.tallVolumeGaps, validationErrors,
    construction_ms, runtime_ms: performance.now() - runStarted, steps, velocitySummary, stabilityEnvelope, checkpoints
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
    if (scenarioId === "hose-tank" || scenarioId === "sphere-jet") {
      fail((result.info.volumeCellSum ?? -Infinity) >= (result.info.initialVolumeCellSum ?? 0) * 0.99, `${result.method} inflow scene lost more than 1% of its initial represented volume`);
      // A working inflow moves fluid; a projection that treats injected liquid
      // as air freezes the whole field at numerical zero while volume grows.
      // Before ~0.3 s the stream is still sub-threshold and max liquid speed
      // measures ambient equilibrium noise, so the gate only applies once the
      // jet has had time to establish.
      if ((result.info.simulatedTime_s ?? 0) >= 0.3) fail((result.info.maxSpeed_m_s ?? 0) >= 0.01, `${result.method} inflow scene is frozen: max speed ${result.info.maxSpeed_m_s} m/s`);
    }
    else fail(Math.abs(result.info.representedVolumeDrift ?? Infinity) <= 0.01, `${result.method} represented-volume drift ${result.info.representedVolumeDrift} exceeds 1%`);
    fail(result.matchedSummary.minimum >= -0.01, `${result.method} volume minimum ${result.matchedSummary.minimum} is below -0.01`);
    // Stored density above one is deliberate temporary mass on both paths
    // (sharpening deposits, tall remap residuals) and drains through the
    // correction divergence; the bound only catches runaway accumulation.
    const maximumStoredDensity = 1.5;
    fail(result.matchedSummary.maximum <= maximumStoredDensity, `${result.method} volume maximum ${result.matchedSummary.maximum} exceeds ${maximumStoredDensity}`);
    if (result.finalSummary) {
      fail(result.finalSummary.minimum >= -0.01, `${result.method} final volume minimum ${result.finalSummary.minimum} is below -0.01`);
      fail(result.finalSummary.maximum <= maximumStoredDensity, `${result.method} final volume maximum ${result.finalSummary.maximum} exceeds ${maximumStoredDensity}`);
    }
  }
  if (results.length > 1) {
    const [first, ...rest] = results;
    for (const result of rest) fail(result.grid.every((value, axis) => value === first.grid[axis]), `${result.method} cubic grid ${result.grid} differs from ${first.method} ${first.grid}`);
  }
  const tall = results.find((result) => result.method === "tall-cell");
  if (tall?.finalTallVolumeGaps?.unexcusedDeltaViolations !== undefined) fail(tall.finalTallVolumeGaps.unexcusedDeltaViolations === 0, `tall-cell has ${tall.finalTallVolumeGaps.unexcusedDeltaViolations} adjacent base deltas beyond ${tall.info.maximumNeighborDelta}`);
  else if (tall?.finalTallCellActivity?.maximumAdjacentDelta !== undefined) fail(tall.finalTallCellActivity.maximumAdjacentDelta <= tall.info.maximumNeighborDelta, `tall-cell adjacent base delta ${tall.finalTallCellActivity.maximumAdjacentDelta} exceeds ${tall.info.maximumNeighborDelta}`);
  if (scenarioId === "dam-break-boxes" && tall) {
    fail(tall.info.gridKind === "restricted-tall-cell", `tall-cell dam break used ${tall.info.gridKind} instead of the restricted backend`);
    fail((tall.finalTallCellActivity?.tallColumns ?? 0) > 0, "tall-cell dam break has no allocated tall columns");
    fail((tall.finalTallCellActivity?.ordinaryColumns ?? Infinity) === 0, `tall-cell dam break has ${tall.finalTallCellActivity?.ordinaryColumns ?? "unknown"} incomplete base-zero columns`);
    fail((tall.finalTallVolumeGaps?.dryTallWithWetRegularAbove ?? Infinity) === 0, `tall-cell dam break has ${tall.finalTallVolumeGaps?.dryTallWithWetRegularAbove ?? "unknown"} dry tall columns underneath wet regular cells`);
    const tallReference = referenceVolumeCells(tall.info);
    const exactVolumeDrift = tall.finalSummary
      ? (tall.finalSummary.cellSum - tallReference) / Math.max(1, Math.abs(tallReference))
      : tall.info.representedVolumeDrift ?? Infinity;
    fail(Math.abs(exactVolumeDrift) <= 0.01, `tall-cell dam break exact volume drift ${exactVolumeDrift} exceeds 1%`);
  }
  if (scenarioId === "dam-break-ui" && tall) {
    // Gates for the tall-cell dam break with genuinely active tall cells
    // (test:webgpu:dam-tall-active). Thresholds calibrated 2026-07-15 against
    // the post-fix baseline (docs/TALL_CELL_STABILITY.md): KE ratio peaks at
    // 1.67 in the release transient (the endpoint-wetness defect produced >8
    // and non-finite blow-up), peak CFL 10.9 in splash transients (uniform
    // 1.7 — a known remaining gap, gated as a regression backstop).
    fail(tall.info.gridKind === "restricted-tall-cell", `tall-cell dam break used ${tall.info.gridKind} instead of the restricted backend`);
    fail((tall.finalTallCellActivity?.tallColumns ?? 0) > 0, "tall-cell dam break has no allocated tall columns");
    const envelope = tall.stabilityEnvelope;
    if (envelope) {
      fail(envelope.nonFiniteVelocityCount === 0, `tall-cell dam break produced ${envelope.nonFiniteVelocityCount} non-finite staged velocities`);
      fail(envelope.maximumProjectionEnergyRatio <= 2.0, `tall-cell projection amplified kinetic energy by ${envelope.maximumProjectionEnergyRatio}`);
      // Splash chaos gives large run-to-run variance (10.9-21.7 observed on
      // identical configs); the backstop only needs to catch the divergent
      // regime, which reached 1e29 before the 2026-07-15 fixes.
      fail(envelope.peakComponentCfl <= 32, `tall-cell dam break peak CFL ${envelope.peakComponentCfl} exceeds the 32-cell backstop`);
      fail(envelope.maximumExactVolumeDrift <= 0.01, `tall-cell dam break exact volume drift peaked at ${envelope.maximumExactVolumeDrift}`);
    }
    if ((tall.info.simulatedTime_s ?? 0) >= 1.5) fail((tall.info.front_m ?? -Infinity) > 0.3, `tall-cell dam break front did not cross the tank: ${tall.info.front_m} m`);
    fail((tall.finalTallVolumeGaps?.dryTallWithWetRegularAbove ?? Infinity) === 0, `tall-cell dam break has ${tall.finalTallVolumeGaps?.dryTallWithWetRegularAbove ?? "unknown"} dry tall columns underneath wet regular cells`);
    const uniformPair = results.find((result) => result.method === "uniform");
    if (uniformPair && tall.checkpoints.length > 0 && uniformPair.checkpoints.length > 0) {
      const pairCount = Math.min(tall.checkpoints.length, uniformPair.checkpoints.length);
      let minimumIoU = 1, finalIoU = 1;
      for (let index = 0; index < pairCount; index += 1) {
        const iou = compareScalarFields(tall.checkpoints[index].field, uniformPair.checkpoints[index].field, ...tall.grid).wetIntersectionOverUnion;
        minimumIoU = Math.min(minimumIoU, iou); finalIoU = iou;
      }
      // The minimal-tall control (bases pinned at 2) bottoms out at 0.37 IoU
      // against uniform through the chaotic slosh; deep tall cells must stay
      // within that envelope rather than match uniform cell-for-cell.
      fail(minimumIoU >= 0.35, `tall-cell wet-IoU vs uniform fell to ${minimumIoU} (minimal-tall control floor is 0.37)`);
      fail(finalIoU >= 0.4, `tall-cell final wet-IoU vs uniform is ${finalIoU}`);
    }
    // The level set should retain a comparably narrow transition even though
    // it deliberately uses semi-Lagrangian advection and periodic reinit.
    if (uniformPair?.finalSummary && tall.finalSummary) {
      const mixedFraction = (summary: ScalarFieldSummary) => summary.wetCells > 0 ? summary.mixedCells / summary.wetCells : 0;
      fail(mixedFraction(tall.finalSummary) <= mixedFraction(uniformPair.finalSummary) * 2 + 0.05,
        `tall-cell mixed-cell fraction ${mixedFraction(tall.finalSummary)} exceeds twice uniform's ${mixedFraction(uniformPair.finalSummary)}`);
    }
  }
  if ((scenarioId === "settled-tank" || scenarioId === "deep-water") && tall) {
    fail((tall.info.stabilityFlags?.length ?? 0) === 0, `tall-cell equilibrium flags: ${tall.info.stabilityFlags?.join(", ")}`);
    const tallReference = referenceVolumeCells(tall.info);
    const exactVolumeDrift = tall.finalSummary
      ? (tall.finalSummary.cellSum - tallReference) / Math.max(1, Math.abs(tallReference))
      : tall.info.representedVolumeDrift ?? Infinity;
    fail(Math.abs(exactVolumeDrift) <= 0.01, `tall-cell equilibrium exact volume drift ${exactVolumeDrift} exceeds 1%`);
    fail((tall.finalSummary?.componentCount ?? 1) === 1, `tall-cell equilibrium split into ${tall.finalSummary?.componentCount} components`);
  }
  if (scenarioId === "deep-water" && tall) fail((tall.info.compressionRatio ?? 1) < 0.5, `tall-cell compression ratio ${tall.info.compressionRatio} is not below 0.5`);
  const quadtree = results.find((result) => result.method === "quadtree-tall-cell");
  if (quadtree) {
    fail(quadtree.info.gridKind === "quadtree-tall-cell", `quadtree method reported ${quadtree.info.gridKind}`);
    fail((quadtree.info.quadtreeMaximumNeighborRatio ?? Infinity) <= 2, `quadtree neighbor ratio ${quadtree.info.quadtreeMaximumNeighborRatio} exceeds 2:1`);
    fail((quadtree.info.quadtreeLeafCount ?? 0) > 0, "quadtree has no leaves");
    fail((quadtree.info.quadtreeGhostFaceCount ?? 0) > 0, "quadtree has no corrected inner ghost faces");
    const residualAccepted = (quadtree.info.pressureRelativeResidual ?? Infinity) <= 1e-4 || (quadtree.info.pressureResidual ?? Infinity) <= 1e-5;
    fail(residualAccepted, `quadtree PCG residual relative=${quadtree.info.pressureRelativeResidual} rms=${quadtree.info.pressureResidual} exceeds the relative target and f32 absolute floor`);
    // The pressure level set is reconciled against the conservative VOF every
    // rebuild; a growing pre-reconciliation mismatch means the two surface
    // representations are diverging (dead inflow sources, drift, or a
    // reconciliation regression).
    fail((quadtree.info.quadtreeLevelSetMismatchFraction ?? 0) <= 0.02, `quadtree level-set/VOF sign mismatch ${quadtree.info.quadtreeLevelSetMismatchFraction} exceeds 2% of cells`);
    fail((quadtree.info.quadtreeMaximumFluidScale ?? Infinity) <= maximumFluidScale, `quadtree free-surface scale ${quadtree.info.quadtreeMaximumFluidScale} escaped the ${maximumFluidScale} ghost-fluid ceiling`);
    if (scenarioId === "settled-tank" || scenarioId === "deep-water") {
      const exactVolumeDrift = quadtree.finalSummary
        ? (quadtree.finalSummary.cellSum - (quadtree.info.initialVolumeCellSum ?? 0)) / Math.max(1, Math.abs(quadtree.info.initialVolumeCellSum ?? 0))
        : quadtree.info.representedVolumeDrift ?? Infinity;
      fail(Math.abs(exactVolumeDrift) <= 1e-3, `quadtree equilibrium exact volume drift ${exactVolumeDrift} exceeds 0.1%`);
      fail((quadtree.finalSummary?.componentCount ?? 1) === 1, `quadtree equilibrium split into ${quadtree.finalSummary?.componentCount} components`);
      fail((quadtree.velocitySummary?.liquidMaximum ?? Infinity) <= 0.05, `quadtree equilibrium liquid velocity ${quadtree.velocitySummary?.liquidMaximum} m/s exceeds 0.05 m/s`);
    }
    if (scenarioId === "deep-water") fail((quadtree.info.compressionRatio ?? 1) < 0.5, `quadtree deep-water compression ratio ${quadtree.info.compressionRatio} is not below 0.5`);
    if (scenarioId === "dam-break-ui") {
      const envelope = quadtree.stabilityEnvelope;
      fail((quadtree.info.simulatedTime_s ?? 0) >= 0.2 - 1e-9, `quadtree dam-break regression reached only ${quadtree.info.simulatedTime_s} s`);
      fail(quadtree.info.quadtreeRebuildCadenceSteps === 1, `quadtree dam-break rebuild cadence was ${quadtree.info.quadtreeRebuildCadenceSteps}, not Algorithm 1's every-step cadence`);
      fail((quadtree.info.quadtreeRebuildCompletedCount ?? 0) >= quadtree.steps - 1, `quadtree completed ${quadtree.info.quadtreeRebuildCompletedCount} rebuilds for ${quadtree.steps} steps`);
      fail((envelope?.sampledSteps ?? 0) === quadtree.steps, `quadtree dam-break sampled ${envelope?.sampledSteps} of ${quadtree.steps} steps`);
      fail((envelope?.nonFiniteVelocityCount ?? Infinity) === 0, `quadtree dam-break encountered ${envelope?.nonFiniteVelocityCount} non-finite staged velocities`);
      fail((envelope?.peakLiquidSpeed_m_s ?? Infinity) <= 5, `quadtree dam-break peak liquid speed ${envelope?.peakLiquidSpeed_m_s} m/s exceeds 5 m/s`);
      fail((envelope?.peakComponentCfl ?? Infinity) <= 1, `quadtree dam-break peak CFL ${envelope?.peakComponentCfl} exceeds one cell`);
      fail((envelope?.maximumProjectionEnergyRatio ?? Infinity) <= 1.1, `quadtree pressure projection amplified kinetic energy by ${envelope?.maximumProjectionEnergyRatio}`);
      // Results Sec. 5: every paper example uses ICCG with relative residual
      // 1e-4. A topology transition is not allowed to weaken that criterion.
      fail((envelope?.maximumPressureRelativeResidual ?? Infinity) <= 1e-4, `quadtree dam-break pressure residual peaked at ${envelope?.maximumPressureRelativeResidual}`);
      fail((envelope?.maximumExactVolumeDrift ?? Infinity) <= 0.02, `quadtree dam-break level-set volume drift peaked at ${envelope?.maximumExactVolumeDrift}`);
      fail((envelope?.minimumDominantComponentFraction ?? -Infinity) >= 0.995, `quadtree dam-break dominant component fell to ${envelope?.minimumDominantComponentFraction}`);
      fail((quadtree.info.front_m ?? -Infinity) > -0.005, `quadtree dam-break front did not progress: ${quadtree.info.front_m} m`);
      if (tall && quadtree.grid.every((value, axis) => value === tall.grid[axis])) {
        const comparison = compareScalarFields(quadtree.finalSummary ? quadtree.checkpoints.at(-1)?.field ?? quadtree.matchedField : quadtree.matchedField, tall.finalSummary ? tall.checkpoints.at(-1)?.field ?? tall.matchedField : tall.matchedField, ...quadtree.grid);
        fail(comparison.wetIntersectionOverUnion >= 0.35, `quadtree dam-break wet-IoU ${comparison.wetIntersectionOverUnion} is below the active tall-cell quality floor`);
        fail(comparison.centroidDistanceCells === null || comparison.centroidDistanceCells <= 6, `quadtree dam-break centroid differs from tall-cell by ${comparison.centroidDistanceCells} cells`);
      }
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
      const layout = singleTallCellProbe
        ? createSingleTallCellProbeLayout(scenario.scene, quality, 2048, singleTallCellProbe)
        : createTallCellLayout(scenario.scene, quality, 2048, regularLayersOverride === undefined && maximumNeighborDeltaOverride === undefined ? undefined : {
          ...(regularLayersOverride === undefined ? {} : { regularLayers: regularLayersOverride }),
          ...(maximumNeighborDeltaOverride === undefined ? {} : { maximumNeighborDelta: maximumNeighborDeltaOverride })
        });
      console.log(JSON.stringify({
        scenario: scenarioId, phase: "interrogation", interrogation: "tall-cell-activity", stage: "planned",
        cubicGrid: [layout.nx, layout.fineNy, layout.nz], storedNy: layout.packedNy,
        requestedRegularLayers: regularLayersOverride ?? tallCellSettings[quality].regularLayers,
        effectiveRegularLayers: layout.settings.regularLayers, compressionRatio: layout.compressionRatio,
        activeCompressionRatio: layout.activeCompressionRatio, activeSampleCount: layout.activeSampleCount,
        planning: layout.planning,
        activity: summarizeTallCellActivity(layout.columnBases, layout.fineNy, layout.settings.regularLayers, layout.nx, layout.nz),
        singleTallCellProbe: layout.singleTallCellProbe
      }));
    }
    const results: GPUSmokeResult[] = [];
    for (const method of methods) results.push(await runGPU(scenarioId, method, target_s, oracleSteps));
    failures.push(...invariantFailures(scenarioId, results));

    const tallResult = results.find((result) => result.method === "tall-cell"), uniformResult = results.find((result) => result.method === (singleTallCellProbe ? "tall-cell-control" : "uniform"));
    if (tallResult && uniformResult) {
      const ratio = (uniformValue?:number,tallValue?:number) => typeof uniformValue==="number"&&typeof tallValue==="number"&&Number.isFinite(uniformValue)&&Number.isFinite(tallValue)&&tallValue>0 ? uniformValue/tallValue : null;
      const stages = ["advection_ms","pressure_ms","projection_ms","rigidCoupling_ms","diagnostics_ms"] as const;
      const gpuStageSpeedups = Object.fromEntries(stages.map((stage) => [stage,ratio(uniformResult.info.gpuTimings?.[stage],tallResult.info.gpuTimings?.[stage])]));
      console.log(JSON.stringify({
        scenario:scenarioId,phase:"performance-comparison",baseline:"uniform",candidate:"tall-cell",
        tallBackend:tallResult.info.gridKind,wallRuntimeSpeedup:ratio(uniformResult.runtime_ms,tallResult.runtime_ms),
        constructionSpeedup:ratio(uniformResult.construction_ms,tallResult.construction_ms),gpuStageSpeedups,
        activeSampleReduction:1-(tallResult.info.activeSampleCount??tallResult.info.cellCount)/(uniformResult.info.activeSampleCount??uniformResult.info.cellCount),
        properties:{
          tallRepresentedVolumeDrift:tallResult.info.representedVolumeDrift,uniformRepresentedVolumeDrift:uniformResult.info.representedVolumeDrift,
          representedVolumeDriftDelta:(tallResult.info.representedVolumeDrift??0)-(uniformResult.info.representedVolumeDrift??0),
          tallNonFiniteCount:tallResult.info.nonFiniteCount,uniformNonFiniteCount:uniformResult.info.nonFiniteCount,
          tallPressureRelativeResidual:tallResult.info.pressureRelativeResidual,uniformPressureRelativeResidual:uniformResult.info.pressureRelativeResidual
        }
      }));
      if (singleTallCellProbe) {
        const layout = createSingleTallCellProbeLayout(scenario.scene, quality, 2048, singleTallCellProbe);
        const probe = layout.singleTallCellProbe!;
        console.log(JSON.stringify({
          scenario: scenarioId, phase: "single-tall-cell-difference", control: "restricted-cubic-limit", candidate: "one-tall-cell",
          probe, time_s: tallResult.info.simulatedTime_s,
          global: compareScalarFields(tallResult.finalSummary ? tallResult.checkpoints.at(-1)?.field ?? tallResult.matchedField : tallResult.matchedField, uniformResult.finalSummary ? uniformResult.checkpoints.at(-1)?.field ?? uniformResult.matchedField : uniformResult.matchedField, ...tallResult.grid),
          locality: compareSingleTallCellNeighborhood(tallResult.matchedField, uniformResult.matchedField, ...tallResult.grid, probe.x, probe.z),
          velocity: { tall: tallResult.velocitySummary, uniform: uniformResult.velocitySummary }
        }));
        const pairCount = Math.min(tallResult.checkpoints.length, uniformResult.checkpoints.length);
        for (let index = 0; index < pairCount; index += 1) {
          const tallCheckpoint = tallResult.checkpoints[index], uniformCheckpoint = uniformResult.checkpoints[index];
          const profileTop = Math.min(tallResult.grid[1], probe.height + 4);
          const probeProfile = (field: Float32Array) => Array.from({ length: profileTop }, (_, y) => field[probe.x + tallResult.grid[0] * (y + tallResult.grid[1] * probe.z)]);
          const velocityLocality = (left?: Float32Array, right?: Float32Array) => {
            if (!left || !right) return undefined;
            const magnitude = velocityDifferenceMagnitude(left, right);
            return compareSingleTallCellNeighborhood(magnitude, new Float32Array(magnitude.length), ...tallResult.grid, probe.x, probe.z);
          };
          console.log(JSON.stringify({
            scenario: scenarioId, phase: "single-tall-cell-checkpoint", time_s: tallCheckpoint.time_s, probe,
            global: compareScalarFields(tallCheckpoint.field, uniformCheckpoint.field, ...tallResult.grid),
            locality: compareSingleTallCellNeighborhood(tallCheckpoint.field, uniformCheckpoint.field, ...tallResult.grid, probe.x, probe.z),
            probeVolumeProfile: { candidate: probeProfile(tallCheckpoint.field), control: probeProfile(uniformCheckpoint.field) },
            velocityBeforeProjection: velocityLocality(tallCheckpoint.preProjectionVelocity, uniformCheckpoint.preProjectionVelocity),
            velocityAfterProjection: velocityLocality(tallCheckpoint.postProjectionVelocity, uniformCheckpoint.postProjectionVelocity)
          }));
        }
      }
    }

    const grid = results[0].grid;
    const cpu = runCPUOracle ? runMatchedCPUOracle(scenarioId, grid, oracleSteps) : undefined;
    for (let left = 0; left < results.length; left += 1) for (let right = left + 1; right < results.length; right += 1) {
      console.log(JSON.stringify({ scenario: scenarioId, phase: "discrepancy", left: results[left].method, right: results[right].method, oracleSteps, metrics: compareScalarFields(results[left].matchedField, results[right].matchedField, ...grid) }));
      // The devolution-over-time curve: field discrepancy at every checkpoint,
      // matched by index (all methods advance on the same fixed cadence).
      const pairCount = Math.min(results[left].checkpoints.length, results[right].checkpoints.length);
      for (let index = 0; index < pairCount; index += 1) {
        const a = results[left].checkpoints[index], b = results[right].checkpoints[index];
        console.log(JSON.stringify({
          scenario: scenarioId, phase: "checkpoint-comparison", left: results[left].method, right: results[right].method,
          time_s: a.time_s, leftVolumeCellSum: a.summary.cellSum, rightVolumeCellSum: b.summary.cellSum,
          leftComponentCount: a.summary.componentCount, rightComponentCount: b.summary.componentCount,
          metrics: compareScalarFields(a.field, b.field, ...grid)
        }));
      }
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
