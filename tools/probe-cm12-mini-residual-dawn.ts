/** Uniform/Sparse CM12 residual-density probe on mini32, including coarse controls. */
import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";
import { sceneDamBreakBox } from "../lib/core/initial-fluid";
import { resolveMethodValues, type SimulationMethod } from "../lib/core/method-contract";
import {
  createMinimalPowerDamBreak32Scene,
  createMinimalPowerDamBreak64Scene,
} from "../lib/core/scenes";
import { sceneAtFinestCellSize } from "../lib/core/scene-scale";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { uniformMethod } from "../lib/methods/uniform/method";

type Dimensions = readonly [number, number, number];

const argument = (name: string): string | undefined => {
  const prefix = `--${name}=`;
  return process.argv.find((value) => value.startsWith(prefix))?.slice(prefix.length);
};

const gridSize = Number(argument("grid") ?? 32);
if (gridSize !== 32 && gridSize !== 64) {
  throw new RangeError("grid must be 32 or 64");
}
const buildScene = gridSize === 64
  ? createMinimalPowerDamBreak64Scene : createMinimalPowerDamBreak32Scene;
const fineDimensions = [gridSize, gridSize, gridSize] as Dimensions;
const dt_s = 1 / 30;
const seconds = Number(argument("seconds") ?? 16 / 3);
const steps = Math.ceil(seconds / dt_s);
const arm = argument("arm") ?? "both";
if (arm !== "both" && arm !== "uniform" && arm !== "sparse") {
  throw new RangeError("arm must be both, uniform, or sparse");
}
const stageLimit = argument("stage-limit");
const stageFromStep = Number(argument("stage-from-step") ?? 2);
const sparseResolutionMode = argument("sparse-resolution") ?? "all-fine";
if (sparseResolutionMode !== "adaptive" && sparseResolutionMode !== "all-fine"
  && sparseResolutionMode !== "all-coarse" && sparseResolutionMode !== "region") {
  throw new RangeError(
    "sparse-resolution must be adaptive, all-fine, all-coarse, or region",
  );
}
const uniformResolutionMode = argument("uniform-resolution") ?? "fine";
if (uniformResolutionMode !== "fine" && uniformResolutionMode !== "matched") {
  throw new RangeError("uniform-resolution must be fine or matched");
}
const uniformVelocityTransport = argument("uniform-velocity-transport")
  ?? "semi-lagrangian";
if (uniformVelocityTransport !== "semi-lagrangian"
  && uniformVelocityTransport !== "maccormack") {
  throw new RangeError(
    "uniform-velocity-transport must be semi-lagrangian or maccormack",
  );
}
const uniformLatticeScale: 1 | 2 = uniformResolutionMode === "matched"
  && sparseResolutionMode === "all-coarse" ? 2 : 1;
const uniformDimensions = fineDimensions.map((value) =>
  value / uniformLatticeScale) as unknown as Dimensions;

async function readDensity(
  device: GPUDevice,
  texture: GPUTexture,
  dimensions: Dimensions,
): Promise<Float32Array> {
  const [nx, ny, nz] = dimensions;
  const bytesPerRow = Math.ceil(nx * 4 / 256) * 256;
  const buffer = device.createBuffer({
    size: bytesPerRow * ny * nz,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow, rowsPerImage: ny }, dimensions);
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const source = new Float32Array(buffer.getMappedRange());
    const output = new Float32Array(nx * ny * nz);
    const stride = bytesPerRow / 4;
    for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
      output.set(source.subarray(stride * (y + ny * z), stride * (y + ny * z) + nx),
        nx * (y + ny * z));
    }
    return output;
  } finally {
    if (buffer.mapState === "mapped") buffer.unmap();
    buffer.destroy();
  }
}

async function readVelocity(
  device: GPUDevice,
  texture: GPUTexture,
  dimensions: Dimensions,
): Promise<Float32Array> {
  const [nx, ny, nz] = dimensions;
  const bytesPerRow = Math.ceil(nx * 16 / 256) * 256;
  const buffer = device.createBuffer({
    size: bytesPerRow * ny * nz,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer(
      { texture }, { buffer, bytesPerRow, rowsPerImage: ny }, dimensions,
    );
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const source = new Float32Array(buffer.getMappedRange());
    const output = new Float32Array(4 * nx * ny * nz);
    const stride = bytesPerRow / 4;
    for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
      output.set(source.subarray(stride * (y + ny * z),
        stride * (y + ny * z) + 4 * nx), 4 * nx * (y + ny * z));
    }
    return output;
  } finally {
    if (buffer.mapState === "mapped") buffer.unmap();
    buffer.destroy();
  }
}

interface ArmFields {
  readonly density: Float32Array;
  readonly velocity: Float32Array;
  readonly velocityLocation: "collocated" | "positive-mac";
  readonly capacity?: Float32Array;
}

async function readArmFields(
  device: GPUDevice,
  method: SimulationMethod,
  solver: Awaited<ReturnType<NonNullable<SimulationMethod["createSolverAsync"]>>>,
  dimensions: Dimensions,
  frameBank: "accepted" | "candidate" = "accepted",
): Promise<ArmFields> {
  if (method.id === "adaptive-mass") {
    const fields = await (solver as WebGPUAdaptiveMassSolver)
      .readDiagnosticFields(true, frameBank);
    return { density: fields.density, velocity: fields.velocity,
      velocityLocation: "collocated", capacity: fields.solidOpenFraction };
  }
  assert.ok(solver.velocityTexture);
  const [density, velocity] = await Promise.all([
    readDensity(device, solver.volumeTexture, dimensions),
    readVelocity(device, solver.velocityTexture, dimensions),
  ]);
  return { density, velocity, velocityLocation: "positive-mac" };
}

function densityCapacityReceipt(
  fields: ArmFields,
  cellVolumeInFineCells: number,
) {
  let excessMass_cells = 0;
  let maximumExcessDensity = 0;
  let cellsOverCapacity = 0;
  let cellsOverTwiceCapacity = 0;
  for (let index = 0; index < fields.density.length; index += 1) {
    const density = Math.max(0, fields.density[index]!);
    const capacity = Math.max(0, fields.capacity?.[index] ?? 1);
    const excess = Math.max(0, density - capacity);
    excessMass_cells += cellVolumeInFineCells * excess;
    maximumExcessDensity = Math.max(maximumExcessDensity, excess);
    cellsOverCapacity += excess > 1e-6 ? 1 : 0;
    cellsOverTwiceCapacity += density > 2 * Math.max(capacity, 1e-6) ? 1 : 0;
  }
  return { excessMass_cells, maximumExcessDensity, cellsOverCapacity,
    cellsOverTwiceCapacity };
}

function mechanicalReceipt(
  fields: ArmFields,
  dimensions: Dimensions,
  container_m: readonly [number, number, number],
  cellVolume_m3: number,
  gravity: readonly [number, number, number],
) {
  const [nx, ny, nz] = dimensions;
  const collocated = (x: number, y: number, z: number, axis: number) => {
    const index = x + nx * (y + ny * z);
    const positive = fields.velocity[4 * index + axis]!;
    if (fields.velocityLocation === "collocated") return positive;
    const q = [x, y, z];
    q[axis] -= 1;
    const negative = q[axis]! < 0 ? 0 : fields.velocity[
      4 * (q[0]! + nx * (q[1]! + ny * q[2]!)) + axis
    ]!;
    return 0.5 * (negative + positive);
  };
  let kineticEnergyProxy = 0;
  let gravitationalPotentialEnergyProxy = 0;
  let maximumLiquidSpeed_m_s = 0;
  let densityWeightedSpeedSum = 0;
  let mass = 0;
  const momentum = [0, 0, 0];
  const spacing = container_m.map((extent, axis) => extent / dimensions[axis]);
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
    for (let x = 0; x < nx; x += 1) {
      const index = x + nx * (y + ny * z);
      const rho = Math.max(0, fields.density[index]!);
      const velocity = [0, 1, 2].map((axis) => collocated(x, y, z, axis));
      const speed2 = velocity.reduce((sum, value) => sum + value * value, 0);
      const weightedMass = rho * cellVolume_m3;
      kineticEnergyProxy += 0.5 * weightedMass * speed2;
      densityWeightedSpeedSum += weightedMass * Math.sqrt(speed2);
      mass += weightedMass;
      for (let axis = 0; axis < 3; axis += 1) {
        momentum[axis] += weightedMass * velocity[axis]!;
      }
      const position = [(x + 0.5 - nx / 2) * spacing[0]!,
        (y + 0.5) * spacing[1]!, (z + 0.5 - nz / 2) * spacing[2]!];
      gravitationalPotentialEnergyProxy -= weightedMass * position.reduce(
        (sum, value, axis) => sum + value * gravity[axis]!, 0,
      );
      if (rho > 0.5) maximumLiquidSpeed_m_s = Math.max(
        maximumLiquidSpeed_m_s, Math.sqrt(speed2),
      );
    }
  }
  return {
    kineticEnergyProxy,
    gravitationalPotentialEnergyProxy,
    mechanicalEnergyProxy: kineticEnergyProxy + gravitationalPotentialEnergyProxy,
    maximumLiquidSpeed_m_s,
    densityWeightedMeanSpeed_m_s: densityWeightedSpeedSum / Math.max(mass, 1e-30),
    momentumProxy: momentum,
  };
}

async function readOwners(
  device: GPUDevice,
  texture: GPUTexture,
  dimensions: Dimensions,
): Promise<Uint32Array> {
  const [nx, ny, nz] = dimensions;
  const bytesPerRow = Math.ceil(nx * 8 / 256) * 256;
  const buffer = device.createBuffer({
    size: bytesPerRow * ny * nz,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow, rowsPerImage: ny }, dimensions);
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const source = new Uint32Array(buffer.getMappedRange());
    const output = new Uint32Array(2 * nx * ny * nz);
    const stride = bytesPerRow / 4;
    for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
      output.set(source.subarray(stride * (y + ny * z), stride * (y + ny * z) + 2 * nx),
        2 * nx * (y + ny * z));
    }
    return output;
  } finally {
    if (buffer.mapState === "mapped") buffer.unmap();
    buffer.destroy();
  }
}

function residual(
  density: Float32Array,
  dimensions: Dimensions,
  cellVolumeInFineCells: number,
) {
  const dam = sceneDamBreakBox(buildScene());
  const maxX = Math.ceil(dam.max.x * dimensions[0]);
  const maxY = Math.ceil(dam.max.y * dimensions[1]);
  const maxZ = Math.ceil(dam.max.z * dimensions[2]);
  const minimumY = Math.ceil(maxY / 2);
  let mass = 0, maximum = 0, cells = 0;
  for (let z = 0; z < maxZ; z += 1) for (let y = minimumY; y < maxY; y += 1) {
    for (let x = 0; x < maxX; x += 1) {
      const rho = density[x + dimensions[0] * (y + dimensions[1] * z)];
      mass += cellVolumeInFineCells * rho;
      maximum = Math.max(maximum, rho);
      cells += rho > 1e-3 ? cellVolumeInFineCells : 0;
    }
  }
  return { mass_cells: mass, maximum, cellsAbove1e3: cells, minimumY, maxY };
}

function upperWallFilm(density: Float32Array, dimensions: Dimensions) {
  const [nx, ny, nz] = dimensions;
  const upperStart = Math.floor(0.75 * ny);
  const wallBand = Math.max(1, Math.floor(Math.min(nx, nz) / 8));
  let upperMass = 0, upperWallMass = 0, upperCornerMass = 0;
  let upperCellsAbove1e3 = 0, upperWallCellsAbove1e3 = 0;
  let upperWallMaximum = 0, upperWallLiquidCells = 0, upperCornerLiquidCells = 0;
  for (let z = 0; z < nz; z += 1) for (let y = upperStart; y < ny; y += 1) {
    for (let x = 0; x < nx; x += 1) {
      const rho = density[x + nx * (y + ny * z)];
      const xWall = x < wallBand || x >= nx - wallBand;
      const zWall = z < wallBand || z >= nz - wallBand;
      upperMass += rho;
      upperCellsAbove1e3 += rho > 1e-3 ? 1 : 0;
      if (xWall || zWall) {
        upperWallMass += rho;
        upperWallCellsAbove1e3 += rho > 1e-3 ? 1 : 0;
        upperWallMaximum = Math.max(upperWallMaximum, rho);
        upperWallLiquidCells += rho >= 0.5 ? 1 : 0;
      }
      if (xWall && zWall) {
        upperCornerMass += rho;
        upperCornerLiquidCells += rho >= 0.5 ? 1 : 0;
      }
    }
  }
  return { upperStart, wallBand, upperMass, upperWallMass, upperCornerMass,
    upperCellsAbove1e3, upperWallCellsAbove1e3, upperWallMaximum,
    upperWallLiquidCells, upperCornerLiquidCells };
}

function integratedMassInFineCells(
  density: Float32Array,
  cellVolumeInFineCells: number,
): number {
  let sum = 0, correction = 0;
  for (const rho of density) {
    const adjusted = cellVolumeInFineCells * rho - correction;
    const next = sum + adjusted;
    correction = next - sum - adjusted;
    sum = next;
  }
  return sum;
}

function ownerScale(owners: Uint32Array, index: number): number {
  const first = owners[2 * index] >>> 0;
  const second = owners[2 * index + 1] >>> 0;
  return (second & 0x8000_0000) !== 0
    ? 2 ** ((first >>> 22) & 0xf)
    : (first >>> 20) & 0x3ff;
}

function presentationInterpolationError(
  density: Float32Array,
  surfaceDensity: Float32Array,
  owners: Uint32Array | undefined,
  dimensions: Dimensions,
) {
  const [nx, ny, nz] = dimensions;
  if (!owners) return { relativeL1: 0, maximumAbsolute: 0, coarseSamples: 0 };
  const restricted = (x: number, y: number, z: number, scale: number) => {
    let value = 0;
    for (let dz = 0; dz < scale; dz += 1) for (let dy = 0; dy < scale; dy += 1) {
      for (let dx = 0; dx < scale; dx += 1) {
        value += density[scale * x + dx + nx
          * (scale * y + dy + ny * (scale * z + dz))];
      }
    }
    return value / scale ** 3;
  };
  let absolute = 0, reference = 0, maximumAbsolute = 0;
  let coarseSamples = 0;
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
    for (let x = 0; x < nx; x += 1) {
      const index = x + nx * (y + ny * z);
      const scale = ownerScale(owners, index);
      if (scale !== 2) continue;
      coarseSamples += 1;
      const coarse = [nx / scale, ny / scale, nz / scale] as const;
      const position = [x, y, z].map((value) => (value + 0.5) / scale - 0.5);
      const lower = position.map(Math.floor);
      const fraction = position.map((value, axis) => value - lower[axis]);
      let expected = 0;
      for (let dz = 0; dz < 2; dz += 1) for (let dy = 0; dy < 2; dy += 1) {
        for (let dx = 0; dx < 2; dx += 1) {
          const offset = [dx, dy, dz];
          const q = offset.map((value, axis) => Math.max(0,
            Math.min(coarse[axis] - 1, lower[axis] + value)));
          const weight = offset.reduce((product, value, axis) => product
            * (value === 0 ? 1 - fraction[axis] : fraction[axis]), 1);
          expected += weight * restricted(q[0], q[1], q[2], scale);
        }
      }
      const actual = surfaceDensity[index];
      const error = Math.abs(actual - expected);
      absolute += error;
      reference += Math.abs(expected);
      maximumAbsolute = Math.max(maximumAbsolute, error);
    }
  }
  return { relativeL1: absolute / Math.max(reference, Number.MIN_VALUE),
    maximumAbsolute, coarseSamples };
}

async function run(
  device: GPUDevice,
  method: SimulationMethod,
  dimensions: Dimensions,
  latticeScale: 1 | 2,
) {
  const authoredScene = buildScene();
  const scene = latticeScale === 1 ? authoredScene : sceneAtFinestCellSize(
    authoredScene,
    latticeScale * authoredScene.voxelDomain.finestCellSize_m,
  );
  if (method.id === "adaptive-mass" && sparseResolutionMode === "region") {
    scene.fluid.refinementRegions = [{
      id: "whole-tank-one-cell",
      rule: "minimum-cell-size",
      minimumCellSize_cells: 1,
      maximumCellSize_cells: 1,
      min_m: { x: -0.5 * scene.container.width_m, y: 0,
        z: -0.5 * scene.container.depth_m },
      max_m: { x: 0.5 * scene.container.width_m, y: scene.container.height_m,
        z: 0.5 * scene.container.depth_m },
    }];
  }
  const optionalNumber = (name: string) => {
    const raw = argument(name);
    if (raw === undefined) return undefined;
    const value = Number(raw);
    if (!Number.isFinite(value)) throw new RangeError(`${name} must be finite`);
    return value;
  };
  const sparseOverrides = {
    timeStep: "paper",
    resolutionMode: sparseResolutionMode === "region" ? "adaptive" : sparseResolutionMode,
    ...(argument("gamma-diffusion") === "off" ? { gammaDiffusion: "off" } : {}),
    ...(argument("surface-sharpening") === "off"
      ? { surfaceSharpening: "off" } : {}),
    ...(optionalNumber("sharpening-strength") === undefined ? {}
      : { sharpeningStrength: optionalNumber("sharpening-strength")! }),
    ...(optionalNumber("pressure-iterations") === undefined ? {}
      : { pressureIterations: optionalNumber("pressure-iterations")! }),
    ...(optionalNumber("pressure-tolerance") === undefined ? {}
      : { pressureRelativeTolerance: optionalNumber("pressure-tolerance")! }),
  };
  const values = resolveMethodValues(method, "balanced", method.id === "uniform"
    ? { timeStep: "paper", densityPostProcessing: "off",
      velocityTransport: uniformVelocityTransport }
    : sparseOverrides);
  const solver = await method.createSolverAsync!(device, scene, "balanced", values,
    undefined, () => {});
  try {
    assert.deepEqual([solver.info.nx, solver.info.ny, solver.info.nz], dimensions);
    await device.queue.onSubmittedWorkDone();
    const initialFields = await readArmFields(device, method, solver, dimensions);
    const initialDensity = initialFields.density;
    const initialMass_cells = integratedMassInFineCells(
      initialDensity, latticeScale ** 3,
    );
    const cellVolume_m3 = scene.container.width_m * scene.container.height_m
      * scene.container.depth_m / (dimensions[0] * dimensions[1] * dimensions[2]);
    const gravity = [scene.fluid.gravity_m_s2.x, scene.fluid.gravity_m_s2.y,
      scene.fluid.gravity_m_s2.z] as const;
    const container_m = [scene.container.width_m, scene.container.height_m,
      scene.container.depth_m] as const;
    const sampleEvery = Math.max(1, Math.round(Number(argument("sample-every") ?? 10)));
    const trajectory = [{ step: 0, time_s: 0,
      ...mechanicalReceipt(initialFields, dimensions, container_m, cellVolume_m3,
        gravity) }];
    let finalFields = initialFields;
    const advanceStartedAt_ms = performance.now();
    for (let step = 1; step <= steps; step += 1) {
      if (method.id === "adaptive-mass" && stageLimit && step >= stageFromStep) {
        (solver as WebGPUAdaptiveMassSolver).sparseWorldTrace
          .setStageLimitForQA(stageLimit as never);
      }
      while (!solver.advanceTo(step * dt_s, [])) await new Promise(setImmediate);
      if (step % sampleEvery === 0 || step === steps) {
        await device.queue.onSubmittedWorkDone();
        finalFields = await readArmFields(device, method, solver, dimensions,
          method.id === "adaptive-mass" && stageLimit && step >= stageFromStep
            ? "candidate" : "accepted");
        trajectory.push({ step, time_s: step * dt_s,
          ...mechanicalReceipt(finalFields, dimensions, container_m, cellVolume_m3,
            gravity) });
      }
    }
    await device.queue.onSubmittedWorkDone();
    const advanceWall_ms = performance.now() - advanceStartedAt_ms;
    const density = finalFields.density;
    const sparseSolver = method.id === "adaptive-mass"
      ? solver as WebGPUAdaptiveMassSolver : undefined;
    const activity = sparseSolver ? await sparseSolver.readGPUActivityPolicy() : undefined;
    const levelSet = method.id === "uniform"
      ? await readDensity(device, solver.surfaceFieldTexture!, dimensions)
      : density;
    const owners = method.id === "uniform" && solver.gridCellTexture
      ? await readOwners(device, solver.gridCellTexture, dimensions) : undefined;
    const finestCellSize_m = Math.min(
      scene.container.width_m / dimensions[0],
      scene.container.height_m / dimensions[1],
      scene.container.depth_m / dimensions[2],
    );
    const surfaceDensity = method.id === "uniform"
      ? Float32Array.from(levelSet, (phi) =>
        Math.max(0, Math.min(1.5, 0.5 - phi / (4 * finestCellSize_m))))
      : density;
    const info = await solver.readStats();
    const mass_cells = integratedMassInFineCells(density, latticeScale ** 3);
    return {
      density,
      receipt: {
        method: method.id,
        advanceWall_ms,
        advanceWallPerStep_ms: advanceWall_ms / steps,
        residual: residual(density, dimensions, latticeScale ** 3),
        upperWallFilm: upperWallFilm(surfaceDensity, dimensions),
        presentationInterpolation: presentationInterpolationError(
          density, surfaceDensity, owners, dimensions,
        ),
        maximumDensity: density.reduce((maximum, value) => Math.max(maximum, value), 0),
        densityCapacity: densityCapacityReceipt(finalFields, latticeScale ** 3),
        maximumSurfaceDensity: surfaceDensity.reduce(
          (maximum, value) => Math.max(maximum, value), 0),
        maximumSpeed_m_s: info.maxSpeed_m_s,
        pressureRelativeResidual: info.pressureRelativeResidual,
        maximumPostProjectionDivergence_s: info.maxDivergenceAfter_s,
        reportedVolumeCellSum: info.volumeCellSum,
        representedVolumeCellSum: info.representedVolumeCellSum,
        representedVolumeDrift: info.representedVolumeDrift,
        initialMass_cells,
        mass_cells,
        massDrift_cells: mass_cells - initialMass_cells,
        relativeMassDrift: (mass_cells - initialMass_cells)
          / Math.max(initialMass_cells, Number.MIN_VALUE),
        fineBricks: info.adaptiveFineBrickCount,
        coarseBricks: info.adaptiveCoarseBrickCount,
        activityMaximumScore: info.adaptiveActivityMaximumScore,
        activityMeasuredBricks: info.adaptiveActivityMeasuredBrickCount,
        activitySurfaceBricks: info.adaptiveActivitySurfaceBrickCount,
        activityHotBricks: info.adaptiveActivityHotBrickCount,
        activityQuietBricks: info.adaptiveActivityQuietBrickCount,
        activityTopologyEpoch: info.adaptiveResolutionTopologyEpoch,
        activeResolutionHistogram: activity ? Object.fromEntries(
          [...new Set(activity.bricks.filter((brick) => brick.active)
            .map((brick) => brick.acceptedResolution))].sort((a, b) => a - b)
            .map((resolution) => [resolution, activity.bricks.filter((brick) =>
              brick.active && brick.acceptedResolution === resolution).length]),
        ) : undefined,
        trajectory,
      },
    };
  } finally {
    solver.destroy();
  }
}

function upsampleDensityNearest(
  source: Float32Array,
  sourceDimensions: Dimensions,
  targetDimensions: Dimensions,
): Float32Array {
  if (sourceDimensions.every((value, axis) => value === targetDimensions[axis])) {
    return source;
  }
  const [sx, sy] = sourceDimensions;
  const [tx, ty, tz] = targetDimensions;
  const result = new Float32Array(tx * ty * tz);
  for (let z = 0; z < tz; z += 1) for (let y = 0; y < ty; y += 1) {
    for (let x = 0; x < tx; x += 1) {
      const sourceX = Math.floor(x * sourceDimensions[0] / tx);
      const sourceY = Math.floor(y * sourceDimensions[1] / ty);
      const sourceZ = Math.floor(z * sourceDimensions[2] / tz);
      result[x + tx * (y + ty * z)] = source[
        sourceX + sx * (sourceY + sy * sourceZ)
      ];
    }
  }
  return result;
}

await acquireWebGPUExclusiveLock("dawn-acceptance", "tools/probe-cm12-mini-residual-dawn.ts");
try {
  const modulePath = process.env.WEBGPU_NODE_MODULE
    ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));
  const dawn = await import(pathToFileURL(modulePath).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter);
  const device = await adapter.requestDevice({
    requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) =>
    validationErrors.push(event.error.message));
  try {
    const uniform = arm === "sparse" ? undefined : await run(
      device, uniformMethod, uniformDimensions, uniformLatticeScale,
    );
    const sparse = arm === "uniform" ? undefined
      : await run(device, adaptiveMassMethod, fineDimensions, 1);
    const uniformDensity = uniform && upsampleDensityNearest(
      uniform.density, uniformDimensions, fineDimensions,
    );
    let absolute = 0, squared = 0, uniformAbsolute = 0, uniformSquared = 0;
    let maximumAbsolute = 0, supportIntersection = 0, supportUnion = 0;
    const comparisonLength = uniformDensity && sparse ? uniformDensity.length : 0;
    for (let index = 0; index < comparisonLength; index += 1) {
      const difference = sparse!.density[index] - uniformDensity![index];
      absolute += Math.abs(difference);
      squared += difference * difference;
      uniformAbsolute += Math.abs(uniformDensity![index]);
      uniformSquared += uniformDensity![index] ** 2;
      maximumAbsolute = Math.max(maximumAbsolute, Math.abs(difference));
      const uniformSupported = uniformDensity![index] > 1e-3;
      const sparseSupported = sparse!.density[index] > 1e-3;
      supportIntersection += uniformSupported && sparseSupported ? 1 : 0;
      supportUnion += uniformSupported || sparseSupported ? 1 : 0;
    }
    console.log(JSON.stringify({
      scene: `minimal-power-dam-break-${gridSize}`,
      sparseResolutionMode,
      uniformResolutionMode,
      grids: { sparse: fineDimensions, uniform: uniformDimensions },
      time_s: steps * dt_s,
      uniform: uniform?.receipt,
      sparse: sparse?.receipt,
      fieldDifference: uniform && sparse ? {
        relativeL1: absolute / Math.max(1e-30, uniformAbsolute),
        relativeL2: Math.sqrt(squared / Math.max(1e-30, uniformSquared)),
        maximumAbsolute,
        supportIntersectionOverUnion1e3:
          supportIntersection / Math.max(1, supportUnion),
      } : undefined,
      validationErrors,
    }, null, 2));
  } finally {
    device.destroy();
  }
} finally {
  await releaseWebGPUExclusiveLock();
}
