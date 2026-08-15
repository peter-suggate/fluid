/** Uniform/Sparse CM12 residual-density probe on mini32, including coarse controls. */
import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";
import { sceneDamBreakBox } from "../lib/core/initial-fluid";
import { resolveMethodValues, type SimulationMethod } from "../lib/core/method-contract";
import { createMinimalPowerDamBreak32Scene } from "../lib/core/scenes";
import { sceneAtFinestCellSize } from "../lib/core/scene-scale";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import { uniformMethod } from "../lib/methods/uniform/method";

type Dimensions = readonly [number, number, number];

const argument = (name: string): string | undefined => {
  const prefix = `--${name}=`;
  return process.argv.find((value) => value.startsWith(prefix))?.slice(prefix.length);
};

const fineDimensions = [32, 32, 32] as const;
const dt_s = 1 / 30;
const seconds = Number(argument("seconds") ?? 16 / 3);
const steps = Math.ceil(seconds / dt_s);
const sparseResolutionMode = argument("sparse-resolution") ?? "all-fine";
if (sparseResolutionMode !== "adaptive" && sparseResolutionMode !== "all-fine"
  && sparseResolutionMode !== "all-coarse") {
  throw new RangeError("sparse-resolution must be adaptive, all-fine, or all-coarse");
}
const uniformResolutionMode = argument("uniform-resolution") ?? "fine";
if (uniformResolutionMode !== "fine" && uniformResolutionMode !== "matched") {
  throw new RangeError("uniform-resolution must be fine or matched");
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
  const dam = sceneDamBreakBox(createMinimalPowerDamBreak32Scene());
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
  const authoredScene = createMinimalPowerDamBreak32Scene();
  const scene = latticeScale === 1 ? authoredScene : sceneAtFinestCellSize(
    authoredScene,
    latticeScale * authoredScene.voxelDomain.finestCellSize_m,
  );
  const values = resolveMethodValues(method, "balanced", method.id === "uniform"
    ? { timeStep: "paper", densityPostProcessing: "off" }
    : { timeStep: "paper", resolutionMode: sparseResolutionMode });
  const solver = await method.createSolverAsync!(device, scene, "balanced", values,
    undefined, () => {});
  try {
    assert.deepEqual([solver.info.nx, solver.info.ny, solver.info.nz], dimensions);
    await device.queue.onSubmittedWorkDone();
    const initialDensity = await readDensity(device, solver.volumeTexture, dimensions);
    const initialMass_cells = integratedMassInFineCells(
      initialDensity, latticeScale ** 3,
    );
    for (let step = 1; step <= steps; step += 1) {
      while (!solver.advanceTo(step * dt_s, [])) await new Promise(setImmediate);
    }
    await device.queue.onSubmittedWorkDone();
    const density = await readDensity(device, solver.volumeTexture, dimensions);
    assert.ok(solver.surfaceFieldTexture);
    const levelSet = await readDensity(device, solver.surfaceFieldTexture, dimensions);
    const owners = solver.gridCellTexture
      ? await readOwners(device, solver.gridCellTexture, dimensions) : undefined;
    const finestCellSize_m = Math.min(
      scene.container.width_m / dimensions[0],
      scene.container.height_m / dimensions[1],
      scene.container.depth_m / dimensions[2],
    );
    const surfaceDensity = Float32Array.from(levelSet, (phi) =>
      Math.max(0, Math.min(1.5, 0.5 - phi / (4 * finestCellSize_m))));
    const info = await solver.readStats();
    const mass_cells = integratedMassInFineCells(density, latticeScale ** 3);
    return {
      density,
      receipt: {
        method: method.id,
        residual: residual(density, dimensions, latticeScale ** 3),
        upperWallFilm: upperWallFilm(surfaceDensity, dimensions),
        presentationInterpolation: presentationInterpolationError(
          density, surfaceDensity, owners, dimensions,
        ),
        maximumDensity: density.reduce((maximum, value) => Math.max(maximum, value), 0),
        maximumSurfaceDensity: surfaceDensity.reduce(
          (maximum, value) => Math.max(maximum, value), 0),
        maximumSpeed_m_s: info.maxSpeed_m_s,
        pressureRelativeResidual: info.pressureRelativeResidual,
        maximumPostProjectionDivergence_s: info.maxDivergenceAfter_s,
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
    const uniform = await run(
      device, uniformMethod, uniformDimensions, uniformLatticeScale,
    );
    const sparse = await run(device, adaptiveMassMethod, fineDimensions, 1);
    const uniformDensity = upsampleDensityNearest(
      uniform.density, uniformDimensions, fineDimensions,
    );
    let absolute = 0, squared = 0, uniformAbsolute = 0, uniformSquared = 0;
    let maximumAbsolute = 0, supportIntersection = 0, supportUnion = 0;
    for (let index = 0; index < uniformDensity.length; index += 1) {
      const difference = sparse.density[index] - uniformDensity[index];
      absolute += Math.abs(difference);
      squared += difference * difference;
      uniformAbsolute += Math.abs(uniformDensity[index]);
      uniformSquared += uniformDensity[index] ** 2;
      maximumAbsolute = Math.max(maximumAbsolute, Math.abs(difference));
      const uniformSupported = uniformDensity[index] > 1e-3;
      const sparseSupported = sparse.density[index] > 1e-3;
      supportIntersection += uniformSupported && sparseSupported ? 1 : 0;
      supportUnion += uniformSupported || sparseSupported ? 1 : 0;
    }
    console.log(JSON.stringify({
      scene: "minimal-power-dam-break-32",
      sparseResolutionMode,
      uniformResolutionMode,
      grids: { sparse: fineDimensions, uniform: uniformDimensions },
      time_s: steps * dt_s,
      uniform: uniform.receipt,
      sparse: sparse.receipt,
      fieldDifference: {
        relativeL1: absolute / Math.max(1e-30, uniformAbsolute),
        relativeL2: Math.sqrt(squared / Math.max(1e-30, uniformSquared)),
        maximumAbsolute,
        supportIntersectionOverUnion1e3:
          supportIntersection / Math.max(1, supportUnion),
      },
      validationErrors,
    }, null, 2));
  } finally {
    device.destroy();
  }
} finally {
  await releaseWebGPUExclusiveLock();
}
