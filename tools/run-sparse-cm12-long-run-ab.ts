/**
 * Long-run physics A/B for Uniform CM12 and mixed-resolution Sparse CM12.
 *
 * Both arms use the same authored symmetric-expansion scene, finest lattice,
 * dt and target time. Construction is excluded; every accepted step is fenced
 * and the dense publications are compared at t=0, halfway and the final time.
 *
 * Run:
 *   WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
 *     node --import tsx tools/run-sparse-cm12-long-run-ab.ts --seconds=2
 *
 * Force every sparse tile one rung coarser and compare it either to the full
 * fine Uniform grid or the physically equivalent reduced Uniform grid:
 *   ... --sparse-resolution=all-coarse --uniform-resolution=fine
 *   ... --sparse-resolution=all-coarse --uniform-resolution=matched
 *
 * Uncalibrated UI-sized timestep stress:
 *   WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
 *     node --import tsx tools/run-sparse-cm12-long-run-ab.ts --seconds=2 --regime=scene --dt=0.004
 */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { fileURLToPath, pathToFileURL } from "node:url";
import type { MethodParamValues, SimulationMethod } from
  "../lib/core/method-contract";
import { resolveMethodValues } from "../lib/core/method-contract";
import { initialFluidBrickUnionBounds } from "../lib/core/initial-fluid";
import { createSymmetricExpansionScene } from "../lib/core/scenes";
import { sceneAtFinestCellSize } from "../lib/core/scene-scale";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import type { AdaptiveMassResolutionMode } from
  "../lib/methods/adaptive-mass/method";
import type { AdaptiveMassStepTelemetry } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { uniformMethod } from "../lib/methods/uniform/method";

type Dimensions = readonly [number, number, number];
type FixedSparseResolutionMode = Extract<
  AdaptiveMassResolutionMode,
  "all-fine" | "all-coarse"
>;

function fieldHash(values: Float32Array): string {
  return createHash("sha256").update(new Uint8Array(
    values.buffer, values.byteOffset, values.byteLength,
  )).digest("hex");
}

const argument = (name: string): string | undefined => {
  const prefix = `--${name}=`;
  return process.argv.slice(2).find((value) => value.startsWith(prefix))
    ?.slice(prefix.length);
};

const positiveNumber = (name: string, fallback: number): number => {
  const value = Number(argument(name) ?? fallback);
  if (!Number.isFinite(value) || value <= 0) {
    throw new RangeError(`${name} must be finite and positive`);
  }
  return value;
};

interface Symmetry {
  readonly densityMaximumAbsolute: number;
  readonly velocityMaximumAbsolute_m_s: number;
}

interface FieldReceipt {
  readonly step: number;
  readonly time_s: number;
  readonly mass_cells: number;
  readonly relativeMassDrift: number;
  readonly densityWeightedKineticEnergyProxy: number;
  readonly maximumLiquidSpeed_m_s: number;
  readonly maximumDensity: number;
  readonly centerOfMassNormalized: readonly [number, number, number];
  readonly massStandardDeviationNormalized: readonly [number, number, number];
  readonly supportExtentNormalized: readonly [number, number, number];
  readonly symmetry: Symmetry;
}

interface StepReceipt {
  readonly maximumPressureRelativeResidual: number;
  readonly maximumPostProjectionDivergence_s: number;
  readonly maximumCfl: number;
  readonly maximumPressureIterations: number;
  readonly maximumSpeed_m_s: number;
  readonly maximumInactiveFaceSpeedBefore_m_s: number;
  readonly maximumInactiveFaceSpeedAfter_m_s: number;
  readonly maximumMixedSeamDivergence_s: number;
  readonly maximumMixedSeamRows: number;
  readonly minimumEvolvedMixedSeamRows: number;
  readonly maximumFineCoarseConnectedPairs: number;
  readonly maximumDensityAfterTransport: number;
  readonly maximumDensityAfterConditioning: number;
}

interface MutableStepReceipt {
  maximumPressureRelativeResidual: number;
  maximumPostProjectionDivergence_s: number;
  maximumCfl: number;
  maximumPressureIterations: number;
  maximumSpeed_m_s: number;
  maximumInactiveFaceSpeedBefore_m_s: number;
  maximumInactiveFaceSpeedAfter_m_s: number;
  maximumMixedSeamDivergence_s: number;
  maximumMixedSeamRows: number;
  minimumEvolvedMixedSeamRows: number;
  maximumFineCoarseConnectedPairs: number;
  maximumDensityAfterTransport: number;
  maximumDensityAfterConditioning: number;
}

interface ArmReceipt {
  readonly method: string;
  readonly checkpoints: readonly FieldReceipt[];
  readonly evolution: StepReceipt;
  readonly performance: {
    readonly totalStepWallTime_ms: number;
    readonly meanStepWallTime_ms: number;
  };
  readonly sparseTopology?: {
    readonly maximumFineBrickCount: number;
    readonly maximumCoarseBrickCount: number;
    readonly maximumActiveCellCount: number;
    readonly finalFineBrickCount: number;
    readonly finalCoarseBrickCount: number;
  };
}

async function readTexture(
  device: GPUDevice,
  texture: GPUTexture,
  dimensions: Dimensions,
  channels: 1 | 4,
): Promise<Float32Array> {
  const [nx, ny, nz] = dimensions;
  const rowBytes = nx * channels * 4;
  const bytesPerRow = Math.ceil(rowBytes / 256) * 256;
  const readback = device.createBuffer({
    label: "Sparse CM12 long-run A/B readback",
    size: bytesPerRow * ny * nz,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer(
      { texture },
      { buffer: readback, bytesPerRow, rowsPerImage: ny },
      [nx, ny, nz],
    );
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const mapped = new Float32Array(readback.getMappedRange());
    const output = new Float32Array(nx * ny * nz * channels);
    const stride = bytesPerRow / 4;
    for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
      output.set(
        mapped.subarray(stride * (y + ny * z), stride * (y + ny * z) + channels * nx),
        channels * nx * (y + ny * z),
      );
    }
    return output;
  } finally {
    if (readback.mapState === "mapped") readback.unmap();
    readback.destroy();
  }
}

function fieldReceipt(
  step: number,
  dt_s: number,
  density: Float32Array,
  velocity: Float32Array,
  dimensions: Dimensions,
  initialMass: number,
  velocityLocation: "collocated" | "negative-mac",
  cellVolumeInFineCells = 1,
): FieldReceipt {
  const [nx, ny, nz] = dimensions;
  let mass = 0;
  let kinetic = 0;
  let maximumLiquidSpeed = 0;
  let maximumDensity = 0;
  const firstMoment = [0, 0, 0];
  const secondMoment = [0, 0, 0];
  const supportMinimum = [nx, ny, nz];
  const supportMaximum = [-1, -1, -1];
  let densitySymmetry = 0;
  let velocitySymmetry = 0;
  const cell = (x: number, y: number, z: number) => x + nx * (y + ny * z);
  const collocated = new Float32Array(3 * nx * ny * nz);
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
    for (let x = 0; x < nx; x += 1) {
      const index = cell(x, y, z);
      for (const component of [0, 1, 2] as const) {
        let value = velocity[4 * index + component];
        if (velocityLocation === "negative-mac") {
          // velocityTexture stores the positive x/y/z face of this cell. The
          // collocated value averages it with the previous cell's positive
          // face (the current cell's negative face). Symmetric expansion is
          // detached from the domain wall, so the omitted negative boundary
          // buffer is zero in every density-weighted sample used here.
          const negative = [x, y, z] as [number, number, number];
          negative[component] -= 1;
          value = 0.5 * (value + (negative[component] >= 0
            ? velocity[4 * cell(...negative) + component] : 0));
        }
        collocated[3 * index + component] = value;
      }
    }
  }
  const compareVelocity = (
    source: number,
    target: number,
    component: 0 | 1 | 2,
    targetComponent: 0 | 1 | 2,
    sign: number,
  ) => {
    // Post-projection air velocity is outside the liquid authority and is
    // rebuilt by Sec. 3.3 before the next characteristic trace. Measuring it
    // as fluid symmetry conflates inactive storage with physical momentum.
    if (density[source] <= 0.5 && density[target] <= 0.5) return;
    velocitySymmetry = Math.max(velocitySymmetry, Math.abs(
      collocated[3 * target + targetComponent] - sign * collocated[3 * source + component],
    ));
  };
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
    for (let x = 0; x < nx; x += 1) {
      const index = cell(x, y, z);
      const rho = density[index];
      const weightedRho = cellVolumeInFineCells * rho;
      const speed = Math.hypot(
        collocated[3 * index], collocated[3 * index + 1], collocated[3 * index + 2],
      );
      mass += weightedRho;
      for (const [axis, coordinate] of [x, y, z].entries()) {
        const centered = coordinate + 0.5;
        firstMoment[axis] += weightedRho * centered;
        secondMoment[axis] += weightedRho * centered * centered;
        if (rho > 1e-3) {
          supportMinimum[axis] = Math.min(supportMinimum[axis], coordinate);
          supportMaximum[axis] = Math.max(supportMaximum[axis], coordinate);
        }
      }
      kinetic += 0.5 * weightedRho * speed * speed;
      maximumDensity = Math.max(maximumDensity, rho);
      if (rho > 0.5) maximumLiquidSpeed = Math.max(maximumLiquidSpeed, speed);
      for (const target of [
        [nx - 1 - x, y, z], [x, y, nz - 1 - z], [z, y, x],
      ] as const) {
        densitySymmetry = Math.max(
          densitySymmetry,
          Math.abs(rho - density[cell(...target)]),
        );
      }
      const reflectX = cell(nx - 1 - x, y, z);
      const reflectZ = cell(x, y, nz - 1 - z);
      const swap = cell(z, y, x);
      for (const component of [0, 1, 2] as const) {
        compareVelocity(index, reflectX, component, component, component === 0 ? -1 : 1);
        compareVelocity(index, reflectZ, component, component, component === 2 ? -1 : 1);
      }
      compareVelocity(index, swap, 0, 2, 1);
      compareVelocity(index, swap, 1, 1, 1);
      compareVelocity(index, swap, 2, 0, 1);
    }
  }
  return {
    step,
    time_s: step * dt_s,
    mass_cells: mass,
    relativeMassDrift: (mass - initialMass) / Math.max(1, initialMass),
    densityWeightedKineticEnergyProxy: kinetic,
    maximumLiquidSpeed_m_s: maximumLiquidSpeed,
    maximumDensity,
    centerOfMassNormalized: firstMoment.map((value, axis) =>
      value / Math.max(1e-30, mass) / dimensions[axis]) as [number, number, number],
    massStandardDeviationNormalized: secondMoment.map((value, axis) => {
      const mean = firstMoment[axis] / Math.max(1e-30, mass);
      return Math.sqrt(Math.max(0, value / Math.max(1e-30, mass) - mean * mean))
        / dimensions[axis];
    }) as [number, number, number],
    supportExtentNormalized: supportMaximum.map((maximum, axis) =>
      maximum >= supportMinimum[axis]
        ? (maximum - supportMinimum[axis] + 1) / dimensions[axis] : 0,
    ) as [number, number, number],
    symmetry: {
      densityMaximumAbsolute: densitySymmetry,
      velocityMaximumAbsolute_m_s: velocitySymmetry,
    },
  };
}

function finiteOrZero(value: number | undefined): number {
  return value !== undefined && Number.isFinite(value) ? value : 0;
}

function downsampleNearest(
  source: Float32Array,
  sourceDimensions: Dimensions,
  scale: number,
  channels: 1 | 4,
): { readonly values: Float32Array; readonly dimensions: Dimensions } {
  if (scale === 1) return { values: source, dimensions: sourceDimensions };
  const dimensions = sourceDimensions.map((value) =>
    value / scale) as unknown as Dimensions;
  const [sx, sy] = sourceDimensions;
  const [nx, ny, nz] = dimensions;
  const values = new Float32Array(channels * nx * ny * nz);
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
    for (let x = 0; x < nx; x += 1) {
      const sourceCell = scale * x + sx * (scale * y + sy * scale * z);
      const targetCell = x + nx * (y + ny * z);
      for (let channel = 0; channel < channels; channel += 1) {
        values[channels * targetCell + channel] =
          source[channels * sourceCell + channel];
      }
    }
  }
  return { values, dimensions };
}

async function runArm(
  device: GPUDevice,
  method: SimulationMethod,
  dimensions: Dimensions,
  dt_s: number,
  steps: number,
  regime: "paper" | "scene",
  latticeScale: 1 | 2,
  sparseResolutionMode: FixedSparseResolutionMode,
  onFinalFields?: (density: Float32Array, velocity: Float32Array) => void,
): Promise<ArmReceipt> {
  const authoredScene = createSymmetricExpansionScene();
  const scene = latticeScale === 1 ? authoredScene : sceneAtFinestCellSize(
    authoredScene,
    latticeScale * authoredScene.voxelDomain.finestCellSize_m,
  );
  if (latticeScale > 1) {
    // An 8^3 seed is a storage primitive, not physical geometry. At the coarse
    // Uniform lattice it would cover twice the intended width on every axis.
    // The symmetric seed union is an exact box, so express that same authored
    // volume as an offset dam box for this resolution-independent control arm.
    const fineDimensions = dimensions.map((value) =>
      value * latticeScale) as unknown as Dimensions;
    const bounds = initialFluidBrickUnionBounds(authoredScene, fineDimensions);
    assert.ok(bounds, "symmetric-expansion seed union must be an exact box");
    scene.fluid.initialCondition = "dam-break";
    scene.fluid.initialDamBreakDimensions_m = {
      x: bounds.maximum.x - bounds.minimum.x,
      y: bounds.maximum.y - bounds.minimum.y,
      z: bounds.maximum.z - bounds.minimum.z,
    };
    scene.fluid.initialDamBreakOrigin_m = {
      x: bounds.minimum.x + 0.5 * scene.container.width_m,
      y: bounds.minimum.y,
      z: bounds.minimum.z + 0.5 * scene.container.depth_m,
    };
    delete scene.fluid.initialBrickSeeds_m;
    delete scene.fluid.initialBrickSeedsAdditive;
  }
  scene.duration_s = steps * dt_s;
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = dt_s;
  const overrides: MethodParamValues = method.id === "uniform"
    ? { timeStep: regime, densityPostProcessing: "off" }
    : { timeStep: regime, resolutionMode: sparseResolutionMode };
  const solver = await method.createSolverAsync!(
    device,
    scene,
    "balanced",
    resolveMethodValues(method, "balanced", overrides),
    undefined,
    () => {},
  );
  const checkpoints: FieldReceipt[] = [];
  const evolution: MutableStepReceipt = {
    maximumPressureRelativeResidual: 0,
    maximumPostProjectionDivergence_s: 0,
    maximumCfl: 0,
    maximumPressureIterations: 0,
    maximumSpeed_m_s: 0,
    maximumInactiveFaceSpeedBefore_m_s: 0,
    maximumInactiveFaceSpeedAfter_m_s: 0,
    maximumMixedSeamDivergence_s: 0,
    maximumMixedSeamRows: 0,
    minimumEvolvedMixedSeamRows: Number.POSITIVE_INFINITY,
    maximumFineCoarseConnectedPairs: 0,
    maximumDensityAfterTransport: 0,
    maximumDensityAfterConditioning: 0,
  };
  let maximumFineBrickCount = 0;
  let maximumCoarseBrickCount = 0;
  let maximumActiveCellCount = 0;
  let finalFineBrickCount = 0;
  let finalCoarseBrickCount = 0;
  try {
    assert.deepEqual([solver.info.nx, solver.info.ny, solver.info.nz], dimensions);
    let initialMass = 0;
    const capture = async (step: number) => {
      const density = await readTexture(device, solver.volumeTexture, dimensions, 1);
      const velocity = await readTexture(device, solver.velocityTexture!, dimensions, 4);
      if (step === steps) onFinalFields?.(density, velocity);
      const publicationScale = method.id === "adaptive-mass"
        && sparseResolutionMode === "all-coarse" ? 2 : 1;
      const analyzedDensity = downsampleNearest(
        density, dimensions, publicationScale, 1,
      );
      const analyzedVelocity = downsampleNearest(
        velocity, dimensions, publicationScale, 4,
      );
      const cellVolumeInFineCells = (latticeScale * publicationScale) ** 3;
      if (step === 0) {
        for (const value of analyzedDensity.values) {
          initialMass += cellVolumeInFineCells * value;
        }
      }
      checkpoints.push(fieldReceipt(
        step,
        dt_s,
        analyzedDensity.values,
        analyzedVelocity.values,
        analyzedDensity.dimensions,
        initialMass,
        method.id === "uniform" ? "negative-mac" : "collocated",
        cellVolumeInFineCells,
      ));
    };
    await capture(0);
    const checkpointSteps = new Set([Math.floor(steps / 2), steps]);
    const stepClockStart = performance.now();
    for (let step = 1; step <= steps; step += 1) {
      while (!solver.advanceTo(step * dt_s, [])) await new Promise(setImmediate);
      const info = await solver.readStats();
      const sparse = info as typeof info & AdaptiveMassStepTelemetry;
      maximumFineBrickCount = Math.max(maximumFineBrickCount,
        finiteOrZero(info.adaptiveFineBrickCount));
      maximumCoarseBrickCount = Math.max(maximumCoarseBrickCount,
        finiteOrZero(info.adaptiveCoarseBrickCount));
      maximumActiveCellCount = Math.max(maximumActiveCellCount,
        finiteOrZero(info.activeSampleCount));
      finalFineBrickCount = finiteOrZero(info.adaptiveFineBrickCount);
      finalCoarseBrickCount = finiteOrZero(info.adaptiveCoarseBrickCount);
      evolution.maximumPressureRelativeResidual = Math.max(
        evolution.maximumPressureRelativeResidual,
        finiteOrZero(info.pressureRelativeResidual),
      );
      evolution.maximumPostProjectionDivergence_s = Math.max(
        evolution.maximumPostProjectionDivergence_s,
        finiteOrZero(info.maxDivergenceAfter_s),
      );
      evolution.maximumCfl = Math.max(evolution.maximumCfl, finiteOrZero(info.maxComponentCfl));
      evolution.maximumPressureIterations = Math.max(
        evolution.maximumPressureIterations, info.pressureIterations,
      );
      evolution.maximumSpeed_m_s = Math.max(
        evolution.maximumSpeed_m_s, finiteOrZero(info.maxSpeed_m_s),
      );
      evolution.maximumInactiveFaceSpeedBefore_m_s = Math.max(
        evolution.maximumInactiveFaceSpeedBefore_m_s,
        finiteOrZero(sparse.adaptiveMaximumInactiveFaceSpeedBefore_m_s),
      );
      evolution.maximumInactiveFaceSpeedAfter_m_s = Math.max(
        evolution.maximumInactiveFaceSpeedAfter_m_s,
        finiteOrZero(sparse.adaptiveMaximumInactiveFaceSpeedAfter_m_s),
      );
      evolution.maximumMixedSeamDivergence_s = Math.max(
        evolution.maximumMixedSeamDivergence_s,
        finiteOrZero(sparse.adaptiveMaximumMixedSeamDivergence_s),
      );
      evolution.maximumMixedSeamRows = Math.max(
        evolution.maximumMixedSeamRows, finiteOrZero(info.adaptiveMixedSeamFaceCount),
      );
      if (method.id === "adaptive-mass") {
        const mixedRows = finiteOrZero(info.adaptiveMixedSeamFaceCount);
        // The t=0 body begins in four fine bricks. The first force step has no
        // outward receiver request yet; measure persistence once the first
        // genuinely connected coarse support brick appears.
        if (mixedRows > 0) {
          evolution.minimumEvolvedMixedSeamRows = Math.min(
            evolution.minimumEvolvedMixedSeamRows,
            mixedRows,
          );
        }
      }
      evolution.maximumFineCoarseConnectedPairs = Math.max(
        evolution.maximumFineCoarseConnectedPairs,
        finiteOrZero(info.adaptiveFineCoarseFaceConnectedPairCount),
      );
      evolution.maximumDensityAfterTransport = Math.max(
        evolution.maximumDensityAfterTransport,
        finiteOrZero(sparse.adaptiveMaximumDensityAfterTransport),
      );
      evolution.maximumDensityAfterConditioning = Math.max(
        evolution.maximumDensityAfterConditioning,
        finiteOrZero(sparse.adaptiveMaximumDensityAfterConditioning),
      );
      if (checkpointSteps.has(step)) await capture(step);
    }
    const totalStepWallTime_ms = performance.now() - stepClockStart;
    if (!Number.isFinite(evolution.minimumEvolvedMixedSeamRows)) {
      evolution.minimumEvolvedMixedSeamRows = 0;
    }
    return {
      method: method.id,
      checkpoints,
      evolution,
      performance: {
        totalStepWallTime_ms,
        meanStepWallTime_ms: totalStepWallTime_ms / steps,
      },
      sparseTopology: method.id === "adaptive-mass" ? {
        maximumFineBrickCount,
        maximumCoarseBrickCount,
        maximumActiveCellCount,
        finalFineBrickCount,
        finalCoarseBrickCount,
      } : undefined,
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
  for (let axis = 0; axis < 3; axis += 1) {
    if (targetDimensions[axis] % sourceDimensions[axis] !== 0) {
      throw new RangeError("comparison grids must have integer refinement ratios");
    }
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

function d4SymmetrizedDensity(
  source: Float32Array,
  dimensions: Dimensions,
): Float32Array {
  const [nx, ny, nz] = dimensions;
  if (nx !== nz) return source;
  const cell = (x: number, y: number, z: number) => x + nx * (y + ny * z);
  return Float32Array.from(source, (_, index) => {
    const x = index % nx;
    const y = Math.floor(index / nx) % ny;
    const z = Math.floor(index / (nx * ny));
    const reflectedX = nx - 1 - x, reflectedZ = nz - 1 - z;
    return (
      source[cell(x, y, z)] + source[cell(reflectedX, y, z)]
      + source[cell(x, y, reflectedZ)] + source[cell(reflectedX, y, reflectedZ)]
      + source[cell(z, y, x)] + source[cell(reflectedZ, y, x)]
      + source[cell(z, y, reflectedX)] + source[cell(reflectedZ, y, reflectedX)]
    ) / 8;
  });
}

const regimeArgument = argument("regime") ?? "paper";
if (regimeArgument !== "paper" && regimeArgument !== "scene") {
  throw new RangeError("regime must be paper or scene");
}
const regime = regimeArgument;
const dt_s = positiveNumber("dt", regime === "paper" ? 1 / 30 : 0.004);
const target_s = positiveNumber("seconds", 2);
const steps = Math.ceil(target_s / dt_s);
const sparseResolutionArgument = argument("sparse-resolution") ?? "all-fine";
if (sparseResolutionArgument !== "all-fine" && sparseResolutionArgument !== "all-coarse") {
  throw new RangeError("sparse-resolution must be all-fine or all-coarse");
}
const sparseResolutionMode = sparseResolutionArgument;
const uniformResolutionMode = argument("uniform-resolution") ?? "fine";
if (uniformResolutionMode !== "fine" && uniformResolutionMode !== "matched") {
  throw new RangeError("uniform-resolution must be fine or matched");
}
const fineDimensions = [32, 16, 32] as const;
const uniformLatticeScale: 1 | 2 = uniformResolutionMode === "matched"
  && sparseResolutionMode === "all-coarse" ? 2 : 1;
const uniformDimensions = fineDimensions.map((value) =>
  value / uniformLatticeScale) as unknown as Dimensions;
await acquireWebGPUExclusiveLock("dawn-acceptance", "tools/run-sparse-cm12-long-run-ab.ts");
try {
  const modulePath = process.env.WEBGPU_NODE_MODULE
    ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));
  const dawn = await import(pathToFileURL(modulePath).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter);
  const features: GPUFeatureName[] = [];
  if (adapter.features.has("timestamp-query")) features.push("timestamp-query");
  const device = await adapter.requestDevice({
    requiredFeatures: features,
    requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => validationErrors.push(event.error.message));
  try {
    let uniformFinalDensity: Float32Array | undefined;
    let uniformFinalVelocity: Float32Array | undefined;
    let sparseFinalDensity: Float32Array | undefined;
    let sparseFinalVelocity: Float32Array | undefined;
    const uniform = await runArm(
      device, uniformMethod, uniformDimensions, dt_s, steps, regime,
      uniformLatticeScale, sparseResolutionMode,
      (density, velocity) => {
        uniformFinalDensity = density;
        uniformFinalVelocity = velocity;
      });
    const sparse = await runArm(
      device, adaptiveMassMethod, fineDimensions, dt_s, steps, regime,
      1, sparseResolutionMode,
      (density, velocity) => {
        sparseFinalDensity = density;
        sparseFinalVelocity = velocity;
      });
    assert.ok(uniformFinalDensity && uniformFinalVelocity
      && sparseFinalDensity && sparseFinalVelocity);
    uniformFinalDensity = upsampleDensityNearest(
      uniformFinalDensity, uniformDimensions, fineDimensions,
    );
    const failures: string[] = [];
    for (const arm of [uniform, sparse]) {
      const final = arm.checkpoints.at(-1)!;
      if (Math.abs(final.relativeMassDrift) > 1e-3) failures.push(`${arm.method}: mass drift`);
      if (!Number.isFinite(final.densityWeightedKineticEnergyProxy)
        || !Number.isFinite(final.maximumLiquidSpeed_m_s)) failures.push(`${arm.method}: energy/speed`);
    }
    const sparseFinal = sparse.checkpoints.at(-1)!;
    if (sparseFinal.symmetry.densityMaximumAbsolute > 1e-3) failures.push("sparse: density D4");
    if (sparseFinal.symmetry.velocityMaximumAbsolute_m_s > 1e-4) failures.push("sparse: velocity D4");
    if (sparse.evolution.maximumPressureRelativeResidual > 1e-8) failures.push("sparse: pressure residual");
    if (sparse.evolution.maximumPostProjectionDivergence_s > 1e-5) failures.push("sparse: divergence");
    if (sparse.evolution.maximumMixedSeamDivergence_s > 1e-5) failures.push("sparse: seam divergence");
    if (sparse.evolution.maximumInactiveFaceSpeedAfter_m_s !== 0) failures.push("sparse: inactive face carry");
    if (sparse.evolution.maximumMixedSeamRows !== 0
      || sparse.evolution.maximumFineCoarseConnectedPairs !== 0) {
      failures.push(`sparse: ${sparseResolutionMode} mode produced a mixed-resolution seam`);
    }
    if (sparseResolutionMode === "all-fine"
      && (sparse.sparseTopology?.maximumCoarseBrickCount ?? 0) !== 0) {
      failures.push("sparse: all-fine mode created a coarse brick");
    }
    if (sparseResolutionMode === "all-coarse"
      && (sparse.sparseTopology?.maximumFineBrickCount ?? 0) !== 0) {
      failures.push("sparse: all-coarse mode created a fine brick");
    }
    const uniformFinal = uniform.checkpoints.at(-1)!;
    const horizontalSpreadRatios = [0, 2].map((axis) =>
      sparseFinal.massStandardDeviationNormalized[axis] / Math.max(
        1e-30,
        uniformFinal.massStandardDeviationNormalized[axis],
      ));
    const supportExtentRatios = [0, 2].map((axis) =>
      sparseFinal.supportExtentNormalized[axis] / Math.max(
        1e-30,
        uniformFinal.supportExtentNormalized[axis],
      ));
    const kineticEnergyRatio = sparseFinal.densityWeightedKineticEnergyProxy
      / Math.max(1e-30, uniformFinal.densityWeightedKineticEnergyProxy);
    const liquidSpeedRatio = sparseFinal.maximumLiquidSpeed_m_s
      / Math.max(1e-30, uniformFinal.maximumLiquidSpeed_m_s);
    const maximumDensityRatio = sparseFinal.maximumDensity
      / Math.max(1e-30, uniformFinal.maximumDensity);
    const centerOfMassYDifference = Math.abs(
      sparseFinal.centerOfMassNormalized[1] - uniformFinal.centerOfMassNormalized[1],
    );
    let densityAbsolute = 0, densitySquared = 0;
    let uniformDensityAbsolute = 0, uniformDensitySquared = 0;
    let densityMaximumAbsolute = 0, supportIntersection = 0, supportUnion = 0;
    let symmetrizedUniformDensityAbsolute = 0;
    const symmetrizedUniformDensity = d4SymmetrizedDensity(
      uniformFinalDensity, fineDimensions,
    );
    for (let index = 0; index < uniformFinalDensity.length; index += 1) {
      const difference = sparseFinalDensity[index] - uniformFinalDensity[index];
      densityAbsolute += Math.abs(difference);
      densitySquared += difference * difference;
      uniformDensityAbsolute += Math.abs(uniformFinalDensity[index]);
      uniformDensitySquared += uniformFinalDensity[index] ** 2;
      densityMaximumAbsolute = Math.max(densityMaximumAbsolute, Math.abs(difference));
      const uniformSupported = uniformFinalDensity[index] > 1e-3;
      const sparseSupported = sparseFinalDensity[index] > 1e-3;
      supportIntersection += uniformSupported && sparseSupported ? 1 : 0;
      supportUnion += uniformSupported || sparseSupported ? 1 : 0;
      symmetrizedUniformDensityAbsolute += Math.abs(
        sparseFinalDensity[index] - symmetrizedUniformDensity[index],
      );
    }
    const densityRelativeL1 = densityAbsolute / Math.max(1e-30, uniformDensityAbsolute);
    const densityRelativeL2 = Math.sqrt(
      densitySquared / Math.max(1e-30, uniformDensitySquared),
    );
    const supportIntersectionOverUnion = supportIntersection / Math.max(1, supportUnion);
    if (horizontalSpreadRatios.some((ratio) => ratio < 0.85 || ratio > 1.15)) {
      failures.push("similarity: horizontal mass spread ratio outside [0.85, 1.15]");
    }
    if (densityRelativeL1 > 0.05) {
      failures.push("similarity: final density relative L1 exceeds 0.05");
    }
    if (supportIntersectionOverUnion < 0.85) {
      failures.push("similarity: rho>1e-3 support intersection/union is below 0.85");
    }
    if (centerOfMassYDifference > 0.03) {
      failures.push("similarity: normalized vertical center of mass differs by more than 0.03");
    }
    if (maximumDensityRatio < 0.75 || maximumDensityRatio > 1.25) {
      failures.push("similarity: maximum-density ratio outside [0.75, 1.25]");
    }
    if (kineticEnergyRatio < 0.65 || kineticEnergyRatio > 1.35) {
      failures.push("similarity: kinetic-energy ratio outside [0.65, 1.35]");
    }
    if (liquidSpeedRatio < 0.2 || liquidSpeedRatio > 2) {
      failures.push("similarity: liquid-speed ratio outside [0.2, 2]");
    }
    console.log(JSON.stringify({
      passed: failures.length === 0 && validationErrors.length === 0,
      scenario: "symmetric-expansion",
      sparseResolutionMode,
      uniformResolutionMode,
      regime,
      grids: { sparse: fineDimensions, uniform: uniformDimensions },
      dt_s,
      steps,
      exactTargetTime_s: steps * dt_s,
      uniform,
      sparse,
      finalRatios: {
        densityWeightedKineticEnergy: kineticEnergyRatio,
        maximumLiquidSpeed: liquidSpeedRatio,
        maximumDensity: maximumDensityRatio,
        centerOfMassYAbsoluteDifference: centerOfMassYDifference,
        massStandardDeviation: sparse.checkpoints.at(-1)!.massStandardDeviationNormalized
          .map((value, axis) => value / Math.max(
            1e-30,
            uniform.checkpoints.at(-1)!.massStandardDeviationNormalized[axis],
          )),
        supportExtent: sparse.checkpoints.at(-1)!.supportExtentNormalized
          .map((value, axis) => value / Math.max(
            1e-30,
            uniform.checkpoints.at(-1)!.supportExtentNormalized[axis],
          )),
      },
      finalDensityDifference: {
        relativeL1: densityRelativeL1,
        relativeL2: densityRelativeL2,
        maximumAbsolute: densityMaximumAbsolute,
        supportIntersectionOverUnion1e3: supportIntersectionOverUnion,
        relativeL1AgainstD4SymmetrizedUniform:
          symmetrizedUniformDensityAbsolute / Math.max(1e-30, uniformDensityAbsolute),
      },
      finalFieldHashes: {
        uniformDensity: fieldHash(uniformFinalDensity),
        uniformVelocity: fieldHash(uniformFinalVelocity),
        sparseDensity: fieldHash(sparseFinalDensity),
        sparseVelocity: fieldHash(sparseFinalVelocity),
      },
      similarityThresholds: {
        horizontalMassSpreadRatio: [0.85, 1.15],
        finalDensityRelativeL1: 0.05,
        supportIntersectionOverUnion1e3: 0.85,
        centerOfMassYAbsoluteDifference: 0.03,
        maximumDensityRatio: [0.75, 1.25],
        densityWeightedKineticEnergyRatio: [0.65, 1.35],
        maximumLiquidSpeedRatio: [0.2, 2],
      },
      uniformBaselineSymmetry: uniform.checkpoints.map((checkpoint) => ({
        time_s: checkpoint.time_s,
        ...checkpoint.symmetry,
      })),
      validationErrors,
      failures,
    }, null, 2));
    if (failures.length > 0 || validationErrors.length > 0) process.exitCode = 1;
  } finally {
    device.destroy();
  }
} finally {
  await releaseWebGPUExclusiveLock();
}
