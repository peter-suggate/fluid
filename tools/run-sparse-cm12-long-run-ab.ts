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
 * UI-sized timestep stress:
 *   WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
 *     node --import tsx tools/run-sparse-cm12-long-run-ab.ts --seconds=2 --dt=0.033
 */
import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";
import type { MethodParamValues, SimulationMethod } from
  "../lib/core/method-contract";
import { resolveMethodValues } from "../lib/core/method-contract";
import { createSymmetricExpansionScene } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import type { AdaptiveMassStepTelemetry } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { uniformMethod } from "../lib/methods/uniform/method";

type Dimensions = readonly [number, number, number];

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
}

interface ArmReceipt {
  readonly method: string;
  readonly checkpoints: readonly FieldReceipt[];
  readonly evolution: StepReceipt;
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
          const positive = [x, y, z] as [number, number, number];
          positive[component] += 1;
          value = 0.5 * (value + (positive[component] < dimensions[component]
            ? velocity[4 * cell(...positive) + component] : 0));
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
    velocitySymmetry = Math.max(velocitySymmetry, Math.abs(
      collocated[3 * target + targetComponent] - sign * collocated[3 * source + component],
    ));
  };
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
    for (let x = 0; x < nx; x += 1) {
      const index = cell(x, y, z);
      const rho = density[index];
      const speed = Math.hypot(
        collocated[3 * index], collocated[3 * index + 1], collocated[3 * index + 2],
      );
      mass += rho;
      for (const [axis, coordinate] of [x, y, z].entries()) {
        const centered = coordinate + 0.5;
        firstMoment[axis] += rho * centered;
        secondMoment[axis] += rho * centered * centered;
        if (rho > 1e-3) {
          supportMinimum[axis] = Math.min(supportMinimum[axis], coordinate);
          supportMaximum[axis] = Math.max(supportMaximum[axis], coordinate);
        }
      }
      kinetic += 0.5 * rho * speed * speed;
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

async function runArm(
  device: GPUDevice,
  method: SimulationMethod,
  dimensions: Dimensions,
  dt_s: number,
  steps: number,
): Promise<ArmReceipt> {
  const scene = createSymmetricExpansionScene();
  scene.duration_s = steps * dt_s;
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = dt_s;
  const overrides: MethodParamValues = method.id === "uniform"
    ? { timeStep: "scene", densityPostProcessing: "off" } : {};
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
  };
  try {
    assert.deepEqual([solver.info.nx, solver.info.ny, solver.info.nz], dimensions);
    let initialMass = 0;
    const capture = async (step: number) => {
      const density = await readTexture(device, solver.volumeTexture, dimensions, 1);
      const velocity = await readTexture(device, solver.velocityTexture!, dimensions, 4);
      if (step === 0) for (const value of density) initialMass += value;
      checkpoints.push(fieldReceipt(
        step,
        dt_s,
        density,
        velocity,
        dimensions,
        initialMass,
        method.id === "uniform" ? "negative-mac" : "collocated",
      ));
    };
    await capture(0);
    const checkpointSteps = new Set([Math.floor(steps / 2), steps]);
    for (let step = 1; step <= steps; step += 1) {
      while (!solver.advanceTo(step * dt_s, [])) await new Promise(setImmediate);
      const info = await solver.readStats();
      const sparse = info as typeof info & AdaptiveMassStepTelemetry;
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
      if (checkpointSteps.has(step)) await capture(step);
    }
    if (!Number.isFinite(evolution.minimumEvolvedMixedSeamRows)) {
      evolution.minimumEvolvedMixedSeamRows = 0;
    }
    return { method: method.id, checkpoints, evolution };
  } finally {
    solver.destroy();
  }
}

const dt_s = positiveNumber("dt", 0.004);
const target_s = positiveNumber("seconds", 2);
const steps = Math.ceil(target_s / dt_s);
const dimensions = [32, 16, 32] as const;
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
    const uniform = await runArm(device, uniformMethod, dimensions, dt_s, steps);
    const sparse = await runArm(device, adaptiveMassMethod, dimensions, dt_s, steps);
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
    if (sparse.evolution.maximumMixedSeamRows === 0
      || sparse.evolution.maximumFineCoarseConnectedPairs === 0) failures.push("sparse: mixed topology absent");
    if (sparse.evolution.minimumEvolvedMixedSeamRows === 0) {
      failures.push("sparse: connected mixed seam did not persist after activation");
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
    if (horizontalSpreadRatios.some((ratio) => ratio < 0.8 || ratio > 1.2)) {
      failures.push("similarity: horizontal mass spread ratio outside [0.8, 1.2]");
    }
    if (supportExtentRatios.some((ratio) => Math.abs(ratio - 1) > 1e-12)) {
      failures.push("similarity: horizontal support extent differs from Uniform");
    }
    if (centerOfMassYDifference > 0.03) {
      failures.push("similarity: normalized vertical center of mass differs by more than 0.03");
    }
    if (maximumDensityRatio < 0.75 || maximumDensityRatio > 1.25) {
      failures.push("similarity: maximum-density ratio outside [0.75, 1.25]");
    }
    if (kineticEnergyRatio < 0.1 || kineticEnergyRatio > 2) {
      failures.push("similarity: kinetic-energy ratio outside [0.1, 2]");
    }
    if (liquidSpeedRatio < 0.2 || liquidSpeedRatio > 2) {
      failures.push("similarity: liquid-speed ratio outside [0.2, 2]");
    }
    console.log(JSON.stringify({
      passed: failures.length === 0 && validationErrors.length === 0,
      scenario: "symmetric-expansion",
      grid: dimensions,
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
      similarityThresholds: {
        horizontalMassSpreadRatio: [0.8, 1.2],
        horizontalSupportExtentRatio: 1,
        centerOfMassYAbsoluteDifference: 0.03,
        maximumDensityRatio: [0.75, 1.25],
        densityWeightedKineticEnergyRatio: [0.1, 2],
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
