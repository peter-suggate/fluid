/**
 * Adaptive versus forced-all-fine Sparse CM12 physics A/B for the canonical
 * long-tank dam break.
 *
 * This is deliberately separate from the frame-time benchmark: exact per-step
 * density readbacks are QA observations and never enter production scheduling.
 *
 * WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
 *   node --import tsx tools/run-sparse-cm12-long-dam-ab-dawn.ts
 */
import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";
import {
  resolveMethodValues,
  type GPUSolverInstance,
  type MethodParamValues,
  type SimulationMethod,
} from "../lib/core/method-contract";
import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { createSparseCM12LongDamBreakScene } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import {
  adaptiveMassMethod,
  type AdaptiveMassResolutionMode,
} from "../lib/methods/adaptive-mass/method";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

type Dimensions = readonly [number, number, number];
type ArmId = "allFine" | "adaptive";
type Threshold = "trace" | "front" | "liquid";

const argument = (name: string): string | undefined => {
  const prefix = `--${name}=`;
  return process.argv.slice(2).find((value) => value.startsWith(prefix))
    ?.slice(prefix.length);
};
const positiveInteger = (name: string, fallback: number): number => {
  const value = Number(argument(name) ?? fallback);
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new RangeError(`${name} must be a positive integer`);
  }
  return value;
};

const scene = createSparseCM12LongDamBreakScene();
const dimensions = [96, 48, 16] as const;
const dt_s = CM12_PAPER_DT_S;
const steps = positiveInteger("steps", 250);
const checkpointEvery = positiveInteger("checkpoint-every", 25);

interface DensityMetrics {
  readonly mass_cells: number;
  readonly relativeMassDrift: number;
  readonly maximumDensity: number;
  readonly liquidVolume_cells: number;
  readonly frontFineCellX: Readonly<Record<Threshold, number>>;
  readonly centerOfMassNormalized: readonly [number, number, number];
  readonly farWallMass_cells: number;
  readonly farWallWetHeightNormalized: number;
  readonly massByBrickX_cells: readonly number[];
}

function densityMetrics(
  density: Float32Array,
  initialMass: number,
): DensityMetrics {
  const [nx, ny, nz] = dimensions;
  const massByBrickX = new Array<number>(Math.ceil(nx / 8)).fill(0);
  let mass = 0, maximumDensity = 0, liquidVolume = 0, farWallMass = 0;
  let momentX = 0, momentY = 0, momentZ = 0, farWallWetY = -1;
  const front: Record<Threshold, number> = { trace: -1, front: -1, liquid: -1 };
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1)
    for (let x = 0; x < nx; x += 1) {
      const rho = Math.max(0, density[x + nx * (y + ny * z)]!);
      mass += rho;
      massByBrickX[Math.floor(x / 8)]! += rho;
      maximumDensity = Math.max(maximumDensity, rho);
      liquidVolume += rho >= 0.5 ? 1 : 0;
      momentX += rho * (x + 0.5) / nx;
      momentY += rho * (y + 0.5) / ny;
      momentZ += rho * (z + 0.5) / nz;
      if (rho > 1e-3) front.trace = Math.max(front.trace, x);
      if (rho > 0.05) front.front = Math.max(front.front, x);
      if (rho >= 0.5) front.liquid = Math.max(front.liquid, x);
      if (x === nx - 1) {
        farWallMass += rho;
        if (rho > 0.05) farWallWetY = Math.max(farWallWetY, y);
      }
    }
  return {
    mass_cells: mass,
    relativeMassDrift: (mass - initialMass) / Math.max(1, initialMass),
    maximumDensity,
    liquidVolume_cells: liquidVolume,
    frontFineCellX: front,
    centerOfMassNormalized: [momentX / mass, momentY / mass, momentZ / mass],
    farWallMass_cells: farWallMass,
    farWallWetHeightNormalized: farWallWetY < 0 ? 0 : (farWallWetY + 1) / ny,
    massByBrickX_cells: massByBrickX,
  };
}

function densityAgreement(allFine: Float32Array, adaptive: Float32Array) {
  let l1 = 0, l2 = 0, referenceL1 = 0, referenceL2 = 0;
  let maximumAbsolute = 0, supportIntersection = 0, supportUnion = 0;
  for (let index = 0; index < allFine.length; index += 1) {
    const difference = adaptive[index]! - allFine[index]!;
    l1 += Math.abs(difference); l2 += difference * difference;
    referenceL1 += Math.abs(allFine[index]!);
    referenceL2 += allFine[index]! * allFine[index]!;
    maximumAbsolute = Math.max(maximumAbsolute, Math.abs(difference));
    const allFineSupported = allFine[index]! > 1e-3;
    const adaptiveSupported = adaptive[index]! > 1e-3;
    supportIntersection += allFineSupported && adaptiveSupported ? 1 : 0;
    supportUnion += allFineSupported || adaptiveSupported ? 1 : 0;
  }
  return {
    relativeL1: l1 / Math.max(Number.MIN_VALUE, referenceL1),
    relativeL2: Math.sqrt(l2 / Math.max(Number.MIN_VALUE, referenceL2)),
    maximumAbsolute,
    supportIntersectionOverUnion1e3: supportIntersection / Math.max(1, supportUnion),
  };
}

function velocityMetrics(density: Float32Array, velocity: Float32Array) {
  let kineticEnergy = 0, maximumLiquidSpeed = 0;
  const momentum = [0, 0, 0];
  for (let cell = 0; cell < density.length; cell += 1) {
    const rho = Math.max(0, density[cell]!);
    const vx = velocity[4 * cell]!, vy = velocity[4 * cell + 1]!;
    const vz = velocity[4 * cell + 2]!;
    const speedSquared = vx * vx + vy * vy + vz * vz;
    kineticEnergy += 0.5 * rho * speedSquared;
    momentum[0] += rho * vx; momentum[1] += rho * vy; momentum[2] += rho * vz;
    if (rho >= 0.5) maximumLiquidSpeed = Math.max(maximumLiquidSpeed,
      Math.sqrt(speedSquared));
  }
  return {
    densityWeightedKineticEnergy_cells_m2_s2: kineticEnergy,
    densityWeightedMomentum_cells_m_s: momentum,
    maximumLiquidSpeed_m_s: maximumLiquidSpeed,
  };
}

async function createArm(
  device: GPUDevice,
  resolutionMode: AdaptiveMassResolutionMode,
) {
  const method: SimulationMethod = adaptiveMassMethod;
  const overrides: MethodParamValues = { timeStep: "paper", resolutionMode };
  const solver = await method.createSolverAsync!(
    device, createSparseCM12LongDamBreakScene(), "balanced",
    resolveMethodValues(method, "balanced", overrides), undefined, () => {},
  );
  assert.deepEqual([solver.info.nx, solver.info.ny, solver.info.nz], dimensions);
  return solver;
}

async function advanceOne(solver: GPUSolverInstance, time_s: number) {
  while (!solver.advanceTo(time_s, [])) await new Promise((resolve) => setImmediate(resolve));
  return solver.readStats();
}

await acquireWebGPUExclusiveLock(
  "dawn-acceptance",
  "tools/run-sparse-cm12-long-dam-ab-dawn.ts",
);
let allFine: GPUSolverInstance | undefined;
let adaptive: GPUSolverInstance | undefined;
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
  assert.ok(adapter, "Dawn did not expose a WebGPU adapter");
  const device = await adapter.requestDevice({
    requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    event.preventDefault(); validationErrors.push(event.error.message);
  });
  allFine = await createArm(device, "all-fine");
  adaptive = await createArm(device, "adaptive");
  const initialAllFine = (await (allFine as WebGPUAdaptiveMassSolver)
    .readDiagnosticFields()).density;
  const initialAdaptive = (await (adaptive as WebGPUAdaptiveMassSolver)
    .readDiagnosticFields()).density;
  const initialMass = {
    allFine: initialAllFine.reduce((sum, value) => sum + value, 0),
    adaptive: initialAdaptive.reduce((sum, value) => sum + value, 0),
  };
  const arrivals: Record<ArmId, Partial<Record<Threshold, number>>> = {
    allFine: {}, adaptive: {},
  };
  const trajectory: unknown[] = [];
  const checkpoints: unknown[] = [];
  let maximumFrontDifference = 0, maximumCenterOfMassXDifference = 0;
  let maximumSparseMassDrift = 0, maximumUniformMassDrift = 0;
  let maximumSparseDivergence = 0, maximumSparsePressureResidual = 0;
  let maximumUniformDivergence = 0, maximumUniformPressureResidual = 0;
  let maximumPlannedNeighborRatio = 1;
  let fineReceiverFloorViolations = 0;
  let unsupportedEmptyActiveBricks = 0;
  let rejectedCandidateTransfers = 0;
  let unresolvedRejectedCandidateTransfers = 0;
  for (let step = 1; step <= steps; step += 1) {
    const time_s = step * dt_s;
    let allFineStats, adaptiveStats;
    if (step % 2 === 0) {
      adaptiveStats = await advanceOne(adaptive, time_s);
      allFineStats = await advanceOne(allFine, time_s);
    } else {
      allFineStats = await advanceOne(allFine, time_s);
      adaptiveStats = await advanceOne(adaptive, time_s);
    }
    const [allFineFields, adaptiveFields] = await Promise.all([
      (allFine as WebGPUAdaptiveMassSolver).readDiagnosticFields(),
      (adaptive as WebGPUAdaptiveMassSolver).readDiagnosticFields(),
    ]);
    const allFineDensity = allFineFields.density;
    const adaptiveDensity = adaptiveFields.density;
    const density = {
      allFine: densityMetrics(allFineDensity, initialMass.allFine),
      adaptive: densityMetrics(adaptiveDensity, initialMass.adaptive),
    };
    const agreement = densityAgreement(allFineDensity, adaptiveDensity);
    const newlyArrived: Threshold[] = [];
    for (const threshold of ["trace", "front", "liquid"] as const) {
      for (const arm of ["allFine", "adaptive"] as const) {
        if (arrivals[arm][threshold] === undefined
          && density[arm].frontFineCellX[threshold] === dimensions[0] - 1) {
          arrivals[arm][threshold] = time_s; newlyArrived.push(threshold);
        }
      }
    }
    const sparsePressureResidual = adaptiveStats.pressureRelativeResidual
      ?? adaptiveStats.pressureResidual ?? 0;
    const sparseDivergence = adaptiveStats.maxDivergenceAfter_s ?? 0;
    const uniformPressureResidual = allFineStats.pressureRelativeResidual
      ?? allFineStats.pressureResidual ?? 0;
    const uniformDivergence = allFineStats.maxDivergenceAfter_s ?? 0;
    maximumFrontDifference = Math.max(maximumFrontDifference,
      Math.abs(density.allFine.frontFineCellX.front - density.adaptive.frontFineCellX.front));
    maximumCenterOfMassXDifference = Math.max(maximumCenterOfMassXDifference,
      Math.abs(density.allFine.centerOfMassNormalized[0]
        - density.adaptive.centerOfMassNormalized[0]));
    maximumSparseMassDrift = Math.max(maximumSparseMassDrift,
      Math.abs(density.adaptive.relativeMassDrift));
    maximumUniformMassDrift = Math.max(maximumUniformMassDrift,
      Math.abs(density.allFine.relativeMassDrift));
    maximumSparseDivergence = Math.max(maximumSparseDivergence, sparseDivergence);
    maximumSparsePressureResidual = Math.max(maximumSparsePressureResidual,
      sparsePressureResidual);
    maximumUniformDivergence = Math.max(maximumUniformDivergence, uniformDivergence);
    maximumUniformPressureResidual = Math.max(maximumUniformPressureResidual,
      uniformPressureResidual);
    trajectory.push({ step, time_s, density, agreement,
      pressureRelativeResidual: { allFine: uniformPressureResidual,
        adaptive: sparsePressureResidual },
      postProjectionDivergence_s: { allFine: uniformDivergence,
        adaptive: sparseDivergence } });
    const capture = step === 1 || step === steps || step % checkpointEvery === 0
      || newlyArrived.length > 0;
    if (capture) {
      const allFineVelocity = allFineFields.velocity;
      const adaptiveVelocity = adaptiveFields.velocity;
      const activity = await (adaptive as WebGPUAdaptiveMassSolver).readGPUActivityPolicy();
      const activeBricks = activity.bricks.filter((brick) => brick.active);
      const resolutionHistogram = Object.fromEntries(([1, 2, 4, 8] as const).map(
        (resolution) => [resolution, activity.bricks.filter(
          (brick) => brick.resolution === resolution).length],
      ));
      const activeResolutionHistogram = Object.fromEntries(([1, 2, 4, 8] as const).map(
        (resolution) => [resolution, activeBricks.filter(
          (brick) => brick.resolution === resolution).length],
      ));
      const plannedResolutionHistogram = Object.fromEntries(([1, 2, 4, 8] as const).map(
        (resolution) => [resolution, activity.bricks.filter(
          (brick) => brick.plannedResolution === resolution).length],
      ));
      const byKey = new Map(activity.bricks.map((brick) => [brick.key, brick] as const));
      const brickDimensions = dimensions.map((size) => Math.ceil(size / 8));
      let plannedMaximumNeighborRatio = 1;
      let checkpointUnsupportedEmptyActiveBricks = 0;
      for (const brick of activity.bricks) {
        for (let axis = 0; axis < 3; axis += 1) {
          const coordinate = [...brick.coordinate] as [number, number, number];
          coordinate[axis] += 1;
          if (coordinate[axis] >= brickDimensions[axis]!) continue;
          const key = coordinate[0] + brickDimensions[0]!
            * (coordinate[1] + brickDimensions[1]! * coordinate[2]);
          const neighbor = byKey.get(key);
          if (!neighbor) continue;
          plannedMaximumNeighborRatio = Math.max(plannedMaximumNeighborRatio,
            Math.max(brick.plannedResolution, neighbor.plannedResolution)
              / Math.min(brick.plannedResolution, neighbor.plannedResolution));
        }
        if (!brick.active || (brick.reasons & 64) !== 0) continue;
        let supported = false;
        for (let dz = -1; dz <= 1; dz += 1) for (let dy = -1; dy <= 1; dy += 1)
          for (let dx = -1; dx <= 1; dx += 1) {
            if (dx === 0 && dy === 0 && dz === 0) continue;
            const coordinate = [brick.coordinate[0] + dx, brick.coordinate[1] + dy,
              brick.coordinate[2] + dz] as const;
            if (coordinate.some((value, axis) => value < 0
              || value >= brickDimensions[axis]!)) continue;
            const key = coordinate[0] + brickDimensions[0]!
              * (coordinate[1] + brickDimensions[1]! * coordinate[2]);
            const neighbor = byKey.get(key);
            const bit = (1 - dx) + 3 * (1 - dy) + 9 * (1 - dz);
            supported ||= neighbor !== undefined
              && (neighbor.supportMask & (1 << bit)) !== 0;
          }
        if (!supported) checkpointUnsupportedEmptyActiveBricks += 1;
      }
      maximumPlannedNeighborRatio = Math.max(
        maximumPlannedNeighborRatio, plannedMaximumNeighborRatio,
      );
      fineReceiverFloorViolations += activity.bricks.filter((brick) =>
        brick.activatedStep === step && brick.plannedResolution !== 8).length;
      unsupportedEmptyActiveBricks = Math.max(
        unsupportedEmptyActiveBricks, checkpointUnsupportedEmptyActiveBricks,
      );
      // Candidate status is retained for inspection after an epoch. Only a
      // scheduled record belongs to the live shadow transaction; otherwise a
      // dormant slot's old status would be reported as a current rejection.
      const scheduledTransfers = activity.bricks.filter((brick) =>
        brick.topologyPreparationScheduled);
      const pendingTransfers = scheduledTransfers.filter((brick) =>
        brick.candidateStatus === 1);
      const rejectedTransfers = scheduledTransfers.filter((brick) =>
        brick.candidateStatus === 2 || (brick.candidateStatus === 1
          && (brick.transferStatus !== 1 || brick.faceTransferStatus !== 1)));
      rejectedCandidateTransfers = Math.max(
        rejectedCandidateTransfers, rejectedTransfers.length,
      );
      unresolvedRejectedCandidateTransfers = rejectedTransfers.length;
      const activeMaximumFineCellX = Math.max(...activeBricks.map(
        (brick) => 8 * (brick.coordinate[0] + 1) - 1));
      checkpoints.push({
        step, time_s, density, agreement,
        velocity: {
          allFine: velocityMetrics(allFineDensity, allFineVelocity),
          adaptive: velocityMetrics(adaptiveDensity, adaptiveVelocity),
        },
        pressureRelativeResidual: { allFine: uniformPressureResidual,
          adaptive: sparsePressureResidual },
        postProjectionDivergence_s: { allFine: uniformDivergence,
          adaptive: sparseDivergence },
        sparseResidency: {
          activeBricks: activeBricks.length,
          activeCells: adaptiveStats.activeSampleCount,
          activeMaximumFineCellX,
          receiverLeadFineCells: activeMaximumFineCellX
            - density.adaptive.frontFineCellX.front,
          generation: adaptiveStats.fluidBrickGeneration,
          resolutionHistogram,
          activeResolutionHistogram,
          plannedResolutionHistogram,
          plannedMaximumNeighborRatio,
          unsupportedEmptyActiveBricks: checkpointUnsupportedEmptyActiveBricks,
          candidateTransfers: {
            pending: pendingTransfers.length,
            passed: pendingTransfers.filter((brick) => brick.transferStatus === 1).length,
            rejected: rejectedTransfers.length,
            maximumAbsoluteMassErrorFineCells: Math.max(0, ...pendingTransfers.map(
              (brick) => Math.abs(brick.transferMassErrorFineCells))),
            maximumAbsoluteGammaErrorFineCells: Math.max(0, ...pendingTransfers.map(
              (brick) => Math.abs(brick.transferGammaErrorFineCells))),
            maximumAbsoluteMomentumErrorFineCells: Math.max(0, ...pendingTransfers.flatMap(
              (brick) => brick.transferMomentumErrorFineCells.map(Math.abs))),
            maximumAbsoluteExteriorFluxErrorFineAreas: Math.max(0, ...pendingTransfers.map(
              (brick) => brick.maximumAbsoluteTransferFluxErrorFineAreas)),
          },
          topBrickRow: activity.bricks.filter((brick) =>
            brick.coordinate[1] === brickDimensions[1]! - 1 && brick.active).map((brick) => ({
            coordinate: brick.coordinate,
            resolution: brick.resolution,
            plannedResolution: brick.plannedResolution,
            acceptedResolution: brick.acceptedResolution,
            candidateResolution: brick.candidateResolution,
            candidateStatus: brick.candidateStatus,
            transferStatus: brick.transferStatus,
            faceTransferStatus: brick.faceTransferStatus,
            transferMassErrorFineCells: brick.transferMassErrorFineCells,
            scoreByte: brick.scoreByte,
            reasons: brick.reasons,
            supportMask: brick.supportMask,
            quietEpochs: brick.quietEpochs,
          })),
        },
      });
    }
  }
  const final = trajectory.at(-1) as {
    density: Record<ArmId, DensityMetrics>;
    agreement: ReturnType<typeof densityAgreement>;
  };
  const expectsFarWallArrival = steps * dt_s >= 1;
  const failures: string[] = [];
  if (maximumUniformMassDrift > 1e-3) failures.push("all-fine mass drift exceeds 1e-3");
  if (maximumSparseMassDrift > 1e-3) failures.push("adaptive mass drift exceeds 1e-3");
  // The production projection performs a fixed 128 f32 PCG iterations. Keep
  // the absolute gate comfortably below 1e-7 while allowing the f32
  // reduction-order floor to differ from the uniform topology.
  if (maximumSparsePressureResidual > Math.max(5e-8, 2 * maximumUniformPressureResidual)) {
    failures.push("adaptive pressure residual exceeds the absolute/all-fine-relative gate");
  }
  if (maximumSparseDivergence > Math.max(1e-5, 2 * maximumUniformDivergence)) {
    failures.push("adaptive divergence exceeds the absolute/all-fine-relative gate");
  }
  if (expectsFarWallArrival && arrivals.adaptive.liquid === undefined) {
    failures.push("adaptive rho >= 0.5 liquid front did not reach the far wall");
  }
  if (arrivals.allFine.liquid !== undefined && arrivals.adaptive.liquid !== undefined
    && Math.abs(arrivals.adaptive.liquid - arrivals.allFine.liquid) > 0.12) {
    failures.push("adaptive/all-fine liquid far-wall arrival differs by more than 0.12 s");
  }
  if (maximumCenterOfMassXDifference > 0.08) {
    failures.push("adaptive/all-fine normalized x centroid differs by more than 0.08");
  }
  if (maximumPlannedNeighborRatio > 2) {
    failures.push("adaptive GPU candidate plan exceeds the maximum 2:1 face ratio");
  }
  if (fineReceiverFloorViolations > 0) {
    failures.push("new adaptive receiver was not planned at the finest 8^3 level");
  }
  if (unsupportedEmptyActiveBricks > 0) {
    failures.push("adaptive residency retained empty bricks outside the air-support ring");
  }
  if (unresolvedRejectedCandidateTransfers > 0) {
    failures.push("adaptive GPU candidate transfer rejection did not recover");
  }
  if ([allFine, adaptive].some((solver) => solver.info.hostFluidAuthority !== "gpu-resident"
    || solver.info.hostSimulationSizedWorkItems !== 0
    || solver.info.hostSchedulingUsesReadback !== false)) {
    failures.push("Sparse CM12 production frame authority is not fully GPU-resident");
  }
  for (const error of validationErrors) failures.push(`WebGPU validation error: ${error}`);
  const report = {
    passed: failures.length === 0,
    scenario: scene.sceneId,
    arms: { reference: "adaptive-mass/all-fine", candidate: "adaptive-mass/adaptive" },
    backend: process.env.FLUID_WEBGPU_BACKEND ?? "metal",
    adapter: (adapter as GPUAdapter & { readonly info?: GPUAdapterInfo }).info,
    dimensions,
    dt_s,
    steps,
    simulatedTime_s: steps * dt_s,
    expectsFarWallArrival,
    initialMass_cells: initialMass,
    farWallArrival_s: {
      allFine: arrivals.allFine,
      adaptive: arrivals.adaptive,
      difference: Object.fromEntries((["trace", "front", "liquid"] as const)
        .map((threshold) => [threshold,
          arrivals.adaptive[threshold] !== undefined && arrivals.allFine[threshold] !== undefined
            ? arrivals.adaptive[threshold]! - arrivals.allFine[threshold]! : null])),
    },
    extrema: {
      maximumAbsoluteFrontDifferenceFineCells: maximumFrontDifference,
      maximumNormalizedCenterOfMassXDifference: maximumCenterOfMassXDifference,
      maximumAbsoluteMassDrift: {
        allFine: maximumUniformMassDrift,
        adaptive: maximumSparseMassDrift,
      },
      maximumPressureRelativeResidual: {
        allFine: maximumUniformPressureResidual,
        adaptive: maximumSparsePressureResidual,
      },
      maximumPostProjectionDivergence_s: {
        allFine: maximumUniformDivergence,
        adaptive: maximumSparseDivergence,
      },
      maximumPlannedNeighborRatio,
      fineReceiverFloorViolations,
      unsupportedEmptyActiveBricks,
      rejectedCandidateTransfers,
      unresolvedRejectedCandidateTransfers,
    },
    final: {
      density: final.density,
      agreement: final.agreement,
    },
    checkpoints,
    trajectory,
    validationErrors,
    failures,
  };
  const output = argument("summary") === "1" ? {
    passed: report.passed,
    simulatedTime_s: report.simulatedTime_s,
    farWallArrival_s: report.farWallArrival_s,
    extrema: report.extrema,
    final: report.final,
    finalResidency: checkpoints.at(-1),
    validationErrors: report.validationErrors,
    failures: report.failures,
  } : report;
  console.log(JSON.stringify(output, null, 2));
  if (failures.length > 0) process.exitCode = 1;
} finally {
  allFine?.destroy(); adaptive?.destroy();
  await releaseWebGPUExclusiveLock();
}
