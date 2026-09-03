import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { resolveMethodValues, type GPUSolverInstance, type SimulationMethod } from
  "../lib/core/method-contract";
import { createOceanSeicheScene, getScenePreset } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { probeOceanWavePropagation } from "../lib/harness/ocean-wave-propagation-probe";
import { readFloatTexture3D, readRgbaTexture3D } from
  "../lib/harness/webgpu-smoke-readbacks";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import { uniformMethod } from "../lib/methods/uniform/method";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
assert.ok(dawnModule, "set WEBGPU_NODE_MODULE to the Dawn webgpu module");
const stepCount = Number.parseInt(process.env.OCEAN_AB_STEPS ?? "60", 10);
const checkpointStride = Number.parseInt(process.env.OCEAN_AB_STRIDE ?? "10", 10);
const stepDt_s = Number(process.env.OCEAN_AB_DT ?? CM12_PAPER_DT_S);
const timeStepMode = process.env.OCEAN_AB_TIME_STEP === "scene" ? "scene" : "paper";
const sparseOverrides = JSON.parse(process.env.OCEAN_AB_SPARSE_OVERRIDES ?? "{}") as
  Record<string, unknown>;
const minimumCellSize = Number.parseInt(process.env.OCEAN_AB_MINIMUM_CELL_SIZE ?? "0", 10);
const outputPath = process.env.OCEAN_AB_OUT;
assert.ok(Number.isSafeInteger(minimumCellSize) && minimumCellSize >= 0,
  "OCEAN_AB_MINIMUM_CELL_SIZE must be a non-negative integer");

await acquireWebGPUExclusiveLock("dawn-probe", "probe-ocean-seiche-uniform-ab-dawn");
let device: GPUDevice | undefined;
try {
  const dawn = await import(pathToFileURL(dawnModule).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter);
  device = await adapter.requestDevice({
    requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    event.preventDefault();
    validationErrors.push(event.error.message);
  });

  const runArm = async (
    method: SimulationMethod,
    sparseOverrides: Record<string, unknown> = {},
  ) => {
    const scene = process.env.OCEAN_AB_DIRECT_SCENE === "1"
      ? createOceanSeicheScene() : getScenePreset("ocean-seiche").create();
    if (process.env.OCEAN_AB_HALF === "1") {
      scene.container = { ...scene.container, width_m: 4, height_m: 1.2,
        depth_m: 1, fillFraction: 2 / 3 };
      const brick = 8 * scene.voxelDomain.finestCellSize_m;
      scene.fluid.initialBrickSeeds_m = Array.from({ length: 5 }, (_, zTier) => {
        const z = -0.5 + (zTier + 0.5) * brick;
        return [
          { x: -2 + 0.5 * brick, y: 4.5 * brick, z },
          { x: -2 + 1.5 * brick, y: 4.5 * brick, z },
        ];
      }).flat();
      if (process.env.OCEAN_AB_FIXED === "1") {
        scene.fluid.refinementRegions = [{
          id: "diagnostic-fixed-lattice", rule: "minimum-cell-size",
          minimumCellSize_cells: 1, maximumCellSize_cells: 1,
          min_m: { x: -2, y: 0, z: -0.5 }, max_m: { x: 2, y: 1.2, z: 0.5 },
        }];
      }
    }
    if (minimumCellSize > 0) {
      scene.fluid.refinementRegions = [{
        id: `diagnostic-full-domain-min-${minimumCellSize}`,
        rule: "minimum-cell-size",
        minimumCellSize_cells: minimumCellSize,
        min_m: {
          x: -0.5 * scene.container.width_m,
          y: 0,
          z: -0.5 * scene.container.depth_m,
        },
        max_m: {
          x: 0.5 * scene.container.width_m,
          y: scene.container.height_m,
          z: 0.5 * scene.container.depth_m,
        },
      }];
    }
    const values = resolveMethodValues(method, "balanced", {
      timeStep: timeStepMode,
      ...(method.id === "uniform" ? { densityPostProcessing: "off" } : {}),
      ...(method.id === "adaptive-mass" ? sparseOverrides : {}),
    });
    const solver = await method.createSolverAsync!(
      device!, scene, "balanced", values, undefined, () => {},
    ) as GPUSolverInstance & {
      waitForSimulationReady?: () => Promise<void>;
      readDiagnosticFields?: (
        includeWorldLeaves?: boolean,
        frameBank?: "accepted" | "candidate",
      ) => Promise<{ density: Float32Array; velocity: Float32Array;
        pressureRhs?: Float32Array; pressure?: Float32Array;
        divergence?: Float32Array }>;
      readGPUActivityPolicy?: () => Promise<{
        acceptedTopologyGeneration?: number;
        bricks: Array<{ active: boolean; acceptedResolution: number;
          coordinate?: readonly [number, number, number]; reasons?: number;
          supportMask?: number; candidateStatus?: number; candidateResolution?: number;
          sweptSupportMask?: number; plannedResolution?: number; planReasons?: number;
          candidateEpoch?: number; transferStatus?: number; faceTransferStatus?: number;
          transferMassErrorFineCells?: number;
          maximumAbsoluteTransferFluxErrorFineAreas?: number;
          topologyPreparationScheduled?: boolean; topologyPreparationEpoch?: number;
          topologyPage?: number; authoredResolutionFloor?: boolean }>;
      }>;
      readStats?: () => Promise<Record<string, unknown>>;
      readAdaptiveRepresentationQA?: () => Promise<Record<string, unknown>>;
    };
    try {
      await solver.waitForSimulationReady?.();
      const grid = [solver.info.nx, solver.info.ny, solver.info.nz] as const;
      type DiagnosticFields = { density: Float32Array; velocity: Float32Array;
        pressureRhs?: Float32Array; pressure?: Float32Array;
        divergence?: Float32Array };
      const readFields = async (): Promise<DiagnosticFields> =>
        solver.readDiagnosticFields
          ? solver.readDiagnosticFields()
          : Promise.all([
            readFloatTexture3D(
              device!, solver.volumeTexture, grid[0], grid[1], grid[2],
            ),
            readRgbaTexture3D(
              device!, solver.velocityTexture!, grid[0], grid[1], grid[2],
            ),
          ]).then(([density, velocity]) => ({ density, velocity }));
      const checkpoints: { time_s: number; field: Float32Array;
        velocity: Float32Array; pressureRhs?: Float32Array;
        pressure?: Float32Array; divergence?: Float32Array }[] = [];
      const telemetry: unknown[] = [];
      const initialField = solver.readDiagnosticFields
        ? await solver.readDiagnosticFields().then((receipt) => receipt.density)
        : await readFloatTexture3D(device!, solver.volumeTexture, grid[0], grid[1], grid[2]);
      const initialActivity = await solver.readGPUActivityPolicy?.();
      if (initialActivity) telemetry.push({
        time_s: 0,
        generation: initialActivity.acceptedTopologyGeneration,
        resolutionHistogram: Object.fromEntries([1, 2, 4, 8].map((resolution) => [
          resolution,
          initialActivity.bricks.filter((brick) => brick.active
            && brick.acceptedResolution === resolution).length,
        ])),
        launchEdge: initialActivity.bricks.filter((brick) =>
          brick.coordinate?.[2] === 5 && ((brick.coordinate?.[1] === 9
            && brick.coordinate[0] <= 3) || (brick.coordinate?.[1] === 8
              && brick.coordinate[0] === 10))).map((brick) => ({
              coordinate: brick.coordinate, active: brick.active, reasons: brick.reasons,
              supportMask: brick.supportMask, candidateStatus: brick.candidateStatus,
              candidateResolution: brick.candidateResolution,
              sweptSupportMask: brick.sweptSupportMask,
              plannedResolution: brick.plannedResolution, planReasons: brick.planReasons,
              candidateEpoch: brick.candidateEpoch,
              transferStatus: brick.transferStatus,
              faceTransferStatus: brick.faceTransferStatus,
              transferMassErrorFineCells: brick.transferMassErrorFineCells,
              maximumAbsoluteTransferFluxErrorFineAreas:
                brick.maximumAbsoluteTransferFluxErrorFineAreas,
              topologyPreparationScheduled: brick.topologyPreparationScheduled,
              topologyPreparationEpoch: brick.topologyPreparationEpoch,
              topologyPage: brick.topologyPage,
              authoredResolutionFloor: brick.authoredResolutionFloor,
            })),
      });
      for (let step = 1; step <= stepCount; step += 1) {
        const time_s = step * stepDt_s;
        assert.equal(solver.advanceTo(time_s, []), true,
          `${method.id} rejected step ${step}`);
        if (step % checkpointStride === 0) {
          await device!.queue.onSubmittedWorkDone();
          const [fields, activity, stats, representationQA] = await Promise.all([
            readFields(),
            solver.readGPUActivityPolicy?.(),
            solver.readStats?.().then((value) => value as unknown as
              Record<string, unknown>),
            process.env.OCEAN_AB_TOPOLOGY_QA === "1"
              ? solver.readAdaptiveRepresentationQA?.() : undefined,
          ]);
          checkpoints.push({
            time_s,
            field: fields.density,
            velocity: fields.velocity,
            pressureRhs: fields.pressureRhs,
            pressure: fields.pressure,
            divergence: fields.divergence,
          });
          if (activity) telemetry.push({
            time_s,
            generation: activity.acceptedTopologyGeneration,
            resolutionHistogram: Object.fromEntries([1, 2, 4, 8].map((resolution) => [
              resolution,
              activity.bricks.filter((brick) => brick.active
                && brick.acceptedResolution === resolution).length,
            ])),
            launchEdge: activity.bricks.filter((brick) =>
              brick.coordinate?.[2] === 5 && ((brick.coordinate?.[1] === 9
                && brick.coordinate[0] <= 3) || (brick.coordinate?.[1] === 8
                  && brick.coordinate[0] === 10))).map((brick) => ({
                  coordinate: brick.coordinate, active: brick.active, reasons: brick.reasons,
                  supportMask: brick.supportMask, candidateStatus: brick.candidateStatus,
                  candidateResolution: brick.candidateResolution,
                  sweptSupportMask: brick.sweptSupportMask,
                  plannedResolution: brick.plannedResolution, planReasons: brick.planReasons,
                  candidateEpoch: brick.candidateEpoch,
                  transferStatus: brick.transferStatus,
                  faceTransferStatus: brick.faceTransferStatus,
                  transferMassErrorFineCells: brick.transferMassErrorFineCells,
                  maximumAbsoluteTransferFluxErrorFineAreas:
                    brick.maximumAbsoluteTransferFluxErrorFineAreas,
                  topologyPreparationScheduled: brick.topologyPreparationScheduled,
                  topologyPreparationEpoch: brick.topologyPreparationEpoch,
                  topologyPage: brick.topologyPage,
                  authoredResolutionFloor: brick.authoredResolutionFloor,
                })),
            pressureRelativeResidual: stats?.pressureRelativeResidual,
            pressureTrueResidualMaximum: stats?.pressureTrueResidualMaximum,
            pressureInitialTrueRelativeResidual: stats?.pressureInitialTrueRelativeResidual,
            pressureIterationsExecuted: stats?.pressureIterationsExecuted,
            pressureSolveConverged: stats?.pressureSolveConverged,
            pressureIterationCapReached: stats?.pressureIterationCapReached,
            pressureConvergenceReason: stats?.pressureConvergenceReason,
            pressureCurvatureBreakdown: stats?.pressureCurvatureBreakdown,
            pressureCurvatureRecoveryCount: stats?.pressureCurvatureRecoveryCount,
            pressureConvergenceFault: stats?.pressureConvergenceFault,
            maximumPostProjectionDivergence_s: stats?.maximumPostProjectionDivergence_s,
            adaptivePressureCellCount: stats?.adaptivePressureCellCount,
            adaptivePressureActiveRowCount: stats?.adaptivePressureActiveRowCount,
            adaptiveAcceptedCellCount: stats?.adaptiveAcceptedCellCount,
            adaptiveAcceptedRowCount: stats?.adaptiveAcceptedRowCount,
            representationQA,
          });
        }
      }
      const result = checkpoints.length >= 3 ? probeOceanWavePropagation(scene, {
        method: method.id,
        grid,
        checkpoints,
      }, {
        stationCount: 12,
        minimumCheckpointCount: 6,
        minimumFarHalfDisturbance_cells: 0.5,
      }) : undefined;
      return { scene, grid, initialField, checkpoints, result, telemetry };
    } finally {
      solver.destroy();
    }
  };

  const uniform = await runArm(uniformMethod);
  const sparse = await runArm(adaptiveMassMethod, sparseOverrides);
  const heightProfiles = (
    field: Float32Array,
    [nx, ny, nz]: readonly [number, number, number],
  ) => {
    return Array.from({ length: nx }, (_, x) => {
      let height = 0;
      for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
        height += field[x + nx * (y + ny * z)]! / nz;
      }
      return height;
    });
  };
  const compare = (candidateArm: typeof sparse) => uniform.checkpoints.map(
    (checkpoint, index) => {
      const reference = heightProfiles(checkpoint.field, uniform.grid);
      const candidate = heightProfiles(
        candidateArm.checkpoints[index]!.field, candidateArm.grid,
      );
      const differences = reference.map((value, x) => candidate[x]! - value);
      const referenceSignal = reference.map((value) => value - 72);
      const candidateSignal = candidate.map((value) => value - 72);
      const norm = (values: number[]) => Math.sqrt(
        values.reduce((sum, value) => sum + value * value, 0),
      );
      const referenceNorm = norm(referenceSignal);
      const candidateNorm = norm(candidateSignal);
      const correlationAtLag = (lag: number) => {
        let dot = 0, referenceEnergy = 0, candidateEnergy = 0;
        for (let x = Math.max(0, -lag); x < Math.min(reference.length,
          reference.length - lag); x += 1) {
          const a = referenceSignal[x]!;
          const b = candidateSignal[x + lag]!;
          dot += a * b;
          referenceEnergy += a * a;
          candidateEnergy += b * b;
        }
        return dot / Math.sqrt(referenceEnergy * candidateEnergy);
      };
      const lags = Array.from({ length: 161 }, (_, lag) => lag - 80);
      const bestLag = lags.reduce((best, lag) =>
        correlationAtLag(lag) > correlationAtLag(best) ? lag : best, 0);
      return {
        time_s: checkpoint.time_s,
        meanAbsoluteHeightError_cells:
          differences.reduce((sum, value) => sum + Math.abs(value), 0) / differences.length,
        maximumAbsoluteHeightError_cells:
          Math.max(...differences.map((value) => Math.abs(value))),
        signedVolumeHeightError_cells:
          differences.reduce((sum, value) => sum + value, 0) / differences.length,
        zeroLagCorrelation: correlationAtLag(0),
        amplitudeRatio: candidateNorm / referenceNorm,
        bestLag_cells: bestLag,
        bestLagCorrelation: correlationAtLag(bestLag),
      };
    },
  );
  const compareFields = (referenceField: Float32Array, candidateField: Float32Array) => {
    const reference = heightProfiles(referenceField, uniform.grid);
    const candidate = heightProfiles(candidateField, sparse.grid);
    const differences = reference.map((value, x) => candidate[x]! - value);
    return {
      meanAbsoluteHeightError_cells:
        differences.reduce((sum, value) => sum + Math.abs(value), 0) / differences.length,
      maximumAbsoluteHeightError_cells:
        Math.max(...differences.map((value) => Math.abs(value))),
      signedVolumeHeightError_cells:
        differences.reduce((sum, value) => sum + value, 0) / differences.length,
    };
  };
  const compareVelocity = (
    referenceVelocity: Float32Array,
    candidateVelocity: Float32Array,
    density: Float32Array,
  ) => {
    const [nx, ny, nz] = uniform.grid;
    const collocatedUniform = (x: number, y: number, z: number, axis: number) => {
      const at = x + nx * (y + ny * z);
      const q = [x, y, z]; q[axis] -= 1;
      const negative = q[axis]! < 0 ? 0 : referenceVelocity[4 *
        (q[0]! + nx * (q[1]! + ny * q[2]!)) + axis]!;
      return 0.5 * (referenceVelocity[4 * at + axis]! + negative);
    };
    return [0, 1, 2].map((axis) => {
      let squaredError = 0, squaredReference = 0, dot = 0;
      let squaredCandidate = 0, maximumError = 0, count = 0;
      let referenceSum = 0, candidateSum = 0;
      let candidateMaximum = 0;
      let candidateMaximumAt = [0, 0, 0];
      for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
        for (let x = 0; x < nx; x += 1) {
          const at = x + nx * (y + ny * z);
          if (density[at]! < 0.05) continue;
          const reference = collocatedUniform(x, y, z, axis);
          const candidate = candidateVelocity[4 * at + axis]!;
          const error = candidate - reference;
          squaredError += error * error;
          squaredReference += reference * reference;
          squaredCandidate += candidate * candidate;
          dot += reference * candidate;
          referenceSum += reference; candidateSum += candidate;
          maximumError = Math.max(maximumError, Math.abs(error));
          if (Math.abs(candidate) > candidateMaximum) {
            candidateMaximum = Math.abs(candidate);
            candidateMaximumAt = [x, y, z];
          }
          count += 1;
        }
      }
      return {
        axis: "xyz"[axis], count,
        rmsError_m_s: Math.sqrt(squaredError / Math.max(1, count)),
        relativeL2Error: Math.sqrt(squaredError / Math.max(squaredReference, 1e-30)),
        maximumError_m_s: maximumError,
        referenceRms_m_s: Math.sqrt(squaredReference / Math.max(1, count)),
        candidateRms_m_s: Math.sqrt(squaredCandidate / Math.max(1, count)),
        referenceMean_m_s: referenceSum / Math.max(1, count),
        candidateMean_m_s: candidateSum / Math.max(1, count),
        candidateMaximum_m_s: candidateMaximum,
        candidateMaximumAt,
        correlation: dot / Math.sqrt(squaredReference * squaredCandidate),
      };
    });
  };
  const scalarSummary = (values?: Float32Array) => values ? ({
    minimum: values.reduce((result, value) => Math.min(result, value), Infinity),
    maximum: values.reduce((result, value) => Math.max(result, value), -Infinity),
    rms: Math.sqrt(values.reduce((sum, value) => sum + value * value, 0)
      / values.length),
    nonzero: values.reduce((count, value) => count + Number(value !== 0), 0),
  }) : undefined;
  const report = {
    configuration: {
      stepCount, checkpointStride, stepDt_s, timeStepMode, minimumCellSize,
      sparseOverrides,
    },
    uniform: uniform.result,
    sparse: sparse.result,
    sparseTelemetry: sparse.telemetry,
    initialComparison: compareFields(uniform.initialField, sparse.initialField),
    comparisons: compare(sparse),
    velocityComparisons: uniform.checkpoints.map((checkpoint, index) => ({
      time_s: checkpoint.time_s,
      components: compareVelocity(checkpoint.velocity,
        sparse.checkpoints[index]!.velocity, checkpoint.field),
    })),
    sparsePressureFields: sparse.checkpoints.map((checkpoint) => ({
      time_s: checkpoint.time_s,
      rhs: scalarSummary(checkpoint.pressureRhs),
      pressure: scalarSummary(checkpoint.pressure),
      divergence: scalarSummary(checkpoint.divergence),
      launchCornerZ: Array.from({ length: sparse.grid[2] }, (_, z) => {
        const at = 15 + sparse.grid[0] * (79 + sparse.grid[1] * z);
        return {
          z,
          density: checkpoint.field[at],
          velocity: Array.from(checkpoint.velocity.subarray(4 * at, 4 * at + 3)),
          pressure: checkpoint.pressure?.[at],
          rhs: checkpoint.pressureRhs?.[at],
        };
      }),
    })),
    validationErrors,
  };
  const serialized = JSON.stringify(report, null, 2);
  if (outputPath) await writeFile(outputPath, `${serialized}\n`);
  console.log(serialized);
} finally {
  device?.destroy();
  await releaseWebGPUExclusiveLock();
}
