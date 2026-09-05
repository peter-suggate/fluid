import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { unpackFineLevelSetPackedPhi } from
  "../lib/core/fine-levelset-packed-sample";
import { resolveMethodValues } from "../lib/core/method-contract";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import { WebGPUAdaptiveMassSolver, type AdaptiveMassStepTelemetry } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { SPARSE_CM12_ACTIVITY_POLICY } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

function densityMass(density: Float32Array): number {
  let mass = 0;
  for (const value of density) mass += Math.max(0, value);
  return mass;
}

async function readGPUWords(device: GPUDevice,
  source: GPUBuffer | GPUBufferBinding, count: number): Promise<Uint32Array> {
  const binding = "buffer" in source ? source : { buffer: source };
  const readback = device.createBuffer({ size: Math.max(4, 4 * count),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(binding.buffer, binding.offset ?? 0,
      readback, 0, 4 * count);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    return new Uint32Array(readback.getMappedRange()).slice();
  } finally {
    if (readback.mapState === "mapped") readback.unmap();
    readback.destroy();
  }
}

const signedPageCoordinate = (key: number): readonly [number, number, number] => [
  (key & 0x7ff) - 1024,
  ((key >>> 11) & 0x3ff) - 512,
  (key >>> 21) - 1024,
];

async function negativePresentationSamplesByCoordinate(device: GPUDevice,
  solver: WebGPUAdaptiveMassSolver): Promise<Map<string, number>> {
  const source = solver.globalFineLevelSetSource;
  const capacity = source.plan.maximumResidentBricks;
  const samplesPerPage = source.plan.samplesPerBrick;
  const [worklist, metadata, samples] = await Promise.all([
    readGPUWords(device, source.worklist, source.worklist.size / 4),
    readGPUWords(device, source.metadata, 4 * capacity),
    readGPUWords(device, source.samples, capacity * samplesPerPage),
  ]);
  const result = new Map<string, number>();
  const generation = worklist[0]!;
  const count = Math.min(worklist[1]!, capacity);
  for (let rank = 0; rank < count; rank += 1) {
    const page = worklist[7 + rank]!;
    const at = 4 * page;
    if (metadata[at] !== page || metadata[at + 2] !== generation) continue;
    let negative = 0;
    for (let local = 0; local < samplesPerPage; local += 1) {
      negative += Number(unpackFineLevelSetPackedPhi(
        samples[page * samplesPerPage + local]!,
      ) < 0);
    }
    result.set(signedPageCoordinate(metadata[at + 1]!).join(","), negative);
  }
  return result;
}

dawnTest("mini32 conserves liquid volume through four seconds",
  { timeout: 240_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/sparse-cm12-mini32-volume-dawn.test.ts");
    let device: GPUDevice | undefined;
    let solver: WebGPUAdaptiveMassSolver | undefined;
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href) as {
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
        event.preventDefault();validationErrors.push(event.error.message);
      });

      const scene = sceneDocument(getSceneDefinition("minimal-power-dam-break-32"));
      const requestedResolutionMode = process.env.FLUID_MINI32_RESOLUTION_MODE;
      const resolutionMode = requestedResolutionMode === "all-fine"
        || requestedResolutionMode === "all-coarse"
        ? requestedResolutionMode : "adaptive";
      const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
        brickFineResolution: "8", resolutionMode,
        selectorMode: "surface",
        surfaceFineRings: Number(process.env.FLUID_MINI32_SURFACE_RINGS ?? 1),
        topologyCadenceSteps: Number(
          process.env.FLUID_MINI32_TOPOLOGY_CADENCE ?? 1,
        ),
        timeStep: "paper",
        ...(process.env.FLUID_MINI32_SHARPENING === "off"
          ? { surfaceSharpening: "off" } : {}),
      });
      solver = process.env.FLUID_MINI32_PHASE1_QA === "1"
        ? await WebGPUAdaptiveMassSolver.createPhase1TransportReceiptOracleForQA(
          device, scene, "balanced", undefined, {
            resolutionMode, brickFineResolution: 8,
            surfaceFineRings: Number(process.env.FLUID_MINI32_SURFACE_RINGS ?? 1),
            activityPolicy: {
              ...SPARSE_CM12_ACTIVITY_POLICY,
              topologyCadenceSteps: Number(
                process.env.FLUID_MINI32_TOPOLOGY_CADENCE ?? 1,
              ),
            },
            timeStep: "paper",
            surfaceSharpeningEnabled:
              process.env.FLUID_MINI32_SHARPENING !== "off",
          }, () => {},
        )
        : await adaptiveMassMethod.createSolverAsync!(
          device, scene, "balanced", values, undefined, () => {},
        ) as WebGPUAdaptiveMassSolver;
      await solver.waitForSimulationReady();

      const [initialFields, initialWorld] = await Promise.all([
        solver.readDiagnosticFields(), solver.readWorldGrowthReceiptQA(),
      ]);
      const initialHostMass = densityMass(initialFields.density);
      const initialMass = initialHostMass + initialWorld.dynamicLiquidMassFineCells;
      const trajectory: Array<Record<string, unknown>> = [];
      const steps = Number(process.env.FLUID_MINI32_VOLUME_STEPS
        ?? Math.round(4 / CM12_PAPER_DT_S));
      const sampleEvery = Number(process.env.FLUID_MINI32_VOLUME_SAMPLE_EVERY ?? 15);
      const sampleFrom = Number(process.env.FLUID_MINI32_VOLUME_SAMPLE_FROM ?? 1);
      const detailSteps = new Set((process.env.FLUID_MINI32_VOLUME_DETAIL_STEPS ?? "")
        .split(",").filter(Boolean).map(Number));
      let previousActive = new Map<number, Awaited<ReturnType<
        WebGPUAdaptiveMassSolver["readGPUActivityPolicy"]>>["bricks"][number]>();
      for (let step = 1; step <= steps; step += 1) {
        assert.equal(solver.advanceTo(step * CM12_PAPER_DT_S, []), true);
        if (step === 1) {
          await device.queue.onSubmittedWorkDone();
          const [stepOneFields, negativeByPage]: [
            Awaited<ReturnType<WebGPUAdaptiveMassSolver["readDiagnosticFields"]>>,
            Map<string, number>,
          ] = await Promise.all([
            solver.readDiagnosticFields(),
            negativePresentationSamplesByCoordinate(device, solver),
          ]);
          let wetCornerSamples = 0;
          for (let z = 0; z < 8; z += 1) for (let y = 0; y < 16; y += 1) {
            for (let x = 0; x < 8; x += 1) {
              wetCornerSamples += Number(stepOneFields.density[
                x + 32 * (y + 32 * z)]! >= 0.5);
            }
          }
          assert.equal(wetCornerSamples, 2 * 8 ** 3,
            "the step-one corner fixture must keep both stacked wet bricks");
          for (const coordinate of ["0,0,0", "0,1,0"]) {
            assert.equal(negativeByPage.get(coordinate), 8 ** 3,
              `step one carved wet presentation brick ${coordinate}`);
          }
        }
        if (step % 2 === 0) await device.queue.onSubmittedWorkDone();
        if ((step < sampleFrom || step % sampleEvery !== 0) && step !== steps) continue;
        await device.queue.onSubmittedWorkDone();
        const [fields, activity, fpp, world, stats, phase1] = await Promise.all([
          solver.readDiagnosticFields(), solver.readGPUActivityPolicy(),
          solver.readFramePlanPresentationHeaderQA(), solver.readWorldGrowthReceiptQA(),
          solver.readStats(),
          process.env.FLUID_MINI32_PHASE1_QA === "1"
            ? solver.readPhase1TransportReceiptQA() : undefined,
        ]);
        const hostMass = densityMass(fields.density);
        const mass = hostMass + world.dynamicLiquidMassFineCells;
        const adaptiveStats = stats as typeof stats & AdaptiveMassStepTelemetry;
        let maximumDensity = -Infinity;
        let maximumDensityIndex = 0;
        for (let index = 0; index < fields.density.length; index += 1) {
          if (fields.density[index]! > maximumDensity) {
            maximumDensity = fields.density[index]!;
            maximumDensityIndex = index;
          }
        }
        const maximumDensityCoordinate = [
          maximumDensityIndex % 32,
          Math.floor(maximumDensityIndex / 32) % 32,
          Math.floor(maximumDensityIndex / (32 * 32)),
        ];
        const currentActive = new Map(activity.bricks.filter((brick) => brick.active)
          .map((brick) => [brick.key, brick] as const));
        const retired = [...previousActive].filter(([brickKey]) =>
          !currentActive.has(brickKey)).map(([, brick]) => ({
          key: brick.key, coordinate: brick.coordinate, meanDensity: brick.meanDensity,
          reasons: brick.reasons, supportMask: brick.supportMask,
          retiredResidueMass: brick.retiredResidueMassFineCells,
        }));
        const activated = [...currentActive].filter(([brickKey]) =>
          !previousActive.has(brickKey)).map(([, brick]) => ({
          key: brick.key, coordinate: brick.coordinate,
        }));
        trajectory.push({ step, time_s: step * CM12_PAPER_DT_S, mass, hostMass,
          dynamicMass: world.dynamicLiquidMassFineCells,
          maximumDensity, maximumDensityCoordinate,
          maximumDivergence_s: stats.maxDivergence_s,
          maximumMixedSeamDivergence_s:
            adaptiveStats.adaptiveMaximumMixedSeamDivergence_s,
          mixedSeamRows: stats.adaptiveAcceptedMixedSeamRowCount,
          phase1,
          relativeMass: mass / initialMass,
          activeBricks: activity.bricks.filter((brick) => brick.active).length,
          retiredResidueMass: activity.bricks.reduce((sum, brick) =>
            sum + brick.retiredResidueMassFineCells, 0),
          topologyGeneration: activity.acceptedTopologyGeneration,
          commitFailed: activity.commitFailed,
          fppFault: (fpp as Record<string, number>).faultCode,
          retired, activated,
          ...(process.env.FLUID_MINI32_VOLUME_DETAIL === "1"
            && (detailSteps.size === 0 || detailSteps.has(step)) ? {
            bricks: activity.bricks.filter((brick) => brick.active).map((brick) => ({
              key: brick.key, coordinate: brick.coordinate,
              meanDensity: brick.meanDensity, reasons: brick.reasons,
              supportMask: brick.supportMask,
              acceptedResolution: brick.acceptedResolution,
              planReasons: brick.planReasons,
              retiredResidueMass: brick.retiredResidueMassFineCells,
            })),
          } : {}),
        });
        previousActive = currentActive;
      }
      if (process.env.FLUID_MINI32_VOLUME_TRACE === "1") {
        process.stderr.write(`[mini32-volume] ${JSON.stringify({
          resolutionMode,
          topologyCadenceSteps: Number(
            process.env.FLUID_MINI32_TOPOLOGY_CADENCE ?? 1,
          ),
          initialMass, initialHostMass,
          initialDynamicMass: initialWorld.dynamicLiquidMassFineCells,
          trajectory, validationErrors,
        })}\n`);
      }
      const minimumRelativeMass = Math.min(...trajectory.map((sample) =>
        sample.relativeMass as number));
      assert.ok(minimumRelativeMass >= 0.995,
        `mini32 lost ${(100 * (1 - minimumRelativeMass)).toFixed(3)}% of its liquid: ${
          JSON.stringify(trajectory)}`);
      assert.deepEqual(validationErrors, []);
    } finally {
      solver?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();
    }
  });
