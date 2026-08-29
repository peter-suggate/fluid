/**
 * Stepwise physical-parity probe for the same mini dam represented as:
 * A. a finer authored lattice with a two-fine-cell minimum-size region; and
 * B. the physically equivalent lattice at half the linear resolution.
 *
 * The A fields are conservatively restricted to the B physical lattice
 * before comparison. This intentionally leaves the brick partition different.
 */
import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";
import type { SceneDescription } from "../lib/core/model";
import { resolveMethodValues } from "../lib/core/method-contract";
import {
  createMinimalPowerDamBreak32Scene,
  createMinimalPowerDamBreak64Scene,
  createMinimalPowerDamBreakScene,
} from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import {
  adaptiveMassMethod,
  adaptiveMassSolverOptions,
} from "../lib/methods/adaptive-mass/method";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const argument = (name: string, fallback: string): string =>
  process.argv.slice(2).find((value) => value.startsWith(`--${name}=`))
    ?.slice(name.length + 3) ?? fallback;
const steps = Number(argument("steps", "8"));
if (!Number.isSafeInteger(steps) || steps < 0) throw new RangeError(
  "steps must be a non-negative integer",
);
const sampleEvery = Number(argument("sample-every", "1"));
if (!Number.isSafeInteger(sampleEvery) || sampleEvery < 1) throw new RangeError(
  "sample-every must be a positive integer",
);
const sampleFrom = Number(argument("sample-from", "0"));
if (!Number.isSafeInteger(sampleFrom) || sampleFrom < 0) throw new RangeError(
  "sample-from must be a non-negative integer",
);
const dt_s = 0.004;
const pair = argument("pair", "32-16");
if (pair !== "32-16" && pair !== "64-32") {
  throw new RangeError("pair must be 32-16 or 64-32");
}
const fineResolution = pair === "64-32" ? 64 : 32;
const coarseResolution = fineResolution / 2;
const vexQA = argument("vex-qa", "off") === "on";
const phase1QA = argument("phase1-qa", "off") === "on";
const stageLimit = argument("stage-limit", "");
const stageFromStep = Number(argument("stage-from-step", "1"));
const sharpeningLimit = argument("sharpening-limit", "");
const topologyMode = argument("topology", "adaptive");
if (topologyMode !== "adaptive" && topologyMode !== "matched-uniform") {
  throw new RangeError("topology must be adaptive or matched-uniform");
}
const shellMode = argument("shell", "on");
if (shellMode !== "on" && shellMode !== "off") {
  throw new RangeError("shell must be on or off");
}
const densityBoxArgument = argument("density-box", "");
const densityBox = densityBoxArgument === "" ? undefined
  : densityBoxArgument.split(",").map(Number);
if (densityBox && (densityBox.length !== 6
  || densityBox.some((value) => !Number.isSafeInteger(value)))) {
  throw new RangeError("density-box must be x0,x1,y0,y1,z0,z1 in the coarse arm");
}
const fieldProbe = Number(argument("field-probe", "-1"));
if (!Number.isSafeInteger(fieldProbe) || fieldProbe < -1
  || fieldProbe >= coarseResolution ** 3) {
  throw new RangeError("field-probe must be a valid coarse-arm scalar cell index");
}
const FIELD_NAMES = ["density", "gamma", "sharpeningDelta",
  "sharpeningReceiptMass", "solidOpenFraction", "velocity", "pressureRhs",
  "pressure", "divergence"] as const;
type FieldName = typeof FIELD_NAMES[number];
type DiagnosticFields = Awaited<ReturnType<WebGPUAdaptiveMassSolver["readDiagnosticFields"]>>;

interface Snapshot {
  readonly step: number;
  readonly fields: DiagnosticFields;
  readonly topology: {
    readonly residentBricks: number;
    readonly resolutionHistogram: Readonly<Record<string, number>>;
    readonly bricks: readonly Readonly<Record<string, unknown>>[];
  };
  readonly stats: Readonly<Record<string, unknown>>;
  readonly phase1?: Awaited<ReturnType<
    WebGPUAdaptiveMassSolver["readPhase1TransportReceiptQA"]
  >>;
}

function physicalRegion(scene: SceneDescription) {
  scene.fluid.refinementRegions = [{
    id: "mini-physical-parity-two-cell-floor",
    rule: "minimum-cell-size",
    minimumCellSize_cells: 2,
    min_m: { x: -0.5 * scene.container.width_m, y: 0,
      z: -0.5 * scene.container.depth_m },
    max_m: { x: 0.5 * scene.container.width_m, y: scene.container.height_m,
      z: 0.5 * scene.container.depth_m },
  }];
  return scene;
}

async function runArm(
  device: GPUDevice,
  scene: SceneDescription,
  resolutionMode: "adaptive" | "all-fine" | "all-coarse",
  vexProbeCell?: number,
): Promise<readonly Snapshot[]> {
  const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
    timeStep: "scene",
    resolutionMode,
    brickFineResolution: "8",
    presentationPageResolution: "8",
    pressureIterations: Number(argument("pressure-iterations", "40")),
    pressureRelativeTolerance: Number(argument("pressure-tolerance", "0.001")),
    gammaDiffusion: argument("gamma-diffusion", "on"),
    surfaceSharpening: argument("surface-sharpening", "on"),
  });
  const solver = phase1QA
    ? await WebGPUAdaptiveMassSolver.createPhase1TransportReceiptOracleForQA(
      device, scene, "balanced", undefined, adaptiveMassSolverOptions(values),
      () => {},
    )
    : await adaptiveMassMethod.createSolverAsync!(
      device, scene, "balanced", values, undefined, () => {},
    ) as WebGPUAdaptiveMassSolver;
  const snapshots: Snapshot[] = [];
  try {
    for (let step = 0; step <= steps; step += 1) {
      if (step > 0) {
        if (stageLimit !== "" && step >= stageFromStep) {
          solver.sparseWorldTrace.setStageLimitForQA(stageLimit as never);
        }
        if (sharpeningLimit !== "" && step >= stageFromStep) {
          solver.sparseWorldTrace.setSharpeningPhaseLimitForQA(sharpeningLimit as never);
        }
        while (!solver.advanceTo(step * dt_s, [])) await new Promise(setImmediate);
        await device.queue.onSubmittedWorkDone();
      }
      if (step !== 0 && step !== steps
        && (step < sampleFrom || step % sampleEvery !== 0)) continue;
      const [fields, activity, stats, transportDispatch, velocityExtension,
        phase1] =
        await Promise.all([
        solver.readDiagnosticFields(false, stageLimit !== "" && step >= stageFromStep
          ? "candidate" : "accepted"),
        solver.readGPUActivityPolicy(), solver.readStats(),
        solver.readTransportPacketIndirectQA(),
        vexQA ? solver.readVelocityExtensionQA() : undefined,
        phase1QA && step > 0
          ? solver.readPhase1TransportReceiptQA(
            stageLimit !== "" && step >= stageFromStep,
            vexProbeCell === undefined ? [] : [vexProbeCell],
          ) : undefined,
      ]);
      const active = activity.bricks.filter((brick) => brick.active);
      const resolutions = [...new Set(active.map((brick) => brick.acceptedResolution))]
        .sort((left, right) => left - right);
      snapshots.push({
        step,
        fields,
        topology: {
          residentBricks: activity.residentBrickCount,
          resolutionHistogram: Object.fromEntries(resolutions.map((resolution) => [
            String(resolution),
            active.filter((brick) => brick.acceptedResolution === resolution).length,
          ])),
          bricks: active.map((brick) => ({
            coordinate: brick.coordinate,
            resolution: brick.acceptedResolution,
            planned: brick.plannedResolution,
            reasons: brick.reasons,
            planReasons: brick.planReasons,
            meanDensity: brick.meanDensity,
            scoreByte: brick.scoreByte,
            hotEpochs: brick.hotEpochs,
            quietEpochs: brick.quietEpochs,
            densityMoments: brick.densityMoments,
            supportMask: brick.supportMask,
            sweptSupportMask: brick.sweptSupportMask,
            maximumVelocityTravelFineCells: brick.maximumVelocityTravelFineCells,
            candidateResolution: brick.candidateResolution,
            candidateStatus: brick.candidateStatus,
            activatedStep: brick.activatedStep,
          })),
        },
        stats: {
          acceptedCells: stats.adaptiveAcceptedCellCount,
          acceptedRows: stats.adaptiveAcceptedRowCount,
          pressureCells: stats.adaptivePressureCellCount,
          pressureRows: stats.adaptivePressureActiveRowCount,
          pressureIterations: stats.pressureIterationsExecuted,
          pressureRelativeResidual: stats.pressureRelativeResidual,
          maxDivergenceAfter_s: stats.maxDivergenceAfter_s,
          maximumSpeed_m_s: stats.maxSpeed_m_s,
          topologyGeneration: stats.adaptiveTopologyShadowGeneration,
          promotedBricks: stats.adaptiveResolutionPromotedBrickCount,
          demotedBricks: stats.adaptiveResolutionDemotedBrickCount,
          hotBricks: stats.adaptiveActivityHotBrickCount,
          quietBricks: stats.adaptiveActivityQuietBrickCount,
          preparedBricks: stats.adaptiveTopologyPreparedBrickCount,
          committedBricks: stats.adaptiveTopologyCommittedBrickCount,
          transportPackets: transportDispatch[0],
          ...(velocityExtension && vexProbeCell !== undefined ? {
            vexValidCellCount: velocityExtension.header[8],
            vexProbeCell,
            vexProbeDepth: velocityExtension.acceptedDepth[vexProbeCell],
            vexProbeVelocity: Array.from(new Float32Array(
              velocityExtension.velocityBits.buffer,
              velocityExtension.velocityBits.byteOffset + 16 * vexProbeCell,
              4,
            )),
          } : {}),
        },
        ...(phase1 ? { phase1 } : {}),
      });
    }
    return snapshots;
  } finally {
    solver.destroy();
  }
}

function restrictScalar2(source: Float32Array): Float32Array {
  assert.equal(source.length, fineResolution ** 3);
  const result = new Float32Array(coarseResolution ** 3);
  for (let z = 0; z < coarseResolution; z += 1) {
    for (let y = 0; y < coarseResolution; y += 1) {
      for (let x = 0; x < coarseResolution; x += 1) {
        let sum = 0;
        for (let dz = 0; dz < 2; dz += 1) for (let dy = 0; dy < 2; dy += 1) {
          for (let dx = 0; dx < 2; dx += 1) {
            sum += source[2 * x + dx + fineResolution
              * ((2 * y + dy) + fineResolution * (2 * z + dz))]!;
          }
        }
        result[x + coarseResolution * (y + coarseResolution * z)] = sum / 8;
      }
    }
  }
  return result;
}

function restrictField2(name: FieldName, source: Float32Array): Float32Array {
  if (name !== "velocity") {
    const restricted = restrictScalar2(source);
    // RHS is an integrated control-volume flux in authored finest-cell units.
    // One coarse-arm unit is the volume of eight fine-arm units.
    if (name === "pressureRhs") {
      for (let index = 0; index < restricted.length; index += 1) {
        restricted[index] /= 8;
      }
    }
    return restricted;
  }
  assert.equal(source.length, 4 * fineResolution ** 3);
  const result = new Float32Array(4 * coarseResolution ** 3);
  for (let component = 0; component < 4; component += 1) {
    const scalar = new Float32Array(fineResolution ** 3);
    for (let index = 0; index < scalar.length; index += 1) {
      scalar[index] = source[4 * index + component]!;
    }
    const restricted = restrictScalar2(scalar);
    for (let index = 0; index < restricted.length; index += 1) {
      result[4 * index + component] = restricted[index]!;
    }
  }
  return result;
}

function difference(
  left: Float32Array,
  right: Float32Array,
  liquidMask?: Float32Array,
) {
  assert.equal(left.length, right.length);
  let absolute = 0, squared = 0, referenceAbsolute = 0, referenceSquared = 0;
  let maximumAbsolute = 0, differing = 0, firstDiffering = -1;
  let maximumIndex = -1;
  const examples: Array<{ index: number; left: number; right: number }> = [];
  for (let index = 0; index < left.length; index += 1) {
    const cell = liquidMask && left.length === 4 * liquidMask.length
      ? Math.floor(index / 4) : index;
    if (liquidMask && liquidMask[cell]! <= 1e-6) continue;
    const delta = left[index]! - right[index]!;
    absolute += Math.abs(delta);
    squared += delta * delta;
    referenceAbsolute += Math.abs(right[index]!);
    referenceSquared += right[index]! ** 2;
    if (Math.abs(delta) > maximumAbsolute) {
      maximumAbsolute = Math.abs(delta);
      maximumIndex = index;
    }
    if (delta !== 0) {
      differing += 1;
      if (firstDiffering < 0) firstDiffering = index;
      if (examples.length < 24) examples.push({
        index, left: left[index]!, right: right[index]!,
      });
    }
  }
  return {
    relativeL1: absolute / Math.max(referenceAbsolute, 1e-30),
    relativeL2: Math.sqrt(squared / Math.max(referenceSquared, 1e-30)),
    maximumAbsolute,
    differing,
    firstDiffering,
    firstValues: firstDiffering < 0 ? undefined
      : { left: left[firstDiffering], right: right[firstDiffering] },
    maximumIndex,
    maximumValues: maximumIndex < 0 ? undefined
      : { left: left[maximumIndex], right: right[maximumIndex] },
    examples,
  };
}

function scalarSum(values: Float32Array): number {
  let sum = 0, correction = 0;
  for (const value of values) {
    const adjusted = value - correction;
    const next = sum + adjusted;
    correction = next - sum - adjusted;
    sum = next;
  }
  return sum;
}

/** Resolution-independent mechanical/activity receipt from a dense diagnostic field. */
function physicalState(fields: DiagnosticFields, scene: SceneDescription) {
  const density = fields.density;
  const velocity = fields.velocity;
  assert.equal(velocity.length, 4 * density.length);
  const resolution = Math.round(Math.cbrt(density.length));
  assert.equal(resolution ** 3, density.length);
  const h = scene.voxelDomain.finestCellSize_m;
  const cellVolume_m3 = h ** 3;
  const fluidDensity_kg_m3 = scene.fluid.density_kg_m3;
  const gravity = scene.fluid.gravity_m_s2;
  let liquidCells = 0;
  let speedSquaredCells = 0;
  let kineticEnergy_J = 0;
  let potentialEnergy_J = 0;
  let movingVolumeAbove01_m3 = 0;
  let movingVolumeAbove05_m3 = 0;
  let maximumSpeed_m_s = 0;
  for (let z = 0; z < resolution; z += 1) {
    const positionZ_m = -0.5 * scene.container.depth_m + (z + 0.5) * h;
    for (let y = 0; y < resolution; y += 1) {
      const positionY_m = (y + 0.5) * h;
      for (let x = 0; x < resolution; x += 1) {
        const index = x + resolution * (y + resolution * z);
        const fill = Math.max(0, density[index]!);
        if (fill <= 0) continue;
        const positionX_m = -0.5 * scene.container.width_m + (x + 0.5) * h;
        const vx = velocity[4 * index]!, vy = velocity[4 * index + 1]!;
        const vz = velocity[4 * index + 2]!;
        const speedSquared = vx * vx + vy * vy + vz * vz;
        const speed = Math.sqrt(speedSquared);
        const volume_m3 = fill * cellVolume_m3;
        const mass_kg = fluidDensity_kg_m3 * volume_m3;
        liquidCells += fill;
        speedSquaredCells += fill * speedSquared;
        kineticEnergy_J += 0.5 * mass_kg * speedSquared;
        potentialEnergy_J += mass_kg * -(gravity.x * positionX_m
          + gravity.y * positionY_m + gravity.z * positionZ_m);
        if (speed > 0.01) movingVolumeAbove01_m3 += volume_m3;
        if (speed > 0.05) movingVolumeAbove05_m3 += volume_m3;
        maximumSpeed_m_s = Math.max(maximumSpeed_m_s, speed);
      }
    }
  }
  const mechanicalEnergy_J = kineticEnergy_J + potentialEnergy_J;
  return {
    liquidVolume_m3: liquidCells * cellVolume_m3,
    kineticEnergy_J,
    potentialEnergy_J,
    mechanicalEnergy_J,
    rmsSpeed_m_s: Math.sqrt(speedSquaredCells / Math.max(liquidCells, 1e-30)),
    maximumSpeed_m_s,
    movingVolumeAbove01_m3,
    movingVolumeAbove05_m3,
  };
}

await acquireWebGPUExclusiveLock("dawn-acceptance",
  "tools/probe-sparse-cm12-mini-physical-ab.ts");
let device: GPUDevice | undefined;
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
  device = await adapter.requestDevice({
    requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    event.preventDefault();
    validationErrors.push(event.error.message);
  });
  const fineScene = pair === "64-32"
    ? createMinimalPowerDamBreak64Scene() : createMinimalPowerDamBreak32Scene();
  const coarseScene = pair === "64-32"
    ? createMinimalPowerDamBreak32Scene() : createMinimalPowerDamBreakScene();
  if (shellMode === "off") {
    fineScene.solidVoxels = [];
    coarseScene.solidVoxels = [];
  }
  const fine = await runArm(device, physicalRegion(fineScene),
    topologyMode === "adaptive" ? "adaptive" : "all-coarse",
    pair === "64-32" ? 82 : undefined);
  const coarse = await runArm(device, coarseScene,
    topologyMode === "adaptive" ? "adaptive" : "all-fine",
    pair === "64-32" ? 76 : undefined);
  const initialFineMechanicalEnergy_J = physicalState(
    fine[0]!.fields, fineScene,
  ).mechanicalEnergy_J;
  const initialCoarseMechanicalEnergy_J = physicalState(
    coarse[0]!.fields, coarseScene,
  ).mechanicalEnergy_J;
  const trajectory = fine.map((fineSnapshot, snapshotIndex) => {
    const coarseSnapshot = coarse[snapshotIndex]!;
    assert.equal(coarseSnapshot.step, fineSnapshot.step,
      "fine and coarse checkpoints must remain synchronized");
    const step = fineSnapshot.step;
    const restricted = Object.fromEntries(FIELD_NAMES.map((name) => [
      name, restrictField2(name, fineSnapshot.fields[name]),
    ])) as Record<FieldName, Float32Array>;
    const liquidMask = Float32Array.from(restricted.density, (density, index) =>
      Math.max(density, coarseSnapshot.fields.density[index]!));
    const bulkLiquidMask = Float32Array.from(liquidMask, (density) =>
      density >= 0.5 ? density : 0);
    const finePhysical = physicalState(fineSnapshot.fields, fineScene);
    const coarsePhysical = physicalState(coarseSnapshot.fields, coarseScene);
    const fieldProbeValues = fieldProbe < 0 ? undefined
      : Object.fromEntries(FIELD_NAMES.map((name) => {
        const width = name === "velocity" ? 4 : 1;
        return [name, {
          fineRestricted: Array.from(restricted[name].subarray(
            width * fieldProbe, width * (fieldProbe + 1))),
          coarse: Array.from(coarseSnapshot.fields[name].subarray(
            width * fieldProbe, width * (fieldProbe + 1))),
        }];
      }));
    const densityBoxSamples = densityBox === undefined ? undefined : (() => {
      const [x0, x1, y0, y1, z0, z1] = densityBox;
      const samples: Array<Readonly<Record<string, unknown>>> = [];
      for (let z = z0!; z < z1!; z += 1) for (let y = y0!; y < y1!; y += 1) {
        for (let x = x0!; x < x1!; x += 1) {
          const index = x + coarseResolution * (y + coarseResolution * z);
          const left = restricted.density[index]!;
          const right = coarseSnapshot.fields.density[index]!;
          const leftOpen = restricted.solidOpenFraction[index]!;
          const rightOpen = coarseSnapshot.fields.solidOpenFraction[index]!;
          const leftFill = left / Math.max(leftOpen, 1e-6);
          const rightFill = right / Math.max(rightOpen, 1e-6);
          if ((leftFill >= 0.35 && leftFill <= 0.65)
            || (rightFill >= 0.35 && rightFill <= 0.65)
            || (leftFill >= 0.5) !== (rightFill >= 0.5)) {
            samples.push({ coordinate: [x, y, z], left, right,
              leftOpen, rightOpen, leftFill, rightFill,
              classificationDiffers: (leftFill >= 0.5) !== (rightFill >= 0.5) });
          }
        }
      }
      return samples;
    })();
    return {
      step,
      time_s: step * dt_s,
      fineTopology: fineSnapshot.topology,
      coarseTopology: coarseSnapshot.topology,
      fineStats: fineSnapshot.stats,
      coarseStats: coarseSnapshot.stats,
      fieldProbe: fieldProbe < 0 ? undefined : {
        index: fieldProbe,
        coordinate: [fieldProbe % coarseResolution,
          Math.floor(fieldProbe / coarseResolution) % coarseResolution,
          Math.floor(fieldProbe / coarseResolution ** 2)],
        values: fieldProbeValues,
      },
      finePhase1: fineSnapshot.phase1,
      coarsePhase1: coarseSnapshot.phase1,
      densityBoxSamples,
      physicalMass_m3: {
        fineRestricted: scalarSum(restricted.density)
          * coarseScene.voxelDomain.finestCellSize_m ** 3,
        coarse: scalarSum(coarseSnapshot.fields.density)
          * coarseScene.voxelDomain.finestCellSize_m ** 3,
        difference: (scalarSum(restricted.density)
          - scalarSum(coarseSnapshot.fields.density))
          * coarseScene.voxelDomain.finestCellSize_m ** 3,
      },
      physicalState: {
        fine: {
          ...finePhysical,
          mechanicalRetention: finePhysical.mechanicalEnergy_J
            / initialFineMechanicalEnergy_J,
        },
        coarse: {
          ...coarsePhysical,
          mechanicalRetention: coarsePhysical.mechanicalEnergy_J
            / initialCoarseMechanicalEnergy_J,
        },
        kineticEnergyDifference_J: finePhysical.kineticEnergy_J
          - coarsePhysical.kineticEnergy_J,
        mechanicalRetentionDifference: finePhysical.mechanicalEnergy_J
            / initialFineMechanicalEnergy_J
          - coarsePhysical.mechanicalEnergy_J
            / initialCoarseMechanicalEnergy_J,
      },
      fields: Object.fromEntries(FIELD_NAMES.map((name) => [name, difference(
        restricted[name], coarseSnapshot.fields[name],
      )])),
      liquidFields: Object.fromEntries(FIELD_NAMES.map((name) => [name, difference(
        restricted[name], coarseSnapshot.fields[name], liquidMask,
      )])),
      bulkLiquidFields: Object.fromEntries(FIELD_NAMES.map((name) => [name, difference(
        restricted[name], coarseSnapshot.fields[name], bulkLiquidMask,
      )])),
    };
  });
  console.log(JSON.stringify({
    probe: "sparse-cm12-mini-physical-ab",
    arms: {
      A: `mini${fineResolution} + whole-domain minimumCellSize_cells=2`,
      B: `mini${coarseResolution}`,
    },
    pair,
    dt_s,
    sampleEvery,
    sampleFrom,
    topologyMode,
    stageLimit: stageLimit || undefined,
    trajectory,
    validationErrors,
  }, null, 2));
} finally {
  device?.destroy();
  await releaseWebGPUExclusiveLock();
}
