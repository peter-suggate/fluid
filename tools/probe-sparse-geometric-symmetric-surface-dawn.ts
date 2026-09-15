/**
 * Dawn surface-evolution probe for the 3-D Sparse Geometric symmetric expansion.
 *
 * This deliberately has no page-count or expected-success assertions: it is a
 * reproduction tool for failures that occur before the canonical symmetry gate
 * can collect an evolution. Run the adaptive and all-fine arms separately and
 * compare their `checkpoints` by step.
 *
 * WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
 *   node --import tsx tools/probe-sparse-geometric-symmetric-surface-dawn.ts \
 *   --arm=adaptive --steps=20 --output=artifacts/sparse-geometric-surface/adaptive.json
 */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { SimulationFailureError } from "../lib/core/simulation-failure";
import { createSymmetricExpansionScene } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { sparseCM12DawnDefaultOptions } from "../lib/harness/sparse-cm12-dawn-defaults";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from
  "../lib/harness/node-dawn-provider";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";
import { SPARSE_CM12_ACTIVITY_POLICY } from
  "../lib/methods/adaptive-volume/features/adaptivity/policy";
import { readPublishedCM12Field } from "./sparse-cm12-published-field";
import { sparseGeometricLsvFaultContextAudit, sparseGeometricLsvStageAudit } from
  "./sparse-geometric-lsv-stage-audit";
import { expectedCentredSquareRadialFront, radialFrontsFromFields } from
  "./sparse-geometric-radial-front";

type Dimensions = readonly [number, number, number];
const argument = (name: string, fallback?: string) => process.argv.slice(2)
  .find(value => value.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const arm = argument("arm", "adaptive");
assert.ok(arm === "adaptive" || arm === "all-fine", "--arm must be adaptive or all-fine");
const steps = Number(argument("steps", "20"));
assert.ok(Number.isSafeInteger(steps) && steps >= 0, "--steps must be a non-negative integer");
const output = argument("output");
const outputPath = output ? resolve(output) : undefined;
const gravityArgument = argument("gravity", "scene");
assert.ok(gravityArgument === "scene" || gravityArgument === "0",
  "--gravity must be scene or 0");
const stageAuditEnabled = argument("stages", "0") === "1";

function scalarD4(field: ArrayLike<number>, [nx, ny, nz]: Dimensions) {
  let maximumAbsoluteError = 0, compared = 0, absentPairCount = 0;
  const byTransform = { reflectX: 0, reflectZ: 0, transposeXZ: 0 };
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    const source = Number(field[x + nx * (y + ny * z)]);
    for (const [name, tx, tz] of [
      ["reflectX", nx - 1 - x, z], ["reflectZ", x, nz - 1 - z],
      ["transposeXZ", z, x],
    ] as const) {
      const target = Number(field[tx + nx * (y + ny * tz)]);
      if (!Number.isFinite(source) || !Number.isFinite(target)) {
        if (Number.isFinite(source) !== Number.isFinite(target)) absentPairCount++;
        continue;
      }
      const error = Math.abs(source - target);
      byTransform[name] = Math.max(byTransform[name], error);
      maximumAbsoluteError = Math.max(maximumAbsoluteError, error); compared++;
    }
  }
  return { maximumAbsoluteError, byTransform, compared, absentPairCount };
}

function velocityD4(field: ArrayLike<number>, [nx, ny, nz]: Dimensions) {
  let maximumAbsoluteError_m_s = 0;
  const byTransform_m_s = { reflectX: 0, reflectZ: 0, transposeXZ: 0 };
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    const at = 4 * (x + nx * (y + ny * z));
    const source = [Number(field[at]), Number(field[at + 1]), Number(field[at + 2])];
    const targets = [
      ["reflectX", nx - 1 - x, z, [-source[0]!, source[1]!, source[2]!]],
      ["reflectZ", x, nz - 1 - z, [source[0]!, source[1]!, -source[2]!]],
      ["transposeXZ", z, x, [source[2]!, source[1]!, source[0]!]],
    ] as const;
    for (const [name, tx, tz, expected] of targets) {
      const targetAt = 4 * (tx + nx * (y + ny * tz));
      for (let axis = 0; axis < 3; axis++) {
        const error = Math.abs(expected[axis]! - Number(field[targetAt + axis]));
        byTransform_m_s[name] = Math.max(byTransform_m_s[name], error);
        maximumAbsoluteError_m_s = Math.max(maximumAbsoluteError_m_s, error);
      }
    }
  }
  return { maximumAbsoluteError_m_s, byTransform_m_s };
}

function heightD4(heights: Float32Array, nx: number, nz: number) {
  let maximumAbsoluteError_cells = 0, absentPairCount = 0;
  const byTransform_cells = { reflectX: 0, reflectZ: 0, transposeXZ: 0 };
  for (let z = 0; z < nz; z++) for (let x = 0; x < nx; x++) {
    const source = heights[x + nx * z]!;
    for (const [name, tx, tz] of [
      ["reflectX", nx - 1 - x, z], ["reflectZ", x, nz - 1 - z],
      ["transposeXZ", z, x],
    ] as const) {
      const target = heights[tx + nx * tz]!;
      if (!Number.isFinite(source) || !Number.isFinite(target)) {
        if (Number.isFinite(source) !== Number.isFinite(target)) absentPairCount++;
        continue;
      }
      const error = Math.abs(source - target);
      byTransform_cells[name] = Math.max(byTransform_cells[name], error);
      maximumAbsoluteError_cells = Math.max(maximumAbsoluteError_cells, error);
    }
  }
  return { maximumAbsoluteError_cells, byTransform_cells, absentPairCount };
}

function summarizeHeights(heights: Float32Array, nx: number, nz: number, cellSize_m: number) {
  const finite = Array.from(heights).filter(Number.isFinite);
  const mean = finite.reduce((sum, value) => sum + value, 0) / Math.max(1, finite.length);
  let minimumX = nx, maximumX = -1, minimumZ = nz, maximumZ = -1;
  for (let z = 0; z < nz; z++) for (let x = 0; x < nx; x++) {
    if (!Number.isFinite(heights[x + nx * z]!)) continue;
    minimumX = Math.min(minimumX, x); maximumX = Math.max(maximumX, x);
    minimumZ = Math.min(minimumZ, z); maximumZ = Math.max(maximumZ, z);
  }
  return { columns: finite.length, missingColumns: nx * nz - finite.length,
    minimum_cells: finite.length ? Math.min(...finite) : null,
    maximum_cells: finite.length ? Math.max(...finite) : null,
    mean_cells: finite.length ? mean : null,
    rmsVariation_cells: finite.length
      ? Math.sqrt(finite.reduce((sum, value) => sum + (value - mean) ** 2, 0) / finite.length)
      : null,
    mean_m: finite.length ? mean * cellSize_m : null,
    footprintCellBounds: finite.length
      ? { minimum: [minimumX, minimumZ], maximumExclusive: [maximumX + 1, maximumZ + 1] }
      : null,
    symmetry: heightD4(heights, nx, nz) };
}

/** Upper zero crossing of the cell-centred published phi in each column. */
function phiUpperHeights(phi: ArrayLike<number>, [nx, ny, nz]: Dimensions) {
  const result = new Float32Array(nx * nz).fill(Number.NaN);
  for (let z = 0; z < nz; z++) for (let x = 0; x < nx; x++) {
    let highestLiquid = -1;
    for (let y = 0; y < ny; y++) {
      if (Number(phi[x + nx * (y + ny * z)]) <= 0) highestLiquid = y;
    }
    if (highestLiquid < 0) continue;
    if (highestLiquid + 1 >= ny) { result[x + nx * z] = ny; continue; }
    const below = Number(phi[x + nx * (highestLiquid + ny * z)]);
    const above = Number(phi[x + nx * (highestLiquid + 1 + ny * z)]);
    result[x + nx * z] = Number.isFinite(below) && Number.isFinite(above) && below !== above
      ? highestLiquid + 0.5 + below / (below - above) : highestLiquid + 1;
  }
  return result;
}

function densityColumnHeights(density: ArrayLike<number>, [nx, ny, nz]: Dimensions) {
  const result = new Float32Array(nx * nz);
  for (let z = 0; z < nz; z++) for (let x = 0; x < nx; x++) {
    let height = 0;
    for (let y = 0; y < ny; y++) height += Math.max(0, Number(density[x + nx * (y + ny * z)]));
    result[x + nx * z] = height > 1e-6 ? height : Number.NaN;
  }
  return result;
}

function pairedHeightError(left: Float32Array, right: Float32Array) {
  let count = 0, sumSquared = 0, maximumAbsolute_cells = 0, phaseMismatchColumns = 0;
  for (let i = 0; i < left.length; i++) {
    if (!Number.isFinite(left[i]!) || !Number.isFinite(right[i]!)) {
      if (Number.isFinite(left[i]!) !== Number.isFinite(right[i]!)) phaseMismatchColumns++;
      continue;
    }
    const error = left[i]! - right[i]!;
    count++; sumSquared += error * error;
    maximumAbsolute_cells = Math.max(maximumAbsolute_cells, Math.abs(error));
  }
  return { count, rms_cells: count ? Math.sqrt(sumSquared / count) : null,
    maximumAbsolute_cells, phaseMismatchColumns };
}

function velocitySummary(velocity: ArrayLike<number>, density: ArrayLike<number>) {
  const maximumAbsolute_m_s = [0, 0, 0];
  const maximumAbsoluteLiquid_m_s = [0, 0, 0];
  let maximumSpeed_m_s = 0, maximumLiquidSpeed_m_s = 0, liquidCells = 0;
  for (let cell = 0; cell < density.length; cell++) {
    const xyz = [Number(velocity[4 * cell]), Number(velocity[4 * cell + 1]),
      Number(velocity[4 * cell + 2])];
    const speed = Math.hypot(...xyz);
    maximumSpeed_m_s = Math.max(maximumSpeed_m_s, speed);
    for (let axis = 0; axis < 3; axis++) maximumAbsolute_m_s[axis] = Math.max(
      maximumAbsolute_m_s[axis]!, Math.abs(xyz[axis]!));
    if (!(Number(density[cell]) > 0.5)) continue;
    liquidCells++; maximumLiquidSpeed_m_s = Math.max(maximumLiquidSpeed_m_s, speed);
    for (let axis = 0; axis < 3; axis++) maximumAbsoluteLiquid_m_s[axis] = Math.max(
      maximumAbsoluteLiquid_m_s[axis]!, Math.abs(xyz[axis]!));
  }
  return { liquidThreshold: 0.5, liquidCells, maximumSpeed_m_s, maximumLiquidSpeed_m_s,
    maximumAbsoluteByAxis_m_s: maximumAbsolute_m_s,
    maximumAbsoluteLiquidByAxis_m_s: maximumAbsoluteLiquid_m_s };
}

function scalarMaximumAbsolute(values: ArrayLike<number>) {
  let maximumAbsolute = 0, nonFiniteCount = 0;
  for (let i = 0; i < values.length; i++) {
    const value = Number(values[i]);
    if (!Number.isFinite(value)) nonFiniteCount++;
    else maximumAbsolute = Math.max(maximumAbsolute, Math.abs(value));
  }
  return { maximumAbsolute, nonFiniteCount };
}

function scalarSummary(values: ArrayLike<number>) {
  let minimum = Infinity, maximum = -Infinity, sum = 0, nonFiniteCount = 0;
  for (let i = 0; i < values.length; i++) {
    const value = Number(values[i]);
    if (!Number.isFinite(value)) { nonFiniteCount++; continue; }
    minimum = Math.min(minimum, value); maximum = Math.max(maximum, value); sum += value;
  }
  return { count: values.length, nonFiniteCount,
    minimum: nonFiniteCount === values.length ? null : minimum,
    maximum: nonFiniteCount === values.length ? null : maximum, sum };
}

function changeFromInitial(current: ArrayLike<number>, initial: ArrayLike<number>) {
  assert.equal(current.length, initial.length);
  let compared = 0, absentPhaseChanges = 0, maximumAbsolute = 0, sumSquared = 0;
  for (let i = 0; i < current.length; i++) {
    const value = Number(current[i]), expected = Number(initial[i]);
    if (!Number.isFinite(value) || !Number.isFinite(expected)) {
      if (Number.isFinite(value) !== Number.isFinite(expected)) absentPhaseChanges++;
      continue;
    }
    const delta = value - expected;
    compared++; maximumAbsolute = Math.max(maximumAbsolute, Math.abs(delta));
    sumSquared += delta * delta;
  }
  return { compared, absentPhaseChanges, maximumAbsolute,
    rms: compared ? Math.sqrt(sumSquared / compared) : null };
}

const modulePath = process.env.WEBGPU_NODE_MODULE ?? `${process.cwd()}/node_modules/webgpu/index.js`;
await acquireWebGPUExclusiveLock("dawn-probe", `symmetric-surface-${arm}`);
let device: GPUDevice | undefined;
let solver: WebGPUAdaptiveMassSolver | undefined;
let faultAudit: ReturnType<typeof sparseGeometricLsvFaultContextAudit> | undefined;
const validationErrors: string[] = [];
const report: Record<string, unknown> & { checkpoints: unknown[] } = {
  probe: "sparse-geometric-symmetric-surface-evolution", arm, steps,
  benchmark: arm === "all-fine"
    ? "forced finest surface-rung control; completed post-presentation receipts prove whether every then-active brick is finest, while transient prephysics support topology is not forced"
    : "compare equal completed steps with the forced finest surface-rung arm as a shared-method resolution-sensitivity control",
  dt_s: CM12_PAPER_DT_S, gravity: gravityArgument, checkpoints: [],
};
const persist = () => {
  if (!outputPath) return;
  mkdirSync(dirname(outputPath), { recursive: true });
  writeFileSync(outputPath, JSON.stringify(report, null, 2) + "\n");
};
try {
  const dawn = await import(pathToFileURL(modulePath).href) as NodeDawnProvider;
  Object.assign(globalThis, dawn.globals);
  const gpu = createProcessRetainedDawnGPU(dawn, [
    `backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`,
    "enable-dawn-features=disable_blob_cache",
  ]);
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter, "Dawn did not expose a WebGPU adapter");
  device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  device.addEventListener("uncapturederror", event => {
    event.preventDefault(); validationErrors.push(event.error.message);
  });
  const scene = createSymmetricExpansionScene();
  if (gravityArgument === "0") scene.fluid.gravity_m_s2 = { x: 0, y: 0, z: 0 };
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = CM12_PAPER_DT_S;
  const defaults = sparseCM12DawnDefaultOptions();
  const options = { ...defaults,
    ...(arm === "all-fine" ? {
      initialResolutionForQA: 8 as const,
      activityPolicy: { ...SPARSE_CM12_ACTIVITY_POLICY, ...defaults.activityPolicy,
        forcedSurfaceResolutionForQA: 8 as const },
    } : {}),
    pressureRelativeTolerance: 0 };
  solver = await WebGPUAdaptiveMassSolver.createCompiledTopologyTransport(
    device, scene, "balanced", undefined, options, () => {});
  await solver.waitForSimulationReady();
  const dimensions = [solver.info.nx, solver.info.ny, solver.info.nz] as Dimensions;
  report.dimensions = dimensions; report.cellSize_m = solver.info.cellSize_m;
  const expectedInitial = { surfaceHeight_cells: 8,
    footprintCellBounds: { minimum: [8, 8], maximumExclusive: [24, 24] },
    volumeFine3: 16 * 8 * 16 };
  report.expectedInitial = expectedInitial;
  report.expectedInitialRadialFront = { shape: "centred 16x16-cell square",
    radii_cells: expectedCentredSquareRadialFront(8) };
  const stageAudit = stageAuditEnabled
    ? sparseGeometricLsvStageAudit(device, solver, dimensions) : undefined;
  faultAudit = stageAuditEnabled ? undefined
    : sparseGeometricLsvFaultContextAudit(device, solver);
  let firstFrameStages: Record<string, unknown> | undefined;
  let initialPublishedPhi: Float32Array | undefined;
  let initialDensity: Float32Array | undefined;
  for (let step = 0; step <= steps; step++) {
    if (step > 0) {
      while (!solver.advanceTo(step * CM12_PAPER_DT_S, [])) await new Promise<void>(setImmediate);
      await solver.awaitFrameCompletion(); await device.queue.onSubmittedWorkDone();
      if (step === 1 && stageAudit) firstFrameStages = await stageAudit.read();
    }
    const [published, fields, physical, adaptivePhi, activity, transport,
      pressureMembership, projectionComponents, stats] = await Promise.all([
      readPublishedCM12Field(device, solver), solver.readDiagnosticFields(true),
      solver.readAcceptedGeometricVolumeQA(), solver.readAdaptiveLevelSetQA(true),
      solver.readGPUActivityPolicy(), step ? solver.readGeometricVolumeTransportReceiptQA() : undefined,
      solver.readPressureCanonicalMembershipQA(), solver.readGeometricProjectionComponentsQA(),
      solver.readStats(),
    ]);
    const phiHeights = phiUpperHeights(published.values, dimensions);
    const densityHeights = densityColumnHeights(fields.density, dimensions);
    initialPublishedPhi ??= published.values.slice();
    initialDensity ??= fields.density.slice();
    const active = activity.bricks.filter(brick => brick.active);
    const phiVertices = adaptivePhi.vertices ?? [];
    const phiLiquidVertices = phiVertices.filter(vertex => vertex.phiFine <= 0).length;
    const allActiveBricksFinest = active.every(brick => brick.acceptedResolution === 8);
    const checkpoint = { step, time_s: step * CM12_PAPER_DT_S,
      acceptedVolumeFine3: physical.volumeFine3,
      relativeVolumeDrift: (physical.volumeFine3 - expectedInitial.volumeFine3) / expectedInitial.volumeFine3,
      publishedPhi: { finiteSamples: Array.from(published.values).filter(Number.isFinite).length,
        valueUnits: "metres (packed f16 publication multiplies phiFine by finest cell size)",
        symmetry: scalarD4(published.values, dimensions),
        upperSurface: summarizeHeights(phiHeights, dimensions[0], dimensions[2], solver.info.cellSize_m),
        // Raw values make a pre-fault checkpoint independently diagnosable.
        values: Array.from(published.values) },
      density: Array.from(fields.density), densitySummary: scalarSummary(fields.density),
      densitySurface: summarizeHeights(densityHeights, dimensions[0], dimensions[2], solver.info.cellSize_m),
      phiVsDensityUpperSurface: pairedHeightError(phiHeights, densityHeights),
      radialFront: radialFrontsFromFields(published.values, fields.density, dimensions),
      adaptivePhi: { acceptedGeneration: adaptivePhi.acceptedGeneration,
        activeVertices: adaptivePhi.activeVertices, activeCells: adaptivePhi.activeCells,
        constrainedVertices: adaptivePhi.constrainedVertices, fault: adaptivePhi.fault,
        liquidVertexCriterion: "phiFine <= 0", liquidVertices: phiLiquidVertices,
        header: adaptivePhi.header, vertices: adaptivePhi.vertices },
      pressure: { canonicalMembership: pressureMembership,
        pressureCells: pressureMembership.cell.activeBitCount,
        pressureRows: pressureMembership.row.activeBitCount,
        generations: { peiAccepted: pressureMembership.cell.acceptedGeneration,
          peiCandidate: pressureMembership.cell.candidateGeneration,
          acceptedTopology: stats.topologyGenerationCount,
          adaptiveLevelSet: adaptivePhi.acceptedGeneration },
        phiLiquidVertices, phiLiquidVerticesPerPressureCell:
          pressureMembership.cell.activeBitCount > 0
            ? phiLiquidVertices / pressureMembership.cell.activeBitCount : null,
        projectionComponents,
        field: scalarMaximumAbsolute(fields.pressure),
        symmetry: scalarD4(fields.pressure, dimensions),
        divergence: scalarMaximumAbsolute(fields.divergence),
        divergenceSymmetry: scalarD4(fields.divergence, dimensions),
        receipt: { relativeResidual: stats.pressureRelativeResidual,
          recursiveRelativeResidual: stats.pressureRecursiveRelativeResidual,
          iterationsExecuted: stats.pressureIterationsExecuted,
          maximumPostProjectionDivergence_s: stats.maxDivergenceAfter_s } },
      velocity: { ...velocitySummary(fields.velocity, fields.density),
        symmetry: velocityD4(fields.velocity, dimensions) },
      analyticZeroMotion: gravityArgument === "0" ? {
        expected: "authored box, volume, phi, density, and velocity remain unchanged",
        publishedPhiChangeMetres: changeFromInitial(published.values, initialPublishedPhi),
        densityChange: changeFromInitial(fields.density, initialDensity),
        volumeErrorFine3: physical.volumeFine3 - expectedInitial.volumeFine3,
      } : undefined,
      ...(step === 1 && firstFrameStages ? { levelSetStageAudit: firstFrameStages } : {}),
      topology: { activeBricks: active.length,
        acceptedGeneration: stats.topologyGenerationCount,
        allActiveBricksFinest, referenceRequirementSatisfied:
          arm !== "all-fine" || allActiveBricksFinest,
        byAcceptedResolution: Object.fromEntries([...new Set(active.map(brick => brick.acceptedResolution))]
          .sort((a, b) => a - b).map(resolution => [resolution,
            active.filter(brick => brick.acceptedResolution === resolution).length])) },
      transport,
    };
    report.checkpoints.push(checkpoint);
    process.stderr.write(JSON.stringify({ arm, step, volume: physical.volumeFine3,
      phiHeight: checkpoint.publishedPhi.upperSurface.mean_cells,
      phiD4: checkpoint.publishedPhi.symmetry.maximumAbsoluteError,
      activeBricks: active.length }) + "\n");
    persist();
  }
  report.validationErrors = validationErrors; report.completed = true;
} catch (error) {
  report.completed = false;
  report.failure = error instanceof Error ? error.stack : String(error);
  if (error instanceof SimulationFailureError) {
    const prior = report.checkpoints.at(-1) as {
      step?: number;
      adaptivePhi?: { acceptedGeneration?: number; header?: { sourceBank?: number };
        activeVertices?: number; activeCells?: number;
        vertices?: Array<{ positionFine: readonly number[]; phiFine: number;
          support: number; constrained: boolean }> };
      topology?: unknown;
    } | undefined;
    const owner = error.failure.ownerId;
    const isLevelSetAdvection = error.failure.code === "ADAPTIVE_LEVEL_SET"
      && error.failure.kernel === "lsvAdvectPhi";
    const samplePositionFine = isLevelSetAdvection
      ? error.failure.operands.slice(1, 4) : undefined;
    report.failureDetail = {
      receipt: error.failure,
      samplePositionFine,
      exactPrephysicsLevelSet: isLevelSetAdvection
        ? await faultAudit?.read(owner, samplePositionFine) : undefined,
      prephysicsAcceptedLevelSet: isLevelSetAdvection && prior ? {
        checkpointStep: prior.step,
        generation: prior.adaptivePhi?.acceptedGeneration,
        sourceBank: prior.adaptivePhi?.header?.sourceBank,
        activeVertices: prior.adaptivePhi?.activeVertices,
        activeCells: prior.adaptivePhi?.activeCells,
        header: prior.adaptivePhi?.header,
        ownerVertexId: owner,
        ownerVertex: prior.adaptivePhi?.vertices?.[owner] ?? null,
        supportLegend: { 0: "absent", 1: "deep air", 2: "deep liquid", 3: "metric" },
      } : null,
      prephysicsTopology: prior?.topology ?? null,
      note: isLevelSetAdvection
        ? "The rolling snapshot is the accepted source image at the projection seam entering lsvAdvectPhi; the sticky receipt sample position is the clipped characteristic departure in fine-cell coordinates."
        : "Owner semantics are kernel-specific; no adaptive-level-set vertex join was attempted for this failure.",
    };
  }
  report.validationErrors = validationErrors;
  persist();
  process.exitCode = 1;
} finally {
  try {
    try { faultAudit?.destroy(); } catch {}
    try { solver?.destroy(); } catch {}
    try { device?.destroy(); } catch {}
  } finally { await releaseWebGPUExclusiveLock(); }
}
const json = JSON.stringify(report, null, 2) + "\n";
if (outputPath) writeFileSync(outputPath, json);
console.log(outputPath ? JSON.stringify({ output: outputPath, completed: report.completed,
  checkpoints: report.checkpoints.length }) : json);
