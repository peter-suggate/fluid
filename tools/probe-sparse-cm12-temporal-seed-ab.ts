#!/usr/bin/env node
/** Construction-only exact temporal pressure-seed A/B; no production toggle. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";

import { createMinimalPowerDamBreak64Scene, createOceanSeicheScene,
  createSymmetricExpansionScene } from
  "../lib/core/scenes";
import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { gpuCompilationManagerFor, invalidateGPUCompilationManager } from
  "../lib/core/gpu-compilation-manager";
import { resolveMethodValues } from "../lib/core/method-contract";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from
  "../lib/harness/node-dawn-provider";
import { adaptiveMassMethod, adaptiveMassSolverOptions } from
  "../lib/methods/adaptive-mass/method";
import { SPARSE_CM12_CANONICAL_MEMBERSHIP_PHASE } from
  "../lib/methods/adaptive-mass/sparse-cm12-canonical-membership";
import { inspectSparseCM12PressureCutoverAuthorities } from
  "../lib/methods/adaptive-mass/sparse-cm12-pressure-cutover-observability";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import type { SparseCM12TemporalSeedQA, SparseCM12TemporalSeedQAMode } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";
import type { SparseCM12FacePreparationTileCensusQA } from
  "../lib/methods/adaptive-mass/sparse-cm12-face-preparation-tile-census";
import { SPARSE_CM12_FPA_VEX_READ_CENSUS_MAX_TILES_PER_ROW,
  type SparseCM12FpaVexReadCensusSummaryQA } from
  "../lib/methods/adaptive-mass/sparse-cm12-fpa-vex-read-census";

const value = (name: string, fallback: string) => process.argv.slice(2)
  .find((item) => item.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
if (process.argv.includes("--help") || process.argv.includes("-h")) {
  console.log(`Sparse CM12 construction-only temporal-seed A/B

Usage:
  WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \\
  FLUID_WEBGPU_BACKEND=metal \\
  node --import tsx tools/probe-sparse-cm12-temporal-seed-ab.ts [options]

Options:
  --scene=symmetric5|symmetric20|symmetric60|dam5|ocean8
                       Fixed paired scene/rung (default symmetric5)
  --out=PATH           JSON success or failure artifact

Both arms use immutable QA constructors. Ordinary production construction has
no temporal-seed selector, and every failure is persisted before rethrow.`);
  process.exit(0);
}
const sceneName = value("scene", "symmetric5");
if (sceneName !== "symmetric5" && sceneName !== "symmetric20"
  && sceneName !== "symmetric60" && sceneName !== "ocean8" && sceneName !== "dam5") {
  throw new RangeError("scene must be symmetric5, symmetric20, symmetric60, dam5, or ocean8");
}
const steps = sceneName === "symmetric20" ? 20
  : sceneName === "symmetric60" ? 60 : sceneName === "ocean8" ? 8 : 5;
const out = resolve(value("out",
  `artifacts/sparse-cm12-temporal-seed-ab-${sceneName}.json`));
const hash = (view: ArrayBufferView): string => createHash("sha256")
  .update(new Uint8Array(view.buffer, view.byteOffset, view.byteLength)).digest("hex");
const hashJSON = (input: unknown): string => createHash("sha256")
  .update(JSON.stringify(input)).digest("hex");

const methodValues = resolveMethodValues(adaptiveMassMethod, "balanced", {
  timeStep: sceneName === "ocean8" ? "scene" : "paper",
  brickFineResolution: "16", presentationPageResolution: "16",
});
const options = adaptiveMassSolverOptions(methodValues);
const buildSymmetricScene = () => {
  const scene = createSymmetricExpansionScene();
  const horizontalGrid = 32;
  scene.voxelDomain.finestCellSize_m = scene.container.width_m / horizontalGrid;
  const verticalGrid = horizontalGrid / 2;
  const brickSize = scene.voxelDomain.brickSize_cells;
  const brickGrid = [horizontalGrid / brickSize, verticalGrid / brickSize,
    horizontalGrid / brickSize] as const;
  scene.fluid.initialBrickSeeds_m = [];
  for (let bz = brickGrid[2] / 4; bz < 3 * brickGrid[2] / 4; bz += 1)
    for (let by = 0; by < brickGrid[1] / 2; by += 1)
      for (let bx = brickGrid[0] / 4; bx < 3 * brickGrid[0] / 4; bx += 1) {
        scene.fluid.initialBrickSeeds_m.push({
          x: -0.5 * scene.container.width_m
            + (bx + 0.5) * brickSize * scene.voxelDomain.finestCellSize_m,
          y: (by + 0.5) * brickSize * scene.voxelDomain.finestCellSize_m,
          z: -0.5 * scene.container.depth_m
            + (bz + 0.5) * brickSize * scene.voxelDomain.finestCellSize_m,
        });
      }
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = CM12_PAPER_DT_S;
  return scene;
};
const buildScene = sceneName.startsWith("symmetric") ? buildSymmetricScene
  : sceneName === "ocean8" ? createOceanSeicheScene : createMinimalPowerDamBreak64Scene;
const timingScene = buildScene();
const dt_s = options.timeStep === "paper"
  ? CM12_PAPER_DT_S : timingScene.numerics.maxDt_s;

type SeedSolver = WebGPUAdaptiveMassSolver & {
  readTemporalSeedQA(): Promise<SparseCM12TemporalSeedQA>;
  readFacePreparationTileCensusQA():
    Promise<SparseCM12FacePreparationTileCensusQA>;
  readFpaVexReadCensusQA(): Promise<SparseCM12FpaVexReadCensusSummaryQA>;
};
type SeedFactory = typeof WebGPUAdaptiveMassSolver.createAsync;
type Constructors = {
  createTemporalCurrentSeedOracleForQA?: SeedFactory;
  createTemporalChangeSeedOracleForQA?: SeedFactory;
};
const constructors = WebGPUAdaptiveMassSolver as unknown as Constructors;
const currentFactory = constructors.createTemporalCurrentSeedOracleForQA;
const changeFactory = constructors.createTemporalChangeSeedOracleForQA;

interface GateError {
  readonly kind: "arm" | "paired" | "validation";
  readonly message: string;
  readonly arm?: SparseCM12TemporalSeedQAMode;
  readonly step?: number;
  readonly field?: string;
  readonly expected?: unknown;
  readonly actual?: unknown;
}
const gateErrors: GateError[] = [];
const validationErrors: string[] = [];
const lifecycleErrors: string[] = [];
const recordGate = (condition: boolean, error: GateError): void => {
  if (!condition) gateErrors.push(error);
};
const serializedError = (error: unknown) => error instanceof Error
  ? { name: error.name, message: error.message, stack: error.stack }
  : { name: "Error", message: String(error) };

interface FrameReceipt {
  readonly step: number;
  readonly fields: Readonly<Record<string, string>>;
  readonly massFineCells: number;
  readonly topologySha256: string;
  readonly temporal: SparseCM12TemporalSeedQA;
  readonly facePreparationTileCensus: SparseCM12FacePreparationTileCensusQA;
  readonly fpaVexReadCensus: SparseCM12FpaVexReadCensusSummaryQA;
  readonly fca: unknown;
  readonly srr: unknown;
  readonly vex: unknown;
  readonly pcm: unknown;
  readonly ptr: unknown;
  readonly pressure: unknown;
}
interface ArmReceipt {
  readonly mode: SparseCM12TemporalSeedQAMode;
  readonly frames: readonly FrameReceipt[];
}

const partialFrames: Record<SparseCM12TemporalSeedQAMode, FrameReceipt[]> = {
  current: [], change: [],
};
const PCM_INVALID = 0xffff_ffff;

const WORK_COUNTER_KEYS = new Set([
  "work", "clean", "nextWork", "nextClean", "scheduledWork", "executedWork",
  "classified", "executed", "dirtyCount", "workCount", "executedCount",
  "skippedCount", "directCount", "closureCount", "expectedProducerReceipts",
  "coveredProducerReceipts", "cellExecutionCount", "rowExecutionCount",
  "brickDirtyLeafCount", "rowDirtyLeafCount", "familyDirtyCount",
  "familyExecutedCount", "rootCount", "blastCount", "maximumDepth",
  "executedCellCount", "reusedCellCount", "temporalScalarCellCount",
  "temporalScalarRowCount", "pcmCellDirtyLeafCount", "pcmRowDirtyLeafCount",
]);
const WORK_OBSERVATION_KEYS = new Set([
  ...WORK_COUNTER_KEYS, "seedCells", "firstDilationCells", "finalCells", "finalRows",
]);
const OMIT_AUTHORITY_BRANCHES = new Set([
  "massCleanTiles", "velocityRejections", "candidateTiles", "stages",
  "cellClassificationScratchQA",
]);
const stableAuthorityIdentity = (input: unknown): unknown => {
  if (Array.isArray(input)) return input.map(stableAuthorityIdentity);
  if (!input || typeof input !== "object") return input;
  return Object.fromEntries(Object.entries(input).flatMap(([key, child]) =>
    WORK_COUNTER_KEYS.has(key) || OMIT_AUTHORITY_BRANCHES.has(key)
      ? [] : [[key, stableAuthorityIdentity(child)]]));
};
const workObservations = (input: unknown, prefix = "",
  result: Record<string, number | readonly number[]> = {}):
Record<string, number | readonly number[]> => {
  if (Array.isArray(input)) {
    input.forEach((child, index) => workObservations(child, `${prefix}[${index}]`, result));
  } else if (input && typeof input === "object") {
    for (const [key, child] of Object.entries(input)) {
      const path = prefix ? `${prefix}.${key}` : key;
      if (WORK_OBSERVATION_KEYS.has(key)
        && (typeof child === "number"
          || (Array.isArray(child) && child.every((value) => typeof value === "number")))) {
        result[path] = child as number | readonly number[];
      } else workObservations(child, path, result);
    }
  }
  return result;
};
const observationDelta = (current: Record<string, number | readonly number[]>,
  change: Record<string, number | readonly number[]>) => Object.fromEntries(
  [...new Set([...Object.keys(current), ...Object.keys(change)])].sort().map((key) => {
    const left = current[key], right = change[key];
    if (typeof left === "number" && typeof right === "number") {
      return [key, right - left];
    }
    if (Array.isArray(left) && Array.isArray(right) && left.length === right.length) {
      return [key, left.map((value, index) => Number(right[index]) - Number(value))];
    }
    return [key, null];
  }),
);
const acceptedTopologyIdentity = (snapshot: Awaited<ReturnType<
  SeedSolver["readGPUActivityPolicy"]>>) => ({
  acceptedSteps: snapshot.acceptedSteps,
  bricks: snapshot.bricks.map(({ key, active, acceptedResolution }) =>
    ({ key, active, acceptedResolution })),
});
const requireCoverageComplete = (input: unknown, arm: SparseCM12TemporalSeedQAMode,
  step: number, prefix = ""): void => {
  if (Array.isArray(input)) {
    input.forEach((child, index) => requireCoverageComplete(child, arm, step,
      `${prefix}[${index}]`));
    return;
  }
  if (!input || typeof input !== "object") return;
  const record = input as Record<string, unknown>;
  if (typeof record.expectedProducerReceipts === "number"
    && typeof record.coveredProducerReceipts === "number") {
    recordGate(record.expectedProducerReceipts === record.coveredProducerReceipts, {
      kind: "arm", arm, step, field: `${prefix}.producerReceipts`,
      expected: record.expectedProducerReceipts, actual: record.coveredProducerReceipts,
      message: `${arm} step ${step} ${prefix} producer receipt coverage is incomplete`,
    });
  }
  for (const [key, child] of Object.entries(record)) {
    requireCoverageComplete(child, arm, step, prefix ? `${prefix}.${key}` : key);
  }
};

const compactPCM = (pcm: Awaited<ReturnType<SeedSolver["readPressureCanonicalMembershipQA"]>>) => {
  const { classificationBitCount, classificationBitsSha256,
    matchesClassification, ...cellAuthority } = pcm.cell;
  return {
    mode: pcm.mode,
    cell: cellAuthority,
    // In local mode layout.liquid aliases conditioned-gamma scratch and only
    // visited pressure cells are overwritten with 0/1 classifications. Keep
    // this whole-capacity threshold census for diagnosis, never paired identity.
    cellClassificationScratchQA: {
      classificationBitCount, classificationBitsSha256, matchesClassification,
    },
    row: pcm.row,
    thetaSha256: pcm.thetaSha256,
    coefficientSha256: pcm.coefficientSha256,
    rhsSha256: pcm.rhsSha256,
    facePreparationSha256: pcm.facePreparationSha256,
    faceABitsSha256: hash(pcm.qaRaw.faceABits),
    faceBBitsSha256: hash(pcm.qaRaw.faceBBits),
    aggregateEdgeSha256: pcm.aggregateEdgeSha256,
    brickDiagonalSha256: pcm.brickDiagonalSha256,
    hierarchyEdgeSha256: pcm.hierarchyEdgeSha256,
    hierarchyDiagonalSha256: pcm.hierarchyDiagonalSha256,
    faceAuthority: pcm.faceAuthority,
  };
};

async function runArm(device: GPUDevice, mode: SparseCM12TemporalSeedQAMode,
  factory: SeedFactory): Promise<ArmReceipt> {
  const scene = buildScene();
  const solver = await factory.call(WebGPUAdaptiveMassSolver,
    device, scene, "balanced", undefined, options, () => {}) as SeedSolver;
  const frames = partialFrames[mode];
  frames.length = 0;
  try {
    // Prime the one-frame-lagged pressure attribution without advancing physics.
    await solver.readStats();
    for (let step = 1; step <= steps; step += 1) {
      if (!solver.advanceTo(step * dt_s, [])) {
        recordGate(false, { kind: "arm", arm: mode, step, field: "advanceTo",
          expected: true, actual: false, message: `${mode} step ${step} did not encode` });
        throw new Error(`${mode} step ${step} did not encode`);
      }
      await device.queue.onSubmittedWorkDone();
      const [fields, topology, temporal, facePreparationTileCensus,
        fpaVexReadCensus, fca,
        srrHeader, srrWork, vex, pcm, stats] =
        await Promise.all([solver.readDiagnosticFields(), solver.readGPUActivityPolicy(),
          solver.readTemporalSeedQA(), solver.readFacePreparationTileCensusQA(),
          solver.readFpaVexReadCensusQA(),
          solver.readFrameControlQA(),
          solver.readScalarAuthorityHeaderQA(), solver.readScalarAuthorityQA(),
          solver.readVelocityExtensionHeaderQA(),
          solver.readPressureCanonicalMembershipQA(), solver.readStats()]);
      const ptr = stats.adaptivePressureTopologyRepair;
      const attribution = stats.adaptivePressureTopologyAttribution;
      const pressureInspection = attribution
        ? inspectSparseCM12PressureCutoverAuthorities(
          attribution.authorities, attribution.inputTopologyGeneration)
        : undefined;
      const massFineCells = fields.density.reduce((sum, density) => sum + density, 0);
      const frame = { step,
        fields: Object.fromEntries(Object.entries(fields).map(([name, field]) =>
          [name, hash(field)])),
        massFineCells,
        topologySha256: hashJSON(acceptedTopologyIdentity(topology)), temporal,
        facePreparationTileCensus, fpaVexReadCensus,
        fca, srr: { header: srrHeader, work: srrWork }, vex, pcm: compactPCM(pcm),
        ptr: ptr ?? null,
        pressure: { attribution: attribution ?? null, inspection: pressureInspection ?? null },
      } satisfies FrameReceipt;
      frames.push(frame);
      const before = gateErrors.length;
      recordGate(temporal.mode === mode, { kind: "arm", arm: mode, step,
        field: "temporal.mode", expected: mode, actual: temporal.mode,
        message: `${mode} factory published ${temporal.mode}` });
      recordGate(temporal.seedCells <= temporal.firstDilationCells
        && temporal.firstDilationCells <= temporal.finalCells, {
        kind: "arm", arm: mode, step, field: "temporal.census",
        expected: "seedCells <= firstDilationCells <= finalCells", actual: temporal,
        message: `${mode} step ${step} temporal census is not monotonic`,
      });
      recordGate(facePreparationTileCensus.fault === 0, { kind: "arm", arm: mode,
        step, field: "facePreparationTileCensus.fault", expected: 0,
        actual: facePreparationTileCensus.fault,
        message: `${mode} step ${step} FPA tile census fault` });
      recordGate(facePreparationTileCensus.omittedChangedRowCount === 0, {
        kind: "arm", arm: mode, step,
        field: "facePreparationTileCensus.omittedChangedRowCount", expected: 0,
        actual: facePreparationTileCensus,
        message: `${mode} step ${step} tile selection omitted changed preparation rows`,
      });
      recordGate(fpaVexReadCensus.fault === 0, { kind: "arm", arm: mode, step,
        field: "fpaVexReadCensus.fault", expected: 0,
        actual: fpaVexReadCensus.fault,
        message: `${mode} step ${step} FVR1 census fault` });
      recordGate(fpaVexReadCensus.oracleMismatchRowCount === 0, {
        kind: "arm", arm: mode, step,
        field: "fpaVexReadCensus.oracleMismatchRowCount", expected: 0,
        actual: fpaVexReadCensus.oracleMismatchRowCount,
        message: `${mode} step ${step} production FPA differs from full oracle` });
      recordGate(fpaVexReadCensus.omittedChangedRowCount === 0, {
        kind: "arm", arm: mode, step,
        field: "fpaVexReadCensus.omittedChangedRowCount", expected: 0,
        actual: fpaVexReadCensus.omittedChangedRowCount,
        message: `${mode} step ${step} FVR1 C is not a subset of predicted S` });
      recordGate(fpaVexReadCensus.acceptedGeneration
        === fpaVexReadCensus.candidateGeneration, {
        kind: "arm", arm: mode, step, field: "fpaVexReadCensus.generation",
        expected: fpaVexReadCensus.candidateGeneration,
        actual: fpaVexReadCensus.acceptedGeneration,
        message: `${mode} step ${step} FVR1 candidate did not commit` });
      recordGate(fpaVexReadCensus.constructionBootstrapPublished === (step === 1), {
        kind: "arm", arm: mode, step,
        field: "fpaVexReadCensus.constructionBootstrapPublished",
        expected: step === 1, actual: fpaVexReadCensus.constructionBootstrapPublished,
        message: `${mode} step ${step} FVR1 construction epoch receipt is invalid` });
      recordGate(fca.fault === 0, { kind: "arm", arm: mode, step, field: "fca.fault",
        expected: 0, actual: fca.fault, message: `${mode} step ${step} FCA fault` });
      recordGate(srrHeader.fault === 0, { kind: "arm", arm: mode, step,
        field: "srr.fault", expected: 0, actual: srrHeader.fault,
        message: `${mode} step ${step} SRR fault` });
      recordGate(vex.faultCount === 0, { kind: "arm", arm: mode, step,
        field: "vex.faultCount", expected: 0, actual: vex.faultCount,
        message: `${mode} step ${step} VEX fault` });
      for (const domain of ["cell", "row"] as const) {
        recordGate(pcm[domain].phase === SPARSE_CM12_CANONICAL_MEMBERSHIP_PHASE.accepted,
          { kind: "arm", arm: mode, step, field: `pcm.${domain}.phase`,
            expected: SPARSE_CM12_CANONICAL_MEMBERSHIP_PHASE.accepted,
            actual: pcm[domain].phase,
            message: `${mode} step ${step} PCM ${domain} is not accepted` });
        recordGate(pcm[domain].fault === 0, { kind: "arm", arm: mode, step,
          field: `pcm.${domain}.fault`, expected: 0, actual: pcm[domain].fault,
          message: `${mode} step ${step} PCM ${domain} fault` });
        recordGate(pcm[domain].firstFault === PCM_INVALID, { kind: "arm", arm: mode,
          step, field: `pcm.${domain}.firstFault`, expected: PCM_INVALID,
          actual: pcm[domain].firstFault,
          message: `${mode} step ${step} PCM ${domain} retained a first fault` });
        recordGate(pcm[domain].candidateGeneration === pcm[domain].acceptedGeneration,
          { kind: "arm", arm: mode, step, field: `pcm.${domain}.generation`,
            expected: pcm[domain].acceptedGeneration,
            actual: pcm[domain].candidateGeneration,
            message: `${mode} step ${step} PCM ${domain} generation is uncommitted` });
        recordGate(pcm[domain].totalCount === pcm[domain].activeBitCount,
          { kind: "arm", arm: mode, step, field: `pcm.${domain}.activeBitCount`,
            expected: pcm[domain].totalCount, actual: pcm[domain].activeBitCount,
            message: `${mode} step ${step} PCM ${domain} count disagrees with authority bits` });
        if (domain === "row" || pcm.mode === "full-refresh-oracle") {
          recordGate(pcm[domain].matchesClassification, { kind: "arm", arm: mode, step,
            field: `pcm.${domain}.matchesClassification`, expected: true,
            actual: pcm[domain].matchesClassification,
            message: `${mode} step ${step} PCM ${domain} domain differs from classification` });
        }
      }
      recordGate(Boolean(ptr), { kind: "arm", arm: mode, step, field: "ptr",
        expected: "available", actual: ptr ?? null,
        message: `${mode} step ${step} PTR receipt unavailable` });
      if (ptr) recordGate(ptr.fault === 0, { kind: "arm", arm: mode, step,
        field: "ptr.fault", expected: 0, actual: ptr.fault,
        message: `${mode} step ${step} PTR fault` });
      recordGate(Boolean(attribution), { kind: "arm", arm: mode, step,
        field: "pressure.attribution", expected: "available", actual: null,
        message: `${mode} step ${step} pressure attribution unavailable` });
      if (pressureInspection) recordGate(pressureInspection.issues.length === 0, {
        kind: "arm", arm: mode, step, field: "pressure.issues", expected: [],
        actual: pressureInspection.issues,
        message: `${mode} step ${step} pressure receipt faults`,
      });
      requireCoverageComplete(frame, mode, step);
      if (gateErrors.length !== before) {
        throw new Error(`${mode} step ${step} failed ${gateErrors.length - before} gate(s)`);
      }
    }
  } finally {
    try {
      await gpuCompilationManagerFor(device).whenIdle();
    } finally {
      try {
        await device.queue.onSubmittedWorkDone();
      } finally {
        solver.destroy();
      }
    }
  }
  return { mode, frames: Object.freeze([...frames]) };
}

let device: GPUDevice | undefined;
let gpu: GPU | undefined;
let lockAcquired = false;
let current: ArmReceipt | undefined;
let change: ArmReceipt | undefined;
let operationError: unknown;
try {
  await acquireWebGPUExclusiveLock("temporal-seed-ab",
    `${sceneName} immutable paired seed probe`);
  lockAcquired = true;
  if (!currentFactory || !changeFactory) {
    throw new Error("temporal seed A/B requires both immutable QA factories; fallback forbidden");
  }
  const modulePath = process.env.WEBGPU_NODE_MODULE
    ?? `${process.cwd()}/node_modules/webgpu/index.js`;
  const dawn = await import(pathToFileURL(modulePath).href) as NodeDawnProvider;
  Object.assign(globalThis, dawn.globals);
  gpu = createProcessRetainedDawnGPU(dawn,
    [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert(adapter, "temporal seed A/B requires a WebGPU adapter");
  device = await adapter.requestDevice({
    requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  });
  device.addEventListener("uncapturederror", (event) => {
    event.preventDefault(); validationErrors.push(event.error.message);
  });
  // Sequential fresh solvers keep compilation/resident memory bounded.
  current = await runArm(device, "current", currentFactory);
  change = await runArm(device, "change", changeFactory);
  recordGate(current.frames.length === change.frames.length, {
    kind: "paired", field: "frames.length", expected: current.frames.length,
    actual: change.frames.length, message: "paired arm frame counts differ",
  });
  const compare = (actual: unknown, expected: unknown, step: number, field: string) => {
    try {
      assert.deepEqual(actual, expected);
    } catch {
      gateErrors.push({ kind: "paired", step, field, expected, actual,
        message: `${field} differs at step ${step}` });
    }
  };
  for (let frame = 0; frame < steps; frame += 1) {
    const left = current.frames[frame], right = change.frames[frame];
    if (!left || !right) continue;
    const step = frame + 1;
    compare(right.fields, left.fields, step, "physical fields");
    recordGate(right.massFineCells === left.massFineCells, {
      kind: "paired", step, field: "massFineCells", expected: left.massFineCells,
      actual: right.massFineCells, message: `mass differs at step ${step}` });
    recordGate(right.topologySha256 === left.topologySha256, {
      kind: "paired", step, field: "topologySha256", expected: left.topologySha256,
      actual: right.topologySha256,
      message: `accepted topology differs at step ${step}` });
    for (const receipt of ["fca", "srr", "vex", "pcm", "ptr", "pressure"] as const) {
      compare(stableAuthorityIdentity(right[receipt]),
        stableAuthorityIdentity(left[receipt]), step, `${receipt} stable authority`);
    }
  }
  if (gateErrors.length > 0) {
    throw new Error(`temporal seed A/B failed ${gateErrors.length} gate(s)`);
  }
} catch (error) {
  operationError = error;
} finally {
  if (device) {
    try {
      await gpuCompilationManagerFor(device).whenIdle();
    } catch (error) {
      lifecycleErrors.push(`compiler idle: ${serializedError(error).message}`);
    }
    try {
      await device.queue.onSubmittedWorkDone();
    } catch (error) {
      lifecycleErrors.push(`queue completion: ${serializedError(error).message}`);
    }
    try {
      invalidateGPUCompilationManager(device, "temporal seed A/B complete");
    } catch (error) {
      lifecycleErrors.push(`compiler invalidation: ${serializedError(error).message}`);
    }
    try {
      device.destroy();
    } catch (error) {
      lifecycleErrors.push(`device destroy: ${serializedError(error).message}`);
    }
  }
  if (gpu) {
    // Let Dawn's final ProcessEvents callback observe device retirement while
    // the process-retained GPU is still strongly reachable.
    await new Promise<void>((resolve) => setImmediate(resolve));
    gpu = undefined;
  }
  if (lockAcquired) {
    try {
      await releaseWebGPUExclusiveLock();
    } catch (error) {
      lifecycleErrors.push(`lock release: ${serializedError(error).message}`);
    }
  }
}

for (const message of validationErrors) gateErrors.push({
  kind: "validation", message: `WebGPU validation error: ${message}`, actual: message,
});
const currentReceipt: ArmReceipt = current ?? {
  mode: "current", frames: Object.freeze([...partialFrames.current]),
};
const changeReceipt: ArmReceipt = change ?? {
  mode: "change", frames: Object.freeze([...partialFrames.change]),
};
const workDeltas = Array.from({ length: Math.min(currentReceipt.frames.length,
  changeReceipt.frames.length) }, (_, index) => {
  const left = currentReceipt.frames[index]!, right = changeReceipt.frames[index]!;
  const currentWork = workObservations({ temporal: left.temporal, srr: left.srr,
    vex: left.vex, pcm: left.pcm, ptr: left.ptr, pressure: left.pressure });
  const changeWork = workObservations({ temporal: right.temporal, srr: right.srr,
    vex: right.vex, pcm: right.pcm, ptr: right.ptr, pressure: right.pressure });
  return { step: index + 1, current: currentWork, change: changeWork,
    deltaChangeMinusCurrent: observationDelta(currentWork, changeWork) };
});
const passed = operationError === undefined && gateErrors.length === 0
  && validationErrors.length === 0 && lifecycleErrors.length === 0
  && currentReceipt.frames.length === steps && changeReceipt.frames.length === steps;
const artifact = { passed, kind: "sparse-cm12-temporal-seed-ab",
  scene: sceneName, steps, fixedSpecialization: true, runtimeToggle: false,
  fpaVexCensus: {
    maximumTilesPerRow: SPARSE_CM12_FPA_VEX_READ_CENSUS_MAX_TILES_PER_ROW,
    capacityStatus: "provisional-census-cap-not-a-production-capacity-claim",
  },
  dt_s, simulatedDuration_s: steps * dt_s,
  fields: ["density", "gamma", "velocity", "pressure", "divergence"],
  methodValues, validationErrors, lifecycleErrors, gateErrors, workDeltas,
  current: currentReceipt, change: changeReceipt,
  ...(operationError === undefined ? {} : { error: serializedError(operationError) }) };
await mkdir(dirname(out), { recursive: true });
await writeFile(out, `${JSON.stringify(artifact, null, 2)}\n`);
console.log(JSON.stringify({ ...artifact, out }, null, 2));
if (!passed) {
  throw operationError instanceof Error ? operationError
    : new Error(`temporal seed A/B failed; evidence preserved at ${out}`);
}
