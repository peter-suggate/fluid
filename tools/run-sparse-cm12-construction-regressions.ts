#!/usr/bin/env node
/**
 * Standalone Dawn construction regression for the production Sparse CM12
 * B16/P16 ladder. This is intentionally not a node:test lane: it emits a
 * durable receipt that the production cutover can consume.
 *
 * The two cases cross the historical receiver-capacity boundary:
 * - CM12 Figure 6 occupies the 512-brick construction floor.
 * - Figure 9 at DETAIL x2 preserves the authored physical scene while filling
 *   a 16 x 16 x 8 = 2,048-brick B16 receiver domain.
 */
import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import type { SceneDescription } from "../lib/core/model";
import { getScenePreset } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import { SPARSE_CM12_VELOCITY_EXTENSION_HEADER } from
  "../lib/methods/adaptive-mass/sparse-cm12-velocity-extension";
import {
  initializeSparseBrickAtlasFromScene,
  sparseBrickAtlasStats,
} from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import {
  adaptiveMassPresentationDimensionsForScene,
  adaptiveMassReceiverScaleForScene,
  dormantReceiverDomain,
  residentSupportAtlas,
  WebGPUAdaptiveMassSolver,
} from "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const ROOT = resolve(new URL("..", import.meta.url).pathname);
const DEFAULT_DAWN_MODULE = `${ROOT}/node_modules/webgpu/index.js`;
const DEFAULT_OUTPUT = `${ROOT}/artifacts/sparse-cm12-construction-b16-p16-512-2048.json`;
const INVALID = 0xffff_ffff;

const argument = (name: string): string | undefined => process.argv.slice(2)
  .find((value) => value.startsWith(`--${name}=`))?.slice(name.length + 3);

if (process.argv.includes("--help") || process.argv.includes("-h")) {
  console.log(`Sparse CM12 standalone B16/P16 construction regression

Usage:
  WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \\
  node --import tsx tools/run-sparse-cm12-construction-regressions.ts [options]

Options:
  --backend=NAME  Dawn backend (default metal)
  --out=PATH      Receipt path (default artifacts/sparse-cm12-construction-b16-p16-512-2048.json)
  --help, -h      Print help without acquiring WebGPU

The lane constructs 512- and 2,048-receiver B16/P16 solvers, advances each
once at the CM12 paper step, and fails on validation, capacity, FCA, or VEX
receipt errors.`);
  process.exit(0);
}

const backend = argument("backend") ?? process.env.FLUID_WEBGPU_BACKEND ?? "metal";
const modulePath = process.env.WEBGPU_NODE_MODULE ?? DEFAULT_DAWN_MODULE;
const outputPath = resolve(argument("out") ?? DEFAULT_OUTPUT);

interface ConstructionCase {
  readonly id: "receiver-512" | "figure9-receiver-2048";
  readonly sourceScene: string;
  readonly expectedReceiverBricks: 512 | 2048;
  readonly createScene: () => SceneDescription;
}

function preset(id: string): SceneDescription {
  const value = getScenePreset(id);
  assert.equal(value.id, id, `scene preset ${id} is unavailable`);
  return value.create();
}

function figure9Detail2(): SceneDescription {
  const scene = preset("mass-conserving-figure-9-dam-break");
  scene.sceneId = "mass-conserving-figure-9-dam-break-detail-2-b16";
  scene.voxelDomain = {
    ...scene.voxelDomain,
    finestCellSize_m: scene.voxelDomain.finestCellSize_m / 2,
  };
  return scene;
}

const cases: readonly ConstructionCase[] = Object.freeze([{
  id: "receiver-512",
  sourceScene: "cm12-figure-6",
  expectedReceiverBricks: 512,
  createScene: () => preset("cm12-figure-6"),
}, {
  id: "figure9-receiver-2048",
  sourceScene: "mass-conserving-figure-9-dam-break / DETAIL x2",
  expectedReceiverBricks: 2048,
  createScene: figure9Detail2,
}]);

const solverOptions = Object.freeze({
  resolutionMode: "adaptive" as const,
  brickFineResolution: 16 as const,
  presentationPageResolution: 16 as const,
  surfaceFineRings: 1,
  receiverSupportRings: 9,
  receiverFloor: "auto" as const,
  timeStep: "paper" as const,
});

function preflight(scene: SceneDescription, expectedReceiverBricks: number) {
  const finestDimensions = adaptiveMassPresentationDimensionsForScene(scene);
  const initial = initializeSparseBrickAtlasFromScene(scene, {
    finestDimensions,
    brickFineResolution: solverOptions.brickFineResolution,
    surfaceFineRings: solverOptions.surfaceFineRings,
  });
  const supported = residentSupportAtlas(initial, solverOptions.resolutionMode);
  const scale = adaptiveMassReceiverScaleForScene(
    scene, solverOptions.receiverSupportRings,
  );
  const receivers = dormantReceiverDomain(supported, solverOptions.resolutionMode,
    scale.supportRings, solverOptions.receiverFloor, scale.minimumCapacityScale);
  const stats = sparseBrickAtlasStats(receivers);
  assert.equal(receivers.brickFineResolution, 16);
  assert.equal(stats.residentBrickCount, expectedReceiverBricks,
    `${scene.sceneId} receiver construction changed`);
  return {
    finestDimensions,
    brickDimensions: receivers.brickDimensions,
    initialBrickCount: initial.bricks.length,
    supportedBrickCount: supported.bricks.length,
    receiverScale: scale,
    receiverStats: stats,
  };
}

function bufferLayout(solver: WebGPUAdaptiveMassSolver) {
  const source = solver.globalFineLevelSetSource;
  return {
    solverAllocatedBytes: solver.info.allocatedBytes,
    packedCellCount: solver.info.cellCount,
    equivalentUniformCells: solver.info.equivalentUniformCells,
    fluidBrickCapacity: solver.info.fluidBrickCapacity,
    presentation: {
      maximumResidentBricks: source.plan.maximumResidentBricks,
      logicalBrickCount: source.plan.logicalBrickCount,
      brickResolution: source.plan.brickResolution,
      samplesPerBrick: source.plan.samplesPerBrick,
      payloadCapacityBytes: source.plan.payloadCapacityBytes,
      metadataCapacityBytes: source.plan.metadataCapacityBytes,
      worklistBytes: source.plan.worklistBytes,
      allocatedBytes: source.plan.allocatedBytes,
    },
    buffers: {
      paramsBytes: source.params.size,
      metadataBytes: source.metadata.size,
      worklistBytes: source.worklist.size,
      samplesBytes: source.samples.size,
      workABytes: source.workA.size,
      workBBytes: source.workB.size,
      rollbackBytes: source.rollbackSamples.size,
    },
  };
}

async function runCase(device: GPUDevice, fixture: ConstructionCase,
  allValidationErrors: string[]) {
  const scene = fixture.createScene();
  const cpu = preflight(scene, fixture.expectedReceiverBricks);
  const progress: Array<{ phase: string; taskId?: string; label: string;
    completed: number; total: number }> = [];
  const validationStart = allValidationErrors.length;
  let solver: WebGPUAdaptiveMassSolver | undefined;
  const constructionStarted = performance.now();
  try {
    solver = await WebGPUAdaptiveMassSolver.createAsync(
      device, scene, "balanced", undefined, solverOptions,
      (value) => progress.push({ phase: value.phase, taskId: value.taskId,
        label: value.label, completed: value.completed, total: value.total }),
    );
    await device.queue.onSubmittedWorkDone();
    const construction_ms = performance.now() - constructionStarted;
    const layout = bufferLayout(solver);
    assert.equal(layout.fluidBrickCapacity, fixture.expectedReceiverBricks);
    assert.equal(layout.presentation.maximumResidentBricks, fixture.expectedReceiverBricks);
    assert.equal(layout.presentation.brickResolution, 16);
    assert.equal(layout.presentation.samplesPerBrick, 16 ** 3);

    const advanceStarted = performance.now();
    assert.equal(solver.advanceTo(CM12_PAPER_DT_S, []), true,
      `${fixture.id} did not encode its first advance`);
    await device.queue.onSubmittedWorkDone();
    const advance_ms = performance.now() - advanceStarted;
    const [stats, fca, vex] = await Promise.all([
      solver.readStats(), solver.readFrameControlQA(), solver.readVelocityExtensionQA(),
    ]);
    const vh = SPARSE_CM12_VELOCITY_EXTENSION_HEADER;
    const laneValidationErrors = allValidationErrors.slice(validationStart);
    assert.deepEqual(laneValidationErrors, [], `${fixture.id} emitted WebGPU validation errors`);
    assert.equal(fca.fault, 0, `${fixture.id} FCA1 faulted`);
    assert.equal(fca.firstFaultOwner, INVALID, `${fixture.id} FCA1 recorded a fault owner`);
    assert.equal(fca.acceptedGeneration, 2, `${fixture.id} did not accept frame generation 2`);
    assert.equal(vex.header[vh.faultCount], 0, `${fixture.id} VEX2 fault count is nonzero`);
    return {
      id: fixture.id,
      sourceScene: fixture.sourceScene,
      scene: scene.sceneId,
      expectedReceiverBricks: fixture.expectedReceiverBricks,
      passed: true,
      cpu,
      construction: { duration_ms: construction_ms, progress, layout },
      advance: {
        dt_s: CM12_PAPER_DT_S,
        duration_ms: advance_ms,
        encodedSteps: stats.encodedSteps,
        completedTime_s: stats.completedTime_s,
        acceptedCells: stats.adaptiveAcceptedCellCount,
        acceptedRows: stats.adaptiveAcceptedRowCount,
        residentBricks: stats.fluidBrickResidentCount,
        fca,
        vex: {
          sourceFrameGeneration: vex.header[vh.sourceFrameGeneration],
          topologyGeneration: vex.header[vh.topologyGeneration],
          packetCapacity: vex.header[vh.packetCapacity],
          dispatchPacketCount: vex.dispatchPacketCount,
          validCellCount: vex.header[vh.validCellCount],
          emptyPacketCount: vex.header[vh.emptyPacketCount],
          faultCount: vex.header[vh.faultCount],
        },
      },
      validationErrors: laneValidationErrors,
    };
  } finally {
    solver?.destroy();
  }
}

const startedAt = new Date().toISOString();
const report: Record<string, unknown> = {
  kind: "sparse-cm12-construction-regression",
  version: 1,
  startedAt,
  backend,
  configuration: {
    brickFineResolution: 16,
    presentationPageResolution: 16,
    receiverCapacities: cases.map((value) => value.expectedReceiverBricks),
    advancesPerCase: 1,
  },
};
const validationErrors: string[] = [];
const lanes: unknown[] = [];
let device: GPUDevice | undefined;
let destroyingDevice = false;
let deviceLostBeforeDestroy: { reason: string; message: string } | undefined;
let failure: string | undefined;

await acquireWebGPUExclusiveLock("dawn-acceptance",
  "tools/run-sparse-cm12-construction-regressions.ts");
try {
  const dawn = await import(pathToFileURL(modulePath).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${backend}`]);
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter, `No Dawn adapter is available for backend ${backend}`);
  device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  device.addEventListener("uncapturederror", (event) => {
    event.preventDefault(); validationErrors.push(event.error.message);
  });
  void device.lost.then((info) => {
    if (!destroyingDevice) deviceLostBeforeDestroy = {
      reason: info.reason, message: info.message,
    };
  });
  for (const fixture of cases) lanes.push(await runCase(device, fixture, validationErrors));
  await device.queue.onSubmittedWorkDone();
  assert.deepEqual(validationErrors, []);
  assert.equal(deviceLostBeforeDestroy, undefined, "device was lost before teardown");
} catch (error) {
  failure = error instanceof Error ? `${error.name}: ${error.message}\n${error.stack ?? ""}`
    : String(error);
} finally {
  destroyingDevice = true;
  device?.destroy();
  await releaseWebGPUExclusiveLock();
}

Object.assign(report, {
  completedAt: new Date().toISOString(),
  passed: failure === undefined && validationErrors.length === 0
    && lanes.length === cases.length,
  lanes,
  validationErrors,
  deviceLostBeforeDestroy,
  failure,
});
await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
console.log(JSON.stringify(report, null, 2));
if (report.passed !== true) process.exitCode = 1;
