import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { unpackFineLevelSetPackedPhi } from
  "../lib/core/fine-levelset-packed-sample";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { sceneAtContainerExtents } from "../lib/core/scene-scale";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { SPARSE_CM12_ACTIVITY_POLICY } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

async function readBuffer(device: GPUDevice, source: GPUBuffer | GPUBufferBinding,
  size: number): Promise<Uint32Array> {
  const binding = "buffer" in source ? source : { buffer: source };
  const readback = device.createBuffer({
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(binding.buffer, binding.offset ?? 0, readback, 0, size);
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

async function readSurfaceCrossings(device: GPUDevice,
  solver: WebGPUAdaptiveMassSolver): Promise<Map<string, number>> {
  const source = solver.globalFineLevelSetSource;
  assert.equal(source.plan.brickResolution, 8);
  const pageCapacity = source.plan.maximumResidentBricks;
  const samplesPerBrick = source.plan.samplesPerBrick;
  const [worklist, metadata, samples] = await Promise.all([
    readBuffer(device, source.worklist, source.worklist.size),
    readBuffer(device, source.metadata, 16 * pageCapacity),
    readBuffer(device, source.samples, 4 * pageCapacity * samplesPerBrick),
  ]);
  const generation = worklist[0]!;
  const count = Math.min(worklist[1]!, pageCapacity);
  const phi = new Map<string, number>();
  for (let rank = 0; rank < count; rank += 1) {
    const page = worklist[7 + rank]!;
    const at = 4 * page;
    if (metadata[at] !== page || metadata[at + 2] !== generation) continue;
    const pageCoordinate = signedPageCoordinate(metadata[at + 1]!);
    for (let local = 0; local < samplesPerBrick; local += 1) {
      const z = Math.floor(local / 64);
      const remainder = local - 64 * z;
      const y = Math.floor(remainder / 8);
      const x = remainder - 8 * y;
      const q = [8 * pageCoordinate[0] + x, 8 * pageCoordinate[1] + y,
        8 * pageCoordinate[2] + z] as const;
      phi.set(q.join(","), unpackFineLevelSetPackedPhi(
        samples[page * samplesPerBrick + local]!,
      ));
    }
  }
  const crossings = new Map<string, number>();
  for (const [coordinate, lowerPhi] of phi) {
    const [x, y, z] = coordinate.split(",").map(Number) as [number, number, number];
    const upperPhi = phi.get(`${x},${y + 1},${z}`);
    if (upperPhi === undefined || (lowerPhi < 0) === (upperPhi < 0)) continue;
    const t = -lowerPhi / (upperPhi - lowerPhi);
    crossings.set(`${x},${z}`, y + 0.5 + t);
  }
  return crossings;
}

function maximumCrossingShift(left: Map<string, number>, right: Map<string, number>): {
  maximum: number;
  column: string;
  positions: readonly [number, number];
} {
  let maximum = 0;
  let maximumColumn = "";
  let positions: readonly [number, number] = [0, 0];
  let common = 0;
  for (const [column, position] of left) {
    const other = right.get(column);
    if (other === undefined) continue;
    common += 1;
    const shift = Math.abs(position - other);
    if (shift > maximum) {
      maximum = shift;
      maximumColumn = column;
      positions = [position, other];
    }
  }
  assert.ok(common > 0, "transition captures must share surface columns");
  return { maximum, column: maximumColumn, positions };
}

dawnTest("surface presentation stays fixed across forced B8-B4-B8 cutovers", {
  timeout: 60_000,
}, async () => {
  await acquireWebGPUExclusiveLock("dawn-test",
    "tests/sparse-cm12-surface-transition-continuity-dawn.test.ts");
  let device: GPUDevice | undefined;
  let solver: WebGPUAdaptiveMassSolver | undefined;
  try {
    const dawn = await import(pathToFileURL(dawnModule!).href) as {
      create(options: string[]): GPU;
      globals: Record<string, unknown>;
    };
    Object.assign(globalThis, dawn.globals);
    const gpu = dawn.create([
      `backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`,
      "enable-dawn-features=disable_blob_cache",
    ]);
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

    const source = sceneDocument(getSceneDefinition("water-box-tank-fill"));
    const scene = sceneAtContainerExtents(source, {
      width_m: 0.8,
      height_m: 0.8,
      depth_m: 0.8,
    });
    scene.rigidBodies = [];
    scene.container.fillFraction = 0.6;
    scene.voxelDomain.finestCellSize_m = 0.05;
    scene.fluid.gravity_m_s2 = { x: 0, y: 0, z: 0 };
    solver = await WebGPUAdaptiveMassSolver.createAsync(
      device, scene, "balanced", undefined, {
        resolutionMode: "all-fine",
        brickFineResolution: 8,
        presentationPageResolution: 8,
        surfaceFineRings: 1,
        timeStep: "paper",
        activityPolicy: {
          ...SPARSE_CM12_ACTIVITY_POLICY,
          forcedSurfaceResolutionForQA: 8,
          prepareBricksPerFrame: 256,
          demoteEpochs: 1,
        },
        gammaDiffusionEnabled: false,
        surfaceSharpeningEnabled: false,
        pressureIterations: 8,
      },
      () => {},
    );
    await solver.waitForSimulationReady();

    assert.equal(solver.advanceTo(CM12_PAPER_DT_S, []), true);
    await device.queue.onSubmittedWorkDone();
    const atB8 = await readSurfaceCrossings(device, solver);

    solver.setForcedSurfaceResolutionForQA(4);
    let coarse: Awaited<ReturnType<WebGPUAdaptiveMassSolver["readGPUActivityPolicy"]>>
      | undefined;
    let coarseStep = 0;
    for (let step = 2; step <= 4; step += 1) {
      assert.equal(solver.advanceTo(step * CM12_PAPER_DT_S, []), true);
      await device.queue.onSubmittedWorkDone();
      const snapshot = await solver.readGPUActivityPolicy();
      const surface = snapshot.bricks.filter((brick) => brick.active
        && (brick.reasons & 1) !== 0);
      if (surface.length > 0
        && surface.every((brick) => brick.acceptedResolution === 4)) {
        coarse = snapshot;
        coarseStep = step;
        break;
      }
    }
    assert.ok(coarse, "forced B4 transition did not commit");
    const coarseSurface = coarse.bricks.filter((brick) => brick.active
      && (brick.reasons & 1) !== 0);
    assert.ok(coarseSurface.length > 0,
      `no classified surface: ${coarse.bricks.filter((brick) => brick.active)
        .map((brick) => `${brick.coordinate.join(",")}:${brick.reasons}`)
        .join(" ")}`);
    assert.ok(coarseSurface.every((brick) => brick.acceptedResolution === 4));
    const atB4 = await readSurfaceCrossings(device, solver);

    solver.setForcedSurfaceResolutionForQA(8);
    assert.equal(solver.advanceTo((coarseStep + 1) * CM12_PAPER_DT_S, []), true);
    await device.queue.onSubmittedWorkDone();
    const fine = await solver.readGPUActivityPolicy();
    const fineSurface = fine.bricks.filter((brick) => brick.active
      && (brick.reasons & 1) !== 0);
    assert.ok(fineSurface.every((brick) => brick.acceptedResolution === 8));
    const backAtB8 = await readSurfaceCrossings(device, solver);

    const downShift = maximumCrossingShift(atB8, atB4);
    const upShift = maximumCrossingShift(atB4, backAtB8);
    assert.ok(downShift.maximum <= 0.05 && upShift.maximum <= 0.05,
      `surface jumped across rung cutover: B8->B4 ${downShift.maximum.toFixed(4)}`
        + ` at ${downShift.column} (${downShift.positions.map((value) => value.toFixed(4))}), `
        + `B4->B8 ${upShift.maximum.toFixed(4)} at ${upShift.column}`
        + ` (${upShift.positions.map((value) => value.toFixed(4))}) fine cells`);
    assert.deepEqual(validationErrors, []);
  } finally {
    solver?.destroy();
    device?.destroy();
    await releaseWebGPUExclusiveLock();
  }
});
