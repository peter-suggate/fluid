#!/usr/bin/env node
/**
 * Native-Dawn production-render capture for the reported mini64/min-cell-8
 * surface case. The PNG comes from RasterWaterPipeline's real compact
 * classifier, scan, tetra emitter and optical composite; the adjacent JSON
 * proves that the frame consumed the current global-fine publication rather
 * than retained or volume fallback geometry.
 *
 * Usage:
 *   WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
 *   FLUID_WEBGPU_BACKEND=${WEBGPU_BACKEND:-metal} \
 *   node --import tsx tools/capture-sparse-cm12-mini64-presentation-dawn.ts
 *
 * Options:
 *   --out=PATH       PNG path
 *   --receipt=PATH   JSON receipt path
 */
import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { environmentIndex } from "../lib/core/environments";
import { resolveMethodValues, type GPUSolverInstance } from
  "../lib/core/method-contract";
import { createGlobalFineLevelSetConsumerSource } from
  "../lib/core/octree-consumer-sampling";
import { encodeRgbPng } from "../lib/core/png-codec";
import { createMinimalPowerDamBreak64Scene } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { RasterWaterPipeline } from "../lib/core/webgpu-water-pipeline";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod, adaptiveMassSolverOptions } from
  "../lib/methods/adaptive-mass/method";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const STEPS = 7;
const WIDTH = 640;
const HEIGHT = 360;

function argument(name: string, fallback: string): string {
  const prefix = `--${name}=`;
  return process.argv.slice(2).find((value) => value.startsWith(prefix))
    ?.slice(prefix.length) ?? fallback;
}

const pngPath = resolve(argument("out",
  "artifacts/sparse-cm12-mini64-min8-t0233333.png"));
const receiptPath = resolve(argument("receipt",
  "artifacts/sparse-cm12-mini64-min8-t0233333.json"));
const dawnModule = process.env.WEBGPU_NODE_MODULE
  ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));

function resolutionHistogram<T extends { readonly acceptedResolution: number }>(
  bricks: readonly T[],
): Readonly<Record<string, number>> {
  return Object.fromEntries([1, 2, 4, 8].map((resolution) => [String(resolution),
    bricks.filter((brick) => brick.acceptedResolution === resolution).length]));
}

function imageReceipt(rgb: Uint8Array) {
  let hash = 0x811c_9dc5;
  let nonBlackPixels = 0;
  let minimum = 255;
  let maximum = 0;
  for (let pixel = 0; pixel < WIDTH * HEIGHT; pixel += 1) {
    const offset = 3 * pixel;
    const red = rgb[offset]!;
    const green = rgb[offset + 1]!;
    const blue = rgb[offset + 2]!;
    if (red !== 0 || green !== 0 || blue !== 0) nonBlackPixels += 1;
    minimum = Math.min(minimum, red, green, blue);
    maximum = Math.max(maximum, red, green, blue);
    hash = Math.imul(hash ^ red, 0x0100_0193) >>> 0;
    hash = Math.imul(hash ^ green, 0x0100_0193) >>> 0;
    hash = Math.imul(hash ^ blue, 0x0100_0193) >>> 0;
  }
  return {
    width: WIDTH,
    height: HEIGHT,
    nonBlackPixels,
    minimumChannel: minimum,
    maximumChannel: maximum,
    rgbFnv1a: hash.toString(16).padStart(8, "0"),
  };
}

await acquireWebGPUExclusiveLock("dawn-probe",
  "tools/capture-sparse-cm12-mini64-presentation-dawn.ts");
let device: GPUDevice | undefined;
let solver: WebGPUAdaptiveMassSolver | undefined;
let pipeline: RasterWaterPipeline | undefined;
let uniformBuffer: GPUBuffer | undefined;
let bodyBuffer: GPUBuffer | undefined;
let output: GPUTexture | undefined;
let columnFallback: GPUTexture | undefined;
let frameReadback: GPUBuffer | undefined;
try {
  const dawn = await import(pathToFileURL(dawnModule).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter, "Dawn must expose a WebGPU adapter");
  device = await adapter.requestDevice({
    requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  });
  const uncapturedErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    event.preventDefault();
    uncapturedErrors.push(event.error.message);
  });
  device.pushErrorScope("validation");

  const scene = createMinimalPowerDamBreak64Scene();
  scene.duration_s = Math.max(scene.duration_s, STEPS * CM12_PAPER_DT_S);
  scene.fluid.refinementRegions = [{
    id: "mini64-production-render-whole-domain-min8",
    rule: "minimum-cell-size",
    minimumCellSize_cells: 8,
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
  const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
    resolutionMode: "adaptive",
    brickFineResolution: "8",
    presentationPageResolution: "8",
    surfaceFineRings: 1,
    timeStep: "paper",
  });
  solver = await WebGPUAdaptiveMassSolver.createCompiledTopologyTransport(
    device, scene, "balanced", undefined, adaptiveMassSolverOptions(values), () => {},
  );
  await solver.waitForSimulationReady();
  assert.deepEqual([solver.info.nx, solver.info.ny, solver.info.nz], [64, 64, 64]);
  for (let step = 1; step <= STEPS; step += 1) {
    while (!solver.advanceTo(step * CM12_PAPER_DT_S, [])) {
      await new Promise<void>((done) => setImmediate(done));
    }
    await device.queue.onSubmittedWorkDone();
  }

  const activity = await solver.readGPUActivityPolicy();
  const activeBricks = activity.bricks.filter((brick) => brick.active);
  const surfaceBricks = activeBricks.filter((brick) => (brick.reasons & 1) !== 0);
  assert.ok(surfaceBricks.length > 0, "the evolved dam must retain surface bricks");
  assert.ok(surfaceBricks.every((brick) => brick.acceptedResolution === 1),
    "every rendered surface brick must exercise the B8 scale-8 presentation branch");

  uniformBuffer = device.createBuffer({
    label: "Mini64 min8 capture uniforms",
    size: 400,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  bodyBuffer = device.createBuffer({
    label: "Mini64 min8 capture bodies",
    size: 12 * 64,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  output = device.createTexture({
    label: "Mini64 min8 production water frame",
    size: [WIDTH, HEIGHT],
    format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  columnFallback = device.createTexture({
    label: "Mini64 min8 non-column fallback",
    size: [1, 1],
    format: "r32float",
    usage: GPUTextureUsage.TEXTURE_BINDING,
  });
  pipeline = new RasterWaterPipeline(
    device, "rgba8unorm", uniformBuffer, bodyBuffer,
  );
  await pipeline.initialize();
  pipeline.setSceneOptics({
    optics: scene.fluid.optics,
    directional: scene.lighting?.directional,
    grade: scene.lighting?.grade,
    terrain: scene.terrain,
    container: {
      width_m: scene.container.width_m,
      depth_m: scene.container.depth_m,
    },
  });
  const presentationSolver = solver as unknown as GPUSolverInstance;
  pipeline.setVolume(presentationSolver.surfaceFieldTexture ?? solver.volumeTexture,
    presentationSolver.columnBaseTexture ?? columnFallback);
  const source = solver.globalFineLevelSetSource;
  const sourceGeneration = source.generation;
  pipeline.setGlobalFineLevelSet(createGlobalFineLevelSetConsumerSource(source));
  pipeline.setCoarseLevelSet(presentationSolver.coarseLevelSetSource);
  pipeline.ensureSize(WIDTH, HEIGHT);

  const span = Math.max(scene.container.width_m, scene.container.height_m,
    scene.container.depth_m);
  const packed = new Float32Array(100);
  packed.set([WIDTH, HEIGHT, solver.info.submittedTime_s ?? 0, 0], 0);
  packed.set([1.55 * span, 1.12 * span, 1.72 * span, 0], 4);
  packed.set([0, 0.38 * scene.container.height_m, 0,
    scene.container.top === "closed" ? 1 : 0], 8);
  packed.set([scene.container.width_m, scene.container.height_m,
    scene.container.depth_m, scene.container.height_m * scene.container.fillFraction], 12);
  packed.set([0, scene.voxelDomain.finestCellSize_m, 0, 0], 16);
  packed.set([solver.info.nx, solver.info.ny, solver.info.nz,
    solver.info.gridKind === "octree" ? 3 : 1], 20);
  packed.set([0, 0.5, 0, 0], 24);
  packed.set([environmentIndex(scene.environment ?? "default"),
    solver.info.lastDt_s ?? 0, solver.info.maxSpeed_m_s ?? 0, 0], 28);
  device.queue.writeBuffer(uniformBuffer, 0, packed);
  device.queue.writeBuffer(bodyBuffer, 0, new Uint8Array(12 * 64));

  const bytesPerRow = Math.ceil(WIDTH * 4 / 256) * 256;
  frameReadback = device.createBuffer({
    label: "Mini64 min8 production frame readback",
    size: bytesPerRow * HEIGHT,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder({
    label: "Capture mini64 min8 production water frame",
  });
  const encoded = pipeline.encode(
    encoder, output.createView(), solver.info.nx, solver.info.ny, solver.info.nz,
    false, solver.info.maximumNeighborDelta ?? 0, solver.info.encodedSteps ?? STEPS,
    undefined, undefined, true, "clear",
  );
  assert.ok(encoded, "production RasterWaterPipeline did not encode");
  assert.equal(encoded.surfaceUpdated, true,
    "the capture must extract a fresh surface, not draw retained geometry");
  assert.equal(encoded.surfaceDiagnosticsCaptured, true,
    "the capture must fence a source-matched renderer receipt");
  encoder.copyTextureToBuffer(
    { texture: output },
    { buffer: frameReadback, bytesPerRow, rowsPerImage: HEIGHT },
    [WIDTH, HEIGHT],
  );
  device.queue.submit([encoder.finish()]);
  const diagnostics = await pipeline.completeSurfaceDiagnostics();
  await device.queue.onSubmittedWorkDone();
  assert.ok(diagnostics, "production renderer did not return surface diagnostics");

  await frameReadback.mapAsync(GPUMapMode.READ);
  const mapped = new Uint8Array(frameReadback.getMappedRange());
  const rgb = new Uint8Array(WIDTH * HEIGHT * 3);
  for (let y = 0; y < HEIGHT; y += 1) for (let x = 0; x < WIDTH; x += 1) {
    const sourceOffset = y * bytesPerRow + 4 * x;
    const targetOffset = 3 * (x + WIDTH * y);
    rgb[targetOffset] = mapped[sourceOffset]!;
    rgb[targetOffset + 1] = mapped[sourceOffset + 1]!;
    rgb[targetOffset + 2] = mapped[sourceOffset + 2]!;
  }
  frameReadback.unmap();

  const presentation = await solver.readPresentationPageAllocatorReceiptQA();
  const validationError = await device.popErrorScope();
  const validationErrors = [
    ...(validationError ? [validationError.message] : []),
    ...uncapturedErrors,
  ];
  assert.equal(diagnostics.surfaceGeometrySource, "global-fine-coarse");
  assert.equal(diagnostics.globalFineCrossingPublished, true);
  assert.equal(diagnostics.presentationFallbackActive, false);
  assert.equal(diagnostics.globalFineAttachedGeneration, sourceGeneration);
  assert.equal(diagnostics.meshPublicationGeneration, sourceGeneration);
  assert.notEqual(diagnostics.globalFineAuthorityLatch, 0);
  assert.ok(diagnostics.vertexCount > 0);
  assert.ok(diagnostics.activeCubeCount > 0);
  assert.equal(presentation.faultCode, 0);
  assert.deepEqual(validationErrors, []);

  const receipt = {
    probe: "sparse-cm12-mini64-production-render",
    configuration: {
      scene: scene.sceneId,
      grid: [solver.info.nx, solver.info.ny, solver.info.nz],
      brickFineResolution: 8,
      presentationPageResolution: 8,
      minimumCellSize_cells: 8,
      refinementRegion: "whole-domain",
      timeStep: "paper",
      steps: STEPS,
      time_s: STEPS * CM12_PAPER_DT_S,
    },
    branchProof: {
      interpretation: "acceptedResolution 1 in B8 means cell scale 8",
      activeBricks: activeBricks.length,
      activeResolutionHistogram: resolutionHistogram(activeBricks),
      surfaceBricks: surfaceBricks.length,
      surfaceResolutionHistogram: resolutionHistogram(surfaceBricks),
      everySurfaceBrickUsesScale8: surfaceBricks.every(
        (brick) => brick.acceptedResolution === 1),
    },
    presentation,
    renderer: diagnostics,
    image: { path: pngPath, ...imageReceipt(rgb) },
    validationErrors,
  };
  assert.ok(receipt.image.nonBlackPixels > 0, "captured frame is entirely black");
  await Promise.all([
    mkdir(dirname(pngPath), { recursive: true }),
    mkdir(dirname(receiptPath), { recursive: true }),
  ]);
  await Promise.all([
    writeFile(pngPath, encodeRgbPng({ width: WIDTH, height: HEIGHT, rgb })),
    writeFile(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`, "utf8"),
  ]);
  process.stdout.write(`${JSON.stringify(receipt, null, 2)}\n`);
} finally {
  try {
    if (frameReadback?.mapState === "mapped") frameReadback.unmap();
    frameReadback?.destroy();
    pipeline?.destroy();
    output?.destroy();
    columnFallback?.destroy();
    uniformBuffer?.destroy();
    bodyBuffer?.destroy();
    solver?.destroy();
    if (device) {
      await device.queue.onSubmittedWorkDone().catch(() => undefined);
      try {
        device.destroy();
      } catch {
        // A failed native pipeline may already have retired its Dawn device.
      }
    }
  } finally {
    await releaseWebGPUExclusiveLock();
  }
}
