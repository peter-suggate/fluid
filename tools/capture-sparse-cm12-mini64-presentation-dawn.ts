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
import { float16BitsToFloat32 } from "../lib/core/fine-levelset-packed-sample";
import { resolveMethodValues, type GPUSolverInstance } from
  "../lib/core/method-contract";
import { createGlobalFineLevelSetConsumerSource } from
  "../lib/core/octree-consumer-sampling";
import { encodeRgbPng } from "../lib/core/png-codec";
import { createCornerBrickDropScene, createMinimalPowerDamBreak64Scene,
  createSparseCM12LongDamBreakScene } from "../lib/core/scenes";
import { solidVoxelShellForScene } from "../lib/core/scene-lattice";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { RasterWaterPipeline } from "../lib/core/webgpu-water-pipeline";
import { GLOBAL_FINE_HEIGHTFIELD_DESCRIPTOR_CODE } from
  "../lib/core/webgpu-water-global-fine-classify";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import { rasterMeshSymmetryMetrics } from
  "../lib/harness/raster-mesh-symmetry";
import { adaptiveMassMethod, adaptiveMassSolverOptions } from
  "../lib/methods/adaptive-mass/method";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const CAPTURE_SCENARIO = process.env.FLUID_PRESENTATION_CAPTURE_SCENARIO ?? "dam";
const LONG_DAM = CAPTURE_SCENARIO === "long-dam";
const CORNER_DROP = CAPTURE_SCENARIO === "corner-drop";
const GEOMETRY_AUDIT = CORNER_DROP
  || process.env.FLUID_PRESENTATION_CAPTURE_GEOMETRY_AUDIT === "1";
const STEPS = Number(process.env.FLUID_PRESENTATION_CAPTURE_STEPS ?? 7);
const MAXIMUM_CELL_SIZE = Number(process.env.FLUID_MAXIMUM_CELL_SIZE ?? 0);
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
let classifiedCubeReadback: GPUBuffer | undefined;
let classifiedOffsetReadback: GPUBuffer | undefined;
let surfaceVertexReadback: GPUBuffer | undefined;
let interfacePositionReadback: GPUBuffer | undefined;
let backInterfacePositionReadback: GPUBuffer | undefined;
let interfaceNormalReadback: GPUBuffer | undefined;
let backInterfaceNormalReadback: GPUBuffer | undefined;
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

  const scene = LONG_DAM ? createSparseCM12LongDamBreakScene()
    : CORNER_DROP ? createCornerBrickDropScene()
    : createMinimalPowerDamBreak64Scene();
  if (CORNER_DROP) {
    scene.solidVoxels = [...solidVoxelShellForScene(scene), ...scene.solidVoxels];
  }
  scene.duration_s = Math.max(scene.duration_s, STEPS * CM12_PAPER_DT_S);
  if (!CORNER_DROP) scene.fluid.refinementRegions = [{
    id: MAXIMUM_CELL_SIZE > 0 ? "production-render-whole-domain-max1"
      : "mini64-production-render-whole-domain-min8",
    rule: "minimum-cell-size",
    minimumCellSize_cells: MAXIMUM_CELL_SIZE > 0 ? 1 : 8,
    ...(MAXIMUM_CELL_SIZE > 0
      ? { maximumCellSize_cells: MAXIMUM_CELL_SIZE } : {}),
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
    ...(LONG_DAM ? { finestTravelCells: 4, fourTravelCells: 2,
      twoTravelCells: 1 } : { surfaceFineRings: 1 }),
    timeStep: "paper",
  });
  solver = await WebGPUAdaptiveMassSolver.createCompiledTopologyTransport(
    device, scene, "balanced", undefined, adaptiveMassSolverOptions(values), () => {},
  );
  await solver.waitForSimulationReady();
  assert.deepEqual([solver.info.nx, solver.info.ny, solver.info.nz],
    LONG_DAM ? [192, 96, 32] : CORNER_DROP ? [24, 16, 24] : [64, 64, 64]);
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
  if (!CORNER_DROP) {
    assert.ok(surfaceBricks.every((brick) => brick.acceptedResolution
        === (MAXIMUM_CELL_SIZE === 1 ? 8 : 1)),
      MAXIMUM_CELL_SIZE === 1
        ? "every rendered surface brick must remain fully fine"
        : "every rendered surface brick must exercise the B8 scale-8 presentation branch");
  }

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
  packed.set(LONG_DAM
    ? [0.52 * span, 0.37 * span, 0.57 * span, 0]
    : CORNER_DROP ? [0.82 * span, 0.58 * span, 0.9 * span, 0]
    : [1.55 * span, 1.12 * span, 1.72 * span, 0], 4);
  packed.set([0, (CORNER_DROP ? 0.14 : 0.38) * scene.container.height_m, 0,
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
  const meshSource = GEOMETRY_AUDIT
    ? pipeline.diagnosticSurfaceVertexSource() : undefined;
  if (GEOMETRY_AUDIT) {
    assert.ok(meshSource, "production renderer did not expose its surface mesh");
    classifiedCubeReadback = device.createBuffer({
      label: "Mini64 classified-cube readback",
      size: meshSource.classifiedCubes.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    classifiedOffsetReadback = device.createBuffer({
      label: "Mini64 classified-offset readback",
      size: meshSource.classifiedOffsets.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    surfaceVertexReadback = device.createBuffer({
      label: "Mini64 surface-vertex readback",
      size: meshSource.buffer.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.copyBufferToBuffer(meshSource.classifiedCubes, 0,
      classifiedCubeReadback, 0, meshSource.classifiedCubes.size);
    encoder.copyBufferToBuffer(meshSource.classifiedOffsets, 0,
      classifiedOffsetReadback, 0, meshSource.classifiedOffsets.size);
    encoder.copyBufferToBuffer(meshSource.buffer, 0,
      surfaceVertexReadback, 0, meshSource.buffer.size);
    const interfaceSource = pipeline.diagnosticCaptureTexture("interface-positions");
    assert.ok(interfaceSource, "production renderer did not expose front positions");
    const interfaceBytesPerRow = Math.ceil(WIDTH * 16 / 256) * 256;
    interfacePositionReadback = device.createBuffer({
      label: "Mini64 front-interface readback",
      size: interfaceBytesPerRow * HEIGHT,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.copyTextureToBuffer({ texture: interfaceSource.texture },
      { buffer: interfacePositionReadback, bytesPerRow: interfaceBytesPerRow,
        rowsPerImage: HEIGHT }, [WIDTH, HEIGHT]);
    const backInterfaceSource = pipeline.diagnosticCaptureTexture(
      "back-interface-positions");
    assert.ok(backInterfaceSource, "production renderer did not expose back positions");
    backInterfacePositionReadback = device.createBuffer({
      label: "Mini64 back-interface readback",
      size: interfaceBytesPerRow * HEIGHT,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.copyTextureToBuffer({ texture: backInterfaceSource.texture },
      { buffer: backInterfacePositionReadback, bytesPerRow: interfaceBytesPerRow,
        rowsPerImage: HEIGHT }, [WIDTH, HEIGHT]);
    const interfaceNormalSource = pipeline.diagnosticCaptureTexture("interfaces");
    assert.ok(interfaceNormalSource, "production renderer did not expose front normals");
    const normalBytesPerRow = Math.ceil(WIDTH * 8 / 256) * 256;
    interfaceNormalReadback = device.createBuffer({
      label: "Mini64 front-interface normal readback",
      size: normalBytesPerRow * HEIGHT,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.copyTextureToBuffer({ texture: interfaceNormalSource.texture },
      { buffer: interfaceNormalReadback, bytesPerRow: normalBytesPerRow,
        rowsPerImage: HEIGHT }, [WIDTH, HEIGHT]);
    const backInterfaceNormalSource = pipeline.diagnosticCaptureTexture(
      "back-interfaces");
    assert.ok(backInterfaceNormalSource, "production renderer did not expose back normals");
    backInterfaceNormalReadback = device.createBuffer({
      label: "Mini64 back-interface normal readback",
      size: normalBytesPerRow * HEIGHT,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.copyTextureToBuffer({ texture: backInterfaceNormalSource.texture },
      { buffer: backInterfaceNormalReadback, bytesPerRow: normalBytesPerRow,
        rowsPerImage: HEIGHT }, [WIDTH, HEIGHT]);
  }
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

  let geometry: Record<string, unknown> | undefined;
  let tallestCubeReceipt: readonly unknown[] | undefined;
  let meshAudit: Record<string, unknown> | undefined;
  let tolerantMeshAudit: Record<string, unknown> | undefined;
  let interfacePixels: Record<string, number> | undefined;
  let interfaceNormals: Record<string, unknown> | undefined;
  if (meshSource && classifiedCubeReadback && classifiedOffsetReadback
    && surfaceVertexReadback) {
    await Promise.all([
    classifiedCubeReadback.mapAsync(GPUMapMode.READ),
    classifiedOffsetReadback.mapAsync(GPUMapMode.READ),
    surfaceVertexReadback.mapAsync(GPUMapMode.READ),
    ]);
  const classifiedCubes = new Uint32Array(classifiedCubeReadback.getMappedRange());
  const classifiedOffsets = new Uint32Array(classifiedOffsetReadback.getMappedRange());
  const surfaceVertices = new Float32Array(surfaceVertexReadback.getMappedRange());
  const descriptorGroups = new Map<string, {
    cubes: number; vertices: number; degenerateTriangles: number;
    minimum: number[]; maximum: number[];
  }>();
  const tallestCubes: Array<{
    kind: string; base: number[]; descriptor: number; vertices: number;
    minimumY: number; maximumY: number;
  }> = [];
  const signed16 = (word: number) => (word << 16) >> 16;
  for (let cube = 0; cube < diagnostics.activeCubeCount; cube += 1) {
    const descriptor = classifiedCubes[2 * cube + 1]! >>> 16;
    const rawCode = descriptor & 255;
    const kind = rawCode === GLOBAL_FINE_HEIGHTFIELD_DESCRIPTOR_CODE ? "height-field"
      : rawCode >= 224 ? `wall-axis-${descriptor >>> 14 & 3}`
      : rawCode & 128 ? "transition-volume" : "regular-volume";
    const group = descriptorGroups.get(kind) ?? {
      cubes: 0, vertices: 0, degenerateTriangles: 0,
      minimum: [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY],
      maximum: [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY],
    };
    group.cubes += 1;
    const first = classifiedOffsets[6 * cube]!;
    const end = cube + 1 < diagnostics.activeCubeCount
      ? classifiedOffsets[6 * (cube + 1)]! : diagnostics.vertexCount;
    group.vertices += end - first;
    let minimumY = Number.POSITIVE_INFINITY;
    let maximumY = Number.NEGATIVE_INFINITY;
    for (let vertex = first; vertex < end; vertex += 1) {
      const offset = vertex * meshSource.strideBytes / 4;
      for (let axis = 0; axis < 3; axis += 1) {
        group.minimum[axis] = Math.min(group.minimum[axis]!, surfaceVertices[offset + axis]!);
        group.maximum[axis] = Math.max(group.maximum[axis]!, surfaceVertices[offset + axis]!);
      }
      minimumY = Math.min(minimumY, surfaceVertices[offset + 1]!);
      maximumY = Math.max(maximumY, surfaceVertices[offset + 1]!);
    }
    for (let vertex = first; vertex + 2 < end; vertex += 3) {
      const stride = meshSource.strideBytes / 4;
      const a = vertex * stride, b = (vertex + 1) * stride, c = (vertex + 2) * stride;
      const ab = [surfaceVertices[b]! - surfaceVertices[a]!,
        surfaceVertices[b + 1]! - surfaceVertices[a + 1]!,
        surfaceVertices[b + 2]! - surfaceVertices[a + 2]!];
      const ac = [surfaceVertices[c]! - surfaceVertices[a]!,
        surfaceVertices[c + 1]! - surfaceVertices[a + 1]!,
        surfaceVertices[c + 2]! - surfaceVertices[a + 2]!];
      const cross = [ab[1]! * ac[2]! - ab[2]! * ac[1]!,
        ab[2]! * ac[0]! - ab[0]! * ac[2]!,
        ab[0]! * ac[1]! - ab[1]! * ac[0]!];
      const areaSquared = cross.reduce((sum, value) => sum + value * value, 0);
      group.degenerateTriangles += Number(!(areaSquared > 0)
        || !Number.isFinite(areaSquared));
    }
    if (end > first) tallestCubes.push({ kind,
      base: [signed16(classifiedCubes[2 * cube]!),
        classifiedCubes[2 * cube + 1]! & 0xffff,
        signed16(classifiedCubes[2 * cube]! >>> 16)],
      descriptor, vertices: end - first, minimumY, maximumY });
    descriptorGroups.set(kind, group);
  }
  if (interfacePositionReadback && backInterfacePositionReadback) {
    await Promise.all([interfacePositionReadback.mapAsync(GPUMapMode.READ),
      backInterfacePositionReadback.mapAsync(GPUMapMode.READ)]);
    const interfaceBytesPerRow = Math.ceil(WIDTH * 16 / 256) * 256;
    const frontPositions = new Float32Array(interfacePositionReadback.getMappedRange());
    const backPositions = new Float32Array(backInterfacePositionReadback.getMappedRange());
    let front = 0, back = 0;
    for (let y = 0; y < HEIGHT; y += 1) for (let x = 0; x < WIDTH; x += 1) {
      const alpha = (y * interfaceBytesPerRow >> 2) + 4 * x + 3;
      front += Number(frontPositions[alpha]! !== 0);
      back += Number(backPositions[alpha]! !== 0);
    }
    interfacePixels = { front, back, total: WIDTH * HEIGHT };
    interfacePositionReadback.unmap();
    backInterfacePositionReadback.unmap();
  }
  if (interfaceNormalReadback && backInterfaceNormalReadback) {
    await Promise.all([interfaceNormalReadback.mapAsync(GPUMapMode.READ),
      backInterfaceNormalReadback.mapAsync(GPUMapMode.READ)]);
    const normalBytesPerRow = Math.ceil(WIDTH * 8 / 256) * 256;
    const summarizeNormals = (mappedRange: ArrayBuffer) => {
      const words = new Uint16Array(mappedRange);
      let occupied = 0, zeroLength = 0, nonFinite = 0;
      let positiveY = 0, negativeY = 0;
      let sumX = 0, sumY = 0, sumZ = 0, sumLength = 0;
      let minimumLength = Number.POSITIVE_INFINITY;
      let maximumLength = 0;
      let rawHash = 0x811c_9dc5;
      for (let y = 0; y < HEIGHT; y += 1) for (let x = 0; x < WIDTH; x += 1) {
        const base = (y * normalBytesPerRow >> 1) + 4 * x;
        const alpha = float16BitsToFloat32(words[base + 3]!);
        if (alpha === 0) continue;
        occupied += 1;
        const nx = float16BitsToFloat32(words[base]!);
        const ny = float16BitsToFloat32(words[base + 1]!);
        const nz = float16BitsToFloat32(words[base + 2]!);
        const length = Math.hypot(nx, ny, nz);
        if (!Number.isFinite(nx + ny + nz + length + alpha)) nonFinite += 1;
        else {
          zeroLength += Number(length < 1e-6);
          positiveY += Number(ny > 0);
          negativeY += Number(ny < 0);
          sumX += nx; sumY += ny; sumZ += nz; sumLength += length;
          minimumLength = Math.min(minimumLength, length);
          maximumLength = Math.max(maximumLength, length);
        }
        for (let channel = 0; channel < 4; channel += 1) {
          const word = words[base + channel]!;
          rawHash = Math.imul(rawHash ^ (word & 255), 0x0100_0193) >>> 0;
          rawHash = Math.imul(rawHash ^ (word >>> 8), 0x0100_0193) >>> 0;
        }
      }
      const divisor = Math.max(1, occupied - nonFinite);
      return { occupied, zeroLength, nonFinite, positiveY, negativeY,
        mean: [sumX / divisor, sumY / divisor, sumZ / divisor],
        meanLength: sumLength / divisor,
        minimumLength: Number.isFinite(minimumLength) ? minimumLength : 0,
        maximumLength,
        rawFnv1a: rawHash.toString(16).padStart(8, "0") };
    };
    interfaceNormals = {
      front: summarizeNormals(interfaceNormalReadback.getMappedRange()),
      back: summarizeNormals(backInterfaceNormalReadback.getMappedRange()),
    };
    interfaceNormalReadback.unmap();
    backInterfaceNormalReadback.unmap();
  }
  geometry = Object.fromEntries([...descriptorGroups].map(([kind, group]) =>
    [kind, { ...group, minimum: group.minimum.map(value => Number(value.toFixed(6))),
      maximum: group.maximum.map(value => Number(value.toFixed(6))) }]));
  tallestCubes.sort((a, b) => b.maximumY - a.maximumY);
  const fullMeshAudit = rasterMeshSymmetryMetrics(surfaceVertices,
    diagnostics.vertexCount, {
      minimum: [-0.5 * scene.container.width_m, 0,
        -0.5 * scene.container.depth_m],
      maximum: [0.5 * scene.container.width_m, scene.container.height_m,
        0.5 * scene.container.depth_m],
      tolerance: 1e-6,
    }, { cubes: classifiedCubes, offsets: classifiedOffsets,
      cubeCount: diagnostics.activeCubeCount });
  assert.equal(fullMeshAudit.interiorOpenEdgeCount, 0,
    "height/volume handoff left an unmatched interior mesh edge");
  assert.equal(fullMeshAudit.nonManifoldEdgeCount, 0,
    "surface extraction emitted a non-manifold mesh edge");
  meshAudit = { ...fullMeshAudit,
    interiorOpenEdges: fullMeshAudit.interiorOpenEdges?.slice(0, 24) };
  tolerantMeshAudit = Object.fromEntries([1e-7, 1e-6, 1e-5, 1e-4].map(
    (tolerance) => {
      const rounded = surfaceVertices.slice(0,
        diagnostics.vertexCount * meshSource.strideBytes / 4);
      for (let vertex = 0; vertex < diagnostics.vertexCount; vertex += 1) {
        for (let axis = 0; axis < 3; axis += 1) {
          const at = vertex * meshSource.strideBytes / 4 + axis;
          rounded[at] = Math.round(rounded[at]! / tolerance) * tolerance;
        }
      }
      const audit = rasterMeshSymmetryMetrics(rounded, diagnostics.vertexCount, {
        minimum: [-0.5 * scene.container.width_m, 0,
          -0.5 * scene.container.depth_m],
        maximum: [0.5 * scene.container.width_m, scene.container.height_m,
          0.5 * scene.container.depth_m],
        tolerance: 1.5 * tolerance,
      });
      return [String(tolerance), { openEdgeCount: audit.openEdgeCount,
        boundaryOpenEdgeCount: audit.boundaryOpenEdgeCount,
        interiorOpenEdgeCount: audit.interiorOpenEdgeCount,
        degenerateTriangleCount: audit.degenerateTriangleCount,
        nonManifoldEdgeCount: audit.nonManifoldEdgeCount }];
    }));
  tallestCubeReceipt = tallestCubes.slice(0, 24);
  classifiedCubeReadback.unmap();
  classifiedOffsetReadback.unmap();
  surfaceVertexReadback.unmap();
  }

  const receipt = {
    probe: "sparse-cm12-mini64-production-render",
    configuration: {
      scene: scene.sceneId,
      grid: [solver.info.nx, solver.info.ny, solver.info.nz],
      brickFineResolution: 8,
      presentationPageResolution: 8,
      minimumCellSize_cells: MAXIMUM_CELL_SIZE > 0 ? 1 : 8,
      maximumCellSize_cells: MAXIMUM_CELL_SIZE || undefined,
      refinementRegion: CORNER_DROP ? "none" : "whole-domain",
      timeStep: "paper",
      steps: STEPS,
      time_s: STEPS * CM12_PAPER_DT_S,
    },
    branchProof: {
      interpretation: MAXIMUM_CELL_SIZE === 1
        ? "acceptedResolution 8 in B8 means cell scale 1"
        : "acceptedResolution 1 in B8 means cell scale 8",
      activeBricks: activeBricks.length,
      activeResolutionHistogram: resolutionHistogram(activeBricks),
      surfaceBricks: surfaceBricks.length,
      surfaceResolutionHistogram: resolutionHistogram(surfaceBricks),
      everySurfaceBrickUsesExpectedScale: CORNER_DROP ? undefined
        : surfaceBricks.every(
          (brick) => brick.acceptedResolution === (MAXIMUM_CELL_SIZE === 1 ? 8 : 1)),
    },
    presentation,
    renderer: diagnostics,
    ...(geometry ? { geometry } : {}),
    ...(tallestCubeReceipt ? { tallestCubes: tallestCubeReceipt } : {}),
    ...(meshAudit ? { meshAudit } : {}),
    ...(tolerantMeshAudit ? { tolerantMeshAudit } : {}),
    ...(interfacePixels ? { interfacePixels } : {}),
    ...(interfaceNormals ? { interfaceNormals } : {}),
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
    classifiedCubeReadback?.destroy();
    classifiedOffsetReadback?.destroy();
    surfaceVertexReadback?.destroy();
    interfacePositionReadback?.destroy();
    backInterfacePositionReadback?.destroy();
    interfaceNormalReadback?.destroy();
    backInterfaceNormalReadback?.destroy();
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
