/** Export the production narrow-band FLIP contour for an offline Cycles render.
 * node --import tsx tools/blender/export-nbflip-water.ts --scene=alpine-lake-sunrise --time=1
 * OBJ coordinates are metres, Blender Z up: application (x,y,z) -> (x,-z,y).
 * No displacement, replacement surface, particle reconstruction or solver override.
 */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { sceneDocument } from "../../lib/core/scene-definition";
import { findSceneDefinition } from "../../lib/core/scenes";
import { resolveMethodValues } from "../../lib/core/method-contract";
import { managedGPUDevice } from "../../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../../lib/core/webgpu-device-limits";
import { RasterWaterPipeline } from "../../lib/core/webgpu-water-pipeline";
import { createProcessRetainedDawnGPU, paceDawnEventPolling } from "../../lib/harness/node-dawn-provider";
import { uniformNarrowBandMethod as method } from "../../lib/methods/uniform/uniform-narrow-band-method";
import type { WebGPUUniformReferenceSolver } from "../../lib/methods/uniform/webgpu-uniform-reference";
import { waterMeshObj } from "./nbflip-water-mesh";
import { initializeRigidBodies } from "../../lib/core/rigid-body";

const options = new Map(process.argv.slice(2).map(argument => {
  const match = /^--(scene|time|out|normals)=(.+)$/.exec(argument);
  assert.ok(match, `Unknown argument ${argument}; use --scene=id --time=seconds --out=file.obj --normals=none|optical`);
  return [match[1]!, match[2]!];
}));
const normalMode = options.get("normals") ?? "none";
assert.ok(normalMode === "none" || normalMode === "optical", "Normals mode must be none or optical");
const definition = findSceneDefinition(options.get("scene") ?? "alpine-lake-sunrise");
assert.ok(definition, "Unknown scene");
assert.equal(definition.methodProfile?.methodId, method.id);
const scene = structuredClone(sceneDocument(definition));
const bodies = initializeRigidBodies(scene.rigidBodies);
const quality = definition.methodProfile!.quality;
const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...definition.methodProfile!.overrides });
const dt = method.effectiveStep_s?.(scene, values) ?? scene.numerics.fixedDt_s;
const requestedTime_s = Number(options.get("time") ?? 1);
assert.ok(Number.isFinite(requestedTime_s) && requestedTime_s > 0 && requestedTime_s <= scene.duration_s);
const steps = Math.ceil(requestedTime_s / dt - 1e-9);
const output = resolve(options.get("out") ?? `artifacts/water-scenes/${definition.id}/water.obj`);
assert.ok(output.endsWith(".obj"));
mkdirSync(dirname(output), { recursive: true });
paceDawnEventPolling();
const dawn = await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);
Object.assign(globalThis, dawn.globals);
const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal"]).requestAdapter();
assert.ok(adapter);
const requiredFeatures = ["float32-filterable", "subgroups"].filter(feature => adapter.features.has(feature as GPUFeatureName)) as GPUFeatureName[];
const device = managedGPUDevice(await adapter.requestDevice({ requiredFeatures,
  requiredLimits: requiredFluidDeviceLimits(adapter.limits) }), { requireWorkerRealm: false });
const errors: string[] = [];
device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
let solver: WebGPUUniformReferenceSolver | undefined, water: RasterWaterPipeline | undefined;
const owned: GPUBuffer[] = [];
const started = performance.now();
try {
  let lastProgress = -Infinity;
  console.log(JSON.stringify({ phase: "initialize", scene: definition.id, steps, dt }));
  solver = await method.createSolverAsync!(device, scene, quality, values, undefined, label => {
    if (performance.now() - lastProgress < 10000) return;
    lastProgress = performance.now();
    console.log(JSON.stringify({ phase: "initialize", label, elapsed_ms: lastProgress - started }));
  }) as WebGPUUniformReferenceSolver;
  console.log(JSON.stringify({ phase: "solver-ready", scene: definition.id, steps, dt, allocatedBytes: solver.info.allocatedBytes }));
  for (let step = 1; step <= steps; step++) {
    assert.ok(solver.advanceTo(step * dt, bodies));
    await solver.awaitFrameCompletion();
    assert.equal(solver.info.simulationPipelineError, undefined);
    assert.deepEqual(errors, []);
    if (step % 30 === 0 || step === steps) console.log(JSON.stringify({ phase: "simulate", step, time_s: solver.info.completedTime_s }));
  }
  const buffer = (size: number, usage: number) => { const b = device.createBuffer({ size, usage }); owned.push(b); return b; };
  const uniforms = buffer(432, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const view = new Float32Array(108);
  const info = solver.info, tank = info.tankCells ?? [info.nx, info.ny, info.nz];
  view.set([scene.container.width_m, scene.container.height_m, scene.container.depth_m, 0], 12);
  view.set([...tank, 1], 20);
  device.queue.writeBuffer(uniforms, 0, view);
  water = new RasterWaterPipeline(device, "bgra8unorm", uniforms, buffer(1536, GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM));
  await water.initialize();
  const source = solver.denseLevelSetVolumeSource;
  assert.ok(source?.mixedFrame && source.mixedOwnership, "Must publish actual mixed NBFLIP state");
  water.setVolume(solver.surfaceFieldTexture, source);
  const encoder = device.createCommandEncoder();
  water.encodeDenseSurfaceExtractionForQA(encoder, info.nx, info.ny, info.nz, "windows");
  device.queue.submit([encoder.finish()]);
  const mesh = await water.readDenseSurfaceExtractionForQA();
  assert.equal(mesh.vertexCount, mesh.vertexAllocator, "Surface overflow: refuse partial water export");
  assert.ok(mesh.vertexCount > 0 && mesh.vertexCount % 3 === 0);
  assert.equal(mesh.vertices.length, mesh.vertexCount * 8);
  assert.deepEqual(errors, []);
  const { obj, bounds } = waterMeshObj(mesh.vertices, `Production ${method.id}: ${definition.id} at ${steps * dt}s`, { opticalNormals: normalMode === "optical" });
  writeFileSync(output, obj);
  const report = { sceneId: definition.id, methodId: method.id, quality, methodValues: values,
    requestedTime_s, completedTime_s: solver.info.completedTime_s, dt, steps,
    cellSize_m: scene.voxelDomain?.finestCellSize_m, sceneHash: createHash("sha256").update(JSON.stringify(scene)).digest("hex"),
    vertexCount: mesh.vertexCount, triangleCount: mesh.vertexCount / 3, activeCubeCount: mesh.activeCubeCount,
    boundsBlender_m: bounds, objSha256: createHash("sha256").update(obj).digest("hex"),
    output, elapsed_ms: performance.now() - started, errors,
    rigidBodies: (await solver.readRigidBodyPoses())?.map((pose, index) => ({ ...pose,
      id: scene.rigidBodies[index]?.id, appearance: scene.rigidBodies[index]?.appearance })),
    opticalNormals: normalMode === "optical" ? "Production phi-gradient vertex lane, normalized exactly as the vertex shader and rotated to Blender axes" : "Not exported; legacy position-only OBJ",
    extraction: "Production RasterWaterPipeline mixed-level-set window contour; no authored water geometry" };
  writeFileSync(output.replace(/\.obj$/, ".json"), JSON.stringify(report, null, 2) + "\n");
  console.log(JSON.stringify(report));
} finally {
  water?.destroy(); solver?.destroy(); owned.forEach(buffer => buffer.destroy());
  await device.queue.onSubmittedWorkDone(); device.destroy();
}
