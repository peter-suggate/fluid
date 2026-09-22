#!/usr/bin/env node
/** Isolated production water extraction: native GPU pass times and queue-fenced
 * latency for fresh meshes versus retained meshes. Does not advance physics.
 * WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/benchmark-water-extraction-dawn.ts [--method=adaptive-volume] [--out=PATH]
 */
import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { pathToFileURL } from "node:url";
import { createSparseCM12LongDamBreakScene } from "../lib/core/scenes";
import { resolveMethodValues, type GPUSolverInstance } from "../lib/core/method-contract";
import { createGlobalFineLevelSetConsumerSource } from "../lib/core/octree-consumer-sampling";
import { GPUPassTimestampRecorder } from "../lib/core/performance-trace";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { RasterWaterPipeline } from "../lib/core/webgpu-water-pipeline";
import { adaptiveMassMethod } from "../lib/methods/adaptive-volume/method";
import { uniformVolumeMethod } from "../lib/methods/uniform/uniform-volume-method";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
const flag = (name: string) => process.argv.find(arg => arg.startsWith(`--${name}=`))?.split("=").slice(1).join("=");
const method = flag("method") === "adaptive-volume" ? adaptiveMassMethod : uniformVolumeMethod;
const outputPath = flag("out") ?? `artifacts/water-extraction-${method.id}.json`;
await acquireWebGPUExclusiveLock("dawn-benchmark", "water extraction");
let device: GPUDevice | undefined, solver: GPUSolverInstance | undefined, water: RasterWaterPipeline | undefined;
try {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? `${process.cwd()}/node_modules/webgpu/index.js`).href);
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits), requiredFeatures: ["timestamp-query"] });
  assert.ok(device);
  const errors: string[] = [];
  device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
  const scene = createSparseCM12LongDamBreakScene();
  solver = await method.createSolverAsync!(device, scene, "balanced", resolveMethodValues(method, "balanced", {}), undefined, () => {});
  await solver.waitForSimulationReady?.();
  const uniform = device.createBuffer({ size: 400, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const bodies = device.createBuffer({ size: 768, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const column = device.createTexture({ size: [1,1], format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING });
  const width = 640, height = 360;
  const output = device.createTexture({ size: [width,height], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT });
  water = new RasterWaterPipeline(device, "rgba8unorm", uniform, bodies);
  await water.initialize();
  water.setSceneOptics({ optics: scene.fluid.optics, directional: scene.lighting?.directional, grade: scene.lighting?.grade, terrain: scene.terrain, container: { width_m: scene.container.width_m, depth_m: scene.container.depth_m } });
  water.setVolume(solver.surfaceFieldTexture ?? solver.volumeTexture, solver.columnBaseTexture ?? column);
  water.setFluidDomain(solver.fluidDomain);
  const source = solver.globalFineLevelSetSource;
  if (source) water.setGlobalFineLevelSet(createGlobalFineLevelSetConsumerSource(source));
  water.setCoarseLevelSet(solver.coarseLevelSetSource);
  water.ensureSize(width, height);
  const packed = new Float32Array(100), span = scene.container.width_m;
  packed.set([width,height,0,0]); packed.set([.6*span,.65*span,1.3*span,0],4);
  packed.set([0,.38*scene.container.height_m,0,0],8);
  packed.set([span,scene.container.height_m,scene.container.depth_m,scene.container.height_m*scene.container.fillFraction],12);
  packed.set([0,scene.voxelDomain.finestCellSize_m,0,0],16);
  packed.set([solver.info.nx,solver.info.ny,solver.info.nz,solver.info.gridKind === "octree" ? 3 : 1],20); packed.set([0,.5,0,0],24);
  device.queue.writeBuffer(uniform,0,packed);
  const samples = [];
  for (const fresh of [true,false]) {
    for (let i = 0; i < 10; i++) {
      if (fresh) water.invalidateSurface();
      const recorder: GPUPassTimestampRecorder | undefined = i === 9 ? new GPUPassTimestampRecorder(device,128) : undefined;
      const begin = performance.now(), raw = device.createCommandEncoder(), encoder = recorder?.instrument(raw) ?? raw;
      const result: ReturnType<RasterWaterPipeline["encode"]> = water.encode(encoder,output,solver.info.nx,solver.info.ny,solver.info.nz,false,solver.info.maximumNeighborDelta ?? 0,0,undefined,undefined,false,"clear",i===0,undefined,true);
      assert.ok(result); assert.equal(result.surfaceUpdated,fresh);
      recorder?.resolve(raw);
      device.queue.submit([raw.finish()]);
      const submitted = performance.now();
      await device.queue.onSubmittedWorkDone();
      const complete = performance.now();
      await water.completeSurfaceDiagnostics();
      const trace = await recorder?.read(); recorder?.destroy();
      if (i >= 2) samples.push({ fresh, instrumented: Boolean(recorder), cpu_ms: submitted-begin, completion_ms: complete-submitted, total_ms: complete-begin, trace });
    }
  }
  assert.deepEqual(errors,[]);
  const result = { method: method.id, scene: scene.sceneId, dimensions: [solver.info.nx,solver.info.ny,solver.info.nz], scope: "production water pipeline only, 640x360, paused t=0; no SVO lighting or browser", diagnostics: water.surfaceRenderDiagnostics, samples };
  assert.ok(result.diagnostics && result.diagnostics.vertexCount > 0, "the benchmark must render a nonempty mesh");
  await mkdir(dirname(outputPath),{recursive:true}); await writeFile(outputPath,JSON.stringify(result,null,2));
  console.log(JSON.stringify(result));
} finally { water?.destroy(); solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock(); }
