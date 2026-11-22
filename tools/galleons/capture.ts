// Capture the native narrow-band presentation field for the companion Blender still.
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { managedGPUDevice } from "../../lib/core/gpu-compilation-manager";
import { resolveMethodValues } from "../../lib/core/method-contract";
import { initializeRigidBodies } from "../../lib/core/rigid-body";
import { sceneDocument } from "../../lib/core/scene-definition";
import { findSceneDefinition } from "../../lib/core/scenes";
import { ProjectileSimulation } from "../../lib/core/simulation/projectiles";
import { requiredFluidDeviceLimits } from "../../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU, paceDawnEventPolling } from "../../lib/harness/node-dawn-provider";
import { uniformNarrowBandMethod as method } from "../../lib/methods/uniform/uniform-narrow-band-method";
import { registerSceneMeshAssets } from "../blender/register-assets";
import { galleons } from "../../lib/core/galleon-layout";

const out = process.argv[2];
assert.ok(out, "Supply a capture directory");
mkdirSync(out, { recursive: true });
const definition = findSceneDefinition("galleon-broadside")!;
const scene = sceneDocument(definition);
await registerSceneMeshAssets(scene);
const bodies = initializeRigidBodies(scene.rigidBodies);
const shots = new ProjectileSimulation();
const profile = definition.methodProfile!;
const values = resolveMethodValues(method, profile.quality, { ...method.appDefaults, ...profile.overrides });
paceDawnEventPolling();
const dawn = await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);
Object.assign(globalThis, dawn.globals);
const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal"]).requestAdapter();
assert.ok(adapter);
const device = managedGPUDevice(await adapter.requestDevice({
  requiredFeatures: ["float32-filterable", "subgroups"].filter(f => adapter.features.has(f as GPUFeatureName)) as GPUFeatureName[],
  requiredLimits: requiredFluidDeviceLimits(adapter.limits),
}), { requireWorkerRealm: false });
const solver = await method.createSolverAsync!(device, scene, profile.quality, values, undefined, () => {});
try {
  const dt = scene.numerics.fixedDt_s;
  // Frame the first volley in flight, before the first breach.
  for (let i = 1; i <= 44; i++) {
    assert.deepEqual(shots.step(scene, bodies, i * dt, dt), []);
    assert.ok(solver.advanceTo(i * dt, bodies));
    await solver.awaitFrameCompletion!();
    assert.equal(solver.info.simulationPipelineError, undefined);
  }
  const field = solver.denseLevelSetVolumeSource?.vertexPhi;
  assert.ok(field, "Missing production vertex level set");
  const dims = [field.width, field.height, field.depthOrArrayLayers] as const;
  const rowBytes = Math.ceil(dims[0] * 4 / 256) * 256;
  const dst = device.createBuffer({ size: rowBytes * dims[1] * dims[2], usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture: field }, { buffer: dst, bytesPerRow: rowBytes, rowsPerImage: dims[1] }, dims);
  device.queue.submit([encoder.finish()]);
  await dst.mapAsync(GPUMapMode.READ);
  writeFileSync(join(out, "water.f32"), Buffer.from(dst.getMappedRange()));
  dst.unmap();
  dst.destroy();
  writeFileSync(join(out, "frame.json"), JSON.stringify({ dims, rowBytes, method: method.id, time: solver.info.completedTime_s,
    container: scene.container, h: scene.voxelDomain!.finestCellSize_m, galleons,
    shots: bodies.map(b => ({ position: b.position_m, radius: b.description.dimensions_m.x,
      muzzle: b.description.position_m, start: b.description.launcher!.start_s, velocity: b.description.launcher!.velocity_m_s })),
  }, null, 2));
} finally {
  solver.destroy();
  device.destroy();
}
