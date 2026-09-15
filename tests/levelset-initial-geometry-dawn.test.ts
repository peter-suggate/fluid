import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { cloneScene, defaultScene, type InitialLiquidVolume } from "../lib/core/model";
import { initialLiquidVolumeSignedDistance } from "../lib/core/initial-fluid";
import { createSymmetricExpansionScene } from "../lib/core/scenes";
import { createInitialLevelSetGeometryWGSL } from "../lib/methods/adaptive-volume/levelset-initial-geometry";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";

const modulePath = process.env.WEBGPU_NODE_MODULE;
(modulePath ? test : test.skip)("adaptive phi authored GPU seeds agree with analytic geometry", { timeout: 60_000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "adaptive phi authored geometry");
  let gpu: GPU | undefined, device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(modulePath!).href);
    Object.assign(globalThis, dawn.globals);
    gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice();
    const center_m = { x: .25, y: 2, z: -.25 };
    const wallBox: InitialLiquidVolume = { shape: "box",
      min_m: { x: -1, y: 0, z: -4 }, max_m: { x: 1, y: 8, z: 4 } };
    const brickBox: InitialLiquidVolume = { shape: "box",
      min_m: { x: -2, y: 0, z: -2 }, max_m: { x: 2, y: 2, z: 2 } };
    const shapes: InitialLiquidVolume[] = [
      { shape: "box", min_m: { x: -1, y: 1, z: -1 }, max_m: { x: 1, y: 3, z: 1 } },
      { shape: "sphere", center_m, radius_m: 1 },
      { shape: "cylinder", center_m, radius_m: 1, halfHeight_m: .75 },
      { shape: "torus", center_m, radius_m: 1, tubeRadius_m: .3 },
      { shape: "hemisphere", center_m, radius_m: 1, outwardNormal: { x: 1, y: 2, z: -1 } },
      wallBox,
      brickBox,
    ];
    const scene = cloneScene(defaultScene);
    Object.assign(scene.container, { width_m: 8, height_m: 8, depth_m: 8,
      fillFraction: 0, top: "closed" });
    delete scene.fluid.initialBrickSeeds_m; delete scene.fluid.initialHeightField;
    const count = 512, h = .25;
    const output = device.createBuffer({ size: count * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readback = device.createBuffer({ size: count * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    for (const shape of shapes) {
      scene.fluid.initialLiquidVolumes = [shape];
      if (shape === brickBox) {
        scene.fluid.initialLiquidVolumes = [];
        scene.fluid.initialBrickSeeds_m = [-1, 1].flatMap(x => [-1, 1]
          .map(z => ({ x, y: 1, z })));
      }
      const code = createInitialLevelSetGeometryWGSL(scene, [32, 32, 32], h) + `
@group(0) @binding(0) var<storage,read_write> result:array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=512u){return;}
 let point=vec3f(-1.75+0.5*f32(i%8u),0.25+0.5*f32((i/8u)%8u),-1.75+0.5*f32(i/64u));
 result[i]=lsvAuthoredPhi((point-vec3f(-4.0,0.0,-4.0))/0.25);
}`;
      const shaderModule: GPUShaderModule = device.createShaderModule({ code });
      const info: GPUCompilationInfo = await shaderModule.getCompilationInfo();
      assert.deepEqual(info.messages.filter(m => m.type === "error"), [], shape.shape);
      const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module: shaderModule, entryPoint: "main" } });
      const bindings = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: output } }] });
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, bindings); pass.dispatchWorkgroups(8); pass.end();
      encoder.copyBufferToBuffer(output, 0, readback, 0, count * 4); device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const values = new Float32Array(readback.getMappedRange());
      for (let i = 0; i < count; i++) {
        const point = { x: -1.75 + .5 * (i % 8), y: .25 + .5 * (Math.floor(i / 8) % 8), z: -1.75 + .5 * Math.floor(i / 64) };
        const expected = (shape === brickBox
          ? Math.max(Math.abs(point.x) - 2, point.y - 2, Math.abs(point.z) - 2)
          : shape === wallBox ? Math.abs(point.x) - 1
          : initialLiquidVolumeSignedDistance(shape, point)) / h;
        assert.ok(Math.abs(values[i]! - expected) < 2e-5, `${shape.shape} sample ${i}: ${values[i]} vs ${expected}`);
      }
      readback.unmap();
    }
    // The production 5 cm lattice must author both reflected faces as exact
    // zeros. A tiny negative value on only the positive face loses its edge
    // crossing when redistance has no exterior air samples yet.
    const expansion = createSymmetricExpansionScene();
    const symmetricModule = device.createShaderModule({ code:
      createInitialLevelSetGeometryWGSL(expansion, [32, 16, 32], .05) + `
@group(0) @binding(0) var<storage,read_write> result:array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=9u){return;}
 var point=vec3f(8.0+16.0*f32(i&1u),8.0*f32((i>>1u)&1u),8.0+16.0*f32((i>>2u)&1u));
 if(i==8u){point=vec3f(16.0,0.0,16.0);}
 result[i]=lsvAuthoredPhi(point);
}` });
    const symmetricPipeline = await device.createComputePipelineAsync({ layout: "auto",
      compute: { module: symmetricModule, entryPoint: "main" } });
    const symmetricBindings = device.createBindGroup({ layout: symmetricPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: output } }] });
    const symmetricEncoder = device.createCommandEncoder();
    const symmetricPass = symmetricEncoder.beginComputePass();
    symmetricPass.setPipeline(symmetricPipeline); symmetricPass.setBindGroup(0, symmetricBindings);
    symmetricPass.dispatchWorkgroups(1); symmetricPass.end();
    symmetricEncoder.copyBufferToBuffer(output, 0, readback, 0, 36);
    device.queue.submit([symmetricEncoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
    const symmetricPhi = new Float32Array(readback.getMappedRange());
    for (let i = 0; i < 8; i++) assert.equal(symmetricPhi[i], 0,
      `symmetric box corner ${i} must preserve its exact zero crossing`);
    assert.equal(symmetricPhi[8], -8, "the floor continues liquid instead of introducing a free surface");
    readback.unmap();
    output.destroy(); readback.destroy();
  } finally {
    device?.destroy();
    await new Promise<void>(resolve => setImmediate(resolve));
    gpu = undefined;
    await releaseWebGPUExclusiveLock();
  }
});
