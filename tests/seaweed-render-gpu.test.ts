import assert from "node:assert/strict";
import test from "node:test";
import { seaweedShaderLibrary } from "../lib/core/seaweed-render";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { createBeachScene } from "../lib/core/beach-scene";
import { initializeRigidBodies } from "../lib/core/rigid-body";
import { WebGPURigidBodySystem } from "../lib/core/webgpu-rigid-body";

test("integrating seaweed retains its render skin while physics uses the box proxy", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice("resident seaweed appearance", async device => {
    const scene = createBeachScene();
    scene.rigidBodies = scene.rigidBodies.filter(body => body.appearance === "seaweed").slice(0, 1);
    scene.rigidBodies[0]!.linearVelocity_m_s = { x: .2, y: 0, z: 0 };
    scene.fluid.gravity_m_s2 = { x: 0, y: 0, z: 0 };
    delete scene.terrain;
    const bodies = initializeRigidBodies(scene.rigidBodies);
    const exchange = device.createBuffer({ size: 12 * 12 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const terrain = device.createTexture({ size: [1, 1], format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const read = device.createBuffer({ size: 96, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const rigid = new WebGPURigidBodySystem(device, scene, exchange, terrain);
    try {
      for (const task of rigid.initializationTasks()) await task.run(new AbortController().signal);
      rigid.syncBodies(bodies);
      const encoder = device.createCommandEncoder();
      rigid.encode(encoder, 1 / 60, .125 ** 3, 1, .125);
      encoder.copyBufferToBuffer(rigid.renderBuffer, 0, read, 0, 64);
      encoder.copyBufferToBuffer(rigid.stateBuffer, 0, read, 64, 32);
      device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const values = new Float32Array(read.getMappedRange());
      assert.equal(values[7], 9, "Integration must not replace the seaweed skin with a visible box");
      assert.equal(values[19], 1, "The physics body must retain the box shape");
      assert.ok(values[0]! > bodies[0]!.position_m.x, "The visible skin follows the integrated pose");
      assert.ok(values[3]! > values[23]!, "Appearance bounds include blades beyond the compact collision proxy");
      read.unmap();
    } finally { rigid.destroy(); read.destroy(); exchange.destroy(); terrain.destroy(); }
  });
});

test("Blender seaweed skin intersects wet blades and rejects empty water", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice("seaweed skin", async device => {
    const output = device.createBuffer({ size: 48, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readback = device.createBuffer({ size: 48, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    try {
      const module = device.createShaderModule({ code: `${seaweedShaderLibrary}
        @group(0) @binding(0) var<storage,read_write> result:array<vec4f>;
        @compute @workgroup_size(1) fn main(){
          let origin=SEAWEED_PARTS[0].center+vec3f(0,1,0);let hit=seaweedHit(origin,vec3f(0,-1,0));
          result[0]=vec4f(hit.t,hit.normal);result[1]=vec4f(seaweedColor(origin+vec3f(0,-hit.t,0)),1);
          let miss=seaweedHit(vec3f(2,1,2),vec3f(0,-1,0));result[2]=vec4f(miss.t,0,0,0);
        }` });
      const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: output } }] });
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
      encoder.copyBufferToBuffer(output, 0, readback, 0, 48); device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const data = new Float32Array(readback.getMappedRange());
      assert.ok(data[0]! > .9 && data[0]! < 1.1);
      assert.ok(Math.abs(Math.hypot(data[1]!,data[2]!,data[3]!) - 1) < 1e-5);
      assert.ok(data[4]! < .3 && data[5]! < .3 && data[6]! < .1);
      assert.ok(data[8]! > 1e19);
      readback.unmap();
    } finally { output.destroy(); readback.destroy(); }
  });
});
