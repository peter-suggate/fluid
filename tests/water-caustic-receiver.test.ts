import assert from "node:assert/strict";
import test from "node:test";
import { waterCausticReceiverWGSL } from "../lib/core/water-caustic-receiver.wgsl";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { readMixedBuffer } from "./helpers/uniform-mixed-native-fields";

test("floor caustics illuminate the mapped terrain and raised floor without extending up a wall", {
  skip: !process.env.WEBGPU_NODE_MODULE,
}, async () => {
  await withUniformDevice("water caustic receiver", async device => {
    const module = device.createShaderModule({ code: `
      struct Settings { container:vec4f }
      const u=Settings(vec4f(8,4,8,0));
      @group(0) @binding(0) var causticMap:texture_2d<f32>;
      @group(0) @binding(1) var linearSampler:sampler;
      @group(0) @binding(2) var<storage,read_write> result:array<f32>;
      fn waterReceiverHeight(x:f32,z:f32)->f32{return .25*x;}
      fn waterCausticStrength()->f32{return 1;}
      fn waterIndexOfRefraction()->f32{return 1.333;}
      fn waterAuthoredKeyDirection()->vec3f{return vec3f(.8,.6,0);}
      ${waterCausticReceiverWGSL}
      @compute @workgroup_size(1) fn main(){
        let wall=vec3f(1,0,0);let floor=vec3f(0,-1,0);
        result[0]=dryReceiverCaustic(vec3f(2,.5,0)).x;
        result[1]=dryReceiverCaustic(vec3f(2,2,0)).x;
        result[2]=solidReceiverCaustic(vec3f(2,2,0),wall).x;
        result[3]=solidReceiverCaustic(vec3f(2,3.5,0),wall).x;
        result[4]=solidReceiverCaustic(vec3f(2,3.5,0),floor).x;
      }` });
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
    const texture = device.createTexture({ size: [32, 32], format: "rgba8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const buffer = device.createBuffer({ size: 20, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    try {
      // A sharp boundary makes a displaced floor distinguish the correct
      // light-ray footprint from the camera pixel or a vertical projection.
      const pixels = new Uint8Array(32 * 32 * 4).fill(255);
      for (let y = 0; y < 32; y++) for (let x = 0; x < 32; x++) pixels[(y * 32 + x) * 4] = x < 16 ? 32 : 128;
      device.queue.writeTexture({ texture }, pixels, { bytesPerRow: 128 }, [32, 32]);
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: texture.createView() },
        { binding: 1, resource: device.createSampler({ minFilter: "linear", magFilter: "linear" }) },
        { binding: 2, resource: { buffer } },
      ] });
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
      device.queue.submit([encoder.finish()]);
      const values = await readMixedBuffer(device, buffer);
      for (const [i, expected] of [128 / 255, 1, 1, 1, 32 / 255].entries()) {
        assert.ok(Math.abs(values[i]! - expected) < .001, `receiver ${i}: ${values[i]} vs ${expected}`);
      }
    } finally {
      buffer.destroy();
      texture.destroy();
    }
  });
});
