import assert from "node:assert/strict";
import test from "node:test";
import { waterRayExitWGSL } from "../lib/core/water-interface-sampling.wgsl";
import { withUniformDevice } from "./helpers/uniform-geometric";

test("a refracted exit follows the ray to a sloping plane and rejects a gap or disconnected sheet", {
  skip: !process.env.WEBGPU_NODE_MODULE,
}, async () => {
  await withUniformDevice("refracted exit geometry", async device => {
    // Orthographic projection onto x/z. The exit plane slopes steeply away
    // from the observer, so projecting along camera depth does not converge.
    const module = device.createShaderModule({ code: `
      @group(0) @binding(0) var surface:texture_2d<f32>;
      @group(0) @binding(1) var<storage,read_write> result:array<vec4f>;
      var<private> mode:u32;
      fn project(p:vec3f)->vec2f{return .5+.1*p.xz;}
      fn waterInterfaceUVValid(uv:vec2f)->bool{return all(uv>=vec2f(0))&&all(uv<=vec2f(1));}
      fn safeInterfaceSample(t:texture_2d<f32>,uv:vec2f)->vec4f{return vec4f(normalize(vec3f(1.5,1,0)),select(1.0,1.5,mode==3u));}
      fn safePositionSample(t:texture_2d<f32>,uv:vec2f)->vec4f{
        let xz=(uv-.5)*10.0;
        if(textureDimensions(t).x==0u||(mode==1u&&xz.x>.1)){return vec4f(0);}
        let height=select(-1.5*xz.x,4.0,mode==2u&&xz.x>.1);
        return vec4f(xz.x,height,xz.y,select(1.0,.75,mode==3u));
      }
      ${waterRayExitWGSL}
      @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3u){
        mode=id.x;
        let d=normalize(vec3f(.4,-1,0));let p=vec3f(0,1,0);
        let hit=waterRayExit(surface,surface,project(p),p,d,.125);
        result[id.x]=vec4f(hit.point.xy,hit.solid,hit.distance);
      }` });
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
    const out = device.createBuffer({ size: 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: 64, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const texture = device.createTexture({ size: [1, 1], format: "rgba8unorm", usage: GPUTextureUsage.TEXTURE_BINDING });
    try {
      const entries = [{ binding: 0, resource: texture.createView() }, { binding: 1, resource: { buffer: out } }];
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(4); pass.end();
      encoder.copyBufferToBuffer(out, 0, read, 0, 64); device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const data = new Float32Array(read.getMappedRange());
      assert.ok(Math.abs(data[0]! - 1) < 1e-4 && Math.abs(data[1]! + 1.5) < 1e-4);
      assert.ok(Math.abs(data[3]! - Math.hypot(1, 2.5)) < 1e-4);
      assert.equal(data[2], 0);
      assert.equal(data[14], 1, "partial coverage retains a solid-clipped exit");
      assert.equal(data[15], data[3]);
      assert.equal(data[7], 0);
      assert.equal(data[11], 0);
    } finally {
      if (read.mapState === "mapped") read.unmap();
      read.destroy(); out.destroy(); texture.destroy();
    }
  });
});
