import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createUniformReferenceComputeShader } from "../lib/methods/uniform/webgpu-uniform-reference.wgsl";

// Exercise the actual generated stencil with analytic fields, independently of
// pressure projection (which can hide an incorrect capillary force).
function shaderFunction(source: string, name: string): string {
  const start = source.indexOf(`fn ${name}(`); assert.ok(start >= 0);
  let depth = 0;
  for (let i = source.indexOf("{", start); i < source.length; i++) {
    if (source[i] === "{") depth++;
    if (source[i] === "}" && --depth === 0) return source.slice(start, i + 1);
  }
  throw new Error(`Unclosed ${name}`);
}

(process.env.WEBGPU_NODE_MODULE ? test : test.skip)("geometric curvature preserves a flat solid contact and a curved free surface", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "uniform geometric curvature");
  let device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as NodeDawnProvider;
    Object.assign(globalThis, dawn.globals);
    const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter); device = await adapter.requestDevice();
    const source = createUniformReferenceComputeShader(true);
    const module = device.createShaderModule({ code: `
struct Params {cellGravity:vec4f}
const params=Params(vec4f(0.05,0.05,0.05,0));
@group(0) @binding(0) var uvPhiIn:texture_3d<f32>;
@group(0) @binding(1) var<storage,read_write> output:array<vec4f>;
fn uvCorner(i:u32)->vec3i{return vec3i(i32(i&1u),i32((i>>1u)&1u),i32((i>>2u)&1u));}
fn cellOpenFraction(p:vec3i)->f32{return select(0.0,1.0,all(p>=vec3i(4,7,0))&&all(p<vec3i(32)));}
${shaderFunction(source, "interfaceNormal")}
${shaderFunction(source, "curvatureAt")}
@compute @workgroup_size(4,4,4) fn probe(@builtin(global_invocation_id) p:vec3u){
 if(any(p>=vec3u(32))){return;}let id=vec3i(p);if(cellOpenFraction(id)==0){return;}
 output[p.x+32u*(p.y+32u*p.z)]=vec4f(interfaceNormal(id),curvatureAt(id));
}` });
    assert.deepEqual((await module.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "probe" } });
    const phi = device.createTexture({ size: [33,33,33], dimension: "3d", format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const output = device.createBuffer({ size: 32**3*16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readback = device.createBuffer({ size: output.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: phi.createView() }, { binding: 1, resource: { buffer: output } }] });
    for (const curved of [false, true]) {
      const values = new Float32Array(33**3);
      for (let z=0;z<=32;z++) for (let y=0;y<=32;y++) for (let x=0;x<=32;x++) {
        // Buried vertices carry the same positive sentinel as construction.
        values[x+33*(y+33*z)] = x<4||y<7 ? 1.6 : curved ? .05*(Math.hypot(x-20,y-20,z-20)-6) : .05*(y-14.6);
      }
      device.queue.writeTexture({ texture: phi }, values, { bytesPerRow: 33*4, rowsPerImage: 33 }, [33,33,33]);
      const encoder=device.createCommandEncoder();const pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(8,8,8);pass.end();encoder.copyBufferToBuffer(output,0,readback,0,output.size);device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);const result=new Float32Array(readback.getMappedRange()).slice();readback.unmap();
      if (!curved) {
        for(let z=1;z<31;z++)for(let y=7;y<31;y++)for(let x=4;x<31;x++)assert.ok(Math.abs(result[4*(x+32*(y+32*z))+3]!)<1e-4, "plane has zero curvature beside the wall and floor");
      } else {
        for(const [x,y,z] of [[25,20,20],[20,25,20],[20,20,25],[16,16,20]]) {
          const actual=result[4*(x!+32*(y!+32*z!))+3]!;
          const expected=2/(.05*Math.hypot(x!+.5-20,y!+.5-20,z!+.5-20));
          assert.ok(Math.abs(actual/expected-1)<.05, `sphere curvature ${actual}, expected ${expected}`);
        }
      }
    }
    phi.destroy(); output.destroy(); readback.destroy();
  } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); }
});
