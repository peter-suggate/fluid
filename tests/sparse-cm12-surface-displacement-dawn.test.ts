import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";

// Execute the production proof function. Supply only its lattice and physical
// units here, so the fixture tests the gate independently of activity history,
// pressure, remapping and the conservative volume field.
const source = readFileSync(new URL("../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url), "utf8");
const start = source.indexOf("fn surfaceProofOutputSampleFailure(");
const end = source.indexOf("// Publish a camera-independent", start);
assert.ok(start >= 0 && end > start);
const proof = source.slice(start, end);

(process.env.WEBGPU_NODE_MODULE ? test : test.skip)(
  "surface proof rejects distant phase changes hidden by small candidate phi", { timeout: 60_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test", "surface displacement certificate");
    let device: GPUDevice | undefined;
    try {
      const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
      Object.assign(globalThis, dawn.globals);
      const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
      const adapter = await gpu.requestAdapter(); assert.ok(adapter);
      device = await adapter.requestDevice();
      assert.ok(device);
      const module = device.createShaderModule({ code: `
@group(0)@binding(0)var<storage,read>samples:array<vec2f>;
@group(0)@binding(1)var<storage,read_write>failures:array<u32>;
struct Params{frame:vec4f};const p=Params(vec4f(0.0,0.05,0.0,0.0));
const cm12PresentationBrickOrigin=vec3i(0);
fn cm12SolidVoxelFractionQ8(q:vec3i)->u32{_=q;return 0u;}
fn surfaceDisplacementToleranceMetres()->f32{return 0.05;}
fn surfaceProofPhiAt(local:vec3i,coarse:bool)->f32{
 let q=vec3u(local+vec3i(1));let s=samples[q.x+10u*(q.y+10u*q.z)];return select(s.x,s.y,coarse);
}
${proof}
@compute @workgroup_size(64)fn check(@builtin(global_invocation_id)gid:vec3u){
 let i=gid.x;if(i>=512u){return;}
 failures[i]=surfaceProofOutputSampleFailure(vec3i(vec3u(i%8u,(i/8u)%8u,i/64u)));
}` });
      assert.deepEqual((await module.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
      const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "check" } });
      const input = device.createBuffer({ size: 8000, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const output = device.createBuffer({ size: 2048, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const readback = device.createBuffer({ size: 2048, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const bindings = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: input } }, { binding: 1, resource: { buffer: output } },
      ] });
      const clearance = (y: number) => y <= 4 ? y - .5 : 3.5 + (y - 4) * (.01 - 3.5) / 4;
      const restricted = (y: number) => -.5 + y * .51 / 8;
      const fixtures = [
        { name: "pool swallows distant air", fine: clearance, coarse: restricted, reject: true },
        { name: "air erases distant liquid", fine: (y: number) => -clearance(y), coarse: (y: number) => -restricted(y), reject: true },
        { name: "unchanged plane", fine: (y: number) => y - 4.25, coarse: (y: number) => y - 4.25, reject: false },
        { name: "subcell displacement", fine: (y: number) => y - 4.25, coarse: (y: number) => y - 4.5, reject: false },
      ];
      for (const fixture of fixtures) {
        const values = new Float32Array(2000);
        for (let z = 0; z < 10; z++) for (let y = 0; y < 10; y++) for (let x = 0; x < 10; x++) {
          const at = 2 * (x + 10 * (y + 10 * z)), position = y - .5;
          values[at] = .05 * fixture.fine(position); values[at + 1] = .05 * fixture.coarse(position);
        }
        device.queue.writeBuffer(input, 0, values);
        const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
        pass.setPipeline(pipeline); pass.setBindGroup(0, bindings); pass.dispatchWorkgroups(8); pass.end();
        encoder.copyBufferToBuffer(output, 0, readback, 0, 2048); device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const rejected = new Uint32Array(readback.getMappedRange()).some(flag => flag !== 0);
        readback.unmap();
        assert.equal(rejected, fixture.reject, fixture.name);
      }
      input.destroy(); output.destroy(); readback.destroy();
    } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); }
  });
