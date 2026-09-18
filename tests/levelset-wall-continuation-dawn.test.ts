import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createLevelSetVolumeWGSL } from "../lib/methods/adaptive-volume/levelset-volume-core.wgsl";
import { createLevelSetVolumeLayout } from "../lib/methods/adaptive-volume/levelset-volume-layout";

(process.env.WEBGPU_NODE_MODULE ? test : test.skip)(
  "closed-wall continuation preserves wet distance, admits contact, and yields to separation", { timeout: 60_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test", "closed-wall signed distance continuation");
    let device: GPUDevice | undefined;
    try {
      const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
      Object.assign(globalThis, dawn.globals);
      const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
      const adapter = await gpu.requestAdapter(); assert.ok(adapter);
      device = await adapter.requestDevice();
      assert.ok(device);
      const production = createLevelSetVolumeWGSL({
        layout: createLevelSetVolumeLayout({ activeCellCapacity: 8, vertexCapacity: 27 }),
        acceptedGenerationExpression: "0u", buildGenerationExpression: "0u", buildSlotExpression: "0u",
        buildCellCountExpression: "0u", buildCellAtOrdinal: () => "0u", acceptedCellOrdinal: () => "0u",
        acceptedOwnerCellAt: () => "vec2u(0)", buildOwnerCellAt: () => "vec2u(0)",
        authoredSample: () => "vec2f(0)", velocitySample: () => "vec4f(0)",
        closedWallPhi: p => `closedPhi(${p})`, dtExpression: "0.033333333", constraintWidthExpression: "1u",
      });
      const start = production.indexOf("fn lsvStoreAdvectedPhi("), end = production.indexOf("fn lsvSlotBase(", start);
      assert.ok(start >= 0 && end > start);
      const module = device.createShaderModule({ code: `
@group(0)@binding(0)var<storage,read>cases:array<vec4f>;
@group(0)@binding(1)var<storage,read_write>result:array<u32>;
const LSV_SUPPORT_DEEP_AIR=1u;const LSV_SUPPORT_DEEP_LIQUID=2u;const LSV_SUPPORT_METRIC=3u;
fn lsvVertexPosition(slot:u32,v:u32)->vec3f{_=slot;return vec3f(f32(v),0,0);}
fn closedPhi(p:vec3f)->vec2f{return cases[2u*u32(p.x)].yz;}
fn lsvPhiBase(slot:u32,bank:u32)->u32{_=slot;_=bank;return 0u;}
fn lsvSupportBase(slot:u32,bank:u32)->u32{_=slot;_=bank;return 8u;}
fn lsvStore(at:u32,v:u32){result[at]=v;}
fn lsvStoreFloat(at:u32,v:f32){result[at]=bitcast<u32>(v);}
${production.slice(start, end)}
@compute @workgroup_size(1)fn check(@builtin(global_invocation_id)i:vec3u){
 let c=cases[2u*i.x];lsvStoreAdvectedPhi(0u,0u,i.x,c.x,u32(c.w),cases[2u*i.x+1u].xy);
}` });
      assert.deepEqual((await module.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
      const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "check" } });
      // [advected phi, interior phi, continuation active, support, release phi, release active]
      const fixtures = [
        { name: "resting coarse pool", values: [-6.5, -5.5, 1, 2, 0, 0], phi: -6.5, support: 2 },
        { name: "new liquid contact", values: [2, -.5, 1, 1, 0, 0], phi: -.5, support: 3 },
        { name: "outgoing wall separation", values: [-6.5, -5.5, 1, 2, .125, 1], phi: .125, support: 3 },
        { name: "air does not erase a wet film", values: [-.25, 1, 1, 3, 0, 0], phi: -.25, support: 3 },
        { name: "inactive wall", values: [-.25, -1, 0, 3, 0, 0], phi: -.25, support: 3 },
      ];
      const values = new Float32Array(fixtures.length * 8);
      fixtures.forEach((f, i) => values.set(f.values, i * 8));
      const input = device.createBuffer({ size: values.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const output = device.createBuffer({ size: 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const readback = device.createBuffer({ size: 64, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(input, 0, values);
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: input } }, { binding: 1, resource: { buffer: output } },
      ] });
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(fixtures.length); pass.end();
      encoder.copyBufferToBuffer(output, 0, readback, 0, 64); device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange()), scalars = new Float32Array(words.buffer);
      fixtures.forEach((fixture, i) => {
        assert.equal(scalars[i], fixture.phi, fixture.name);
        assert.equal(words[8 + i], fixture.support, `${fixture.name} support`);
      });
      readback.unmap(); input.destroy(); output.destroy(); readback.destroy();
    } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); }
  });
