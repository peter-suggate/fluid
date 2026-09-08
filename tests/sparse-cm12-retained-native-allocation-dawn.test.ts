import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";

const live = new Set<GPU>();
(process.env.WEBGPU_NODE_MODULE ? test : test.skip)(
  "retained native compilation ignores poisoned free pages and rejects real allocated-cell mismatches", async () => {
    await acquireWebGPUExclusiveLock("dawn-test", "retained-native-allocation");
    let device: GPUDevice | undefined;
    try {
      const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
      Object.assign(globalThis, dawn.globals);
      const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); live.add(gpu);
      const adapter = await gpu.requestAdapter(); assert.ok(adapter);
      device = await adapter.requestDevice(); assert.ok(device);
      const source = readFileSync(new URL(
        "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url), "utf8");
      const actualFunction = (name: string) => {
        const match = source.match(new RegExp(`fn ${name}\\([\\s\\S]*?\\n}`));
        assert.ok(match, `production function ${name}`);
        // This fixture has no rigid arena. Match the production generator's
        // optional capacity-plane line, leaving its actual integral guard intact.
        return match[0].replace(/\$\{layout\.rigid \? [^\n]+\}/g, "");
      };
      const count = 1025;
      const code = `
@group(0) @binding(0) var<storage,read_write> topologyArena:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read_write> state:array<f32>;
const BRICK_FINE_RESOLUTION=8u;
const CM12_WDR_INITIAL_LEAVES=1u;
const CM12_RETAINED_INTEGRAL_BASE=${count}u;
struct Params { counts:vec4u }
const p=Params(vec4u(${count}u,0u,0u,0u));
fn ta(_at:u32)->u32{return 1u;}
fn candidateTopologyPageBase(page:u32)->u32{return 4u+4u*page;}
fn cm12WorldLeafAllocated(leaf:u32)->bool{return atomicLoad(&topologyArena[16u+leaf])==1u;}
fn cellBrick(cell:u32)->u32{
  if(cell==0u){return 0u;}
  return atomicLoad(&topologyArena[candidateTopologyPageBase((cell-1u)/512u)]);
}
fn cellCenter(cell:u32)->vec3f{return vec3f(f32(8u*cellBrick(cell)),0.0,0.0)+vec3f(0.5);}
fn cellWidths(_cell:u32)->vec3f{return vec3f(1.0);}
fn cellVolume(_cell:u32)->f32{return 1.0;}
fn cellActive(_cell:u32)->bool{return true;}
fn cm12RetainedDensitySupportMomentsAt(_q:vec3i,_bank:u32)->vec2f{return vec2f(0.25,1.0);}
fn cm12RetainedDensityEnabled()->bool{return true;}
fn cm12RetainedDensityAcceptedBank()->u32{return 0u;}
fn destinationDensity()->u32{return 0u;}
fn cm12RetainedDensityCellMean(cell:u32)->f32{return state[CM12_RETAINED_INTEGRAL_BASE+cell];}
fn cm12RecordFailure(reason:u32,owner:u32,operands:vec4u){
  atomicStore(&topologyArena[0],reason);atomicStore(&topologyArena[1],owner);
  atomicStore(&topologyArena[2],operands.x);
}
${actualFunction("cm12RetainedDensityCellAllocated")}
${actualFunction("cm12RetainedDensityIntegrateCellMoments")}
@compute @workgroup_size(64)
${actualFunction("compileRetainedDensityNativeIntegrals")}
`;
      const module = device.createShaderModule({ code });
      assert.deepEqual((await module.getCompilationInfo()).messages.filter(message => message.type === "error"), []);
      const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "compileRetainedDensityNativeIntegrals" } });
      const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
      const topology = device.createBuffer({ size: 32 * 4, usage: storage });
      const fields = device.createBuffer({ size: 2 * count * 4, usage: storage });
      const output = device.createBuffer({ size: topology.size + fields.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const binding = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: topology } }, { binding: 1, resource: { buffer: fields } },
      ] });
      const run = async (descriptor: readonly number[], allocated: boolean, mismatch: boolean) => {
        const words = new Uint32Array(32); words[16] = 1; words[17] = Number(allocated); words.set(descriptor, 4);
        // Free slots deliberately alias host leaf zero and contain poisoned
        // rho/cache values. Actual host and allocated page values are explicit.
        const values = new Float32Array(2 * count).fill(.875); values[0] = .25;
        if (descriptor[0] === 1 && allocated) values.fill(.25, 1, 513);
        if (mismatch) values[1] = .5;
        device!.queue.writeBuffer(topology, 0, words); device!.queue.writeBuffer(fields, 0, values);
        const encoder = device!.createCommandEncoder();
        const pass = encoder.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, binding);
        pass.dispatchWorkgroups(Math.ceil(count / 64)); pass.end();
        encoder.copyBufferToBuffer(topology, 0, output, 0, topology.size);
        encoder.copyBufferToBuffer(fields, 0, output, topology.size, fields.size);
        device!.queue.submit([encoder.finish()]); await output.mapAsync(GPUMapMode.READ);
        const receipt = new Uint32Array(output.getMappedRange()).slice(); output.unmap();
        return { receipt, fields: new Float32Array(receipt.buffer, topology.size) };
      };
      const free = await run([0, 0, 0, 0], false, false);
      assert.equal(free.receipt[0], 0, "free-page poison cannot fault a valid host field");
      assert.equal(free.fields[count], .25);
      assert.ok(free.fields.subarray(count + 1).every(value => value === 0), "free pages have no native integrals");
      assert.equal(free.fields[1], .875, "integral compilation never rounds or rewrites poisoned rho");
      const allocated = await run([1, 8, 512, 0], true, false);
      assert.equal(allocated.receipt[0], 0);
      assert.ok(allocated.fields.subarray(count + 1, count + 513).every(value => value === .25));
      const retired = await run([1, 8, 512, 0], false, false);
      assert.equal(retired.receipt[0], 0);
      assert.equal(retired.fields[count + 1], 0, "retired physical leaf is excluded even with an old descriptor");
      const mismatch = await run([1, 8, 512, 0], true, true);
      assert.deepEqual([...mismatch.receipt.subarray(0, 3)], [6, 1, 3], "allocated native mismatch still halts");
      output.destroy(); fields.destroy(); topology.destroy();
    } finally { device?.destroy(); live.clear(); await releaseWebGPUExclusiveLock(); }
  });
