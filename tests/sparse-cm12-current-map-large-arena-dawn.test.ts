import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createSparseCM12CurrentMapLayout, createSparseCM12CurrentMapWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-current-map.wgsl";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const live = new Set<GPU>();

(dawnModule ? test : test.skip)("production map archive publication and commit at full quarter-scene arena offsets",
  { timeout: 120_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test", "current-map-large-arena");
    let gpu: GPU | undefined, device: GPUDevice | undefined;
    const buffers: GPUBuffer[] = [];
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href); Object.assign(globalThis, dawn.globals);
      gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); live.add(gpu!);
      const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
      device = await adapter.requestDevice({ requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      } });
      const errors: string[] = [];
      device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
      const layout = createSparseCM12CurrentMapLayout(19_434_556, [32, 24, 32]);
      const controlBase = 18_004_152;
      const module = device.createShaderModule({ code: /* wgsl */ `
struct Parameters{frame:vec4f}
@group(0) @binding(0) var<storage,read_write> state:array<f32>;
@group(0) @binding(1) var<uniform> p:Parameters;
@group(0) @binding(2) var<storage,read_write> failures:array<atomic<u32>>;
fn cm12RetainedDensityAcceptedBank()->u32{return u32(state[${controlBase + 1}u])&1u;}
fn cm12CurrentMapNativeVelocity(_point:vec3f)->vec4f{return vec4f(0.0,0.0,0.0,1.0);}
fn cm12CurrentMapVelocityBoundary(_point:vec3f,value:vec3f)->vec3f{return value;}
fn cm12CurrentMapCoefficientBoundaryPoint(point:vec3f)->vec3f{return point;}
fn cm12CurrentMapCoefficientBoundaryValue(_point:vec3f,value:vec3f)->vec3f{return value;}
fn cm12CurrentMapInitializeVelocity(id:u32,sample:vec4f){cm12CurrentMapWrite(CM12_CURRENT_MAP_VELOCITY_BASE,id,sample.xyz);}
fn cm12CurrentMapFail(code:u32,_id:u32){atomicOr(&failures[0],1u<<code);}
fn cm12CurrentMapFailed()->bool{return atomicLoad(&failures[0])!=0u;}
${createSparseCM12CurrentMapWGSL(layout)}
@compute @workgroup_size(1)
fn commitRetainedDensityGeneration(){
  state[${controlBase + 3}u]+=1.0;
  if(!(state[${controlBase}u]>0.5)||cm12CurrentMapFailed()){return;}
  cm12CurrentMapCommitIncrement();
  state[${controlBase + 1}u]=f32(1u-cm12RetainedDensityAcceptedBank());
  state[${controlBase + 2}u]+=1.0;state[${controlBase}u]=2.0;
}` });
      assert.deepEqual((await module.getCompilationInfo()).messages.filter(message => message.type === "error")
        .map(message => `${message.lineNum}: ${message.message}`), []);
      const bindingLayout = device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ] });
      const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindingLayout] });
      const publish = await device.createComputePipelineAsync({ layout: pipelineLayout, compute: { module, entryPoint: "publishCurrentMapIncrement" } });
      const commit = await device.createComputePipelineAsync({ layout: pipelineLayout, compute: { module, entryPoint: "commitRetainedDensityGeneration" } });
      const allocate = (size: number, usage: GPUBufferUsageFlags) => {
        const buffer = device!.createBuffer({ size, usage }); buffers.push(buffer); return buffer;
      };
      const state = allocate(4 * layout.endWords, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
      const parameters = allocate(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      const failures = allocate(4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
      const readback = allocate(64, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
      const bindings = device.createBindGroup({ layout: bindingLayout,
        entries: [state, parameters, failures].map((buffer, binding) => ({ binding, resource: { buffer } })) });
      device.queue.writeBuffer(state, 4 * controlBase, new Float32Array([2, 0, 3, 0]));
      device.queue.writeBuffer(state, 4 * layout.chainCountBaseWords, new Float32Array([2]));
      const probes = [0, Math.floor(layout.nodeCount / 2), layout.nodeCount - 1];
      for (let step = 0; step < 7; step++) {
        const candidate = 1 - step % 2;
        const values = [step + .125, -step - .25, step + .5];
        for (const id of probes) device.queue.writeBuffer(state,
          4 * (layout.coefficientBaseWords[candidate]! + 3 * id), new Float32Array(values));
        const encoder = device.createCommandEncoder();
        // Exercise the original same-pass publication/commit ordering too.
        const pass = encoder.beginComputePass(); pass.setBindGroup(0, bindings);
        pass.setPipeline(publish); pass.dispatchWorkgroups(Math.ceil(layout.nodeCount / 64));
        pass.setPipeline(commit); pass.dispatchWorkgroups(1); pass.end();
        encoder.copyBufferToBuffer(state, 4 * controlBase, readback, 0, 16);
        encoder.copyBufferToBuffer(state, 4 * layout.chainCountBaseWords, readback, 16, 4);
        encoder.copyBufferToBuffer(failures, 0, readback, 20, 4);
        for (let i = 0; i < probes.length; i++) encoder.copyBufferToBuffer(state,
          4 * (layout.chainBaseWords + 3 * layout.nodeCount * (step + 2) + 3 * probes[i]!), readback, 24 + 12 * i, 12);
        device.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(readback.getMappedRange()).slice(); readback.unmap();
        assert.deepEqual([...data.slice(0, 5)], [2, candidate, step + 4, step + 1, step + 3], `commit ${step}`);
        assert.equal(data[5], 0);
        for (let i = 0; i < probes.length; i++) assert.deepEqual([...data.slice(6 + 3 * i, 9 + 3 * i)], values,
          `archive slot ${step + 2}, node ${probes[i]}`);
      }
      await device.queue.onSubmittedWorkDone(); assert.deepEqual(errors, []);
    } finally {
      for (const buffer of buffers) buffer.destroy(); device?.destroy(); if (gpu) live.delete(gpu);
      await releaseWebGPUExclusiveLock();
    }
  });
