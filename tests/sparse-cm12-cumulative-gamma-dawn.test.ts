import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createCm12NumericsWGSL, CM12_TRANSPORT_FIXED_SCALE } from "../lib/core/cm12-numerics";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";

// Execute both actual production gathers. With zero motion the density
// operator is identity, so it must also leave non-unit historical gamma alone.
// The old backward A*1 / forward A*gamma hybrid fails on both sides of one.
const source = readFileSync(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url), "utf8");
const entryPoints = ["gatherConservativeDensity", "gatherConservativeDensityPackedCoarse"];
const production = (name: string) => {
  const body = source.match(new RegExp(`fn ${name}\\([\\s\\S]*?\\n}`))?.[0];
  assert.ok(body, name);
  return body.replace(/\$\{phase1QA\w+\}/g, "");
};

(process.env.WEBGPU_NODE_MODULE ? test : test.skip)(
  "stationary fine and packed-coarse gathers retain cumulative gamma", async () => {
    await acquireWebGPUExclusiveLock("dawn-test", "cumulative-gamma");
    let device: GPUDevice | undefined, gpu: GPU | undefined;
    try {
      const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
      Object.assign(globalThis, dawn.globals);
      gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
      const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
      device = await adapter.requestDevice();
      const errors: string[] = [];
      device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
      const shaderModule = device.createShaderModule({ code: `
${createCm12NumericsWGSL()}
const INVALID=0xffffffffu;
const CM12_SPARSE_TRANSPORT_FIXED=CM12_TRANSPORT_FIXED;
struct Params{counts:vec4u}
const p=Params(vec4u(4,0,0,0));
@group(0)@binding(0)var<storage,read_write>state:array<f32>;
@group(0)@binding(1)var<storage,read_write>conditioning:array<atomic<i32>>;
fn stageSparseCM12TransportExecutionImage(w:u32,l:u32,s:u32){_=w;_=l;_=s;}
fn cm12MassExecutionCell(w:u32,l:u32,s:u32)->u32{_=w;_=s;return select(INVALID,l,l<4u);}
struct Owner{cell:u32}
fn cm12PackedCoarseCell(i:u32)->Owner{return Owner(select(INVALID,i,i<4u));}
fn cellActive(c:u32)->bool{_=c;return true;}
fn cellTransportActive(c:u32)->bool{_=c;return true;}
fn dynamicallyCoveredCell(c:u32)->bool{_=c;return false;}
fn sourceDensity()->u32{return 0u;}
fn sourceGamma()->u32{return 4u;}
fn destinationDensity()->u32{return 8u;}
fn destinationGamma()->u32{return 12u;}
fn sourceCellVelocity()->u32{return 16u;}
fn destinationCellVelocity()->u32{return 32u;}
fn massDepartureStencilCell(c:u32,k:u32)->u32{_=k;return c;}
fn massDepartureStencilWeight(c:u32,k:u32)->f32{_=c;return select(0.0,1.0,k==0u);}
fn transportBeta(c:u32)->f32{return state[sourceGamma()+c];}
fn configuredTransportCoefficient(g:f32,w:f32,b:f32)->f32{return cm12ConditionedRowCoefficient(g,w,b);}
fn cm12PublishTransferredEffectiveVelocity(c:u32,v:vec3f){_=c;_=v;}
${entryPoints.map(name => `@compute @workgroup_size(64)\n${production(name)}`).join("\n")}
` });
      assert.deepEqual((await shaderModule.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
      const gamma = [0.5, 0.75, 1.25, 2];
      const initial = new Float32Array(48);
      const receipts = new Int32Array(24);
      for (let i = 0; i < 4; i++) {
        initial[i] = 1;
        initial[4 + i] = initial[12 + i] = gamma[i];
        const deficit = Math.max(0, 1 - gamma[i]);
        receipts[4 + i] = deficit * CM12_TRANSPORT_FIXED_SCALE;
        receipts[8 + i] = gamma[i] * deficit * CM12_TRANSPORT_FIXED_SCALE;
      }
      const buffers = [initial.byteLength, receipts.byteLength].map(size => device!.createBuffer({
        size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      }));
      const readback = device.createBuffer({ size: initial.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      for (const entryPoint of entryPoints) {
        const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module: shaderModule, entryPoint } });
        const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
          entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
        device.queue.writeBuffer(buffers[0], 0, initial);
        device.queue.writeBuffer(buffers[1], 0, receipts);
        const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
        pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
        encoder.copyBufferToBuffer(buffers[0], 0, readback, 0, readback.size);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(readback.getMappedRange());
        assert.deepEqual([...result.slice(8, 12)], [1, 1, 1, 1], entryPoint);
        assert.deepEqual([...result.slice(12, 16)], gamma, entryPoint);
        readback.unmap();
      }
      readback.destroy(); buffers.forEach(buffer => buffer.destroy());
      assert.deepEqual(errors, []);
    } finally {
      device?.destroy(); await releaseWebGPUExclusiveLock();
      // Keep Dawn's owner alive until after device destruction and lease release.
      assert.ok(gpu);
    }
  });
