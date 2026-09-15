import assert from "node:assert/strict";
import test from "node:test";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { pathToFileURL } from "node:url";
import { createSparseCM12VelocityExtensionLayout } from "../lib/methods/adaptive-volume/sparse-cm12-velocity-extension";
import { createSparseCM12VelocityExtensionWGSL } from "../lib/methods/adaptive-volume/sparse-cm12-velocity-extension.wgsl";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";

(process.env.WEBGPU_NODE_MODULE ? test : test.skip)("VEX keeps projected effective velocity while air remains an extension receiver", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "trace-gravity");
  let device: GPUDevice | undefined, gpu: GPU | undefined;
  try {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
    Object.assign(globalThis, dawn.globals);
    gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice();
    const layout = createSparseCM12VelocityExtensionLayout({ cellCapacity: 64, packetCapacity: 64, brickFineResolution: 8 });
    const generated = createSparseCM12VelocityExtensionWGSL({layout, effectiveVelocityHookPrefix: "cm12"});
    const initialize = generated.match(/fn initializeVelocityExtensionPackets\([\s\S]*?\n}/)?.[0];
    assert.ok(initialize);
    // Exercise the production initializer across trace and partial fill.
    // Only bulk liquid may seed the velocity extension used at the surface.
    const shaderModule = device.createShaderModule({code: `
@group(0)@binding(0)var<storage,read_write>state:array<f32>;
@group(0)@binding(1)var<storage,read_write>activity:array<atomic<u32>>;
@group(0)@binding(2)var<storage,read_write>effective:array<vec4f>;
const CM12_LIQUID_ISOVALUE=0.5;
const cm12ExtensionInvalid=0xffffffffu;
const cm12ExtensionCapacity=64u;
const cm12ExtensionDispatchWidth=32768u;
const cm12ExtensionValidityA=0u;
const cm12ExtensionValidityB=64u;
const cm12ExtensionAcceptedDepth=2u;
struct Packet{first:u32,strideY:u32,strideZ:u32,counts:vec3u}
fn cm12TeiPacket(p:u32,s:u32)->Packet{_=p;_=s;return Packet(0u,4u,16u,vec3u(4,2,1));}
fn acceptedTopologySlot()->u32{return 0u;}
fn cm12ExtensionStablePacket(p:u32)->u32{return p;}
fn cm12ExtensionStore(at:u32,v:u32){atomicStore(&activity[at],v);}
fn sourceDensity()->u32{return 0u;}
fn sourceCellVelocity()->u32{return 8u;}
fn cm12EffectiveTransportVelocity(c:u32)->vec4f{return effective[c];}
fn cm12PublishVexAcceptedEffectiveVelocity(c:u32,v:vec4f){effective[c]=v;}
@compute @workgroup_size(64)
${initialize}
`});
    assert.deepEqual((await shaderModule.getCompilationInfo()).messages.filter(m=>m.type==="error"), []);
    const pipeline = await device.createComputePipelineAsync({layout:"auto",compute:{module:shaderModule,entryPoint:"initializeVelocityExtensionPackets"}});
    const values = new Float32Array(40);
    values.set([0, 1e-8, 1e-5, .01, .1, .49, .5, 1]);
    // The source bank is deliberately stale. The effective plane represents
    // the projection published immediately before the second VEX rebuild.
    for(let i=0;i<8;i++) values[8+4*i+1]=-91;
    const buffers = [values.byteLength, 66*4, 8*16].map(size=>device!.createBuffer({size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC}));
    const readback=device.createBuffer({size:8*16,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
    try {
      device.queue.writeBuffer(buffers[0],0,values);
      const projected = new Float32Array(8*4);
      for(let i=0;i<8;i++) projected.set([1.25+i,-9.81*.2,3.5-i,1],4*i);
      device.queue.writeBuffer(buffers[2],0,projected);
      const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:buffers.map((buffer,binding)=>({binding,resource:{buffer}}))});
      const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();
      pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();
      encoder.copyBufferToBuffer(buffers[2],0,readback,0,readback.size);device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const result=new Float32Array(readback.getMappedRange());
      for(let i=0;i<7;i++) {
        assert.deepEqual([...result.slice(4*i,4*i+4)], [0,0,0,0],
          `air-side density ${values[i]} must receive liquid extension`);
      }
      assert.deepEqual([...result.slice(4*7,4*7+4)], [...projected.slice(4*7,4*7+4)],
        "bulk liquid retains the projected plane rather than stale source velocity");
      assert.equal(result[4*7+3],1,"bulk liquid seeds extension");
    } finally {if(readback.mapState==="mapped")readback.unmap();readback.destroy();buffers.forEach(b=>b.destroy());}
  } finally {device?.destroy();await releaseWebGPUExclusiveLock();assert.ok(gpu);}
});

(process.env.WEBGPU_NODE_MODULE ? test : test.skip)("air-side velocity handling preserves the mixed-rung hydrostatic surface", {timeout:120_000}, async () => {
  await promisify(execFile)(process.execPath, ["--import", "tsx", "tools/probe-sparse-cm12-mini64-surface-dawn.ts"], {
    env: {...process.env, FLUID_MIN8_SURFACE_REGION:"right-x", FLUID_MIN8_SURFACE_STEPS:"48",
      FLUID_MIN8_SURFACE_SCENARIO:"large-offset"}, timeout:110_000,
  });
});
