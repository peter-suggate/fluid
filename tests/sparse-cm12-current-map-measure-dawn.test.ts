import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createSparseCM12CurrentMapMeasureWGSL } from "../lib/methods/adaptive-mass/sparse-cm12-current-map-measure.wgsl";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const live = new Set<GPU>();

(dawnModule ? test : test.skip)("cooperative current-map quadrature completes every support and matches the scalar rule",
  { timeout: 120_000 }, async t => {
    await acquireWebGPUExclusiveLock("dawn-test", "current-map-cooperative-measure");
    let gpu: GPU | undefined, device: GPUDevice | undefined;
    const buffers: GPUBuffer[] = [];
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href); Object.assign(globalThis, dawn.globals);
      gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); live.add(gpu!);
      const adapter = await gpu!.requestAdapter(); assert.ok(adapter); device = await adapter.requestDevice();
      const errors: string[] = [];
      device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
      const count = 64, baseWords = 64, words = baseWords + 8 * count;
      const module = device.createShaderModule({ code: /* wgsl */ `
struct Parameters{frame:vec4f,transform:vec4f}
@group(0) @binding(0) var<storage,read_write> state:array<f32>;
@group(0) @binding(1) var<uniform> p:Parameters;
@group(0) @binding(2) var<storage,read_write> failures:array<atomic<u32>>;
const CM12_RETAINED_FIELD_BASE:u32=0u;
struct CurrentMapEvaluation{point:vec3f,jacobian:mat3x3f}
fn cm12RetainedDensityAcceptedBank()->u32{return 0u;}
fn cm12CurrentMapCandidateBank()->u32{return 1u;}
fn cm12CurrentMapFail(code:u32,id:u32){atomicStore(&failures[0],code);atomicStore(&failures[1],id);}
fn cm12CurrentMapFailed()->bool{return atomicLoad(&failures[0])!=0u;}
fn cm12CurrentMapMeasureFailure(id:u32,_error:vec4f,_tolerance:vec4f,_value:vec4f){cm12CurrentMapFail(21u,id);}
fn cm12CurrentMapMeasureCompleted(id:u32){
  if(id<64u&&!cm12CurrentMapFailed()){atomicAdd(&failures[2],1u);}
}
fn cm12RetainedDensityVector(at:u32)->vec3f{return vec3f(state[at],state[at+1u],state[at+2u]);}
fn cm12SolidVoxelFractionQ8(_q:vec3i)->u32{return 0u;}
fn cm12RetainedDensityRigidPointOpen(_point:vec3f)->bool{return true;}
fn cm12RetainedDensityPhiMetres(point:vec3f)->f32{
  if(any(point<vec3f(0.0))||any(point>vec3f(4.0))){return 1.0;}
  let delta=(point-cm12RetainedDensityVector(20u))/cm12RetainedDensityVector(24u);
  let sphere=0.5*state[24]*(dot(delta,delta)-1.0);
  return min(sphere,point.y-state[37]);
}
fn cm12CurrentMapEvaluate(point:vec3f,_bank:u32)->CurrentMapEvaluation{
  let jacobian=mat3x3f(vec3f(1.0,0.0,0.0),vec3f(p.transform.x,1.0,0.0),vec3f(0.0,0.0,1.0));
  return CurrentMapEvaluation(jacobian*point+vec3f(0.0,p.transform.y,0.0),jacobian);
}
fn cm12CurrentMapEvaluateIncrement(point:vec3f,bank:u32)->CurrentMapEvaluation{return cm12CurrentMapEvaluate(point,bank);}
fn cm12CurrentMapEvaluatePoint(point:vec3f,bank:u32)->vec3f{return cm12CurrentMapEvaluate(point,bank).point;}
fn cm12CurrentMapVelocity(point:vec3f)->vec3f{return vec3f(0.2*point.y,-0.7+0.13*point.x,0.11*point.z);}
fn cm12CurrentMapRangeOnFineSupport(q:vec3i,bank:u32)->mat2x3f{
  var lower=vec3f(1e6);var upper=vec3f(-1e6);
  for(var corner=0u;corner<8u;corner++){
    let offset=vec3f(f32(corner&1u),f32((corner>>1u)&1u),f32(corner>>2u));
    let point=cm12CurrentMapEvaluate(vec3f(q)+offset,bank).point;
    lower=min(lower,point);upper=max(upper,point);
  }
  return mat2x3f(lower,upper);
}
${createSparseCM12CurrentMapMeasureWGSL({ baseWords, dimensions: [4, 4, 4] })}` });
      const info = await module.getCompilationInfo();
      assert.deepEqual(info.messages.filter(m => m.type === "error").map(m => `${m.lineNum}:${m.linePos} ${m.message}`), []);
      const bindingLayout = device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ] });
      const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindingLayout] });
      const pipelines = await Promise.all(["integrateCurrentMapFineMeasure", "integrateCurrentMapFineMeasureCooperative"]
        .map(entryPoint => device!.createComputePipelineAsync({ layout: pipelineLayout, compute: { module, entryPoint } })));
      const allocate = (size: number, usage: GPUBufferUsageFlags) => {
        const buffer = device!.createBuffer({ size, usage }); buffers.push(buffer); return buffer;
      };
      const state = allocate(4 * words, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
      const parameters = allocate(32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      const failures = allocate(12, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
      const readback = allocate(16 * count + 12, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
      const bindings = device.createBindGroup({ layout: bindingLayout,
        entries: [state, parameters, failures].map((buffer, binding) => ({ binding, resource: { buffer } })) });
      const initial = new Float32Array(words).fill(-777);
      initial.fill(0, 0, baseWords); initial.set([1, 2, 1, 1]);
      initial.set([0, 0, 0, 4], 4); initial.set([4, 4, 4, 4], 8); initial.set([0, 0, 0, 4], 12);
      initial[16] = 2; initial.set([2, 2.13, 2], 20); initial.set([1.1, 1.1, 1.1], 24);
      initial[32] = 3; initial.set([0, .35, 0], 36);
      const run = async (cooperative: boolean, shear: number) => {
        device!.queue.writeBuffer(state, 0, initial);
        device!.queue.writeBuffer(failures, 0, new Uint32Array(3));
        device!.queue.writeBuffer(parameters, 0, new Float32Array([.1, 1, 0, 0, shear, .137, 0, 0]));
        const encoder = device!.createCommandEncoder(), pass = encoder.beginComputePass();
        pass.setPipeline(pipelines[cooperative ? 1 : 0]!); pass.setBindGroup(0, bindings);
        pass.dispatchWorkgroups(cooperative ? count : 1); pass.end();
        encoder.copyBufferToBuffer(state, 4 * (baseWords + 4 * count), readback, 0, 16 * count);
        encoder.copyBufferToBuffer(failures, 0, readback, 16 * count, 12);
        device!.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
        const mapped = readback.getMappedRange();
        const values = new Float32Array(mapped, 0, 4 * count).slice();
        const receipt = new Uint32Array(mapped, 16 * count, 3).slice(); readback.unmap();
        assert.equal(receipt[0], 0, `shear=${shear}, cooperative=${cooperative}, failure=${receipt}`);
        if (cooperative) assert.equal(receipt[2], count, "all wet and dry supports complete once");
        assert.ok(values.every(Number.isFinite) && values.every(v => v !== -777), "no candidate sentinel survives");
        return values;
      };
      for (const shear of [0, .17]) {
        const scalar = await run(false, shear), cooperative = await run(true, shear);
        let maximumDifference = 0;
        for (let i = 0; i < scalar.length; i++) maximumDifference = Math.max(maximumDifference, Math.abs(scalar[i]! - cooperative[i]!));
        assert.ok(maximumDifference < 2e-6, `same nodes, cuts, adaptive criteria and sum order: ${maximumDifference}`);
        t.diagnostic(`shear=${shear}, scalar/cooperative maximum difference=${maximumDifference}`);
      }
      assert.deepEqual(errors, []);
    } finally {
      for (const buffer of buffers) buffer.destroy();
      device?.destroy(); if (gpu) live.delete(gpu);
      await releaseWebGPUExclusiveLock();
    }
  });
