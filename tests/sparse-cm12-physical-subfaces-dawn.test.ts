import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";

const resident = readFileSync(new URL("../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url), "utf8");
const extension = readFileSync(new URL("../lib/methods/adaptive-mass/sparse-cm12-velocity-extension.wgsl.ts", import.meta.url), "utf8");
function production(name: string, source = resident): string {
  const result = source.match(new RegExp(`fn ${name}\\([\\s\\S]*?\\n}`))?.[0];
  assert.ok(result, name); return result;
}
const dawnTest = process.env.WEBGPU_NODE_MODULE ? test : test.skip;
const live = new Set<GPU>();
Object.assign(globalThis, { cm12PhysicalSubfaceTestGPUs: live });
dawnTest("capacity receipts follow physical face area and extension excludes same-side fine siblings", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "physical-subfaces");
  let gpu: GPU | undefined, device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
    Object.assign(globalThis, dawn.globals);
    gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); live.add(gpu!);
    const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice();
    const errors: string[] = [];
    device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); console.error(e.error.message); });
    const shader = device.createShaderModule({ code: `
const INVALID=0xffffffffu;
struct Params{counts:vec4u}
const p=Params(vec4u(10,0,0,0));
@group(0)@binding(0)var<storage,read_write>state:array<f32>;
@group(0)@binding(1)var<storage,read_write>conditioning:array<atomic<i32>>;
@group(0)@binding(2)var<storage,read_write>output:array<vec4f>;
fn destinationDensity()->u32{return 0u;}
fn cellTransportActive(c:u32)->bool{return c<10u;}
fn cellOpenFraction(c:u32)->f32{_=c;return 1.0;}
fn cellVolume(c:u32)->f32{return select(1.0,8.0,c<6u);}
fn cm12PhysicalMassFixedScale()->f32{return 1048576.0;}
fn dynamicallyCoveredCell(c:u32)->bool{_=c;return false;}
fn acceptedTemplateCellInvocation(c:u32)->u32{return select(INVALID,c,c<10u);}
// Coarse cell 0 has five ordinary width-two neighbours and four width-one
// neighbours across its sixth face. Every physical coarse face has area 4.
fn incidenceBegin(c:u32)->u32{return select(5u+c,0u,c==0u);}
fn incidenceEnd(c:u32)->u32{return select(6u+c,6u,c==0u);}
fn incidenceRow(at:u32)->u32{
  if(at<6u){return at;}return min(at-6u,5u);
}
fn incidenceTerm(at:u32)->u32{
  if(at<6u){return 2u*at;}
  if(at<11u){return 2u*(at-6u)+1u;}
  return at;
}
fn rowAccepted(r:u32)->bool{return r<6u;}
fn rowArea(r:u32)->f32{_=r;return 4.0;}
fn rowDistance(r:u32)->f32{return select(2.0,1.5,r==5u);}
fn rowTermOffset(r:u32)->u32{return 2u*r;}
fn rowTermCount(r:u32)->u32{return select(2u,5u,r==5u);}
fn termCell(t:u32)->u32{
  if(t<10u){return select(0u,t/2u+1u,(t&1u)!=0u);}
  return select(t-5u,0u,t==10u);
}
fn termCoefficient(t:u32)->f32{
  if(t<10u){return select(-0.5,0.5,(t&1u)!=0u);}
  return select(1.0/6.0,-2.0/3.0,t==10u);
}
fn cm12HotRowTermCoefficient(r:u32,o:u32)->f32{return termCoefficient(rowTermOffset(r)+o);}
fn cm12HotRowDistance(r:u32)->f32{return rowDistance(r);}
${production("cm12PhysicalSubfaceArea")}
${production("densityCapacityRepairMass")}
${production("densityCapacityRepairArea")}
${production("densityCapacityRepairShare")}
${production("scatterDensityCapacityRepairCellAtPlane")}
${production("cm12VelocityExtensionNeighborWeight", extension)}
@compute @workgroup_size(64)
fn scatter(@builtin(global_invocation_id)id:vec3u){
  if(id.x<10u){scatterDensityCapacityRepairCellAtPlane(id.x,0u);}
}
@compute @workgroup_size(64)
fn prepare(@builtin(global_invocation_id)id:vec3u){
  if(id.x>=10u){return;}
  atomicStore(&conditioning[50u+id.x],bitcast<i32>(densityCapacityRepairArea(id.x)));
  atomicStore(&conditioning[60u+id.x],densityCapacityRepairMass(id.x));
}
@compute @workgroup_size(64)
${production("gatherDensityCapacityRepair")}
@compute @workgroup_size(1)
fn weights(){
  output[0]=vec4f(cm12VelocityExtensionNeighborWeight(5u,1u,0u),
    cm12VelocityExtensionNeighborWeight(5u,1u,2u),
    cm12VelocityExtensionNeighborWeight(5u,0u,1u),
    cm12PhysicalSubfaceArea(5u,-2.0/3.0,1.0/6.0));
  output[1]=vec4f(f32(densityCapacityRepairShare(100000000,1.0,6.0)));
}
` });
    assert.deepEqual((await shader.getCompilationInfo()).messages.filter(m => m.type === "error").map(m => m.message), []);
    const state = device.createBuffer({ size: 40, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const receipts = device.createBuffer({ size: 280, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const output = device.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readback = device.createBuffer({ size: 112, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const layouts = [device.createBindGroupLayout({ entries: [0,1,2].map(binding => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } })) })];
    const layout = device.createPipelineLayout({ bindGroupLayouts: layouts });
    const group = device.createBindGroup({ layout: layouts[0]!, entries: [state,receipts,output].map((buffer,binding) => ({ binding,resource:{buffer} })) });
    const pipelines = Object.fromEntries(await Promise.all(["scatter","prepare","gatherDensityCapacityRepair","weights"].map(async entryPoint =>
      [entryPoint, await device!.createComputePipelineAsync({ layout, compute:{module:shader,entryPoint} }).catch(e=>{console.error(entryPoint,e.message);throw e;})])));
    for (const source of [0,6]) {
      const density = new Float32Array(10).fill(1); density[source] = source === 0 ? 1.75 : 2;
      device.queue.writeBuffer(state,0,density); device.queue.writeBuffer(receipts,0,new Int32Array(70));
      let e = device.createCommandEncoder();
      for(const entry of ["scatter","weights"]){const pass=e.beginComputePass();pass.setPipeline(pipelines[entry]!);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();}
      e.copyBufferToBuffer(receipts,0,readback,0,40);e.copyBufferToBuffer(output,0,readback,80,32);device.queue.submit([e.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const mapped=readback.getMappedRange();
      const result = new Int32Array(mapped.slice(0,40));
      const weights = new Float32Array(mapped.slice(80,112));
      assert.equal(result.reduce((a,b)=>a+b,0),0,"every integer debit has an identical credit");
      assert.equal(weights[1],0,"same-side fine sibling has no extension edge");
      assert.ok(Math.abs(weights[0]!-2/3)<1e-6 && Math.abs(weights[2]!-2/3)<1e-6);
      assert.ok(Math.abs(weights[3]!-1)<1e-6);
      assert.equal(weights[4],16666666,"large equal-area receipts must use the exact integer quotient");
      if(source===0){
        for(let cell=1;cell<10;cell++) assert.ok(Math.abs(result[cell]!/1048576-(cell<6?1:.25))<2e-6,`cell ${cell}: ${result[cell]}`);
      }else{
        assert.ok(result[0]!>0);for(let cell=1;cell<10;cell++) if(cell!==6)assert.equal(result[cell],0,`fine sibling ${cell} must not receive mass`);
      }
      readback.unmap();
      e=device.createCommandEncoder();
      for(const entry of ["prepare","gatherDensityCapacityRepair"]){const pass=e.beginComputePass();pass.setPipeline(pipelines[entry]!);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();}
      e.copyBufferToBuffer(state,0,readback,40,40);device.queue.submit([e.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const gathered=new Float32Array(readback.getMappedRange().slice(40,80));
      for(let cell=0;cell<10;cell++) assert.equal(gathered[cell],Math.fround(density[cell]!+result[cell]!/1048576/(cell<6?8:1)),`gather/scatter cell ${cell}`);
      readback.unmap();
    }
    state.destroy();receipts.destroy();output.destroy();readback.destroy();assert.deepEqual(errors,[]);
  } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); if(gpu)live.delete(gpu); }
});
