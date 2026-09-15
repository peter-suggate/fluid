import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { gridOverlayShader } from "../lib/core/webgpu-grid-overlay";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
const live = new Set<GPU>();
(process.env.WEBGPU_NODE_MODULE ? test : test.skip)("coarse slice addresses all rungs and sparse-air clearance refuses represented pages", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "coarse-readers");
  let device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href); Object.assign(globalThis, dawn.globals);
    const gpu = dawn.create(["backend=metal"]); live.add(gpu);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter); device = await adapter.requestDevice(); assert.ok(device);
    const source = readFileSync(new URL("../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url), "utf8");
    const extract = (s: string, name: string) => s.match(new RegExp(`fn ${name}\\([\\s\\S]*?\\n}`))![0];
    const code = `
struct Overlay { worldDirectory:vec4u, dynamicCells:vec4u, rungOffsets:vec4u }
struct Params { counts:vec4u, dispatch:vec4u }
var<private>sparseOverlayP:Overlay;var<private>sparseP:Params;
var<private>sparseTopologyArena:array<u32,32>;
const INVALID=0xffffffffu;const BRICK_FINE_RESOLUTION=8u;
fn sparseWorldDirectoryEnabled()->bool{return true;}
fn sparseBrickFineResolution()->u32{return 8u;}
fn lsvAcceptedSlot()->u32{return 0u;}
// The nonnegative x half-space is represented, even if its phi hash is broken.
fn brickDirectoryLookupAtSignedCoordinate(q:vec3i)->u32{return select(INVALID,0u,q.x>=0);}
fn lsvBrickPhiResolution(slot:u32,brick:u32)->u32{return 2u;}
${extract(gridOverlayShader,"sparseTemplateLevelIndex")}
${extract(gridOverlayShader,"sparseTemplateCellRange")}
${extract(source,"cm12UnrepresentedAirClearance")}
struct ResidentParams {dispatch:vec4u}
const p=ResidentParams(vec4u(0u,0u,0u,2u));
fn brickActive(brick:u32)->bool{return true;}
fn acceptedBrickResolution(brick:u32)->u32{return 2u;}
fn cm12PhiBandBrick(brick:u32)->bool{return true;}
fn templateBrickCellRange(brick:u32,resolution:u32)->vec2u{
 // Fixed page 0 aliases every rung to B2; page 1 has a real B8 entry.
 if(brick==1u&&resolution==8u){return vec2u(80u,512u);}
 return vec2u(20u,8u);
}
fn cellResolution(cell:u32)->u32{return select(2u,8u,cell==80u);}
${extract(source,"cm12PhiBrickPlan")}
var<workgroup> activity:array<atomic<u32>,128>;
const ACTIVITY_SURFACE_PROOF_FAILURE_WORD=4u;
const CM12_WDR_INITIAL_LEAVES=2u;
const ACTIVITY_LIFECYCLE_CHANGED=16u;
fn activityRecord(brick:u32)->u32{return brick*64u;}
fn brickCandidatePlanningEnabled(brick:u32)->bool{return brick==1u;}
fn applySparseCM12RefinementRegionBounds(brick:u32,resolution:u32)->u32{return resolution;}
fn setCandidateBrickActiveAt(output:u32,enabled:bool){atomicStore(&activity[output+10u],select(0u,1u,enabled));}
${extract(source,"cachedRefinementGradingCap")}
${extract(source,"setRefinementGradingCap")}
${extract(source,"stageFrontierPageAtRung")}


@group(0)@binding(0)var<storage,read_write>result:array<vec4u>;
@compute @workgroup_size(1)fn main(){
 sparseOverlayP.worldDirectory=vec4u(0u,1u,1u,48u);
 sparseOverlayP.dynamicCells=vec4u(585u,0u,0u,0u);sparseOverlayP.rungOffsets=vec4u(584u,576u,512u,0u);
 sparseP.counts=vec4u(10000u);sparseTopologyArena[2u]=1000u;
 for(var level=0u;level<4u;level+=1u){result[level]=vec4u(sparseTemplateCellRange(3u,1u<<level),0u,0u);}
 sparseOverlayP.dynamicCells.x=0u;
 result[4]=vec4u(sparseTemplateCellRange(3u,8u),sparseTemplateCellRange(3u,2u));
 setRefinementGradingCap(0u,8u);setRefinementGradingCap(1u,8u);
 stageFrontierPageAtRung(0u,1u);stageFrontierPageAtRung(1u,1u);
 result[7]=vec4u(cachedRefinementGradingCap(0u),cachedRefinementGradingCap(1u),
 atomicLoad(&activity[8u]),atomicLoad(&activity[72u]));
 result[6]=vec4u(cm12PhiBrickPlan(0u,true),cm12PhiBrickPlan(1u,true));
 result[5]=bitcast<vec4u>(vec4f(cm12UnrepresentedAirClearance(vec3f(-1.5,4.0,4.0)),
 cm12UnrepresentedAirClearance(vec3f(-0.2,4.0,4.0)),cm12UnrepresentedAirClearance(vec3f(0.0,4.0,4.0)),
 cm12UnrepresentedAirClearance(vec3f(4.0,4.0,4.0))));
}`;
    const module = device.createShaderModule({ code });
    assert.deepEqual((await module.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
    const output = device.createBuffer({ size: 128, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readback = device.createBuffer({ size: 128, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: output } }] });
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();
    encoder.copyBufferToBuffer(output,0,readback,0,128);device.queue.submit([encoder.finish()]);await readback.mapAsync(GPUMapMode.READ);
    const raw = readback.getMappedRange();const words = new Uint32Array(raw);
    for (let level=0;level<4;level++) assert.deepEqual([...words.slice(4*level,4*level+2)], [2170+[584,576,512,0][level]!, (1<<level)**3]);
    assert.deepEqual([...words.slice(16,20)], [2024,512,0,0]);
    const f = new Float32Array(raw);assert.equal(f[20],.5);assert.ok(Math.abs(f[21]!-.2)<1e-6);assert.equal(f[22],0);assert.equal(f[23],0);
    assert.deepEqual([...words.slice(24,28)], [8,2,512,8], "aliased fine entry must retain coarse phi geometry");
    assert.deepEqual([...words.slice(28,32)], [2,8,2,1], "fixed pages cap grading and stage only their prepared rung");
    readback.unmap();output.destroy();readback.destroy();
  } finally { device?.destroy();live.clear();await releaseWebGPUExclusiveLock(); }
});
