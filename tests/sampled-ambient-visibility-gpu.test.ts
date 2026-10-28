import assert from "node:assert/strict";
import test from "node:test";
import { meshVolumeWGSL } from "../lib/core/imported-assets/mesh-volume.wgsl";
import { sampledSurfaceProjectionWGSL } from "../lib/svo/features/lighting-visibility/sampled-surface-projection.wgsl";
import { sampledAmbientVisibilityWGSL } from "../lib/svo/features/lighting-visibility/sampled-ambient-visibility.wgsl";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { createUIStore } from "../lib/core/stores/ui-store";
import { lightingQuery } from "../lib/svo/features/lighting-visibility/persistence";
import { DEFAULT_SVO_LIGHTING_OPTIONS } from "../lib/svo/pipeline/svo-render-options";

test("native voxel AO is opt-in and persists independently of surface and lighting visibility",()=>{
  assert.equal(DEFAULT_SVO_LIGHTING_OPTIONS.nativeAmbientOcclusionEnabled,false);
  const store=createUIStore(),before=store.getState();assert.equal(before.svoNativeAmbientOcclusionEnabled,false);
  before.setSvoNativeAmbientOcclusionEnabled(true);
  assert.equal(store.getState().svoNativeAmbientOcclusionEnabled,true);
  assert.equal(store.getState().svoConeTracingMode,before.svoConeTracingMode);
  assert.equal(store.getState().svoPrimaryTraversal,before.svoPrimaryTraversal);
  const query=new URLSearchParams(),state=lightingQuery.read(query);state.svoNativeAmbientOcclusionEnabled=true;lightingQuery.write(query,state);
  assert.equal(lightingQuery.read(query).svoNativeAmbientOcclusionEnabled,true);
});

function fixture(blocker:boolean){
  const words=new Uint32Array(18+729*2),f=new Float32Array(words.buffer);f[3]=.1;words.set([8,8,8],4);f[7]=.2;words.set([1,1,1],8);words[16]=18;
  for(let i=0;i<729;i++){
    const y=Math.floor(i/9)%9*.1;
    // A sampled plane and a separate 5cm sheet in the same FVX owner. The
    // interpolated narrow sheet is thinner than a .5m merged-world AO voxel.
    f[18+i*2]=Math.min(y-.2,blocker?Math.abs(y-.5)-.025:Infinity);words[19+i*2]=0xffff8000;
  }return words;
}

test("GPU native FVX AO distinguishes empty space, thin blockers and unresolved rays",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
  await withUniformDevice("native FVX AO status",async device=>{
    const stride=fixture(false).length,words=new Uint32Array(stride*4);words.set(fixture(false));words.set(fixture(true),stride);
    const wall=fixture(false),wf=new Float32Array(wall.buffer);for(let i=0;i<729;i++)wf[18+i*2]=Math.min(wf[18+i*2]!, .6-i%9*.1);words.set(wall,stride*3);
    const data=device.createBuffer({size:words.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(data,0,words);
    const output=device.createBuffer({size:11*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),read=device.createBuffer({size:output.size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
    const module=device.createShaderModule({code:`
@group(0) @binding(0) var<storage,read> words:array<u32>;
@group(0) @binding(1) var<storage,read_write> output:array<vec4f>;
fn load(at:u32)->u32{return words[at];}
${meshVolumeWGSL("volume","load")}
${sampledSurfaceProjectionWGSL("load")}
${sampledAmbientVisibilityWGSL("load")}
fn rayResult(at:u32,r:SampledAoRay){output[at]=vec4f(f32(r.status),f32(r.steps),r.distance,0);}
fn aoResult(at:u32,r:SampledAmbientVisibility){output[at]=vec4f(r.visibility,f32(r.status),f32(r.steps),f32(r.blocked));}
@compute @workgroup_size(1) fn main(){
  let origin=vec3f(.4,.202,.4);let up=vec3f(0,1,0);
  rayResult(0u,sampledAoRay(0u,origin,up,.5,96u));
  rayResult(1u,sampledAoRay(${stride}u,origin,up,.5,96u));
  rayResult(2u,sampledAoRay(0u,origin,up,.5,1u));
  rayResult(3u,sampledAoRay(${stride*2}u,origin,up,.5,96u));
  aoResult(4u,sampledAmbientVisibility(0u,vec3f(.4,.2,.4),.5,96u));
  aoResult(5u,sampledAmbientVisibility(${stride}u,vec3f(.4,.2,.4),.5,96u));
  aoResult(6u,sampledAmbientVisibility(${stride}u,vec3f(.4,.2,.4),.5,1u));
  aoResult(7u,sampledAmbientVisibility(0u,vec3f(9),.5,96u));
  rayResult(8u,sampledAoRay(0u,vec3f(.1,.202,.4),vec3f(1,0,0),.55,96u));
  rayResult(9u,sampledAoRay(0u,vec3f(.1,.202,.4),vec3f(1,0,0),.55,512u));
  rayResult(10u,sampledAoRay(${stride*3}u,vec3f(.1,.21,.4),vec3f(1,0,0),.55,512u));
}`});
    assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
    const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}}),group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:data}},{binding:1,resource:{buffer:output}}]});
    const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();encoder.copyBufferToBuffer(output,0,read,0,read.size);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);const values=Array.from(new Float32Array(read.getMappedRange()));
    assert.equal(values[0],0,"empty plane ray clears");assert.equal(values[4],1,"thin same-owner sheet blocks");
    assert.ok(Math.abs(values[6]!-.273)<.001,"blocker intersection uses the actual trilinear zero surface");
    assert.equal(values[8],2,"budget exhaustion is distinct from clear");assert.equal(values[12],3,"invalid volume is distinct from clear");
    assert.equal(values[16],1,"empty plane receives no added AO");assert.equal(values[17],0);assert.ok(values[18]!<=384);
    assert.ok(values[20]!<1,"thin sheet produces measured occlusion");assert.equal(values[21],1);assert.ok(values[23]!>0);
    assert.equal(values[25],2,"incomplete four-ray estimate is unavailable, never accepted darkness");assert.equal(values[29],3,"invalid projection is unavailable");
    assert.equal(values[32],2,"short budget explicitly cannot certify a long grazing segment");
    assert.equal(values[36],0,"bounded quality budget certifies the same empty grazing segment");assert.ok(values[37]!>96&&values[37]!<=512);
    assert.equal(values[40],1,"larger reach preserves a farther same-owner wall");assert.ok(Math.abs(values[42]!-.5)<.001);
    read.unmap();for(const b of[data,output,read])b.destroy();
  });
});

test("native AO control is placed and rendered in the shared Lighting section",async()=>{
  const {SVO_LIGHTING_VISIBILITY_FEATURE:feature}=await import("../lib/svo/features/lighting-visibility/definition");
  const {SVO_FEATURE_VIEWS}=await import("../lib/svo/pipeline/ui-slots");
  assert.ok(feature.placements.some(p=>p.slot==="view.lighting"&&p.control==="native-ao"));
  assert.ok(SVO_FEATURE_VIEWS["svo.lighting-visibility/native-ao"]);
});

test("GPU native AO diagnostic resolves actual receiver owners and explicitly skips sculpted fields",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
  const {probeNativeAo}=await import("../tools/probe-native-ao");
  const {SVO_DRY_SCENE_ARENA_LAYOUT:A}=await import("../lib/svo/features/shading/program");
  const {SVO_PRIMITIVE_KINDS}=await import("../lib/svo/contracts/svo-primitive-abi");
  await withUniformDevice("native AO diagnostic publication",async device=>{
    const volume=fixture(false),base=A.sizeBytes/4,words=new Uint32Array(base+volume.length),f=new Float32Array(words.buffer),primitive=A.primitiveOffsetBytes/4,field=A.fieldProgramOffsetBytes/4;
    words[primitive+3]=SVO_PRIMITIVE_KINDS.fieldProgram;words[primitive+7]=12<<16;f[primitive+11]=1;words[primitive+13]=field;
    words[field]=0xfffffffe;words[field+1]=base-field;words.set(volume,base);
    const sceneArenaBuffer=device.createBuffer({size:words.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(sceneArenaBuffer,0,words);
    const params=new Uint32Array(176),pf=new Float32Array(params.buffer);params.set([1,12,0,0],12);pf.set([.5,.5,.5],4);pf[90]=1;
    const paramsBuffer=device.createBuffer({size:params.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(paramsBuffer,0,params);
    const splitGeometry=device.createTexture({size:[1,1],format:"rgba32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});device.queue.writeTexture({texture:splitGeometry},new Float32Array([0,1,0,1]),{bytesPerRow:16},[1,1]);
    const splitOpaqueIdentity=device.createTexture({size:[1,1],format:"rg32uint",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});device.queue.writeTexture({texture:splitOpaqueIdentity},new Uint32Array([7,12]),{bytesPerRow:8},[1,1]);
    const view=new Float32Array([1,1,0,0,.4,.2,1.4,.4,.4,.2,.4,0,64,8,80,0]);
    const pipeline={sceneArenaBuffer,paramsBuffer,splitGeometry,splitOpaqueIdentity};
    const clear=await probeNativeAo(device,pipeline,view);assert.equal(clear.counts.clear,16000);assert.equal(clear.counts.exhausted,undefined);
    device.queue.writeBuffer(sceneArenaBuffer,A.sculptCutOffsetBytes,new Uint32Array([1]));
    const sculpted=await probeNativeAo(device,pipeline,view);assert.equal(sculpted.counts.notApplicable,16000,"edited surfaces keep existing AO rather than querying the uncut source");
    for(const b of[sceneArenaBuffer,paramsBuffer])b.destroy();splitGeometry.destroy();splitOpaqueIdentity.destroy();
  });
});

function sparseSkipFixture(blocker: boolean, missingDistance = .002) {
  const words=new Uint32Array(22+2*729*2),f=new Float32Array(words.buffer);
  f[3]=.1;words.set([24,8,8],4);f[7]=.002;words.set([3,1,1],8);
  words[16]=22;f[19]=missingDistance;words[20]=22+729*2;
  for(let brick=0;brick<2;brick++)for(let i=0;i<729;i++){
    const x=(i%9+brick*16)*.1;
    f[22+brick*729*2+i*2]=blocker?Math.min(.002,Math.abs(x-2)-.025):.002;
    words[23+brick*729*2+i*2]=0xffff8000;
  }
  return words;
}

test("GPU native AO skips only proven-empty cells and bricks without skipping thin or constant-solid blockers",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
  await withUniformDevice("native AO empty-cell DDA",async device=>{
    const stride=sparseSkipFixture(false).length,words=new Uint32Array(stride*5);
    words.set(sparseSkipFixture(false));words.set(sparseSkipFixture(true),stride);
    words.set(sparseSkipFixture(false,-.002),stride*2);words.set(sparseSkipFixture(false,NaN),stride*3);
    const broad=sparseSkipFixture(false,.8),bf=new Float32Array(broad.buffer);
    for(let brick=0;brick<2;brick++)for(let i=0;i<729;i++)bf[22+brick*729*2+i*2]=.8;
    words.set(broad,stride*4);
    const data=device.createBuffer({size:words.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(data,0,words);
    const output=device.createBuffer({size:8*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    const read=device.createBuffer({size:output.size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
    const module=device.createShaderModule({code:`
@group(0) @binding(0) var<storage,read> words:array<u32>;
@group(0) @binding(1) var<storage,read_write> output:array<vec4f>;
fn load(at:u32)->u32{return words[at];}
${sampledSurfaceProjectionWGSL("load")}
${sampledAmbientVisibilityWGSL("load")}
fn write(at:u32,r:SampledAoRay){output[at]=vec4f(f32(r.status),f32(r.steps),r.distance,0);}
@compute @workgroup_size(1) fn main(){
  // The scalar field is positive but only .002m wide: distance-only marching
  // would need over2000 steps. Convex-cell skipping certifies it in <=17.
  let p=vec3f(.05,.300001,.4);let direction=normalize(vec3f(1,.00001,0));
  write(0u,sampledAoRay(0u,p,direction,2.3,96u));
  write(1u,sampledAoRay(${stride}u,p,direction,2.3,96u));
  write(2u,sampledAoRay(${stride*2}u,p,direction,2.3,96u));
  write(3u,sampledAoRay(${stride*3}u,p,direction,2.3,96u));
  write(4u,sampledAoRay(${stride}u,vec3f(2.35,.3,.4),vec3f(-1,0,0),2.3,96u));
  write(5u,sampledAoRay(0u,vec3f(.8,.3,.4),vec3f(-1,0,0),.75,96u));
  write(6u,sampledAoRay(0u,p,direction,2.3,1u));
  write(7u,sampledAoRay(${stride*4}u,p,direction,2.3,96u));
}`});
    assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
    const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
    const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:data}},{binding:1,resource:{buffer:output}}]});
    const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();encoder.copyBufferToBuffer(output,0,read,0,read.size);device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);const result=Array.from(new Float32Array(read.getMappedRange()));
    assert.equal(result[0],0,"grazing ray through proven positive cells and air brick clears");assert.ok(result[1]!<=17,"cell/brick certificate avoids tiny scalar-distance steps");
    assert.equal(result[4],1,"empty brick skip retains farther thin blocker");assert.ok(result[6]!>1.85&&result[6]!<1.87);
    assert.equal(result[8],1,"absent constant-solid brick must block");assert.ok(Math.abs(result[10]!-.75)<1e-5);
    assert.equal(result[12],3,"nonfinite absent brick is unavailable");
    assert.equal(result[16],1,"negative-direction DDA retains thin blocker");assert.ok(result[18]!>.25&&result[18]!<.27);
    assert.equal(result[20],0,"exact brick boundary in reverse direction terminates without an epsilon nudge");
    assert.equal(result[24],2,"DDA remains status-aware at its step budget");
    assert.equal(result[28],0);assert.ok(result[29]!<=6,"large conservative scalar steps are not clipped to every native cell");
    read.unmap();for(const buffer of[data,output,read])buffer.destroy();
  });
});
