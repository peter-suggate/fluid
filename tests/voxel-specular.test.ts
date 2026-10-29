import assert from "node:assert/strict";
import test from "node:test";
import {voxelSpecularWGSL} from "../lib/svo/features/radiance/voxel-specular.wgsl";
import {svoTetrahedralRadianceConeWGSL} from "../lib/svo/features/radiance/svo-tetrahedral-radiance-cone";
import {createSvoDrySceneFragmentWGSL,SVO_DRY_VISIBILITY_FLAGS} from "../lib/svo/features/shading/program";
import {createUIStore} from "../lib/core/stores/ui-store";
import {svoFeatureQuery} from "../lib/svo/pipeline/persistence";
import {SVO_LIGHTING_VISIBILITY_FEATURE} from "../lib/svo/features/lighting-visibility/definition";

test("voxel reflections are opt-in, persistent, and select their required cone/GI producer",()=>{
 const store=createUIStore();assert.equal(store.getState().svoVoxelSpecularReflectionsEnabled,false);
 assert.equal(store.getState().svoVoxelSpecularSteps,128);
 store.getState().setSvoVoxelSpecularReflectionsEnabled(true);
 assert.equal(store.getState().svoConeTracingMode,"cones");assert.equal(store.getState().svoGlobalIlluminationEnabled,true);
 store.getState().setSvoVoxelSpecularSteps(512);
 const query=new URLSearchParams();svoFeatureQuery.write(query,store.getState());
 const decoded=svoFeatureQuery.read(query);assert.equal(decoded.svoVoxelSpecularReflectionsEnabled,true);assert.equal(decoded.svoVoxelSpecularSteps,512);
 assert.throws(()=>store.getState().setSvoVoxelSpecularSteps(513),RangeError);
 for(const control of ["lens-bloom","voxel-specular","global-illumination"])assert.ok(SVO_LIGHTING_VISIBILITY_FEATURE.placements.some(p=>p.slot==="view.lighting"&&p.control===control));
});

test("GPU reflected cone gathers emitter plus transmitted sky, reports exhaustion and missing pages",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 const {withUniformDevice}=await import("./helpers/uniform-geometric");
 await withUniformDevice("voxel specular closure",async device=>{
  for(const scale of [1,.5] as const){
   const production=device.createShaderModule({code:createSvoDrySceneFragmentWGSL(scale,"hybrid","bounds","split")});
   assert.deepEqual((await production.getCompilationInfo()).messages.filter(m=>m.type==="error"),[],"full production shader validates");
  }
  const productionSource=createSvoDrySceneFragmentWGSL(1,"hybrid","bounds","split");
  const start=productionSource.indexOf("fn dryOpaqueSpecular(");
  const productionClosure=productionSource.slice(start,productionSource.indexOf("\n}",start)+2);
  assert.ok(productionClosure.includes("return sky*brdf*legacyVisibility"));
  const code=`${svoTetrahedralRadianceConeWGSL}\n${voxelSpecularWGSL}
@group(0) @binding(0) var<storage,read> config:array<vec4f>;
@group(0) @binding(1) var<storage,read_write> out:array<vec4f>;
var<private> mode:u32;
struct Mapping{cellSize:vec3f}
struct Params{materialPublication:vec4u,mapping:Mapping,tuningRays1:vec4f,giCones:vec4f}
struct DryNodeMipPageCache{a:vec3u,b:u32,c:vec3u,d:u32,e:u32,f:u32,g:u32}
var<private> dry:Params;
struct View{options:vec4f}
var<private> uniforms:View;
var<private> dryGiPageCache:DryNodeMipPageCache;
var<private> dryGiRadiancePageCache:DryNodeMipPageCache;
var<private> dryDerivedPageFailure:u32;
@group(0) @binding(2) var<storage,read_write> drySpecularCounters:array<atomic<u32>,4>;
fn dryEnvironment(direction:vec3f,roughness:f32)->vec3f{return vec3f(.1,.2,.3);}
fn dryNodeMipSceneExitDistance(position:vec3f,direction:vec3f)->f32{return 10.0;}
${productionClosure}
fn svoTetraRadianceConeLoad(q:SvoTetraRadianceConeQuery)->SvoTetraRadianceConeSourceSample{
 let coverage=select(0.0,1.0,mode!=0u&&q.distance_m>=2.0);
 let c=vec3f(2.0,.5,.25)*coverage;
 return SvoTetraRadianceConeSourceSample(coverage,SvoTetraRadiance(c,c,c,c),1u,select(1u,0u,mode==3u));
}
@compute @workgroup_size(1) fn main(){
 for(var i=0u;i<4u;i++){
  mode=u32(config[i].x);
  let v=voxelSpecularTrace(vec3f(0),vec3f(0,0,1),config[i].y,.125,10.0,u32(config[i].z));
  out[i*2u]=vec4f(voxelSpecularCompose(v,vec3f(.1,.2,.3)),v.transmittance);
  out[i*2u+1u]=vec4f(f32(v.valid),f32(v.exhausted),voxelSpecularAperture(config[i].y),0);
 }
 dry.materialPublication=vec4u(0u);
 let brdf=config[1].xyz*.1;let legacyVisibility=config[1].y;
 out[8]=vec4f(dryOpaqueSpecular(vec3f(0),vec3f(0,1,0),vec3f(0,0,1),.5,brdf,legacyVisibility),0);
 out[9]=vec4f(dryEnvironment(vec3f(0,0,1),.5)*brdf*legacyVisibility,0);
 dry.materialPublication.w=${SVO_DRY_VISIBILITY_FLAGS.voxelSpecular}u;
 uniforms.options.z=config[1].x;
 out[10]=vec4f(dryOpaqueSpecular(vec3f(0),vec3f(0,1,0),vec3f(0,0,1),.5,brdf,legacyVisibility),0);
 uniforms.options.z=0.0;dry.mapping.cellSize=vec3f(.125);dry.giCones.z=config[1].z;
 mode=3u;
 out[11]=vec4f(dryOpaqueSpecular(vec3f(0),vec3f(0,1,0),vec3f(0,0,1),.5,brdf,legacyVisibility),0);
 out[12]=vec4f(f32(atomicLoad(&drySpecularCounters[0])),f32(atomicLoad(&drySpecularCounters[1])),f32(atomicLoad(&drySpecularCounters[2])),f32(atomicLoad(&drySpecularCounters[3])));
}`;
  const module=device.createShaderModule({code});assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
  const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
  const input=device.createBuffer({size:64,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  const output=device.createBuffer({size:208,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const counters=device.createBuffer({size:16,usage:GPUBufferUsage.STORAGE});
  const read=device.createBuffer({size:208,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  device.queue.writeBuffer(input,0,new Float32Array([0,.1,512,0, 1,.5,128,0, 1,0,1,0, 3,.5,128,0]));
  try{
   const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:input}},{binding:1,resource:{buffer:output}},{binding:2,resource:{buffer:counters}}]}));p.dispatchWorkgroups(1);p.end();e.copyBufferToBuffer(output,0,read,0,208);device.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);
   const v=new Float32Array(read.getMappedRange()).slice();read.unmap();
   for(let k=0;k<3;k++)assert.ok(Math.abs(v[k]!-[.1,.2,.3][k]!)<1e-6,"empty scene preserves sky");assert.equal(v[3],1);assert.equal(v[4],1);assert.equal(v[5],0);
   assert.ok(v[8]!>1.99&&v[9]!>.49&&v[10]!>.24,"opaque emitter replaces sky");assert.equal(v[11],0);assert.equal(v[12],1);
   assert.equal(v[21],1,"finite budget exhaustion is explicit");assert.equal(v[20],1);
   assert.equal(v[28],0,"missing radiance is invalid, never silently valid black");
   assert.deepEqual(v.slice(32,36),v.slice(36,40),"disabled production closure is bit-identical to the existing sky expression");
   assert.deepEqual(v.slice(40,44),v.slice(36,40),"rigid-body scene explicitly retains legacy shading");
   assert.deepEqual(v.slice(44,48),v.slice(36,40),"invalid source retains legacy occlusion, not unoccluded sky");
   assert.deepEqual(Array.from(v.slice(48,52)),[1,0,1,1],"invalid and unsupported closures are separately counted");
  }finally{input.destroy();output.destroy();read.destroy();counters.destroy();}
 });
});
