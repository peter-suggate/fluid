import assert from "node:assert/strict";
import test from "node:test";
import { createSvoDrySceneFragmentWGSL, SVO_DRY_WORLD_GI_CACHE_CONTRACT } from "../lib/svo/features/shading/program";

function producerBody():string{
  const source=createSvoDrySceneFragmentWGSL(.5,"hybrid","bounds","split");
  const start=source.indexOf("@compute @workgroup_size(8,8) fn dryWorldGiCacheMain");
  assert.ok(start>=0);const bodyStart=source.indexOf("{",start);let depth=1,end=bodyStart+1;
  for(;depth&&end<source.length;end++){if(source[end]==="{")depth++;else if(source[end]==="}")depth--;}
  return source.slice(start,end);
}

test("GI cache preference has a dedicated existing frame lane without moving diagnostics",()=>{
  assert.equal(SVO_DRY_WORLD_GI_CACHE_CONTRACT.cacheEnabledOffsetBytes,12);
  assert.equal(SVO_DRY_WORLD_GI_CACHE_CONTRACT.frameBytes,144);
  assert.equal(SVO_DRY_WORLD_GI_CACHE_CONTRACT.exhaustedConesOffsetBytes,20);
});

test("GPU reduced GI producer evaluates current bounce when persistence is off, and caches only when enabled",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
  const {withUniformDevice}=await import("./helpers/uniform-geometric");
  // Run the entire production producer with deterministic input textures and
  // gather/cache oracles. Only resource I/O is adapted to a compact fixture.
  const producer=producerBody()
    .replace("@compute @workgroup_size(8,8) fn dryWorldGiCacheMain(@builtin(global_invocation_id) globalId:vec3u)","fn evaluate(globalId:vec3u)")
    .replace("textureDimensions(dryWorldGiOutput)","vec2u(1)")
    .replace("textureLoad(dryPrepassGeometryTexture,coordinate,0)","vec4f(1,0,0,0)")
    .replace("textureLoad(dryPrepassIdentityTexture,coordinate,0)","vec4u(0)")
    .replaceAll("textureStore(dryWorldGiOutput,","storeOutput(");
  await withUniformDevice("GI persistence-independent producer",async device=>{
    const module=device.createShaderModule({code:`
const DRY_OWNER_NONE=65535u;const DRY_GBUFFER_MOTION_RIGID=1u;
struct DryGlobalIllumination{radiance:vec3f,visibility:f32,valid:u32}
struct Hit{t:f32,ownerId:u32,motionKind:u32,normal:vec3f}
struct Influence{bodyMask:u32,movingMask:u32,signature:u32}
struct Lookup{value:DryGlobalIllumination,hit:u32,claimSlot:u32,claimState:u32}
struct Frame{cacheEnabled:u32,invalidGiPages:atomic<u32>,exhaustedGiCones:atomic<u32>}
var<workgroup> dryWorldGiFrame:Frame;
var<private> dryWorldGiIgnoreRigidBodies:u32;var<private> dryWorldGiBodyMask:u32;
var<private> dryGiExhaustedCones:u32;var<private> dryPrepassGiState:u32;var<private> dryNodeMipAwaitingBuild:u32;
var<private> exhausted:u32;var<private> moving:u32;var<private> cached:u32;var<private> valid:u32;var<private> value:f32;
var<private> gathers:u32;var<private> finds:u32;var<private> inserts:u32;var<private> caseIndex:u32;
@group(0) @binding(0) var<storage,read_write> output:array<vec4f,21>;
fn dryPrepassUnpackHit(g:vec4f,i:u32)->Hit{return Hit(1,3u,0u,vec3f(0,1,0));}
fn dryWorldGiFrameRay(c:vec2u,d:vec2u)->mat2x3f{return mat2x3f(vec3f(0),vec3f(0,0,1));}
fn dryGeometricNormal(h:Hit)->vec3f{return h.normal;}
fn dryWorldGiBodyInfluence(p:vec3f,o:u32)->Influence{return Influence(0u,moving,7u);}
fn dryWorldGiKey(p:vec3f,n:vec3f,b:u32)->u32{return 11u;}
fn dryWorldGiFind(key:u32)->Lookup{finds++;return Lookup(DryGlobalIllumination(vec3f(.75),.5,1u),cached,0u,0u);}
fn dryWorldGiInsert(key:u32,slot:u32,state:u32,v:DryGlobalIllumination){inserts++;}
fn dryGlobalIlluminationFaced(p:vec3f,n:vec3f,g:vec3f,o:u32)->DryGlobalIllumination{gathers++;dryGiExhaustedCones=exhausted;return DryGlobalIllumination(vec3f(value,value*.5,value*.25),.625,valid);}
fn storeOutput(c:vec2i,v:vec4f){output[caseIndex*3u]=v;}
${producer}
@compute @workgroup_size(1) fn main(){
 for(var i=0u;i<7u;i++){
  caseIndex=i;exhausted=0u;gathers=0u;finds=0u;inserts=0u;moving=0u;cached=1u;valid=1u;value=.5;dryNodeMipAwaitingBuild=0u;
  dryWorldGiFrame.cacheEnabled=select(0u,1u,i==1u||i==3u||i==4u||i==6u);atomicStore(&dryWorldGiFrame.invalidGiPages,0u);atomicStore(&dryWorldGiFrame.exhaustedGiCones,0u);
  if(i==2u){value=.25;}if(i==3u){cached=0u;}if(i==4u){moving=1u;}if(i==5u){valid=0u;}if(i==6u){cached=0u;exhausted=2u;}
  evaluate(vec3u(0));output[i*3u+1u]=vec4f(f32(gathers),f32(finds),f32(inserts),f32(atomicLoad(&dryWorldGiFrame.invalidGiPages)));
  output[i*3u+2u]=vec4f(f32(atomicLoad(&dryWorldGiFrame.exhaustedGiCones)),0,0,0);
 }
}`});
    assert.deepEqual((await module.getCompilationInfo()).messages.filter(x=>x.type==="error"),[]);
    const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
    const out=device.createBuffer({size:336,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    const read=device.createBuffer({size:336,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
    try{
      const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:out}}]}));p.dispatchWorkgroups(1);p.end();e.copyBufferToBuffer(out,0,read,0,336);device.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);
      const data=[...new Float32Array(read.getMappedRange())];read.unmap();
      const expected=[
        [.5,.25,.125,.625,1,0,0,0], [.75,.75,.75,.5,0,1,0,0],
        [.25,.125,.0625,.625,1,0,0,0], [.5,.25,.125,.625,1,1,1,0],
        [.5,.25,.125,.625,1,0,0,0], [.5,.25,.125,-1,1,0,0,1], [.5,.25,.125,.625,1,1,0,0],
      ];
      expected.forEach((v,i)=>assert.deepEqual(data.slice(i*12,i*12+8),v,`GI producer case${i}`));
      assert.equal(data[6*12+8],2,"finite partial gather reports exhaustion without persisting");
    }finally{out.destroy();read.destroy();}
  });
});
