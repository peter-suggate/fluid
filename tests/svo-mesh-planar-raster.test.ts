import assert from "node:assert/strict";
import test from "node:test";
import { packPlanarBoundaryPatches, planarBoundaryWGSL, type PlanarBoundaryPatch } from "../lib/core/planar-boundary";
import { svoMeshPlanarRasterWGSL } from "../lib/svo/features/primary-visibility/svo-mesh-planar-raster";
import { withUniformDevice } from "./helpers/uniform-geometric";

test("GPU projected planar candidates match the full catalogue at every pixel", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice("projected planar catalogue parity", async device => {
    const slab = (x:number,y:number,z:number,angle:number,ownerId:number):PlanarBoundaryPatch => ({
      center_m:[x,y,z], normal:[Math.sin(angle),0,Math.cos(angle)], tangentU:[Math.cos(angle),0,-Math.sin(angle)],
      tangentV:[0,1,0], halfExtentU_m:.8, halfExtentV_m:.7, halfThickness_m:.08, materialId:ownerId+1, ownerId,
    });
    const patches = [slab(-.7,0,3,.7,1), slab(.6,.3,4,-.5,2), slab(0,0,2,0,99),
      slab(0,-.5,3.5,0,3), slab(0,-.5,3.5,0,4), slab(9,0,3,0,5), slab(0,0,-3,0,6)];
    const words=packPlanarBoundaryPatches(patches);
    const arena=device.createBuffer({size:words.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    const eye=device.createBuffer({size:16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    device.queue.writeBuffer(arena,0,words);
    const module=device.createShaderModule({code:`
${planarBoundaryWGSL}
@group(0) @binding(0) var<storage,read> dryPlanarBoundaries:array<PlanarBoundaryPatch>;
@group(0) @binding(1) var<storage,read> eye:vec4f;
struct Dry { planarBoundaries:vec4u } const dry=Dry(vec4u(${patches.length},1,64,0));
struct Uniforms { viewport:vec4f } const uniforms=Uniforms(vec4f(64,64,0,0));
const DRY_REVERSED_Z_NEAR_M=.001;const DRY_MISS=1e30;
const SVO_FEATURE_BOX_X=0u;const DRY_GBUFFER_FIELD_ANALYTIC=0u;const DRY_GBUFFER_MOTION_STATIC=0u;const SVO_GBUFFER_PRODUCER_RASTER_BACKGROUND=0u;
struct DryHit {t:f32,normal:vec3f,material:u32,owner:u32,feature:u32,field:u32,motion:u32,valid:u32,value:f32,aux:vec3u}
struct DryRasterPrimaryOut {@location(0) data:vec4u,@builtin(frag_depth) depth:f32}
fn cameraTanHalfFov()->f32{return .7;}
fn dryRasterPrimaryReset(){}
fn dryOpaqueOwnerSuppressed(owner:u32)->bool{return owner==99u;}
fn dryRasterPrimaryCamera()->mat4x3f{return mat4x3f(eye.xyz,vec3f(0,0,1),vec3f(1,0,0),vec3f(0,1,0));}
fn dryRasterPrimaryRay(pixel:vec2f,camera:mat4x3f)->vec3f{
 let uv=vec2f(pixel.x/64.0,1.0-pixel.y/64.0);let ndc=uv*2.0-1.0;
 return normalize(camera[1]+camera[2]*ndc.x*.7+camera[3]*ndc.y*.7);
}
fn dryRasterPrimarySurface(hit:DryHit,ro:vec3f,rd:vec3f,forward:vec3f,producer:u32)->DryRasterPrimaryOut{
 return DryRasterPrimaryOut(vec4u(bitcast<u32>(hit.t),hit.material|(hit.owner<<16u),bitcast<u32>(hit.normal.x),bitcast<u32>(hit.normal.y)),clamp(.001/(hit.t*dot(rd,forward)),0.0,1.0));
}
${svoMeshPlanarRasterWGSL}
@vertex fn fullVertex(@builtin(vertex_index) vertex:u32)->@builtin(position) vec4f{
 let corners=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));return vec4f(corners[vertex],0,1);
}
@fragment fn fullFragment(@builtin(position) pixel:vec4f)->DryRasterPrimaryOut{
 let camera=dryRasterPrimaryCamera();let rd=dryRasterPrimaryRay(pixel.xy,camera);var best=DRY_MISS;
 var result=DryRasterPrimaryOut(vec4u(0),0);
 for(var i=0u;i<dry.planarBoundaries.x;i++){
  let boundary=dryPlanarBoundaries[i];let exact=intersectPlanarBoundary(boundary,camera[0],rd,1e-4,best);
  let identity=planarBoundaryIdentity(boundary);if(exact.valid==0u||dryOpaqueOwnerSuppressed(identity>>16u)){continue;}
  best=exact.tHit;let hit=DryHit(best,exact.normal,identity&0xffffu,identity>>16u,exact.featureAxis,0,0,1,0,vec3u(0));
  result=dryRasterPrimarySurface(hit,camera[0],rd,camera[1],0);
 }return result;
}`});
    assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
    const layout=device.createBindGroupLayout({entries:[0,1].map(binding=>({binding,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage" as const}}))});
    const group=device.createBindGroup({layout,entries:[arena,eye].map((buffer,binding)=>({binding,resource:{buffer}}))});
    const pipelines=await Promise.all([false,true].map(projected=>device.createRenderPipelineAsync({
      layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),
      vertex:{module,entryPoint:projected?"meshPlanarVertex":"fullVertex"},
      fragment:{module,entryPoint:projected?"meshPlanarFragment":"fullFragment",targets:[{format:"rgba32uint"}]},
      primitive:{topology:projected?"triangle-strip":"triangle-list"},
      depthStencil:{format:"depth32float",depthWriteEnabled:true,depthCompare:projected?"greater-equal":"always"},
    })));
    const color=device.createTexture({size:[64,64],format:"rgba32uint",usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC});
    const depth=device.createTexture({size:[64,64],format:"depth32float",usage:GPUTextureUsage.RENDER_ATTACHMENT});
    const read=device.createBuffer({size:64*64*16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    try {
      // Outside, inside a slab, near-plane crossing, behind, and off-axis eyes.
      for(const origin of [[0,0,0],[-.7,0,3],[-.7,0,2.93],[0,0,5],[1.4,.4,2.5]]){
        device.queue.writeBuffer(eye,0,new Float32Array([...origin,0]));
        const results:Uint32Array[]=[];
        for(let projected=0;projected<2;projected++){
          const encoder=device.createCommandEncoder();const pass=encoder.beginRenderPass({
            colorAttachments:[{view:color.createView(),loadOp:"clear",storeOp:"store",clearValue:[0,0,0,0]}],
            depthStencilAttachment:{view:depth.createView(),depthLoadOp:"clear",depthStoreOp:"store",depthClearValue:0},
          });
          pass.setPipeline(pipelines[projected]!);pass.setBindGroup(0,group);pass.draw(projected?4:3,projected?patches.length:1);pass.end();
          encoder.copyTextureToBuffer({texture:color},{buffer:read,bytesPerRow:1024},[64,64]);device.queue.submit([encoder.finish()]);
          await read.mapAsync(GPUMapMode.READ);results.push(new Uint32Array(read.getMappedRange()).slice());read.unmap();
        }
        assert.deepEqual(results[1],results[0],`identical depth, material, owner and normal at eye ${origin}`);
      }
    } finally {read.destroy();color.destroy();depth.destroy();arena.destroy();eye.destroy();}
  });
});
