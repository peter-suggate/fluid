import assert from "node:assert/strict";
import test from "node:test";
import { uniformTerrainSamplingWGSL } from "../lib/methods/uniform/uniform-terrain-sampling.wgsl";
import { narrowBandTerrainContactWGSL } from "../lib/methods/uniform/uniform-particle-contact.wgsl";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { readMixedBuffer } from "./helpers/uniform-mixed-native-fields";

/** Exercise the production sampler and contact, not a CPU transcription.
 * The omitted solid kinds are absent in this isolated heightfield fixture. */
test("terrain edge patches preserve planar heights, normals and downhill particle contact",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("terrain edge contact",async device=>{
  for(const fixture of [
   {size:[8,8],slope:[-1/16,0],frame:[0,0,0],h:[0.02,0.02,0.02]},
   {size:[8,8],slope:[1/16,0],frame:[0,0,0],h:[0.02,0.02,0.02]},
   {size:[8,8],slope:[0,-1/16],frame:[0,0,0],h:[0.02,0.04,0.01]},
   {size:[8,8],slope:[0,0],frame:[0,0,0],h:[0.02,0.02,0.02]},
   {size:[1,8],slope:[0,1/16],frame:[1,-1,0],h:[0.02,0.02,0.02]},
   {size:[8,8],slope:[-1/16,1/32],frame:[-1,1,1],h:[0.02,0.02,0.02],analytic:true},
  ]){
   const [nx,nz]=fixture.size as [number,number], [sx,sz]=fixture.slope as [number,number];
   const offset=fixture.frame.map(v=>32*v),analytic="analytic" in fixture && fixture.analytic;
   const points=[[0,0],[0.01,0.01],[0.49,0.49],[0.5,0.5],[nx-0.5,nz-0.5],[nx-0.01,nz-0.01],[nx,nz],[nx/2,nz/2],[-0.75,nz+0.75]];
   const terrain=device.createTexture({size:[nx,nz],format:"r32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
   const heights=Float32Array.from({length:nx*nz},(_,i)=>17+sx*(i%nx+0.5)+sz*(Math.floor(i/nx)+0.5));
   device.queue.writeTexture({texture:terrain},heights,{bytesPerRow:4*nx},[nx,nz]);
   const output=device.createBuffer({size:points.length*64,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
   try{
    const module=device.createShaderModule({code:`
@group(0) @binding(0) var umSolidTerrain:texture_2d<f32>;
@group(0) @binding(1) var<storage,read_write> result:array<vec4f>;
struct Analytic{frame:vec4f,ground:vec4f} const umAnalyticTerrain=Analytic(vec4f(0,0,1,1),vec4f(0,0,0,${analytic?1:0}));
struct Params{hDt:vec4f} const params=Params(vec4f(${fixture.h.join(",")},1.0/60.0));
fn umFramePage()->vec3i{return vec3i(${fixture.frame.join(",")});}
fn owHeight(x:f32,z:f32)->f32{return 17.0+${sx}*x+${sz}*z;}
${uniformTerrainSamplingWGSL}
fn umBoundaryRecord(c:vec3i)->u32{return 0u;}
fn umBoundaryBase()->u32{return 0u;}
fn umBoundarySample(p:vec3f)->vec4f{return vec4f(0,1,0,1);}
fn umBoundarySampleAt(p:vec3f,c:vec3i,r:u32)->vec4f{return vec4f(0,1,0,1);}
fn umSolidHasTerrain()->bool{return true;}
fn umSolidTraceWorld(p:vec3f)->vec3f{return p;}
fn umBodyAt(p:vec3f)->i32{return -1;}
fn umSolidVoxel(c:vec3i)->bool{return false;}
fn umCellOpen(c:vec3i)->f32{return 1.0;}
${narrowBandTerrainContactWGSL}
@compute @workgroup_size(1) fn probe(@builtin(global_invocation_id) id:vec3u){
 let points=array<vec2f,${points.length}>(${points.map(p=>`vec2f(${p.join(",")})`).join(",")});
 let i=id.x;let point=points[i]-32.0*vec2f(umFramePage().xz);var q=vec3f(point.x,0,point.y);
 let surface=umTerrainSurface(q);result[4u*i]=vec4f(surface,0);result[4u*i+1u]=vec4f(nbTerrainNormal(surface),0);
 q.y=surface.x+0.001;var v=vec3f(0,-9.81/60.0,0);
 for(var step=0u;step<20u;step++){
  let contact=nbTerrainContact(q,q+v*params.hDt.w/params.hDt.xyz,v);q=contact.position;v=contact.velocity+vec3f(0,-9.81/60.0,0);
 }
 result[4u*i+2u]=vec4f(q,0);result[4u*i+3u]=vec4f(v,0);
}`});
    const layout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
    const pipeline=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint:"probe"}});
    const group=device.createBindGroup({layout,entries:[{binding:0,resource:terrain.createView()},{binding:1,resource:{buffer:output}}]});
    const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(points.length);pass.end();device.queue.submit([encoder.finish()]);
    const values=await readMixedBuffer(device,output);
    for(const [i,p] of points.entries()){
     const x=analytic?p[0]!:Math.max(0,Math.min(nx,p[0]!)),z=analytic?p[1]!:Math.max(0,Math.min(nz,p[1]!));
     const expected=[17+sx*x+sz*z-offset[1]!,analytic || (p[0]!>=0 && p[0]!<=nx)?sx:0,analytic || (p[1]!>=0 && p[1]!<=nz)?sz:0];
     expected.forEach((v,a)=>assert.ok(Math.abs(values[16*i+a]!-v)<3e-5,`size ${fixture.size}, slope ${fixture.slope}, point ${p}, component ${a}: ${values[16*i+a]} vs ${v}`));
     const normal=[-expected[1]!/fixture.h[0]!,1/fixture.h[1]!,-expected[2]!/fixture.h[2]!],length=Math.hypot(...normal);
     normal.forEach((v,a)=>assert.ok(Math.abs(values[16*i+4+a]!-v/length)<3e-5));
    }
    // Start inside the half-cell border on whichever edge is uphill.
    const axis=sx!==0?0:2,i=(axis===0?sx:sz)>0?5:1;
    const start=points[i]![axis===0?0:1]!-offset[axis]!;
    const displacement=values[16*i+8+axis]!-start;
    if(sx===0&&sz===0)assert.ok(Math.abs(displacement)<1e-6,"flat terrain must not create lateral motion");
    else assert.ok(displacement*Math.sign(-(sx||sz))>0.5,`particle must leave uphill border: ${displacement}`);
   }finally{terrain.destroy();output.destroy();}
  }
 });
});
