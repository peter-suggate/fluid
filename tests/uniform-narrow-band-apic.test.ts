import assert from "node:assert/strict";
import test from "node:test";
import { narrowBandParticleStride, narrowBandApicMomentWGSL } from "../lib/methods/uniform/uniform-narrow-band-apic.wgsl";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { readMixedBuffer } from "./helpers/uniform-mixed-native-fields";

test("GPU quadratic APIC reproduces translation, rotation and shear on staggered faces",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("APIC moments",async device=>{
  const rows=[[0,0,0],[0,-2,0],[2,0,0],[0.7,-0.3,1.2]];
  const points=[[0,0,0],[0.5,0.5,0.5],[3.25,4.75,2.1],[-1.31,0.001,8.4999]];
  const module=device.createShaderModule({code:`
@group(0) @binding(0) var<storage,read_write> out:array<vec4f>;
var<private> row:vec3f;
fn nbApicTap(index:vec3i,axis:u32)->f32{
 var offset=vec3f(0.5);offset[axis]=1.0;return 1.7+dot(row,vec3f(index)+offset);
}
${narrowBandApicMomentWGSL}
@compute @workgroup_size(1) fn main(){
 let rows=array<vec3f,4>(${rows.map(v=>`vec3f(${v.join(",")})`).join(",")});
 let points=array<vec3f,4>(${points.map(v=>`vec3f(${v.join(",")})`).join(",")});
 for(var r=0u;r<4u;r++){row=rows[r];for(var p=0u;p<4u;p++){for(var a=0u;a<3u;a++){
  out[12u*r+3u*p+a]=nbApicRow(points[p],a);
 }}}
}`});
  const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
  const output=device.createBuffer({size:48*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  try{
   const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);
   pass.setBindGroup(0,device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:output}}]}));pass.dispatchWorkgroups(1);pass.end();device.queue.submit([encoder.finish()]);
   const values=await readMixedBuffer(device,output);
   rows.forEach((row,r)=>points.forEach((point,p)=>{for(let a=0;a<3;a++){
    const expected=[...row,1.7+row.reduce((v,c,i)=>v+c*point[i]!,0)],start=4*(12*r+3*p+a);
    expected.forEach((v,k)=>assert.ok(Math.abs(values[start+k]!-v)<2e-5,`row ${r}, point ${p}, face ${a}, component ${k}: ${values[start+k]} vs ${v}`));
    // Reconstruct velocity at a different face from the recovered C and v.
    const d=[0.6,-0.9,1.1],reconstructed=values[start+3]!+d.reduce((sum,v,k)=>sum+v*values[start+k]!,0);
    assert.ok(Math.abs(reconstructed-(1.7+row.reduce((v,c,k)=>v+c*(point[k]!+d[k]!),0)))<5e-5);
   }}));
  }finally{output.destroy();}
 });
});

test("production APIC fine gather reproduces an affine MAC field in both reduction paths",{skip:!process.env.WEBGPU_NODE_MODULE},async t=>{
 const {narrowBandFineTransferSource}=await import("../lib/methods/uniform/uniform-narrow-band-surface.wgsl");
 const {narrowBandParticleWGSL}=await import("../lib/methods/uniform/uniform-narrow-band-apic.wgsl");
 await withUniformDevice("APIC particle to grid",async device=>{
  const rows=[[0.3,-0.8,0.2],[0.7,0.1,-0.4],[-0.2,0.9,0.5]],shift=[1.2,-0.7,0.3];
  const stride=narrowBandParticleStride(true),data=new Float32Array(512*stride);
  for(let i=0;i<512;i++){
   const p=[i%8+0.25,Math.floor(i/8)%8+0.37,Math.floor(i/64)+0.61];data.set(p,stride*i);
   rows.forEach((row,a)=>{data[stride*i+4+a]=shift[a]!+row.reduce((sum,v,k)=>sum+v*p[k]!,0);data.set(row,stride*i+12+4*a);});
  }
  const particles=device.createBuffer({size:data.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(particles,0,data);
  const dummy=device.createBuffer({size:4096,usage:GPUBufferUsage.STORAGE});
  const velocity=device.createTexture({size:[8,8,8],dimension:"3d",format:"rgba32float",usage:GPUTextureUsage.TEXTURE_BINDING});
  const output=device.createTexture({size:[8,8,8],dimension:"3d",format:"rgba32float",usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC});
  const read=device.createBuffer({size:256*8*8,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{for(const subgroups of [false,...(device.features.has("subgroups")&&(device.adapterInfo.subgroupMinSize??0)>=32?[true]:[])]){
   t.diagnostic(`P2G path: ${subgroups?"subgroup":"workgroup"}`);
   const module=device.createShaderModule({code:`${subgroups?"enable subgroups;":""}
${narrowBandParticleWGSL(true)}
@group(0) @binding(0) var<storage,read> particles:array<Particle>;
@group(0) @binding(1) var<storage,read_write> bins:array<atomic<u32>>;
@group(0) @binding(2) var<storage,read_write> state:array<atomic<u32>>;
@group(0) @binding(3) var velocity:texture_3d<f32>;
@group(0) @binding(4) var output:texture_storage_3d<rgba32float,write>;
struct UMOwner{index:u32,width:u32,tile:u32}
const UM_CELL_KEYS=512u;const UM_RESIDENT=1u;const umCounts=vec4u(8,0,0,1);
fn umAllOwner(id:vec3u)->UMOwner{return UMOwner(id.x,select(0u,1u,id.x<512u),0);}
fn umOrigin(o:UMOwner)->vec3u{return vec3u(o.index%8u,(o.index/8u)%8u,o.index/64u);}
fn nbTransferPropertiesFit()->bool{return false;}
fn umAcrossKind(p:vec3i,a:u32,s:i32)->u32{return 1u;}
fn particleDepth(p:vec3f)->f32{return -1.0;}
fn nbTheta(p:vec3f)->f32{return 1.0;}
fn nbSurfaceTheta(p:vec3f)->f32{return 1.0;}
fn umCellNearFromPage(t:u32,page:vec3i,p:vec3i)->vec3i{return vec3i(clamp(p.x,0,7),p.y,p.z);}
fn umCellPresent(p:vec3i)->bool{return all(p>=vec3i(0))&&all(p<vec3i(8));}
fn nbRun(a:vec3i,b:vec3i)->vec2u{return vec2u(u32(a.x+8*(a.y+8*a.z)),u32(b.x+8*(b.y+8*b.z)+1));}
fn nbMotion(i:u32)->vec4f{return vec4f(particles[i].velocity.xyz,0);}
fn nbPosition(i:u32)->vec3f{return particles[i].position.xyz;}
fn weight(x:f32)->f32{let a=abs(x);if(a<0.5){return 0.75-a*a;}let b=max(0.0,1.5-a);return 0.5*b*b;}
${narrowBandFineTransferSource(subgroups,true)}
`});
   const layout=device.createBindGroupLayout({entries:[
    {binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
    ...[1,2].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}})),
    {binding:3,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},
    {binding:4,visibility:GPUShaderStage.COMPUTE,storageTexture:{format:"rgba32float",viewDimension:"3d",access:"write-only"}},
   ]});
   const pipeline=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint:"transfer"}});
   const group=device.createBindGroup({layout,entries:[{binding:0,resource:{buffer:particles}},{binding:1,resource:{buffer:dummy,offset:0,size:256}},{binding:2,resource:{buffer:dummy,offset:256,size:256}},{binding:3,resource:velocity.createView()},{binding:4,resource:output.createView()}]});
   const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(subgroups?128:64);pass.end();
   encoder.copyTextureToBuffer({texture:output},{buffer:read,bytesPerRow:256,rowsPerImage:8},[8,8,8]);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
   const values=new Float32Array(read.getMappedRange());
   for(let z=1;z<7;z++)for(let y=1;y<7;y++)for(let x=1;x<7;x++)for(let a=0;a<3;a++){
    const q=[x+0.5,y+0.5,z+0.5];q[a]!+=0.5;
    const expected=shift[a]!+rows[a]!.reduce((sum,c,k)=>sum+c*q[k]!,0),actual=values[4*x+64*(y+8*z)+a]!;
    assert.ok(Math.abs(actual-expected)<1e-5,`subgroups=${subgroups} ${x},${y},${z},${a}: ${actual} vs ${expected}`);
   }
   read.unmap();
  }}finally{if(read.mapState==="mapped")read.unmap();for(const r of [particles,dummy,velocity,output,read])r.destroy();}
 },["subgroups"]);
});
