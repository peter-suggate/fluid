/** GPU differential check for compilation-oriented WGSL changes.
 * WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/check-cold-startup-kernels-dawn.ts
 */
import assert from "node:assert/strict";
import { narrowBandSeedShellWGSL } from "../lib/methods/uniform/uniform-narrow-band-seed.wgsl";
import { uniformMixedFaceAddressWGSL, uniformMixedFaceDispatchWGSL } from "../lib/methods/uniform/uniform-mixed-face-dispatch.wgsl";
import { withUniformDevice } from "../tests/helpers/uniform-geometric";

// Independent previous traversal, retained as a differential oracle.
const referenceFaces = /* wgsl */ `
@compute @workgroup_size(64) fn reference(@builtin(global_invocation_id) gid:vec3u){
 let owner=umOwner(gid);if(owner.width==0u){return;}let origin=umOrigin(owner);
 let low=umLowOpen(vec3i(origin));
 for(var axis=0u;axis<3u;axis++){
  if(low[axis]){let face=umFace(owner,axis,-1,0u);boundary[umLowFaceIndex(origin,axis)]=evaluate(owner,face);}
 }
 for(var axis=0u;axis<select(3u,1u,owner.width==1u);axis++){
  let first=umFace(owner,axis,1,0u);
  for(var part=0u;part<first.count;part++){
   let ownedFace=umFacePatch(first,part);var earlier=false;
   for(var other=0u;other<axis;other++){earlier=earlier||umPositiveFaceAtAnchor(owner,other,ownedFace.anchor).width!=0u;}
   if(earlier){continue;}var value=vec4f(0);
   for(var other=0u;other<3u;other++){
    let face=umPositiveFaceAtAnchor(owner,other,ownedFace.anchor);
    if(face.width!=0u){value[other]=evaluate(owner,face);}
   }
   for(var a=0u;a<3u;a++){if(low[a]){value.w+=evaluate(owner,umFace(owner,a,-1,0u));}}
   textureStore(output,ownedFace.anchor,value);
  }
 }
}`;
const referenceSeed = /* wgsl */ `
fn referenceSlope(q:vec3f,d:f32,axis:u32)->f32{
 var e=vec3f(0);e[axis]=0.5;
 var open=array<bool,2>(false,false);var depth=array<f32,2>(d,d);
 for(var s=0u;s<2u;s++){
  let x=q+select(-e,e,s==1u);open[s]=!nbParticleBlocked(x);
  if(open[s]){depth[s]=particleDepth(x);}
 }
 let span=select(2.0,1.0,open[0]&&open[1]);
 return select(0.0,span*(depth[1]-depth[0]),open[0]||open[1]);
}
fn referenceShell(lattice:vec3f)->vec4f{
 var q=lattice;var d=particleDepth(q);
 for(var i=0u;i<4u;i++){
  let rise=d+NB_SEED_DEPTH;if(rise<=1e-3){break;}
  var g=vec3f(0);for(var a=0u;a<3u;a++){g[a]=referenceSlope(q,d,a);}
  let g2=dot(g,g);if(g2<1e-8){break;}
  let next=nbPointNearIn(vec3i(floor(q)),q-min(rise,0.5)*g*inverseSqrt(g2),0.01);
  let depth=particleDepth(next);if(depth>=d){break;}
  q=next;d=depth;
 }
 return vec4f(q,d);
}`;
await withUniformDevice("cold startup kernels", async device => {
  const output = device.createTexture({ size: [32,4,4], dimension: "3d", format: "rgba32float",
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST });
  const boundary = device.createBuffer({ size: 8*3*4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const read = device.createBuffer({ size: 32*4*4*16 + boundary.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const module = device.createShaderModule({ code: /* wgsl */ `
struct UMOwner{index:u32,width:u32}
struct UMFace{anchor:vec3i,axis:u32,sign:i32,width:u32,count:u32}
@group(0) @binding(0) var output:texture_storage_3d<rgba32float,write>;
@group(0) @binding(1) var<storage,read_write> boundary:array<f32>;
fn umOwner(gid:vec3u)->UMOwner{return UMOwner(gid.x,select(select(4u,1u,gid.x%3u==0u),0u,gid.x>=8u));}
fn umOrigin(o:UMOwner)->vec3u{return vec3u(4u*o.index,0,0);}
fn umLowOpen(p:vec3i)->vec3<bool>{let n=u32(p.x)/4u;return vec3<bool>((n&1u)==0u,(n&2u)==0u,(n&4u)==0u);}
fn umLowFaceIndex(p:vec3u,a:u32)->u32{return 3u*(p.x/4u)+a;}
fn umFace(o:UMOwner,a:u32,s:i32,part:u32)->UMFace{
 let w=select(o.width,1u,(o.index+a)%2u==1u);var p=vec3i(umOrigin(o));
 p[a]+=select(-1,i32(o.width)-1,s==1);let u=(a+1u)%3u;let v=(a+2u)%3u;
 p[u]+=i32((part%(o.width/w))*w);p[v]+=i32((part/(o.width/w))*w);
 return UMFace(p,a,s,w,(o.width/w)*(o.width/w));
}
fn umFacePatch(f:UMFace,part:u32)->UMFace{
 var p=f;let side=u32(sqrt(f32(f.count)));let u=(f.axis+1u)%3u;let v=(f.axis+2u)%3u;
 p.anchor[u]+=i32((part%side)*f.width);p.anchor[v]+=i32((part/side)*f.width);return p;
}
fn evaluate(o:UMOwner,f:UMFace)->f32{return f32(o.index*1000u+f.axis*100u)+f32(f.sign)*f32(1+f.anchor.x+32*f.anchor.y+128*f.anchor.z);}
${uniformMixedFaceAddressWGSL}
${referenceFaces}
${uniformMixedFaceDispatchWGSL("candidate","evaluate(owner,face)",false,"for(var a=0u;a<3u;a++){if(low[a]){value.w+=lowValues[a];}}")}
` });
    const layout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32float", viewDimension: "3d" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ] });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const group = device.createBindGroup({ layout, entries: [{ binding: 0, resource: output.createView() }, { binding: 1, resource: { buffer: boundary } }] });
    let reference: Uint32Array | undefined;
    for (const entryPoint of ["reference", "candidate"]) {
      const pipeline = await device.createComputePipelineAsync({ layout: pipelineLayout, compute: { module, entryPoint } });
      device.queue.writeTexture({ texture: output }, new Uint8Array(32*4*4*16), { bytesPerRow: 512, rowsPerImage: 4 }, [32,4,4]);
      const e = device.createCommandEncoder();e.clearBuffer(boundary);
      const p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(1);p.end();
      e.copyTextureToBuffer({texture:output},{buffer:read,bytesPerRow:512,rowsPerImage:4},[32,4,4]);
      e.copyBufferToBuffer(boundary,0,read,32*4*4*16,boundary.size);device.queue.submit([e.finish()]);
      await read.mapAsync(GPUMapMode.READ);const result=new Uint32Array(read.getMappedRange().slice(0));read.unmap();
      if(reference)assert.deepEqual(result,reference,"mixed faces and low-boundary metadata must be bit-identical");else reference=result;
    }
  } finally { output.destroy();boundary.destroy();read.destroy(); }

  const count=4096, result=device.createBuffer({size:count*32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const readSeeds=device.createBuffer({size:result.size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
  try {
    const module=device.createShaderModule({code:/* wgsl */`
@group(0) @binding(0) var<storage,read_write> result:array<vec4f>;
const NB_SEED_DEPTH:f32=0.35;
var<private> mode:u32;
fn particleDepth(p:vec3f)->f32{
 if(mode==0u){return p.y-2.0;}
 if(mode==1u){return length(p-vec3f(2))-1.5;}
 if(mode==2u){return 0.0;}
 return p.x*p.x+0.5*p.z-p.y-1.0;
}
fn nbParticleBlocked(p:vec3f)->bool{return p.x<0.1||p.z<0.3||(mode==3u&&p.y>2.0);}
fn nbPointNearIn(home:vec3i,p:vec3f,pad:f32)->vec3f{return clamp(p,vec3f(pad),vec3f(4.0-pad));}
${referenceSeed}
${narrowBandSeedShellWGSL}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;mode=i%4u;let n=i/4u;
 let p=vec3f(f32(n%16u)/4.0+0.01,f32((n/16u)%8u)/2.0+0.01,f32((n/128u)%8u)/2.0+0.01);
 result[2u*i]=referenceShell(p);result[2u*i+1u]=seedShell(p,particleDepth(p));
}`});
    const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
    const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:result}}]});
    const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(count/64);p.end();
    e.copyBufferToBuffer(result,0,readSeeds,0,result.size);device.queue.submit([e.finish()]);await readSeeds.mapAsync(GPUMapMode.READ);
    const words=new Uint32Array(readSeeds.getMappedRange());
    for(let i=0;i<count;i++)assert.deepEqual(words.slice(i*8,i*8+4),words.slice(i*8+4,i*8+8),`seed ${i} must match bit for bit`);
    readSeeds.unmap();
  }finally{result.destroy();readSeeds.destroy();}
  console.log(JSON.stringify({status:"pass",seedCases:count,faceOwners:8,comparison:"bit-identical"}));
});
