import { uniformCompiledExtensionNeighborWGSL } from "../lib/methods/uniform/uniform-compiled-extension.wgsl";
import { uniformExtensionNeighborReferenceWGSL } from "./helpers/uniform-extension-neighbor-reference.wgsl";
import assert from "node:assert/strict";
import test from "node:test";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { createUniformMixedLayoutFromWidths } from "../lib/methods/uniform/uniform-mixed-layout";
import { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import { UniformMixedLayoutBuilder } from "../lib/methods/uniform/uniform-mixed-layout-builder";
import { uniformMixedTopologyWGSL } from "../lib/methods/uniform/uniform-mixed-topology.wgsl";
import { uniformMixedVertexSamplingWGSL } from "../lib/methods/uniform/uniform-mixed-vertex-sampling.wgsl";
import { uniformCompiledVertexResolveWGSL } from "../lib/methods/uniform/uniform-compiled-topology";
import { uniformCompiledVelocityTapWGSL } from "../lib/methods/uniform/uniform-compiled-velocity-taps.wgsl";
import { uniformMixedVelocitySamplingSource } from "../lib/methods/uniform/uniform-mixed-velocity-sampling.wgsl";

// Frozen geometric reference from before compiled recipes. Intentionally
// resolves every candidate and patch through the general owner lookup.
const reference = /* wgsl */ `
fn referenceAuthority(p:vec3u)->UMOwner{
 var best=UMOwner();
 for(var k=0u;k<8u;k++){
  let candidate=umOwnerAt(vec3i(p)+vec3i(umCorner(k,2u))-vec3i(1));
  if(candidate.width>best.width||(candidate.width==best.width&&candidate.width!=0u&&candidate.index<best.index)){best=candidate;}
 }return best;
}
fn referenceFace(owner:UMOwner,axis:u32,sign:i32,part:u32)->UMFace{
 let origin=vec3i(umOrigin(owner));var probe=origin;
 probe[axis]+=select(-1,i32(owner.width),sign>0);var neighbor=umOwnerAt(probe);
 let width=min(owner.width,select(owner.width,neighbor.width,neighbor.width!=0u));
 let side=owner.width/width;let count=side*side;if(part>=count){return UMFace();}
 let u=(axis+1u)%3u;let v=(axis+2u)%3u;
 probe[u]+=i32((part%side)*width);probe[v]+=i32((part/side)*width);
 if(part!=0u){neighbor=umOwnerAt(probe);}
 var anchor=probe;anchor[axis]-=select(0,1,sign>0);
 return UMFace(neighbor,anchor,width,count,axis,sign);
}
fn sameOwner(a:UMOwner,b:UMOwner)->bool{return a.tile==b.tile&&a.lane==b.lane&&a.width==b.width&&a.index==b.index;}
fn sameFace(a:UMFace,b:UMFace)->bool{return sameOwner(a.neighbor,b.neighbor)&&all(a.anchor==b.anchor)&&a.width==b.width&&a.count==b.count&&a.axis==b.axis&&a.sign==b.sign;}
fn referenceVertex(p:vec3u)->f32{
 let a=referenceAuthority(p);let origin=umOrigin(a);
 if(all((p-origin)%a.width==vec3u(0))){return umLoadVertex(p);}
 let t=vec3f(p-origin)/4.0;var terms:array<f32,8>;
 for(var k=0u;k<8u;k++){let c=umCorner(k,2u);let w=umVertexWeight(t,c);if(w>0.0){terms[k]=w*umLoadVertex(origin+4u*c);}}
 return umVertexSum8(terms);
}
`;

(process.env.WEBGPU_NODE_MODULE?test:test.skip)("GPU compiled topology matches CPU builders and original vertex/face traversal through all 256 layouts",{timeout:180000},async()=>{
 await withUniformDevice("Uniform compiled topology parity",async device=>{
  const lattice={dimensions:[8,8,8] as const,cellSize_m:[1,1,1] as const,origin_m:{x:0,y:0,z:0}};
  const initial=createUniformMixedLayoutFromWidths(lattice,new Uint8Array(8).fill(4),[]);
  const ownership=new UniformMixedOwnership(device,initial);
  const band=device.createBuffer({size:4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  const builder=new UniformMixedLayoutBuilder(device,{buffer:band,wordOffset:0},ownership);
  const errors=device.createBuffer({size:32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const readback=device.createBuffer({size:256*160+32,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{
   await builder.initialize();
   const resources=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
   const group=device.createBindGroup({layout:resources,entries:[{binding:0,resource:{buffer:errors}}]});
   const module=device.createShaderModule({code:uniformMixedTopologyWGSL(initial,0)+/* wgsl */`
@group(1) @binding(0) var<storage,read_write> failures:array<atomic<u32>>;
fn umLoadVertex(p:vec3u)->f32{let n=p.x+p.y*9u+p.z*81u;return f32((n*1664525u+1013904223u)&65535u)/32768.0-1.0;}
fn umLoadMixedFace(p:vec3i,axis:u32)->f32{return umLoadVertex(bitcast<vec3u>(p+vec3i(i32(axis)*19)));}
fn umLoadCoarseFace(p:vec3i,axis:u32)->f32{return umLoadVertex(bitcast<vec3u>(p+vec3i(i32(axis)*31+97)));}
${uniformMixedVelocitySamplingSource(false,true)}
${uniformCompiledVelocityTapWGSL}
${uniformMixedVertexSamplingWGSL}
${reference}
const UM_INF=1e20;const h=vec4f(0.5,1,2,0);
fn umSlotState(anchor:vec3i,axis:u32,width:u32)->vec2f{
 let k=bitcast<u32>(anchor.x+17*anchor.y+101*anchor.z)+axis*733u+width*937u;
 return vec2f(umLoadVertex(bitcast<vec3u>(anchor+vec3i(i32(axis*17u+width)))),select(f32(k%5u)*0.5,UM_INF,k%7u==0u));
}
${uniformExtensionNeighborReferenceWGSL}
${uniformCompiledExtensionNeighborWGSL}
fn referenceDirect(point:vec3f,component:u32,width:u32)->bool{
 var below=vec3i(floor(point));let plane=i32(round(point[component]));below[component]=plane-1;
 var above=below;above[component]=plane;
 var lo=0u;if(plane>0){lo=umTileWidth(umTileAt(vec3u(below)/4u));}
 var hi=0u;if(plane<i32(UM_D[component])){hi=umTileWidth(umTileAt(vec3u(above)/4u));}
 return select((lo==width&&(hi==0u||hi>=width))||(hi==width&&lo>width),hi==width,lo==0u);
}
@compute @workgroup_size(64) fn compareExtension(@builtin(global_invocation_id) gid:vec3u){
 let owner=umAllOwner(gid);if(owner.width==0u){return;}
 for(var axis=0u;axis<3u;axis++){
  let first=umFaceFirst(owner,axis,1);
  for(var part=0u;part<first.count;part++){
   let face=umFacePatch(first,part);let center=umFaceCenter(face);
   for(var k=0u;k<6u;k++){
    var point=center;point[k/2u]+=select(-f32(face.width),f32(face.width),(k&1u)!=0u);
    if(any(point<vec3f(0))||any(point>vec3f(UM_D))||referenceDirect(point,axis,face.width)){continue;}
    let before=umNeighbor(point,center,axis,k/2u,face.width);let after=ueCompiledNeighbor(point,center,axis,k/2u,face.width);
    if(any(bitcast<vec3u>(vec3f(before.value,before.distance,before.spacing))!=bitcast<vec3u>(vec3f(after.value,after.distance,after.spacing)))){atomicAdd(&failures[6],1u);}
    atomicAdd(&failures[7],1u);
   }
  }
 }
}
@compute @workgroup_size(64) fn compareTaps(@builtin(global_invocation_id) gid:vec3u){
 let tile=gid.x/256u;let lane=gid.x%256u;if(tile>=UM_TILES||umTileWidth(tile)!=4u||lane>=240u){return;}
 var local=umCorner(lane%64u,4u);var axis=lane/64u;let negativePlane=lane>=192u;
 if(negativePlane){let k=lane-192u;axis=k/16u;if(umTileCoord(tile)[axis]!=0u){return;}
  local=vec3u(0);local[(axis+1u)%3u]=k%4u;local[(axis+2u)%3u]=(k%16u)/4u;
 }
 var index=vec3i(umTileCoord(tile)*4u+local);if(negativePlane){index[axis]=-1;}
 if(bitcast<u32>(umCompiledVelocityTap(tile,local,axis,negativePlane))!=bitcast<u32>(umVelocityTap1(index,axis))){atomicAdd(&failures[5],1u);}
}
var<workgroup> lattice:array<f32,27>;
@compute @workgroup_size(125) fn compareResolve(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let tile=group.x;let base=vec3i(umTileCoord(tile));
 if(lane>=9u&&lane<35u){let v=(base+vec3i(umCorner(lane-8u,3u))-vec3i(1))*4;if(all(v>=vec3i(0))){lattice[lane-8u]=umLoadVertex(vec3u(v));}}
 workgroupBarrier();
 ${uniformCompiledVertexResolveWGSL("lattice","umVertexWeight","umVertexSum8")}
 if(bitcast<u32>(value)!=bitcast<u32>(referenceVertex(p))){atomicAdd(&failures[3],1u);}
}
@compute @workgroup_size(64) fn compare(@builtin(global_invocation_id) gid:vec3u){
 if(gid.x<UM_TILES){let tile=umTileCoord(gid.x);
  for(var k=0u;k<64u;k++){
   let d=umCorner(k,4u);let p=(vec3i(tile)+vec3i(d)-vec3i(1))*4;let owner=umOwnerAt(p);if(owner.width==0u){continue;}
   let c=d/2u;let local=d%2u;let mask=umWindowFineOctant(tile,c.x+2u*(c.y+2u*c.z));
   let fine=(mask&(1u<<(local.x+2u*(local.y+2u*local.z))))!=0u;
   if(fine!=(owner.width==1u)){atomicAdd(&failures[4],1u);}
  }
 }
 if(gid.x<729u){let p=umCorner(gid.x,9u);
  if(!sameOwner(umVertexAuthority(p),referenceAuthority(p))){atomicAdd(&failures[0],1u);}
  if(bitcast<u32>(umVertexValue(p))!=bitcast<u32>(referenceVertex(p))){atomicAdd(&failures[2],1u);}
 }
 let owner=umAllOwner(gid);if(owner.width==0u){return;}
 for(var axis=0u;axis<3u;axis++){for(var side=0u;side<2u;side++){
  let sign=select(-1,1,side==1u);let first=umFaceFirst(owner,axis,sign);
  // One extra part exercises the empty sentinel as well as every real patch.
  for(var part=0u;part<=first.count;part++){
   let expected=referenceFace(owner,axis,sign,part);
   if(!sameFace(umFacePatch(first,part),expected)||!sameFace(umFace(owner,axis,sign,part),expected)){atomicAdd(&failures[1],1u);}
  }
 }}
}`});
   const messages=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");assert.deepEqual(messages.map(m=>m.message),[]);
   const pipeline=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[ownership.bindLayout,resources]}),compute:{module,entryPoint:"compare"}});
   const resolve=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[ownership.bindLayout,resources]}),compute:{module,entryPoint:"compareResolve"}});
   const taps=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[ownership.bindLayout,resources]}),compute:{module,entryPoint:"compareTaps"}});
   const extension=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[ownership.bindLayout,resources]}),compute:{module,entryPoint:"compareExtension"}});
   const expected:Uint32Array[]=[];
   for(let bits=0;bits<256;bits++){
    const widths=Uint8Array.from({length:8},(_,k)=>bits&(1<<k)?1:4);
    const cpu=createUniformMixedLayoutFromWidths(lattice,widths,[]);
    expected.push(new Uint32Array([...cpu.tiles,...cpu.fineTiles,...cpu.coarseTiles,...cpu.stencils,...cpu.blendMasks]));
    builder.setStatic(Uint8Array.from(widths,w=>Number(w===1)));
    const e=device.createCommandEncoder();builder.encodeClear(e);builder.encode(e);
    e.copyBufferToBuffer(builder.generation.topology,0,readback,160*bits,160);
    ownership.adoptGpu(e,builder.generation);
    const pass=e.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,ownership.bindGroup);pass.setBindGroup(1,group);pass.dispatchWorkgroups(12);pass.setPipeline(resolve);pass.dispatchWorkgroups(8);pass.setPipeline(taps);pass.dispatchWorkgroups(32);pass.setPipeline(extension);pass.dispatchWorkgroups(8);pass.end();
    device.queue.submit([e.finish()]);
   }
   const e=device.createCommandEncoder();e.copyBufferToBuffer(errors,0,readback,256*160,32);device.queue.submit([e.finish()]);
   await readback.mapAsync(GPUMapMode.READ);const values=new Uint32Array(readback.getMappedRange().slice(0));readback.unmap();
   for(let bits=0;bits<256;bits++)assert.deepEqual(values.slice(bits*40,(bits+1)*40),expected[bits],`builder layout ${bits}`);
   assert.deepEqual([...values.slice(256*40,256*40+7)],[0,0,0,0,0,0,0],"authority, face, sampled-value, staged-resolve, window-mask velocity-tap and extension-neighbor mismatches");
   assert.equal(values[256*40+7],75648,"All searching extension requests across 256 layouts");
  }finally{readback.destroy();errors.destroy();builder.destroy();band.destroy();ownership.destroy();}
 });
});
