import assert from "node:assert/strict";
import test from "node:test";
import {narrowBandActivityWGSL} from "../lib/methods/uniform/uniform-narrow-band-activity.wgsl";
import {narrowBandTiledSurfaceWGSL,narrowBandTransferPropertiesSource} from "../lib/methods/uniform/uniform-narrow-band-surface.wgsl";
import {withUniformDevice} from "./helpers/uniform-geometric";

test("NB-FLIP heat excludes solid padding from momentum and surface coupling",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("wall heat interpolation",async device=>{
  const start=narrowBandActivityWGSL.indexOf("fn nbInterpolatedHeat(");
  const helper=narrowBandActivityWGSL.slice(start,narrowBandActivityWGSL.indexOf("// A point inside",start));
  const couple=narrowBandTiledSurfaceWGSL.slice(narrowBandTiledSurfaceWGSL.indexOf("var<workgroup> nbCoupleHeat:"),narrowBandTiledSurfaceWGSL.indexOf("// The vertices a fine tile"));
  const transferSource=narrowBandTransferPropertiesSource();
  const transfer=transferSource.slice(transferSource.indexOf("var<workgroup> nbTransferHeat:"),transferSource.lastIndexOf("@compute",transferSource.indexOf("fn transferProperties(")));
  const bins=device.createBuffer({size:1024*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  const values=new Float32Array(1024);
  for(let i=0;i<512;i++){
   values[i]=(i>>6)<4?.2+.1*(i%8):0;
   values[512+i]=(i>>6)<4?1+values[i]!:0;
  }
  device.queue.writeBuffer(bins,0,values);
  const output=device.createBuffer({size:20*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const module=device.createShaderModule({code:`
@group(0) @binding(0) var<storage,read_write> bins:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read_write> output:array<f32>;
const NB_ACTIVITY_THETA=0u;const NB_ACTIVITY_SURFACE_HEAT=512u;
fn umCorner(k:u32,n:u32)->vec3u{return vec3u(k%n,(k/n)%n,k/(n*n));}
fn nbHomeBlock(page:vec3i,lo:vec3i,hi:vec3i)->u32{return 0u;}
fn umTileNearFromPage(block:u32,page:vec3i,p:vec3i)->vec3i{return clamp(p,vec3i(0),vec3i(7));}
fn umTileAt(p:vec3u)->u32{return p.x+8u*p.y+64u*p.z;}
fn umTileCoord(t:u32)->vec3u{return vec3u(t%8u,(t/8u)%8u,t/64u);}
fn umTileOpen(t:u32)->f32{return select(0.0,1.0,t<256u);}
fn bandPhi(p:vec3f)->f32{return -1.0;}
${helper}
${couple}
${transfer}
@compute @workgroup_size(64) fn main(@builtin(local_invocation_index) i:u32,@builtin(workgroup_id) group:vec3u){
 let high=group.x!=0u;let tile=umTileAt(vec3u(1,1,select(0u,3u,high)));
 nbCoupleHeatLoad(tile,i);
 if(i<27u){nbTransferHeat[i]=nbCoupleHeat[i];nbTransferHot[i]=nbCoupleHot[i];}
 workgroupBarrier();
 if(group.x==0u&&i<12u){
  let z=array<f32,6>(0,16,1,15,4,12)[i%6u];
  let p=vec3f(6,6,z);output[i]=nbInterpolatedHeat(vec3i(floor(p/4.0)),p,i>=6u);
 }
 if(i==0u){
  let local=vec3i(2,2,select(0,4,high));let at=12u+4u*group.x;
  output[at]=nbCoupleTheta(local,false);output[at+1u]=nbCoupleTheta(local,true);
  let weights=nbTransferTheta(vec3f(local));output[at+2u]=weights.x;output[at+3u]=weights.y;
 }
}`});
  assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error").map(m=>m.message),[]);
  const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
  const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[bins,output].map((buffer,binding)=>({binding,resource:{buffer}}))});
  const read=device.createBuffer({size:output.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{
   const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(2);p.end();e.copyBufferToBuffer(output,0,read,0,output.size);device.queue.submit([e.finish()]);
   await read.mapAsync(GPUMapMode.READ);const actual=new Float32Array(read.getMappedRange());
   for(let i=0;i<20;i++)assert.ok(Math.abs(actual[i]!-.3)<1e-6,`sample ${i}: ${actual[i]}`);
  }finally{if(read.mapState==="mapped")read.unmap();read.destroy();bins.destroy();output.destroy();}
 });
});
