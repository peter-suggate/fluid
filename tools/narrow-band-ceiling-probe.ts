/** Read-only surface-volume and ceiling diagnostics for sustained NB-FLIP runs. */
import {levelSetFillWGSL} from '../lib/core/level-set-fill.wgsl';
import {UNIFORM_DETAIL_CANONICAL_LOAD} from '../lib/core/uniform-detail-abi';
import {uniformDetailBindLayout,uniformDetailModule,uniformDetailGroup} from '../lib/methods/uniform/uniform-detail-fields';
import {uniformMixedTopologyWGSL,uniformMixedCountedEntriesWGSL,UNIFORM_MIXED_COUNTED} from '../lib/methods/uniform/uniform-mixed-topology.wgsl';
import {uniformMixedVertexSamplingSource} from '../lib/methods/uniform/uniform-mixed-vertex-sampling.wgsl';
import type {UniformMixedFrame} from '../lib/methods/uniform/uniform-mixed-frame';
import type {WebGPUUniformReferenceSolver} from '../lib/methods/uniform/webgpu-uniform-reference';
export async function narrowBandCeilingProbe(device:GPUDevice,solver:WebGPUUniformReferenceSolver){
 const frame=(solver as unknown as {mixedFrame:Pick<UniformMixedFrame,"ownership"|"narrowBandFlip">&{fields:{phi:GPUTexture;phiScratch:GPUTexture}}}).mixedFrame;
 // Field layouts are registered on the solver's initialization-scoped device.
 // Using its outer device would bypass packed-field addressing in the shader.
 device=(solver as unknown as {device:GPUDevice}).device;
 const o=frame.ownership,stage=frame.narrowBandFlip!;
 const totals=device.createBuffer({size:1280,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
 const read=device.createBuffer({size:1280,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
 const resources=uniformDetailBindLayout(device,{entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'unfilterable-float',viewDimension:'3d'}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}}]});
 const groups=[frame.fields.phi,frame.fields.phiScratch,frame.fields.phi,frame.fields.phi,frame.fields.phi].map((t,i)=>uniformDetailGroup(device,{layout:resources,entries:[{binding:0,resource:t},{binding:1,resource:{buffer:totals,offset:i*256,size:256}}]}));
 const module=uniformDetailModule(device,{label:'NB ceiling stage probe',code:uniformMixedCountedEntriesWGSL(uniformMixedTopologyWGSL(o.capacity,0)+/* wgsl */`
@group(1) @binding(0) var phi:texture_3d<f32>;
@group(1) @binding(1) var<storage,read_write> total:array<atomic<u32>>;
fn umLoadVertex(p:vec3u)->f32{return ${UNIFORM_DETAIL_CANONICAL_LOAD}textureLoad(phi,vec3i(p),0).x;}
${uniformMixedVertexSamplingSource('',false)}
${levelSetFillWGSL}
var<workgroup> values:array<u32,64>;
@compute @workgroup_size(64) fn probe(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 let owner=umAllOwner(gid);var value=0u;
 if(owner.width!=0u){
  var v:array<f32,8>;let origin=umOrigin(owner);
  for(var k=0u;k<8u;k++){v[k]=umSampleVertex(vec3f(origin+umCorner(k,2u)*owner.width));}
  value=u32(fill(v)*f32(owner.width*owner.width*owner.width)*2048.0+0.5);
  let height=${o.capacity.lattice.dimensions[1]}u;
  let band=min(15u,16u*origin.y/height);
  atomicAdd(&total[1u+band],value);
  if(origin.y+owner.width==height){
   for(var k=0u;k<8u;k++){if((k&2u)!=0u&&v[k]<=0.0){atomicAdd(&total[17],1u);}}
  }
 }
 values[lane]=value;workgroupBarrier();
 for(var stride=32u;stride>0u;stride/=2u){if(lane<stride){values[lane]+=values[lane+stride];}workgroupBarrier();}
 if(lane==0u){atomicAdd(&total[0],values[0]);}
}`,['probe'])});
 const pipeline=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[o.bindLayout,resources]}),compute:{module,entryPoint:'probe',constants:{umDispatchX:o.dispatchX,umCountedJobs:UNIFORM_MIXED_COUNTED.all}}});
 function encode(encoder:GPUCommandEncoder,index:number){
  encoder.clearBuffer(totals,index*256,256);const pass=encoder.beginComputePass({label:`NB volume probe ${index}`});pass.setBindGroup(0,o.bindGroup);pass.setBindGroup(1,groups[index]!.group);o.dispatchAllCounted(pass,pipeline);pass.end();
 }
 const move=stage.move.bind(stage),reconstruct=stage.reconstruct.bind(stage),redistance=stage.redistance.bind(stage);
 stage.move=(...args)=>{encode(args[0],0);move(...args);};
 stage.reconstruct=encoder=>{encode(encoder,1);reconstruct(encoder);encode(encoder,2);};
 stage.redistance=encoder=>{encode(encoder,3);redistance(encoder);encode(encoder,4);};
 return {
  async read(){const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(totals,0,read,0,1280);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);const words=new Uint32Array(read.getMappedRange());const result=Object.fromEntries(['start','advected','reconstructed','retired','redistanced'].map((name,i)=>[name,{volume:words[64*i]!/2048,heightBands:Array.from(words.slice(64*i+1,64*i+17),v=>v/2048),wetCeilingCorners:words[64*i+17]}]));read.unmap();return result;},
  destroy(){stage.move=move;stage.reconstruct=reconstruct;stage.redistance=redistance;totals.destroy();read.destroy();},
 };
}
