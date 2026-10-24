/** Spatial readback of the NB phi-volume gauge on the current GPU ownership.
 * Avoids stale host page maps and the distinct pressure fill-field estimate. */
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import type { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "../lib/methods/uniform/uniform-mixed-topology.wgsl";
import { uniformMixedSolidWGSL, type UniformMixedSolid } from "../lib/methods/uniform/uniform-mixed-solid.wgsl";
import { uniformDetailBindLayout, uniformDetailModule, uniformDetailPipeline, uniformDetailPick, uniformDetailGroup } from "../lib/methods/uniform/uniform-detail-fields";
import { levelSetFillWGSL } from "../lib/core/level-set-fill.wgsl";
import { readMixedBuffer } from "../tests/helpers/uniform-mixed-native-fields";

export async function createNbflipVolumeProbe(device:GPUDevice,solver:WebGPUUniformReferenceSolver){
 // Field rewrites are registered on the solver initialization device wrapper.
 device=(solver as unknown as {device:GPUDevice}).device;
 const frame=(solver as unknown as {mixedFrame:{ownership:UniformMixedOwnership;solid:UniformMixedSolid;fields:{phi:GPUTexture}}}).mixedFrame;
 const o=frame.ownership,solid=frame.solid,{nx,ny,nz}=solver.info;
 const jobs=Math.ceil(nx*ny*nz/64),gridX=Math.min(jobs,device.limits.maxComputeWorkgroupsPerDimension),gridY=Math.ceil(jobs/gridX);
 const output=device.createBuffer({size:4*nx*ny*nz,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
 const layout=uniformDetailBindLayout(device,{entries:[
  {binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},
  {binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
 ]});
 const module=uniformDetailModule(device,{label:"NB phi volume probe",code:uniformMixedTopologyWGSL(o.capacity,0)+`
@group(1) @binding(0) var phi:texture_3d<f32>;
@group(1) @binding(1) var<storage,read_write> result:array<f32>;
${uniformMixedSolidWGSL(2,solid.coarse?.count)}
${levelSetFillWGSL}
@compute @workgroup_size(64) fn measure(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let i=64u*(group.x+${gridX}u*group.y)+lane; if(i>=${nx*ny*nz}u){return;}result[i]=0.0;
 let c=vec3i(vec3u(i%${nx}u,(i/${nx}u)%${ny}u,i/${nx*ny}u))+umWallLow();
 if(!umCellPresent(c)){return;}let owner=umOwnerAt(c);let origin=umOrigin(owner);
 if(any(vec3i(origin)!=c)){return;}
 let capacity=select(umTileOpen(owner.tile),umCellOpen(c),owner.width==1u);
 if(capacity<=0.0){return;}
 var values:array<f32,8>;var negative=0u;
 for(var k=0u;k<8u;k++){
  let p=vec3i(origin+owner.width*umCorner(k,2u));
  if(owner.width==4u){values[k]=/*4h*/textureLoad(phi,p,0).x;}else{values[k]=textureLoad(phi,p,0).x;}
  negative+=u32(values[k]<0.0);
 }
 let mass=f32(owner.width*owner.width*owner.width)*capacity;
 if(negative==8u){result[i]=mass;return;}if(negative==0u){return;}
 // A 4h owner, full or cut, sums its 64 h subcells: fill()'s one 6-tet split
 // is orientation-biased (an aligned 16^3 block read 7.68 of 8.192 L).
 if(owner.width==1u){result[i]=fill(values)*mass;return;}
 var sum=0.0;
 for(var s=0u;s<64u;s++){
  let cell=vec3u(s&3u,(s>>2u)&3u,s>>4u);let open=umCellOpen(c+vec3i(cell));if(open<=0.0){continue;}
  var v:array<f32,8>;
  for(var k=0u;k<8u;k++){
   let f=0.25*vec3f(cell+umCorner(k,2u));
   v[k]=mix(mix(mix(values[0],values[1],f.x),mix(values[2],values[3],f.x),f.y),mix(mix(values[4],values[5],f.x),mix(values[6],values[7],f.x),f.y),f.z);
  }
  sum+=open*fill(v);
 }
 result[i]=sum;
}`});
 const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
 if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
 const pipeline=await uniformDetailPipeline(device,o,{layout:device.createPipelineLayout({bindGroupLayouts:[o.bindLayout,layout,solid.tileLayout]}),compute:{module,entryPoint:"measure",constants:{umDispatchX:65535}}});
 const group=uniformDetailGroup(device,{layout,entries:[{binding:0,resource:frame.fields.phi},{binding:1,resource:{buffer:output}}]});
 return {async read(){
  const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(uniformDetailPick(pipeline));pass.setBindGroup(0,o.bindGroup);pass.setBindGroup(1,group.group);pass.setBindGroup(2,solid.tileGroup);pass.dispatchWorkgroups(gridX,gridY);pass.end();device.queue.submit([encoder.finish()]);return readMixedBuffer(device,output);
 },destroy(){output.destroy();}};
}
