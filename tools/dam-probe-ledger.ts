/** Read-only stage ledger using production geometry and owner-sized velocity samples. */
import { uniformDetailBindLayout, uniformDetailModule, uniformDetailPipeline, uniformDetailPick, uniformDetailGroup } from '../lib/methods/uniform/uniform-detail-fields';
import { uniformMixedTopologyWGSL } from '../lib/methods/uniform/uniform-mixed-topology.wgsl';
import { uniformMixedVertexSamplingSource } from '../lib/methods/uniform/uniform-mixed-vertex-sampling.wgsl';
import { uniformVolumeTargetWGSL } from '../lib/methods/uniform/uniform-volume.wgsl';
import { geometricPlaneBoxWGSL } from '../lib/core/geometric-plane-box.wgsl';
import { uniformMixedSolidWGSL, uniformMixedSolidPipeline } from '../lib/methods/uniform/uniform-mixed-solid.wgsl';
import { UNIFORM_DETAIL_4H_LOAD, UNIFORM_DETAIL_RING_4H_LOAD } from '../lib/core/uniform-detail-abi';
import { writeFile } from 'node:fs/promises';

export async function damProbeLedger(device:GPUDevice,f:any,records=false){
 device=f.surfaceVolume.device;
 const o=f.ownership,solid=f.solid,h=o.capacity.lattice.cellSize_m;
 const groups=Math.ceil((63*o.capacity.fineTiles+o.capacity.tiles)/64);
 const sums=device.createBuffer({size:groups*32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
 const cells=records?device.createBuffer({size:groups*64*64,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}):undefined;
 const layout=uniformDetailBindLayout(device,{entries:[
  ...[0,1].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'unfilterable-float' as const,viewDimension:'3d' as const}})),
  {binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}},
  ...(records?[{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage' as const}}]:[]),
 ]});
 const module=uniformDetailModule(device,{label:'Dam stage mass and energy',code:uniformMixedTopologyWGSL(o.capacity,0)+`
@group(1) @binding(0) var phi:texture_3d<f32>;
@group(1) @binding(1) var velocity:texture_3d<f32>;
@group(1) @binding(2) var<storage,read_write> sums:array<vec4f>;
${records?'@group(1) @binding(3) var<storage,read_write> cellLedger:array<vec4f>;':''}
fn umLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p),0).x;}
fn umLoadCorner(p:vec3u)->f32{return ${UNIFORM_DETAIL_4H_LOAD}textureLoad(phi,vec3i(p),0).x;}
fn umLoadVertex8(p:vec3u,s:u32)->array<f32,8>{return udLoad8_phi(vec3i(p),s);}
fn umLoadCorner8(p:vec3u,s:u32)->array<f32,8>{return udLoad48_phi(vec3i(p),s);}
${uniformMixedVertexSamplingSource('',true,undefined,'umLoadCorner',true)}
${geometricPlaneBoxWGSL}
fn uvCorner(k:u32)->vec3i{return vec3i(umCorner(k,2u));}
fn d4Sum8(v:array<f32,8>)->f32{return ((v[0]+v[5])+(v[1]+v[4]))+((v[2]+v[7])+(v[3]+v[6]));}
${uniformVolumeTargetWGSL(true,true)}
${uniformMixedSolidWGSL(solid?2:undefined,solid?.coarse?.count)}
var<workgroup> ledgerTerms:array<vec4f,64>;
var<workgroup> ledgerPhi:array<vec2f,64>;
@compute @workgroup_size(64) fn measure(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 let owner=umAllOwner(gid);var result=vec4f(0);var coordinate=0.0;var corners:array<f32,8>;var motion=vec3f(0);var range=vec2f(0);
 if(owner.width!=0u){
  let origin=umOrigin(owner);let width=owner.width;
  coordinate=f32(origin.x+128u*(origin.y+128u*origin.z));
  if(width==4u){corners=umLoadCorner8(origin,4u);}else{corners=umLoadVertex8(origin,1u);}
  var low=corners[0];var high=corners[0];for(var k=1u;k<8u;k++){low=min(low,corners[k]);high=max(high,corners[k]);}
  range=vec2f(low,high);
  var fill=0.0;if(high < -1e-20&&4.0*(high-low)<= -high){fill=1.0;}else if(!(low>1e-20&&4.0*(high-low)<=low)){fill=umSurfaceTarget(owner,corners);}
  if(width==1u){fill*=umCellOpen(vec3i(origin));}else{let open=umTileOpen(owner.tile);if(open<0.99999){fill=umSurfaceTargetOpen(corners,vec3i(origin));}}
  let mass=fill*f32(width*width*width);var u=vec3f(0);
  for(var a=0u;a<3u;a++){var face=origin;face[a]+=width-1u;
   if(width==4u){u[a]=${UNIFORM_DETAIL_RING_4H_LOAD}textureLoad(velocity,vec3i(face),0)[a];}else{u[a]=textureLoad(velocity,vec3i(face),0)[a];}
  }
  result=vec4f(mass,0.5*mass*dot(u,u),mass*9.81*(f32(origin.y)+0.5*f32(width))*${h[1]},select(0.0,max(abs(u.x),max(abs(u.y),abs(u.z))),mass>0.01));
  motion=u;
 }
 ${records?`let record=4u*(gid.x+64u*umDispatchX*gid.y);cellLedger[record]=vec4f(result.x,coordinate,f32(owner.width),result.y);
 cellLedger[record+1u]=vec4f(corners[0],corners[1],corners[2],corners[3]);cellLedger[record+2u]=vec4f(corners[4],corners[5],corners[6],corners[7]);cellLedger[record+3u]=vec4f(motion,0);`:''}
 ledgerTerms[lane]=result;ledgerPhi[lane]=range;workgroupBarrier();
 for(var stride=32u;stride>0u;stride/=2u){if(lane<stride){ledgerTerms[lane]=vec4f(ledgerTerms[lane].xyz+ledgerTerms[lane+stride].xyz,max(ledgerTerms[lane].w,ledgerTerms[lane+stride].w));ledgerPhi[lane]=vec2f(min(ledgerPhi[lane].x,ledgerPhi[lane+stride].x),max(ledgerPhi[lane].y,ledgerPhi[lane+stride].y));}workgroupBarrier();}
 if(lane==0u){let at=2u*(gid.x/64u+umDispatchX*gid.y);sums[at]=ledgerTerms[0];sums[at+1u]=vec4f(ledgerPhi[0],0,0);}
}`});
 const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error');if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join('\n'));
 const pipeline=await uniformMixedSolidPipeline(solid,(s:any)=>uniformDetailPipeline(device,o,{layout:device.createPipelineLayout({bindGroupLayouts:[o.bindLayout,layout,...(solid?[solid.tileLayout]:[])]}),compute:{module,entryPoint:'measure',constants:{umDispatchX:o.dispatchX,...s}}}),{entry:'measure'});
 const cache=new Map<GPUTexture,Map<GPUTexture,ReturnType<typeof uniformDetailGroup>>>();
 return {
  capture(e:GPUCommandEncoder,phi:GPUTexture,velocity:GPUTexture,recordPath?:string){
   let row=cache.get(phi);if(!row){row=new Map();cache.set(phi,row);}let group=row.get(velocity);
   if(!group){group=uniformDetailGroup(device,{layout,entries:[{binding:0,resource:phi},{binding:1,resource:velocity},{binding:2,resource:{buffer:sums}},...(cells?[{binding:3,resource:{buffer:cells}}]:[])]});row.set(velocity,group);}
   const read=device.createBuffer({size:sums.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
   const pass=e.beginComputePass({label:'Dam read-only stage ledger'});pass.setBindGroup(0,o.bindGroup);pass.setBindGroup(1,group.group);if(solid)pass.setBindGroup(2,solid.tileGroup);
   pass.setPipeline(uniformDetailPick(solid?.select(pipeline)??pipeline));pass.dispatchWorkgroups(Math.min(groups,o.dispatchX),Math.ceil(groups/o.dispatchX));pass.end();e.copyBufferToBuffer(sums,0,read,0,sums.size);
   const cellRead=recordPath&&cells?device.createBuffer({size:cells.size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}):undefined;
   if(cellRead)e.copyBufferToBuffer(cells!,0,cellRead,0,cellRead.size);
   return async()=>{try{await read.mapAsync(GPUMapMode.READ);const a=new Float32Array(read.getMappedRange());let mass=0,kinetic=0,potential=0,max=0,minPhi=0,maxPhi=0;for(let i=0;i<a.length;i+=8){mass+=a[i]!;kinetic+=a[i+1]!;potential+=a[i+2]!;max=Math.max(max,a[i+3]!);minPhi=Math.min(minPhi,a[i+4]!);maxPhi=Math.max(maxPhi,a[i+5]!);}if(cellRead){await cellRead.mapAsync(GPUMapMode.READ);await writeFile(recordPath!,new Uint8Array(cellRead.getMappedRange()));}return {mass,kinetic,potential,energy:kinetic+potential,max,minPhi,maxPhi};}finally{read.unmap();read.destroy();if(cellRead){cellRead.unmap();cellRead.destroy();}}};
  },
  destroy(){sums.destroy();cells?.destroy();}
 };
}
