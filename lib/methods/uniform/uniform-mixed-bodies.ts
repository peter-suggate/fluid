import { uniformDetailBindLayout, uniformDetailModule, uniformDetailGroup, type UniformDetailGroup } from "./uniform-detail-fields";
import {uniformKeyGroups} from "./uniform-lattice-address";
import type {UniformMixedOwnership} from "./uniform-mixed-ownership";
import {UNIFORM_MIXED_COUNTED,uniformMixedCountedEntriesWGSL,uniformMixedTopologyWGSL} from "./uniform-mixed-topology.wgsl";
import {uniformMixedSolidWGSL,type UniformMixedSolid,type UniformMixedSolidPool} from "./uniform-mixed-solid.wgsl";
import {UNIFORM_LIVE_KEY_GROUPS,uniformLiveListsWGSL} from "./uniform-live-lists";
import {GPU_RIGID_BODY_CAPACITY} from "../../core/rigid-capacity";

/** The rigid coupling kernel (couple) and its per-body face fraction, over
 * the solid library and the helpers its module defines: umBodyPressure,
 * umBodyOwnerWet, rigidExchange and
 * umAllOwner over the h list (its job is an h tile, a lane a cell). Exported
 * for the all-body reference test. */
export const uniformMixedCoupleWGSL=/* wgsl */`
// Match the pressure operator's four transverse aperture samples.
fn umBodyFaceFraction(i:u32,p:vec3i,axis:u32)->f32{
 let world=umSolidFaceWorld(p,axis);let h=umSolidParams.cellGravity.xyz;var blocked=0.0;
 for(var k=0u;k<4u;k++){
  var sample=world;let a=(axis+1u)%3u;let b=(axis+2u)%3u;
  sample[a]+=select(-0.35,0.35,(k&1u)!=0u)*h[a];sample[b]+=select(-0.35,0.35,(k&2u)!=0u)*h[b];
  blocked+=select(0.0,0.25,umInsideBody(i,sample));
 }return blocked;
}
// The exchange's fixed-point words of one h tile's job, summed in workgroup
// memory (integer adds commute: the sums equal the per-cell global adds) and
// flushed once a job; a word whose sum is zero is not touched.
var<workgroup> coupleSum:array<atomic<i32>,${12*GPU_RIGID_BODY_CAPACITY}>;
// The bodies whose bounding sphere, grown by the per-cell margin, reaches
// the tile's box (a superset of every cell's umBodyNear), and those some
// cell is near (word 10: pressure coupled).
var<workgroup> coupleBodies:atomic<u32>;
var<workgroup> coupleNear:atomic<u32>;
fn coupleAdd(word:u32,value:i32){if(value!=0){atomicAdd(&coupleSum[word],value);}}
// One job an h tile (the h list), a lane a cell.
@compute @workgroup_size(64) fn couple(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let o=umAllOwner(vec3u(64u*group.x+lane,0u,0u));
 let id=vec3i(umOrigin(o));let h=umSolidParams.cellGravity.xyz;let world=umSolidWorldCell(id);
 let count=umBodyCount();
 if(lane==0u){atomicStore(&coupleBodies,0u);atomicStore(&coupleNear,0u);}
 workgroupBarrier();
 // Every cell centre lies in the tile's box: a body whose sphere misses the
 // box by more than the margin is near no cell of it.
 if(lane<count&&o.width==1u){
  let lo=umSolidWorldCell(vec3i(4u*umTileCoord(o.tile)));let hi=lo+3.0*h;
  let b=umSolidBodies[lane];let c=b.positionShape.xyz;
  if(b.positionShape.w<0.0||distance(clamp(c,lo,hi),c)<=b.dimensions.w+1.01*length(h)){atomicOr(&coupleBodies,1u<<lane);}
 }
 let bodies=workgroupUniformLoad(&coupleBodies);
 if(bodies==0u){return;}
 for(var i=lane;i<12u*count;i+=64u){if(((bodies>>(i/12u))&1u)!=0u){atomicStore(&coupleSum[i],0);}}
 workgroupBarrier();
 var remaining=bodies;
 while(remaining!=0u){
  let owner=firstTrailingBit(remaining);remaining&=remaining-1u;
  if(!umBodyNear(owner,world,length(h))){continue;}
  atomicOr(&coupleNear,1u<<owner);let base=owner*12u;
  // The shapes' distances are exact (1-Lipschitz, signed as umInsideBody):
  // every face and corner sample lies within half the cell's diagonal of its
  // centre, so a centre farther than that from the surface has all of them on
  // its own side. Its face fractions then cancel and its fraction is 0 or 1.
  let distance=umBodyDistance(owner,world);let clear=abs(distance)>0.505*length(h);
  var area=vec3f(0.0);
  if(!clear){
   for(var axis=0u;axis<3u;axis++){
    var low=id;low[axis]-=1;
    area[axis]=(umBodyFaceFraction(owner,id,axis)-umBodyFaceFraction(owner,low,axis))*h[(axis+1u)%3u]*h[(axis+2u)%3u];
   }
  }
  // No cut face: the reaction and its torque are zero whatever the pressure.
  if(any(area!=vec3f(0.0))){
   let reaction=umBodyPressure(id)*area*umSolidParams.dimsDt.w;
   let torque=cross(world-umSolidBodies[owner].positionShape.xyz,reaction);
   coupleAdd(base,i32(round(reaction.x*1000000.0)));coupleAdd(base+1u,i32(round(reaction.y*1000000.0)));coupleAdd(base+2u,i32(round(reaction.z*1000000.0)));
   coupleAdd(base+3u,i32(round(torque.x*1000000.0)));coupleAdd(base+4u,i32(round(torque.y*1000000.0)));coupleAdd(base+5u,i32(round(torque.z*1000000.0)));
  }
  var fraction=select(0.0,1.0,distance<0.0);if(!clear){fraction=umBodyFraction(owner,id);}
  if(fraction==0.0){continue;}
  // The displaced wet volume (word 6). No ambient velocity (words 7-9, its
  // weight 11): the integrator reads it only for the drag of a body no cell
  // couples by pressure (webgpu-rigid-body.ts: drag 0 where word 10 is set),
  // and every body this kernel would weigh it for is one some cell is near,
  // which sets word 10. Its six probes a displacing cell were dead work.
  coupleAdd(base+6u,i32(round(umBodyOwnerWet(o)*fraction*65536.0)));
 }
 workgroupBarrier();
 let near=atomicLoad(&coupleNear);
 for(var i=lane;i<12u*count;i+=64u){
  let body=i/12u;if(((bodies>>body)&1u)==0u){continue;}
  if(i%12u==10u){if(((near>>body)&1u)!=0u){atomicStore(&rigidExchange[i],1);}continue;}
  let v=atomicLoad(&coupleSum[i]);if(v!=0){atomicAdd(&rigidExchange[i],v);}
 }
}
`;

export interface UniformMixedBodyFields {
 /** The frame's velocity (the projected field when coupling runs). */
 velocity:GPUTexture;
 /** Vertex phi. */
 phi:GPUTexture;
 /** The rigid system's exchange (12 i32 words per body). */
 exchange:GPUBuffer;
 /** Accepted h pressures: slot map at word, followed by 64 pressures per slot.
  * Read each encode because the frame may grow its backing buffer. */
 pressure:()=>{buffer:GPUBuffer;word:number};
}

/** Rigid bodies on the mixed frame, GPU resident.
 *
 * couple: integrates the solved h pressure over the body's cut faces. The
 * reaction p (blockedPlus - blockedMinus) area dt is the transpose of the
 * moving-wall flux in the pressure RHS. It includes hydrostatic buoyancy and
 * impact pressure, so the integrator must not add empirical drag, buoyancy
 * or added mass a second time. Interior velocity matching was not this
 * reaction: its drag depended on extension into the solid and removed
 * momentum without a corresponding fluid impulse.
 *
 * Bodies are h wherever liquid can reach them. The displaced wet volume
 * remains a diagnostic in the native fixed-point exchange; the ambient
 * velocity (the drag of a body no pressure couples) is not sampled.
 *
 * markTiles: one lane per tile (with a page pool, a fixed grid striding the
 * resident blocks' tile keys: a free block's tile is never marked), ors every tile a body's shape,
 * dilated by its surface's travel over `horizon` seconds plus one cell, can touch into
 * the census's solid-coupled mask, from the GPU body state (free bodies are
 * integrated on the GPU; the host roster is stale). The bounding sphere is
 * only the prefilter: a long flat hull's sphere holds whole columns of wet
 * tiles its surface never reaches. */
export class UniformMixedBodies {
 private readonly coupleLayout:GPUBindGroupLayout;
 private readonly tilesLayout:GPUBindGroupLayout;
 private readonly params:GPUBuffer;
 private coupleGroup?:UniformDetailGroup;
 private pressureBuffer?:GPUBuffer;
 private tilesGroup?:{tiles:GPUBuffer;group:UniformDetailGroup};
 private couplePipeline?:GPUComputePipeline;
 private tilesPipeline?:GPUComputePipeline;
 /** Workgroups of markTiles without a pool: one lane per tile key. */
 private readonly tileGroups:[number,number];
 /** encodeTiles' horizon words: one array for every write. */
 private readonly horizon=new Float32Array(4);
 readonly allocatedBytes=16;
 constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,private readonly solid:UniformMixedSolid,private readonly fields:UniformMixedBodyFields,private readonly pool?:UniformMixedSolidPool){
  const texture={sampleType:"unfilterable-float" as const,viewDimension:"3d" as const};
  this.tileGroups=uniformKeyGroups(ownership.capacity.address.tileKeys,64,ownership.dispatchX,[0,0]);
  this.coupleLayout=uniformDetailBindLayout(device,{label:"Uniform mixed rigid coupling",entries:[
   {binding:0,visibility:GPUShaderStage.COMPUTE,texture},{binding:1,visibility:GPUShaderStage.COMPUTE,texture},
   {binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
   {binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
  ]});
  this.tilesLayout=uniformDetailBindLayout(device,{label:"Uniform mixed rigid body tiles",entries:[
   {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
   {binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
   ...(pool?[{binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage" as const}}]:[]),
  ]});
  this.params=device.createBuffer({label:"Uniform mixed rigid body tiles horizon",size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
 }
 async initialize():Promise<void>{
  // couple strides the GPU-counted owners of every tier (umAllOwner); markTiles covers the lattice.
  const module=uniformDetailModule(this.device,{label:"Uniform mixed rigid bodies",code:uniformMixedCountedEntriesWGSL(uniformMixedTopologyWGSL(this.ownership.capacity,0)+/* wgsl */`
@group(1) @binding(0) var bodyPhi:texture_3d<f32>;
// binding 1 (the velocity) fed the ambient probes, no longer sampled.
@group(1) @binding(2) var<storage,read_write> rigidExchange:array<atomic<i32>>;
@group(1) @binding(5) var<storage,read> bodyPressure:array<f32>;
@group(1) @binding(3) var<storage,read_write> bodyTiles:array<atomic<u32>>;
@group(1) @binding(4) var<uniform> bodyHorizon:vec4f;
${this.pool?uniformLiveListsWGSL(1,6,this.pool.layout):""}
${uniformMixedSolidWGSL(2,undefined,true,false,this.solid.features)}
// Air rows publish zero; closed cells have no physical pressure traction.
fn umBodyPressure(p:vec3i)->f32{
 let tile=umTileKey(p>>vec3u(2u));let slot=bitcast<u32>(bodyPressure[${this.fields.pressure().word}u+tile]);
 if(slot==0u||umCellOpen(p)<=1e-5){return 0.0;}
 let l=vec3u(p)&vec3u(3u);
 return bodyPressure[${this.fields.pressure().word+this.ownership.capacity.tiles}u+(slot-1u)*64u+l.x+4u*(l.y+4u*l.z)];
}
// An owner's centre phi (the trilinear vertex phi at its centre) and wet
// fraction, surfaceOccupancy's level-set branch.
fn umBodyOwnerWet(o:UMOwner)->f32{
 let origin=vec3i(umOrigin(o));let w=i32(o.width);var phi=0.0;
 for(var k=0u;k<8u;k++){phi+=textureLoad(bodyPhi,origin+w*vec3i(umCorner(k,2u)),0).x;}
 return clamp(0.5-0.125*phi/(4.0*umSolidParams.cellGravity.y),0.0,1.0);
}
// Average opposing MAC faces at the owner centre, where the rigid velocity
// and torque arm are evaluated. A positive-face sample biases impact forces
// and spins a vertically falling sphere toward the positive grid diagonal.
${uniformMixedCoupleWGSL}
fn umBodyMarkTile(tile:u32){
 if(!umTileLive(tile)){return;}
 // bodyTiles is a bit per tile key (the census's body words).
 let coord=umTileCoord(tile);
 let h=umSolidParams.cellGravity.xyz;let t=vec3f(coord);
 let low=umSolidParams.frame.xyz+4.0*t*h;let high=low+4.0*h;
 let middle=0.5*(low+high);let half=length(2.0*h);let cell=max(h.x,max(h.y,h.z));
 for(var i=0u;i<umBodyCount();i++){
  let b=umSolidBodies[i];let centre=b.positionShape.xyz;let radius=max(b.dimensions.w,0.0);
  let travel=length(b.linearVelocity.xyz)*bodyHorizon.x;
  if(distance(clamp(centre,low,high),centre)>radius+travel+cell){continue;}
  // Within the ball: the shape's own 1-Lipschitz distance bounds every point
  // of the tile below by its centre's less the half diagonal. The surface
  // moves at most (|v| + |w| radius) over the horizon. An inverted shape's
  // solid lies outside its ball, so the ball alone decides.
  let margin=(length(b.linearVelocity.xyz)+length(b.angularVelocity.xyz)*radius)*bodyHorizon.x+cell;
  if(b.positionShape.w<0.0||umBodyDistance(i,middle)-half<=margin){atomicOr(&bodyTiles[tile/32u],1u<<(tile%32u));return;}
 }
}
${this.pool?/* wgsl */`@compute @workgroup_size(64) fn markTiles(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 for(var i=gid.x;i<512u*ulResidentCount();i+=64u*groups.x){umBodyMarkTile(512u*ulResidentBlock(i>>9u)+(i&511u));}
}`:/* wgsl */`@compute @workgroup_size(64) fn markTiles(@builtin(global_invocation_id) gid:vec3u){umBodyMarkTile(gid.x+umDispatchX*64u*gid.y);}`}
`,["couple"],"umCounts.x")});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const couple=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.coupleLayout,this.solid.bindLayout]});
  const tiles=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.tilesLayout,this.solid.bindLayout]});
  this.couplePipeline=await this.device.createComputePipelineAsync({layout:couple,compute:{module,entryPoint:"couple",constants:{umDispatchX:this.ownership.dispatchX,umCountedJobs:UNIFORM_MIXED_COUNTED.all}}});
  this.tilesPipeline=await this.device.createComputePipelineAsync({layout:tiles,compute:{module,entryPoint:"markTiles",constants:{umDispatchX:this.ownership.dispatchX}}});
 }
 /** Clears the exchange, then couples every body-covered h owner. */
 encodeCoupling(encoder:GPUCommandEncoder):void{
  if(!this.couplePipeline)throw new Error("Mixed rigid bodies are not initialized");
  encoder.clearBuffer(this.fields.exchange);
  const pressure=this.fields.pressure().buffer;
  if(this.pressureBuffer!==pressure){this.coupleGroup=undefined;this.pressureBuffer=pressure;}
  this.coupleGroup??=uniformDetailGroup(this.device,{layout:this.coupleLayout,entries:[
   {binding:0,resource:this.fields.phi},{binding:1,resource:this.fields.velocity},{binding:2,resource:{buffer:this.fields.exchange}},{binding:5,resource:{buffer:pressure}}]});
  const pass=encoder.beginComputePass({label:"Uniform mixed rigid coupling"});
  pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.coupleGroup.group);pass.setBindGroup(2,this.solid.bindGroup);
  this.ownership.dispatchAllCounted(pass,this.couplePipeline);pass.end();
 }
 /** Ors the tiles bodies may touch within `horizon` seconds into `tiles`. */
 encodeTiles(encoder:GPUCommandEncoder,tiles:GPUBuffer,horizon:number):void{
  if(!this.tilesPipeline)throw new Error("Mixed rigid bodies are not initialized");
  if(!Number.isFinite(horizon)||horizon<0)throw new Error(`Mixed rigid body tile horizon must be finite and non-negative: ${horizon}`);
  this.horizon[0]=horizon;this.device.queue.writeBuffer(this.params,0,this.horizon);
  if(this.tilesGroup?.tiles!==tiles)this.tilesGroup={tiles,group:uniformDetailGroup(this.device,{layout:this.tilesLayout,entries:[{binding:3,resource:{buffer:tiles}},{binding:4,resource:{buffer:this.params}},...(this.pool?[{binding:6,resource:{buffer:this.pool.state}}]:[])]})};
  const pass=encoder.beginComputePass({label:"Uniform mixed rigid body tiles"});
  pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.tilesGroup.group.group);pass.setBindGroup(2,this.solid.bindGroup);
  pass.setPipeline(this.tilesPipeline);if(this.pool)pass.dispatchWorkgroups(UNIFORM_LIVE_KEY_GROUPS);else pass.dispatchWorkgroups(this.tileGroups[0],this.tileGroups[1]);pass.end();
 }
 destroy():void{this.params.destroy();}
}
