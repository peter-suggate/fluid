import { uniformDetailBindLayout, uniformDetailModule, uniformDetailGroup, type UniformDetailGroup } from "./uniform-detail-fields";
import {uniformKeyGroups} from "./uniform-lattice-address";
import type {UniformMixedOwnership} from "./uniform-mixed-ownership";
import {UNIFORM_MIXED_COUNTED,uniformMixedCountedEntriesWGSL,uniformMixedTopologyWGSL} from "./uniform-mixed-topology.wgsl";
import {uniformMixedSolidWGSL,type UniformMixedSolid} from "./uniform-mixed-solid.wgsl";

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
 * Bodies are h wherever liquid can reach them. The displaced wet volume and
 * ambient probes remain diagnostics in the native fixed-point exchange.
 *
 * markTiles: one lane per tile, ors every tile a body's bounding sphere,
 * dilated by its travel over `horizon` seconds plus one cell, can touch into
 * the census's solid-coupled mask, from the GPU body state (free bodies are
 * integrated on the GPU; the host roster is stale). */
export class UniformMixedBodies {
 private readonly coupleLayout:GPUBindGroupLayout;
 private readonly tilesLayout:GPUBindGroupLayout;
 private readonly params:GPUBuffer;
 private coupleGroup?:UniformDetailGroup;
 private pressureBuffer?:GPUBuffer;
 private tilesGroup?:{tiles:GPUBuffer;group:UniformDetailGroup};
 private couplePipeline?:GPUComputePipeline;
 private tilesPipeline?:GPUComputePipeline;
 /** Workgroups of markTiles: one lane per tile key. */
 private readonly tileGroups:[number,number];
 /** encodeTiles' horizon words: one array for every write. */
 private readonly horizon=new Float32Array(4);
 readonly allocatedBytes=16;
 constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,private readonly solid:UniformMixedSolid,private readonly fields:UniformMixedBodyFields){
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
  ]});
  this.params=device.createBuffer({label:"Uniform mixed rigid body tiles horizon",size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
 }
 async initialize():Promise<void>{
  // couple strides the GPU-counted owners of every tier (umAllOwner); markTiles covers the lattice.
  const module=uniformDetailModule(this.device,{label:"Uniform mixed rigid bodies",code:uniformMixedCountedEntriesWGSL(uniformMixedTopologyWGSL(this.ownership.capacity,0)+/* wgsl */`
@group(1) @binding(0) var bodyPhi:texture_3d<f32>;
@group(1) @binding(1) var bodyVelocity:texture_3d<f32>;
@group(1) @binding(2) var<storage,read_write> rigidExchange:array<atomic<i32>>;
@group(1) @binding(5) var<storage,read> bodyPressure:array<f32>;
@group(1) @binding(3) var<storage,read_write> bodyTiles:array<atomic<u32>>;
@group(1) @binding(4) var<uniform> bodyHorizon:vec4f;
${uniformMixedSolidWGSL(2)}
// Air rows publish zero; closed cells have no physical pressure traction.
fn umBodyPressure(p:vec3i)->f32{
 let tile=umTileKey(p>>vec3u(2u));let slot=bitcast<u32>(bodyPressure[${this.fields.pressure().word}u+tile]);
 if(slot==0u||umCellOpen(p)<=1e-5){return 0.0;}
 let l=vec3u(p)&vec3u(3u);
 return bodyPressure[${this.fields.pressure().word+this.ownership.capacity.tiles}u+(slot-1u)*64u+l.x+4u*(l.y+4u*l.z)];
}
// Match the pressure operator's four transverse aperture samples.
fn umBodyFaceFraction(i:u32,p:vec3i,axis:u32)->f32{
 let world=umSolidFaceWorld(p,axis);let h=umSolidParams.cellGravity.xyz;var blocked=0.0;
 for(var k=0u;k<4u;k++){
  var sample=world;let a=(axis+1u)%3u;let b=(axis+2u)%3u;
  sample[a]+=select(-0.35,0.35,(k&1u)!=0u)*h[a];sample[b]+=select(-0.35,0.35,(k&2u)!=0u)*h[b];
  blocked+=select(0.0,0.25,umInsideBody(i,sample));
 }return blocked;
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
fn umBodyOwnerVelocity(o:UMOwner)->vec3f{
 let origin=vec3i(umOrigin(o));let last=i32(o.width)-1;
 return 0.5*vec3f(
  textureLoad(bodyVelocity,origin+vec3i(last,0,0),0).x+textureLoad(bodyVelocity,origin-vec3i(1,0,0),0).x,
  textureLoad(bodyVelocity,origin+vec3i(0,last,0),0).y+textureLoad(bodyVelocity,origin-vec3i(0,1,0),0).y,
  textureLoad(bodyVelocity,origin+vec3i(0,0,last),0).z+textureLoad(bodyVelocity,origin-vec3i(0,0,1),0).z);
}
// ambientFluidVelocity: six wet, open probes beyond the bounding sphere.
fn umBodyAmbient(i:u32,p:vec3i,fallback:vec3f)->vec3f{
 let h=umSolidParams.cellGravity.xyz;let radius=max(umSolidBodies[i].dimensions.w,0.0);
 let reach=vec3i(ceil(vec3f(2.0*radius)/h))+vec3i(2);
 var total=vec3f(0.0);var weight=0.0;
 for(var n=0u;n<6u;n++){
  let axis=n/2u;var q=p;q[axis]+=select(-reach[axis],reach[axis],(n&1u)!=0u);
  if(umSolidVoxel(q)||umBodyAt(umSolidWorldCell(q))>=0||umCellInsideTerrain(q)){continue;}
  let o=umOwnerAt(q);if(o.width==0u){continue;}
  let wet=umBodyOwnerWet(o);total+=wet*umBodyOwnerVelocity(o);weight+=wet;
 }
 return select(fallback,total/max(weight,1e-6),weight>0.0);
}
@compute @workgroup_size(64) fn couple(@builtin(global_invocation_id) gid:vec3u){
 let o=umAllOwner(gid);if(o.width!=1u){return;}
 let id=vec3i(umOrigin(o));let h=umSolidParams.cellGravity.xyz;let world=umSolidWorldCell(id);
 for(var owner=0u;owner<umBodyCount();owner++){
  if(!umBodyNear(owner,world,length(h))){continue;}
  let pressure=umBodyPressure(id);let base=owner*12u;
  var area=vec3f(0.0);
  for(var axis=0u;axis<3u;axis++){
   var low=id;low[axis]-=1;
   area[axis]=(umBodyFaceFraction(owner,id,axis)-umBodyFaceFraction(owner,low,axis))*h[(axis+1u)%3u]*h[(axis+2u)%3u];
  }
  let reaction=pressure*area*umSolidParams.dimsDt.w;
  let torque=cross(world-umSolidBodies[owner].positionShape.xyz,reaction);
  atomicAdd(&rigidExchange[base],i32(round(reaction.x*1000000.0)));atomicAdd(&rigidExchange[base+1u],i32(round(reaction.y*1000000.0)));atomicAdd(&rigidExchange[base+2u],i32(round(reaction.z*1000000.0)));
  atomicAdd(&rigidExchange[base+3u],i32(round(torque.x*1000000.0)));atomicAdd(&rigidExchange[base+4u],i32(round(torque.y*1000000.0)));atomicAdd(&rigidExchange[base+5u],i32(round(torque.z*1000000.0)));
  atomicStore(&rigidExchange[base+10u],1);
  let fraction=umBodyFraction(owner,id);if(fraction==0.0){continue;}
  let displaced=umBodyOwnerWet(o)*fraction;
  atomicAdd(&rigidExchange[base+6u],i32(round(displaced*65536.0)));
  atomicAdd(&rigidExchange[base+11u],i32(round(displaced*65536.0)));
  let ambient=umBodyAmbient(owner,id,umBodyOwnerVelocity(o));
  atomicAdd(&rigidExchange[base+7u],i32(round(displaced*ambient.x*10000.0)));atomicAdd(&rigidExchange[base+8u],i32(round(displaced*ambient.y*10000.0)));atomicAdd(&rigidExchange[base+9u],i32(round(displaced*ambient.z*10000.0)));
 }
}
@compute @workgroup_size(64) fn markTiles(@builtin(global_invocation_id) gid:vec3u){
 let tile=gid.x+umDispatchX*64u*gid.y;if(!umTileLive(tile)){return;}
 // bodyTiles is a bit per tile key (the census's body words).
 let coord=umTileCoord(tile);
 let h=umSolidParams.cellGravity.xyz;let t=vec3f(coord);
 let low=umSolidParams.frame.xyz+4.0*t*h;let high=low+4.0*h;
 for(var i=0u;i<umBodyCount();i++){
  let b=umSolidBodies[i];let centre=b.positionShape.xyz;
  let reach=max(b.dimensions.w,0.0)+length(b.linearVelocity.xyz)*bodyHorizon.x+max(h.x,max(h.y,h.z));
  if(distance(clamp(centre,low,high),centre)<=reach){atomicOr(&bodyTiles[tile/32u],1u<<(tile%32u));return;}
 }
}
`,["couple"])});
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
  if(this.tilesGroup?.tiles!==tiles)this.tilesGroup={tiles,group:uniformDetailGroup(this.device,{layout:this.tilesLayout,entries:[{binding:3,resource:{buffer:tiles}},{binding:4,resource:{buffer:this.params}}]})};
  const pass=encoder.beginComputePass({label:"Uniform mixed rigid body tiles"});
  pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.tilesGroup.group.group);pass.setBindGroup(2,this.solid.bindGroup);
  pass.setPipeline(this.tilesPipeline);pass.dispatchWorkgroups(this.tileGroups[0],this.tileGroups[1]);pass.end();
 }
 destroy():void{this.params.destroy();}
}
