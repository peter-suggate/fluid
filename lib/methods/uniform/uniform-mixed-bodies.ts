import { uniformDetailBindLayout, uniformDetailModule, uniformDetailGroup, type UniformDetailGroup } from "./uniform-detail-fields";
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
}

/** Rigid bodies on the mixed frame, GPU resident.
 *
 * couple: the native coupleRigid over h owners, after the projection. The
 * body with the greatest sub-cell coverage owns a cell; its reaction is the
 * drag-like impulse -rho h^3 wet s (u_s - u) blend, its torque arm x
 * reaction, and it accumulates the displaced wet volume and the ambient
 * velocity (six wet probes beyond the bounding sphere) for the integrator's
 * buoyancy and form drag, in the native fixed-point words. Bodies are h by
 * the census wherever liquid can reach them, so 4h owners carry no body
 * cells with liquid (the band certificate fails a liquid row in a cut 4h
 * tile); 4h owners only serve as ambient probes.
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
 private tilesGroup?:{tiles:GPUBuffer;group:UniformDetailGroup};
 private couplePipeline?:GPUComputePipeline;
 private tilesPipeline?:GPUComputePipeline;
 readonly allocatedBytes=16;
 constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,private readonly solid:UniformMixedSolid,private readonly fields:UniformMixedBodyFields){
  const texture={sampleType:"unfilterable-float" as const,viewDimension:"3d" as const};
  this.coupleLayout=uniformDetailBindLayout(device,{label:"Uniform mixed rigid coupling",entries:[
   {binding:0,visibility:GPUShaderStage.COMPUTE,texture},{binding:1,visibility:GPUShaderStage.COMPUTE,texture},
   {binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
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
@group(1) @binding(3) var<storage,read_write> bodyTiles:array<atomic<u32>>;
@group(1) @binding(4) var<uniform> bodyHorizon:vec4f;
${uniformMixedSolidWGSL(2)}
// An owner's centre phi (the trilinear vertex phi at its centre) and wet
// fraction, surfaceOccupancy's level-set branch.
fn umBodyOwnerWet(o:UMOwner)->f32{
 let origin=vec3i(umOrigin(o));let w=i32(o.width);var phi=0.0;
 for(var k=0u;k<8u;k++){phi+=textureLoad(bodyPhi,origin+w*vec3i(umCorner(k,2u)),0).x;}
 return clamp(0.5-0.125*phi/(4.0*umSolidParams.cellGravity.y),0.0,1.0);
}
// Its +face velocities, each at its anchor (origin + (width-1) on the axis).
fn umBodyOwnerVelocity(o:UMOwner)->vec3f{
 let origin=vec3i(umOrigin(o));let last=i32(o.width)-1;
 return vec3f(textureLoad(bodyVelocity,origin+vec3i(last,0,0),0).x,textureLoad(bodyVelocity,origin+vec3i(0,last,0),0).y,textureLoad(bodyVelocity,origin+vec3i(0,0,last),0).z);
}
// ambientFluidVelocity: six wet, open probes beyond the bounding sphere.
fn umBodyAmbient(i:u32,p:vec3i,fallback:vec3f)->vec3f{
 let h=umSolidParams.cellGravity.xyz;let radius=max(umSolidBodies[i].dimensions.w,0.0);
 let reach=vec3i(ceil(vec3f(2.0*radius)/h))+vec3i(2);
 var total=vec3f(0.0);var weight=0.0;
 for(var n=0u;n<6u;n++){
  let axis=n/2u;var q=p;q[axis]+=select(-reach[axis],reach[axis],(n&1u)!=0u);
  if(!umSolidValid(q)||umSolidVoxel(q)||umBodyAt(umSolidWorldCell(q))>=0||umCellInsideTerrain(q)){continue;}
  let o=umOwnerAt(q);if(o.width==0u){continue;}
  let wet=umBodyOwnerWet(o);total+=wet*umBodyOwnerVelocity(o);weight+=wet;
 }
 return select(fallback,total/max(weight,1e-6),weight>0.0);
}
@compute @workgroup_size(64) fn couple(@builtin(global_invocation_id) gid:vec3u){
 let o=umAllOwner(gid);if(o.width!=1u){return;}
 let id=vec3i(umOrigin(o));var owner=12u;var fraction=0.0;
 for(var i=0u;i<umBodyCount();i++){let s=umBodyFraction(i,id);if(s>fraction){fraction=s;owner=i;}}
 if(owner>=12u){return;}
 let h=umSolidParams.cellGravity.xyz;let wet=umBodyOwnerWet(o);let v=umBodyOwnerVelocity(o);
 let cellMass=umSolidParams.physical.x*h.x*h.y*h.z*wet;let blend=clamp(45.0*umSolidParams.dimsDt.w,0.0,1.0);
 let world=umSolidWorldCell(id);let arm=world-umSolidBodies[owner].positionShape.xyz;
 let reaction=-cellMass*fraction*(umBodyVelocity(owner,world)-v)*blend;let torque=cross(arm,reaction);
 let ambient=umBodyAmbient(owner,id,v);let base=owner*12u;
 atomicAdd(&rigidExchange[base],i32(round(reaction.x*1000000.0)));atomicAdd(&rigidExchange[base+1u],i32(round(reaction.y*1000000.0)));atomicAdd(&rigidExchange[base+2u],i32(round(reaction.z*1000000.0)));
 atomicAdd(&rigidExchange[base+3u],i32(round(torque.x*1000000.0)));atomicAdd(&rigidExchange[base+4u],i32(round(torque.y*1000000.0)));atomicAdd(&rigidExchange[base+5u],i32(round(torque.z*1000000.0)));
 let displaced=wet*fraction;
 atomicAdd(&rigidExchange[base+6u],i32(round(displaced*65536.0)));
 atomicAdd(&rigidExchange[base+7u],i32(round(displaced*ambient.x*10000.0)));atomicAdd(&rigidExchange[base+8u],i32(round(displaced*ambient.y*10000.0)));atomicAdd(&rigidExchange[base+9u],i32(round(displaced*ambient.z*10000.0)));
}
@compute @workgroup_size(64) fn markTiles(@builtin(global_invocation_id) gid:vec3u){
 let tile=gid.x+umDispatchX*64u*gid.y;if(tile>=UM_TILES){return;}
 let h=umSolidParams.cellGravity.xyz;let t=vec3f(umTileCoord(tile));
 let low=vec3f(-0.5*umSolidParams.container.x,0.0,-0.5*umSolidParams.container.z)+4.0*t*h;let high=low+4.0*h;
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
  this.coupleGroup??=uniformDetailGroup(this.device,{layout:this.coupleLayout,entries:[
   {binding:0,resource:this.fields.phi},{binding:1,resource:this.fields.velocity},{binding:2,resource:{buffer:this.fields.exchange}}]});
  const pass=encoder.beginComputePass({label:"Uniform mixed rigid coupling"});
  pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.coupleGroup.group);pass.setBindGroup(2,this.solid.bindGroup);
  this.ownership.dispatchAllCounted(pass,this.couplePipeline);pass.end();
 }
 /** Ors the tiles bodies may touch within `horizon` seconds into `tiles`. */
 encodeTiles(encoder:GPUCommandEncoder,tiles:GPUBuffer,horizon:number):void{
  if(!this.tilesPipeline)throw new Error("Mixed rigid bodies are not initialized");
  if(!Number.isFinite(horizon)||horizon<0)throw new Error(`Mixed rigid body tile horizon must be finite and non-negative: ${horizon}`);
  this.device.queue.writeBuffer(this.params,0,new Float32Array([horizon,0,0,0]));
  if(this.tilesGroup?.tiles!==tiles)this.tilesGroup={tiles,group:uniformDetailGroup(this.device,{layout:this.tilesLayout,entries:[{binding:3,resource:{buffer:tiles}},{binding:4,resource:{buffer:this.params}}]})};
  const groups=Math.ceil(this.ownership.capacity.tiles/64),x=this.ownership.dispatchX;
  const pass=encoder.beginComputePass({label:"Uniform mixed rigid body tiles"});
  pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.tilesGroup.group.group);pass.setBindGroup(2,this.solid.bindGroup);
  pass.setPipeline(this.tilesPipeline);pass.dispatchWorkgroups(Math.min(groups,x),Math.ceil(groups/x));pass.end();
 }
 destroy():void{this.params.destroy();}
}
