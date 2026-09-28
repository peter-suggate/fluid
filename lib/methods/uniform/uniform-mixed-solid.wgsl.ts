/** Static embedded solids for fine mixed owners: the native packed voxel mask
 * (in the host's active scratch, at params.dropExtent.z) and the terrain
 * heightfield, read through the native parameter block. Every helper is a
 * transcription of its native namesake in webgpu-uniform-reference.wgsl.ts;
 * body terms are absent because rigid bodies are rejected by the host.
 * A frame compiles this library only for a scene with cut cells; otherwise
 * stages get inert stubs and keep their solid-free code, bit-identical.
 * Coarse (2h/4h) simulation owners never reach these helpers: promotion keeps
 * them one full tile away from any cut cell. Band pressure's all-4h levels
 * read the static coarse record instead (see UniformMixedSolid.coarse). */
import {uniformMixedPressureStorage} from "./uniform-mixed-pressure-boundary.wgsl";
import type {UniformMixedLayout} from "./uniform-mixed-layout";

export interface UniformMixedSolidResources {
 params:GPUBuffer;scratch:GPUBuffer;terrain:GPUTexture;
 /** Tiles the host promotes as solid-coupled (uniformMixedSolidTiles().coupled):
  * the h pressure band's reserve on top of its surface capacity. */
 coupledTiles:number;
}

/** The static all-4h solid record of band pressure (one vec4 per 4h owner,
 * owner index = tile key): (open, V+x, V+y, V+z), the two-level native
 * mgDownsampleTopology of the h CM11a record (open is the mean of the 64 h
 * cells, a face V the mean of its 16 h faces). Then one vec4 per all-4h halo
 * slot (umBoundaryIndex order) whose x is that wall's mean h V: together the
 * level-0 topology of band pressure. Then one vec4 per tile whose x is 1 when
 * the tile is cut: an h cell of it or of its one-cell ring has open < 1 (the
 * host's coupled promotion, so a cut tile is always h in the simulation). */
export interface UniformMixedSolidCoarse {
 readonly bindLayout:GPUBindGroupLayout;readonly bindGroup:GPUBindGroup;readonly record:GPUBuffer;
 /** Owners plus halo slots: the level-0 topology record's vec4 count. */
 readonly count:number;
}

export class UniformMixedSolid {
 readonly bindLayout:GPUBindGroupLayout;
 readonly bindGroup:GPUBindGroup;
 readonly coupledTiles:number;
 /** The all-4h record for band pressure (constructed with its all-4h layout). */
 readonly coarse?:UniformMixedSolidCoarse;
 private builder?:{pipeline:GPUComputePipeline;group:GPUBindGroup};
 private built=false;
 constructor(private readonly device:GPUDevice,resources:UniformMixedSolidResources,private readonly coarseLayout?:UniformMixedLayout){
  if(resources.params.size<272||resources.terrain.format!=="r32float")throw new Error("Mixed solids require the native parameter block and terrain heightfield");
  if(!Number.isSafeInteger(resources.coupledTiles)||resources.coupledTiles<0)throw new Error("Mixed solids require the host's coupled tile count");
  this.coupledTiles=resources.coupledTiles;
  const entries:GPUBindGroupLayoutEntry[]=[
   {binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
   {binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
   {binding:2,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"2d"}},
  ];
  const resourcesOf=():GPUBindGroupEntry[]=>[
   {binding:0,resource:{buffer:resources.params,size:272}},{binding:1,resource:{buffer:resources.scratch}},
   {binding:2,resource:resources.terrain.createView()},
  ];
  this.bindLayout=device.createBindGroupLayout({label:"Uniform mixed static solids",entries});
  this.bindGroup=device.createBindGroup({layout:this.bindLayout,entries:resourcesOf()});
  if(coarseLayout){
   if(coarseLayout.tiles.some(word=>(word&0xc0000000)!==0)||coarseLayout.cellCount!==coarseLayout.tiles.length)throw new Error("The coarse solid record requires the all-4h layout");
   const count=uniformMixedPressureStorage(coarseLayout).count,tiles=coarseLayout.tiles.length;
   const record=device.createBuffer({label:"Uniform mixed all-4h solid record",size:16*(count+tiles),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
   const bindLayout=device.createBindGroupLayout({label:"Uniform mixed static solids with the all-4h record",entries:[...entries,
    {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}}]});
   this.coarse={bindLayout,record,count,bindGroup:device.createBindGroup({layout:bindLayout,entries:[...resourcesOf(),{binding:3,resource:{buffer:record}}]})};
  }
 }
 get allocatedBytes():number{return this.coarse?.record.size??0;}
 async initialize():Promise<void>{
  const layout=this.coarseLayout,coarse=this.coarse;if(!layout||!coarse||this.builder)return;
  const d=layout.lattice.dimensions,dispatchX=this.device.limits.maxComputeWorkgroupsPerDimension;
  const module=this.device.createShaderModule({label:"Uniform mixed all-4h solid record",code:/* wgsl */`
const UM_D=vec3u(${d.map(n=>`${n}u`).join(",")});const UM_T=UM_D/4u;const UM_TILES:u32=${layout.tiles.length}u;const UM_SOLID_COUNT:u32=${coarse.count}u;
${uniformMixedSolidWGSL(0)}
@group(1) @binding(0) var<storage,read_write> record:array<vec4f>;
override umDispatchX:u32=65535u;
var<workgroup> sums:array<vec4f,64>;
var<workgroup> lows:array<vec4f,64>;
var<workgroup> cut:atomic<u32>;
// umBoundaryIndex of the all-4h layout (cells = tiles, width 4).
fn umSolidHalo(t:vec3u,axis:u32,side:u32)->u32{
 let d=UM_T;
 if(axis==0u){return UM_TILES+side*d.y*d.z+t.y+d.y*t.z;}
 if(axis==1u){return UM_TILES+2u*d.y*d.z+side*d.x*d.z+t.x+d.x*t.z;}
 return UM_TILES+2u*(d.y*d.z+d.x*d.z)+side*d.x*d.y+t.x+d.x*t.y;
}
@compute @workgroup_size(64) fn build(@builtin(workgroup_id) g:vec3u,@builtin(local_invocation_index) lane:u32){
 let t=g.x+umDispatchX*g.y;if(t>=UM_TILES){return;}
 let tc=vec3u(t%UM_T.x,(t/UM_T.x)%UM_T.y,t/(UM_T.x*UM_T.y));
 let l=vec3u(lane%4u,(lane/4u)%4u,lane/16u);let c=vec3i(tc*4u+l);
 if(lane==0u){atomicStore(&cut,0u);}
 var own=vec4f(umCellOpen(c),0.0,0.0,0.0);var low=vec4f(0.0);
 for(var axis=0u;axis<3u;axis++){
  if(l[axis]==3u){own[axis+1u]=umPressureFaceV(c,axis);}
  if(l[axis]==0u){var q=c;q[axis]-=1;low[axis+1u]=umPressureFaceV(q,axis);}
 }
 workgroupBarrier();
 for(var i=lane;i<216u;i+=64u){
  let q=vec3i(tc*4u)-vec3i(1)+vec3i(vec3u(i%6u,(i/6u)%6u,i/36u));
  if(umSolidValid(q)&&umCellOpen(q)<1.0){atomicStore(&cut,1u);}
 }
 sums[lane]=own;lows[lane]=low;
 workgroupBarrier();
 for(var stride=32u;stride>0u;stride/=2u){if(lane<stride){sums[lane]+=sums[lane+stride];lows[lane]+=lows[lane+stride];}workgroupBarrier();}
 if(lane==0u){
  let s=sums[0];let lo=lows[0];
  record[t]=vec4f(s.x/64.0,s.y/16.0,s.z/16.0,s.w/16.0);
  record[UM_SOLID_COUNT+t]=vec4f(select(0.0,1.0,atomicLoad(&cut)!=0u),0.0,0.0,0.0);
  for(var axis=0u;axis<3u;axis++){
   if(tc[axis]==0u){record[umSolidHalo(tc,axis,0u)]=vec4f(lo[axis+1u]/16.0,0.0,0.0,0.0);}
   if(tc[axis]==UM_T[axis]-1u){record[umSolidHalo(tc,axis,1u)]=vec4f(s[axis+1u]/16.0,0.0,0.0,0.0);}
  }
 }
}`});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const output=this.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
  const pipeline=await this.device.createComputePipelineAsync({layout:this.device.createPipelineLayout({bindGroupLayouts:[this.bindLayout,output]}),compute:{module,entryPoint:"build",constants:{umDispatchX:dispatchX}}});
  this.builder={pipeline,group:this.device.createBindGroup({layout:output,entries:[{binding:0,resource:{buffer:coarse.record}}]})};
 }
 /** Builds the static all-4h record once, from the solids the native host has
  * published by the first advance (its parameter block and voxel mask). */
 encodeCoarse(encoder:GPUCommandEncoder):void{
  const layout=this.coarseLayout;if(!layout||this.built)return;
  if(!this.builder)throw new Error("The coarse solid record is not initialized");
  const tiles=layout.tiles.length,x=Math.min(tiles,this.device.limits.maxComputeWorkgroupsPerDimension);
  const pass=encoder.beginComputePass({label:"Uniform mixed all-4h solid record"});
  pass.setPipeline(this.builder.pipeline);pass.setBindGroup(0,this.bindGroup);pass.setBindGroup(1,this.builder.group);pass.dispatchWorkgroups(x,Math.ceil(tiles/x));pass.end();
  this.built=true;
 }
 destroy():void{this.coarse?.record.destroy();}
}

/** Requires UM_D. `group` undefined emits inert stubs with the same ABI.
 * `coarse` (the record's owner/halo count) binds the all-4h record: the
 * group is then UniformMixedSolid.coarse.bindLayout, and umSolidCoarse(i) and
 * umSolidCut(tile) read it. */
export function uniformMixedSolidWGSL(group?:number,coarse?:number):string{
 if(group===undefined)return /* wgsl */ `
fn umSolidEnabled()->bool{return false;}
fn umSolidValid(p:vec3i)->bool{return all(p>=vec3i(0))&&all(p<vec3i(UM_D));}
fn umSolidVoxel(p:vec3i)->bool{return false;}
fn umCellOpen(p:vec3i)->f32{return select(0.0,1.0,umSolidValid(p));}
fn umCellInsideSolid(p:vec3i)->bool{return false;}
fn umFaceOpen(id:vec3i,axis:u32)->f32{return 1.0;}
fn umPressureFaceV(id:vec3i,axis:u32)->f32{return 1.0;}
fn umSolidAtWorldCell(p:vec3f)->bool{return false;}
`;
 return /* wgsl */ `
struct UMSolidParams {
 dimsDt:vec4f,cellGravity:vec4f,container:vec4f,physical:vec4f,boundary:vec4f,
 inflowA:vec4f,inflowB:vec4f,inflowC:vec4f,tuning:vec4f,drop:vec4f,dropExtent:vec4f,
 twoLevel:vec4f,agreement:vec4f,lean:vec4f,splash:vec4f,splashB:vec4f,cleanup:vec4f,
}
@group(${group}) @binding(0) var<uniform> umSolidParams:UMSolidParams;
@group(${group}) @binding(1) var<storage,read> umSolidScratch:array<u32>;
@group(${group}) @binding(2) var umSolidTerrain:texture_2d<f32>;
fn umSolidEnabled()->bool{return true;}
${coarse===undefined?"":`@group(${group}) @binding(3) var<storage,read> umSolidRecord:array<vec4f>;
fn umSolidCoarse(i:u32)->vec4f{return umSolidRecord[i];}
fn umSolidCut(t:u32)->bool{return umSolidRecord[${coarse}u+t].x>0.5;}`}
fn umSolidValid(p:vec3i)->bool{return all(p>=vec3i(0))&&all(p<vec3i(UM_D));}
// staticSolidVoxelOccupied: lattice plus a one-cell halo (the box shell).
fn umSolidVoxel(p:vec3i)->bool{
 let shape=vec3i(UM_D)+vec3i(2);let q=p+vec3i(1);
 if(any(q<vec3i(0))||any(q>=shape)){return false;}
 let index=u32(q.x+shape.x*(q.y+shape.y*q.z));
 return (umSolidScratch[u32(round(umSolidParams.dropExtent.z))+4u+(index>>5u)]&(1u<<(index&31u)))!=0u;
}
fn umSolidHasTerrain()->bool{return umSolidParams.container.w>0.5;}
fn umSolidTerrainHeight(x:i32,z:i32)->f32{let d=vec3i(UM_D);return textureLoad(umSolidTerrain,vec2i(clamp(x,0,d.x-1),clamp(z,0,d.z-1)),0).x;}
fn umCellInsideTerrain(p:vec3i)->bool{if(!umSolidHasTerrain()){return false;}return f32(p.y)+0.5<umSolidTerrainHeight(p.x,p.z);}
fn umCellTerrainFraction(p:vec3i)->f32{if(!umSolidHasTerrain()){return 0.0;}return clamp(umSolidTerrainHeight(p.x,p.z)-f32(p.y),0.0,1.0);}
// cellOpenFraction without bodies.
fn umCellOpen(p:vec3i)->f32{
 if(!umSolidValid(p)||umSolidVoxel(p)){return 0.0;}
 return clamp((1.0-0.0)*(1.0-umCellTerrainFraction(p)),0.0,1.0);
}
// cellInsideSolid||cellInsideTerrain, the CM11a p_min=0 rows.
fn umCellInsideSolid(p:vec3i)->bool{return umSolidVoxel(p)||(umSolidValid(p)&&umCellInsideTerrain(p));}
fn umSolidWorldCell(id:vec3i)->vec3f{let h=umSolidParams.cellGravity.xyz;
 return vec3f(-0.5*umSolidParams.container.x+(f32(id.x)+0.5)*h.x,(f32(id.y)+0.5)*h.y,-0.5*umSolidParams.container.z+(f32(id.z)+0.5)*h.z);}
fn umSolidFaceWorld(id:vec3i,axis:u32)->vec3f{var world=umSolidWorldCell(id);world[axis]+=0.5*umSolidParams.cellGravity.xyz[axis];return world;}
fn umSolidTraceWorld(p:vec3f)->vec3f{let h=umSolidParams.cellGravity.xyz;
 return vec3f(-0.5*umSolidParams.container.x+p.x*h.x,p.y*h.y,-0.5*umSolidParams.container.z+p.z*h.z);}
fn umSolidVoxelAtWorld(world:vec3f)->bool{
 let h=umSolidParams.cellGravity.xyz;
 let p=vec3i(floor(vec3f((world.x+0.5*umSolidParams.container.x)/h.x,world.y/h.y,(world.z+0.5*umSolidParams.container.z)/h.z)));
 return umSolidVoxel(p);
}
fn umSolidTerrainAtWorld(world:vec3f)->bool{
 if(!umSolidHasTerrain()){return false;}
 let h=umSolidParams.cellGravity.xyz;let d=vec3i(UM_D);
 let x=clamp(i32(floor((world.x+0.5*umSolidParams.container.x)/h.x)),0,d.x-1);
 let z=clamp(i32(floor((world.z+0.5*umSolidParams.container.z)/h.z)),0,d.z-1);
 return world.y<umSolidTerrainHeight(x,z)*h.y;
}
fn umSolidAtWorld(world:vec3f)->bool{return umSolidVoxelAtWorld(world)||umSolidTerrainAtWorld(world);}
// insideAnyTraceSolid for a lattice-coordinate point: voxels only (terrain
// is not a trace solid natively).
fn umSolidAtWorldCell(p:vec3f)->bool{return umSolidVoxelAtWorld(umSolidTraceWorld(p));}
// faceOpenFraction: the Sec. 3.6 transverse face aperture.
fn umFaceOpen(id:vec3i,axis:u32)->f32{
 var neighbor=id;neighbor[axis]+=1;
 if(umSolidVoxel(id)||umSolidVoxel(neighbor)){return 0.0;}
 if(!umSolidValid(id)||!umSolidValid(neighbor)){return 1.0;}
 let world=umSolidFaceWorld(id,axis);let h=umSolidParams.cellGravity.xyz;
 let tangentA=(axis+1u)%3u;let tangentB=(axis+2u)%3u;var solid=0.0;
 for(var sampleIndex=0u;sampleIndex<4u;sampleIndex+=1u){
  var sampleWorld=world;
  sampleWorld[tangentA]+=select(-0.35,0.35,(sampleIndex&1u)!=0u)*h[tangentA];
  sampleWorld[tangentB]+=select(-0.35,0.35,(sampleIndex&2u)!=0u)*h[tangentB];
  solid+=select(0.0,1.0,umSolidAtWorld(sampleWorld));
 }
 return 1.0-solid*0.25;
}
// pressureFaceData(id,axis).w, geometric and static: the CM11a dual-cell V.
fn umPressureFaceV(id:vec3i,axis:u32)->f32{
 var neighbor=id;neighbor[axis]+=1;
 if(umSolidValid(id)!=umSolidValid(neighbor)){
  if(axis==2u&&umSolidParams.tuning.w>0.5){return 0.0;}
  let ambient=axis==1u&&max(id.y,neighbor.y)==i32(UM_D.y)&&umSolidParams.boundary.w>0.5;
  let open=umCellOpen(select(neighbor,id,umSolidValid(id)));
  return select(0.5*open,0.5*(1.0+open),ambient);
 }
 if(umSolidVoxel(id)||umSolidVoxel(neighbor)){return 0.5*(umCellOpen(id)+umCellOpen(neighbor));}
 if(!umSolidValid(id)||!umSolidValid(neighbor)){return 0.0;}
 let world=umSolidFaceWorld(id,axis);let h=umSolidParams.cellGravity.xyz;var solid=0.0;
 for(var sampleIndex=0u;sampleIndex<8u;sampleIndex+=1u){
  let sampleWorld=world+vec3f(select(-0.4,0.4,(sampleIndex&1u)!=0u)*h.x,select(-0.4,0.4,(sampleIndex&2u)!=0u)*h.y,select(-0.4,0.4,(sampleIndex&4u)!=0u)*h.z);
  solid+=select(0.0,1.0,umSolidAtWorld(sampleWorld));
 }
 return 1.0-solid/8.0;
}
`;
}
