/** Embedded solids for fine mixed owners: the native packed voxel mask (in
 * the host's active scratch, at params.dropExtent.z), the terrain
 * heightfield and the GPU-resident rigid bodies (WebGPURigidBodySystem's
 * state buffer, params.boundary.z of them), read through the native
 * parameter block. Every helper is a transcription of its native namesake in
 * webgpu-uniform-reference.wgsl.ts (cellOpenFraction, faceOpenFraction,
 * pressureFaceData, insideAnyTraceSolid). Bodies move: the host rebuilds the
 * all-4h record every frame they exist, displaces liquid out of the cells
 * they enter, and promotes the tiles they sweep (uniform-mixed-bodies.ts).
 * The host's mixed frame always compiles it (live voxel edits and bodies
 * reach a scene built without solids); stages built without it get inert stubs.
 * Coarse (4h) simulation owners never reach these helpers: promotion keeps
 * liquid one full tile away from any cut cell a 4h owner holds (a dry cut
 * tile may run 4h, without solid terms). Band pressure's all-4h levels read
 * the static coarse record instead (see UniformMixedSolid.coarse). */
import {uniformMixedPressureStorage} from "./uniform-mixed-pressure-boundary.wgsl";
import {sceneShapeWgsl} from "../../core/scene-shape";
import type {UniformMixedLayout} from "./uniform-mixed-layout";

export interface UniformMixedSolidResources {
 params:GPUBuffer;scratch:GPUBuffer;terrain:GPUTexture;
 /** WebGPURigidBodySystem.stateBuffer: twelve 128-byte body records,
  * copied into the library's uniform mirror by encodeBodies (a storage
  * binding would exceed the band's per-stage storage budget). */
 bodies:GPUBuffer;
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
 * host's coupled tiles), and whose y is 1 when the simulation holds it at h
 * (encodeSimulation, per relayout). Promotion is liquid-conditional: a dry
 * cut tile may be 4h, and its h texels are then stale, so the all-4h levels
 * treat it as uncut (umSolidCut); the band certificate fails any such tile
 * holding a liquid row. */
export interface UniformMixedSolidCoarse {
 readonly bindLayout:GPUBindGroupLayout;readonly bindGroup:GPUBindGroup;readonly record:GPUBuffer;
 /** Owners plus halo slots: the level-0 topology record's vec4 count. */
 readonly count:number;
}

export class UniformMixedSolid {
 readonly bindLayout:GPUBindGroupLayout;
 readonly bindGroup:GPUBindGroup;
 readonly coupledTiles:number;
 private readonly bodySource:GPUBuffer;
 private readonly bodies:GPUBuffer;
 /** The all-4h record for band pressure (constructed with its all-4h layout). */
 readonly coarse?:UniformMixedSolidCoarse;
 private builder?:{pipeline:GPUComputePipeline;bodies:GPUComputePipeline;group:GPUBindGroup};
 /** The body mirror the record was last built with: a bodies-only rebuild
  * revisits the tiles near it or the current mirror. */
 private builtBodies?:GPUBuffer;
 /** The next build must visit every tile (first build, voxel edit). */
 private full=true;
 private simulation?:{pipeline:GPUComputePipeline;layout:GPUBindGroupLayout;group?:GPUBindGroup;topology?:GPUBuffer};
 private built=false;
 constructor(private readonly device:GPUDevice,resources:UniformMixedSolidResources,private readonly coarseLayout?:UniformMixedLayout){
  if(resources.params.size<272||resources.terrain.format!=="r32float")throw new Error("Mixed solids require the native parameter block and terrain heightfield");
  if(resources.bodies.size<12*128)throw new Error("Mixed solids require the rigid-body state buffer");
  if(!Number.isSafeInteger(resources.coupledTiles)||resources.coupledTiles<0)throw new Error("Mixed solids require the host's coupled tile count");
  this.coupledTiles=resources.coupledTiles;
  this.bodySource=resources.bodies;
  this.bodies=device.createBuffer({label:"Uniform mixed solid body mirror",size:12*128,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
  const bodies=this.bodies;
  const entries:GPUBindGroupLayoutEntry[]=[
   {binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
   {binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
   {binding:2,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"2d"}},
   {binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
  ];
  const resourcesOf=():GPUBindGroupEntry[]=>[
   {binding:0,resource:{buffer:resources.params,size:272}},{binding:1,resource:{buffer:resources.scratch}},
   {binding:2,resource:resources.terrain.createView()},{binding:4,resource:{buffer:bodies}},
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
 get allocatedBytes():number{return (this.coarse?.record.size??0)+this.bodies.size+(this.builtBodies?.size??0);}
 /** Mirror the rigid state into the library: at a frame's head, and after
  * the rigid integration for the census that follows it. */
 encodeBodies(encoder:GPUCommandEncoder):void{encoder.copyBufferToBuffer(this.bodySource,0,this.bodies,0,12*128);}
 async initialize():Promise<void>{
  const layout=this.coarseLayout,coarse=this.coarse;if(!layout||!coarse||this.builder)return;
  const d=layout.lattice.dimensions,dispatchX=this.device.limits.maxComputeWorkgroupsPerDimension;
  const module=this.device.createShaderModule({label:"Uniform mixed all-4h solid record",code:/* wgsl */`
const UM_D=vec3u(${d.map(n=>`${n}u`).join(",")});const UM_T=UM_D/4u;const UM_TILES:u32=${layout.tiles.length}u;const UM_SOLID_COUNT:u32=${coarse.count}u;
${uniformMixedSolidWGSL(0)}
@group(1) @binding(0) var<storage,read_write> record:array<vec4f>;
override umDispatchX:u32=65535u;
@group(1) @binding(1) var<uniform> umBuiltBodies:array<UMRigidBody,12>;
// Bodies-only rebuild: a tile far from every body, as last built and now,
// keeps its record (its cut ring and face samples reach two cells out).
override umBodyTiles:bool=false;
fn umBodyTileNear(b:UMRigidBody,lo:vec3f,hi:vec3f,margin:f32)->bool{
 if(b.dimensions.w<=0.0){return false;}
 let p=b.positionShape.xyz;return distance(clamp(p,lo,hi),p)<=b.dimensions.w+margin;
}
fn umBodyTileDirty(tc:vec3u)->bool{
 let h=umSolidParams.cellGravity.xyz;let lo=umSolidWorldCell(vec3i(tc*4u))-0.5*h;let hi=lo+4.0*h;
 let margin=2.0*max(h.x,max(h.y,h.z));
 for(var i=0u;i<12u;i++){if(umBodyTileNear(umSolidBodies[i],lo,hi,margin)||umBodyTileNear(umBuiltBodies[i],lo,hi,margin)){return true;}}
 return false;
}
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
 if(umBodyTiles&&!umBodyTileDirty(tc)){return;}
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
  const output=this.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}]});
  const builderLayout=this.device.createPipelineLayout({bindGroupLayouts:[this.bindLayout,output]});
  const [pipeline,bodies]=await Promise.all([0,1].map(umBodyTiles=>this.device.createComputePipelineAsync({layout:builderLayout,compute:{module,entryPoint:"build",constants:{umDispatchX:dispatchX,umBodyTiles}}})));
  const built=this.builtBodies=this.device.createBuffer({label:"Uniform mixed solid record body mirror",size:12*128,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  this.builder={pipeline:pipeline!,bodies:bodies!,group:this.device.createBindGroup({layout:output,entries:[{binding:0,resource:{buffer:coarse.record}},{binding:1,resource:{buffer:built}}]})};
  const flagModule=this.device.createShaderModule({label:"Uniform mixed solid simulation widths",code:/* wgsl */`
const UM_TILES:u32=${layout.tiles.length}u;const UM_SOLID_COUNT:u32=${coarse.count}u;
override umDispatchX:u32=65535u;
@group(0) @binding(0) var<storage,read_write> record:array<vec4f>;
@group(0) @binding(1) var<storage,read> topology:array<u32>;
// y of each tile's cut record: the simulation tile word is h.
@compute @workgroup_size(64) fn widths(@builtin(global_invocation_id) g:vec3u){
 let t=g.x+umDispatchX*64u*g.y;if(t>=UM_TILES){return;}
 record[UM_SOLID_COUNT+t].y=select(0.0,1.0,(topology[t]&0x80000000u)!=0u);
}`});
  const flagErrors=(await flagModule.getCompilationInfo()).messages.filter(m=>m.type==="error");if(flagErrors.length)throw new Error(flagErrors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const flagLayout=this.device.createBindGroupLayout({label:"Uniform mixed solid simulation widths",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}}]});
  this.simulation={layout:flagLayout,pipeline:await this.device.createComputePipelineAsync({layout:this.device.createPipelineLayout({bindGroupLayouts:[flagLayout]}),compute:{module:flagModule,entryPoint:"widths",constants:{umDispatchX:dispatchX}}})};
 }
 /** Builds the all-4h record from the solids the native host has published
  * (its parameter block, voxel mask and body mirror): once, in full again
  * after a live edit, and after bodies move only the tiles near them as
  * last built or now. */
 encodeCoarse(encoder:GPUCommandEncoder):void{
  const layout=this.coarseLayout;if(!layout||this.built)return;
  if(!this.builder)throw new Error("The coarse solid record is not initialized");
  const tiles=layout.tiles.length,x=Math.min(tiles,this.device.limits.maxComputeWorkgroupsPerDimension);
  const pass=encoder.beginComputePass({label:"Uniform mixed all-4h solid record"});
  pass.setPipeline(this.full?this.builder.pipeline:this.builder.bodies);pass.setBindGroup(0,this.bindGroup);pass.setBindGroup(1,this.builder.group);pass.dispatchWorkgroups(x,Math.ceil(tiles/x));pass.end();
  encoder.copyBufferToBuffer(this.bodies,0,this.builtBodies!,0,12*128);this.full=false;
  this.built=true;
 }
 /** A live voxel edit changed the mask (every tile), or only bodies moved
  * (the tiles near them): rebuild the all-4h record. */
 invalidate(bodiesOnly=false):void{this.built=false;if(!bodiesOnly)this.full=true;}
 /** Records which tiles the simulation holds at h (the record's y), from
  * its tile words. Encode after encodeCoarse and after every relayout,
  * before the all-4h levels read umSolidCut. */
 encodeSimulation(encoder:GPUCommandEncoder,topology:GPUBufferBinding):void{
  const layout=this.coarseLayout,coarse=this.coarse,s=this.simulation;if(!layout||!coarse)return;
  if(!s)throw new Error("The coarse solid record is not initialized");
  if(s.topology!==topology.buffer){s.topology=topology.buffer;s.group=this.device.createBindGroup({layout:s.layout,entries:[{binding:0,resource:{buffer:coarse.record}},{binding:1,resource:topology}]});}
  const groups=Math.ceil(layout.tiles.length/64),x=Math.min(groups,this.device.limits.maxComputeWorkgroupsPerDimension);
  const pass=encoder.beginComputePass({label:"Uniform mixed solid simulation widths"});
  pass.setPipeline(s.pipeline);pass.setBindGroup(0,s.group!);pass.dispatchWorkgroups(x,Math.ceil(groups/x));pass.end();
 }
 destroy():void{this.coarse?.record.destroy();this.bodies.destroy();this.builtBodies?.destroy();}
}

/** Requires UM_D. `group` undefined emits inert stubs with the same ABI.
 * `coarse` (the record's owner/halo count) binds the all-4h record: the
 * group is then UniformMixedSolid.coarse.bindLayout, and umSolidCoarse(i),
 * umSolidCut(tile) (cut and simulated at h) and umSolidStaticCut(tile) read it. */
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
fn umSolidFaceVelocity(id:vec3i,axis:u32)->f32{return 0.0;}
fn umBodyCount()->u32{return 0u;}
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
struct UMRigidBody {
 positionShape:vec4f,dimensions:vec4f,orientation:vec4f,linearVelocity:vec4f,
 angularVelocity:vec4f,inverseMassInertia:vec4f,angularMomentumRestitution:vec4f,material:vec4f,
}
@group(${group}) @binding(4) var<uniform> umSolidBodies:array<UMRigidBody,12>;
fn umSolidEnabled()->bool{return true;}
${coarse===undefined?"":`@group(${group}) @binding(3) var<storage,read> umSolidRecord:array<vec4f>;
fn umSolidCoarse(i:u32)->vec4f{return umSolidRecord[i];}
fn umSolidStaticCut(t:u32)->bool{return umSolidRecord[${coarse}u+t].x>0.5;}
// A cut tile whose h texels are live: the simulation holds it at h.
fn umSolidCut(t:u32)->bool{let r=umSolidRecord[${coarse}u+t];return r.x>0.5&&r.y>0.5;}`}
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
fn umSolidWorldCell(id:vec3i)->vec3f{let h=umSolidParams.cellGravity.xyz;
 return vec3f(-0.5*umSolidParams.container.x+(f32(id.x)+0.5)*h.x,(f32(id.y)+0.5)*h.y,-0.5*umSolidParams.container.z+(f32(id.z)+0.5)*h.z);}
// Rigid bodies: the native RigidBody helpers over the shape table. Every
// body query first rejects on the body's bounding sphere (state dimensions.w,
// boundingRadius), so cells away from every body pay one distance per body.
${sceneShapeWgsl()}
fn umBodyCount()->u32{return min(u32(max(round(umSolidParams.boundary.z),0.0)),12u);}
fn umQuatRotate(q:vec4f,v:vec3f)->vec3f{let uv=cross(q.yzw,v);let uuv=cross(q.yzw,uv);return v+2.0*(q.x*uv+uuv);}
fn umBodyLocal(i:u32,world:vec3f)->vec3f{let b=umSolidBodies[i];return umQuatRotate(vec4f(b.orientation.x,-b.orientation.yzw),world-b.positionShape.xyz);}
fn umBodyNear(i:u32,world:vec3f,margin:f32)->bool{let b=umSolidBodies[i];return distance(world,b.positionShape.xyz)<=b.dimensions.w+margin;}
fn umInsideBody(i:u32,world:vec3f)->bool{let b=umSolidBodies[i];return umBodyNear(i,world,0.0)&&rigidShapeInside(i32(round(b.positionShape.w)),b.dimensions.xyz,umBodyLocal(i,world));}
fn umBodyDistance(i:u32,world:vec3f)->f32{let b=umSolidBodies[i];return rigidShapeDistance(i32(round(b.positionShape.w)),b.dimensions.xyz,umBodyLocal(i,world));}
fn umBodyVelocity(i:u32,world:vec3f)->vec3f{let b=umSolidBodies[i];return b.linearVelocity.xyz+cross(b.angularVelocity.xyz,world-b.positionShape.xyz);}
// rigidBodyIndexAt: the first body containing the point, -1 for none.
fn umBodyAt(world:vec3f)->i32{
 for(var i=0u;i<umBodyCount();i++){if(umInsideBody(i,world)){return i32(i);}}
 return -1;
}
// bodySolidFraction: the CPU voxelizer's eight sub-cell samples.
fn umBodyFraction(i:u32,p:vec3i)->f32{
 let h=umSolidParams.cellGravity.xyz;let centre=umSolidWorldCell(p);
 if(!umBodyNear(i,centre,length(0.4*h))){return 0.0;}
 var inside=0.0;
 for(var corner=0u;corner<8u;corner++){
  let offset=vec3f(select(-0.4,0.4,(corner&1u)!=0u),select(-0.4,0.4,(corner&2u)!=0u),select(-0.4,0.4,(corner&4u)!=0u));
  if(umInsideBody(i,centre+offset*h)){inside+=1.0;}
 }
 return inside/8.0;
}
// bodySolidFractionAt: the largest body coverage of the cell.
fn umCellBodyFraction(p:vec3i)->f32{
 var fraction=0.0;
 for(var i=0u;i<umBodyCount();i++){fraction=max(fraction,umBodyFraction(i,p));}
 return fraction;
}
// cellOpenFraction.
fn umCellOpen(p:vec3i)->f32{
 if(!umSolidValid(p)||umSolidVoxel(p)){return 0.0;}
 return clamp((1.0-umCellBodyFraction(p))*(1.0-umCellTerrainFraction(p)),0.0,1.0);
}
// cellInsideSolid: voxel, terrain or a body at the centre -- the CM11a p_min=0 rows.
fn umCellInsideSolid(p:vec3i)->bool{
 if(umSolidVoxel(p)){return true;}
 if(!umSolidValid(p)){return false;}
 return umCellInsideTerrain(p)||umBodyAt(umSolidWorldCell(p))>=0;
}
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
fn umSolidAtWorld(world:vec3f)->bool{return umSolidVoxelAtWorld(world)||umSolidTerrainAtWorld(world)||umBodyAt(world)>=0;}
// insideAnyTraceSolid for a lattice-coordinate point: voxels and bodies
// (terrain is not a trace solid natively).
fn umSolidAtWorldCell(p:vec3f)->bool{let world=umSolidTraceWorld(p);return umSolidVoxelAtWorld(world)||umBodyAt(world)>=0;}
// pressureFaceData(id,axis)[axis]: the moving-wall velocity u_s of a MAC
// face. Its dual-cell samples classify ownership; each covering body's
// rigid velocity is evaluated at the face centre (static voxels and terrain
// are zero). A face whose samples hold no solid takes the nearest body's
// velocity within one cell (extrapolatedRigidVelocityAtFace), else zero.
// Domain faces and voxel-adjacent faces are static walls: zero.
fn umSolidFaceVelocity(id:vec3i,axis:u32)->f32{
 if(umBodyCount()==0u){return 0.0;}
 var neighbor=id;neighbor[axis]+=1;
 if(!umSolidValid(id)||!umSolidValid(neighbor)||umSolidVoxel(id)||umSolidVoxel(neighbor)){return 0.0;}
 let world=umSolidFaceWorld(id,axis);let h=umSolidParams.cellGravity.xyz;let reach=max(h.x,max(h.y,h.z));
 var near=false;
 for(var i=0u;i<umBodyCount();i++){near=near||umBodyNear(i,world,2.0*reach);}
 if(!near){return 0.0;}
 var solid=0.0;var velocity=0.0;
 for(var sampleIndex=0u;sampleIndex<8u;sampleIndex+=1u){
  let sampleWorld=world+vec3f(select(-0.4,0.4,(sampleIndex&1u)!=0u)*h.x,select(-0.4,0.4,(sampleIndex&2u)!=0u)*h.y,select(-0.4,0.4,(sampleIndex&4u)!=0u)*h.z);
  if(umSolidVoxelAtWorld(sampleWorld)||umSolidTerrainAtWorld(sampleWorld)){solid+=1.0;continue;}
  let body=umBodyAt(sampleWorld);
  if(body>=0){solid+=1.0;velocity+=umBodyVelocity(u32(body),world)[axis];}
 }
 if(solid>0.0){return velocity/solid;}
 var nearest=reach;var result=0.0;
 for(var i=0u;i<umBodyCount();i++){
  let d=abs(umBodyDistance(i,world));
  if(d<=nearest){nearest=d;result=umBodyVelocity(i,world)[axis];}
 }
 return result;
}
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
// pressureFaceData(id,axis).w, geometric and static: the CM11a dual-cell V
// (except a depth symmetry plane, below).
fn umPressureFaceV(id:vec3i,axis:u32)->f32{
 var neighbor=id;neighbor[axis]+=1;
 if(umSolidValid(id)!=umSolidValid(neighbor)){
  // A depth symmetry plane is a closed wall here, as for solid-free mixed
  // frames and every coarse owner (the native CM11a record used V=0).
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
