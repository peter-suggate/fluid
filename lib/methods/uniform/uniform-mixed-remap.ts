import {UniformMixedOwnership,type UniformMixedBuiltOwnership} from "./uniform-mixed-ownership";
import type {UniformMixedLayout} from "./uniform-mixed-layout";
import {uniformMixedTopologyWGSL} from "./uniform-mixed-topology.wgsl";
import {uniformMixedFaceAddressWGSL} from "./uniform-mixed-face-dispatch.wgsl";
import {uniformVolumeTargetWGSL} from "./uniform-volume.wgsl";
import {geometricPlaneBoxWGSL} from "../../core/geometric-plane-box.wgsl";

interface Fields{volume:GPUTexture;velocity:GPUTexture;phi:GPUTexture;negative:GPUBuffer}
/** Conservative edit-boundary remap. Only tiles at a width change are
 * visited, and each does only what the edit alters; every other owner's remap
 * is the identity, so its live value stays in place. Two lists (markChanged):
 * - Cells: changed tiles. Volume and phi are remapped in place: a tile's
 *   donors lie inside it, and a vertex is written only when it becomes stored
 *   (next to a refinement), while the old sampler reads stored vertices only.
 * - Faces: changed tiles and 4h tiles whose positive face neighbour changed
 *   (an h tile's faces are its own cells' patches). Faces read the tile
 *   below across the plane, so they go through the scratch field and back. */
export class UniformMixedRemap {
 readonly target:UniformMixedOwnership;
 get allocatedBytes(){return this.target.allocatedBytes+this.worklist.size;}
 private readonly cellResources:GPUBindGroupLayout;
 private readonly faceResources:GPUBindGroupLayout;
 private readonly cellGroup:GPUBindGroup;
 private readonly faceGroups:readonly [GPUBindGroup,GPUBindGroup];
 private readonly pipelines=new Map<string,GPUComputePipeline>();
 /** [three unused words, cell count, three unused, face count, cell tiles
  * (n), face tiles (n)]: the two changed-tile worklists. */
 private readonly worklist:GPUBuffer;
 /** Workgroups of each remap launch: a fixed grid strides over the listed
  * tiles, bounded by the layout's tiles and capped where the GPU saturates. */
 private readonly grid:number;
 constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,input:Fields,scratch:Fields){
  const n=ownership.capacity.tileCount;
  this.target=new UniformMixedOwnership(device,ownership.layout,false);
  this.worklist=device.createBuffer({label:"Uniform mixed remap worklists",size:(8+2*n)*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  this.grid=Math.min(1024,n);
  const C=GPUShaderStage.COMPUTE;
  this.cellResources=device.createBindGroupLayout({entries:[
   ...[0,1].map(binding=>({binding,visibility:C,storageTexture:{access:"read-write" as const,format:"r32float" as GPUTextureFormat,viewDimension:"3d" as const}})),
   {binding:2,visibility:C,buffer:{type:"storage"}},
  ]});
  this.faceResources=device.createBindGroupLayout({entries:[
   {binding:0,visibility:C,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},
   {binding:1,visibility:C,buffer:{type:"read-only-storage"}},
   {binding:2,visibility:C,storageTexture:{access:"write-only",format:"rgba32float",viewDimension:"3d"}},
   {binding:3,visibility:C,buffer:{type:"storage"}},
   {binding:4,visibility:C,buffer:{type:"read-only-storage"}},
  ]});
  this.cellGroup=device.createBindGroup({layout:this.cellResources,entries:[
   {binding:0,resource:input.volume.createView()},{binding:1,resource:input.phi.createView()},{binding:2,resource:{buffer:this.worklist}}]});
  this.faceGroups=[this.bindFaces(input,scratch),this.bindFaces(scratch,input)];
 }
 private bindFaces(a:{velocity:GPUTexture;negative:GPUBuffer},b:{velocity:GPUTexture;negative:GPUBuffer}):GPUBindGroup{
  return this.device.createBindGroup({layout:this.faceResources,entries:[
   {binding:0,resource:a.velocity.createView()},{binding:1,resource:{buffer:a.negative}},
   {binding:2,resource:b.velocity.createView()},{binding:3,resource:{buffer:b.negative}},{binding:4,resource:{buffer:this.worklist}}]});
 }
 async initialize():Promise<void>{
  const layout=this.ownership.layout;
  const compile=async(code:string,resources:GPUBindGroupLayout,entries:string[])=>{
   const module=this.device.createShaderModule({code});
   const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
   const pipelineLayout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.target.bindLayout,resources]});
   for(const entryPoint of entries)this.pipelines.set(entryPoint,await this.device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint,constants:{umDispatchX:this.ownership.dispatchX}}}));
  };
  await compile(uniformMixedRemapCellsWGSL(layout),this.cellResources,["markChanged","remapCells"]);
  await compile(uniformMixedRemapFacesWGSL(layout),this.faceResources,["remapFaces","copyFaces"]);
 }
 apply(layout:UniformMixedLayout):void{
  if(this.pipelines.size!==4)throw new Error("Live remap has not been initialized");
  this.target.update(layout);
  const encode=(copy:boolean)=>{
   const e=this.device.createCommandEncoder({label:copy?"Uniform publish remapped owners":"Uniform remap changed ownership"});
   this.encodePass(e,copy);this.device.queue.submit([e.finish()]);
  };
  encode(false);this.ownership.update(layout);encode(true);
 }
 /** The same remap for a GPU-built generation, in one encoder: adopt into
  * the target, remap, adopt into the live ownership, publish. */
 applyBuilt(encoder:GPUCommandEncoder,built:UniformMixedBuiltOwnership):void{
  if(this.pipelines.size!==4)throw new Error("Live remap has not been initialized");
  this.target.adopt(encoder,built);this.encodePass(encoder,false);
  this.ownership.adopt(encoder,built);this.encodePass(encoder,true);
 }
 private encodePass(e:GPUCommandEncoder,copy:boolean):void{
  // The lists compare the live (old) and target widths, so they are built
  // before the live ownership adopts the target and reused by the publish.
  const begin=(label:string,group:GPUBindGroup)=>{
   const pass=e.beginComputePass({label});pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.target.bindGroup);pass.setBindGroup(2,group);return pass;
  };
  const run=(label:string,group:GPUBindGroup,name:string,groups=this.grid)=>{const pass=begin(label,group);pass.setPipeline(this.pipelines.get(name)!);pass.dispatchWorkgroups(groups);pass.end();};
  if(!copy){
   e.clearBuffer(this.worklist,0,32);
   const groups=Math.ceil(this.target.tileCount/64);
   const list=begin("Uniform mixed remap worklist",this.cellGroup);
   list.setPipeline(this.pipelines.get("markChanged")!);list.dispatchWorkgroups(Math.min(groups,this.target.dispatchX),Math.ceil(groups/this.target.dispatchX));
   list.end();
   run("Uniform mixed remap cells",this.cellGroup,"remapCells");
   run("Uniform mixed remap faces",this.faceGroups[0],"remapFaces");
   return;
  }
  run("Uniform mixed remap publish",this.faceGroups[1],"copyFaces");
 }
 destroy():void{this.target.destroy();this.worklist.destroy();}
}

/** Frame-internal transfer between the h/4h simulation ownership and the
 * all-4h pressure ownership of one lattice (the band pressure's split).
 * Neither ownership changes, and every tile is one 4h pressure owner. Each
 * direction is two fixed direct launches over the simulation ownership's own
 * tier lists, their live counts read on the GPU (umCounts): one lane per 4h
 * tile (a regular 4h owner is a handful of texels; a seam one walks its split
 * planes serially), then one workgroup job per h tile. No whole-field copy.
 * Only what the other side reads is written: a 4h owner's origin volume texel,
 * its three positive-face anchor texels (origin + 3 on the face's axis, the
 * face in that component, the released-wall bits in w) and its negative
 * domain-wall entries; every h owner's texel and wall entries. Pressure phi
 * needs no transfer: the simulation phi is resolved, so its texels at the 4h
 * corners are the pressure ownership's vertex values.
 * - To pressure: a 4h face is the mean of the simulation patches on it (a
 *   4h simulation face with one patch is copied), so each owner's flux is
 *   preserved exactly; volume is the mean of the h cells; a wall face is
 *   released only when its whole footprint was.
 * - To simulation: an h face interpolates the 4h owner's two faces on its
 *   axis at its plane, a 4h face with one patch is copied, one of sixteen
 *   patches takes its 4h face. */
const TRANSFER_GRID=1024;
export class UniformMixedOwnershipTransfer {
 readonly allocatedBytes=0;
 private readonly resources:GPUBindGroupLayout;
 private readonly pipelines=new Map<string,GPUComputePipeline>();
 /** Fixed grids: lanes over the 4h list, workgroup jobs over the h list. */
 private readonly coarseGroups:number;
 private readonly fineGroups:number;
 constructor(private readonly device:GPUDevice,readonly simulation:UniformMixedOwnership,readonly pressure:UniformMixedOwnership){
  if(simulation.layout.tiles.length!==pressure.layout.tiles.length)throw new Error("Ownership transfer requires one tile lattice");
  const tiles=simulation.layout.tiles.length;
  this.coarseGroups=Math.max(1,Math.min(TRANSFER_GRID,Math.ceil(tiles/64)));this.fineGroups=Math.max(1,Math.min(TRANSFER_GRID,tiles));
  if(pressure.layout.tiles.some(word=>(word&0xc0000000)!==0))throw new Error("Ownership transfer targets the all-4h pressure ownership");
  this.resources=device.createBindGroupLayout({entries:[
   ...[0,1].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
   {binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
   ...[3,4].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only" as const,format:(binding===4?"rgba32float":"r32float") as GPUTextureFormat,viewDimension:"3d" as const}})),
   {binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
  ]});
 }
 /** Fields read in the source ownership and written in the target's. The
  * volume pair is read and written by the transfer to pressure only. */
 bind(input:{volume:GPUTexture;velocity:GPUTexture;negative:GPUBuffer},output:{volume:GPUTexture;velocity:GPUTexture;negative:GPUBuffer}):GPUBindGroup{
  return this.device.createBindGroup({layout:this.resources,entries:[
   {binding:0,resource:input.volume.createView()},{binding:1,resource:input.velocity.createView()},{binding:2,resource:{buffer:input.negative}},
   {binding:3,resource:output.volume.createView()},{binding:4,resource:output.velocity.createView()},{binding:5,resource:{buffer:output.negative}},
  ]});
 }
 async initialize():Promise<void>{
  const module=this.device.createShaderModule({label:"Uniform mixed ownership transfer",code:uniformMixedOwnershipTransferWGSL(this.simulation.layout)});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.simulation.bindLayout,this.resources]});
  for(const entryPoint of ["toPressureCoarse","toPressureFine","toSimulationCoarse","toSimulationFine"])this.pipelines.set(entryPoint,await this.device.createComputePipelineAsync({layout,compute:{module,entryPoint,constants:{umDispatchX:this.simulation.dispatchX}}}));
 }
 /** Both launches write disjoint tiles and read only the source fields. */
 private encode(e:GPUCommandEncoder,entry:"toPressure"|"toSimulation",group:GPUBindGroup):void{
  const coarse=this.pipelines.get(`${entry}Coarse`),fine=this.pipelines.get(`${entry}Fine`);if(!coarse||!fine)throw new Error("Ownership transfer has not been initialized");
  const pass=e.beginComputePass({label:entry==="toPressure"?"Uniform mixed transfer to pressure":"Uniform mixed transfer to simulation"});
  pass.setBindGroup(0,this.simulation.bindGroup);pass.setBindGroup(1,group);
  pass.setPipeline(coarse);pass.dispatchWorkgroups(this.coarseGroups);
  pass.setPipeline(fine);pass.dispatchWorkgroups(this.fineGroups);
  pass.end();
 }
 /** Simulation to pressure: volume, faces and negative walls. */
 encodeToPressure(e:GPUCommandEncoder,group:GPUBindGroup):void{this.encode(e,"toPressure",group);}
 /** Pressure to simulation: faces and negative walls. */
 encodeToSimulation(e:GPUCommandEncoder,group:GPUBindGroup):void{this.encode(e,"toSimulation",group);}
 destroy():void{}
}

/** The transfer's kernels (UniformMixedOwnershipTransfer). Group 0 is the
 * simulation ownership in both directions; the pressure side is all-4h. */
function uniformMixedOwnershipTransferWGSL(layout:UniformMixedLayout):string{
 return uniformMixedTopologyWGSL(layout,0)+uniformMixedFaceAddressWGSL+/* wgsl */`
@group(1) @binding(0) var volume:texture_3d<f32>;
@group(1) @binding(1) var velocity:texture_3d<f32>;
@group(1) @binding(2) var<storage,read> negative:array<f32>;
@group(1) @binding(3) var outputVolume:texture_storage_3d<r32float,write>;
@group(1) @binding(4) var output:texture_storage_3d<rgba32float,write>;
@group(1) @binding(5) var<storage,read_write> boundary:array<f32>;
fn tBits(texel:vec4f)->u32{return u32(round(texel.w));}
// The simulation tile at tile coordinate c across the tile's positive face
// on axis a is h: that 4h face holds sixteen simulation patches.
fn tSplit(c:vec3u,a:u32)->bool{var q=c;q[a]+=1u;return q[a]<UM_T[a]&&umTileWidth(umTileAt(q))==1u;}
// Job j of the simulation ownership's h and 4h tile lists.
fn tFineTile(j:u32)->u32{return umTopology[UM_TILES+j];}
fn tCoarseTile(j:u32)->u32{return umTopology[UM_TILES+umCounts.x+j];}
// A 4h tile, one lane: the origin volume texel, the three positive-face
// anchor texels (a split face is the mean of its sixteen patch anchors on the
// plane, summed in tMean's order; never released) and the negative walls.
fn tCoarseToPressure(tile:u32){
 let c=umTileCoord(tile);let base=c*4u;let origin=vec3i(base);
 textureStore(outputVolume,origin,vec4f(textureLoad(volume,origin,0).x));
 var texels:array<vec4f,3>;
 for(var a=0u;a<3u;a++){var anchor=origin;anchor[a]+=3;texels[a]=textureLoad(velocity,anchor,0);}
 var negativeBits=0u;
 for(var b=0u;b<3u;b++){
  if(c[b]!=0u){continue;}
  let index=umNegativeBoundaryIndex(base,b);
  negativeBits|=((tBits(texels[b])>>(b+3u))&1u)<<(b+3u);
  boundary[index]=negative[index];
 }
 for(var a=0u;a<3u;a++){
  var anchor=origin;anchor[a]+=3;let texel=texels[a];
  var face=vec2f(texel[a],f32((tBits(texel)>>a)&1u));
  if(tSplit(c,a)){
   let u=(a+1u)%3u;let v=(a+2u)%3u;var value=0.0;
   for(var y=0u;y<4u;y++){for(var x=0u;x<4u;x++){
    var q=base;q[a]+=3u;q[u]+=x;q[v]+=y;value+=textureLoad(velocity,vec3i(q),0)[a];
   }}
   face=vec2f(value/16.0,0.0);
  }
  var packed=vec4f(0);packed[a]=face.x;
  packed.w=f32(select(0u,u32(face.y)<<a,c[a]+1u==UM_T[a])|negativeBits);
  textureStore(output,anchor,packed);
 }
}
var<workgroup> tVolume:array<f32,64>;
// Per axis: the simulation value on the tile's positive plane (positive
// faces) and on its negative domain plane (wall entries), with the released
// bit of each, at lane + 64 axis.
var<workgroup> tPositive:array<vec2f,192>;
var<workgroup> tNegative:array<vec2f,192>;
// Sixteen plane samples summed in the order the remap's patch mean uses.
// wall: the negative domain plane's samples, else the positive plane's.
fn tMean(wall:bool,a:u32)->vec2f{
 let u=(a+1u)%3u;let v=(a+2u)%3u;var value=0.0;var released=1.0;
 for(var y=0u;y<4u;y++){for(var x=0u;x<4u;x++){
  var q=vec3u(0);q[a]=select(3u,0u,wall);q[u]=x;q[v]=y;let i=q.x+4u*(q.y+4u*q.z)+64u*a;
  let s=select(tPositive[i],tNegative[i],wall);value+=s.x;released=min(released,s.y);
 }}return vec2f(value/16.0,released);
}
// An h tile, one workgroup (a lane per cell): volume is the mean of its
// cells, each 4h face (and negative wall) the mean of its sixteen patches.
fn tFineToPressure(tile:u32,lane:u32){
 let c=umTileCoord(tile);let local=umCorner(lane,4u);let base=c*4u;let cell=vec3i(base+local);
 tVolume[lane]=textureLoad(volume,cell,0).x;let texel=textureLoad(velocity,cell,0);let bits=tBits(texel);
 for(var a=0u;a<3u;a++){
  tPositive[lane+64u*a]=vec2f(texel[a],f32((bits>>a)&1u));
  if(c[a]==0u&&local[a]==0u){tNegative[lane+64u*a]=vec2f(negative[umNegativeBoundaryIndex(vec3u(cell),a)],f32((bits>>(a+3u))&1u));}
 }
 workgroupBarrier();
 if(lane==0u){
  var value=0.0;
  for(var z=0u;z<4u;z++){for(var y=0u;y<4u;y++){for(var x=0u;x<4u;x++){value+=tVolume[x+4u*(y+4u*z)];}}}
  textureStore(outputVolume,cell,vec4f(value/64.0));
 }
 if(lane>=3u){return;}
 // Lane a: the owner's negative wall (every lane, for the released bits it
 // packs), then its positive face on axis a.
 var negativeBits=0u;
 for(var b=0u;b<3u;b++){
  if(c[b]!=0u){continue;}
  let wall=tMean(true,b);
  negativeBits|=u32(wall.y)<<(b+3u);
  if(b==lane){boundary[umNegativeBoundaryIndex(base,b)]=wall.x;}
 }
 let a=lane;var anchor=vec3i(base);anchor[a]+=3;let face=tMean(false,a);
 var packed=vec4f(0);packed[a]=face.x;
 packed.w=f32(select(0u,u32(face.y)<<a,c[a]+1u==UM_T[a])|negativeBits);
 textureStore(output,anchor,packed);
}
@compute @workgroup_size(64) fn toPressureCoarse(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 for(var j=gid.x;j<umCounts.y;j+=64u*groups.x){tCoarseToPressure(tCoarseTile(j));}
}
@compute @workgroup_size(64) fn toPressureFine(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 for(var job=group.x;job<umCounts.x;job+=groups.x){tFineToPressure(tFineTile(job),lane);workgroupBarrier();}
}
// A 4h tile, one lane: its non-split positive faces copied at their anchors,
// split ones written at their plane cells, and its negative walls.
fn tCoarseToSimulation(tile:u32){
 let c=umTileCoord(tile);let base=c*4u;let origin=vec3i(base);
 var low=vec3f(0);var high=vec3f(0);var negativeBits=0u;var splits=0u;
 for(var a=0u;a<3u;a++){
  var anchor=origin;anchor[a]+=3;let texel=textureLoad(velocity,anchor,0);high[a]=texel[a];
  if(c[a]==0u){low[a]=negative[umNegativeBoundaryIndex(base,a)];negativeBits|=tBits(texel)&(1u<<(a+3u));boundary[umNegativeBoundaryIndex(base,a)]=low[a];}
  else{var below=origin;below[a]-=1;low[a]=textureLoad(velocity,below,0)[a];}
  if(tSplit(c,a)){splits|=1u<<a;}else{textureStore(output,anchor,texel);}
 }
 for(var a=0u;a<3u;a++){
  if((splits&(1u<<a))==0u){continue;}
  let u=(a+1u)%3u;let v=(a+2u)%3u;
  for(var y=0u;y<4u;y++){for(var x=0u;x<4u;x++){
   var q=vec3u(0);q[a]=3u;q[u]=x;q[v]=y;
   // A cell on several split planes is written once, by the first.
   var packed=vec4f(0);var first=true;
   for(var b=0u;b<3u;b++){if(q[b]==3u&&(splits&(1u<<b))!=0u){packed[b]=mix(low[b],high[b],1.0);if(b<a){first=false;}}}
   if(first){packed.w=f32(negativeBits);textureStore(output,vec3i(base+q),packed);}
  }}
 }
}
// An h tile, one workgroup: each h face interpolates the 4h owner's two
// faces on its axis at its plane.
fn tFineToSimulation(tile:u32,lane:u32){
 let c=umTileCoord(tile);let local=umCorner(lane,4u);let base=c*4u;let origin=vec3i(base);let cell=vec3i(base+local);
 var low=vec3f(0);var high=vec3f(0);var positiveBits=0u;var negativeBits=0u;
 for(var a=0u;a<3u;a++){
  var anchor=origin;anchor[a]+=3;let texel=textureLoad(velocity,anchor,0);let bits=tBits(texel);high[a]=texel[a];
  if(c[a]+1u==UM_T[a]){positiveBits|=bits&(1u<<a);}
  if(c[a]==0u){low[a]=negative[umNegativeBoundaryIndex(base,a)];negativeBits|=bits&(1u<<(a+3u));}
  else{var below=origin;below[a]-=1;low[a]=textureLoad(velocity,below,0)[a];}
 }
 var packed=vec4f(0);var bits=0u;
 for(var a=0u;a<3u;a++){
  packed[a]=mix(low[a],high[a],f32(local[a]+1u)/4.0);
  if(local[a]==3u){bits|=positiveBits&(1u<<a);}
  if(local[a]==0u&&c[a]==0u){boundary[umNegativeBoundaryIndex(vec3u(cell),a)]=mix(low[a],high[a],0.0);bits|=negativeBits&(1u<<(a+3u));}
 }
 packed.w=f32(bits);textureStore(output,cell,packed);
}
@compute @workgroup_size(64) fn toSimulationCoarse(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 for(var j=gid.x;j<umCounts.y;j+=64u*groups.x){tCoarseToSimulation(tCoarseTile(j));}
}
@compute @workgroup_size(64) fn toSimulationFine(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 for(var job=group.x;job<umCounts.x;job+=groups.x){tFineToSimulation(tFineTile(job),lane);}
}
`;
}

/** Remap kernels between two ownership generations of one lattice. Group 0
 * is the source ("old"-prefixed WGSL), group 1 the target. Only the lattice
 * dimensions are compiled in; both ownerships are live buffers. */
function uniformMixedRemapCommonWGSL(layout:UniformMixedLayout):string{
 return uniformMixedTopologyWGSL(layout,0,"old")+uniformMixedTopologyWGSL(layout,1)+/* wgsl */`
// One workgroup job per listed tile; a fixed grid strides over the list.
const UM_UNLISTED:u32=0xffffffffu;
var<workgroup> listedTile:u32;
var<workgroup> listedJobs:u32;
fn listedJobCount(lane:u32,list:u32)->u32 {
 if(lane==0u){listedJobs=umListLoad(4u*list+3u);}
 return workgroupUniformLoad(&listedJobs);
}
fn listed(job:u32,lane:u32,list:u32)->u32 {
 if(lane==0u){listedTile=umListLoad(8u+list*UM_TILES+job);}
 return workgroupUniformLoad(&listedTile);
}
`;
}
function uniformMixedRemapCellsWGSL(layout:UniformMixedLayout):string{
 return uniformMixedRemapCommonWGSL(layout)+/* wgsl */`
@group(2) @binding(0) var volume:texture_storage_3d<r32float,read_write>;
@group(2) @binding(1) var phi:texture_storage_3d<r32float,read_write>;
@group(2) @binding(2) var<storage,read_write> worklist:array<atomic<u32>>;
fn umListLoad(i:u32)->u32{return atomicLoad(&worklist[i]);}
fn umListPush(list:u32,tile:u32){atomicStore(&worklist[8u+list*UM_TILES+atomicAdd(&worklist[4u*list+3u],1u)],tile);}
// Cells (list 0): every tile that changed width. A vertex becomes stored
// only next to a refinement, so the refined tiles write every such vertex.
// Faces (list 1): every tile that changed width, and every tile 4h in both
// whose positive face neighbour on some axis changed width: that face's
// patches, anchored in this tile's plane cells, split or merge. An h tile's
// patches are its own cells' faces; negative faces belong to the tile below.
@compute @workgroup_size(64) fn markChanged(@builtin(global_invocation_id) gid:vec3u){
 let tile=gid.x+umDispatchX*64u*gid.y;if(tile>=UM_TILES){return;}
 let width=umTileWidth(tile);let before=oldumTileWidth(tile);
 if(width!=before){umListPush(0u,tile);umListPush(1u,tile);return;}
 if(width==1u){return;}
 let p=vec3i(umTileCoord(tile));
 for(var a=0u;a<3u;a++){
  var q=p;q[a]+=1;if(q[a]>=i32(UM_T[a])){continue;}
  let t=umTileAt(vec3u(q));if(umTileWidth(t)!=oldumTileWidth(t)){umListPush(1u,tile);return;}
 }
}
${geometricPlaneBoxWGSL}
fn uvCorner(k:u32)->vec3i{return vec3i(umCorner(k,2u));}
fn d4Sum8(v:array<f32,8>)->f32{return ((v[0]+v[5])+(v[1]+v[4]))+((v[2]+v[7])+(v[3]+v[6]));}
${uniformVolumeTargetWGSL(true,true)}
var<workgroup> tileVolume:array<f32,64>;
var<workgroup> tileFill:array<f32,64>;
// The old layout's phi at the tile's 5x5x5 closure (refined tiles): every
// new owner corner and every vertex this tile writes. The widest incident
// owner holds a vertex, and a refined tile was a 4h owner, so each is its
// aligned corners' trilinear interpolant (on a face or edge shared with
// another 4h owner both interpolate the same corners), evaluated as the
// mixed sampler's regular branch does.
var<workgroup> tileCorners:array<f32,8>;
var<workgroup> tileVertices:array<f32,125>;
fn tileVertex(q:vec3u)->f32{return tileVertices[q.x+5u*(q.y+5u*q.z)];}
// Only a vertex that becomes stored is written (a stored vertex keeps its
// texel: the old sampler returns it exactly). The widest incident owner
// holds a vertex, so off the 4h lattice a vertex is stored exactly when every
// incident tile is h: it becomes stored when all are h now and one was 4h,
// which refined. The lowest such tile writes it, from its closure samples.
fn remapWrites(p:vec3u,tile:u32)->bool{
 if(all(p%4u==vec3u(0))){return false;}
 var first=0xffffffffu;
 for(var k=0u;k<8u;k++){
  let c=vec3i(p)-vec3i(umCorner(k,2u));
  if(any(c<vec3i(0))||any(c>=vec3i(UM_D))){continue;}
  let t=umTileAt(vec3u(c)/4u);
  if(umTileWidth(t)!=1u){return false;}
  if(oldumTileWidth(t)!=1u){first=min(first,t);}
 }
 return first==tile;
}
// In place, one tile per workgroup (the tile's donors lie inside it; reads
// precede the barrier its writes follow).
// Refined (4h to h): every cell's donor is the tile's 4h owner, which does
// not broadcast its volume: its cells take the geometric fill of their new
// owner (the same target the solver compares against, from the remapped phi),
// scaled so the donor's volume is conserved exactly. Filled fractions shrink
// toward zero when the donor holds less than its phi implies, empty
// fractions toward zero when it holds more (up to full), so no fraction
// leaves [0,1]. Re-coarsening averages the cells back to the donor's volume,
// and its phi corners were never altered, so coarse -> fine -> coarse is the
// identity. Coarsened (h to 4h): the owner averages its cells; it stores no
// new vertex.
fn remapTileCells(job:u32,lane:u32){
 let tile=listed(job,lane,0u);
 let base=umTileCoord(tile)*4u;let cell=base+umCorner(lane,4u);
 let width=umTileWidth(tile);let before=oldumTileWidth(tile);
 let refined=width<before;
 if(refined){
  if(lane<8u){tileCorners[lane]=textureLoad(phi,vec3i(base+umCorner(lane,2u)*4u)).x;}
  tileVolume[lane]=textureLoad(volume,vec3i(base)).x;
 }else{tileVolume[lane]=textureLoad(volume,vec3i(cell)).x;}
 workgroupBarrier();textureBarrier();
 if(refined){
  for(var v=lane;v<125u;v+=64u){
   let t=vec3f(umCorner(v,5u))/4.0;var values:array<f32,8>;
   for(var k=0u;k<8u;k++){let w=select(vec3f(1)-t,t,umCorner(k,2u)!=vec3u(0));values[k]=tileCorners[k]*w.x*w.y*w.z;}
   tileVertices[v]=d4Sum8(values);
  }
 }
 workgroupBarrier();
 if(refined){
  var vertices:array<f32,8>;let local=umCorner(lane,4u);
  for(var j=0u;j<8u;j++){vertices[j]=tileVertex(local+umCorner(j,2u));}
  tileFill[lane]=umSurfaceTarget(umOwnerAt(vec3i(cell)),vertices);
 }
 workgroupBarrier();
 if(refined){
  var fill=0.0;
  for(var z=0u;z<4u;z++){for(var y=0u;y<4u;y++){for(var x=0u;x<4u;x++){fill+=tileFill[x+4u*(y+4u*z)];}}}
  // A donor whose phi is (nearly) all empty or all full has no shape to
  // follow, and an overfull donor (V > 1) keeps its excess uniform.
  let held=tileVolume[lane]*64.0;let f=tileFill[lane];let room=64.0-fill;var value=tileVolume[lane];
  if(held<=fill&&fill>=0.5){value=f*held/fill;}
  else if(held>fill&&held<=64.0&&room>=0.5){value=1.0-(1.0-f)*(64.0-held)/room;}
  textureStore(volume,vec3i(cell),vec4f(value));
  for(var v=lane;v<125u;v+=64u){let p=base+umCorner(v,5u);if(remapWrites(p,tile)){textureStore(phi,vec3i(p),vec4f(tileVertices[v]));}}
 }else if(lane==0u){
   var value=0.0;
   for(var z=0u;z<4u;z++){for(var y=0u;y<4u;y++){for(var x=0u;x<4u;x++){value+=tileVolume[x+4u*(y+4u*z)];}}}
   textureStore(volume,vec3i(base),vec4f(value/64.0));
 }
}
@compute @workgroup_size(64) fn remapCells(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let jobs=listedJobCount(lane,0u);for(var job=group.x;job<jobs;job+=groups.x){remapTileCells(job,lane);workgroupBarrier();}
}
`;
}
// Faces on the ungraded h/4h lattice, addressed from tile widths alone: a
// patch is as wide as the narrower of its two owners, anchored in the lower
// one's plane cells (umFace), so the tile, the tile below and the tile above
// on the axis locate every old and new patch. One lane per cell and axis
// evaluates the positive patch anchored at that cell (each texel has one
// writer) plus its owner's negative wall patch; the cell lane packs the
// texel. A patch's footprint cells are the tile cells on its plane: each
// such lane samples its own cell (and its old released bit), and the anchor
// lane sums them in footprint order. No footprint is walked serially.
function uniformMixedRemapFacesWGSL(layout:UniformMixedLayout):string{
 return uniformMixedRemapCommonWGSL(layout)+/* wgsl */`
@group(2) @binding(0) var velocity:texture_3d<f32>;
@group(2) @binding(1) var<storage,read> negative:array<f32>;
@group(2) @binding(2) var output:texture_storage_3d<rgba32float,write>;
@group(2) @binding(3) var<storage,read_write> boundary:array<f32>;
@group(2) @binding(4) var<storage,read> worklist:array<u32>;
fn umListLoad(i:u32)->u32{return worklist[i];}
${uniformMixedFaceAddressWGSL}
// Tile widths, 0 outside the lattice.
fn oldWidthAt(t:vec3i)->u32{if(any(t<vec3i(0))||any(t>=vec3i(UM_T))){return 0u;}return oldumTileWidth(umTileAt(vec3u(t)));}
fn widthAt(t:vec3i)->u32{if(any(t<vec3i(0))||any(t>=vec3i(UM_T))){return 0u;}return umTileWidth(umTileAt(vec3u(t)));}
// An old patch value at its anchor; a negative domain wall lives in the buffer.
fn oldFace(anchor:vec3i,axis:u32)->f32{
 if(anchor[axis]<0){return negative[umNegativeBoundaryIndex(vec3u(max(anchor,vec3i(0))),axis)];}
 return textureLoad(velocity,anchor,0)[axis];
}
// The tile's old owner at cell p (width w0, the widths nb below and ab above
// on axis): its faces on the axis, interpolated at plane p[axis] = plane.
fn remapSample(p:vec3i,plane:i32,axis:u32,base:vec3i,w0:u32,nb:u32,ab:u32)->f32{
 let u=(axis+1u)%3u;let v=(axis+2u)%3u;
 let origin=select(base[axis],p[axis],w0==1u);
 let t=f32(plane+1-origin)/f32(w0);
 var low=p;low[axis]=origin-1;if(w0==4u&&nb!=1u){low[u]=base[u];low[v]=base[v];}
 var high=p;high[axis]=origin+i32(w0)-1;if(w0==4u&&ab!=1u){high[u]=base[u];high[v]=base[v];}
 return mix(oldFace(low,axis),oldFace(high,axis),t);
}
// A footprint cell's old released bit (its old owner's positive anchor).
fn oldReleased(p:vec3i,axis:u32,bit:u32,base:vec3i,w0:u32)->u32{
 var anchor=p;if(w0==4u){anchor=base;anchor[axis]+=3;}
 return (u32(round(textureLoad(velocity,anchor,0).w))>>bit)&1u;
}
var<workgroup> tileFaces:array<vec2f,192>;
var<workgroup> tileSamples:array<f32,192>;
// The negative domain wall's samples (plane -1) and, per cell, its owner's
// old released bits: 1 the positive domain wall, 2 the negative one.
var<workgroup> tileWalls:array<f32,192>;
var<workgroup> tileReleased:array<u32,192>;
fn footprintSum(walls:bool,q:vec3u,w:u32,axis:u32)->f32{
 let u=(axis+1u)%3u;let v=(axis+2u)%3u;var value=0.0;
 for(var y=0u;y<w;y++){for(var x=0u;x<w;x++){var c=q;c[u]+=x;c[v]+=y;let i=c.x+4u*(c.y+4u*c.z)+64u*axis;value+=select(tileSamples[i],tileWalls[i],walls);}}
 return value/f32(w*w);
}
// A coarse wall patch is released only when its entire old footprint was
// released. Refinement copies the parent's classification. An OR would turn
// a partially attached patch into a completely separated wall contact.
fn footprintReleased(q:vec3u,w:u32,axis:u32,flag:u32)->bool{
 let u=(axis+1u)%3u;let v=(axis+2u)%3u;var acc=flag;
 for(var y=0u;y<w;y++){for(var x=0u;x<w;x++){var c=q;c[u]+=x;c[v]+=y;acc&=tileReleased[c.x+4u*(c.y+4u*c.z)+64u*axis];}}
 return acc!=0u;
}
fn remapTileFaces(job:u32,lane:u32,remap:bool){
 let tile=listed(job,lane,1u);
 let cellLane=lane%64u;let axis=lane/64u;let u=(axis+1u)%3u;let v=(axis+2u)%3u;
 let tc=vec3i(umTileCoord(tile));let base=tc*4;let local=umCorner(cellLane,4u);let p=base+vec3i(local);
 var below=tc;below[axis]-=1;var above=tc;above[axis]+=1;
 let w1=umTileWidth(tile);let w0=oldumTileWidth(tile);let nb=oldWidthAt(below);let ab=oldWidthAt(above);let next=widthAt(above);
 // The new owner of p, its positive patch width, and whether one of its
 // patches is anchored at p.
 let ownerLocal=select(vec3u(0),local,w1==1u);let origin=base+vec3i(ownerLocal);
 let width=select(w1,1u,next==1u);
 let anchored=w1==1u||(local[axis]==3u&&local[u]%width==0u&&local[v]%width==0u);
 let plane=origin[axis]+i32(w1)-1;
 // Domain walls: the owner's low plane at 0, the tile's last plane below none.
 let wall=origin[axis]==0;let top=next==0u&&local[axis]==3u;
 if(remap){
  var released=0u;
  if(p[axis]==plane){tileSamples[lane]=remapSample(p,plane,axis,base,w0,nb,ab);if(top){released|=oldReleased(p,axis,axis,base,w0);}}
  if(wall&&p[axis]==0){tileWalls[lane]=remapSample(p,-1,axis,base,w0,nb,ab);released|=oldReleased(p,axis,axis+3u,base,w0)<<1u;}
  tileReleased[lane]=released;
 }
 workgroupBarrier();
 if(wall&&all(p==origin)){
  let index=umNegativeBoundaryIndex(vec3u(origin),axis);
  var value=0.0;if(remap){value=footprintSum(true,ownerLocal,w1,axis);}else{value=negative[index];}
  boundary[index]=value;
 }
 var bits=0u;var value=0.0;
 if(anchored){
  bits=1u;
  if(remap){
   value=footprintSum(false,local,width,axis);
   if(top&&footprintReleased(local,width,axis,1u)){bits|=2u<<axis;}
  }else{value=textureLoad(velocity,p,0)[axis];}
 }
 if(remap&&wall&&footprintReleased(ownerLocal,w1,axis,2u)){bits|=2u<<(axis+3u);}
 tileFaces[lane]=vec2f(value,f32(bits));
 workgroupBarrier();
 if(axis!=0u){return;}
 var packed=vec4f(0);var written=false;var released=0u;
 for(var a=0u;a<3u;a++){
  let entry=tileFaces[cellLane+64u*a];let flags=u32(entry.y);
  if((flags&1u)!=0u){packed[a]=entry.x;written=true;}
  released|=flags>>1u;
 }
 if(!written){return;}
 if(remap){packed.w=f32(released);}else{packed.w=textureLoad(velocity,p,0).w;}
 textureStore(output,p,packed);
}
@compute @workgroup_size(192) fn remapFaces(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let jobs=listedJobCount(lane,1u);for(var job=group.x;job<jobs;job+=groups.x){remapTileFaces(job,lane,true);workgroupBarrier();}
}
@compute @workgroup_size(192) fn copyFaces(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let jobs=listedJobCount(lane,1u);for(var job=group.x;job<jobs;job+=groups.x){remapTileFaces(job,lane,false);workgroupBarrier();}
}
`;
}
