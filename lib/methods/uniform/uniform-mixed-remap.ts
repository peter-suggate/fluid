import {UniformMixedOwnership,type UniformMixedBuiltOwnership} from "./uniform-mixed-ownership";
import type {UniformMixedLayout} from "./uniform-mixed-layout";
import {uniformMixedTopologyWGSL} from "./uniform-mixed-topology.wgsl";
import {uniformMixedVertexSamplingSource} from "./uniform-mixed-vertex-sampling.wgsl";
import {uniformMixedFaceAddressWGSL} from "./uniform-mixed-face-dispatch.wgsl";
import {uniformVolumeTargetWGSL} from "./uniform-volume.wgsl";
import {geometricPlaneBoxWGSL} from "../../core/geometric-plane-box.wgsl";

interface Fields{volume:GPUTexture;velocity:GPUTexture;phi:GPUTexture;negative:GPUBuffer}
/** Conservative edit-boundary remap, followed by canonical copy back into the
 * persistent fields. The temporary ownership and fields are reused for every
 * edit. No fine-grid expansion or field downloads are involved. Only tiles
 * within one tile of a width change are visited: their owners' faces, cells
 * and authoritative vertices are the only ones the edit can alter; every
 * other owner's remap is the identity, so its live value stays in place. */
export class UniformMixedRemap {
 readonly target:UniformMixedOwnership;
 get allocatedBytes(){return this.target.allocatedBytes+this.worklist.size;}
 private readonly resources:GPUBindGroupLayout;
 private readonly groups:readonly [GPUBindGroup,GPUBindGroup];
 /** The census extension (scratch velocity and negative walls) remapped
  * through its own fields, before the velocity remap reuses the scratch. */
 private extensionGroups?:readonly [GPUBindGroup,GPUBindGroup];
 private readonly pipelines=new Map<string,GPUComputePipeline>();
 /** [three unused words, count, tiles...]: the dilated changed-tile worklist. */
 private readonly worklist:GPUBuffer;
 /** Workgroups of each remap launch: a fixed grid strides over the listed
  * tiles, bounded by the layout's tiles and capped where the GPU saturates. */
 private readonly grid:number;
 constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,input:Fields,scratch:Fields){
  this.target=new UniformMixedOwnership(device,ownership.layout,false);
  this.worklist=device.createBuffer({label:"Uniform mixed remap worklist",size:(4+ownership.layout.tiles.length)*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  this.grid=Math.min(1024,ownership.layout.tiles.length);
  this.resources=device.createBindGroupLayout({entries:[
   ...[0,1,2].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
   {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
   ...[4,5,6].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only" as const,format:(binding===5?"rgba32float":"r32float") as GPUTextureFormat,viewDimension:"3d" as const}})),
   {binding:7,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
   {binding:8,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
  ]});
  const bind=(a:Fields,b:Fields)=>device.createBindGroup({layout:this.resources,entries:[
   ...[a.volume,a.velocity,a.phi].map((t,binding)=>({binding,resource:t.createView()})),{binding:3,resource:{buffer:a.negative}},
   ...[b.volume,b.velocity,b.phi].map((t,i)=>({binding:4+i,resource:t.createView()})),{binding:7,resource:{buffer:b.negative}},
   {binding:8,resource:{buffer:this.worklist}},
  ]});
  this.groups=[bind(input,scratch),bind(scratch,input)];
  this.bind=bind;this.scratchFields=scratch;
 }
 private readonly bind:(a:Fields,b:Fields)=>GPUBindGroup;
 private readonly scratchFields:Fields;
 /** Extension fields: the scratch velocity and negative walls are remapped
  * into these and copied back after the live ownership adopts. */
 bindExtension(input:Fields,extension:{velocity:GPUTexture;negative:GPUBuffer}):void{
  const s=this.scratchFields,remapped={volume:s.volume,velocity:extension.velocity,phi:s.phi,negative:extension.negative};
  this.extensionGroups=[this.bind({volume:input.volume,velocity:s.velocity,phi:input.phi,negative:s.negative},remapped),
   this.bind(remapped,{volume:input.volume,velocity:s.velocity,phi:input.phi,negative:s.negative})];
 }
 async initialize():Promise<void>{
  const module=this.device.createShaderModule({code:uniformMixedRemapWGSL(this.ownership.layout)});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.target.bindLayout,this.resources]});
  for(const entryPoint of ["markChanged","remapCells","remapFaces","copyCells","copyFaces"])this.pipelines.set(entryPoint,await this.device.createComputePipelineAsync({layout,compute:{module,entryPoint,constants:{umDispatchX:this.ownership.dispatchX}}}));
 }
 apply(layout:UniformMixedLayout):void{
  if(this.pipelines.size!==5)throw new Error("Live remap has not been initialized");
  this.target.update(layout);
  const encode=(copy:boolean)=>{
   const e=this.device.createCommandEncoder({label:copy?"Uniform publish remapped owners":"Uniform remap changed ownership"});
   this.encodePass(e,copy);this.device.queue.submit([e.finish()]);
  };
  encode(false);this.ownership.update(layout);encode(true);
 }
 /** The same remap for a GPU-built generation, in one encoder: adopt into
  * the target, remap, adopt into the live ownership, publish. With
  * extension, the scratch velocity (the census extension) is remapped too. */
 applyBuilt(encoder:GPUCommandEncoder,built:UniformMixedBuiltOwnership,extension=false):void{
  if(this.pipelines.size!==5)throw new Error("Live remap has not been initialized");
  if(extension&&!this.extensionGroups)throw new Error("Remap has no extension fields");
  this.target.adopt(encoder,built);this.encodePass(encoder,false,extension);
  this.ownership.adopt(encoder,built);this.encodePass(encoder,true,extension);
 }
 private encodePass(e:GPUCommandEncoder,copy:boolean,extension=false):void{
  // The list compares the live (old) and target widths, so it is built
  // before the live ownership adopts the target and reused by the publish.
  const begin=(label:string)=>{
   const pass=e.beginComputePass({label});pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.target.bindGroup);pass.setBindGroup(2,this.groups[copy?1:0]);return pass;
  };
  if(!copy){
   e.clearBuffer(this.worklist,0,16);
   const list=begin("Uniform mixed remap worklist"),groups=Math.ceil(this.target.layout.tiles.length/64);
   list.setPipeline(this.pipelines.get("markChanged")!);list.dispatchWorkgroups(Math.min(groups,this.target.dispatchX),Math.ceil(groups/this.target.dispatchX));
   list.end();
  }
  // The extension leaves the scratch velocity before the velocity remap
  // writes it, and returns after the velocity publish read it.
  const faces=(label:string,group:GPUBindGroup,name:string)=>{const pass=begin(label);pass.setBindGroup(2,group);pass.setPipeline(this.pipelines.get(name)!);pass.dispatchWorkgroups(this.grid);pass.end();};
  if(extension&&!copy)faces("Uniform mixed remap extension",this.extensionGroups![0],"remapFaces");
  const pass=begin(copy?"Uniform mixed remap publish":"Uniform mixed remap");
  for(const name of copy?["copyCells","copyFaces"]:["remapCells","remapFaces"]){pass.setPipeline(this.pipelines.get(name)!);pass.dispatchWorkgroups(this.grid);}
  pass.end();
  if(extension&&copy)faces("Uniform mixed remap extension publish",this.extensionGroups![1],"copyFaces");
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
function uniformMixedRemapWGSL(layout:UniformMixedLayout):string{
 const vertexSampling=uniformMixedVertexSamplingSource("",false);
 const old=(s:string)=>s.replace(/\b(?:um[A-Z]\w*|UM_\w+|UMOwner|UMFace)\b/g,n=>"old"+n);
 return uniformMixedTopologyWGSL(layout,0,"old")+uniformMixedTopologyWGSL(layout,1)+/* wgsl */`
@group(2) @binding(0) var volume:texture_3d<f32>;
@group(2) @binding(1) var velocity:texture_3d<f32>;
@group(2) @binding(2) var phi:texture_3d<f32>;
@group(2) @binding(3) var<storage,read> negative:array<f32>;
@group(2) @binding(4) var outputVolume:texture_storage_3d<r32float,write>;
@group(2) @binding(5) var output:texture_storage_3d<rgba32float,write>;
@group(2) @binding(6) var outputPhi:texture_storage_3d<r32float,write>;
@group(2) @binding(7) var<storage,read_write> boundary:array<f32>;
@group(2) @binding(8) var<storage,read_write> worklist:array<atomic<u32>>;
// A tile is visited when any tile of its 3x3x3 neighbourhood changed width.
// A tile that is h in both keeps its cells and face patches (seam patches are
// anchored in its cells), but when a neighbour refines it may inherit
// authority over a shared vertex (equal widths tie-break by owner,
// umVertexAuthority) that the old layout derived from the coarse corners and
// never stored: it must write that vertex, or the stale texel becomes the
// surface. A coarsening neighbour takes its shared vertices over instead.
@compute @workgroup_size(64) fn markChanged(@builtin(global_invocation_id) gid:vec3u){
 let tile=gid.x+umDispatchX*64u*gid.y;if(tile>=UM_TILES){return;}
 let fine=umTileWidth(tile)==1u&&oldumTileWidth(tile)==1u;
 let p=vec3i(umTileCoord(tile));
 for(var z=max(p.z-1,0);z<=min(p.z+1,i32(UM_T.z)-1);z++){for(var y=max(p.y-1,0);y<=min(p.y+1,i32(UM_T.y)-1);y++){for(var x=max(p.x-1,0);x<=min(p.x+1,i32(UM_T.x)-1);x++){
  let q=umTileAt(vec3u(vec3i(x,y,z)));
  if(select(umTileWidth(q)!=oldumTileWidth(q),umTileWidth(q)<oldumTileWidth(q),fine)){atomicStore(&worklist[4u+atomicAdd(&worklist[3],1u)],tile);return;}
 }}}
}
// One workgroup job per listed tile; a fixed grid strides over the list. Lanes are the tile's cells (and, for faces,
// cell and axis), so a coarse owner's footprint loops run side by side
// instead of serially on one lane.
const UM_UNLISTED:u32=0xffffffffu;
var<workgroup> listedTile:u32;
fn listed(group:vec3u,lane:u32)->u32 {
 if(lane==0u){
  let slot=group.x;
  listedTile=select(UM_UNLISTED,atomicLoad(&worklist[4u+slot]),slot<atomicLoad(&worklist[3]));
 }
 return workgroupUniformLoad(&listedTile);
}
var<workgroup> listedJobs:u32;
fn listedJobCount(lane:u32)->u32 {
 if(lane==0u){listedJobs=atomicLoad(&worklist[3]);}
 return workgroupUniformLoad(&listedJobs);
}
fn oldumLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p),0).x;}
${old(vertexSampling)}
fn umLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p),0).x;}
${vertexSampling}
${uniformMixedFaceAddressWGSL}
fn oldFaceValue(f:oldUMFace)->f32{
 if(f.anchor[f.axis]<0){return negative[umNegativeBoundaryIndex(vec3u(max(f.anchor,vec3i(0))),f.axis)];}
 return textureLoad(velocity,f.anchor,0)[f.axis];
}
fn oldPatch(o:oldUMOwner,p:vec3u,axis:u32,sign:i32)->oldUMFace{
 let first=oldumFace(o,axis,sign,0u);let local=p-oldumOrigin(o);let u=(axis+1u)%3u;let v=(axis+2u)%3u;
 return oldumFace(o,axis,sign,local[u]/first.width+(o.width/first.width)*(local[v]/first.width));
}
// One footprint cell of a face on plane p[axis] (anchor[axis] = plane).
fn remapSample(p:vec3u,plane:i32,axis:u32)->f32{
 let o=oldumOwnerAt(vec3i(p));let origin=oldumOrigin(o);
 let t=f32(plane+1-i32(origin[axis]))/f32(o.width);
 return mix(oldFaceValue(oldPatch(o,p,axis,-1)),oldFaceValue(oldPatch(o,p,axis,1)),t);
}
fn remapFace(f:UMFace)->f32{
 let axis=f.axis;let u=(axis+1u)%3u;let v=(axis+2u)%3u;var value=0.0;
 for(var y=0u;y<f.width;y++){for(var x=0u;x<f.width;x++){
  var p=vec3u(max(f.anchor,vec3i(0)));p[u]+=x;p[v]+=y;value+=remapSample(p,f.anchor[axis],axis);
 }}return value/f32(f.width*f.width);
}
// A coarse wall patch is released only when its entire old footprint was
// released. Refinement copies the parent's classification. An OR would turn
// a partially attached patch into a completely separated wall contact.
fn remapReleased(f:UMFace)->bool {
 if(f.neighbor.width!=0u){return false;}
 let axis=f.axis;let u=(axis+1u)%3u;let v=(axis+2u)%3u;
 let bit=axis+select(0u,3u,f.sign<0);
 for(var y=0u;y<f.width;y++){for(var x=0u;x<f.width;x++){
  var p=vec3u(max(f.anchor,vec3i(0)));p[u]+=x;p[v]+=y;
  let o=oldumOwnerAt(vec3i(p));var anchor=oldumOrigin(o);
  anchor[axis]+=o.width-1u;
  if((u32(round(textureLoad(velocity,vec3i(anchor),0).w))&(1u<<bit))==0u){return false;}
 }}return true;
}
fn copyFace(f:UMFace)->f32{
 if(f.anchor[f.axis]<0){return negative[umNegativeBoundaryIndex(vec3u(max(f.anchor,vec3i(0))),f.axis)];}
 return textureLoad(velocity,f.anchor,0)[f.axis];
}
${geometricPlaneBoxWGSL}
fn uvCorner(k:u32)->vec3i{return vec3i(umCorner(k,2u));}
fn d4Sum8(v:array<f32,8>)->f32{return ((v[0]+v[5])+(v[1]+v[4]))+((v[2]+v[7])+(v[3]+v[6]));}
${uniformVolumeTargetWGSL(true,true)}
var<workgroup> tileVolume:array<f32,64>;
var<workgroup> tileFill:array<f32,64>;
// The old layout's phi at the tile's 5x5x5 closure: every new owner corner
// and every vertex this tile writes, each sampled once.
var<workgroup> tileVertices:array<f32,125>;
fn tileVertex(q:vec3u)->f32{return tileVertices[q.x+5u*(q.y+5u*q.z)];}
// Cells: every cell loads its donor's volume; each new owner's origin lane
// averages its cells. Vertices: each lattice vertex of the tile's closure is
// written by the tile holding its authority (the widest incident owner).
// A donor split into narrower owners does not broadcast its volume: its
// cells take the geometric fill of their new owner (the same target the
// solver compares against, from the remapped phi), scaled so the donor's
// volume is conserved exactly. Filled fractions shrink toward zero when
// the donor holds less than its phi implies, empty fractions toward zero
// when it holds more (up to full), so no fraction leaves [0,1]. Re-coarsening
// averages the cells back to the donor's volume, and its phi corners were
// never altered, so coarse -> fine -> coarse is the identity.
fn remapTileCells(group:vec3u,lane:u32,remap:bool){
 let tile=listed(group,lane);if(tile==UM_UNLISTED){return;}
 let base=umTileCoord(tile)*4u;let cell=base+umCorner(lane,4u);
 let o=umOwnerAt(vec3i(cell));
 var split=false;var donor:oldUMOwner;
 if(remap){
  for(var v=lane;v<125u;v+=64u){tileVertices[v]=oldumSampleVertex(vec3f(base+umCorner(v,5u)));}
  donor=oldumOwnerAt(vec3i(cell));
  tileVolume[lane]=textureLoad(volume,vec3i(oldumOrigin(donor)),0).x;
  split=o.width<donor.width;
 }
 workgroupBarrier();
 // Tile widths are uniform: a new owner's cells share its origin lane's fill.
 let ownerLocal=(umCorner(lane,4u)/o.width)*o.width;
 if(split&&all(cell==umOrigin(o))){
  var vertices:array<f32,8>;
  for(var j=0u;j<8u;j++){vertices[j]=tileVertex(ownerLocal+umCorner(j,2u)*o.width);}
  tileFill[lane]=umSurfaceTarget(o,vertices);
 }
 workgroupBarrier();
 // A split donor is wider than one cell, so it lies inside this tile.
 if(split){
  let local=oldumOrigin(donor)-base;let n=donor.width*donor.width*donor.width;var fill=0.0;
  for(var z=0u;z<donor.width;z++){for(var y=0u;y<donor.width;y++){for(var x=0u;x<donor.width;x++){
   let q=((local+vec3u(x,y,z))/o.width)*o.width;fill+=tileFill[q.x+4u*(q.y+4u*q.z)];
  }}}
  // A donor whose phi is (nearly) all empty or all full has no shape to
  // follow, and an overfull donor (V > 1) keeps its excess uniform.
  let held=tileVolume[lane]*f32(n);let f=tileFill[ownerLocal.x+4u*(ownerLocal.y+4u*ownerLocal.z)];let room=f32(n)-fill;
  if(held<=fill&&fill>=0.5){tileVolume[lane]=f*held/fill;}
  else if(held>fill&&held<=f32(n)&&room>=0.5){tileVolume[lane]=1.0-(1.0-f)*(f32(n)-held)/room;}
 }
 workgroupBarrier();
 if(all(cell==umOrigin(o))){
  var value=0.0;
  if(remap){
   let local=umCorner(lane,4u);
   for(var z=0u;z<o.width;z++){for(var y=0u;y<o.width;y++){for(var x=0u;x<o.width;x++){
    let q=local+vec3u(x,y,z);value+=tileVolume[q.x+4u*(q.y+4u*q.z)];
   }}}
   value/=f32(o.width*o.width*o.width);
  }else{value=textureLoad(volume,vec3i(cell),0).x;}
  textureStore(outputVolume,vec3i(cell),vec4f(value));
 }
 for(var v=lane;v<125u;v+=64u){
  let p=base+umCorner(v,5u);let a=umVertexAuthority(p);
  if(a.width!=0u&&a.tile==tile&&umVertexIsCanonical(p,a)){
   var value=0.0;if(remap){value=tileVertices[v];}else{value=textureLoad(phi,vec3i(p),0).x;}
   textureStore(outputPhi,vec3i(p),vec4f(value));
  }
 }
}
@compute @workgroup_size(64) fn remapCells(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let jobs=listedJobCount(lane);for(var job=group.x;job<jobs;job+=groups.x){remapTileCells(vec3u(job,0u,0u),lane,true);workgroupBarrier();}
}
@compute @workgroup_size(64) fn copyCells(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let jobs=listedJobCount(lane);for(var job=group.x;job<jobs;job+=groups.x){remapTileCells(vec3u(job,0u,0u),lane,false);workgroupBarrier();}
}
var<workgroup> tileFaces:array<vec2f,192>;
var<workgroup> tileSamples:array<f32,192>;
// Faces: one lane per cell and axis evaluates the positive patch anchored at
// that cell (anchors lie inside their owner, so each texel has one writer)
// plus its owner's negative wall patch. The cell lane packs the texel.
// A positive patch's footprint cells are the tile cells on its plane: each
// such lane samples its own cell, and the anchor lane sums them in
// remapFace's order. A coarse patch no longer walks its footprint serially.
fn remapTileFaces(group:vec3u,lane:u32,remap:bool){
 let tile=listed(group,lane);if(tile==UM_UNLISTED){return;}
 let cellLane=lane%64u;let axis=lane/64u;
 let anchor=vec3i(umTileCoord(tile)*4u+umCorner(cellLane,4u));
 let owner=umOwnerAt(anchor);let origin=umOrigin(owner);
 if(origin[axis]==0u&&all(vec3u(anchor)==origin)){
  let face=umFace(owner,axis,-1,0u);
  var value=0.0;if(remap){value=remapFace(face);}else{value=copyFace(face);}
  boundary[umNegativeBoundaryIndex(origin,axis)]=value;
 }
 let face=umPositiveFaceAtAnchor(owner,axis,anchor);
 let plane=i32(origin[axis]+owner.width)-1;
 if(remap&&anchor[axis]==plane){tileSamples[lane]=remapSample(vec3u(anchor),plane,axis);}
 workgroupBarrier();
 var bits=0u;var value=0.0;
 if(face.width!=0u){
  bits=1u;
  if(remap){
   let u=(axis+1u)%3u;let v=(axis+2u)%3u;let local=umCorner(cellLane,4u);
   for(var y=0u;y<face.width;y++){for(var x=0u;x<face.width;x++){
    var q=local;q[u]+=x;q[v]+=y;value+=tileSamples[q.x+4u*(q.y+4u*q.z)+64u*axis];
   }}
   value/=f32(face.width*face.width);
   if(remapReleased(face)){bits|=2u<<axis;}
  }else{value=copyFace(face);}
 }
 if(remap&&origin[axis]==0u&&remapReleased(umFace(owner,axis,-1,0u))){bits|=2u<<(axis+3u);}
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
 if(remap){packed.w=f32(released);}else{packed.w=textureLoad(velocity,anchor,0).w;}
 textureStore(output,anchor,packed);
}
@compute @workgroup_size(192) fn remapFaces(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let jobs=listedJobCount(lane);for(var job=group.x;job<jobs;job+=groups.x){remapTileFaces(vec3u(job,0u,0u),lane,true);workgroupBarrier();}
}
@compute @workgroup_size(192) fn copyFaces(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let jobs=listedJobCount(lane);for(var job=group.x;job<jobs;job+=groups.x){remapTileFaces(vec3u(job,0u,0u),lane,false);workgroupBarrier();}
}

`;
}
