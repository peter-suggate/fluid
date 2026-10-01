import {UniformMixedOwnership,type UniformMixedGenerationBuffers} from "./uniform-mixed-ownership";
import type {UniformMixedLayout} from "./uniform-mixed-layout";
import {UNIFORM_MIXED_COUNTED,UNIFORM_MIXED_OVERFLOW_HANGING,uniformMixedCountedEntriesWGSL,uniformMixedPageCount,uniformMixedResidencyWord,uniformMixedTopologyWGSL} from "./uniform-mixed-topology.wgsl";
import {uniformMixedVertexSamplingSource} from "./uniform-mixed-vertex-sampling.wgsl";
import {uniformMixedFaceAddressWGSL} from "./uniform-mixed-face-dispatch.wgsl";
import {UNIFORM_MIXED_FAILURE,UNIFORM_MIXED_STATUS,UNIFORM_MIXED_STATUS_WORDS,uniformMixedFrameStatusWGSL} from "./uniform-mixed-frame-status";
import {uniformVolumeTargetWGSL} from "./uniform-volume.wgsl";
import {geometricPlaneBoxWGSL} from "../../core/geometric-plane-box.wgsl";
import {UNIFORM_MIXED_RELAYOUT_RECEIPT,uniformMixedChangedTilesWGSL} from "./uniform-mixed-layout-builder";

interface Fields{volume:GPUTexture;velocity:GPUTexture;phi:GPUTexture;negative:GPUBuffer}
/** Workgroups of each remap launch (see applyGpu). */
const REMAP_GRID=4096;
/** Conservative edit-boundary remap. Volume and phi are remapped in place on
 * the persistent fields; faces go through the scratch velocity and are copied
 * back. The target generation is bound as group 1: a GPU-built generation's
 * own buffers, or a transient ownership for a host layout. No fine-grid
 * expansion or field downloads are involved. Only tiles
 * within one tile of a width change are visited: their owners' faces, cells
 * and authoritative vertices are the only ones the edit can alter; every
 * other owner's remap is the identity, so its live value stays in place. */
export class UniformMixedRemap {
 get allocatedBytes(){return this.worklist.size;}
 /** Group 1 over a GPU-built generation's buffers, bound per topology buffer. */
 private targetGroup?:{readonly topology:GPUBuffer;readonly group:GPUBindGroup};
 private readonly resources:GPUBindGroupLayout;
 private readonly groups:readonly [GPUBindGroup,GPUBindGroup];
 /** The in-place cells remap: the live volume and phi read_write, and the
  * worklist. */
 private readonly cellsResources:GPUBindGroupLayout;
 private readonly cellsGroup:GPUBindGroup;
 /** The census extension (scratch velocity and negative walls) remapped
  * through its own fields, before the velocity remap reuses the scratch. */
 private extensionGroups?:readonly [GPUBindGroup,GPUBindGroup];
 private readonly pipelines=new Map<string,GPUComputePipeline>();
 /** [unused x3, count, tiles...]: the changed-tile worklist. */
 private readonly worklist:GPUBuffer;
 /** Group 3: the frame status record (uniform-mixed-frame-status), bound
  * read_write. markChanged latches a builder fatal into it and lists nothing
  * once any failure is latched. A private zero record until bindStatus. */
 private readonly statusLayout:GPUBindGroupLayout;
 private readonly idleStatus:GPUBuffer;
 private statusGroup:GPUBindGroup;
 /** Tiles in the lattice: the worklist's capacity. */
 private readonly tiles:number;
 /** markListed's group 2: the worklist, a generation's changed tiles
  * (UniformMixedGenerationBuffers.changes) and its builder's relayout
  * receipt, bound per changes buffer. */
 private readonly listResources:GPUBindGroupLayout;
 private listGroup?:{readonly changes:GPUBuffer;readonly receipt:GPUBuffer;readonly group:GPUBindGroup};
 constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,input:Fields,scratch:Fields){
  this.worklist=device.createBuffer({label:"Uniform mixed remap worklist",size:(4+ownership.capacity.tiles)*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  this.tiles=ownership.capacity.tiles;
  if(this.tiles>0x08000000)throw new Error("Remap worklist entries pack a tile index below 2^27");
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
  this.listResources=device.createBindGroupLayout({entries:[
   {binding:8,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:9,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
   {binding:10,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
  ]});
  this.cellsResources=device.createBindGroupLayout({entries:[
   ...[0,1].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"read-write" as const,format:"r32float" as const,viewDimension:"3d" as const}})),
   {binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
  ]});
  this.cellsGroup=device.createBindGroup({layout:this.cellsResources,entries:[
   {binding:0,resource:input.volume.createView()},{binding:1,resource:input.phi.createView()},{binding:2,resource:{buffer:this.worklist}},
  ]});
  this.bind=bind;this.scratchFields=scratch;
  this.statusLayout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
  this.idleStatus=device.createBuffer({label:"Uniform mixed remap idle status",size:UNIFORM_MIXED_STATUS_WORDS*4,usage:GPUBufferUsage.STORAGE});
  this.statusGroup=this.bindStatusGroup(this.idleStatus);
 }
 private bindStatusGroup(status:GPUBuffer):GPUBindGroup{return this.device.createBindGroup({layout:this.statusLayout,entries:[{binding:0,resource:{buffer:status,size:UNIFORM_MIXED_STATUS_WORDS*4}}]});}
 /** The frame's status record (UniformMixedFrame.status). */
 bindStatus(status:GPUBuffer):void{this.statusGroup=this.bindStatusGroup(status);}
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
  // One module per part: the face and list entries share the faces module.
  const modules=new Map<"faces"|"cells",Promise<GPUShaderModule>>(),moduleOf=(part:"faces"|"cells")=>{
   let module=modules.get(part);
   if(!module)modules.set(part,module=(async()=>{const m=this.device.createShaderModule({code:uniformMixedRemapWGSL(this.ownership.layout,part)});
    const errors=(await m.getCompilationInfo()).messages.filter(e=>e.type==="error");if(errors.length)throw new Error(errors.map(e=>`${e.lineNum}: ${e.message}`).join("\n"));return m;})());
   return module;
  };
  const compile=async(part:"faces"|"cells",resources:GPUBindGroupLayout,entryPoints:readonly string[])=>{
   const module=await moduleOf(part);
   const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.ownership.bindLayout,resources,...(part==="faces"?[this.statusLayout]:[])]});
   await Promise.all(entryPoints.map(async entryPoint=>{this.pipelines.set(entryPoint,await this.device.createComputePipelineAsync({layout,compute:{module,entryPoint,constants:{umDispatchX:this.ownership.dispatchX}}}));}));
  };
  await Promise.all([compile("faces",this.resources,["markChanged","remapFaces","copyFaces"]),compile("faces",this.listResources,["markListed"]),compile("cells",this.cellsResources,["remapCells"])]);
 }
 apply(layout:UniformMixedLayout):void{
  if(this.pipelines.size!==5)throw new Error("Live remap has not been initialized");
  // A transient target: host layouts are rare (authored regions, the initial layout).
  const target=new UniformMixedOwnership(this.device,layout,false);
  const encode=(copy:boolean)=>{
   const e=this.device.createCommandEncoder({label:copy?"Uniform publish remapped owners":"Uniform remap changed ownership"});
   if(!copy)e.clearBuffer(this.worklist,0,16);
   this.encodePass(e,copy,target.bindGroup);this.device.queue.submit([e.finish()]);
  };
  encode(false);this.ownership.update(layout);encode(true);target.destroy();
 }
 /** The same remap for a GPU-built generation with no host object, in one
  * encoder at the head of the frame that runs on it: remap with the
  * builder's buffers bound as the target (its support words are bound but
  * never read), adopt them into the live ownership, publish. Encoded every frame; an unchanged generation lists no tile.
  * receipt: the builder's relayout receipt (UNIFORM_MIXED_RELAYOUT_RECEIPT),
  * read by markListed: a set fatal bit latches frame failure 6 (hanging
  * capacity) or 7 (tier sum or tile words: invalid support) and lists
  * nothing; a frame already failed lists nothing either. Its generation word
  * then becomes the frame status's current generation. encodeClear must
  * precede it in the encoder. The buffer copies are not gated: a failed frame
  * adopts the built words unremapped, and the host throws on its record.
  * With extension, the scratch velocity (the census extension) is remapped
  * too. The host does not know the changed count, so every launch is the
  * fixed REMAP_GRID striding the GPU list. The worklist is built from the
  * generation's dilated changed tiles (markListed), not a lattice scan.
  * adopted: encoded right after the live ownership adopts (in that blit
  * run, between the remap and publish passes). */
 applyGpu(encoder:GPUCommandEncoder,source:UniformMixedGenerationBuffers,receipt:{readonly buffer:GPUBuffer;readonly offset:number},extension=false,adopted?:(encoder:GPUCommandEncoder)=>void):void{
  if(this.pipelines.size!==5)throw new Error("Live remap has not been initialized");
  if(extension&&!this.extensionGroups)throw new Error("Remap has no extension fields");
  if(!this.cleared)throw new Error("Live remap encoded without encodeClear");
  this.cleared=false;
  if(receipt.offset%256)throw new Error("Remap binds the relayout receipt at a 256 B aligned offset");
  if(this.listGroup?.changes!==source.changes||this.listGroup.receipt!==receipt.buffer)this.listGroup={changes:source.changes,receipt:receipt.buffer,group:this.device.createBindGroup({layout:this.listResources,entries:[
   {binding:8,resource:{buffer:this.worklist}},{binding:9,resource:{buffer:source.changes}},
   {binding:10,resource:{buffer:receipt.buffer,offset:receipt.offset,size:UNIFORM_MIXED_RELAYOUT_RECEIPT.words*4}}]})};
  if(this.targetGroup?.topology!==source.topology)this.targetGroup={topology:source.topology,group:this.device.createBindGroup({layout:this.ownership.bindLayout,entries:[
   {binding:0,resource:{buffer:source.topology}},{binding:1,resource:{buffer:source.counts.buffer,offset:source.counts.offset,size:16}},{binding:2,resource:{buffer:source.support}}]})};
  const target=this.targetGroup.group;
  this.encodePass(encoder,false,target,extension,Math.min(REMAP_GRID,this.tiles),this.listGroup.group);
  this.ownership.adoptGpu(encoder,source);adopted?.(encoder);this.encodePass(encoder,true,target,extension);
 }
 /** Zero the worklist header for the next applyGpu: encode it ahead of the
  * frame head's first pass, in the same encoder, so it joins that blit run. */
 encodeClear(encoder:GPUCommandEncoder):void{encoder.clearBuffer(this.worklist,0,16);this.cleared=true;}
 private cleared=false;
 /** The worklist header is cleared before (apply, encodeClear). listed: markListed's
  * group (a GPU-built generation's changed tiles); without it the worklist
  * scans the lattice (markChanged). Each call is one compute pass: every
  * launch reads what the one before it wrote. target: group 1, the
  * generation being adopted. */
 private encodePass(e:GPUCommandEncoder,copy:boolean,target:GPUBindGroup,extension=false,grid=Math.min(REMAP_GRID,this.tiles),listed?:GPUBindGroup):void{
  const x=this.ownership.dispatchX,dispatch=(pass:GPUComputePassEncoder)=>pass.dispatchWorkgroups(Math.min(grid,x),Math.ceil(grid/x));
  const pass=e.beginComputePass({label:copy?"Uniform mixed remap publish":"Uniform mixed remap"});
  pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,target);pass.setBindGroup(3,this.statusGroup);
  const faces=(group:GPUBindGroup,name:string)=>{pass.setBindGroup(2,group);pass.setPipeline(this.pipelines.get(name)!);dispatch(pass);};
  // The list compares the live (old) and target widths, so it is built
  // before the live ownership adopts the target and reused by the publish.
  // Either launch is one lane per tile, the list's bound.
  if(!copy){
   const groups=Math.ceil(this.tiles/64);
   pass.setBindGroup(2,listed??this.groups[0]);pass.setPipeline(this.pipelines.get(listed?"markListed":"markChanged")!);
   pass.dispatchWorkgroups(Math.min(groups,x),Math.ceil(groups/x));
  }
  // The extension leaves the scratch velocity before the velocity remap
  // writes it, and returns after the velocity publish read it.
  if(extension&&!copy)faces(this.extensionGroups![0],"remapFaces");
  // Cells run in place before the faces, which read neither volume nor phi.
  if(!copy)faces(this.cellsGroup,"remapCells");
  faces(this.groups[copy?1:0],copy?"copyFaces":"remapFaces");
  if(extension&&copy)faces(this.extensionGroups![1],"copyFaces");
  pass.end();
 }
 destroy():void{this.worklist.destroy();this.idleStatus.destroy();}
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
 *   patches takes its 4h face.
 * The 4h side runs over the resident pages only (uniformMixedResidencyWord):
 * an absent page is certified far air whose velocity nobody reads, so its
 * faces are neither transferred nor projected. encodeToPressure first copies
 * the simulation's residency words into the pressure ownership's support,
 * so the all-4h pressure stages stride the same resident pages. */
const TRANSFER_GRID=4096;
export class UniformMixedOwnershipTransfer {
 readonly allocatedBytes=0;
 private readonly resources:GPUBindGroupLayout;
 private readonly pipelines=new Map<string,GPUComputePipeline>();
 /** Fixed grids: lanes over the 4h list, workgroup jobs over the h list. */
 private readonly fineGroups:number;
 /** Resident pages bound the coarse launches (one job per page). */
 private readonly pages:number;
 constructor(private readonly device:GPUDevice,readonly simulation:UniformMixedOwnership,readonly pressure:UniformMixedOwnership){
  if(simulation.layout.tiles.length!==pressure.layout.tiles.length)throw new Error("Ownership transfer requires one tile lattice");
  const tiles=simulation.layout.tiles.length;
  this.fineGroups=Math.max(1,Math.min(TRANSFER_GRID,tiles));this.pages=uniformMixedPageCount(simulation.capacity.lattice);
  if(pressure.layout.tiles.some(word=>(word&0xc0000000)!==0))throw new Error("Ownership transfer targets the all-4h pressure ownership");
  this.resources=device.createBindGroupLayout({entries:[
   ...[0,1].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
   {binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
   ...[3,4].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only" as const,format:(binding===4?"rgba32float":"r32float") as GPUTextureFormat,viewDimension:"3d" as const}})),
   {binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
   {binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
  ]});
 }
 /** Fields read in the source ownership and written in the target's. The
  * volume pair is read and written by the transfer to pressure only. status
  * is the frame status record (uniform-mixed-frame-status): once a failure
  * is latched the transfer to simulation writes nothing, so a rejected
  * frame's unprojected field never reaches simulation ownership. */
 bind(input:{volume:GPUTexture;velocity:GPUTexture;negative:GPUBuffer},output:{volume:GPUTexture;velocity:GPUTexture;negative:GPUBuffer},status:GPUBuffer):GPUBindGroup{
  return this.device.createBindGroup({layout:this.resources,entries:[
   {binding:0,resource:input.volume.createView()},{binding:1,resource:input.velocity.createView()},{binding:2,resource:{buffer:input.negative}},
   {binding:3,resource:output.volume.createView()},{binding:4,resource:output.velocity.createView()},{binding:5,resource:{buffer:output.negative}},
   {binding:6,resource:{buffer:status,size:64}},
  ]});
 }
 async initialize():Promise<void>{
  const module=this.device.createShaderModule({label:"Uniform mixed ownership transfer",code:uniformMixedOwnershipTransferWGSL(this.simulation.layout)});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.simulation.bindLayout,this.resources]});
  await Promise.all(["toPressureCoarse","toPressureFine","toSimulationCoarse","toSimulationFine"].map(async entryPoint=>{this.pipelines.set(entryPoint,await this.device.createComputePipelineAsync({layout,compute:{module,entryPoint,
   constants:{umDispatchX:this.simulation.dispatchX,...(entryPoint.endsWith("Coarse")?{umCountedJobs:UNIFORM_MIXED_COUNTED.residentPages}:{})}}}));}));
 }
 /** Both launches write disjoint tiles and read only the source fields. */
 private encode(e:GPUCommandEncoder,entry:"toPressure"|"toSimulation",group:GPUBindGroup):void{
  const coarse=this.pipelines.get(`${entry}Coarse`),fine=this.pipelines.get(`${entry}Fine`);if(!coarse||!fine)throw new Error("Ownership transfer has not been initialized");
  const pass=e.beginComputePass({label:entry==="toPressure"?"Uniform mixed transfer to pressure":"Uniform mixed transfer to simulation"});
  pass.setBindGroup(0,this.simulation.bindGroup);pass.setBindGroup(1,group);
  this.simulation.dispatchCounted(pass,coarse,this.pages);
  pass.setPipeline(fine);pass.dispatchWorkgroups(this.fineGroups);
  pass.end();
 }
 /** Simulation to pressure: volume, faces and negative walls. */
 encodeToPressure(e:GPUCommandEncoder,group:GPUBindGroup):void{
  const word=uniformMixedResidencyWord(this.simulation.capacity.tiles);
  if(uniformMixedResidencyWord(this.pressure.capacity.tiles)!==word)throw new Error("Ownership transfer residency words differ");
  // Count, audit words, a flag per page and the resident list.
  e.copyBufferToBuffer(this.simulation.support,4*word,this.pressure.support,4*word,4*(4+2*this.pages));
  this.encode(e,"toPressure",group);
 }
 /** Pressure to simulation: faces and negative walls. */
 encodeToSimulation(e:GPUCommandEncoder,group:GPUBindGroup):void{this.encode(e,"toSimulation",group);}
 destroy():void{}
}

/** The transfer's kernels (UniformMixedOwnershipTransfer). Group 0 is the
 * simulation ownership in both directions; the pressure side is all-4h. */
function uniformMixedOwnershipTransferWGSL(layout:UniformMixedLayout):string{
 return uniformMixedCountedEntriesWGSL(uniformMixedTopologyWGSL(layout,0)+uniformMixedFaceAddressWGSL+/* wgsl */`
@group(1) @binding(0) var volume:texture_3d<f32>;
@group(1) @binding(1) var velocity:texture_3d<f32>;
@group(1) @binding(2) var<storage,read> negative:array<f32>;
@group(1) @binding(3) var outputVolume:texture_storage_3d<r32float,write>;
@group(1) @binding(4) var output:texture_storage_3d<rgba32float,write>;
@group(1) @binding(5) var<storage,read_write> boundary:array<f32>;
${uniformMixedFrameStatusWGSL(1,6,"read")}
fn tBits(texel:vec4f)->u32{return u32(round(texel.w));}
// The simulation tile at tile coordinate c across the tile's positive face
// on axis a is h: that 4h face holds sixteen simulation patches.
fn tSplit(c:vec3u,a:u32)->bool{var q=c;q[a]+=1u;return q[a]<UM_T[a]&&umTileWidth(umTileAt(q))==1u;}
// Job j of the simulation ownership's h and 4h tile lists.
fn tFineTile(j:u32)->u32{return umTopology[UM_TILES+j];}
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
// The 4h tiles of resident page job g, one lane each (residentPages).
fn tResidentCoarseTile(g:vec3u,lane:u32)->u32{let tile=umResidentPageTile(g.x,lane);if(tile>=UM_TILES||umTileWidth(tile)!=4u){return UM_TILES;}return tile;}
@compute @workgroup_size(64) fn toPressureCoarse(@builtin(workgroup_id) g:vec3u,@builtin(local_invocation_index) lane:u32){
 let tile=tResidentCoarseTile(g,lane);if(tile<UM_TILES){tCoarseToPressure(tile);}
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
// A latched frame failure (the verdict withheld the projection) transfers nothing back.
@compute @workgroup_size(64) fn toSimulationCoarse(@builtin(workgroup_id) g:vec3u,@builtin(local_invocation_index) lane:u32){
 if(umFrameFailed()){return;}
 let tile=tResidentCoarseTile(g,lane);if(tile<UM_TILES){tCoarseToSimulation(tile);}
}
@compute @workgroup_size(64) fn toSimulationFine(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(umFrameFailed()){return;}
 for(var job=group.x;job<umCounts.x;job+=groups.x){tFineToSimulation(tFineTile(job),lane);}
}
`,["toPressureCoarse","toSimulationCoarse"]);
}

/** Remap kernels between two ownership generations of one lattice. Group 0
 * is the source ("old"-prefixed WGSL), group 1 the target. Only the lattice
 * dimensions are compiled in; both ownerships are live buffers. */
/** The remap's two modules. Faces (group 2 = resources): the changed-tile
 * worklist, face remap from the live fields into the scratch fields, and
 * their publish. Cells (group 2 = cellsResources): volume and phi remapped
 * in place on the live fields. */
function uniformMixedRemapWGSL(layout:UniformMixedLayout,part:"faces"|"cells"):string{
 const topology=uniformMixedTopologyWGSL(layout,0,"old")+uniformMixedTopologyWGSL(layout,1);
 if(part==="faces")return topology+/* wgsl */`
@group(2) @binding(1) var velocity:texture_3d<f32>;
@group(2) @binding(3) var<storage,read> negative:array<f32>;
@group(2) @binding(5) var output:texture_storage_3d<rgba32float,write>;
@group(2) @binding(7) var<storage,read_write> boundary:array<f32>;
@group(2) @binding(8) var<storage,read_write> worklist:array<atomic<u32>>;
${uniformMixedChangedTilesWGSL(layout.tiles.length,2,9)}
${uniformMixedFrameStatusWGSL(3,0,"read_write")}
const REMAP_FATAL_HANGING:u32=${UNIFORM_MIXED_OVERFLOW_HANGING}u;
// A tile is visited when any tile of its 3x3x3 neighbourhood changed width.
// A tile that is h in both keeps its cells and face patches (seam patches are
// anchored in its cells), but when a neighbour refines it may inherit
// authority over a shared vertex (equal widths tie-break by owner,
// umVertexAuthority) that the old layout derived from the coarse corners and
// never stored: it must write that vertex, or the stale texel becomes the
// surface. A coarsening neighbour takes its shared vertices over instead.
// A frame that already failed lists nothing: its fields stay as they were.
@compute @workgroup_size(64) fn markChanged(@builtin(global_invocation_id) gid:vec3u){
 let tile=gid.x+umDispatchX*64u*gid.y;if(tile>=UM_TILES){return;}
 if(umFrameFailed()){return;}
 markTile(tile);
}
// The builder's relayout receipt (UNIFORM_MIXED_RELAYOUT_RECEIPT).
@group(2) @binding(10) var<storage,read> receipt:array<u32>;
const RECEIPT_FATAL:u32=${UNIFORM_MIXED_RELAYOUT_RECEIPT.fatal}u;const RECEIPT_FATAL_BUILD:u32=${UNIFORM_MIXED_RELAYOUT_RECEIPT.fatalBuild}u;
const RECEIPT_GENERATION:u32=${UNIFORM_MIXED_RELAYOUT_RECEIPT.generation}u;
// The same list from a GPU-built generation's dilated changed tiles
// (uniformMixedChangedTilesWGSL): a tile markTile lists has a width change in
// its 3x3x3 neighbourhood, so it is in that list, and every entry runs the
// same test. A fatal receipt bit lists nothing; lane 0 of the first group
// latches it once, then publishes the receipt's generation as the frame's
// current one (a failure latched later names it).
@compute @workgroup_size(64) fn markListed(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x+umDispatchX*64u*gid.y;let fatal=receipt[RECEIPT_FATAL];
 if(i==0u){
  if(fatal!=0u){umLatchFailure(select(${UNIFORM_MIXED_FAILURE.layoutCapacity}u,${UNIFORM_MIXED_FAILURE.invalidSupport}u,(fatal&~REMAP_FATAL_HANGING)!=0u),fatal,receipt[RECEIPT_FATAL_BUILD]);}
  atomicStore(&umStatus[${UNIFORM_MIXED_STATUS.currentGeneration}u],receipt[RECEIPT_GENERATION]);
 }
 if(fatal!=0u||umFrameFailed()||i>=umDilatedCount()){return;}
 markTile(umDilatedTile(i));
}
fn markTile(tile:u32){
 let fine=umTileWidth(tile)==1u&&oldumTileWidth(tile)==1u;
 let p=vec3i(umTileCoord(tile));
 for(var z=max(p.z-1,0);z<=min(p.z+1,i32(UM_T.z)-1);z++){for(var y=max(p.y-1,0);y<=min(p.y+1,i32(UM_T.y)-1);y++){for(var x=max(p.x-1,0);x<=min(p.x+1,i32(UM_T.x)-1);x++){
  let q=umTileAt(vec3u(vec3i(x,y,z)));
  if(select(umTileWidth(q)!=oldumTileWidth(q),umTileWidth(q)<oldumTileWidth(q),fine)){atomicStore(&worklist[4u+atomicAdd(&worklist[3],1u)],tile|listedFlags(tile));return;}
 }}}
}
// A listed entry is the tile and its width change, taken here while both
// generations are bound: the publish runs after the live ownership adopted
// the target, when the two widths agree. Bit 31: its width changed; bit 30:
// it refined; bits 27..29: the +x/+y/+z neighbour's width changed.
const LISTED_TILE:u32=0x07ffffffu;
fn listedFlags(tile:u32)->u32{
 var flags=select(0u,1u<<31u,umTileWidth(tile)!=oldumTileWidth(tile))|select(0u,1u<<30u,umTileWidth(tile)<oldumTileWidth(tile));
 let c=umTileCoord(tile);
 for(var a=0u;a<3u;a++){
  var q=c;q[a]+=1u;
  if(q[a]<UM_T[a]){let t=umTileAt(q);if(umTileWidth(t)!=oldumTileWidth(t)){flags|=1u<<(27u+a);}}
 }
 return flags;
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
var<workgroup> tileFaces:array<vec2f,192>;
var<workgroup> tileSamples:array<f32,192>;
// Faces: one lane per cell and axis evaluates the positive patch anchored at
// that cell (anchors lie inside their owner, so each texel has one writer)
// plus its owner's negative wall patch. The cell lane packs the texel.
// A positive patch's footprint cells are the tile cells on its plane: each
// such lane samples its own cell, and the anchor lane sums them in
// remapFace's order. A coarse patch no longer walks its footprint serially.
// A listed tile whose own width did not change keeps its owners and every
// face but those on its positive tile planes toward a neighbour that changed
// width (anchored here: a 4h owner's plane anchors and seam patches, an h
// cell's plane faces). Only texels holding such a face are remapped, all
// three components; its negative domain walls are the identity.
fn remapFaceTexel(entry:u32,local:vec3u)->bool{
 if((entry&(1u<<31u))!=0u){return true;}
 for(var a=0u;a<3u;a++){if(local[a]==3u&&(entry&(1u<<(27u+a)))!=0u){return true;}}
 return false;
}
// An entry with no width change of its own or of a +x/+y/+z neighbour
// writes no texel and no wall (remapFaceTexel is false in every lane): its
// job leaves at once (entry is workgroup-uniform).
const LISTED_FACES:u32=0x80000000u|(7u<<27u);
fn remapTileFaces(group:vec3u,lane:u32,remap:bool){
 let entry=listed(group,lane);if(entry==UM_UNLISTED||(entry&LISTED_FACES)==0u){return;}
 let tile=entry&LISTED_TILE;let cellLane=lane%64u;let axis=lane/64u;
 let anchor=vec3i(umTileCoord(tile)*4u+umCorner(cellLane,4u));
 let texel=remapFaceTexel(entry,umCorner(cellLane,4u));
 let owner=umOwnerAt(anchor);let origin=umOrigin(owner);
 if((entry&(1u<<31u))!=0u&&origin[axis]==0u&&all(vec3u(anchor)==origin)){
  let face=umFace(owner,axis,-1,0u);
  var value=0.0;if(remap){value=remapFace(face);}else{value=copyFace(face);}
  boundary[umNegativeBoundaryIndex(origin,axis)]=value;
 }
 let face=umPositiveFaceAtAnchor(owner,axis,anchor);
 let plane=i32(origin[axis]+owner.width)-1;
 if(texel&&remap&&anchor[axis]==plane){tileSamples[lane]=remapSample(vec3u(anchor),plane,axis);}
 workgroupBarrier();
 var bits=0u;var value=0.0;
 if(texel&&face.width!=0u){
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
 if(texel&&remap&&origin[axis]==0u&&remapReleased(umFace(owner,axis,-1,0u))){bits|=2u<<(axis+3u);}
 tileFaces[lane]=vec2f(value,f32(bits));
 workgroupBarrier();
 if(axis!=0u||!texel){return;}
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
 let jobs=listedJobCount(lane);for(var job=group.x+groups.x*group.y;job<jobs;job+=groups.x*groups.y){remapTileFaces(vec3u(job,0u,0u),lane,true);workgroupBarrier();}
}
@compute @workgroup_size(192) fn copyFaces(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let jobs=listedJobCount(lane);for(var job=group.x+groups.x*group.y;job<jobs;job+=groups.x*groups.y){remapTileFaces(vec3u(job,0u,0u),lane,false);workgroupBarrier();}
}
`;
 const vertexSampling=uniformMixedVertexSamplingSource("",false);
 const old=(s:string)=>s.replace(/\b(?:um[A-Z]\w*|UM_\w+|UMOwner|UMFace)\b/g,n=>"old"+n);
 return topology+/* wgsl */`
@group(2) @binding(0) var volume:texture_storage_3d<r32float,read_write>;
@group(2) @binding(1) var phi:texture_storage_3d<r32float,read_write>;
@group(2) @binding(2) var<storage,read_write> worklist:array<atomic<u32>>;
const LISTED_TILE:u32=0x07ffffffu;
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
fn oldumLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p)).x;}
${old(vertexSampling)}
fn umLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p)).x;}
${vertexSampling}
${geometricPlaneBoxWGSL}
fn uvCorner(k:u32)->vec3i{return vec3i(umCorner(k,2u));}
fn d4Sum8(v:array<f32,8>)->f32{return ((v[0]+v[5])+(v[1]+v[4]))+((v[2]+v[7])+(v[3]+v[6]));}
${uniformVolumeTargetWGSL(true,true)}
var<workgroup> tileVolume:array<f32,64>;
var<workgroup> tileFill:array<f32,64>;
// The old layout's phi at a refining tile's 5x5x5 closure: every new owner
// corner and every vertex it writes. The widest incident owner holds a
// vertex and a refining tile was one 4h owner, so each is its aligned
// corners' trilinear interpolant (a face or edge shared with another 4h
// owner interpolates the same corners), evaluated as the old sampler's
// regular branch does.
var<workgroup> tileCorners:array<f32,8>;
var<workgroup> tileVertices:array<f32,125>;
fn tileVertex(q:vec3u)->f32{return tileVertices[q.x+5u*(q.y+5u*q.z)];}
// In place, on the live volume and phi. Cells: every cell loads its donor's
// volume; each new owner's origin lane averages its cells. A donor lies in
// the tile (widths are per tile), so a tile reads and writes only its own
// volume texels, every read before the barrier. Vertices: each lattice
// vertex of the tile's closure is written by the tile holding its authority
// (the widest incident owner), and only if the old layout did not store it:
// at a stored vertex the old sampler returns the texel exactly, and the old
// sampler loads only stored vertices, so no tile writes a texel any other
// tile's old sample reads.
// A donor split into narrower owners does not broadcast its volume: its
// cells take the geometric fill of their new owner (the same target the
// solver compares against, from the remapped phi), scaled so the donor's
// volume is conserved exactly. Filled fractions shrink toward zero when
// the donor holds less than its phi implies, empty fractions toward zero
// when it holds more (up to full), so no fraction leaves [0,1]. Re-coarsening
// averages the cells back to the donor's volume, and its phi corners were
// never altered, so coarse -> fine -> coarse is the identity.
// A listed tile whose own width did not change (a neighbour's did) keeps its
// owners, so its cells are the identity; it may still take authority over a
// shared vertex the old layout derived from a coarse neighbour's corners.
// A listed 4h tile whose width did not change writes nothing: its cells are
// the identity, and its only canonical vertices are its eight tile corners,
// stored by every layout. Its job leaves at once (workgroup-uniform).
fn remapTileCells(group:vec3u,lane:u32){
 let entry=listed(group,lane);if(entry==UM_UNLISTED){return;}
 let tile=entry&LISTED_TILE;let same=(entry&(1u<<31u))==0u;let refine=(entry&(1u<<30u))!=0u;
 if(same&&umTileWidth(tile)==4u){return;}
 let base=umTileCoord(tile)*4u;let cell=base+umCorner(lane,4u);
 let o=umOwnerAt(vec3i(cell));
 var split=false;var donor:oldUMOwner;
 if(refine&&lane<8u){tileCorners[lane]=textureLoad(phi,vec3i(base+umCorner(lane,2u)*4u)).x;}
 if(!same){
  donor=oldumOwnerAt(vec3i(cell));
  tileVolume[lane]=textureLoad(volume,vec3i(oldumOrigin(donor))).x;
  split=o.width<donor.width;
 }
 workgroupBarrier();textureBarrier();
 if(refine){
  for(var v=lane;v<125u;v+=64u){
   let t=vec3f(umCorner(v,5u))*0.25;var values:array<f32,8>;
   for(var k=0u;k<8u;k++){let w=select(vec3f(1)-t,t,umCorner(k,2u)!=vec3u(0));values[k]=tileCorners[k]*w.x*w.y*w.z;}
   tileVertices[v]=d4Sum8(values);
  }
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
 if(!same&&all(cell==umOrigin(o))){
  let local=umCorner(lane,4u);var value=0.0;
  for(var z=0u;z<o.width;z++){for(var y=0u;y<o.width;y++){for(var x=0u;x<o.width;x++){
   let q=local+vec3u(x,y,z);value+=tileVolume[q.x+4u*(q.y+4u*q.z)];
  }}}
  textureStore(volume,vec3i(cell),vec4f(value/f32(o.width*o.width*o.width)));
 }
 for(var v=lane;v<125u;v+=64u){
  let p=base+umCorner(v,5u);let a=umVertexAuthority(p);
  if(a.width==0u||a.tile!=tile||!umVertexIsCanonical(p,a)){continue;}
  let stored=oldumVertexAuthority(p);
  if(stored.width!=0u&&oldumVertexIsCanonical(p,stored)){continue;}
  var value=0.0;
  if(refine){value=tileVertices[v];}else{value=oldumSampleVertex(vec3f(p));}
  textureStore(phi,vec3i(p),vec4f(value));
 }
}
@compute @workgroup_size(64) fn remapCells(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let jobs=listedJobCount(lane);for(var job=group.x+groups.x*group.y;job<jobs;job+=groups.x*groups.y){remapTileCells(vec3u(job,0u,0u),lane);workgroupBarrier();}
}
`;
}
