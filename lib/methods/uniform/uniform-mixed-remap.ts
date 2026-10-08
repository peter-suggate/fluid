import { uniformDetailBindLayout, uniformDetailModule, uniformDetailPipeline, uniformDetailPick, uniformDetailGroup, type UniformDetailGroup } from "./uniform-detail-fields";
import {UNIFORM_DETAIL_4H_LOAD,UNIFORM_DETAIL_CANONICAL_LOAD} from "../../core/uniform-detail-abi";
import {UniformMixedOwnership,type UniformMixedGenerationBuffers} from "./uniform-mixed-ownership";
import type {UniformMixedLayout} from "./uniform-mixed-layout";
import {UNIFORM_MIXED_COUNTED,UNIFORM_MIXED_OVERFLOW_FINE,UNIFORM_MIXED_OVERFLOW_HANGING,uniformMixedCountedEntriesWGSL,uniformMixedPageCount,uniformMixedResidencyWord,uniformMixedTopologyWGSL} from "./uniform-mixed-topology.wgsl";
import {uniformMixedVertexSamplingSource} from "./uniform-mixed-vertex-sampling.wgsl";
import {uniformMixedFaceAddressWGSL} from "./uniform-mixed-face-dispatch.wgsl";
import {uniformMixedSolidPipeline,uniformMixedSolidWGSL,type UniformMixedSolid} from "./uniform-mixed-solid.wgsl";
import {UNIFORM_MIXED_FAILURE,UNIFORM_MIXED_STATUS,UNIFORM_MIXED_STATUS_WORDS,uniformMixedFrameStatusWGSL} from "./uniform-mixed-frame-status";
import {uniformVolumeTargetWGSL} from "./uniform-volume.wgsl";
import {geometricPlaneBoxWGSL} from "../../core/geometric-plane-box.wgsl";
import {UNIFORM_MIXED_RELAYOUT_RECEIPT,uniformMixedChangedTilesWGSL} from "./uniform-mixed-layout-builder";
import {uniformBufferedWork} from "./uniform-buffered-work";

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
 * other owner's remap is the identity, so its live value stays in place.
 * With solids (group 3 of the face and cell kernels: the library and its
 * tile records) a statically cut tile is remapped by capacity: a 4h face is
 * the V-weighted mean of the h faces it replaces (the flux the root row sums,
 * as the ownership transfer keeps it), a split donor fills its children up
 * to their open fraction, and a tile handed to a cut 4h owner gives the
 * corners the h rule buried its plane (see remapTileCells). */
export class UniformMixedRemap {
 get allocatedBytes(){return this.worklist.size;}
 /** Group 1 over a GPU-built generation's buffers, bound per topology buffer. */
 private targetGroup?:{readonly topology:GPUBuffer;readonly group:UniformDetailGroup};
 private readonly resources:GPUBindGroupLayout;
 private readonly groups:readonly [UniformDetailGroup,UniformDetailGroup];
 /** The in-place cells remap: the live volume and phi read_write, and the
  * worklist. */
 private readonly cellsResources:GPUBindGroupLayout;
 private readonly cellsGroup:UniformDetailGroup;
 /** The census extension (scratch velocity and negative walls) remapped
  * through its own fields, before the velocity remap reuses the scratch. */
 private extensionGroups?:readonly [UniformDetailGroup,UniformDetailGroup];
 private readonly pipelines=new Map<string,GPUComputePipeline>();
 /** [unused x3, count, tiles...]: the changed-tile worklist. */
 private readonly worklist:GPUBuffer;
 /** Group 3: the frame status record (uniform-mixed-frame-status), bound
  * read_write. markChanged latches a builder fatal into it and lists nothing
  * once any failure is latched. A private zero record until bindStatus. */
 private readonly statusLayout:GPUBindGroupLayout;
 private readonly idleStatus:GPUBuffer;
 private statusGroup:UniformDetailGroup;
 /** Tiles in the lattice: the worklist's capacity. */
 private readonly tiles:number;
 /** markListed's group 2: the worklist, a generation's changed tiles
  * (UniformMixedGenerationBuffers.changes) and its builder's relayout
  * receipt, bound per changes buffer. */
 private readonly listResources:GPUBindGroupLayout;
 private listGroup?:{readonly changes:GPUBuffer;readonly receipt:GPUBuffer;readonly group:UniformDetailGroup};
 constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,input:Fields,scratch:Fields,private readonly solid?:UniformMixedSolid,private readonly surfaceOnly=false){
  this.worklist=device.createBuffer({label:"Uniform mixed remap worklist",size:(4+ownership.capacity.tiles)*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  this.tiles=ownership.capacity.tiles;
  if(this.tiles>0x04000000)throw new Error("Remap worklist entries pack a tile index below 2^26");
  this.resources=uniformDetailBindLayout(device,{entries:[
   ...[0,1,2].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
   {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
   ...[4,5,6].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only" as const,format:(binding===5?"rgba32float":"r32float") as GPUTextureFormat,viewDimension:"3d" as const}})),
   {binding:7,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
   {binding:8,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
  ]});
  const bind=(a:Fields,b:Fields)=>uniformDetailGroup(device,{layout:this.resources,entries:[
   ...[a.volume,a.velocity,a.phi].map((t,binding)=>({binding,resource:t})),{binding:3,resource:{buffer:a.negative}},
   ...[b.volume,b.velocity,b.phi].map((t,i)=>({binding:4+i,resource:t})),{binding:7,resource:{buffer:b.negative}},
   {binding:8,resource:{buffer:this.worklist}},
  ]});
  this.groups=[bind(input,scratch),bind(scratch,input)];
  this.listResources=uniformDetailBindLayout(device,{entries:[
   {binding:8,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:9,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
   {binding:10,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
  ]});
  this.cellsResources=uniformDetailBindLayout(device,{entries:[
   ...[0,1].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"read-write" as const,format:"r32float" as const,viewDimension:"3d" as const}})),
   {binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
  ]});
  this.cellsGroup=uniformDetailGroup(device,{layout:this.cellsResources,entries:[
   {binding:0,resource:input.volume},{binding:1,resource:input.phi},{binding:2,resource:{buffer:this.worklist}},
  ]});
  this.bind=bind;this.scratchFields=scratch;
  this.statusLayout=uniformDetailBindLayout(device,{entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
  this.idleStatus=device.createBuffer({label:"Uniform mixed remap idle status",size:UNIFORM_MIXED_STATUS_WORDS*4,usage:GPUBufferUsage.STORAGE});
  this.statusGroup=this.bindStatusGroup(this.idleStatus);
 }
 private bindStatusGroup(status:GPUBuffer):UniformDetailGroup{return uniformDetailGroup(this.device,{layout:this.statusLayout,entries:[{binding:0,resource:{buffer:status,size:UNIFORM_MIXED_STATUS_WORDS*4}}]});}
 /** The frame's status record (UniformMixedFrame.status). */
 bindStatus(status:GPUBuffer):void{this.statusGroup=this.bindStatusGroup(status);}
 private readonly bind:(a:Fields,b:Fields)=>UniformDetailGroup;
 private readonly scratchFields:Fields;
 /** Extension fields: the scratch velocity and negative walls are remapped
  * into these and copied back after the live ownership adopts. */
 bindExtension(input:Fields,extension:{velocity:GPUTexture;negative:GPUBuffer}):void{
  const s=this.scratchFields,remapped={volume:s.volume,velocity:extension.velocity,phi:s.phi,negative:extension.negative};
  this.extensionGroups=[this.bind({volume:input.volume,velocity:s.velocity,phi:input.phi,negative:s.negative},remapped),
   this.bind(remapped,{volume:input.volume,velocity:s.velocity,phi:input.phi,negative:s.negative})];
 }
 async initialize():Promise<void>{
  // One module per part. Group 3 is the frame status for the list entries,
  // the solid library for the face and cell kernels (twinned, see select).
  const modules=new Map<RemapPart,Promise<GPUShaderModule>>(),moduleOf=(part:RemapPart)=>{
   let module=modules.get(part);
   if(!module)modules.set(part,module=(async()=>{const m=uniformDetailModule(this.device,{label:`Uniform mixed remap ${part}`,code:uniformMixedRemapWGSL(this.ownership.layout,part,this.solid,this.surfaceOnly)});
    const errors=(await m.getCompilationInfo()).messages.filter(e=>e.type==="error");if(errors.length)throw new Error(errors.map(e=>`${e.lineNum}: ${e.message}`).join("\n"));return m;})());
   return module;
  };
  const compile=async(part:RemapPart,resources:GPUBindGroupLayout,entryPoints:readonly string[])=>{
   const module=await moduleOf(part),solid=part==="list"?undefined:this.solid;
   const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.ownership.bindLayout,resources,...(part==="list"?[this.statusLayout]:solid?[solid.tileLayout]:[])]});
   await Promise.all(entryPoints.map(async entryPoint=>{this.pipelines.set(entryPoint,await uniformMixedSolidPipeline(solid,s=>uniformDetailPipeline(this.device,this.ownership,{layout,compute:{module,entryPoint,constants:{umDispatchX:this.ownership.dispatchX,...s}}})));}));
  };
  await Promise.all([compile("list",this.resources,["markChanged"]),compile("faces",this.resources,["remapFaces","copyFaces"]),compile("list",this.listResources,["markListed"]),compile("cells",this.cellsResources,["remapCells"])]);
 }
 /** before: encoded ahead of the remap pass with the target's topology
  * (tile words from word 0): the detail storage's patch admission. */
 apply(layout:UniformMixedLayout,before?:(encoder:GPUCommandEncoder,target:GPUBuffer)=>void):void{
  if(this.pipelines.size!==5)throw new Error("Live remap has not been initialized");
  this.ownership.assertHolds(layout);
  // A transient target: host layouts are rare (authored regions, the initial layout).
  const target=new UniformMixedOwnership(this.device,layout,false);
  const encode=(copy:boolean)=>{
   const e=this.device.createCommandEncoder({label:copy?"Uniform publish remapped owners":"Uniform remap changed ownership"});
   if(!copy){e.clearBuffer(this.worklist,0,16);before?.(e,target.presentation.buffer);}
   this.encodePass(e,copy,target.bindGroup);this.device.queue.submit([e.finish()]);
  };
  encode(false);this.ownership.update(layout);encode(true);target.destroy();
 }
 /** The same remap for a GPU-built generation with no host object, in one
  * encoder at the head of the frame that runs on it: remap with the
  * builder's buffers bound as the target (its support words are bound but
  * never read), adopt them into the live ownership, publish. Encoded every frame; an unchanged generation lists no tile.
  * receipt: the builder's relayout receipt (UNIFORM_MIXED_RELAYOUT_RECEIPT),
  * read by markListed: a set fatal bit latches frame failure 6 (hanging tap or h-tile
  * capacity) or 7 (tier sum or tile words: invalid support) and lists
  * nothing; a frame already failed lists nothing either. Its generation word
  * then becomes the frame status's current generation. encodeClear must
  * precede it in the encoder. The buffer copies are not gated: a failed frame
  * adopts the built words unremapped, and the host throws on its record.
  * With extension, the scratch velocity (the census extension) is remapped
  * too. The host does not know the changed count: every launch strides the
  * GPU list on a grid budgeted from the lists of completed frames
  * (observeWork), REMAP_GRID without one. The worklist is built from the
  * generation's dilated changed tiles (markListed), not a lattice scan.
  * adopted: encoded right after the live ownership adopts (in that blit
  * run, between the remap and publish passes). */
 applyGpu(encoder:GPUCommandEncoder,source:UniformMixedGenerationBuffers,receipt:{readonly buffer:GPUBuffer;readonly offset:number},extension=false,adopted?:(encoder:GPUCommandEncoder)=>void):void{
  if(this.pipelines.size!==5)throw new Error("Live remap has not been initialized");
  if(extension&&!this.extensionGroups)throw new Error("Remap has no extension fields");
  if(!this.cleared)throw new Error("Live remap encoded without encodeClear");
  this.cleared=false;
  if(receipt.offset%256)throw new Error("Remap binds the relayout receipt at a 256 B aligned offset");
  if(this.listGroup?.changes!==source.changes||this.listGroup.receipt!==receipt.buffer)this.listGroup={changes:source.changes,receipt:receipt.buffer,group:uniformDetailGroup(this.device,{layout:this.listResources,entries:[
   {binding:8,resource:{buffer:this.worklist}},{binding:9,resource:{buffer:source.changes}},
   {binding:10,resource:{buffer:receipt.buffer,offset:receipt.offset,size:UNIFORM_MIXED_RELAYOUT_RECEIPT.words*4}}]})};
  if(this.targetGroup?.topology!==source.topology)this.targetGroup={topology:source.topology,group:uniformDetailGroup(this.device,{layout:this.ownership.bindLayout,entries:[
   {binding:0,resource:{buffer:source.topology}},{binding:1,resource:{buffer:source.counts.buffer,offset:source.counts.offset,size:16}},{binding:2,resource:{buffer:source.support}}]})};
  const target=this.targetGroup.group.group;
  const grid=this.work??Math.min(REMAP_GRID,this.tiles);
  this.encodePass(encoder,false,target,extension,grid,this.listGroup.group);
  this.ownership.adoptGpu(encoder,source);adopted?.(encoder);this.encodePass(encoder,true,target,extension,grid);
 }
 /** applyGpu's launch width (uniformBufferedWork: parallelism only, every
  * kernel strides the live list). listed: a completed frame's list. reach:
  * tiles within one of a request edit no receipt reports yet, its list's bound. */
 private work?:number;
 observeWork(listed:number,reserve=0,reach=0):void{this.work=uniformBufferedWork(this.work??1,listed+reach,Math.min(REMAP_GRID,this.tiles),reserve);}
 /** No evidence of the lists to come: the ceiling. */
 forgetWork():void{this.work=undefined;}
 /** The list's length (the last applyGpu's) into a frame receipt: one word. */
 encodeWorkReceipt(encoder:GPUCommandEncoder,target:GPUBuffer,offset:number):void{encoder.copyBufferToBuffer(this.worklist,12,target,offset,4);}
 /** Zero the worklist header for the next applyGpu: encode it ahead of the
  * frame head's first pass, in the same encoder, so it joins that blit run. */
 encodeClear(encoder:GPUCommandEncoder):void{encoder.clearBuffer(this.worklist,0,16);this.cleared=true;}
 private cleared=false;
 /** The worklist header is cleared before (apply, encodeClear). listed: markListed's
  * group (a GPU-built generation's changed tiles); without it the worklist
  * scans the lattice (markChanged). Each call is one compute pass: every
  * launch reads what the one before it wrote. target: group 1, the
  * generation being adopted. */
 private encodePass(e:GPUCommandEncoder,copy:boolean,target:GPUBindGroup,extension=false,grid=Math.min(REMAP_GRID,this.tiles),listed?:UniformDetailGroup):void{
  const x=this.ownership.dispatchX,dispatch=(pass:GPUComputePassEncoder)=>pass.dispatchWorkgroups(Math.min(grid,x),Math.ceil(grid/x));
  const pass=e.beginComputePass({label:copy?"Uniform mixed remap publish":"Uniform mixed remap"});
  pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,target);
  const solid=this.solid,faces=(group:UniformDetailGroup,name:string)=>{pass.setBindGroup(2,group.group);pass.setPipeline(uniformDetailPick(solid?solid.select(this.pipelines.get(name)!):this.pipelines.get(name)!));dispatch(pass);};
  // The list compares the live (old) and target widths, so it is built
  // before the live ownership adopts the target and reused by the publish.
  // Either launch is one lane per tile, the list's bound.
  if(!copy){
   const groups=Math.ceil(this.tiles/64);
   pass.setBindGroup(2,(listed??this.groups[0]).group);pass.setBindGroup(3,this.statusGroup.group);pass.setPipeline(uniformDetailPick(this.pipelines.get(listed?"markListed":"markChanged")!));
   pass.dispatchWorkgroups(Math.min(groups,x),Math.ceil(groups/x));
  }
  if(solid)pass.setBindGroup(3,solid.tileGroup);
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
 *   released only when its whole footprint was. On a cut tile the mean is
 *   weighted by each patch's V (umPressureFaceV): the 4h face then carries
 *   sum(V u)/sum(V), the flux the root row sums from the h faces through
 *   its record V, and the band's Neumann faces take that flux back after
 *   the projection. The plain mean of a face the floor shuts a row of lost
 *   that row's share of every seam flux.
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
 constructor(private readonly device:GPUDevice,readonly simulation:UniformMixedOwnership,readonly pressure:UniformMixedOwnership,private readonly solid?:UniformMixedSolid){
  if(simulation.layout.tiles.length!==pressure.layout.tiles.length)throw new Error("Ownership transfer requires one tile lattice");
  const tiles=simulation.layout.tiles.length;
  this.fineGroups=Math.max(1,Math.min(TRANSFER_GRID,tiles));this.pages=uniformMixedPageCount(simulation.capacity.lattice);
  if(pressure.layout.tiles.some(word=>(word&0xc0000000)!==0))throw new Error("Ownership transfer targets the all-4h pressure ownership");
  this.resources=uniformDetailBindLayout(device,{entries:[
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
 bind(input:{volume:GPUTexture;velocity:GPUTexture;negative:GPUBuffer},output:{volume:GPUTexture;velocity:GPUTexture;negative:GPUBuffer},status:GPUBuffer):UniformDetailGroup{
  return uniformDetailGroup(this.device,{layout:this.resources,entries:[
   {binding:0,resource:input.volume},{binding:1,resource:input.velocity},{binding:2,resource:{buffer:input.negative}},
   {binding:3,resource:output.volume},{binding:4,resource:output.velocity},{binding:5,resource:{buffer:output.negative}},
   {binding:6,resource:{buffer:status,size:64}},
  ]});
 }
 async initialize():Promise<void>{
  const module=uniformDetailModule(this.device,{label:"Uniform mixed ownership transfer",code:uniformMixedOwnershipTransferWGSL(this.simulation.layout,this.solid)});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  // Only the transfer to pressure reads the solid library (the patch weights).
  const weighted=(entryPoint:string)=>!!this.solid&&entryPoint.startsWith("toPressure");
  const layoutOf=(entryPoint:string)=>this.device.createPipelineLayout({bindGroupLayouts:[this.simulation.bindLayout,this.resources,...(weighted(entryPoint)?[this.solid!.tileLayout]:[])]});
  await Promise.all(["toPressureCoarse","toPressureFine","toSimulationCoarse","toSimulationFine"].map(async entryPoint=>{this.pipelines.set(entryPoint,await uniformMixedSolidPipeline(weighted(entryPoint)?this.solid:undefined,s=>uniformDetailPipeline(this.device,this.simulation,{layout:layoutOf(entryPoint),compute:{module,entryPoint,
   constants:{umDispatchX:this.simulation.dispatchX,...(entryPoint.endsWith("Coarse")?{umCountedJobs:UNIFORM_MIXED_COUNTED.residentPages}:{}),...s}}})));}));
 }
 /** Both launches write disjoint tiles and read only the source fields. */
 private encode(e:GPUCommandEncoder,entry:"toPressure"|"toSimulation",group:UniformDetailGroup):void{
  const coarse=this.pipelines.get(`${entry}Coarse`),fine=this.pipelines.get(`${entry}Fine`);if(!coarse||!fine)throw new Error("Ownership transfer has not been initialized");
  const pass=e.beginComputePass({label:entry==="toPressure"?"Uniform mixed transfer to pressure":"Uniform mixed transfer to simulation"});
  pass.setBindGroup(0,this.simulation.bindGroup);pass.setBindGroup(1,group.group);
  const weighted=!!this.solid&&entry==="toPressure";if(weighted)pass.setBindGroup(2,this.solid!.tileGroup);
  const variant=(p:GPUComputePipeline)=>weighted?this.solid!.select(p):p;
  this.simulation.dispatchCounted(pass,variant(coarse),this.pages);
  // A grid-stride over the h tiles (umCounts.x): sized by them, and not
  // issued while the simulation holds no h-tile capacity.
  this.simulation.dispatchBuffered(pass,variant(fine),"fine",this.fineGroups);
  pass.end();
 }
 /** Simulation to pressure: volume, faces and negative walls. */
 encodeToPressure(e:GPUCommandEncoder,group:UniformDetailGroup):void{
  const word=uniformMixedResidencyWord(this.simulation.capacity.tiles);
  if(uniformMixedResidencyWord(this.pressure.capacity.tiles)!==word)throw new Error("Ownership transfer residency words differ");
  // Count, audit words, a flag per page and the resident list.
  e.copyBufferToBuffer(this.simulation.support,4*word,this.pressure.support,4*word,4*(4+2*this.pages));
  this.encode(e,"toPressure",group);
 }
 /** Pressure to simulation: faces and negative walls. */
 encodeToSimulation(e:GPUCommandEncoder,group:UniformDetailGroup):void{this.encode(e,"toSimulation",group);}
 destroy():void{}
}

/** The transfer's kernels (UniformMixedOwnershipTransfer). Group 0 is the
 * simulation ownership in both directions; the pressure side is all-4h. */
function uniformMixedOwnershipTransferWGSL(layout:UniformMixedLayout,solid?:UniformMixedSolid):string{
 // A statically cut tile's patches are weighted by their V.
 const cut=(tile:string)=>solid?.coarse?`umSolidEnabled()&&umSolidStaticCut(${tile})`:"false";
 // A +face patch's V: the record's where it holds it (umPatchFaceV).
 const patchV=(tile:string,id:string,axis:string)=>solid?.coarse?`umPatchFaceV(${tile},${id},${axis})`:`umPressureFaceV(${id},${axis})`;
 return uniformMixedCountedEntriesWGSL(uniformMixedTopologyWGSL(layout,0)+uniformMixedFaceAddressWGSL+/* wgsl */`
${solid?uniformMixedSolidWGSL(2,solid.coarse?.count):""}
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
// plane, summed in tMean's order and V-weighted on a cut tile; never
// released) and the negative walls.
fn tCoarseToPressure(tile:u32){
 let c=umTileCoord(tile);let base=c*4u;let origin=vec3i(base);
 // The owner's own canonical texels: the base blocks (UNIFORM_DETAIL_4H_LOAD).
 textureStore(outputVolume,origin,vec4f(${UNIFORM_DETAIL_4H_LOAD}textureLoad(volume,origin,0).x));
 var texels:array<vec4f,3>;
 for(var a=0u;a<3u;a++){var anchor=origin;anchor[a]+=3;texels[a]=${UNIFORM_DETAIL_4H_LOAD}textureLoad(velocity,anchor,0);}
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
   let u=(a+1u)%3u;let v=(a+2u)%3u;var value=0.0;var weight=16.0;
   if(${cut("tile")}){
    weight=0.0;
    for(var y=0u;y<4u;y++){for(var x=0u;x<4u;x++){
     var q=base;q[a]+=3u;q[u]+=x;q[v]+=y;let w=${patchV("tile","vec3i(q)","a")};value+=w*textureLoad(velocity,vec3i(q),0)[a];weight+=w;
    }}
   }else{
    for(var y=0u;y<4u;y++){for(var x=0u;x<4u;x++){
     var q=base;q[a]+=3u;q[u]+=x;q[v]+=y;value+=textureLoad(velocity,vec3i(q),0)[a];
    }}
   }
   if(weight>0.0){value/=weight;}
   face=vec2f(value,0.0);
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
// z: the patch's weight (1, or its V on a cut tile).
var<workgroup> tPositive:array<vec3f,192>;
var<workgroup> tNegative:array<vec3f,192>;
// Sixteen plane samples summed in the order the remap's patch mean uses.
// wall: the negative domain plane's samples, else the positive plane's.
fn tMean(wall:bool,a:u32,cut:bool)->vec2f{
 let u=(a+1u)%3u;let v=(a+2u)%3u;var value=0.0;var released=1.0;var weight=0.0;
 for(var y=0u;y<4u;y++){for(var x=0u;x<4u;x++){
  var q=vec3u(0);q[a]=select(3u,0u,wall);q[u]=x;q[v]=y;let i=q.x+4u*(q.y+4u*q.z)+64u*a;
  var s=tPositive[i];if(wall){s=tNegative[i];}
  if(cut){value+=s.z*s.x;weight+=s.z;}else{value+=s.x;}released=min(released,s.y);
 }}
 if(!cut){weight=16.0;}
 if(weight>0.0){value/=weight;}
 return vec2f(value,released);
}
// An h tile, one workgroup (a lane per cell): volume is the mean of its
// cells, each 4h face (and negative wall) the mean of its sixteen patches.
fn tFineToPressure(tile:u32,lane:u32){
 let c=umTileCoord(tile);let local=umCorner(lane,4u);let base=c*4u;let cell=vec3i(base+local);
 tVolume[lane]=textureLoad(volume,cell,0).x;let texel=textureLoad(velocity,cell,0);let bits=tBits(texel);
 let cut=${cut("tile")};
 for(var a=0u;a<3u;a++){
  var weight=1.0;if(cut&&local[a]==3u){weight=${patchV("tile","cell","a")};}
  tPositive[lane+64u*a]=vec3f(texel[a],f32((bits>>a)&1u),weight);
  if(c[a]==0u&&local[a]==0u){
   var wall=1.0;if(cut){var outside=cell;outside[a]-=1;wall=umPressureFaceV(outside,a);}
   tNegative[lane+64u*a]=vec3f(negative[umNegativeBoundaryIndex(vec3u(cell),a)],f32((bits>>(a+3u))&1u),wall);
  }
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
  let wall=tMean(true,b,cut);
  negativeBits|=u32(wall.y)<<(b+3u);
  if(b==lane){boundary[umNegativeBoundaryIndex(base,b)]=wall.x;}
 }
 let a=lane;var anchor=vec3i(base);anchor[a]+=3;let face=tMean(false,a,cut);
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
// split ones written at their plane cells, and its negative walls. Both
// transfers to simulation read the all-4h pressure side at tile anchors
// only (an owner's and its negative neighbour's): the base block.
fn tCoarseToSimulation(tile:u32){
 let c=umTileCoord(tile);let base=c*4u;let origin=vec3i(base);
 var low=vec3f(0);var high=vec3f(0);var negativeBits=0u;var splits=0u;
 for(var a=0u;a<3u;a++){
  var anchor=origin;anchor[a]+=3;let texel=${UNIFORM_DETAIL_4H_LOAD}textureLoad(velocity,anchor,0);high[a]=texel[a];
  if(c[a]==0u){low[a]=negative[umNegativeBoundaryIndex(base,a)];negativeBits|=tBits(texel)&(1u<<(a+3u));boundary[umNegativeBoundaryIndex(base,a)]=low[a];}
  else{var below=origin;below[a]-=1;low[a]=${UNIFORM_DETAIL_4H_LOAD}textureLoad(velocity,below,0)[a];}
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
  var anchor=origin;anchor[a]+=3;let texel=${UNIFORM_DETAIL_4H_LOAD}textureLoad(velocity,anchor,0);let bits=tBits(texel);high[a]=texel[a];
  if(c[a]+1u==UM_T[a]){positiveBits|=bits&(1u<<a);}
  if(c[a]==0u){low[a]=negative[umNegativeBoundaryIndex(base,a)];negativeBits|=bits&(1u<<(a+3u));}
  else{var below=origin;below[a]-=1;low[a]=${UNIFORM_DETAIL_4H_LOAD}textureLoad(velocity,below,0)[a];}
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
 * in place on the live fields. NB only remaps phi; geometry regenerates V. */
type RemapPart="list"|"faces"|"cells";
function uniformMixedRemapWGSL(layout:UniformMixedLayout,part:RemapPart,solid?:UniformMixedSolid,surfaceOnly=false):string{
 const topology=uniformMixedTopologyWGSL(layout,0,"old")+uniformMixedTopologyWGSL(layout,1);
 // A statically cut tile's h faces are weighted by their V.
 const cut=(tile:string)=>solid?.coarse?`umSolidEnabled()&&umSolidStaticCut(${tile})`:"false";
 if(part==="list")return topology+/* wgsl */`
@group(2) @binding(8) var<storage,read_write> worklist:array<atomic<u32>>;
${uniformMixedChangedTilesWGSL(layout.tiles.length,2,9)}
${uniformMixedFrameStatusWGSL(3,0,"read_write")}
// The builder's capacity bits (hanging tap slots, h tiles): failure 6.
const REMAP_FATAL_CAPACITY:u32=${UNIFORM_MIXED_OVERFLOW_HANGING|UNIFORM_MIXED_OVERFLOW_FINE}u;
// A tile is visited when any tile of its 3x3x3 neighbourhood changed width.
// A tile that is h in both keeps its cells and face patches (seam patches are
// anchored in its cells), but when a neighbour refines it may inherit
// authority over a shared vertex (equal widths tie-break by owner,
// umVertexAuthority) that the old layout derived from the coarse corners and
// never stored: it must write that vertex, or the stale texel becomes the
// surface. A coarsening neighbour takes its shared vertices over instead.
// Equal-width authority is the lowest owner, and h owners are numbered in
// tile order: an h tile holds a vertex only when every other incident tile
// lies at a positive offset. The refined tile is then a +face or +edge
// neighbour (a +corner neighbour shares one tile corner, stored by every
// layout), so only those six give a tile that stays h a vertex to write
// (LISTED_VERTICES).
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
  if(fatal!=0u){umLatchFailure(select(${UNIFORM_MIXED_FAILURE.layoutCapacity}u,${UNIFORM_MIXED_FAILURE.invalidSupport}u,(fatal&~REMAP_FATAL_CAPACITY)!=0u),fatal,receipt[RECEIPT_FATAL_BUILD]);}
  atomicStore(&umStatus[${UNIFORM_MIXED_STATUS.currentGeneration}u],receipt[RECEIPT_GENERATION]);
 }
 if(fatal!=0u||umFrameFailed()||i>=umDilatedCount()){return;}
 markTile(umDilatedTile(i));
}
fn refinedAt(c:vec3u,d:vec3u)->bool{
 let q=c+d;if(any(q>=UM_T)){return false;}
 let t=umTileAt(q);return umTileWidth(t)<oldumTileWidth(t);
}
fn markTile(tile:u32){
 let fine=umTileWidth(tile)==1u&&oldumTileWidth(tile)==1u;
 let flags=listedFlags(tile);let faces=(flags&LISTED_FACES)!=0u;
 // A 4h tile that stays 4h beside no changed +x/+y/+z neighbour has no job:
 // its cells are the identity and it anchors no remapped face.
 if(umTileWidth(tile)==4u&&!faces){return;}
 var vertices=0u;
 if(fine){
  let c=umTileCoord(tile);
  if(refinedAt(c,vec3u(1u,0u,0u))||refinedAt(c,vec3u(0u,1u,0u))||refinedAt(c,vec3u(0u,0u,1u))||refinedAt(c,vec3u(1u,1u,0u))||refinedAt(c,vec3u(1u,0u,1u))||refinedAt(c,vec3u(0u,1u,1u))){vertices=LISTED_VERTICES;}
  // Neither a remapped face nor a vertex to take over: no job. (A +face
  // neighbour that refined sets both.)
  if(!faces&&vertices==0u){return;}
 }
 let p=vec3i(umTileCoord(tile));
 for(var z=max(p.z-1,0);z<=min(p.z+1,i32(UM_T.z)-1);z++){for(var y=max(p.y-1,0);y<=min(p.y+1,i32(UM_T.y)-1);y++){for(var x=max(p.x-1,0);x<=min(p.x+1,i32(UM_T.x)-1);x++){
  let q=umTileAt(vec3u(vec3i(x,y,z)));
  if(select(umTileWidth(q)!=oldumTileWidth(q),umTileWidth(q)<oldumTileWidth(q),fine)){atomicStore(&worklist[4u+atomicAdd(&worklist[3],1u)],tile|flags|vertices);return;}
 }}}
}
// A listed entry is the tile and its width change, taken here while both
// generations are bound: the publish runs after the live ownership adopted
// the target, when the two widths agree. Bit 31: its width changed; bit 30:
// it refined; bits 27..29: the +x/+y/+z neighbour's width changed; bit 26:
// it stays h and a +face or +edge neighbour refined (a vertex to take over).
const LISTED_TILE:u32=0x03ffffffu;
const LISTED_VERTICES:u32=1u<<26u;
const LISTED_FACES:u32=0x80000000u|(7u<<27u);
fn listedFlags(tile:u32)->u32{
 var flags=select(0u,1u<<31u,umTileWidth(tile)!=oldumTileWidth(tile))|select(0u,1u<<30u,umTileWidth(tile)<oldumTileWidth(tile));
 let c=umTileCoord(tile);
 for(var a=0u;a<3u;a++){
  var q=c;q[a]+=1u;
  if(q[a]<UM_T[a]){let t=umTileAt(q);if(umTileWidth(t)!=oldumTileWidth(t)){flags|=1u<<(27u+a);}}
 }
 return flags;
}
`;
 if(part==="faces")return topology+/* wgsl */`
${solid?uniformMixedSolidWGSL(3,solid.coarse?.count):uniformMixedSolidWGSL()}
@group(2) @binding(1) var velocity:texture_3d<f32>;
@group(2) @binding(3) var<storage,read> negative:array<f32>;
@group(2) @binding(5) var output:texture_storage_3d<rgba32float,write>;
@group(2) @binding(7) var<storage,read_write> boundary:array<f32>;
@group(2) @binding(8) var<storage,read_write> worklist:array<atomic<u32>>;
// A listed entry (listedFlags): the tile below its width-change bits.
const LISTED_TILE:u32=0x03ffffffu;
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
// A face wider than the faces it replaces is their mean, weighted by each
// one's V on a cut tile: the coarse face then carries sum(V u)/sum(V), the
// flux its pressure row sums from them, and a divergence-free h field
// coarsens to a divergence-free 4h one. The plain mean of a face the solid
// shuts a row of handed that row's share of the flux to the next solve.
fn remapFace(f:UMFace,cut:bool)->f32{
 let axis=f.axis;let u=(axis+1u)%3u;let v=(axis+2u)%3u;var value=0.0;var weight=0.0;
 for(var y=0u;y<f.width;y++){for(var x=0u;x<f.width;x++){
  var p=vec3u(max(f.anchor,vec3i(0)));p[u]+=x;p[v]+=y;var w=1.0;
  if(cut&&f.width>1u){var low=vec3i(p);low[axis]=f.anchor[axis];w=umPressureFaceV(low,axis);}
  value+=w*remapSample(p,f.anchor[axis],axis);weight+=w;
 }}
 if(weight>0.0){value/=weight;}
 return value;
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
 return ${UNIFORM_DETAIL_CANONICAL_LOAD}textureLoad(velocity,f.anchor,0)[f.axis];
}
var<workgroup> tileFaces:array<vec2f,192>;
var<workgroup> tileSamples:array<f32,192>;
// Each sample's weight: 1, or its h face's V on a cut tile.
var<workgroup> tileWeights:array<f32,192>;
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
 let owner=umOwnerAt(anchor);let origin=umOrigin(owner);let cut=remap&&${cut("tile")};
 if((entry&(1u<<31u))!=0u&&origin[axis]==0u&&all(vec3u(anchor)==origin)){
  let face=umFace(owner,axis,-1,0u);
  var value=0.0;if(remap){value=remapFace(face,cut);}else{value=copyFace(face);}
  boundary[umNegativeBoundaryIndex(origin,axis)]=value;
 }
 let face=umPositiveFaceAtAnchor(owner,axis,anchor);
 let plane=i32(origin[axis]+owner.width)-1;
 if(texel&&remap&&anchor[axis]==plane){
  tileSamples[lane]=remapSample(vec3u(anchor),plane,axis);var w=1.0;
  // Sixteen patches of a 4h face are copied one to one.
  if(cut&&owner.width>1u&&face.width!=1u){w=umPressureFaceV(anchor,axis);}
  tileWeights[lane]=w;
 }
 workgroupBarrier();
 var bits=0u;var value=0.0;
 if(texel&&face.width!=0u){
  bits=1u;
  if(remap){
   let u=(axis+1u)%3u;let v=(axis+2u)%3u;let local=umCorner(cellLane,4u);
   var weight=0.0;
   for(var y=0u;y<face.width;y++){for(var x=0u;x<face.width;x++){
    var q=local;q[u]+=x;q[v]+=y;let i=q.x+4u*(q.y+4u*q.z)+64u*axis;value+=tileWeights[i]*tileSamples[i];weight+=tileWeights[i];
   }}
   if(weight>0.0){value/=weight;}
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
 if(remap){packed.w=f32(released);}else{packed.w=${UNIFORM_DETAIL_CANONICAL_LOAD}textureLoad(velocity,anchor,0).w;}
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
${solid?uniformMixedSolidWGSL(3,solid.coarse?.count):uniformMixedSolidWGSL()}
@group(2) @binding(0) var volume:texture_storage_3d<r32float,read_write>;
@group(2) @binding(1) var phi:texture_storage_3d<r32float,read_write>;
@group(2) @binding(2) var<storage,read_write> worklist:array<atomic<u32>>;
const LISTED_TILE:u32=0x03ffffffu;
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
${surfaceOnly?"":uniformVolumeTargetWGSL(true,true)}
${surfaceOnly?"":`var<workgroup> tileVolume:array<f32,64>;
var<workgroup> tileFill:array<f32,64>;
// Each h cell's open fraction, and each new owner's (its cells' mean).
var<workgroup> tileOpen:array<f32,64>;
var<workgroup> tileCapacity:array<f32,64>;`}
// umVertexBuried under the old layout's widths.
fn oldBuried(p:vec3i)->bool{
 let base=p-vec3i(1);
 for(var k=0u;k<8u;k++){
  let cell=base+vec3i(umCorner(k,2u));if(!umSolidValid(cell)){continue;}
  let t=umTileAt(vec3u(cell)/4u);var open=0.0;
  if(oldumTileWidth(t)==4u){open=umTileOpen(t);}else{open=umCellOpen(cell);}
  if(open>1e-5){return false;}
 }
 return true;
}
// A tile handed to a cut 4h owner: it was h and has capacity.
fn planeTile(t:u32)->bool{return umTileWidth(t)==4u&&oldumTileWidth(t)!=4u&&umTileOpen(t)>1e-5;}
// The tile whose workgroup computes the 4-aligned vertex p's plane: the lowest such tile at it.
fn planeWriter(p:vec3i)->u32{
 let base=p/4-vec3i(1);var writer=UM_UNLISTED;
 for(var k=0u;k<8u;k++){
  let c=base+vec3i(umCorner(k,2u));if(any(c<vec3i(0))||any(c>=vec3i(UM_T))){continue;}
  let t=umTileAt(vec3u(c));if(planeTile(t)){writer=min(writer,t);}
 }
 return writer;
}
// The h rule buries a 4-aligned vertex whose eight h cells are closed and
// leaves its phi alone (an air sentinel, or whatever an earlier layout left);
// a cut 4h owner reads its plane at all eight corners. So a corner the old
// layout buried, handed to a cut 4h owner, takes the least-squares plane
// through the old h surface around it, read at the corner: a plane surface
// stays that plane across the change.
// The fit is the corner's own (the stored vertices of the open h cells
// within two tiles of it, whichever tile computes it: planeWriter only names
// the workgroup), so it has no low/high side. It goes through the vertices of
// the cells the surface crosses, each weighted by its crossing cells: the h
// redistance bends air-side vertices over dry land toward the shoreline's
// distance, which is not the surface's plane, and a corner 2e-4 cells off it
// unbalances the pressure phi of the wet tile beside it. Two tiles reach
// every corner of a tile beside one the surface crosses (the pressure
// couples face neighbours).
// With no crossing cell in reach no tile at the corner or beside it holds
// the surface: the corner takes the plane through every such vertex, kept on
// the side they are all on (an extrapolated air plane never puts liquid
// under dry land).
// The normal equations are centred, in cell units; a vertex set in one
// lattice plane leaves the slope across it zero (the minimum-norm fit: its
// moment is exactly zero and the diagonal shift keeps the system definite).
struct PlaneCell{count:u32,value:array<f32,8>}
// An open h cell's stored vertices under the old layout (count 0: not one).
// Inside an h tile every vertex is stored; on its faces a wider neighbour
// may hold the authority.
fn planeCell(cell:vec3i)->PlaneCell{
 var out:PlaneCell;
 if(umCellOpen(cell)<=1e-5){return out;}
 for(var k=0u;k<8u;k++){
  let at=vec3u(cell)+umCorner(k,2u);
  if(any(at%vec3u(4u)==vec3u(0))){
   let stored=oldumVertexAuthority(at);
   if(stored.width==0u||!oldumVertexIsCanonical(at,stored)){out.value[k]=0.0;continue;}
  }
  out.value[k]=textureLoad(phi,vec3i(at)).x;out.count|=1u<<k;
 }
 return out;
}
fn planeCrossing(c:PlaneCell)->bool{
 var below=false;var above=false;
 for(var k=0u;k<8u;k++){if((c.count&(1u<<k))!=0u){if(c.value[k]<0.0){below=true;}else{above=true;}}}
 return below&&above;
}
fn planeOldFine(t:vec3i)->bool{
 if(any(t<vec3i(0))||any(t>=vec3i(UM_T))){return false;}
 let tile=umTileAt(vec3u(t));return oldumTileWidth(tile)!=4u&&umTileOpen(tile)>1e-5;
}
// The sums are split over the workgroup: lane i takes tile i of the corner's
// 4^3 tiles and lane 0 adds the lanes' partials in tile order (one lane over
// all 4096 cells, twice, held the frame for tens of milliseconds). Counts and
// offset sums are whole numbers, so the mean is the serial sum's exactly.
var<workgroup> planeA:array<vec4f,64>;
var<workgroup> planeB:array<vec4f,64>;
var<workgroup> planeC:array<vec4f,64>;
// (mean, 1: through crossing cells' vertices, 0: through every vertex, -1: no vertex), (their count, low, high).
var<workgroup> planeHead:array<vec4f,2>;
// The listed tile's corners that take a plane, a bit each.
var<workgroup> planeCorners:u32;
fn planeCorner(p:vec3i,lane:u32){
 let t=p/4-vec3i(2)+vec3i(umCorner(lane,4u));let fine=planeOldFine(t);
 // Offsets from p; [0]: every stored vertex, [1]: those of crossing cells.
 var count=array<f32,2>(0.0,0.0);var first=array<vec3f,2>(vec3f(0),vec3f(0));
 var low=1e30;var high=-1e30;
 if(fine){
  for(var j=0u;j<64u;j++){
   let cell=t*4+vec3i(umCorner(j,4u));let c=planeCell(cell);if(c.count==0u){continue;}
   let which=select(0u,1u,planeCrossing(c));
   for(var k=0u;k<8u;k++){if((c.count&(1u<<k))==0u){continue;}
    let r=vec3f(cell+vec3i(umCorner(k,2u))-p);
    count[0]+=1.0;first[0]+=r;low=min(low,c.value[k]);high=max(high,c.value[k]);
    if(which==1u){count[1]+=1.0;first[1]+=r;}
   }
  }
 }
 planeA[lane]=vec4f(first[0],count[0]);planeB[lane]=vec4f(first[1],count[1]);planeC[lane]=vec4f(low,high,0.0,0.0);
 workgroupBarrier();
 if(lane==0u){
  var every=vec4f(0);var crossed=vec4f(0);var limits=vec2f(1e30,-1e30);
  for(var i=0u;i<64u;i++){every+=planeA[i];crossed+=planeB[i];limits=vec2f(min(limits.x,planeC[i].x),max(limits.y,planeC[i].y));}
  let through=crossed.w!=0.0;var s=every;if(through){s=crossed;}
  planeHead[0]=vec4f(s.xyz/s.w,select(select(0.0,1.0,through),-1.0,every.w==0.0));planeHead[1]=vec4f(s.w,limits,0.0);
 }
 workgroupBarrier();
 let head=planeHead[0];let mean=head.xyz;let crossing=head.w>0.5;
 var total=0.0;var diagonal=vec3f(0);var cross3=vec3f(0);var moment=vec3f(0);
 if(fine&&head.w>=0.0){
  for(var j=0u;j<64u;j++){
   let cell=t*4+vec3i(umCorner(j,4u));let c=planeCell(cell);if(c.count==0u){continue;}
   if(crossing&&!planeCrossing(c)){continue;}
   for(var k=0u;k<8u;k++){if((c.count&(1u<<k))==0u){continue;}
    let r=vec3f(cell+vec3i(umCorner(k,2u))-p)-mean;
    total+=c.value[k];diagonal+=r*r;cross3+=r.xxy*r.yzz;moment+=r*c.value[k];
   }
  }
 }
 planeA[lane]=vec4f(diagonal,total);planeB[lane]=vec4f(cross3,0.0);planeC[lane]=vec4f(moment,0.0);
 workgroupBarrier();
 if(lane==0u&&head.w>=0.0){
  total=0.0;diagonal=vec3f(0);cross3=vec3f(0);moment=vec3f(0);
  for(var i=0u;i<64u;i++){total+=planeA[i].w;diagonal+=planeA[i].xyz;cross3+=planeB[i].xyz;moment+=planeC[i].xyz;}
  // Centred offsets sum to zero, so the moment needs no level term. The
  // shifted solve is refined once on the unshifted residual, which removes
  // the diagonal shift's bias on the slope (1e-6 of it: 1e-5 cells here).
  let level=total/planeHead[1].x;low=planeHead[1].y;high=planeHead[1].z;
  let shift=1e-6*max(diagonal.x+diagonal.y+diagonal.z,1.0);
  let c0=vec3f(diagonal.x+shift,cross3.x,cross3.y);let c1=vec3f(cross3.x,diagonal.y+shift,cross3.z);let c2=vec3f(cross3.y,cross3.z,diagonal.z+shift);
  let det=dot(c0,cross(c1,c2));var slope=vec3f(0);
  for(var fit=0u;fit<2u;fit++){
   let rest=moment-vec3f(dot(vec3f(diagonal.x,cross3.x,cross3.y),slope),dot(vec3f(cross3.x,diagonal.y,cross3.z),slope),dot(vec3f(cross3.y,cross3.z,diagonal.z),slope));
   slope+=vec3f(dot(rest,cross(c1,c2)),dot(c0,cross(rest,c2)),dot(c0,cross(c1,rest)))/det;
  }
  var value=level-dot(slope,mean);
  if(!crossing){if(low>=0.0){value=max(value,low);}else if(high<0.0){value=min(value,high);}}
  textureStore(phi,p,vec4f(value));
 }
 workgroupBarrier();
}
// The old layout's phi at a refining tile's 5x5x5 closure: every new owner
// corner and every vertex it writes. The widest incident owner holds a
// vertex and a refining tile was one 4h owner, so each is its aligned
// corners' trilinear interpolant (a face or edge shared with another 4h
// owner interpolates the same corners), evaluated as the old sampler's
// regular branch does.
// A corner the old layout buried (umVertexBuried: every incident tile closed
// throughout) is not state. Its tile is closed, so every vertex it weights is
// buried under the new layout too: that vertex takes the corner's value as it
// is (the air sentinel of construction), never a blend of it with the plane.
var<workgroup> tileCorners:array<f32,8>;
var<workgroup> tileCornerBuried:array<u32,8>;
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
// solver compares against, from the remapped phi, times the owner's open
// fraction), scaled so the donor's volume is conserved exactly. Filled
// fractions shrink toward zero when the donor holds less than its phi
// implies, empty fractions toward zero when it holds more (up to its
// capacity, the cells' open fractions), so no child leaves [0,open]: a cut
// donor puts nothing in a closed cell. Past capacity, or with no shape to
// follow, a cut donor's volume is spread by open fraction. Re-coarsening
// averages the cells back to the donor's volume, and its phi corners were
// never altered, so coarse -> fine -> coarse is the identity.
// A listed tile whose own width did not change (a neighbour's did) keeps its
// owners, so its cells are the identity; it may still take authority over a
// shared vertex the old layout derived from a coarse neighbour's corners.
// A listed 4h tile whose width did not change writes nothing: its cells are
// the identity, and its only canonical vertices are its eight tile corners,
// stored by every layout. Its job leaves at once (workgroup-uniform), and so
// does an h tile's that is listed for its faces alone (no bit 26: no +face or
// +edge neighbour refined, so it takes over no vertex).
fn remapTileCells(group:vec3u,lane:u32){
 let entry=listed(group,lane);if(entry==UM_UNLISTED){return;}
 let tile=entry&LISTED_TILE;let same=(entry&(1u<<31u))==0u;let refine=(entry&(1u<<30u))!=0u;
 if(same&&(umTileWidth(tile)==4u||(entry&(1u<<26u))==0u)){return;}
 let base=umTileCoord(tile)*4u;let cell=base+umCorner(lane,4u);
 let o=umOwnerAt(vec3i(cell));
 ${surfaceOnly?"":"var split=false;var donor:oldUMOwner;"}
 if(refine&&lane<8u){
  let corner=vec3i(base+umCorner(lane,2u)*4u);tileCorners[lane]=textureLoad(phi,corner).x;
  tileCornerBuried[lane]=select(0u,1u,umSolidEnabled()&&oldBuried(corner));
 }
 // A tile handed to a cut 4h owner (workgroup-uniform): a closed h cell
 // leaves the tile's mean open fraction under 63/64.
 let plane=!same&&!refine&&umSolidEnabled()&&planeTile(tile)&&umTileOpen(tile)<0.99;
 ${surfaceOnly?"":` if(!same){
  donor=oldumOwnerAt(vec3i(cell));
  tileVolume[lane]=textureLoad(volume,vec3i(oldumOrigin(donor))).x;
  split=o.width<donor.width;
  tileOpen[lane]=umCellOpen(vec3i(cell));
 }
`}
 workgroupBarrier();textureBarrier();
 if(refine){
  for(var v=lane;v<125u;v+=64u){
   let t=vec3f(umCorner(v,5u))*0.25;var values:array<f32,8>;var dead=8u;
   for(var k=0u;k<8u;k++){let w=select(vec3f(1)-t,t,umCorner(k,2u)!=vec3u(0));values[k]=tileCorners[k]*w.x*w.y*w.z;
    if(dead==8u&&w.x*w.y*w.z!=0.0&&tileCornerBuried[k]!=0u){dead=k;}}
   if(dead<8u){tileVertices[v]=tileCorners[dead];}else{tileVertices[v]=d4Sum8(values);}
  }
 }
 workgroupBarrier();
 ${surfaceOnly?"":` // Tile widths are uniform: a new owner's cells share its origin lane's fill.
 let ownerLocal=(umCorner(lane,4u)/o.width)*o.width;
 if(split&&all(cell==umOrigin(o))){
  var vertices:array<f32,8>;
  for(var j=0u;j<8u;j++){vertices[j]=tileVertex(ownerLocal+umCorner(j,2u)*o.width);}
  var open=0.0;let corner=umCorner(lane,4u);
  for(var z=0u;z<o.width;z++){for(var y=0u;y<o.width;y++){for(var x=0u;x<o.width;x++){
   let q=corner+vec3u(x,y,z);open+=tileOpen[q.x+4u*(q.y+4u*q.z)];
  }}}
  open/=f32(o.width*o.width*o.width);
  tileCapacity[lane]=open;tileFill[lane]=umSurfaceTarget(o,vertices)*open;
 }
`}
 // The tile's corners that take a plane, one at a time across the workgroup.
 if(lane==0u){
  var wanted=0u;
  if(plane){for(var k=0u;k<8u;k++){let corner=vec3i(base+umCorner(k,2u)*4u);if(planeWriter(corner)==tile&&oldBuried(corner)){wanted|=1u<<k;}}}
  planeCorners=wanted;
 }
 let corners=workgroupUniformLoad(&planeCorners);
 for(var k=0u;k<8u;k++){if((corners&(1u<<k))!=0u){planeCorner(vec3i(base+umCorner(k,2u)*4u),lane);}}
 workgroupBarrier();
 ${surfaceOnly?"":` // A split donor is wider than one cell, so it lies inside this tile.
 if(split){
  let local=oldumOrigin(donor)-base;let n=donor.width*donor.width*donor.width;var fill=0.0;var capacity=0.0;
  for(var z=0u;z<donor.width;z++){for(var y=0u;y<donor.width;y++){for(var x=0u;x<donor.width;x++){
   let q=((local+vec3u(x,y,z))/o.width)*o.width;let i=q.x+4u*(q.y+4u*q.z);fill+=tileFill[i];capacity+=tileCapacity[i];
  }}}
  // A donor whose phi is (nearly) all empty or all full has no shape to
  // follow, and an overfull donor (V > its capacity) keeps its excess
  // uniform: over its open fraction when the solid cuts it.
  let i=ownerLocal.x+4u*(ownerLocal.y+4u*ownerLocal.z);
  let held=tileVolume[lane]*f32(n);let f=tileFill[i];let open=tileCapacity[i];let room=capacity-fill;
  if(held<=fill&&fill>=0.5){tileVolume[lane]=f*held/fill;}
  else if(held>fill&&held<=capacity&&room>=0.5){tileVolume[lane]=open-(open-f)*(capacity-held)/room;}
  else if(capacity>1e-5&&capacity<f32(n)){tileVolume[lane]=open*held/capacity;}
 }
 workgroupBarrier();
 if(!same&&all(cell==umOrigin(o))){
  let local=umCorner(lane,4u);var value=0.0;
  for(var z=0u;z<o.width;z++){for(var y=0u;y<o.width;y++){for(var x=0u;x<o.width;x++){
   let q=local+vec3u(x,y,z);value+=tileVolume[q.x+4u*(q.y+4u*q.z)];
  }}}
  textureStore(volume,vec3i(cell),vec4f(value/f32(o.width*o.width*o.width)));
 }
`}
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
