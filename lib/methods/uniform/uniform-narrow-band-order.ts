import { gpuCompilationManagerFor } from "../../core/gpu-compilation-manager";
import { narrowBandMergeWGSL } from "./uniform-narrow-band-merge.wgsl";
import { narrowBandParticleStride, narrowBandParticleWGSL } from "./uniform-narrow-band-apic.wgsl";
import { uniformWindowAddressWGSL } from "./uniform-window-address";
import { ParticleBinScan } from "./particle-bin-scan";
import { uniformLatticeAddressWGSL, type UniformLatticeAddress } from "./uniform-lattice-address";

/** The stage's pass descriptors: begun every frame, built once. */
const PREPARE_PASS:GPUComputePassDescriptor={label:"Narrow-band FLIP clear occupied bins"};
const PREFIX_PASS:GPUComputePassDescriptor={label:"Narrow-band FLIP occupied cell offsets"},PACK_PASS:GPUComputePassDescriptor={label:"Narrow-band FLIP order pack"};
const CROWDED_PASS:GPUComputePassDescriptor={label:"Narrow-band FLIP crowded cell parallel merge"};
const CANONICAL_PASS:GPUComputePassDescriptor={label:"Narrow-band FLIP canonical cell order"};
/** Workgroups of every launch over the live samples, one row. A kernel reads
 * the count on the GPU and strides the launch through it, 64 samples a group
 * a turn, so any width is complete at any count up to the capacity. This one
 * is the ownership's counted launches' (262,144 lanes): enough to fill the
 * GPU, few enough that a launch with little live stays cheap. Letters, the
 * five passes that hold these launches, against a group a sample block: 1.28
 * ms for 1.30 under half a million samples (most groups idle), 5.62 for 5.58
 * over 1.25 million (five turns a group; the snapshot alone 0.85 for 0.74). */
export const NARROW_BAND_PARTICLE_GRID=4096;
/** Parallelism for a frame whose lagged heat contains no HOT region or live
 * particle. Keep a nonzero direct grid: every kernel strides the actual GPU
 * count, so an unexpected promotion still processes all arrivals. */
export const NARROW_BAND_QUIET_PARTICLE_GRID=64;

/** Reorder the live particles by cell without a bucket limit or another
 * particle allocation. The expired source epoch becomes the sorted output.
 * The input is every slot of the last epoch, moved in place: a sample that
 * left the liquid's domain has a negative x and is dropped here.
 * Cells are ordered a topology tile at a time, x fastest inside it, so all
 * 4 cubed fine cells of a tile are adjacent in memory; linear h-cell order
 * separates z-neighbors by an entire domain plane. The move counted each
 * cell's samples at that order in the first bank of the bins. Their prefix
 * sum, written to the head of the links, is each cell's first sample: a
 * cell's samples are one contiguous run, and so are those of consecutive
 * cells. Keys scatter in parallel through a cursor in the bins' second
 * bank so a crowded bin cannot serialize an entire workgroup; each sample
 * then sorts fixed-size runs and merges across run boundaries for every
 * cell. The exact order does not depend on scheduling.
 * Scratch holds compact position keys and per-stage run lists, not a record
 * allocation. The keys stay in the links: the
 * NB-FLIP tile marking, one workgroup a tile's run, writes the sorted
 * positions over them as it reads the moved records. */
export class UniformNarrowBandOrder {
 /** Device-scoped diagnostic construction policy. Existing instances never
  * switch layouts; production always uses the shared occupied-tile domain. */
 private static readonly legacyScanDevices=new WeakSet<object>();
 static useLegacyScanForDevice(device:GPUDevice,enabled:boolean):void{
  // Initialization uses a scoped device proxy; its compilation authority is
  // the stable identity shared with the caller's device.
  const identity=gpuCompilationManagerFor(device);
  if(enabled)this.legacyScanDevices.add(identity);else this.legacyScanDevices.delete(identity);
 }
 workgroups=NARROW_BAND_PARTICLE_GRID;
 /** Diagnostic ablation of the former unbounded pairwise rank. */
 legacyCanonical=false;
 private readonly orderScratch:GPUBuffer;
 private readonly scratchKeys:number;
 private readonly enabled:GPUBuffer;
 private readonly dimensions:GPUBuffer;
 private readonly tileOffsets?:GPUBuffer;
 private preparePipeline!:GPUComputePipeline;
 private prefixPipeline!:GPUComputePipeline;
 private scan!:ParticleBinScan;
 private pipeline!:GPUComputePipeline;
 private canonicalPipeline!:GPUComputePipeline;
 private legacyPipeline!:GPUComputePipeline;
 private mergePipelines:GPUComputePipeline[]=[];
 private readonly mergeLevels:number;
 private readonly mergeQueues:number[];
 private readonly headerWords:number;
 private groups!:readonly [GPUBindGroup,GPUBindGroup];
 /** Cell keys of the lattice: the bins' first two banks. */
 private readonly cells:number;
 get allocatedBytes():number{return this.orderScratch.size+this.enabled.size+this.dimensions.size+(this.tileOffsets?.size??0)+(this.scan?.allocatedBytes??0);}
 /** topology: the simulation ownership's, whose tail holds the pages. The
  * optional policy override lets small fixtures exercise both paths.
 * occupied: the bins word of the occupied tile list (its length, entries
 * from 4 words on), which every tile with coverage is on: prepare and the
 * cell offsets stride it on a fixed grid, and prepare empties it. */
 constructor(private readonly device:GPUDevice,private readonly address:UniformLatticeAddress,private readonly topology:GPUBuffer,private readonly particles:readonly [GPUBuffer,GPUBuffer],private readonly bins:GPUBuffer,private readonly links:GPUBuffer,private readonly state:GPUBuffer,private readonly occupied:number,readonly sparse=!UniformNarrowBandOrder.legacyScanDevices.has(gpuCompilationManagerFor(device))||address.cellKeys>=4_194_304,private readonly apic=false,
  /** false (NB-FLIP particleOrder "scheduled"): pack scatters each record
   * straight to its cell's run, in the order its cursor hands out slots; no
   * canonical sort, merge or key scratch. Within a cell the order then
   * depends on scheduling. */
  readonly canonical=true){
  this.cells=address.cellKeys;
  const capacity=Math.floor(particles[0].size/(4*narrowBandParticleStride(apic)));
  this.mergeLevels=Math.max(0,Math.ceil(Math.log2(capacity/128)));
  // A merge list has at most N/256 + N/(width+1) runs: only cells larger
  // than width join it. Keep counters separate from non-atomic key storage.
  // Every merge consumes this frame’s queues on a bounded direct grid.
  this.headerWords=4*(this.mergeLevels+1);
  let next=this.headerWords;
  this.mergeQueues=Array.from({length:this.mergeLevels},(_,level)=>{
   const base=next;next+=2*(Math.ceil(capacity/256)+Math.floor(capacity/(128*2**level+1)));return base;
  });
  this.scratchKeys=next;
  this.orderScratch=device.createBuffer({label:"FLIP canonical merge keys",size:canonical?4*(next+4*capacity):4*(next+4),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  // Shared APIC scan's activation record: no failure and an active step.
  // This immutable record makes our sort unconditional, including bootstrap.
  this.dimensions=device.createBuffer({label:"FLIP spatial order dimensions",size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(this.dimensions,0,new Uint32Array([...address.cells,address.blocks]));
  this.enabled=device.createBuffer({label:"FLIP spatial order scan activation",size:88,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  if(this.sparse)this.tileOffsets=device.createBuffer({label:"FLIP tile offsets",size:4*(address.tileKeys+Math.ceil(address.tileKeys/128)),usage:GPUBufferUsage.STORAGE});
  const activation=new Float32Array(22);activation[21]=1;device.queue.writeBuffer(this.enabled,0,activation);
 }
 async initialize():Promise<void>{
  const cells=this.address.cellKeys,tiles=this.address.tileKeys;
  // Every size consumes the same occupied-tile list already produced by
  // movement and seeding. Prefix only tile totals, then occupied cells;
  // empty cells neither participate in the scan nor need a full-bank clear.
  // The dense path remains an exact diagnostic reference.
  this.scan=await ParticleBinScan.create(this.device,this.sparse?tiles:cells,
   {buffer:this.bins,offset:this.sparse?8*cells:0,size:this.sparse?4*tiles:4*cells},
   this.sparse?{buffer:this.tileOffsets!,offset:0,size:4*tiles}:{buffer:this.links,offset:0,size:4*cells},this.enabled,undefined,
   this.sparse?{buffer:this.tileOffsets!,offset:4*tiles,size:4*Math.ceil(tiles/128)}:undefined);
  const module=this.device.createShaderModule({label:"FLIP spatial order",code:/* wgsl */`
${narrowBandParticleWGSL(this.apic)}
@group(0) @binding(7) var<uniform> orderDimensions:vec4u;
// The box and the blocks are this module's uniform, the pages the
// topology's tail: the address functions read them.
@group(0) @binding(6) var<storage,read> orderTopology:array<u32>;
fn orderTiles()->u32{return 512u*orderDimensions.w;}
${uniformWindowAddressWGSL("orderTopology","5u*orderTiles()","orderDimensions.w")}
${uniformLatticeAddressWGSL({constant:false,paged:{blocks:"orderDimensions.w"}})}
@group(0) @binding(0) var<storage,read> input:array<Particle>;
@group(0) @binding(1) var<storage,read_write> output:array<Particle>;
// Each cell's count, then its scatter cursor.
@group(0) @binding(2) var<storage,read_write> bins:array<atomic<u32>>;
// Each cell's first sample, then the compact positions the gathers read.
@group(0) @binding(3) var<storage,read_write> links:array<u32>;
// The receipt: [0] the input's slots, [1] the live samples packed.
@group(0) @binding(4) var<storage,read_write> state:array<u32>;
// The occupied tile list's word in the bins (UniformNarrowBandFlip.occupiedWord).
const ORDER_OCCUPIED:u32=${this.occupied}u;
${this.sparse?`@group(0) @binding(5) var<storage,read> tileOffsets:array<u32>;
// The shared tile-total scan leaves local offsets and the complete parent
// prefix. Add it only at occupied consumers, never across the whole plane.
fn orderTileStart(tile:u32)->u32{return tileOffsets[tile]+tileOffsets[orderTiles()+tile/128u];}`:""}
${this.sparse?/* wgsl */`
var<workgroup> prefix:array<u32,64>;
var<workgroup> prefixListed:u32;
var<workgroup> prefixTileKey:u32;
// Only the occupied tiles (the list: every tile with coverage) hold cell
// counts to clear; the list is emptied after this pass.
@compute @workgroup_size(64) fn prepare(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let count=min(atomicLoad(&bins[ORDER_OCCUPIED]),orderTiles());
 for(var j=group.x;j<count;j+=groups.x){
  let tile=atomicLoad(&bins[ORDER_OCCUPIED+4u+j]);
  atomicStore(&bins[64u*tile+lane],0u);atomicStore(&bins[UM_CELL_KEYS()+64u*tile+lane],0u);
  if(lane==0u){atomicStore(&bins[2u*UM_CELL_KEYS()+tile],0u);}
 }
}
@compute @workgroup_size(64) fn prefixTile(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(group.x==0u&&lane==0u){let last=orderTiles()-1u;state[1]=orderTileStart(last)+atomicLoad(&bins[2u*UM_CELL_KEYS()+last]);}
 if(lane==0u){prefixListed=min(atomicLoad(&bins[ORDER_OCCUPIED]),orderTiles());}
 let listed=workgroupUniformLoad(&prefixListed);
 for(var j=group.x;j<listed;j+=groups.x){
  if(lane==0u){prefixTileKey=atomicLoad(&bins[ORDER_OCCUPIED+4u+j]);}
  let tile=workgroupUniformLoad(&prefixTileKey);
  if(umTileLive(tile)){
   let cell=64u*tile+lane;let count=atomicLoad(&bins[cell]);prefix[lane]=count;workgroupBarrier();
   for(var stride=1u;stride<64u;stride*=2u){
    var add=0u;if(lane>=stride){add=prefix[lane-stride];}
    workgroupBarrier();prefix[lane]+=add;workgroupBarrier();
   }
   links[cell]=orderTileStart(tile)+prefix[lane]-count;
   atomicStore(&bins[UM_CELL_KEYS()+cell],0u);
  }
  workgroupBarrier();
 }
}
`:""}
// The key of a stored cell, or 0xffffffff: a sample whose cell holds no
// storage (its arrival was fatal) is in no count and joins no run.
fn orderCellKey(c:vec3u)->u32{
 let t=umTileKey(vec3i(c>>vec3u(2u)));if(!umKeyStored(t)){return 0xffffffffu;}
 return 64u*t+((c.x&3u)|((c.y&3u)<<2u)|((c.z&3u)<<4u));
}
var<workgroup> packKeys:array<u32,64>;
var<workgroup> packStarts:array<u32,64>;
var<workgroup> packCount:u32;
@compute @workgroup_size(64) fn pack(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let job=group.x+groups.x*group.y;
 if(lane==0u){packCount=min(state[0],arrayLength(&input));}
 let count=workgroupUniformLoad(&packCount);
 if(job==0u&&lane==0u){state[1]=${this.sparse?"orderTileStart(orderTiles()-1u)+atomicLoad(&bins[2u*UM_CELL_KEYS()+orderTiles()-1u])":"links[UM_CELL_KEYS()-1u]+atomicLoad(&bins[UM_CELL_KEYS()-1u])"};}
 for(var base=64u*job;base<count;base+=64u*groups.x*groups.y){
  let i=base+lane;var position=vec4f(-1.0);var key=0xffffffffu;
  if(i<count){position=input[i].position;if(position.x>=0.0){
   key=orderCellKey(vec3u(position.xyz));
  }}
  packKeys[lane]=key;workgroupBarrier();
  if(key!=0xffffffffu&&(lane==0u||packKeys[lane-1u]!=key)){
   var end=lane+1u;while(end<64u&&packKeys[end]==key){end++;}
   packStarts[lane]=links[key]+atomicAdd(&bins[UM_CELL_KEYS()+key],end-lane);
  }
  workgroupBarrier();
  if(key!=0xffffffffu){
   var first=lane;while(first>0u&&packKeys[first-1u]==key){first--;}
   let next=packStarts[first]+lane-first;
   ${this.canonical?`// A sort key, not yet the record: position and source slot.
   let a=UM_CELL_KEYS()+4u*next;let bits=bitcast<vec4u>(position);
   links[a]=bits.x;links[a+1u]=bits.y;links[a+2u]=bits.z;links[a+3u]=i;`:`// The record itself: its cell's run in cursor order.
   output[next]=input[i];`}
  }
  workgroupBarrier();
 }
}
// Atomic scatter chooses slots, not a floating-point summation order. Give
// each cell a canonical order before reconstruction, transfer and retention.
// Positions decide almost always; an exact position tie compares every
// stored component, and records equal in every bit are interchangeable, so
// their source slots only make the ranks distinct.
fn orderCompare(a:Particle,b:Particle)->i32{
 ${["position","velocity",...(this.apic?["before","cx","cy","cz","lifetime"]:[])].map(field=>`{let x=bitcast<vec4u>(a.${field});let y=bitcast<vec4u>(b.${field});for(var k=0u;k<4u;k++){if(x[k]!=y[k]){return select(1,-1,x[k]<y[k]);}}}`).join("\n ")}
 ${this.apic?"":`{let x=bitcast<vec4u>(vec4f(a.before.xyz,nbParticleMode(a)));let y=bitcast<vec4u>(vec4f(b.before.xyz,nbParticleMode(b)));for(var k=0u;k<4u;k++){if(x[k]!=y[k]){return select(1,-1,x[k]<y[k]);}}}
 {let x=bitcast<u32>(nbParticleAge(a));let y=bitcast<u32>(nbParticleAge(b));if(x!=y){return select(1,-1,x<y);}}`}
 return 0;
}
fn orderBefore(a:vec4u,b:vec4u)->bool{
 for(var k=0u;k<3u;k++){if(a[k]!=b[k]){return a[k]<b[k];}}
 let order=orderCompare(input[a.w],input[b.w]);
 return order<0||(order==0&&a.w<b.w);
}
fn orderKey(slot:u32)->vec4u{let a=UM_CELL_KEYS()+4u*slot;return vec4u(links[a],links[a+1u],links[a+2u],links[a+3u]);}
// The former quadratic rank is retained only for diagnostic ablation;
// production uses the same run-sort/merge pipeline for every cell.
fn orderRank(s:u32){
  let own=orderKey(s);let cell=orderCellKey(vec3u(bitcast<vec3f>(own.xyz)));
  let start=links[cell];let end=start+atomicLoad(&bins[cell]);
  var rank=0u;var tie=false;
  for(var k=start;k<end;k++){let o=orderKey(k).xyz;
   rank+=select(0u,1u,o.x<own.x||(o.x==own.x&&(o.y<own.y||(o.y==own.y&&o.z<own.z))));tie=tie||(k!=s&&all(o==own.xyz));}
  if(tie){for(var k=start;k<end;k++){let o=orderKey(k);if(k!=s&&all(o.xyz==own.xyz)&&orderBefore(o,own)){rank++;}}}
  output[start+rank]=input[own.w];
}
@compute @workgroup_size(64) fn canonicalLegacy(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let live=min(state[1],arrayLength(&output));
 for(var s=64u*(group.x+groups.x*group.y)+lane;s<live;s+=64u*groups.x*groups.y){orderRank(s);}
}
// One uniform run-sort for all cells: no particle compares more than 128
// keys, regardless of its cell's population. Completed runs publish their
// records; longer cells continue through the same parallel merge pipeline.
@compute @workgroup_size(64) fn canonical(@builtin(global_invocation_id) id:vec3u,@builtin(num_workgroups) groups:vec3u){
 let live=min(state[1],arrayLength(&output));
 for(var slot=id.x;slot<live;slot+=64u*groups.x){
  let own=orderKey(slot);let cell=orderCellKey(vec3u(bitcast<vec3f>(own.xyz)));
  let start=links[cell];let count=atomicLoad(&bins[cell]);let run=start+((slot-start)/128u)*128u;
  // Build merge jobs where the cell metadata is already in registers.
  // Keeping this out of pack avoids slowing its bandwidth-bound scatter.
  if((slot-start)%256u==0u){for(var level=0u;level<ORDER_MERGE_LEVELS;level++){
   if(count<=128u*(1u<<level)){break;}
   let header=4u*(level+1u);let merge=atomicAdd(&orderScratch.counts[header+3u],1u);
   let at=ORDER_MERGE_QUEUES[level]-ORDER_HEADER_WORDS+2u*merge;orderScratch.words[at]=slot;orderScratch.words[at+1u]=cell;
  }}
  var rank=0u;var tie=false;let end=min(run+128u,start+count);
  for(var k=run;k<end;k++){let other=orderKey(k).xyz;
   rank+=select(0u,1u,other.x<own.x||(other.x==own.x&&(other.y<own.y||(other.y==own.y&&other.z<own.z))));
   tie=tie||(k!=slot&&all(other==own.xyz));
  }
  if(tie){for(var k=run;k<end;k++){let other=orderKey(k);if(k!=slot&&all(other.xyz==own.xyz)&&orderBefore(other,own)){rank++;}}}
  if(count<=128u){output[run+rank]=input[own.w];}else{mergeStore(run+rank,1u,own);}
 }
}
${narrowBandMergeWGSL(this.scratchKeys,this.headerWords,this.mergeQueues)}
`});
  const layout=this.device.createBindGroupLayout({entries:[{binding:7,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},...(this.sparse?[0,1,2,3,4,6,5,8]:[0,1,2,3,4,6,8]).map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===0||binding===5||binding===6?"read-only-storage" as const:"storage" as const}}))]});
  const pipelineLayout=this.device.createPipelineLayout({bindGroupLayouts:[layout]});
  const canonical=this.canonical?["canonical","canonicalLegacy"]:[];
  const pipelines=await Promise.all([...(this.sparse?["pack","prepare","prefixTile"]:["pack"]),...canonical].map(entryPoint=>this.device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint}})));
  [this.pipeline]=pipelines;if(this.sparse)[this.preparePipeline,this.prefixPipeline]=pipelines.slice(1);
  if(this.canonical)[this.canonicalPipeline,this.legacyPipeline]=pipelines.slice(-2);
  this.mergePipelines=!this.canonical?[]:await Promise.all(Array.from({length:this.mergeLevels},(_,level)=>this.device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint:"canonicalMerge",constants:{ORDER_MERGE_WIDTH:128*2**level,ORDER_MERGE_BANK:1-level%2,ORDER_MERGE_LEVEL:level}}})));
  this.groups=[0,1].map(parity=>this.device.createBindGroup({layout,entries:[{binding:8,resource:{buffer:this.orderScratch}},{binding:7,resource:{buffer:this.dimensions}},{binding:6,resource:{buffer:this.topology}},...[this.particles[1-parity]!,this.particles[parity]!,this.bins,this.links,this.state,...(this.sparse?[this.tileOffsets!]:[])].map((buffer,binding)=>({binding,resource:{buffer}}))]})) as [GPUBindGroup,GPUBindGroup];
 }
 prepare(encoder:GPUCommandEncoder,parity:number):void{
  if(!this.sparse)encoder.clearBuffer(this.bins,0,this.cells*8+4*this.address.tileKeys);
  else{const pass=encoder.beginComputePass(PREPARE_PASS);pass.setPipeline(this.preparePipeline);pass.setBindGroup(0,this.groups[parity]!);pass.dispatchWorkgroups(Math.min(128,this.workgroups));pass.end();}
  encoder.clearBuffer(this.bins,4*this.occupied,4);
 }
 /** Tile-total scan followed by cell offsets only for occupied tiles. */
 encodeOffsets(encoder:GPUCommandEncoder,parity:number):void{
  if(this.sparse){
   const prefix=encoder.beginComputePass(PREFIX_PASS);this.scan.encode(encoder,true,prefix);
   // This domain contains tiles, not particles. A crowded tile must not
   // inflate its setup grid; each group strides the complete occupied list.
   prefix.setPipeline(this.prefixPipeline);prefix.setBindGroup(0,this.groups[parity]!);prefix.dispatchWorkgroups(Math.min(128,this.workgroups));prefix.end();
  }else this.scan.encode(encoder);
 }
 encode(encoder:GPUCommandEncoder,parity:number):void{
  if(this.canonical&&!this.legacyCanonical){encoder.clearBuffer(this.orderScratch,0,16*(this.mergeLevels+1));}
  this.encodeOffsets(encoder,parity);
  const pass=encoder.beginComputePass(PACK_PASS);pass.setPipeline(this.pipeline);pass.setBindGroup(0,this.groups[parity]!);pass.dispatchWorkgroups(this.workgroups);pass.end();
  if(!this.canonical)return;
  const canonical=encoder.beginComputePass(CANONICAL_PASS);canonical.setPipeline(this.legacyCanonical?this.legacyPipeline:this.canonicalPipeline);canonical.setBindGroup(0,this.groups[parity]!);canonical.dispatchWorkgroups(this.workgroups);canonical.end();
  if(!this.legacyCanonical){const crowded=encoder.beginComputePass(CROWDED_PASS);crowded.setBindGroup(0,this.groups[parity]!);for(let level=0;level<this.mergePipelines.length;level++){crowded.setPipeline(this.mergePipelines[level]);crowded.dispatchWorkgroups(Math.min(256,this.workgroups));}crowded.end();}
 }
 destroy():void{this.orderScratch.destroy();this.scan?.destroy();this.enabled.destroy();this.dimensions.destroy();this.tileOffsets?.destroy();}
}
