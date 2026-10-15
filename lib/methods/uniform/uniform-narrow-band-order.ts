import { narrowBandParticleWGSL } from "./uniform-narrow-band-apic.wgsl";
import { uniformWindowAddressWGSL } from "./uniform-window-address";
import { ParticleBinScan } from "./particle-bin-scan";
import { UNIFORM_PAGE_TABLE_WORDS, uniformKeyGroups, uniformLatticeAddressWGSL, type UniformLatticeAddress } from "./uniform-lattice-address";

/** The stage's pass descriptors: begun every frame, built once. */
const PREPARE_PASS:GPUComputePassDescriptor={label:"Narrow-band FLIP clear occupied bins"};
const PREFIX_PASS:GPUComputePassDescriptor={label:"Narrow-band FLIP occupied cell offsets"},PACK_PASS:GPUComputePassDescriptor={label:"Narrow-band FLIP order pack"};
const CANONICAL_PASS:GPUComputePassDescriptor={label:"Narrow-band FLIP canonical cell order"},POSITIONS_PASS:GPUComputePassDescriptor={label:"Narrow-band FLIP sorted positions"};
/** Workgroups of every launch over the live samples, one row. A kernel reads
 * the count on the GPU and strides the launch through it, 64 samples a group
 * a turn, so any width is complete at any count up to the capacity. This one
 * is the ownership's counted launches' (262,144 lanes): enough to fill the
 * GPU, few enough that a launch with little live stays cheap. Letters, the
 * five passes that hold these launches, against a group a sample block: 1.28
 * ms for 1.30 under half a million samples (most groups idle), 5.62 for 5.58
 * over 1.25 million (five turns a group; the snapshot alone 0.85 for 0.74). */
export const NARROW_BAND_PARTICLE_GRID=4096;

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
 * then ranks itself within its cell and moves its record once, so the
 * order does not depend on GPU scheduling. */
export class UniformNarrowBandOrder {
 private readonly enabled:GPUBuffer;
 private readonly dimensions:GPUBuffer;
 private readonly tileOffsets?:GPUBuffer;
 private preparePipeline!:GPUComputePipeline;
 private prefixPipeline!:GPUComputePipeline;
 private scan!:ParticleBinScan;
 private pipeline!:GPUComputePipeline;
 private canonicalPipeline!:GPUComputePipeline;
 private positionsPipeline!:GPUComputePipeline;
 private groups!:readonly [GPUBindGroup,GPUBindGroup];
 /** Cell keys of the lattice: the bins' first two banks. */
 private readonly cells:number;
 /** The per-tile launches' grid: the lattice's, for life. */
 private readonly tileGroups:[number,number]=[0,0];
 get allocatedBytes():number{return this.enabled.size+this.dimensions.size+(this.tileOffsets?.size??0)+(this.scan?.allocatedBytes??0);}
 /** topology: the simulation ownership's, whose tail holds the pages. The
  * optional policy override lets small fixtures exercise both paths. */
 constructor(private readonly device:GPUDevice,private readonly address:UniformLatticeAddress,private readonly topology:GPUBuffer,private readonly particles:readonly [GPUBuffer,GPUBuffer],private readonly bins:GPUBuffer,private readonly links:GPUBuffer,private readonly state:GPUBuffer,readonly sparse=address.cellKeys>=4_194_304,private readonly apic=false){
  this.cells=address.cellKeys;uniformKeyGroups(address.tileKeys,1,65535,this.tileGroups);
  // Shared APIC scan's activation record: no failure and an active step.
  // This immutable record makes our sort unconditional, including bootstrap.
  this.dimensions=device.createBuffer({label:"FLIP spatial order dimensions",size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(this.dimensions,0,new Uint32Array([...address.cells,address.blocks]));
  this.enabled=device.createBuffer({label:"FLIP spatial order scan activation",size:88,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  if(this.sparse)this.tileOffsets=device.createBuffer({label:"FLIP tile offsets",size:4*address.tileKeys,usage:GPUBufferUsage.STORAGE});
  const activation=new Float32Array(22);activation[21]=1;device.queue.writeBuffer(this.enabled,0,activation);
 }
 async initialize():Promise<void>{
  const cells=this.address.cellKeys,tiles=this.address.tileKeys;
  // Extra tile counting pays for itself only on large domains. The dense
  // path retains the original scan and clears, without tile storage/atomics.
  this.scan=await ParticleBinScan.create(this.device,this.sparse?tiles:cells,
   {buffer:this.bins,offset:this.sparse?8*cells:0,size:this.sparse?4*tiles:4*cells},
   this.sparse?this.tileOffsets!:{buffer:this.links,offset:0,size:4*cells},this.enabled);
  const module=this.device.createShaderModule({label:"FLIP spatial order",code:/* wgsl */`
${narrowBandParticleWGSL(this.apic)}
@group(0) @binding(7) var<uniform> orderDimensions:vec4u;
// The box and the blocks are this module's uniform, the pages the
// topology's tail: the address functions read them.
@group(0) @binding(6) var<storage,read> orderTopology:array<u32>;
fn orderTiles()->u32{return 512u*orderDimensions.w;}
${uniformWindowAddressWGSL("orderTopology","5u*orderTiles()","p.blocks.x")}
${uniformLatticeAddressWGSL({constant:false,paged:{blocks:"orderDimensions.w"}})}
@group(0) @binding(0) var<storage,read> input:array<Particle>;
@group(0) @binding(1) var<storage,read_write> output:array<Particle>;
// Each cell's count, then its scatter cursor.
@group(0) @binding(2) var<storage,read_write> bins:array<atomic<u32>>;
// Each cell's first sample, then the compact positions the gathers read.
@group(0) @binding(3) var<storage,read_write> links:array<u32>;
// The receipt: [0] the input's slots, [1] the live samples packed.
@group(0) @binding(4) var<storage,read_write> state:array<u32>;
${this.sparse?"@group(0) @binding(5) var<storage,read> tileOffsets:array<u32>;":""}
${this.sparse?/* wgsl */`
var<workgroup> prefix:array<u32,64>;
var<workgroup> tilePopulation:u32;
// Only tiles populated by the previous move contain cell counts to clear.
// Every tile key is visited, one a page holds or not: the scan sums them all.
@compute @workgroup_size(64) fn prepare(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let tile=group.x+65535u*group.y;if(tile>=orderTiles()){return;}
 let live=atomicLoad(&bins[2u*UM_CELL_KEYS()+tile]);
 if(live!=0u){atomicStore(&bins[64u*tile+lane],0u);atomicStore(&bins[UM_CELL_KEYS()+64u*tile+lane],0u);}
 workgroupBarrier();
 if(lane==0u){atomicStore(&bins[2u*UM_CELL_KEYS()+tile],0u);}
}
@compute @workgroup_size(64) fn prefixTile(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let tile=group.x+65535u*group.y;if(tile>=orderTiles()){return;}
 if(tile==orderTiles()-1u&&lane==0u){state[1]=tileOffsets[tile]+atomicLoad(&bins[2u*UM_CELL_KEYS()+tile]);}
 if(!umTileLive(tile)){return;}
 if(lane==0u){tilePopulation=atomicLoad(&bins[2u*UM_CELL_KEYS()+tile]);}
 if(workgroupUniformLoad(&tilePopulation)==0u){return;}
 let cell=64u*tile+lane;let count=atomicLoad(&bins[cell]);prefix[lane]=count;workgroupBarrier();
 for(var stride=1u;stride<64u;stride*=2u){
  var add=0u;if(lane>=stride){add=prefix[lane-stride];}
  workgroupBarrier();prefix[lane]+=add;workgroupBarrier();
 }
 links[cell]=tileOffsets[tile]+prefix[lane]-count;
 atomicStore(&bins[UM_CELL_KEYS()+cell],0u);
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
 if(job==0u&&lane==0u){state[1]=${this.sparse?"tileOffsets[orderTiles()-1u]+atomicLoad(&bins[2u*UM_CELL_KEYS()+orderTiles()-1u])":"links[UM_CELL_KEYS()-1u]+atomicLoad(&bins[UM_CELL_KEYS()-1u])"};}
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
   // A sort key, not yet the record: position and source slot.
   let next=packStarts[first]+lane-first;
   let a=UM_CELL_KEYS()+4u*next;let bits=bitcast<vec4u>(position);
   links[a]=bits.x;links[a+1u]=bits.y;links[a+2u]=bits.z;links[a+3u]=i;
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
 ${["position","velocity","before",...(this.apic?["cx","cy","cz"]:[]),"lifetime"].map(field=>`{let x=bitcast<vec4u>(a.${field});let y=bitcast<vec4u>(b.${field});for(var k=0u;k<4u;k++){if(x[k]!=y[k]){return select(1,-1,x[k]<y[k]);}}}`).join("\n ")}
 return 0;
}
fn orderBefore(a:vec4u,b:vec4u)->bool{
 for(var k=0u;k<3u;k++){if(a[k]!=b[k]){return a[k]<b[k];}}
 let order=orderCompare(input[a.w],input[b.w]);
 return order<0||(order==0&&a.w<b.w);
}
fn orderKey(slot:u32)->vec4u{let a=UM_CELL_KEYS()+4u*slot;return vec4u(links[a],links[a+1u],links[a+2u],links[a+3u]);}
// One lane a sample: its rank among its cell's keys is its slot in the run.
// Work is linear in the cell's population a lane, all lanes in parallel.
@compute @workgroup_size(64) fn canonical(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let live=min(state[1],arrayLength(&output));
 for(var s=64u*(group.x+groups.x*group.y)+lane;s<live;s+=64u*groups.x*groups.y){
  let own=orderKey(s);let cell=orderCellKey(vec3u(bitcast<vec3f>(own.xyz)));
  let start=links[cell];let end=start+atomicLoad(&bins[cell]);
  var rank=0u;
  for(var k=start;k<end;k++){if(k!=s&&orderBefore(orderKey(k),own)){rank++;}}
  output[start+rank]=input[own.w];
 }
}
// The gathers read sorted positions from the links.
@compute @workgroup_size(64) fn positions(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let live=min(state[1],arrayLength(&output));
 for(var s=64u*(group.x+groups.x*group.y)+lane;s<live;s+=64u*groups.x*groups.y){
  let a=UM_CELL_KEYS()+4u*s;let p=bitcast<vec4u>(output[s].position);
  links[a]=p.x;links[a+1u]=p.y;links[a+2u]=p.z;links[a+3u]=p.w;
 }
}
`});
  const layout=this.device.createBindGroupLayout({entries:[{binding:7,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},...(this.sparse?[0,1,2,3,4,6,5]:[0,1,2,3,4,6]).map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===0||binding>=5?"read-only-storage" as const:"storage" as const}}))]});
  const pipelineLayout=this.device.createPipelineLayout({bindGroupLayouts:[layout]});
  [this.pipeline,this.canonicalPipeline,this.positionsPipeline,this.preparePipeline,this.prefixPipeline]=await Promise.all((this.sparse?["pack","canonical","positions","prepare","prefixTile"]:["pack","canonical","positions"]).map(entryPoint=>this.device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint}})));
  this.groups=[0,1].map(parity=>this.device.createBindGroup({layout,entries:[{binding:7,resource:{buffer:this.dimensions}},{binding:6,resource:{buffer:this.topology}},...[this.particles[1-parity]!,this.particles[parity]!,this.bins,this.links,this.state,...(this.sparse?[this.tileOffsets!]:[])].map((buffer,binding)=>({binding,resource:{buffer}}))]})) as [GPUBindGroup,GPUBindGroup];
 }
 prepare(encoder:GPUCommandEncoder,parity:number):void{
  if(!this.sparse){encoder.clearBuffer(this.bins,0,this.cells*8);return;}
  const pass=encoder.beginComputePass(PREPARE_PASS);pass.setPipeline(this.preparePipeline);pass.setBindGroup(0,this.groups[parity]!);pass.dispatchWorkgroups(this.tileGroups[0],this.tileGroups[1]);pass.end();
 }
 /** Dense scan for small domains; occupied-tile cell scans for large ones. */
 encodeOffsets(encoder:GPUCommandEncoder,parity:number):void{
  this.scan.encode(encoder);
  if(this.sparse){
   const prefix=encoder.beginComputePass(PREFIX_PASS);prefix.setPipeline(this.prefixPipeline);prefix.setBindGroup(0,this.groups[parity]!);prefix.dispatchWorkgroups(this.tileGroups[0],this.tileGroups[1]);prefix.end();
  }
 }
 encode(encoder:GPUCommandEncoder,parity:number):void{
  this.encodeOffsets(encoder,parity);
  const pass=encoder.beginComputePass(PACK_PASS);pass.setPipeline(this.pipeline);pass.setBindGroup(0,this.groups[parity]!);pass.dispatchWorkgroups(NARROW_BAND_PARTICLE_GRID);pass.end();
  const canonical=encoder.beginComputePass(CANONICAL_PASS);canonical.setPipeline(this.canonicalPipeline);canonical.setBindGroup(0,this.groups[parity]!);canonical.dispatchWorkgroups(NARROW_BAND_PARTICLE_GRID);canonical.end();
  const positions=encoder.beginComputePass(POSITIONS_PASS);positions.setPipeline(this.positionsPipeline);positions.setBindGroup(0,this.groups[parity]!);positions.dispatchWorkgroups(NARROW_BAND_PARTICLE_GRID);positions.end();
 }
 destroy():void{this.scan?.destroy();this.enabled.destroy();this.dimensions.destroy();this.tileOffsets?.destroy();}
}
