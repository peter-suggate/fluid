import { ParticleBinScan } from "../particle/scan";

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
 * cells. Particles scatter in parallel through a cursor in the bins' second
 * bank so a crowded bin cannot serialize an entire workgroup. */
export class UniformNarrowBandOrder {
 private readonly enabled:GPUBuffer;
 private readonly tileOffsets?:GPUBuffer;
 private readonly spatialCounts?:GPUBuffer;
 private spatialPipeline?:GPUComputePipeline;
 private preparePipeline!:GPUComputePipeline;
 private prefixPipeline!:GPUComputePipeline;
 private scan!:ParticleBinScan;
 private pipeline!:GPUComputePipeline;
 private groups!:readonly [GPUBindGroup,GPUBindGroup];
 get allocatedBytes():number{return this.enabled.size+(this.spatialCounts?.size??0)+(this.tileOffsets?.size??0)+(this.scan?.allocatedBytes??0);}
 /** The optional policy override lets small fixtures exercise both paths. */
 constructor(private readonly device:GPUDevice,private readonly dims:readonly number[],private readonly particles:readonly [GPUBuffer,GPUBuffer],private readonly bins:GPUBuffer,private readonly links:GPUBuffer,private readonly state:GPUBuffer,readonly sparse=dims.reduce((n,v)=>n*v,1)>=4_194_304,readonly spatialOrder=sparse&&dims.every(n=>n%16===0)){
  // Shared APIC scan's activation record: no failure and an active step.
  // This immutable record makes our sort unconditional, including bootstrap.
  this.enabled=device.createBuffer({label:"FLIP spatial order scan activation",size:88,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  if(this.sparse)this.tileOffsets=device.createBuffer({label:"FLIP tile offsets",size:dims.reduce((n,v)=>n*v,1)/16,usage:GPUBufferUsage.STORAGE});
  if(spatialOrder)this.spatialCounts=device.createBuffer({label:"FLIP spatial tile counts",size:dims.reduce((n,v)=>n*v,1)/16,usage:GPUBufferUsage.STORAGE});
  const activation=new Float32Array(22);activation[21]=1;device.queue.writeBuffer(this.enabled,0,activation);
 }
 async initialize():Promise<void>{
  const cells=this.dims.reduce((n,v)=>n*v,1);
  // Extra tile counting pays for itself only on large domains. The dense
  // path retains the original scan and clears, without tile storage/atomics.
  this.scan=await ParticleBinScan.create(this.device,this.sparse?cells/64:cells,
   this.spatialCounts??{buffer:this.bins,offset:this.sparse?8*cells:0,size:this.sparse?cells/16:4*cells},
   this.sparse?this.tileOffsets!:{buffer:this.links,offset:0,size:4*cells},this.enabled);
  const module=this.device.createShaderModule({label:"FLIP spatial order",code:/* wgsl */`
struct Particle{position:vec4f,velocity:vec4f,before:vec4f}
@group(0) @binding(0) var<storage,read> input:array<Particle>;
@group(0) @binding(1) var<storage,read_write> output:array<Particle>;
// Each cell's count, then its scatter cursor.
@group(0) @binding(2) var<storage,read_write> bins:array<atomic<u32>>;
// Each cell's first sample, then the compact positions the gathers read.
@group(0) @binding(3) var<storage,read_write> links:array<u32>;
// The receipt: [0] the input's slots, [1] the live samples packed.
@group(0) @binding(4) var<storage,read_write> state:array<u32>;
${this.sparse?"@group(0) @binding(5) var<storage,read> tileOffsets:array<u32>;":""}
${this.spatialOrder?`@group(0) @binding(6) var<storage,read_write> spatialCounts:array<u32>;
fn spatialRank(tile:u32)->u32{
 let dims=vec3u(${this.dims.map(n=>`${n/4}u`).join(',')});
 let t=vec3u(tile%dims.x,(tile/dims.x)%dims.y,tile/(dims.x*dims.y));let b=t/4u;let l=t%4u;
 let k=(l.x&1u)|((l.y&1u)<<1u)|((l.z&1u)<<2u)|((l.x&2u)<<2u)|((l.y&2u)<<3u)|((l.z&2u)<<4u);
 return 64u*(b.x+(dims.x/4u)*(b.y+(dims.y/4u)*b.z))+k;
}
@compute @workgroup_size(64) fn countTiles(@builtin(global_invocation_id) gid:vec3u){
 let tile=gid.x;if(tile<TILES){spatialCounts[spatialRank(tile)]=atomicLoad(&bins[2u*CELLS+tile]);}
}`:""}
const CELLS:u32=${cells}u;
const TILES:u32=CELLS/64u;
${this.sparse?/* wgsl */`
var<workgroup> prefix:array<u32,64>;
var<workgroup> tilePopulation:u32;
// Only tiles populated by the previous move contain cell counts to clear.
@compute @workgroup_size(64) fn prepare(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let tile=group.x+65535u*group.y;if(tile>=TILES){return;}
 let live=atomicLoad(&bins[2u*CELLS+tile]);
 if(live!=0u){atomicStore(&bins[64u*tile+lane],0u);atomicStore(&bins[CELLS+64u*tile+lane],0u);}
 workgroupBarrier();
 if(lane==0u){atomicStore(&bins[2u*CELLS+tile],0u);}
}
@compute @workgroup_size(64) fn prefixTile(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let tile=group.x+65535u*group.y;if(tile>=TILES){return;}
 if(lane==0u){tilePopulation=atomicLoad(&bins[2u*CELLS+tile]);}
 if(workgroupUniformLoad(&tilePopulation)==0u){return;}
 let cell=64u*tile+lane;let count=atomicLoad(&bins[cell]);prefix[lane]=count;workgroupBarrier();
 for(var stride=1u;stride<64u;stride*=2u){
  var add=0u;if(lane>=stride){add=prefix[lane-stride];}
  workgroupBarrier();prefix[lane]+=add;workgroupBarrier();
 }
 links[cell]=tileOffsets[${this.spatialOrder?"spatialRank(tile)":"tile"}]+prefix[lane]-count;
 atomicStore(&bins[CELLS+cell],0u);
}
`:""}
var<workgroup> packKeys:array<u32,64>;
var<workgroup> packStarts:array<u32,64>;
var<workgroup> packCount:u32;
@compute @workgroup_size(64) fn pack(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let dims=vec3u(${this.dims.map(n=>`${n}u`).join(',')});let tiles=dims/4u;
 let job=group.x+groups.x*group.y;
 if(lane==0u){packCount=min(state[0],arrayLength(&input));}
 let count=workgroupUniformLoad(&packCount);
 if(job==0u&&lane==0u){state[1]=${this.sparse?`tileOffsets[TILES-1u]+${this.spatialOrder?"spatialCounts[TILES-1u]":"atomicLoad(&bins[2u*CELLS+TILES-1u])"}`:"links[CELLS-1u]+atomicLoad(&bins[CELLS-1u])"};}
 for(var base=64u*job;base<count;base+=64u*groups.x*groups.y){
  let i=base+lane;var p=Particle();var key=0xffffffffu;
  if(i<count){p=input[i];if(p.position.x>=0.0){
   let c=vec3u(p.position.xyz);let t=c/4u;let l=c%4u;
   key=64u*(t.x+tiles.x*(t.y+tiles.y*t.z))+l.x+4u*l.y+16u*l.z;
  }}
  packKeys[lane]=key;workgroupBarrier();
  if(key!=0xffffffffu&&(lane==0u||packKeys[lane-1u]!=key)){
   var end=lane+1u;while(end<64u&&packKeys[end]==key){end++;}
   packStarts[lane]=links[key]+atomicAdd(&bins[CELLS+key],end-lane);
  }
  workgroupBarrier();
  if(key!=0xffffffffu){
   var first=lane;while(first>0u&&packKeys[first-1u]==key){first--;}
   let next=packStarts[first]+lane-first;output[next]=p;
   let a=CELLS+4u*next;let position=bitcast<vec4u>(p.position);
   links[a]=position.x;links[a+1u]=position.y;links[a+2u]=position.z;links[a+3u]=position.w;
  }
  workgroupBarrier();
 }
}
`});
  const layout=this.device.createBindGroupLayout({entries:(this.spatialOrder?[0,1,2,3,4,5,6]:this.sparse?[0,1,2,3,4,5]:[0,1,2,3,4]).map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===0||binding===5?"read-only-storage" as const:"storage" as const}}))});
  const pipelineLayout=this.device.createPipelineLayout({bindGroupLayouts:[layout]});
  [this.pipeline,this.preparePipeline,this.prefixPipeline]=await Promise.all((this.sparse?["pack","prepare","prefixTile"]:["pack"]).map(entryPoint=>this.device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint}})));
  if(this.spatialOrder)this.spatialPipeline=await this.device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint:"countTiles"}});
  this.groups=[0,1].map(parity=>this.device.createBindGroup({layout,entries:[this.particles[1-parity]!,this.particles[parity]!,this.bins,this.links,this.state,...(this.sparse?[this.tileOffsets!]:[]),...(this.spatialOrder?[this.spatialCounts!]:[])].map((buffer,binding)=>({binding,resource:{buffer}}))})) as [GPUBindGroup,GPUBindGroup];
 }
 prepare(encoder:GPUCommandEncoder,parity:number):void{
  if(!this.sparse){encoder.clearBuffer(this.bins,0,this.dims.reduce((n,v)=>n*v,1)*8);return;}
  const tiles=this.dims.reduce((n,v)=>n*v,1)/64;
  const pass=encoder.beginComputePass({label:"Narrow-band FLIP clear occupied bins"});pass.setPipeline(this.preparePipeline);pass.setBindGroup(0,this.groups[parity]!);pass.dispatchWorkgroups(Math.min(tiles,65535),Math.ceil(tiles/65535));pass.end();
 }
 /** Dense scan for small domains; occupied-tile cell scans for large ones. */
 encode(encoder:GPUCommandEncoder,parity:number,work?:GPUBuffer):void{
  if(this.spatialOrder){const pass=encoder.beginComputePass({label:"Narrow-band FLIP spatial tile counts"});pass.setPipeline(this.spatialPipeline!);pass.setBindGroup(0,this.groups[parity]!);pass.dispatchWorkgroups(Math.ceil(this.dims.reduce((n,v)=>n*v,1)/4096));pass.end();}
  this.scan.encode(encoder);
  if(this.sparse){
   const tiles=this.dims.reduce((n,v)=>n*v,1)/64;
   const prefix=encoder.beginComputePass({label:"Narrow-band FLIP occupied cell offsets"});prefix.setPipeline(this.prefixPipeline);prefix.setBindGroup(0,this.groups[parity]!);prefix.dispatchWorkgroups(Math.min(tiles,65535),Math.ceil(tiles/65535));prefix.end();
  }
  const pass=encoder.beginComputePass({label:"Narrow-band FLIP order pack"});pass.setPipeline(this.pipeline);pass.setBindGroup(0,this.groups[parity]!);if(work)pass.dispatchWorkgroupsIndirect(work,0);else pass.dispatchWorkgroups(1024);pass.end();
 }
 destroy():void{this.scan?.destroy();this.enabled.destroy();this.tileOffsets?.destroy();this.spatialCounts?.destroy();}
}
