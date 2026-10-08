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
 private scan!:ParticleBinScan;
 private pipeline!:GPUComputePipeline;
 private groups!:readonly [GPUBindGroup,GPUBindGroup];
 get allocatedBytes():number{return this.enabled.size+(this.scan?.allocatedBytes??0);}
 constructor(private readonly device:GPUDevice,private readonly dims:readonly number[],private readonly particles:readonly [GPUBuffer,GPUBuffer],private readonly bins:GPUBuffer,private readonly links:GPUBuffer,private readonly state:GPUBuffer){
  // Shared APIC scan's activation record: no failure and an active step.
  // This immutable record makes our sort unconditional, including bootstrap.
  this.enabled=device.createBuffer({label:"FLIP spatial order scan activation",size:88,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  const activation=new Float32Array(22);activation[21]=1;device.queue.writeBuffer(this.enabled,0,activation);
 }
 async initialize():Promise<void>{
  const cells=this.dims.reduce((n,v)=>n*v,1);
  this.scan=await ParticleBinScan.create(this.device,cells,{buffer:this.bins,offset:0,size:4*cells},{buffer:this.links,offset:0,size:4*cells},this.enabled);
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
const CELLS:u32=${cells}u;
@compute @workgroup_size(64) fn pack(@builtin(global_invocation_id) gid:vec3u){
 let dims=vec3u(${this.dims.map(n=>`${n}u`).join(',')});let tiles=dims/4u;
 if(gid.x==0u){state[1]=links[CELLS-1u]+atomicLoad(&bins[CELLS-1u]);}
 for(var i=gid.x;i<min(state[0],arrayLength(&input));i+=65536u){
  let p=input[i];if(p.position.x<0.0){continue;}
  let c=vec3u(p.position.xyz);let t=c/4u;let l=c%4u;
  let order=64u*(t.x+tiles.x*(t.y+tiles.y*t.z))+l.x+4u*l.y+16u*l.z;
  let next=links[order]+atomicAdd(&bins[CELLS+order],1u);output[next]=p;
  let a=CELLS+4u*next;let position=bitcast<vec4u>(p.position);
  links[a]=position.x;links[a+1u]=position.y;links[a+2u]=position.z;links[a+3u]=position.w;
 }
}`});
  const layout=this.device.createBindGroupLayout({entries:[0,1,2,3,4].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===0?"read-only-storage" as const:"storage" as const}}))});
  this.pipeline=await this.device.createComputePipelineAsync({layout:this.device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint:"pack"}});
  this.groups=[0,1].map(parity=>this.device.createBindGroup({layout,entries:[this.particles[1-parity]!,this.particles[parity]!,this.bins,this.links,this.state].map((buffer,binding)=>({binding,resource:{buffer}}))})) as [GPUBindGroup,GPUBindGroup];
 }
 /** The bins hold the counts of the samples to pack and a clear cursor. */
 encode(encoder:GPUCommandEncoder,parity:number):void{
  this.scan.encode(encoder);
  const pass=encoder.beginComputePass({label:"Narrow-band FLIP order pack"});pass.setPipeline(this.pipeline);pass.setBindGroup(0,this.groups[parity]!);pass.dispatchWorkgroups(1024);pass.end();
 }
 destroy():void{this.scan?.destroy();this.enabled.destroy();}
}
