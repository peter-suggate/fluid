import { ParticleBinScan } from "../particle/scan";

/** Reorder the live particles by cell without a bucket limit or another
 * particle allocation. The expired source epoch becomes the sorted output.
 * Count actual links, excluding failed seed reservations.
 * Prefix offsets give each cell a contiguous interval; particles scatter in
 * parallel so a crowded bin cannot serialize an entire workgroup. */
export class UniformNarrowBandOrder {
 private readonly counts:GPUBuffer;
 private readonly offsets:GPUBuffer;
 private readonly enabled:GPUBuffer;
 private scan!:ParticleBinScan;
 private pipelines!:Record<string,GPUComputePipeline>;
 private groups!:readonly [GPUBindGroup,GPUBindGroup];
 get allocatedBytes():number{return this.counts.size+this.offsets.size+this.enabled.size+(this.scan?.allocatedBytes??0);}
 constructor(private readonly device:GPUDevice,private readonly dims:readonly number[],private readonly particles:readonly [GPUBuffer,GPUBuffer],private readonly bins:GPUBuffer,private readonly links:GPUBuffer){
  const buffer=(label:string,size:number)=>device.createBuffer({label:`FLIP spatial order ${label}`,size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  this.counts=buffer("counts",4*dims.reduce((n,v)=>n*v,1));this.offsets=buffer("offsets",this.counts.size);
  // Shared APIC scan's activation record: no failure and an active step.
  // This immutable record makes our sort unconditional, including bootstrap.
  this.enabled=buffer("scan activation",88);const activation=new Float32Array(22);activation[21]=1;device.queue.writeBuffer(this.enabled,0,activation);
 }
 async initialize():Promise<void>{
  this.scan=await ParticleBinScan.create(this.device,this.counts.size/4,this.counts,this.offsets,this.enabled);
  const module=this.device.createShaderModule({label:"FLIP spatial order",code:/* wgsl */`
struct Particle{position:vec4f,velocity:vec4f,before:vec4f}
@group(0) @binding(0) var<storage,read> input:array<Particle>;
@group(0) @binding(1) var<storage,read_write> output:array<Particle>;
@group(0) @binding(2) var<storage,read_write> bins:array<atomic<u32>>;
@group(0) @binding(3) var<storage,read_write> links:array<u32>;
@group(0) @binding(4) var<storage,read_write> counts:array<u32>;
@group(0) @binding(5) var<storage,read> offsets:array<u32>;

// Keep all 4 cubed fine cells of a topology tile adjacent in memory.
// Linear h-cell order separates z-neighbors by an entire domain plane.
fn cellAt(order:u32)->u32{
 let dims=vec3u(${this.dims.map(n=>`${n}u`).join(',')});let tiles=dims/4u;let t=order/64u;let l=order%64u;
 let c=4u*vec3u(t%tiles.x,(t/tiles.x)%tiles.y,t/(tiles.x*tiles.y))+vec3u(l%4u,(l/4u)%4u,l/16u);
 return c.x+dims.x*(c.y+dims.y*c.z);
}
@compute @workgroup_size(64) fn count(@builtin(global_invocation_id) gid:vec3u){
 for(var order=gid.x;order<arrayLength(&counts);order+=65536u){
  let cell=cellAt(order);
  var n=0u;var link=atomicLoad(&bins[2u*cell]);while(link!=0u){n++;link=links[link-1u];}
  counts[order]=n;atomicStore(&bins[2u*cell],0u);
 }
}
@compute @workgroup_size(64) fn pack(@builtin(global_invocation_id) gid:vec3u){
 let dims=vec3u(${this.dims.map(n=>`${n}u`).join(',')});let tiles=dims/4u;
 let last=arrayLength(&counts)-1u;let live=offsets[last]+counts[last];
 for(var i=gid.x;i<live;i+=65536u){
  let p=input[i];let c=vec3u(p.position.xyz);let t=c/4u;let l=c%4u;
  let cell=c.x+dims.x*(c.y+dims.y*c.z);let order=64u*(t.x+tiles.x*(t.y+tiles.y*t.z))+l.x+4u*l.y+16u*l.z;
  let next=offsets[order]+atomicAdd(&bins[2u*cell],1u);output[next]=p;
  let a=arrayLength(&input)+4u*next;let position=bitcast<vec4u>(p.position);
  links[a]=position.x;links[a+1u]=position.y;links[a+2u]=position.z;links[a+3u]=position.w;
 }
}
@compute @workgroup_size(64) fn rewire(@builtin(global_invocation_id) gid:vec3u){
 for(var order=gid.x;order<arrayLength(&counts);order+=65536u){
  let cell=cellAt(order);
  let n=counts[order];let start=offsets[order];atomicStore(&bins[2u*cell],select(0u,start+1u,n>0u));
  for(var j=0u;j<n;j++){links[start+j]=select(start+j+2u,0u,j+1u==n);}
 }
}`});
  const layout=this.device.createBindGroupLayout({entries:[0,1,2,3,4,5].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===0||binding===5?"read-only-storage" as const:"storage" as const}}))});
  const pipelineLayout=this.device.createPipelineLayout({bindGroupLayouts:[layout]});
  this.pipelines=Object.fromEntries(await Promise.all(["count","pack","rewire"].map(async entryPoint=>[entryPoint,await this.device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint}})])));
  this.groups=[0,1].map(parity=>this.device.createBindGroup({layout,entries:[this.particles[1-parity]!,this.particles[parity]!,this.bins,this.links,this.counts,this.offsets].map((buffer,binding)=>({binding,resource:{buffer}}))})) as [GPUBindGroup,GPUBindGroup];
 }
 encode(encoder:GPUCommandEncoder,parity:number):void{
  const run=(entry:string)=>{const pass=encoder.beginComputePass({label:`Narrow-band FLIP order ${entry}`});pass.setPipeline(this.pipelines[entry]!);pass.setBindGroup(0,this.groups[parity]!);pass.dispatchWorkgroups(1024);pass.end();};
  run("count");this.scan.encode(encoder);run("pack");run("rewire");
 }
 destroy():void{this.scan?.destroy();this.counts.destroy();this.offsets.destroy();this.enabled.destroy();}
}
