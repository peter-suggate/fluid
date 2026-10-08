import { ParticleBinScan } from "../lib/methods/particle/scan";

/** Diagnostic tile index over an existing particle epoch. It changes neither
 * particle order nor any simulation parameter. The caller lends the expired
 * cell-link arena until the next spatial sort overwrites it.
 *
 * workspace: 16 header words, capacity particle indices, then (first,end,tile,
 * regular-fine) per packet. Header 0 is packet count; 4..6 are indirect args;
 * 8..10 dispatch one lane per live particle. No fixed per-tile population cap.
 */
export class NarrowBandParticlePackets {
  readonly indirect: GPUBuffer;
  readonly packetBase: number;
  private readonly counts: GPUBuffer;
  private readonly offsets: GPUBuffer;
  private readonly enabled: GPUBuffer;
  private scan!: ParticleBinScan;
  private pipelines!: Record<string, GPUComputePipeline>;
  private groups!: GPUBindGroup[];
  private readonly tiles: number;
  constructor(private readonly device: GPUDevice, private readonly dims: readonly number[],
    private readonly particles: readonly GPUBuffer[], private readonly state: GPUBuffer,
    private readonly workspace: GPUBuffer, private readonly topology: GPUBuffer,
    private readonly support: GPUBuffer) {
    this.tiles = dims.reduce((a,b)=>a*b,1)/64;
    const capacity=particles[0]!.size/48, maxPackets=Math.ceil(capacity/64)+this.tiles;
    this.packetBase=16+capacity;
    if (workspace.size<4*(this.packetBase+4*maxPackets)) throw new Error("Packet workspace too small");
    const buffer=(label:string,size:number,extra=0)=>device.createBuffer({label,size,
      usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST|extra});
    this.counts=buffer("Particle packet tile counts and cursors",8*this.tiles);
    this.offsets=buffer("Particle packet tile offsets",4*this.tiles);
    this.enabled=buffer("Particle packet scan activation",88);
    this.indirect=buffer("Particle packet indirect dispatches",48,GPUBufferUsage.INDIRECT);
    const activation=new Float32Array(22);activation[21]=1;device.queue.writeBuffer(this.enabled,0,activation);
  }
  async initialize():Promise<void> {
    this.scan=await ParticleBinScan.create(this.device,this.tiles,
      {buffer:this.counts,size:4*this.tiles},this.offsets,this.enabled);
    const module=this.device.createShaderModule({label:"Particle tile packet index",code:/* wgsl */`
struct Particle{position:vec4f,velocity:vec4f,before:vec4f}
@group(0) @binding(0) var<storage,read> particles:array<Particle>;
@group(0) @binding(1) var<storage,read> state:array<u32>;
@group(0) @binding(2) var<storage,read_write> counts:array<atomic<u32>>;
@group(0) @binding(3) var<storage,read> offsets:array<u32>;
@group(0) @binding(4) var<storage,read_write> work:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read> topology:array<u32>;
@group(0) @binding(6) var<storage,read> support:array<u32>;
const D=vec3u(${this.dims.map(n=>n+'u').join(',')});const T=D/4u;
const NT:u32=${this.tiles}u;const JOB:u32=${this.packetBase}u;
fn tileOf(q:vec3f)->u32{let t=vec3u(clamp(q,vec3f(0),vec3f(D)-0.001))/4u;return t.x+T.x*(t.y+T.y*t.z);}
@compute @workgroup_size(64) fn count(@builtin(global_invocation_id) id:vec3u){
 for(var i=id.x;i<min(state[0],arrayLength(&particles));i+=65536u){atomicAdd(&counts[tileOf(particles[i].position.xyz)],1u);}
}
@compute @workgroup_size(64) fn scatter(@builtin(global_invocation_id) id:vec3u){
 for(var i=id.x;i<min(state[0],arrayLength(&particles));i+=65536u){
  let t=tileOf(particles[i].position.xyz);let slot=offsets[t]+atomicAdd(&counts[NT+t],1u);
  atomicStore(&work[16u+slot],i);
 }
}
@compute @workgroup_size(64) fn packets(@builtin(global_invocation_id) id:vec3u){
 let tile=id.x+65535u*64u*id.y;if(tile>=NT){return;}
 let n=atomicLoad(&counts[tile]);let packets=(n+63u)/64u;if(packets==0u){return;}
 let first=offsets[tile];let job=atomicAdd(&work[0],packets);
 let regular=u32((support[3u*NT+tile]&1u)!=0u&&(topology[2u*NT+2u*tile]>>27u)==1u);
 for(var k=0u;k<packets;k++){
  let base=JOB+4u*(job+k);atomicStore(&work[base],first+64u*k);atomicStore(&work[base+1u],min(first+64u*(k+1u),first+n));
  atomicStore(&work[base+2u],tile);atomicStore(&work[base+3u],regular);
 }
}
@compute @workgroup_size(1) fn finish(){
 let n=atomicLoad(&work[0]);atomicStore(&work[4],min(n,65535u));atomicStore(&work[5],(n+65534u)/65535u);atomicStore(&work[6],1u);
 let p=(min(state[0],arrayLength(&particles))+63u)/64u;atomicStore(&work[8],min(p,65535u));atomicStore(&work[9],(p+65534u)/65535u);atomicStore(&work[10],1u);
}
`});
    const layout=this.device.createBindGroupLayout({entries:[0,1,2,3,4,5,6].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,
      buffer:{type:binding===2||binding===4?"storage" as const:"read-only-storage" as const}}))});
    const pipelineLayout=this.device.createPipelineLayout({bindGroupLayouts:[layout]});
    this.pipelines=Object.fromEntries(await Promise.all(["count","scatter","packets","finish"].map(async entryPoint=>
      [entryPoint,await this.device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint}})])));
    this.groups=this.particles.map(particle=>this.device.createBindGroup({layout,entries:
      [particle,this.state,this.counts,this.offsets,this.workspace,this.topology,this.support].map((buffer,binding)=>({binding,resource:{buffer}}))}));
  }
  encode(encoder:GPUCommandEncoder,parity:number):void {
    encoder.clearBuffer(this.counts);encoder.clearBuffer(this.workspace,0,64);
    const dispatch=(entry:string)=>{
      const pass=encoder.beginComputePass({label:"Particle packet index "+entry});pass.setPipeline(this.pipelines[entry]!);pass.setBindGroup(0,this.groups[parity]!);
      const n=entry==="finish"?1:entry==="packets"?Math.ceil(this.tiles/64):1024;
      pass.dispatchWorkgroups(Math.min(n,65535),Math.ceil(n/65535));pass.end();
    };
    dispatch("count");this.scan.encode(encoder);dispatch("scatter");dispatch("packets");dispatch("finish");
    encoder.copyBufferToBuffer(this.workspace,0,this.indirect,0,48);
  }
  destroy():void {this.scan?.destroy();this.counts.destroy();this.offsets.destroy();this.enabled.destroy();this.indirect.destroy();}
}
