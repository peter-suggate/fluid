import type {GPUSecondaryParticleSource} from "../../core/webgpu-secondary-particles";

/** Packs only ballistic samples for the existing optical water renderer.
 * This is a representation of live samples, not a second spray simulation.
 * No lifetime deletion or inferred transfer of mass from particle counts. */
export class UniformNarrowBandSpray {
 readonly renderSource:GPUSecondaryParticleSource;
 readonly allocatedBytes:number;
 private readonly output:GPUBuffer;
 private readonly draw:GPUBuffer;
 private pipeline!:GPUComputePipeline;
 private groups!:GPUBindGroup[];
 constructor(private readonly device:GPUDevice,private readonly particles:readonly [GPUBuffer,GPUBuffer],private readonly state:GPUBuffer,private readonly h:readonly number[],private readonly dims:readonly number[]){
  const capacity=particles[0].size/48;
  this.output=device.createBuffer({label:"NB ballistic optical droplets",size:capacity*64,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  this.draw=device.createBuffer({label:"NB spray draw",size:16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.INDIRECT|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
  device.queue.writeBuffer(this.draw,0,new Uint32Array([6,0,0,0]));
  this.renderSource={buffer:this.output,capacity,strideBytes:64,indirectBuffer:this.draw};
  this.allocatedBytes=this.output.size+16;
 }
 async initialize(){
  const code=/* wgsl */`
struct Sample{position:vec4f,velocity:vec4f,before:vec4f}
struct Drop{positionRadius:vec4f,velocityAge:vec4f,birthNormalLifetime:vec4f,shape:vec4f}
@group(0) @binding(0) var<storage,read> samples:array<Sample>;
@group(0) @binding(1) var<storage,read> state:array<u32>;
@group(0) @binding(2) var<storage,read_write> drops:array<Drop>;
@group(0) @binding(3) var<storage,read_write> draw:array<atomic<u32>>;
@compute @workgroup_size(64) fn pack(@builtin(global_invocation_id) gid:vec3u){
 for(var i=gid.x;i<min(state[0],arrayLength(&samples));i+=65536u){
  let p=samples[i];if(p.position.w<3.0){continue;}
  let at=atomicAdd(&draw[1],1u);
  let h=vec3f(${this.h.join(",")});let dims=vec3f(${this.dims.join(",")});
  let world=p.position.xyz*h-vec3f(0.5*dims.x*h.x,0,0.5*dims.z*h.z);
  let radius=0.22*min(h.x,min(h.y,h.z));
  drops[at]=Drop(vec4f(world,radius),vec4f(p.velocity.xyz,0),vec4f(0,1,0,1),vec4f(1,0,1,0.02));
 }
}`;
  this.pipeline=await this.device.createComputePipelineAsync({layout:"auto",compute:{module:this.device.createShaderModule({label:"NB spray publication",code}),entryPoint:"pack"}});
  this.groups=this.particles.map(buffer=>this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[buffer,this.state,this.output,this.draw].map((buffer,binding)=>({binding,resource:{buffer}}))}));
 }
 encode(encoder:GPUCommandEncoder,parity:number){
  encoder.clearBuffer(this.draw,4,4);
  const pass=encoder.beginComputePass({label:"Publish NB optical spray"});pass.setPipeline(this.pipeline);pass.setBindGroup(0,this.groups[parity]!);pass.dispatchWorkgroups(1024);pass.end();
 }
 destroy(){this.output.destroy();this.draw.destroy();}
}
