import type {UniformClothGridFields} from './uniform-cloth-grid-coupling';
import type {UniformNarrowBandFlip} from './uniform-narrow-band-flip';
import type {WetClothSourceField} from '../../cloth/source-types';
import {narrowBandParticleWGSL} from './uniform-narrow-band-apic.wgsl';

/** Retires actual production FLIP records before native move/sort. Erasing
 * phi alone leaves samples capable of reconstructing the removed water.
 * Released source terminators are rounded boxes; unknown kinds fail. */
export class UniformSourceParticleTermination {
 private readonly layout:GPUBindGroupLayout;
 private pipeline?:GPUComputePipeline;
 private shapes?:GPUBuffer;
 private params:GPUBuffer;
 private group?:GPUBindGroup;
 private count=0;
 readonly diagnostics:GPUBuffer;
 constructor(private readonly fields:UniformClothGridFields,private readonly particles:Pick<UniformNarrowBandFlip,'particles'|'state'|'capacity'|'particleStride'|'transferMode'>){
  const d=fields.device,b=(binding:number,type:GPUBufferBindingType):GPUBindGroupLayoutEntry=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type}});
  if(particles.transferMode!=='flip'||particles.particleStride!==12)throw new Error('Source termination requires native FLIP records');
  this.layout=d.createBindGroupLayout({entries:[b(0,'storage'),b(1,'read-only-storage'),b(2,'read-only-storage'),b(3,'uniform'),b(4,'storage')]});
  this.params=d.createBuffer({size:48,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  this.diagnostics=d.createBuffer({label:'Actual terminated FLIP sample count',size:16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
 }
 async initialize():Promise<void>{
  const d=this.fields.device,module=d.createShaderModule({label:'Source exact FLIP particle termination',code:narrowBandParticleWGSL(false)+/* wgsl */`
struct Box {center:vec4f,extents:vec4f,rotation:vec4f}
struct Params {origin:vec4f,h:vec4f,counts:vec4u}
@group(0) @binding(0) var<storage,read_write> particles:array<Particle>;
@group(0) @binding(1) var<storage,read> state:array<u32>;
@group(0) @binding(2) var<storage,read> boxes:array<Box>;
@group(0) @binding(3) var<uniform> params:Params;
@group(0) @binding(4) var<storage,read_write> diagnostics:array<atomic<u32>>;
fn rotate(p:vec3f,q:vec4f)->vec3f{return p+2.0*cross(q.xyz,cross(q.xyz,p)+q.w*p);}
@compute @workgroup_size(64) fn terminate(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 let held=32.0*vec3f(bitcast<vec3i>(vec3u(state[12],state[13],state[14])));
 for(var i=gid.x;i<params.counts.x;i+=64u*groups.x){
  let p=particles[i];if(p.position.w<=0.0){continue;}
  let world=params.origin.xyz+(p.position.xyz+held)*params.h.xyz;
  for(var b=0u;b<params.counts.y;b++){
   let box=boxes[b];let local=rotate(world-box.center.xyz,vec4f(-box.rotation.xyz,box.rotation.w));
   let q=abs(local)-box.extents.xyz;let distance=min(max(q.x,max(q.y,q.z)),0.0)+length(max(q,vec3f(0.0)))-box.extents.w;
   if(select(distance,-distance,box.center.w!=0.0)<0.0){particles[i].position.w=0.0;atomicAdd(&diagnostics[0],1u);break;}
  }
 }
}`});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error');if(errors.length)throw new Error(errors.map(m=>m.message).join('\n'));
  this.pipeline=await d.createComputePipelineAsync({layout:d.createPipelineLayout({bindGroupLayouts:[this.layout]}),compute:{module,entryPoint:'terminate'}});
 }
 /** Prepare only after prior submission completes. Source poses are the same
  * accepted scripted poses used for the water subtraction in this substep. */
 prepare(fields:readonly WetClothSourceField[]):void{
  const roots=fields.filter(f=>f.root&&f.usage==='terminator');
  if(roots.some(f=>f.kind!=='box'))throw new Error('Source particle terminator needs an exact supported field; only released box terminators are admitted');
  this.shapes?.destroy();this.group=undefined;this.count=roots.length;if(!roots.length)return;
  const data=new Float32Array(12*roots.length);
  roots.forEach((f,i)=>{
   const axis=f.axis??[0,1,0],angle=f.angle_rad??0,length=Math.hypot(...axis),s=angle?Math.sin(angle/2)/length:0;
   if(angle&&!length)throw new Error('Rotated source terminator has zero axis');
   data.set([...(f.center_m??[0,0,0]),f.inside?1:0,...f.halfExtents_m!,f.roundingRadius_m??0,axis[0]*s,axis[1]*s,axis[2]*s,Math.cos(angle/2)],12*i);
  });
  const d=this.fields.device;this.shapes=d.createBuffer({size:data.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});d.queue.writeBuffer(this.shapes,0,data);
  const raw=new ArrayBuffer(48),floats=new Float32Array(raw),ints=new Uint32Array(raw);floats.set([...this.fields.origin_m,0,...this.fields.cellSize_m,0]);ints.set([this.particles.capacity,roots.length,0,0],8);d.queue.writeBuffer(this.params,0,raw);
  this.group=d.createBindGroup({layout:this.layout,entries:[{binding:0,resource:{buffer:this.particles.particles[1],size:4*this.particles.capacity*this.particles.particleStride}},{binding:1,resource:{buffer:this.particles.state}},{binding:2,resource:{buffer:this.shapes}},{binding:3,resource:{buffer:this.params}},{binding:4,resource:{buffer:this.diagnostics}}]});
 }
 encode(encoder:GPUCommandEncoder):void{
  if(!this.count)return;if(!this.pipeline||!this.group)throw new Error('Particle source termination is not initialized');
  const p=encoder.beginComputePass({label:'Retire source-terminated FLIP samples'});p.setPipeline(this.pipeline);p.setBindGroup(0,this.group);p.dispatchWorkgroups(Math.max(1,Math.min(4096,Math.ceil(this.particles.capacity/64))));p.end();
 }
 destroy():void{this.shapes?.destroy();this.params.destroy();this.diagnostics.destroy();}
}
