import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformMixedVertexSamplingWGSL } from "./uniform-mixed-vertex-sampling.wgsl";

/** Native post-transport floors on canonical owners. The immutable input is
 * retained for the entire orphan census; one discarded neighbour must not
 * make the next neighbour eligible. All fields and accounting are borrowed.
 * Embedded solids must be excluded by the host until open fractions are bound. */
export class UniformMixedCleanup {
 readonly allocatedBytes=0;
 private readonly resources:GPUBindGroupLayout;
 private readonly pipelines=new Map<string,GPUComputePipeline>();
 constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership){
  this.resources=device.createBindGroupLayout({entries:[
   ...[0,1].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
   {binding:2,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"r32float",viewDimension:"3d"}},
   {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
   {binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
  ]});
 }
 bind(input:GPUTexture,output:GPUTexture,phi:GPUTexture,params:GPUBuffer,reductions:GPUBuffer):GPUBindGroup{
  const d=this.ownership.layout.lattice.dimensions;
  for(const [i,t] of [input,output,phi].entries())if(t.format!=="r32float"||[t.width,t.height,t.depthOrArrayLayers].some((v,a)=>v!==d[a]!+(i===2?1:0)))throw new Error("Mixed cleanup requires native volume and vertex fields");
  if(input===output)throw new Error("Mixed cleanup requires an immutable input");
  if(reductions.size<48)throw new Error("Mixed cleanup requires twelve accounting words");
  return this.device.createBindGroup({layout:this.resources,entries:[
   {binding:0,resource:input.createView()},{binding:1,resource:phi.createView()},{binding:2,resource:output.createView()},
   {binding:3,resource:{buffer:params,size:32}},{binding:4,resource:{buffer:reductions,size:48}},
  ]});
 }
 async initialize():Promise<void>{
  const h=this.ownership.layout.lattice.cellSize_m;
  const module=this.device.createShaderModule({code:uniformMixedTopologyWGSL(this.ownership.layout,0)+/* wgsl */`
@group(1) @binding(0) var volume:texture_3d<f32>;
@group(1) @binding(1) var phi:texture_3d<f32>;
@group(1) @binding(2) var output:texture_storage_3d<r32float,write>;
// Same tuning ABI as sharpening: strength, distance, regular floor, orphan floor.
@group(1) @binding(3) var<uniform> tuning:array<vec4f,2>;
@group(1) @binding(4) var<storage,read_write> reductions:array<atomic<u32>>;
fn umLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p),0).x;}
${uniformMixedVertexSamplingWGSL}
fn umVolume(o:UMOwner)->f32{return textureLoad(volume,vec3i(umOrigin(o)),0).x;}
fn umDiscard(o:UMOwner,value:f32,threshold:f32,word:u32)->f32{
 atomicAdd(&reductions[word],1u);
 atomicAdd(&reductions[word+1u],min(u32(abs(value)/threshold*64.0),64u)*o.width*o.width*o.width);
 return 0.0;
}
override umOrphan:bool=false;
fn umClean(o:UMOwner)->f32{
 let value=umVolume(o);let floor=tuning[0].z;let orphan=tuning[0].w;
 let origin=vec3i(umOrigin(o));let w=i32(o.width);
 let band=${4*Math.max(...h)}*f32(o.width);
 if(!umOrphan&&value!=0.0&&abs(value)<floor){
  if(!(value>0.0&&umSampleVertex(vec3f(origin)+vec3f(0.5*f32(o.width)))<band)){
   return umDiscard(o,value,floor,5u);
  }
 }
 if(!umOrphan||!(floor>0.0&&orphan>floor&&value>0.0&&value<orphan)){return value;}
 // Protect the complete neighbouring footprint, including fine surface
 // vertices inside a coarse owner's neighbourhood and hanging samples.
 for(var z=-w;z<=2*w;z++){for(var y=-w;y<=2*w;y++){for(var x=-w;x<=2*w;x++){
  let p=clamp(origin+vec3i(x,y,z),vec3i(0),vec3i(UM_D));
  if(umVertexValue(vec3u(p))<band){return value;}
 }}}
 var mass=0.0;
 // Integrate fill over the 3w cubical footprint in fine-cell mass units.
 // Canonical lookup preserves partial overlaps at both kinds of interface.
 for(var z=-w;z<2*w;z++){for(var y=-w;y<2*w;y++){for(var x=-w;x<2*w;x++){
  let donor=umOwnerAt(origin+vec3i(x,y,z));if(donor.width==0u){continue;}
  let v=umVolume(donor);if(v>=0.05){return value;}mass+=max(v,0.0);
 }}}
 if(mass>=0.25*f32(o.width*o.width*o.width)){return value;}
 return umDiscard(o,value,orphan,10u);
}
@compute @workgroup_size(64) fn clean(@builtin(global_invocation_id) gid:vec3u){
 let o=umAllOwner(gid);if(o.width!=0u){textureStore(output,vec3i(umOrigin(o)),vec4f(umClean(o)));}
}
`});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources]});
  for(const entry of ["floor","orphan"])this.pipelines.set(entry,await this.device.createComputePipelineAsync({layout,compute:{module,entryPoint:"clean",constants:{umDispatchX:this.ownership.dispatchX,umOrphan:Number(entry==="orphan")}}}));
 }
 encode(encoder:GPUCommandEncoder,groups:readonly [GPUBindGroup,GPUBindGroup]):void{
  for(const [i,entry] of ["floor","orphan"].entries()){
   const pipeline=this.pipelines.get(entry);if(!pipeline)throw new Error("Mixed cleanup is not initialized");
   const pass=encoder.beginComputePass({label:`Uniform mixed cleanup ${entry}`});pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,groups[i]!);
   this.ownership.dispatchAll(pass,pipeline);pass.end();
  }
 }
}
