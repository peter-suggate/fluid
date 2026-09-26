import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformMixedFaceAddressWGSL } from "./uniform-mixed-face-dispatch.wgsl";

export interface UniformMixedMomentumCacheFields {
  extended:GPUTexture;
  physical:GPUTexture;
  phase:GPUTexture;
  negative:GPUBuffer;
  coarseExtended:GPUTexture;
  coarsePhysical:GPUTexture;
  coarseWeight:GPUTexture;
}
/** 4h MAC sampling restriction, including negative domain faces. These are
 * interpolation cache values, never an independently advanced velocity grid.
 * The three borrowed RGBA fields have extent D/4+2 and may reuse prepared
 * coarse resources. Fine/transition patches are averaged once per update. */
export class UniformMixedMomentumCache {
  readonly allocatedBytes=0;
  private pipeline?:GPUComputePipeline;
  private readonly resources:GPUBindGroupLayout;
  constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership){
    this.resources=device.createBindGroupLayout({entries:[
      ...[0,1,2].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
      {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      ...[4,5,6].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only" as const,format:"rgba32float" as const,viewDimension:"3d" as const}})),
    ]});
  }
  bind(f:UniformMixedMomentumCacheFields):GPUBindGroup{
    const d=this.ownership.layout.lattice.dimensions;
    for(const [i,t] of [f.extended,f.physical,f.phase,f.coarseExtended,f.coarsePhysical,f.coarseWeight].entries())
      if([t.width,t.height,t.depthOrArrayLayers].some((n,a)=>n!==(i<3?d[a]:d[a]!/4+2))||t.format!==(i===2?"r32float":"rgba32float"))throw new Error("Mixed momentum cache requires native inputs and 4h halo outputs");
    if(new Set([f.coarseExtended,f.coarsePhysical,f.coarseWeight]).size!==3)throw new Error("Mixed momentum cache outputs must be disjoint");
    return this.device.createBindGroup({layout:this.resources,entries:[
      ...[f.extended,f.physical,f.phase].map((t,binding)=>({binding,resource:t.createView()})),
      {binding:3,resource:{buffer:f.negative}},
      ...[f.coarseExtended,f.coarsePhysical,f.coarseWeight].map((t,i)=>({binding:4+i,resource:t.createView()})),
    ]});
  }
  async initialize():Promise<void>{
    const module=this.device.createShaderModule({code:uniformMixedTopologyWGSL(this.ownership.layout,0)+/* wgsl */`
@group(1) @binding(0) var extended:texture_3d<f32>;
@group(1) @binding(1) var physical:texture_3d<f32>;
@group(1) @binding(2) var phase:texture_3d<f32>;
@group(1) @binding(3) var<storage,read> negative:array<f32>;
@group(1) @binding(4) var coarseExtended:texture_storage_3d<rgba32float,write>;
@group(1) @binding(5) var coarsePhysical:texture_storage_3d<rgba32float,write>;
@group(1) @binding(6) var coarseWeight:texture_storage_3d<rgba32float,write>;
${uniformMixedFaceAddressWGSL}
fn umCachePayload(anchor:vec3i,axis:u32)->vec3f {
 var next=anchor;next[axis]+=1;let left=umOwnerAt(anchor);let right=umOwnerAt(next);var supported=false;
 if(left.width!=0u){supported=textureLoad(phase,vec3i(umOrigin(left)),0).x>0.5;}
 if(right.width!=0u){supported=supported||textureLoad(phase,vec3i(umOrigin(right)),0).x>0.5;}
 var e=0.0;var p=0.0;
 if(anchor[axis]<0){e=negative[umNegativeBoundaryIndex(vec3u(max(anchor,vec3i(0))),axis)];if(supported){p=e;}}
 else{e=textureLoad(extended,anchor,0)[axis];if(supported){p=textureLoad(physical,anchor,0)[axis];}}
 return vec3f(e,p,select(0.0,1.0,supported));
}
@compute @workgroup_size(4,4,4) fn cache(@builtin(global_invocation_id) gid:vec3u){
 if(any(gid>=UM_T+vec3u(2))){return;}let id=vec3i(gid)-vec3i(1);
 var e=vec4f(0);var p=vec4f(0);var weight=vec4f(0);
 for(var axis=0u;axis<3u;axis++){
  let u=(axis+1u)%3u;let v=(axis+2u)%3u;
  if(id[axis]< -1||id[axis]>=i32(UM_T[axis])||id[u]<0||id[u]>=i32(UM_T[u])||id[v]<0||id[v]>=i32(UM_T[v])){continue;}
  var anchor=id*4;anchor[axis]+=3;
  var owner=umOwnerAt(anchor);var sign=1;
  if(owner.width==0u){var next=anchor;next[axis]+=1;owner=umOwnerAt(next);sign=-1;}
  let width=umFace(owner,axis,sign,0u).width;let side=4u/width;var value=vec3f(0);
  for(var y=0u;y<side;y++){for(var x=0u;x<side;x++){
   var q=anchor;q[u]+=i32(x*width);q[v]+=i32(y*width);value+=umCachePayload(q,axis);
  }}value/=f32(side*side);e[axis]=value.x;p[axis]=value.y;weight[axis]=value.z;
 }
 textureStore(coarseExtended,vec3i(gid),e);textureStore(coarsePhysical,vec3i(gid),p);textureStore(coarseWeight,vec3i(gid),weight);
}`});
    const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
    this.pipeline=await this.device.createComputePipelineAsync({layout:this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources]}),compute:{module,entryPoint:"cache"}});
  }
  encode(encoder:GPUCommandEncoder,group:GPUBindGroup):void{
    if(!this.pipeline)throw new Error("Mixed momentum cache is not initialized");
    const pass=encoder.beginComputePass({label:"Uniform mixed 4h sampling cache"});pass.setPipeline(this.pipeline);pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group);
    pass.dispatchWorkgroups(...this.ownership.layout.lattice.dimensions.map(n=>Math.ceil((n/4+2)/4)) as [number,number,number]);pass.end();
  }
}
