import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";

export interface UniformMixedCellFields {
  volume:GPUTexture;
  velocity:GPUTexture;
  negativeFaces:GPUBuffer;
}

/** Mixed -> fine cell handoff, encoded only when leaving mixed mode. Mirrors
 * native global prolongation: constant cell volume and face-normal velocity
 * interpolation. Each canonical face patch retains its integrated flux.
 * Vertex phi and the live host handoff are separate responsibilities. */
export class UniformMixedCellProlongation {
  readonly allocatedBytes=0;
  private pipeline?:GPUComputePipeline;
  private readonly resources:GPUBindGroupLayout;
  private readonly group:GPUBindGroup;
  constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,input:UniformMixedCellFields,output:UniformMixedCellFields){
    const d=ownership.layout.lattice.dimensions;
    for(const fields of [input,output]){
      for(const [texture,format] of [[fields.volume,"r32float"],[fields.velocity,"rgba32float"]] as const)
        if(texture.format!==format||[texture.width,texture.height,texture.depthOrArrayLayers].some((n,a)=>n!==d[a]))throw new Error("Mixed cell handoff requires matching native fields");
      if(fields.negativeFaces.size<4*(d[0]*d[1]+d[0]*d[2]+d[1]*d[2]))throw new Error("Mixed cell handoff requires negative face planes");
    }
    if(input.volume===output.volume||input.velocity===output.velocity||input.negativeFaces===output.negativeFaces)throw new Error("Mixed cell handoff fields must be disjoint");
    this.resources=device.createBindGroupLayout({entries:[
      ...[0,1].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
      {binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      ...(["r32float","rgba32float"] as const).map((format,i)=>({binding:3+i,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only" as const,format,viewDimension:"3d" as const}})),
      {binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
    ]});
    this.group=device.createBindGroup({layout:this.resources,entries:[
      {binding:0,resource:input.volume.createView()},{binding:1,resource:input.velocity.createView()},{binding:2,resource:{buffer:input.negativeFaces}},
      {binding:3,resource:output.volume.createView()},{binding:4,resource:output.velocity.createView()},{binding:5,resource:{buffer:output.negativeFaces}},
    ]});
  }
  async initialize():Promise<void>{
    const module=this.device.createShaderModule({code:uniformMixedTopologyWGSL(this.ownership.layout,0)+/* wgsl */`
@group(1) @binding(0) var volume:texture_3d<f32>;
@group(1) @binding(1) var velocity:texture_3d<f32>;
@group(1) @binding(2) var<storage,read> boundary:array<f32>;
@group(1) @binding(3) var outputVolume:texture_storage_3d<r32float,write>;
@group(1) @binding(4) var outputVelocity:texture_storage_3d<rgba32float,write>;
@group(1) @binding(5) var<storage,read_write> outputBoundary:array<f32>;
fn umBoundaryAt(p:vec3i,axis:u32)->u32 {
 if(axis==0u){return u32(p.y)+UM_D.y*u32(p.z);}
 if(axis==1u){return UM_D.y*UM_D.z+u32(p.x)+UM_D.x*u32(p.z);}
 return UM_D.y*UM_D.z+UM_D.x*UM_D.z+u32(p.x)+UM_D.x*u32(p.y);
}
fn umReadFace(face:UMFace)->f32 {
 if(face.anchor[face.axis]<0){return boundary[umBoundaryAt(face.anchor,face.axis)];}
 return textureLoad(velocity,face.anchor,0)[face.axis];
}
fn umContainingPatch(owner:UMOwner,id:vec3u,axis:u32,sign:i32)->UMFace {
 let first=umFace(owner,axis,sign,0u);let local=id-umOrigin(owner);
 let u=(axis+1u)%3u;let v=(axis+2u)%3u;
 return umFace(owner,axis,sign,local[u]/first.width+(owner.width/first.width)*(local[v]/first.width));
}
@compute @workgroup_size(4,4,4) fn prolongCells(@builtin(global_invocation_id) id:vec3u){
 if(any(id>=UM_D)){return;}
 let owner=umOwnerAt(vec3i(id));let origin=umOrigin(owner);
 textureStore(outputVolume,vec3i(id),textureLoad(volume,vec3i(origin),0));
 var value=vec4f(0);
 for(var axis=0u;axis<3u;axis++){
  let left=umReadFace(umContainingPatch(owner,id,axis,-1));let right=umReadFace(umContainingPatch(owner,id,axis,1));
  value[axis]=mix(left,right,f32(id[axis]-origin[axis]+1u)/f32(owner.width));
  if(id[axis]==0u){outputBoundary[umBoundaryAt(vec3i(id),axis)]=left;}
 }
 textureStore(outputVelocity,vec3i(id),value);
}`});
    const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
    this.pipeline=await this.device.createComputePipelineAsync({layout:this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources]}),compute:{module,entryPoint:"prolongCells"}});
  }
  encode(encoder:GPUCommandEncoder):void{
    if(!this.pipeline)throw new Error("Mixed cell handoff is not initialized");
    const pass=encoder.beginComputePass({label:"Uniform mixed to fine cell handoff"});pass.setPipeline(this.pipeline);pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.group);
    const d=this.ownership.layout.lattice.dimensions;pass.dispatchWorkgroups(Math.ceil(d[0]/4),Math.ceil(d[1]/4),Math.ceil(d[2]/4));pass.end();
  }
}
