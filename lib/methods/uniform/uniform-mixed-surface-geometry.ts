import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformMixedVertexSamplingWGSL } from "./uniform-mixed-vertex-sampling.wgsl";
import { uniformVolumeTargetWGSL } from "./uniform-volume.wgsl";
import { geometricPlaneBoxWGSL } from "../../core/geometric-plane-box.wgsl";
import { uniformMixedSolidWGSL, type UniformMixedSolid } from "./uniform-mixed-solid.wgsl";

/** The native eight-probe / planar-exact fill rule on mixed owners. Target
 * fractions are intensive, phi is physical, and no fine cell is expanded.
 * Subsequent pressure authority and conditioning consume these same values. */
export class UniformMixedSurfaceGeometry {
  readonly allocatedBytes=0;
  private pipeline?:GPUComputePipeline;
  private readonly resources:GPUBindGroupLayout;
  constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,private readonly solid?:UniformMixedSolid){
    this.resources=device.createBindGroupLayout({entries:[
      {binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},
      ...[1,2].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only" as const,format:"r32float" as const,viewDimension:"3d" as const}})),
    ]});
  }
  bind(phi:GPUTexture,targetFill:GPUTexture,centerPhi:GPUTexture):GPUBindGroup{
    const d=this.ownership.layout.lattice.dimensions;
    for(const [i,t] of [phi,targetFill,centerPhi].entries())
      if(t.format!=="r32float"||[t.width,t.height,t.depthOrArrayLayers].some((n,a)=>n!==d[a]!+(i===0?1:0)))throw new Error("Mixed geometry requires native cell and vertex fields");
    if(targetFill===centerPhi)throw new Error("Mixed geometry outputs must be disjoint");
    return this.device.createBindGroup({layout:this.resources,entries:[phi,targetFill,centerPhi].map((t,binding)=>({binding,resource:t.createView()}))});
  }
  async initialize():Promise<void>{
    const module=this.device.createShaderModule({code:uniformMixedTopologyWGSL(this.ownership.layout,0)+/* wgsl */`
@group(1) @binding(0) var phi:texture_3d<f32>;
@group(1) @binding(1) var targetFill:texture_storage_3d<r32float,write>;
@group(1) @binding(2) var centerPhi:texture_storage_3d<r32float,write>;
fn umLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p),0).x;}
${uniformMixedVertexSamplingWGSL}
${geometricPlaneBoxWGSL}
fn uvCorner(k:u32)->vec3i{return vec3i(umCorner(k,2u));}
fn d4Sum8(v:array<f32,8>)->f32{return ((v[0]+v[5])+(v[1]+v[4]))+((v[2]+v[7])+(v[3]+v[6]));}
${uniformVolumeTargetWGSL(true,true)}
${uniformMixedSolidWGSL(this.solid?2:undefined)}
@compute @workgroup_size(64) fn geometry(@builtin(global_invocation_id) gid:vec3u){
 let owner=umAllOwner(gid);if(owner.width==0u){return;}let origin=umOrigin(owner);
 var vertices:array<f32,8>;var centre:array<f32,8>;
 for(var j=0u;j<8u;j++){vertices[j]=umVertexValue(origin+vec3u(uvCorner(j))*owner.width);centre[j]=vertices[j]*0.125;}
 // Native uvTarget scales by the open fraction; coarse owners are uncut.
 textureStore(targetFill,vec3i(origin),vec4f(umSurfaceTarget(owner,vertices)*select(1.0,umCellOpen(vec3i(origin)),owner.width==1u)));
 // umSampleVertex at the owner centre: all eight weights are exactly 1/8 of
 // these same corner values (stored, or reconstructed where hanging).
 textureStore(centerPhi,vec3i(origin),vec4f(d4Sum8(centre)));
}`});
    const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
    this.pipeline=await this.device.createComputePipelineAsync({layout:this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources,...(this.solid?[this.solid.bindLayout]:[])]}),compute:{module,entryPoint:"geometry",constants:{umDispatchX:this.ownership.dispatchX}}});
  }
  encode(encoder:GPUCommandEncoder,group:GPUBindGroup):void{
    if(!this.pipeline)throw new Error("Mixed surface geometry is not initialized");
    const pass=encoder.beginComputePass({label:"Uniform mixed geometric fill"});pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group);if(this.solid)pass.setBindGroup(2,this.solid.bindGroup);this.ownership.dispatchAll(pass,this.pipeline);pass.end();
  }
}
