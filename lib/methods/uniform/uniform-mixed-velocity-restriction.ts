import { uniformMixedFaceAddressWGSL, uniformMixedFaceDispatchWGSL } from "./uniform-mixed-face-dispatch.wgsl";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";

/** One-time fine-to-mixed MAC restriction into a borrowed native velocity
 * field. Shared anchors pack all their components in one invocation, avoiding
 * racing rgba writes where several coarse/fine patches meet. No allocation. */
export class UniformMixedVelocityRestriction {
  readonly allocatedBytes=0;
  private pipelines:GPUComputePipeline[]=[];
  private readonly resources:GPUBindGroupLayout;
  private readonly group:GPUBindGroup;
  constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,input:GPUTexture,output:GPUTexture,boundary:GPUBuffer){
    const d=ownership.layout.lattice.dimensions;
    if([input.width,input.height,input.depthOrArrayLayers].some((v,a)=>v!==d[a]!+2)
      ||[output.width,output.height,output.depthOrArrayLayers].some((v,a)=>v!==d[a])||input===output||boundary.size<4*(d[0]*d[1]+d[0]*d[2]+d[1]*d[2]))
      throw new Error("Mixed velocity restriction requires native halo input and disjoint domain output");
    this.resources=device.createBindGroupLayout({entries:[
      {binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},
      {binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:1,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"rgba32float",viewDimension:"3d"}},
    ]});
    this.group=device.createBindGroup({layout:this.resources,entries:[{binding:0,resource:input.createView()},{binding:1,resource:output.createView()},{binding:2,resource:{buffer:boundary}}]});
  }
  async initialize():Promise<void>{
    const module=this.device.createShaderModule({code:uniformMixedTopologyWGSL(this.ownership.layout,0)+/* wgsl */`
@group(1) @binding(0) var input:texture_3d<f32>;
@group(1) @binding(1) var output:texture_storage_3d<rgba32float,write>;
@group(1) @binding(2) var<storage,read_write> boundary:array<f32>;
${uniformMixedFaceAddressWGSL}
fn umRestrictedFace(face:UMFace)->f32 {
 var sum=0.0;let u=(face.axis+1u)%3u;let v=(face.axis+2u)%3u;
 for(var y=0u;y<face.width;y++){for(var x=0u;x<face.width;x++){
  var q=face.anchor+vec3i(1);q[u]+=i32(x);q[v]+=i32(y);sum+=textureLoad(input,q,0)[face.axis];
 }}
 return sum/f32(face.width*face.width);
}
${uniformMixedFaceDispatchWGSL("restrictVelocity", "umRestrictedFace(face)")}
`});
    const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
    if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
    const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources]});
    this.pipelines=await Promise.all([1,2,4].map(umCellWidth=>this.device.createComputePipelineAsync({layout,compute:{module,entryPoint:"restrictVelocity",constants:{umCellWidth,umDispatchX:this.ownership.dispatchX}}})));
  }
  encode(encoder:GPUCommandEncoder):void{
    if(this.pipelines.length!==3)throw new Error("Mixed velocity restriction is not initialized");
    const pass=encoder.beginComputePass({label:"Uniform mixed MAC restriction"});pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.group);this.ownership.dispatch(pass,this.pipelines);pass.end();
  }
}
