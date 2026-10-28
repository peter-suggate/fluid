import { createSvoDrySceneFragmentWGSL, drySceneVertexShader, type SvoDryTraversalMode, type SvoBrickOccupancyMode } from "./program";
import { SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT } from "../../pipeline/svo-render-tuning";

export interface SvoWaterWorldReflectionInputs {
  readonly position: GPUTextureView;
  readonly normal: GPUTextureView;
  readonly dryScene: GPUTextureView;
  readonly screenReflections: GPUTextureView;
  readonly target: GPUTextureView;
  readonly rayKind?: "transmission";
}

/** One bounce, only for SSR misses. The live SVO is the geometry authority.
 * Negative alpha -2 records traversal exhaustion; -1 remains hidden water.
 * Ordinary misses retain the water renderer's own environment, avoiding a
 * second sky model or an extra analytic sun in the reflected background. */
export function createWaterWorldReflectionWGSL(traversal: SvoDryTraversalMode = "hybrid", occupancy: SvoBrickOccupancyMode = "bounds"): string {
  let shader = createSvoDrySceneFragmentWGSL(1, traversal, occupancy, "inline", 0, false, false, false,
    { voxelLightCache: false, primaryEntryPrepass: false, rasterAo: false });
  // The optional secondary-ray path has its own bounded traversal budget. Do
  // not inherit screen-primary seeds, screen-space AO, or cached receiver data.
  const leaf = `let leafBudget=clamp(dry.tuningCounts0.x,1u,${SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT}u);`;
  if (!shader.includes(leaf) || !shader.includes("if(!traversalFinished){}")) throw new Error("World reflection traversal contract changed");
  shader = shader.replace(leaf, `let leafBudget=${SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT}u;`)
    .replace("if(!traversalFinished){}", "if(!traversalFinished){waterWorldTraversalExhausted=true;}")
    .replace("if(leaf.status==SVO_STATUS_WORK_EXHAUSTED||leaf.status==SVO_STATUS_STACK_OVERFLOW||leaf.status==SVO_STATUS_SOURCE_OVERFLOW){}", "if(leaf.status==SVO_STATUS_WORK_EXHAUSTED||leaf.status==SVO_STATUS_STACK_OVERFLOW||leaf.status==SVO_STATUS_SOURCE_OVERFLOW){waterWorldTraversalExhausted=true;}")
    .replace("else if(leaf.status!=SVO_STATUS_MISS){}", "else if(leaf.status!=SVO_STATUS_MISS){waterWorldTraversalExhausted=true;}");
  return shader + /* wgsl */ `
var<private> waterWorldTraversalExhausted:bool;
@group(1) @binding(0) var waterWorldPosition:texture_2d<f32>;
@group(1) @binding(1) var waterWorldNormal:texture_2d<f32>;
@group(1) @binding(2) var waterWorldScreen:texture_2d<f32>;
@fragment fn waterWorldReflectionMain(@builtin(position) pixel:vec4f)->@location(0) vec4f{
  let at=vec2i(pixel.xy);let screen=textureLoad(waterWorldScreen,at,0);
  if(screen.a!=0.0){return screen;}
  let packed=textureLoad(waterWorldPosition,at,0);let nn=textureLoad(waterWorldNormal,at,0);
  if(packed.a<.5||nn.a<.001||dot(nn.xyz,nn.xyz)<1e-12){return screen;}
  let p=packed.xyz/packed.a;var n=normalize(nn.xyz);let incoming=normalize(p-uniforms.cameraPosition.xyz);
  if(dot(n,incoming)>0.0){n=-n;}
  let cell=min(dry.mapping.cellSize.x,min(dry.mapping.cellSize.y,dry.mapping.cellSize.z));
  let ro=p+n*max(.002,.025*cell);let rd=reflect(incoming,n);
  waterWorldTraversalExhausted=false;dryDerivedPageFailure=0u;dryVisibilityFailure=0u;dryVisibilityIgnoredBody=DRY_OWNER_NONE;
  let hit=traceOpaqueScene(ro,rd);
  if(waterWorldTraversalExhausted){return vec4f(screen.rgb,-2);}
  if(hit.t>=DRY_MISS){return screen;}
  let radiance=shadeDrySurface(hit,ro,rd);
  if(dryDerivedPageFailure!=0u||dryVisibilityFailure!=0u||any(radiance!=radiance)||any(abs(radiance)>vec3f(65504))){return vec4f(screen.rgb,-2);}
  return vec4f(max(radiance,vec3f(0)),max(hit.t,.0001));
}
// Request RGB is the authored-IOR refracted direction computed by the water
// pass; alpha zero preserves the existing on-screen fast path.
@fragment fn waterWorldTransmissionMain(@builtin(position) pixel:vec4f)->@location(0) vec4f{
  let at=vec2i(pixel.xy);let request=textureLoad(waterWorldScreen,at,0);if(request.a<.5){return vec4f(0);}
  let packed=textureLoad(waterWorldPosition,at,0);if(packed.a<.5||dot(request.xyz,request.xyz)<.5){return vec4f(0);}
  let p=packed.xyz/packed.a;let rd=normalize(request.xyz);let cell=min(dry.mapping.cellSize.x,min(dry.mapping.cellSize.y,dry.mapping.cellSize.z));
  let epsilon=max(.0001,.001*cell);let ro=p+rd*epsilon;
  waterWorldTraversalExhausted=false;dryDerivedPageFailure=0u;dryVisibilityFailure=0u;dryVisibilityIgnoredBody=DRY_OWNER_NONE;
  let hit=traceOpaqueScene(ro,rd);
  if(waterWorldTraversalExhausted){return vec4f(0,0,0,-2);}
  if(hit.t>=DRY_MISS){return vec4f(0,0,0,-3);}
  let radiance=shadeDrySurface(hit,ro,rd);
  if(dryDerivedPageFailure!=0u||dryVisibilityFailure!=0u||any(radiance!=radiance)||any(abs(radiance)>vec3f(65504))){return vec4f(0,0,0,-2);}
  return vec4f(max(radiance,vec3f(0)),max(hit.t+epsilon,.0001));
}`;
}

export class SvoWaterWorldReflections {
  private pipeline?: GPURenderPipeline;
  private transmissionPipeline?: GPURenderPipeline;
  private layout?: GPUBindGroupLayout;
  private compilation?: Promise<void>;
  private failure?: string;
  private destroyed = false;
  constructor(private readonly device: GPUDevice, private readonly sceneLayout: GPUBindGroupLayout,
    private readonly traversal: SvoDryTraversalMode, private readonly occupancy: SvoBrickOccupancyMode) {}
  get status(): {state:"idle"|"compiling"|"ready"|"failed";detail:string} {
    return this.failure ? {state:"failed",detail:this.failure} : this.pipeline ? {state:"ready",detail:"World water reflections ready"}
      : {state:this.compilation ? "compiling":"idle",detail:"Compiling world water reflections"};
  }
  ensure(): Promise<void> {
    return this.compilation ??= this.compile().catch(error => {this.failure=error instanceof Error ? error.message:String(error);throw error;});
  }
  private async compile(): Promise<void> {
    const module=this.device.createShaderModule({label:"SVO water reflection rays",code:createWaterWorldReflectionWGSL(this.traversal,this.occupancy)});
    const info=await module.getCompilationInfo();const errors=info.messages.filter(m=>m.type==="error");
    if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}:${m.linePos} ${m.message}`).join("\n"));
    const layout=this.device.createBindGroupLayout({label:"World water interface inputs",entries:[
      {binding:0,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"unfilterable-float"}},
      ...[1,2].map(binding=>({binding,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float" as const}})),
    ]});
    const pipeline=await this.device.createRenderPipelineAsync({label:"SVO reflected water rays",layout:this.device.createPipelineLayout({bindGroupLayouts:[this.sceneLayout,layout]}),
      vertex:{module:this.device.createShaderModule({code:drySceneVertexShader}),entryPoint:"vertexMain"},
      fragment:{module,entryPoint:"waterWorldReflectionMain",targets:[{format:"rgba16float"}]},primitive:{topology:"triangle-list"}});
    const transmissionPipeline=await this.device.createRenderPipelineAsync({label:"SVO refracted opaque water rays",layout:this.device.createPipelineLayout({bindGroupLayouts:[this.sceneLayout,layout]}),
      vertex:{module:this.device.createShaderModule({code:drySceneVertexShader}),entryPoint:"vertexMain"},
      fragment:{module,entryPoint:"waterWorldTransmissionMain",targets:[{format:"rgba16float"}]},primitive:{topology:"triangle-list"}});
    if(!this.destroyed){this.layout=layout;this.pipeline=pipeline;this.transmissionPipeline=transmissionPipeline;}
  }
  encode(encoder:GPUCommandEncoder,scene:GPUBindGroup,inputs:SvoWaterWorldReflectionInputs):boolean {
    if(this.failure)throw new Error(this.failure);
    const pipeline=inputs.rayKind==="transmission"?this.transmissionPipeline:this.pipeline;
    if(!pipeline||!this.layout)return false;
    const group=this.device.createBindGroup({layout:this.layout,entries:[{binding:0,resource:inputs.position},{binding:1,resource:inputs.normal},{binding:2,resource:inputs.screenReflections}]});
    const pass=encoder.beginRenderPass({label:"Live SVO water reflection misses",colorAttachments:[{view:inputs.target,clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}]});
    pass.setPipeline(pipeline);pass.setBindGroup(0,scene);pass.setBindGroup(1,group);pass.draw(3);pass.end();return true;
  }
  destroy():void {this.destroyed=true;this.pipeline=undefined;this.transmissionPipeline=undefined;this.layout=undefined;}
}
