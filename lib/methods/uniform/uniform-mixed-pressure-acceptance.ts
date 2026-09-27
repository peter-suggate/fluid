import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformMixedPressureBoundaryIndexWGSL, uniformMixedPressureBoundaryLoop, uniformMixedPressureStorage } from "./uniform-mixed-pressure-boundary.wgsl";

export interface UniformMixedPressureAcceptanceFields {
 residual:GPUBufferBinding;
 /** Eight u32 words; reset once before the initial checkpoint. */
 state:GPUBuffer;
 /** dt/rho, absolute divergence tolerance, unused, unused. */
 params:GPUBuffer;
}
/** GPU-only convergence gate. A nonfinite or worsening iterate latches a
 * terminal failure. This stage cannot mutate pressure or launch repair work.
 * Root residuals include the projected wall constraints. */
export class UniformMixedPressureAcceptance {
 readonly allocatedBytes=0;
 private readonly resources:GPUBindGroupLayout;
 private readonly pipelines=new Map<string,GPUComputePipeline>();
 /** The receipt each group binds: the reset kernels clear it in the pass. */
 private readonly states=new WeakMap<GPUBindGroup,GPUBuffer>();
 constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership){
  this.resources=device.createBindGroupLayout({entries:[...[0,1].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}})),
   {binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}]});
 }
 bind(f:UniformMixedPressureAcceptanceFields):GPUBindGroup{
  const size=4*uniformMixedPressureStorage(this.ownership.layout).count;
  const group=this.device.createBindGroup({layout:this.resources,entries:[
   {binding:0,resource:{...f.residual,size}},
   {binding:1,resource:{buffer:f.state,size:32}},
   {binding:2,resource:{buffer:f.params,size:16}},
  ]});
  this.states.set(group,f.state);return group;
 }
 async initialize():Promise<void>{
  const l=this.ownership.layout;
  const module=this.device.createShaderModule({code:uniformMixedTopologyWGSL(l,0)+/* wgsl */`
@group(1) @binding(0) var<storage,read_write> residual:array<f32>;
@group(1) @binding(1) var<storage,read_write> state:array<atomic<u32>>;
@group(1) @binding(2) var<uniform> params:vec4f;
${uniformMixedPressureBoundaryIndexWGSL(l)}
var<workgroup> maxima:array<u32,64>;
@compute @workgroup_size(64) fn reduce(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 let o=umAllOwner(gid);var norm=0.0;
 if(o.width!=0u){norm=residual[o.index];${uniformMixedPressureBoundaryLoop("norm=max(norm,residual[halo]);")}}
 maxima[lane]=select(bitcast<u32>(norm*params.x),0x7f800000u,norm>=3.402823e38);workgroupBarrier();
 for(var stride=32u;stride>0u;stride/=2u){if(lane<stride){maxima[lane]=max(maxima[lane],maxima[lane+stride]);}workgroupBarrier();}
 // state[0] >= 0u: a zero group maximum (all-air groups) is a no-op max.
 if(lane==0u&&maxima[0]!=0u){atomicMax(&state[0],maxima[0]);}
}
fn umCheck(initial:bool){
 let candidate=atomicLoad(&state[0]);
 let finite=candidate<0x7f800000u;
 let good=finite&&(initial||candidate<=atomicLoad(&state[1]));
 if(initial){atomicStore(&state[2],candidate);}
 if(atomicLoad(&state[4])!=0u){return;}
 atomicStore(&state[3],select(1u,0u,good));
 if(!good){atomicStore(&state[4],1u);atomicStore(&state[5],0u);atomicAdd(&state[6],1u);return;}
 atomicStore(&state[1],candidate);
 atomicStore(&state[5],select(0u,1u,params.y>0.0&&bitcast<f32>(candidate)<=params.y));
}
// clearBuffer of the candidate (a cycle) or the whole receipt (initial), in the pass.
@compute @workgroup_size(1) fn resetInitial(){for(var i=0u;i<8u;i++){atomicStore(&state[i],0u);}}
@compute @workgroup_size(1) fn resetCycle(){atomicStore(&state[0],0u);}
@compute @workgroup_size(1) fn initial(){umCheck(true);}
@compute @workgroup_size(1) fn cycle(){umCheck(false);}

`});
  const info=await module.getCompilationInfo(),errors=info.messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources]});
  for(const entryPoint of ["resetInitial","resetCycle","reduce","initial","cycle"])this.pipelines.set(entryPoint,await this.device.createComputePipelineAsync({layout,compute:{module,entryPoint,constants:{umDispatchX:this.ownership.dispatchX}}}));
 }
 encode(encoder:GPUCommandEncoder,group:GPUBindGroup,state:GPUBuffer,kind:"initial"|"cycle"):void{
  if(this.pipelines.size!==5)throw new Error("Mixed pressure acceptance is not initialized");
  if(this.states.get(group)!==state)throw new Error("Mixed pressure acceptance group does not bind this state");
  const pass=encoder.beginComputePass({label:`Uniform mixed pressure ${kind} checkpoint`});pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group);
  for(const entry of [kind==="initial"?"resetInitial":"resetCycle","reduce",kind]){const pipeline=this.pipelines.get(entry)!;pass.setPipeline(pipeline);
   if(entry==="reduce")this.ownership.dispatchAll(pass,pipeline);else pass.dispatchWorkgroups(1);
  }
  pass.end();
 }
}
