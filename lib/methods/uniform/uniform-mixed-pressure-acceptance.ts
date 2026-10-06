import { uniformDetailModule, uniformDetailPipeline, uniformDetailPick } from "./uniform-detail-fields";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { UNIFORM_MIXED_COUNTED, uniformMixedCountedEntriesWGSL, uniformMixedPageCount, uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformMixedPressureBoundaryIndexWGSL, uniformMixedPressureBoundaryLoop, uniformMixedPressureStorage } from "./uniform-mixed-pressure-boundary.wgsl";

export interface UniformMixedPressureAcceptanceFields {
 residual:GPUBufferBinding;
 /** Eight u32 words; reset once before the initial checkpoint. */
 state:GPUBuffer;
 /** dt/rho, h-equivalent divergence tolerance (1/s), relative reduction of
  * the initial residual, absolute floor of the relative bound (1/s). */
 params:GPUBuffer;
}
/** The checkpoint verdict on the reduced candidate (state[0]), run by the
 * pressure schedule's next gate (one lane, `params` the acceptance uniform):
 * a nonfinite or worsening candidate latches a terminal failure. The initial
 * verdict records the initial norm. */
export const uniformMixedPressureVerdictWGSL=(params:string)=>/* wgsl */`
fn umCheck(initial:bool){
 let candidate=state[0];
 let finite=candidate<0x7f800000u;
 let good=finite&&(initial||candidate<=state[1]);
 if(initial){state[2]=candidate;}
 if(state[4]!=0u){return;}
 state[3]=select(1u,0u,good);
 if(!good){state[4]=1u;state[5]=0u;state[6]+=1u;return;}
 state[1]=candidate;
 // state[2] is the initial norm (the RHS for the band's p=0 start; the
 // warm-started root's carried residual): the absolute tolerance and a
 // relative reduction of it must both hold, so the start is never the answer.
 let bound=min(${params}.y,max(${params}.z*bitcast<f32>(state[2]),${params}.w));
 state[5]=select(0u,1u,${params}.y>0.0&&bitcast<f32>(candidate)<=bound);
}`;
/** GPU-only convergence measure: the reduction of a checkpoint's candidate
 * norm, whose verdict (uniformMixedPressureVerdictWGSL) the schedule's next
 * gate runs, and which that gate clears before each open slot. This stage
 * cannot mutate pressure or launch repair work.
 * Root residuals include the projected wall constraints.
 *
 * The norm is h-equivalent: an owner's divergence times its width. A face
 * velocity error du is a divergence du/(w h), so an absolute divergence bound
 * lets a width-w owner keep w times the velocity error of an h cell (an
 * all-4h solve accepted a resting pool's uncancelled gravity for 3 frames).
 * The reduction strides the resident pages' owners (residentAll): an absent
 * page is certified far air whose residual the root setup zeroes. The all-4h
 * ownership has no h jobs, so its launch is sized to its pages. */
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
  const module=uniformDetailModule(this.device,{label:"Uniform mixed pressure acceptance",code:uniformMixedCountedEntriesWGSL(uniformMixedTopologyWGSL(l,0)+/* wgsl */`
@group(1) @binding(0) var<storage,read_write> residual:array<f32>;
@group(1) @binding(1) var<storage,read_write> state:array<atomic<u32>>;
@group(1) @binding(2) var<uniform> params:vec4f;
${uniformMixedPressureBoundaryIndexWGSL(l)}
var<workgroup> maxima:array<u32,64>;
@compute @workgroup_size(64) fn reduce(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 // A closed slot reduces nothing: no early return ahead of the barriers.
 var o=umResidentAllOwner(gid);if(umSlotClosed()){o=UMOwner();}var norm=0.0;
 if(o.width!=0u){norm=residual[o.index];${uniformMixedPressureBoundaryLoop("norm=max(norm,residual[halo]);")}norm*=f32(o.width);}
 maxima[lane]=select(bitcast<u32>(norm*params.x),0x7f800000u,norm>=3.402823e38);workgroupBarrier();
 for(var stride=32u;stride>0u;stride/=2u){if(lane<stride){maxima[lane]=max(maxima[lane],maxima[lane+stride]);}workgroupBarrier();}
 // state[0] >= 0u: a zero group maximum (all-air groups) is a no-op max.
 if(lane==0u&&maxima[0]!=0u){atomicMax(&state[0],maxima[0]);}
}
// clearBuffer of the whole receipt before the initial reduction, in the pass.
@compute @workgroup_size(1) fn resetInitial(){for(var i=0u;i<8u;i++){atomicStore(&state[i],0u);}}

`,["reduce"])});
  const info=await module.getCompilationInfo(),errors=info.messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources]});
  await Promise.all(["resetInitial","reduce"].map(async entryPoint=>{this.pipelines.set(entryPoint,await uniformDetailPipeline(this.device,this.ownership,{layout,compute:{module,entryPoint,constants:{umDispatchX:this.ownership.dispatchX,umCountedJobs:UNIFORM_MIXED_COUNTED.residentAll}}}));}));
 }
 encode(encoder:GPUCommandEncoder,group:GPUBindGroup,state:GPUBuffer,kind:"initial"|"cycle"):void{
  if(this.pipelines.size!==2)throw new Error("Mixed pressure acceptance is not initialized");
  if(this.states.get(group)!==state)throw new Error("Mixed pressure acceptance group does not bind this state");
  const pass=encoder.beginComputePass({label:`Uniform mixed pressure ${kind} checkpoint`});pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group);
  for(const entry of kind==="initial"?["resetInitial","reduce"]:["reduce"]){const pipeline=uniformDetailPick(this.pipelines.get(entry)!);pass.setPipeline(pipeline);
   if(entry==="reduce")this.ownership.dispatchCounted(pass,pipeline,uniformMixedPageCount(this.ownership.capacity.lattice));else pass.dispatchWorkgroups(1);
  }
  pass.end();
 }
}
