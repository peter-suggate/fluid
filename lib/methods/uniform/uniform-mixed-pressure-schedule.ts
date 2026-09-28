import type { UniformCM11aSchedule } from "./pressure-policy";

/** Frames a skipped V phase stays skipped before V-cycles are probed again. */
const UNIFORM_MIXED_SCHEDULE_V_PROBE=8;

/** One frame's encoded slot list: V-cycles, then Full-Cycles. */
export interface UniformMixedPressurePlan {readonly vCycles:number;readonly fullCycles:number}

/** GPU-resident CM11a schedule for the mixed pressure solve. The host encodes
 * the slot list the previous frame's planner chose (the conservative
 * schedule on the first frame): its V slots, its Full slots, then the
 * projection. Every launch is direct. A one-workgroup gate before each slot
 * applies the previous slot's decision (the rules of
 * nextUniformPressureCorrection: a stalled V phase jumps to the encoded
 * Full-Cycles, a stall tightens coarse accuracy 1 -> 0.1 -> 0) and writes the
 * slot gate word every kernel of a slot reads first: the native hierarchy's
 * (mgSkipCycle) and the pressure root's (umSlotClosed). A closed slot's
 * launches return at once. Running out of slots unconverged withholds the
 * projection and is fatal on the host; the projection gate reopens the
 * native hierarchy, and an accepted solve leaves every word open.
 * The last gate plans the next frame from the cycles this one ran, plus one
 * spare: a stalled V phase is dropped (re-probed every
 * UNIFORM_MIXED_SCHEDULE_V_PROBE frames), and the coarse accuracy a stall
 * forced is kept, relaxing one step after a stall-free solve.
 *
 * Control words: 0 gate step, 1 next slot, 2 coarse accuracy, 3 previous
 * residual, 4 cycles run, 5 current slot enabled, 6 V phase stalled, 7 any
 * stall, 9 planned accuracy, 10 V skip streak, 12-13 this frame's V/Full
 * slots (host), 14-15 the next frame's (planner), 16-17 V/Full cycles run.
 * Words 9-10 and 14-15 persist. The gate writes the cycle count into
 * acceptance word 7. */
export class UniformMixedPressureSchedule {
 /** The schedule's cap per kind; plans never exceed it. */
 readonly maximum:UniformMixedPressurePlan;
 private plan?:UniformMixedPressurePlan;
 get slots():number{if(!this.plan)throw new Error("Mixed pressure schedule has no frame open");return this.plan.vCycles+this.plan.fullCycles;}
 private readonly control:GPUBuffer;
 private pipeline?:GPUComputePipeline;
 private group?:GPUBindGroup;
 private gates=0;
 constructor(private readonly device:GPUDevice,private readonly schedule:UniformCM11aSchedule,
  private readonly state:GPUBuffer,private readonly tolerance:GPUBuffer,
  /** The slot gate words: the native hierarchy's state (mgSkipCycle word
   * 17) and the pressure root's support (umSlotClosed word 9n+24). */
  private readonly words:{readonly native:GPUBuffer;readonly fine:GPUBuffer;readonly supportWord:number}){
  this.maximum={vCycles:schedule.vCycles,fullCycles:schedule.fullCycles};
  this.control=device.createBuffer({label:"Uniform mixed pressure schedule",size:128,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
  device.queue.writeBuffer(this.control,36,new Float32Array([1]));
 }
 async initialize():Promise<void>{
  const {vCycles:VMAX,fullCycles:FMAX}=this.maximum;
  const module=this.device.createShaderModule({label:"Uniform mixed pressure schedule gate",code:/* wgsl */`
@group(0) @binding(0) var<storage,read_write> control:array<u32,32>;
@group(0) @binding(1) var<storage,read_write> state:array<u32,8>;
@group(0) @binding(2) var<storage,read_write> tolerance:array<f32,4>;
@group(0) @binding(3) var<storage,read_write> native:array<u32>;
@group(0) @binding(4) var<storage,read_write> fine:array<u32>;
const VMAX=${VMAX}u;const FMAX=${FMAX}u;const SUPPORT_GATE=${this.words.supportWord}u;
fn umF(word:u32)->f32{return bitcast<f32>(control[word]);}
// The next frame's slot list and coarse accuracy, after an accepted solve:
// the cycles of each kind this frame ran, one spare where the solve ended.
fn umPlan(){
 let V=control[12];let vRan=control[16];let fRan=control[17];
 var v=min(VMAX,vRan+select(0u,1u,fRan==0u));var f=select(0u,min(FMAX,fRan+1u),fRan>0u);
 // A V phase at its cap keeps a Full slot to jump to.
 if(fRan==0u&&vRan+1u>VMAX){f=min(FMAX,1u);}
 var streak=0u;
 if(control[6]!=0u){v=0u;f=min(FMAX,fRan+1u);}
 else if(V==0u){streak=control[10]+1u;if(streak>=${UNIFORM_MIXED_SCHEDULE_V_PROBE}u){v=1u;streak=0u;}}
 if(v+f==0u){v=min(VMAX,1u);f=select(0u,1u,v==0u);}
 control[14]=v;control[15]=f;control[10]=streak;
 let a=umF(2);
 control[9]=bitcast<u32>(select(select(select(0.1,1.0,a!=0.0),a,a==1.0),a,control[7]!=0u));
}
@compute @workgroup_size(1) fn main(){
  let step=control[0];control[0]=step+1u;
  let V=control[12];let F=control[13];let SLOTS=V+F;
  if(step==0u){
   control[1]=0u;control[2]=control[9];control[3]=state[2];control[4]=0u;control[6]=0u;control[7]=0u;control[16]=0u;control[17]=0u;
  }else if(control[5]!=0u){
   // The slot just gated ran; its checkpoint wrote the accepted residual.
   let residual=bitcast<f32>(state[1]);
   let stalled=(state[1]&0x7f800000u)==0x7f800000u||residual>umF(3)*0.5;
   var next=step;
   if(stalled&&next<V&&F>0u){next=V;control[6]=1u;}
   control[1]=next;control[3]=state[1];control[4]+=1u;control[select(17u,16u,step-1u<V)]+=1u;
   if(stalled){control[7]=1u;control[2]=bitcast<u32>(select(0.0,0.1,umF(2)==1.0));}
  }
  var open=state[5]==0u&&state[4]==0u&&control[1]==step;
  if(step==SLOTS){open=state[5]!=0u&&state[4]==0u;if(open){umPlan();}state[7]=control[4];}
  control[5]=u32(open);
  // setCoarseAccuracy's words, for the native 4h continuation.
  let a=umF(2);tolerance[2]=0.1*a;tolerance[3]=tolerance[0]*0.1*a;
  // The slot gate words (0 open). The projection launches on the root
  // alone, so its gate reopens the native hierarchy.
  let closed=u32(!open);let cycle=step<SLOTS;
  native[17]=select(0u,closed,cycle);fine[SUPPORT_GATE]=closed;
}
`});
  const info=await module.getCompilationInfo(),errors=info.messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  this.pipeline=await this.device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
  const w=this.words;
  this.group=this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[this.control,this.state,this.tolerance,w.native,w.fine]
   .map((buffer,binding)=>({binding,resource:{buffer,size:binding===1?32:binding===2?16:binding===3?112:undefined}}))});
 }
 /** Starts a frame's slot list for `plan`. The step counter restarts here. */
 begin(plan:UniformMixedPressurePlan):void{
  if(!this.pipeline)throw new Error("Mixed pressure schedule is not initialized");
  if(![plan.vCycles,plan.fullCycles].every(n=>Number.isInteger(n)&&n>=0)||plan.vCycles>this.maximum.vCycles||plan.fullCycles>this.maximum.fullCycles||plan.vCycles+plan.fullCycles===0)
   throw new Error(`Invalid mixed pressure plan ${JSON.stringify(plan)}`);
  this.plan=plan;this.gates=0;
  this.device.queue.writeBuffer(this.control,0,new Uint32Array(1));
  this.device.queue.writeBuffer(this.control,48,new Uint32Array([plan.vCycles,plan.fullCycles]));
 }
 /** Copy the planner's next slot list (two u32: V, Full) for the host. */
 encodePlanCopy(encoder:GPUCommandEncoder,destination:GPUBuffer,offset:number):void{
  encoder.copyBufferToBuffer(this.control,56,destination,offset,8);
 }
 /** Encode gate `slot` (the projection is slot `slots`). The slot's kernels
  * that follow read its gate word themselves; clears and copies encoded
  * inside a slot run whether or not it is open (only per-cycle scratch). */
 gate(encoder:GPUCommandEncoder,slot:number):GPUCommandEncoder{
  if(slot!==this.gates||slot>this.slots)throw new Error(`Mixed pressure gate ${slot} encoded out of order`);
  this.gates++;
  const pass=encoder.beginComputePass({label:`Uniform mixed pressure gate ${slot}`});
  pass.setPipeline(this.pipeline!);pass.setBindGroup(0,this.group!);pass.dispatchWorkgroups(1);pass.end();
  return encoder;
 }
 /** Close the frame's slot list. Call after the last gate. */
 end():void{
  if(this.gates!==this.slots+1)throw new Error(`Mixed pressure schedule encoded ${this.gates} of ${this.slots+1} gates`);
 }
 destroy():void{this.control.destroy();}
}
