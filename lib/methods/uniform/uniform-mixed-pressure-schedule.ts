import type { UniformCM11aSchedule } from "./pressure-policy";
import { UNIFORM_MIXED_FAILURE as FAIL, uniformMixedFrameStatusWGSL } from "./uniform-mixed-frame-status";
import { uniformMixedPressureVerdictWGSL } from "./uniform-mixed-pressure-acceptance";

/** Frames a skipped V phase stays skipped before V-cycles are probed again. */
const UNIFORM_MIXED_SCHEDULE_V_PROBE=8;

/** Accepted frames whose largest cycle count the plan covers (at most 4:
 * control word 11 packs the previous three). */
const UNIFORM_MIXED_SCHEDULE_NEED_WINDOW=4;

/** One frame's encoded slot list: V-cycles, then Full-Cycles. */
export interface UniformMixedPressurePlan {readonly vCycles:number;readonly fullCycles:number}

/** Fewest slots a frame encodes, the host reserve included. The warm-started
 * root converges a quiet frame in one cycle, so a window of quiet frames
 * planned two; the dam 128^3's first far-wall impact then needed three (its
 * second-cycle residual sat at 2.5-5.4 against 5 across equivalent
 * trajectories) and failed. */
export const UNIFORM_MIXED_SCHEDULE_FLOOR=3;

/** Add capacity without reviving a V phase the planner dropped after a stall.
 * Positive surface coarsening needs another spare across abrupt impacts;
 * the ordinary GPU plan already includes its own one-cycle headroom. */
export function uniformMixedPressureReserve(plan:UniformMixedPressurePlan,maximum:UniformMixedPressurePlan,spare:number):UniformMixedPressurePlan{
 let {vCycles,fullCycles}=plan,remaining=Math.max(0,Math.floor(spare));
 if(fullCycles===0){const add=Math.min(remaining,maximum.vCycles-vCycles);vCycles+=add;remaining-=add;}
 const add=Math.min(remaining,maximum.fullCycles-fullCycles);fullCycles+=add;remaining-=add;
 if(vCycles>0)vCycles=Math.min(maximum.vCycles,vCycles+remaining);
 return {vCycles,fullCycles};
}

/** A V-only plan gives its last V slot to a Full-Cycle, so the stall jump
 * has somewhere to go; the slot count does not change. The dam 128^3
 * all-fine frame fails without it (V cycles at 0.5-0.7 of the cycle before
 * against a plan of three, 3 of 8 perturbed runs by frame 34), and no lagged
 * receipt separates those frames from their neighbours. */
export function uniformMixedPressureSpareFull(plan:UniformMixedPressurePlan,maximum:UniformMixedPressurePlan):UniformMixedPressurePlan{
 return plan.fullCycles===0&&plan.vCycles>1&&maximum.fullCycles>0?{vCycles:plan.vCycles-1,fullCycles:1}:plan;
}

/** GPU-resident CM11a schedule for the mixed pressure solve. The host encodes
 * the slot list the previous frame's planner chose (the conservative
 * schedule on the first frame): its V slots, its Full slots, then the
 * projection. Every launch is direct. A one-workgroup gate before each slot
 * first runs the verdict of the checkpoint before it (the initial one, or
 * the previous slot's when that slot ran), then applies its decision (the rules of
 * nextUniformPressureCorrection: a stalled V phase jumps to the encoded
 * Full-Cycles) and writes the slot gate word every kernel of a slot reads
 * first (the pressure root's, umSlotClosed), clearing the
 * candidate an open slot's checkpoint reduces into. A closed slot's
 * launches return at once. A latched frame status (uniform-mixed-frame-status)
 * closes every slot, this frame's and every later one's.
 * The projection gate is the frame's verdict: it accepts only a converged,
 * never-rejected solve whose h band fits.
 * Otherwise it latches the first cause in the frame status. It writes the
 * projection's gate, the band's closed word (UniformPressureBand.closedWord:
 * the band solve, projection and presentation stride no slots) and, on
 * acceptance, the status's last accepted frame. An accepted solve leaves
 * every word open.
 * The last gate plans the next frame from the cycles this one ran, plus one
 * spare, and never below the largest count of the last
 * UNIFORM_MIXED_SCHEDULE_NEED_WINDOW accepted frames plus the spare: a
 * stalled V phase is dropped (re-probed every
 * UNIFORM_MIXED_SCHEDULE_V_PROBE frames).
 *
 * Control words: 0 gate step, 1 next slot, 3 previous
 * residual, 4 cycles run, 5 current slot enabled, 6 V phase stalled,
 * 10 V skip streak, 11 the previous three accepted
 * frames' cycles (8 bits each), 12-13 this frame's V/Full
 * slots (host), 14-15 the next frame's (planner), 16-17 V/Full cycles run.
 * Diagnostics (read by FLUID_MIXED_PRESSURE_TRACE only): 7 any stall,
 * 8 initial residual, 18-24 each run slot's candidate residual.
 * Words 10-11 and 14-15 persist. The gate writes the cycle count into
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
  private readonly state:GPUBuffer,
  /** The slot gate word: the pressure root's support (umSlotClosed word
   * 9n+24). */
  private readonly words:{readonly fine:GPUBuffer;readonly supportWord:number;
   /** The frame status record and the band's index header (count, overflow,
    * closed word). */
   readonly status:GPUBuffer;readonly band:GPUBuffer;readonly bandClosedWord:number;
   /** The acceptance uniform (UniformMixedPressureAcceptanceFields.params). */
   readonly acceptance:GPUBuffer}){
  this.maximum={vCycles:schedule.vCycles,fullCycles:schedule.fullCycles};
  this.control=device.createBuffer({label:"Uniform mixed pressure schedule",size:128,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
 }
 async initialize():Promise<void>{
  const {vCycles:VMAX,fullCycles:FMAX}=this.maximum;
  const module=this.device.createShaderModule({label:"Uniform mixed pressure schedule gate",code:/* wgsl */`
@group(0) @binding(0) var<storage,read_write> control:array<u32,32>;
@group(0) @binding(1) var<storage,read_write> state:array<u32,8>;
@group(0) @binding(4) var<storage,read_write> fine:array<u32>;
@group(0) @binding(6) var<storage,read_write> band:array<u32,8>;
@group(0) @binding(7) var<uniform> acceptance:vec4f;
${uniformMixedPressureVerdictWGSL("acceptance")}
${uniformMixedFrameStatusWGSL(0,5,"read_write")}const VMAX=${VMAX}u;const FMAX=${FMAX}u;const SUPPORT_GATE=${this.words.supportWord}u;const BAND_CLOSED=${this.words.bandClosedWord}u;
fn umF(word:u32)->f32{return bitcast<f32>(control[word]);}
// The next frame's slot list, after an accepted solve:
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
 // Never fewer slots than the last UNIFORM_MIXED_SCHEDULE_NEED_WINDOW
 // accepted frames' largest cycle count plus the spare: the plan is lagged
 // two frames, and after an impact a frame's need swings between one and
 // three cycles from frame to frame (dam 128^3 at 1/60).
 let ran=vRan+fRan;let seen=control[11];
 var need=ran;for(var k=0u;k<${UNIFORM_MIXED_SCHEDULE_NEED_WINDOW-1}u;k++){need=max(need,(seen>>(8u*k))&0xffu);}
 control[11]=((seen<<8u)|ran)&${`0x${(2**(8*(UNIFORM_MIXED_SCHEDULE_NEED_WINDOW-1))-1).toString(16)}u`};
 var add=select(0u,need+1u-(v+f),need+1u>v+f);
 if(f==0u){let dv=min(add,VMAX-v);v+=dv;add-=dv;}
 let df=min(add,FMAX-f);f+=df;add-=df;
 if(control[6]==0u){v=min(VMAX,v+add);}
 control[14]=v;control[15]=f;control[10]=streak;
}
@compute @workgroup_size(1) fn main(){
  let step=control[0];control[0]=step+1u;
  // A closed slot's checkpoint reduced nothing and has no verdict.
  if(step==0u){umCheck(true);}else if(control[5]!=0u){umCheck(false);}
  let V=control[12];let F=control[13];let SLOTS=V+F;
  if(step==0u){
   control[1]=0u;control[3]=state[2];control[4]=0u;control[6]=0u;control[7]=0u;control[16]=0u;control[17]=0u;
   control[8]=state[2];for(var i=18u;i<25u;i++){control[i]=0u;}
  }else if(control[5]!=0u){
   // The slot just gated ran; its checkpoint wrote the accepted residual.
   let residual=bitcast<f32>(state[1]);
   let stalled=(state[1]&0x7f800000u)==0x7f800000u||residual>umF(3)*0.5;
   var next=step;
   if(stalled&&next<V&&F>0u){next=V;control[6]=1u;}
   control[1]=next;control[3]=state[1];control[4]+=1u;control[select(17u,16u,step-1u<V)]+=1u;control[17u+step]=state[0];
   if(stalled){control[7]=1u;}
  }
  var open=!umFrameFailed()&&state[5]==0u&&state[4]==0u&&control[1]==step;
  if(step==SLOTS){
   // The verdict: each of this frame's failures latches (the first cause
   // wins; a frame after a latched one ran nothing and adds none), then the
   // projection, the band and the plan open only on a healthy record.
   if(!umFrameFailed()){
    if(state[4]!=0u){umLatchFailure(select(${FAIL.pressureRejected}u,${FAIL.pressureNonfinite}u,state[0]>=0x7f800000u),state[0],state[1]);}
    else if(state[5]==0u){umLatchFailure(${FAIL.pressureUnconverged}u,state[1],control[4]);}
    if(band[1]!=0u){umLatchFailure(${FAIL.bandCapacity}u,band[0],0u);}
   }
   open=!umFrameFailed();
   if(open){umPlan();atomicStore(&umStatus[6],atomicLoad(&umStatus[7]));}
   band[BAND_CLOSED]=u32(!open);state[7]=control[4];
  }
  control[5]=u32(open);
  // An open slot's checkpoint reduces into a clear candidate.
  if(open&&step<SLOTS){state[0]=0u;}
  // The slot gate word (0 open).
  fine[SUPPORT_GATE]=u32(!open);
}
`});
  const info=await module.getCompilationInfo(),errors=info.messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  this.pipeline=await this.device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
  const w=this.words;
  const entries:readonly (readonly [number,GPUBuffer,number?])[]=[[0,this.control],[1,this.state,32],[4,w.fine],[5,w.status,64],[6,w.band,32],[7,w.acceptance,16]];
  this.group=this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:entries.map(([binding,buffer,size])=>({binding,resource:{buffer,size}}))});
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
 /** Diagnostics: copy the 32 control words (FLUID_MIXED_PRESSURE_TRACE). */
 encodeTraceCopy(encoder:GPUCommandEncoder,destination:GPUBuffer,offset:number):void{encoder.copyBufferToBuffer(this.control,0,destination,offset,128);}
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
