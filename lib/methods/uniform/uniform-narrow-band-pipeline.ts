import type { FluidPipelineGraph, FluidPipelineContext } from "../../core/fluid-pipeline";
import { UNIFORM_VOLUME_PIPELINE } from "./uniform-volume-pipeline";
import { uniformNarrowBandMethod } from "./uniform-narrow-band-method";

const particles=(context:FluidPipelineContext)=>(context.info as unknown as {narrowBandFlipParticles?:number}|null)?.narrowBandFlipParticles;
const declared=new Set(uniformNarrowBandMethod.params!.map(p=>p.key));
/** These stages include the new work inside the existing timestamp seams. */
export const UNIFORM_NARROW_BAND_PIPELINE:FluidPipelineGraph={
 ...UNIFORM_VOLUME_PIPELINE,methodId:"uniform-narrow-band-flip",
 stages:UNIFORM_VOLUME_PIPELINE.stages.map(original=>{
  const stage={...original,controls:original.controls?.filter(c=>!("param" in c)||declared.has(c.param))};
  if(stage.id==="uniform-volume-phi")return {...stage,label:"Band particles + level set",
   tip:{...stage.tip,summary:"Advect persistent near-surface velocity samples with substepped RK2 trajectories and solid collision checks. Advect the geometric level set. This prototype retains geometric redistance and volume reconciliation."},
   chip:(c:FluidPipelineContext)=>`${particles(c)?.toLocaleString()??"—"} particles · inner 4h band`};
  if(stage.id==="velocity-advection")return {...stage,label:"FLIP transfer + bulk momentum",
   tip:{...stage.tip,summary:"Semi-Lagrangian interior momentum, reseeding and quadratic particle-to-grid velocity transfer in the surface band. A 2h overlap joins the fields. Save the pre-force grid sample, then apply forces."},
   chip:()=>"surface P2G · Eulerian interior · 2h overlap"};
  if(stage.id==="uniform-volume-band")return {...stage,label:"h band pressure + FLIP update",
   tip:{...stage.tip,summary:"Solve the connected fine pressure band using the projected coarse flux. Update particle velocities with 95% FLIP / 5% PIC from the total grid change across forces, global pressure and band pressure."}};
  if(stage.id==="uniform-volume-sharpen")return {...stage,toggle:undefined,controls:undefined,state:()=>"off" as const,chip:()=>"disabled for narrow-band FLIP"};
  return stage;
 }),
};
