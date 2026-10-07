import type { FluidPipelineGraph, FluidPipelineContext } from "../../core/fluid-pipeline";
import { UNIFORM_VOLUME_PIPELINE } from "./uniform-volume-pipeline";
import { uniformNarrowBandMethod } from "./uniform-narrow-band-method";

const particles=(context:FluidPipelineContext)=>(context.info as unknown as {narrowBandFlipParticles?:number}|null)?.narrowBandFlipParticles;
const coarseParticles=(c:FluidPipelineContext)=>c.values.coarseParticleMode==="on";
const declared=new Set(uniformNarrowBandMethod.params!.map(p=>p.key));
/** These stages include the new work inside the existing timestamp seams. */
export const UNIFORM_NARROW_BAND_PIPELINE:FluidPipelineGraph={
 ...UNIFORM_VOLUME_PIPELINE,methodId:"uniform-narrow-band-flip",
 stages:UNIFORM_VOLUME_PIPELINE.stages.map(original=>{
  const stage={...original,controls:original.controls?.filter(c=>!("param" in c)||declared.has(c.param))};
  if(stage.id==="uniform-volume-support")return {...stage,chip:(c:FluidPipelineContext)=>(c.info?.uniformMixedFineTiles??0)===0?"4h support · no h tiles":stage.chip?.(c)};
  if(stage.id==="uniform-volume-phi")return {...stage,label:"Band particles + level set",
   tip:{...stage.tip,summary:"Advect persistent near-surface velocity samples with substepped RK2 trajectories and solid collision checks. Advect the geometric level set. The default uses the geometric surface and seeds particles only inside the selected h surface regions. Coarsened regions retire their particles. The all-4h experiment reconstructs the visible surface from particles, independently of the geometric bulk."},
   chip:(c:FluidPipelineContext)=>`${particles(c)?.toLocaleString()??"—"} particles · ${coarseParticles(c)?"4h seed band":"inner 4h band"}`};
  if(stage.id==="velocity-advection")return {...stage,label:"FLIP transfer + bulk momentum",
   tip:{...stage.tip,summary:"Semi-Lagrangian interior momentum, reseeding and quadratic particle-to-grid velocity transfer in the surface band. A 2h overlap joins the fields. Save the pre-force grid sample, then apply forces."},
   chip:(c:FluidPipelineContext)=>coarseParticles(c)?"surface P2G · 4h baseline · optional h":"surface P2G · Eulerian interior · 2h overlap"};
  if(stage.id==="uniform-volume-band")return {...stage,label:"Band pressure + FLIP update",
   tip:{...stage.tip,summary:"The default solves the connected fine surface-pressure band using projected coarse flux. The all-4h experiment skips fine pressure when no h tiles are requested. Update particle velocities with 95% FLIP / 5% PIC from the total grid change across forces, global pressure and any band pressure. In the experiment, reconstruct the particle surface independently of simulation refinement."},chip:(c:FluidPipelineContext)=>!coarseParticles(c)?((c.info?.uniformMixedFineTiles??0)>0?"h surface band · FLIP":"no h regions · no FLIP particles"):(c.info?.uniformMixedFineTiles??0)>0?"requested h band · FLIP · particle surface":"no h solve · FLIP · particle surface"};
  if(stage.id==="density-post-process")return {...stage,label:"Surface publication",chip:(c:FluidPipelineContext)=>coarseParticles(c)?"particle surface + coarse interior":"geometric surface",tip:{...stage.tip,summary:"Publish the geometric surface by default. The all-4h experiment instead publishes a particle-reconstructed surface with a coarse bulk interior; render vertices do not create h simulation tiles."}};
  if(stage.id==="uniform-volume-sharpen")return {...stage,toggle:undefined,controls:undefined,state:()=>"off" as const,chip:()=>"disabled for narrow-band FLIP"};
  return stage;
 }),
};
