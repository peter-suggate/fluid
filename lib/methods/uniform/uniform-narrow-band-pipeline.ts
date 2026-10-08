import type { FluidPipelineGraph, FluidPipelineContext } from "../../core/fluid-pipeline";
import { UNIFORM_VOLUME_PIPELINE } from "./uniform-volume-pipeline";
import { uniformNarrowBandMethod } from "./uniform-narrow-band-method";

const particles=(context:FluidPipelineContext)=>(context.info as unknown as {narrowBandFlipParticles?:number}|null)?.narrowBandFlipParticles;
const clipped=(c:FluidPipelineContext)=>(c.info as unknown as {narrowBandFlipReseedClipped?:number}|null)?.narrowBandFlipReseedClipped??0;
const unsupported=(c:FluidPipelineContext)=>(c.info as unknown as {narrowBandFlipUnsupportedParticles?:number}|null)?.narrowBandFlipUnsupportedParticles??0;
const coarseParticles=(c:FluidPipelineContext)=>c.values.coarseParticleMode==="on";
const declared=new Set(uniformNarrowBandMethod.params!.map(p=>p.key));
/** These stages include the new work inside the existing timestamp seams. */
export const UNIFORM_NARROW_BAND_PIPELINE:FluidPipelineGraph={
 ...UNIFORM_VOLUME_PIPELINE,methodId:"uniform-narrow-band-flip",
 stages:UNIFORM_VOLUME_PIPELINE.stages.map(original=>{
  const stage={...original,controls:original.controls?.filter(c=>!("param" in c)||declared.has(c.param))};
  if(stage.id==="uniform-volume-support")return {...stage,chip:(c:FluidPipelineContext)=>(c.info?.uniformMixedFineTiles??0)===0?"4h support · no h tiles":stage.chip?.(c)};
  if(stage.id==="uniform-volume-phi")return {...stage,label:"FLIP surface reconstruction",
   tip:{...stage.tip,summary:"Advect particles and the interior level set with matching substepped RK4 traces and solid collision checks. Join the particle surface to the interior eroded by h, then reinitialize distance before pressure. Measure volume from this surface without target-volume corrections. Preserve the outer h layer and reseed the inner band after the projected velocity update. Unresolved droplets retain a gravity fallback and are reported in diagnostics. At the particle budget, defer new seeds while preserving existing samples. Automatic refinement retains swept particle support; explicit Requested regions still control FLIP coverage."},
   chip:(c:FluidPipelineContext)=>`${particles(c)?.toLocaleString()??"—"} particles · ${coarseParticles(c)?"5h seed band":"inner 5h band"}${unsupported(c)?` · ${unsupported(c).toLocaleString()} unresolved`:""}${clipped(c)?" · reseeding limited by particle budget":""}`};
  if(stage.id==="velocity-advection")return {...stage,label:"FLIP transfer + bulk momentum",
   tip:{...stage.tip,summary:"Semi-Lagrangian interior momentum and quadratic particle-to-grid velocity transfer in the surface band. A sharp switch at depth 2h uses particle velocity near the surface; the 5h particle band supplies its interpolation support. Extrapolate valid liquid-face velocities before saving the pre-force grid sample, then apply forces."},
   chip:(c:FluidPipelineContext)=>coarseParticles(c)?"surface P2G · 4h baseline · optional h":"surface P2G · Eulerian interior · 2h switch"};
  if(stage.id==="uniform-volume-band")return {...stage,label:"Band pressure + FLIP update",
   tip:{...stage.tip,summary:"The default solves the connected fine surface-pressure band using projected coarse flux. The all-4h experiment skips fine pressure when no h tiles are requested. Update particle velocities with 95% FLIP / 5% PIC from the total grid change across forces, global pressure and any band pressure. Use the same liquid-face extrapolation before and after pressure so clearing air faces cannot damp surface particles. Unsupported droplets receive gravity directly. Both pressure levels use the coupled liquid geometry."},chip:(c:FluidPipelineContext)=>!coarseParticles(c)?((c.info?.uniformMixedFineTiles??0)>0?"h surface band · FLIP":"no h regions · no FLIP particles"):(c.info?.uniformMixedFineTiles??0)>0?"requested h band · FLIP · particle surface":"no h solve · FLIP · particle surface"};
  if(stage.id==="density-post-process")return {...stage,label:"Surface publication",chip:(c:FluidPipelineContext)=>coarseParticles(c)?"coupled surface · 4h owners":"coupled particle + bulk surface",tip:{...stage.tip,summary:"Publish the same particle-defined surface used by the simulation and pressure. The coarse experiment samples that surface on dense render vertices without requiring h simulation owners."}};
  if(stage.id==="uniform-volume-coupling")return {...stage,label:"Volume transport",controls:undefined,toggle:undefined,state:()=>"off" as const,chip:()=>"surface-derived volume",tip:{reads:"phi",writes:"occupancy at geometry stage",feeds:"pressure",summary:"Liquid occupancy is measured from the reconstructed surface. NB-FLIP does not transport a separate conservative volume or shift the surface to a target volume."}};
  if(stage.id==="uniform-volume-gather")return {...stage,label:"Surface geometry",controls:undefined,chip:()=>"particle surface · measured volume",tip:{summary:"Build pressure geometry and liquid occupancy from the particle-defined level set. No target-volume shift or volume-recovery pressure source.",reads:"phi",writes:"geometry, measured volume",feeds:"pressure"}};
  if(stage.id==="uniform-volume-sharpen")return {...stage,toggle:undefined,controls:undefined,state:()=>"off" as const,chip:()=>"disabled for narrow-band FLIP"};
  return stage;
 }),
};
