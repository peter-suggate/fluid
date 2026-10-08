import { narrowBandActivityParams, narrowBandActivityValues } from "./uniform-narrow-band-controls";
import type { MethodParamValues, SimulationMethod } from "../../core/method-contract";
import { uniformVolumeMethod } from "./uniform-volume-method";
import { uniformGeometricSolverOptions } from "./uniform-geometric-options";
import { resolveUniformGeometricValues } from "./uniform-geometric-parameters";
import { WebGPUUniformReferenceSolver } from "./webgpu-uniform-reference";

/** A 4h particle overlap, with either fixed or activity-driven surface coverage. */
const fixed:MethodParamValues={sharpeningSweeps:0,sharpeningDistance:0,totalSurfaceVolume:"on",surfaceVolumeRounds:2,surfaceDeficitBalancing:"off",volumeDustThreshold:0,orphanDustThreshold:0,phiDrain:"off"};
// Fine tiles at surface crossings, and nothing else: shape at tolerance 0 asks
// for every surface tile, the distance filter keeps only the crossing ones, and
// no other criterion, margin or hold is on. Particle swept-support joins and
// mandatory solid promotion are not elective and stay.
export const narrowBandFixedDetail:MethodParamValues={detailPolicy:"dynamic",detailSolidContact:"on",detailSurface:"on",detailSurfaceDistance:0,
 detailShape:"on",detailShapeTolerance:0,detailShapeMetric:"value",detailThin:"off",detailStrain:"off",detailRotation:"off",detailImpact:"off",detailApproach:"off",
 detailNearFocus:"off",detailBulk:"off",detailMarginTiles:0,detailHoldSteps:0};
/** The app's adaptive profile. Particle heat supplies the hold in physical
 * time, so the tile census needs no additional step-count hold. */
export const narrowBandAdaptiveDetail:MethodParamValues={...narrowBandFixedDetail,...narrowBandActivityValues(),adaptiveSurface:"on",
 detailShapeTolerance:0.5,detailShapeMetric:"displacement",detailThin:"on",detailStrain:"on",detailImpact:"on",detailApproach:"on"};
export const narrowBandFlipValues=(values:MethodParamValues={})=>({
 ...resolveUniformGeometricValues({...narrowBandFixedDetail,...values}),...narrowBandActivityValues(values),...fixed,
 fineGridPadding:Math.max(0,Math.min(4,Number.isFinite(Number(values.fineGridPadding))?Number(values.fineGridPadding):1)),
 adaptiveSurface:values.adaptiveSurface==="on"?"on":"off",
 coarseParticleMode:values.coarseParticleMode==="on"?"on":"off",
});
// Derive the renderer lifetime key from this method's own schema; inheriting
// Uniform Geometric's list misses NB-specific live controls.
const params:SimulationMethod["params"]=[...narrowBandActivityParams,{kind:"select",key:"adaptiveSurface",label:"Adaptive surface",default:"off",tier:"coarse",update:"solver",dedicated:true,options:[{value:"off",label:"Fixed particle band"},{value:"on",label:"Activity transition"}],hint:"In Dynamic detail, cool and retire particles where the refinement criteria permit a coarse surface. Use a nonzero shape tolerance to release calm h tiles. Full and Requested retain the fixed particle band."},{kind:"number",key:"fineGridPadding",label:"Fine-grid padding",default:1,tier:"fine",update:"runtime",min:0,max:4,step:1,digits:0,unit:"h",hint:"Extra h cells around swept surface crossings. The particle band stays 4h wide. Zero padding is experimental."},{kind:"select",key:"coarseParticleMode",label:"Experimental all-4h FLIP",default:"off",tier:"coarse",update:"solver",dedicated:true,
  options:[{value:"off",label:"Off"},{value:"on",label:"On"}],hint:"Publish a separate particle render surface for the all-4h experiment; retain the selected refinement policy. Restarts the simulation."},...uniformVolumeMethod.params!.filter(p=>!Object.hasOwn(fixed,p.key)).map(p=>Object.hasOwn(narrowBandFixedDetail,p.key)?{...p,default:narrowBandFixedDetail[p.key]} as typeof p:p)];
export const uniformNarrowBandMethod:SimulationMethod={
 ...uniformVolumeMethod,
 id:"uniform-narrow-band-flip",label:"Uniform Narrow-band FLIP",shortLabel:"Narrow-band FLIP",badge:"NARROW-BAND FLIP · EXPERIMENTAL",
 description:"Activity-driven FLIP particles preserve detailed surfaces while calm regions cool to an Eulerian level set at 4h. Shape error, thin features and approaching contact retain fine coverage. Full and Requested detail use the fixed particle band. Volume is measured against sources and outflow, not corrected.",
 resource:{...uniformVolumeMethod.resource!,id:"fluid.uniform-narrow-band-flip",label:"Narrow-band FLIP fluid"},
 // The geometric method's layers, plus the band's velocity samples as spheres.
 capabilities:{...uniformVolumeMethod.capabilities,visualLayers:{hidden:["pages","window","release"]}},
 params,
 runtimeParamKeys:params.filter(p=>p.update==="runtime").map(p=>p.key),
 appDefaults:{...uniformVolumeMethod.appDefaults,...narrowBandAdaptiveDetail,...fixed,fineGridPadding:1,coarseParticleMode:"off"},
 normalizeValues:narrowBandFlipValues,
 createSolverAsync:(device,scene,quality,values,loads,progress,signal)=>WebGPUUniformReferenceSolver.createAsync(device,scene,quality,loads,
  {...uniformGeometricSolverOptions(narrowBandFlipValues(values),scene),narrowBandFlip:true,narrowBandAdaptiveSurface:values.adaptiveSurface==="on",narrowBandAdaptiveBudgetPercent:narrowBandActivityValues(values).adaptiveBudgetPercent,narrowBandAdaptiveFadeSeconds:narrowBandActivityValues(values).adaptiveFadeSeconds,narrowBandFinePadding:narrowBandFlipValues(values).fineGridPadding,narrowBandCoarseParticles:values.coarseParticleMode==="on",retainStageDiagnosticsForQA:values.retainStageDiagnosticsForQA===true,sharpeningSweeps:0,sharpeningDistance:0},progress,signal),
 pipelineGraph:async()=> (await import("./uniform-narrow-band-pipeline")).UNIFORM_NARROW_BAND_PIPELINE,
 harness:async()=>({...await import("./harness").then(m=>m.uniformHarnessPlugin),methodId:"uniform-narrow-band-flip"}),
};
