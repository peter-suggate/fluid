import type { MethodParamValues, SimulationMethod } from "../../core/method-contract";
import { uniformVolumeMethod } from "./uniform-volume-method";
import { uniformGeometricSolverOptions } from "./uniform-geometric-options";
import { resolveUniformGeometricValues } from "./uniform-geometric-parameters";
import { WebGPUUniformReferenceSolver } from "./webgpu-uniform-reference";

/** A fixed 4h particle band with independently controlled grid coverage. */
const fixed:MethodParamValues={sharpeningSweeps:0,sharpeningDistance:0,totalSurfaceVolume:"on",surfaceVolumeRounds:2,surfaceDeficitBalancing:"off",volumeDustThreshold:0,orphanDustThreshold:0,phiDrain:"off"};
// Fine tiles at surface crossings, and nothing else: shape at tolerance 0 asks
// for every surface tile, the distance filter keeps only the crossing ones, and
// no other criterion, margin or hold is on. Particle swept-support joins and
// mandatory solid promotion are not elective and stay.
const fineBand:MethodParamValues={detailPolicy:"dynamic",detailSolidContact:"on",detailSurface:"on",detailSurfaceDistance:0,
 detailShape:"on",detailShapeTolerance:0,detailThin:"off",detailStrain:"off",detailRotation:"off",detailImpact:"off",detailApproach:"off",
 detailNearFocus:"off",detailBulk:"off",detailMarginTiles:0,detailHoldSteps:0};
export const narrowBandFlipValues=(values:MethodParamValues={})=>({
 ...resolveUniformGeometricValues({...fineBand,...values}),...fixed,
 fineGridPadding:Math.max(0,Math.min(4,Number.isFinite(Number(values.fineGridPadding))?Number(values.fineGridPadding):1)),
 coarseParticleMode:values.coarseParticleMode==="on"?"on":"off",
});
export const uniformNarrowBandMethod:SimulationMethod={
 ...uniformVolumeMethod,
 id:"uniform-narrow-band-flip",label:"Uniform Narrow-band FLIP",shortLabel:"Narrow-band FLIP",badge:"NARROW-BAND FLIP · EXPERIMENTAL",
 description:"FLIP particles in a band under the surface carry its velocity and overrule the advected level set there (Ferstl et al. 2016): the sheets and droplets they carry are liquid. The interior is a grid. Volume is measured against a budget of sources and open-top outflow and, as in FLIP, not corrected.",
 resource:{...uniformVolumeMethod.resource!,id:"fluid.uniform-narrow-band-flip",label:"Narrow-band FLIP fluid"},
 // The geometric method's layers, plus the band's velocity samples as spheres.
 capabilities:{...uniformVolumeMethod.capabilities,visualLayers:{hidden:["pages","window","release"]}},
 params:[{kind:"number",key:"fineGridPadding",label:"Fine-grid padding",default:1,tier:"fine",update:"runtime",min:0,max:4,step:1,digits:0,unit:"h",hint:"Extra h cells around swept surface crossings. The particle band stays 4h wide. Zero padding is experimental."},{kind:"select",key:"coarseParticleMode",label:"Experimental all-4h FLIP",default:"off",tier:"coarse",update:"solver",dedicated:true,
  options:[{value:"off",label:"Off"},{value:"on",label:"On"}],hint:"Publish a separate particle render surface for the all-4h experiment; retain the selected refinement policy. Restarts the simulation."},...uniformVolumeMethod.params!.filter(p=>!Object.hasOwn(fixed,p.key)).map(p=>Object.hasOwn(fineBand,p.key)?{...p,default:fineBand[p.key]} as typeof p:p)],
 appDefaults:{...uniformVolumeMethod.appDefaults,...fineBand,...fixed,fineGridPadding:1,coarseParticleMode:"off"},
 normalizeValues:narrowBandFlipValues,
 createSolverAsync:(device,scene,quality,values,loads,progress,signal)=>WebGPUUniformReferenceSolver.createAsync(device,scene,quality,loads,
  {...uniformGeometricSolverOptions(narrowBandFlipValues(values),scene),narrowBandFlip:true,narrowBandFinePadding:narrowBandFlipValues(values).fineGridPadding,narrowBandCoarseParticles:values.coarseParticleMode==="on",retainStageDiagnosticsForQA:values.retainStageDiagnosticsForQA===true,sharpeningSweeps:0,sharpeningDistance:0},progress,signal),
 pipelineGraph:async()=> (await import("./uniform-narrow-band-pipeline")).UNIFORM_NARROW_BAND_PIPELINE,
 harness:async()=>({...await import("./harness").then(m=>m.uniformHarnessPlugin),methodId:"uniform-narrow-band-flip"}),
};
