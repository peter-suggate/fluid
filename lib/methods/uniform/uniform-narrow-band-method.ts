import type { MethodParamValues, SimulationMethod } from "../../core/method-contract";
import { uniformVolumeMethod } from "./uniform-volume-method";
import { uniformGeometricSolverOptions } from "./uniform-geometric-options";
import { resolveUniformGeometricValues } from "./uniform-geometric-parameters";
import { WebGPUUniformReferenceSolver } from "./webgpu-uniform-reference";

/** The original fine band is the default, while all refinement controls remain live. */
const fixed:MethodParamValues={sharpeningSweeps:0,sharpeningDistance:0};
const fineBand:MethodParamValues={detailPolicy:"dynamic",detailSolidContact:"on",detailSurface:"on",detailSurfaceDistance:2,
 detailShape:"on",detailShapeTolerance:0,detailThin:"off",detailNearFocus:"off",detailBulk:"off",detailMarginTiles:1};
export const narrowBandFlipValues=(values:MethodParamValues={})=>({
 ...resolveUniformGeometricValues({...fineBand,...values}),...fixed,
 coarseParticleMode:values.coarseParticleMode==="on"?"on":"off",
});
export const uniformNarrowBandMethod:SimulationMethod={
 ...uniformVolumeMethod,
 id:"uniform-narrow-band-flip",label:"Uniform Narrow-band FLIP",shortLabel:"Narrow-band FLIP",badge:"NARROW-BAND FLIP · EXPERIMENTAL",
 description:"4h global pressure with FLIP particles and a fine surface-pressure band.",
 detail:"Geometric surface and conservative volume with a fine FLIP pressure band. An optional all-4h particle-surface experiment also permits particles on 4h tiles. Uniform refinement criteria control h regions in both modes. No sharpening.",
 resource:{...uniformVolumeMethod.resource!,id:"fluid.uniform-narrow-band-flip",label:"Narrow-band FLIP fluid"},
 // The geometric method's layers, plus the band's velocity samples as spheres.
 capabilities:{...uniformVolumeMethod.capabilities,visualLayers:{hidden:["pages","window","release"]}},
 params:[{kind:"select",key:"coarseParticleMode",label:"Experimental all-4h FLIP",default:"off",tier:"coarse",update:"solver",dedicated:true,
  options:[{value:"off",label:"Off"},{value:"on",label:"On"}],hint:"Allow particles on 4h tiles with a particle-reconstructed surface; retain the selected h refinement policy. Restarts the simulation."},...uniformVolumeMethod.params!.filter(p=>!Object.hasOwn(fixed,p.key)).map(p=>Object.hasOwn(fineBand,p.key)?{...p,default:fineBand[p.key]} as typeof p:p)],
 appDefaults:{...uniformVolumeMethod.appDefaults,...fineBand,...fixed,coarseParticleMode:"off"},
 normalizeValues:narrowBandFlipValues,
 createSolverAsync:(device,scene,quality,values,loads,progress,signal)=>WebGPUUniformReferenceSolver.createAsync(device,scene,quality,loads,
  {...uniformGeometricSolverOptions(narrowBandFlipValues(values),scene),narrowBandFlip:true,narrowBandCoarseParticles:values.coarseParticleMode==="on",retainStageDiagnosticsForQA:values.retainStageDiagnosticsForQA===true,sharpeningSweeps:0,sharpeningDistance:0},progress,signal),
 pipelineGraph:async()=> (await import("./uniform-narrow-band-pipeline")).UNIFORM_NARROW_BAND_PIPELINE,
 harness:async()=>({...await import("./harness").then(m=>m.uniformHarnessPlugin),methodId:"uniform-narrow-band-flip"}),
};
