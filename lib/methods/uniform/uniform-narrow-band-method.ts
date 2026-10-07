import type { MethodParamValues, SimulationMethod } from "../../core/method-contract";
import { uniformVolumeMethod } from "./uniform-volume-method";
import { uniformGeometricSolverOptions } from "./uniform-geometric-options";
import { resolveUniformGeometricValues } from "./uniform-geometric-parameters";
import { WebGPUUniformReferenceSolver } from "./webgpu-uniform-reference";

/** Keep a complete particle interpolation/pressure support band, independent
 * of the app's remembered Requested-detail preferences. */
const fixed:MethodParamValues={detailPolicy:"dynamic",detailSurface:"on",detailSurfaceDistance:2,
 detailShape:"on",detailShapeTolerance:0,detailThin:"off",detailNearFocus:"off",detailBulk:"off",detailMarginTiles:1,
 sharpeningSweeps:0,sharpeningDistance:0};
export const narrowBandFlipValues=(values:MethodParamValues={})=>({...resolveUniformGeometricValues(values),...fixed});
export const uniformNarrowBandMethod:SimulationMethod={
 ...uniformVolumeMethod,
 id:"uniform-narrow-band-flip",label:"Uniform Narrow-band FLIP",shortLabel:"Narrow-band FLIP",badge:"NARROW-BAND FLIP · EXPERIMENTAL",
 description:"Experimental 4h global pressure with FLIP velocity samples near the surface.",
 detail:"Persistent FLIP samples in a 4h surface band, 95% FLIP / 5% PIC, Eulerian interior, global 4h pressure and a coupled fine surface solve. No sharpening. This first version retains geometric surface/volume reconciliation and fine band topology.",
 resource:{...uniformVolumeMethod.resource!,id:"fluid.uniform-narrow-band-flip",label:"Narrow-band FLIP fluid"},
 // The geometric method's layers, plus the band's velocity samples as spheres.
 capabilities:{...uniformVolumeMethod.capabilities,visualLayers:{hidden:["pages","window","release"]}},
 params:uniformVolumeMethod.params!.filter(p=>!Object.hasOwn(fixed,p.key)),
 appDefaults:{...uniformVolumeMethod.appDefaults,...fixed},
 normalizeValues:narrowBandFlipValues,
 createSolverAsync:(device,scene,quality,values,loads,progress,signal)=>WebGPUUniformReferenceSolver.createAsync(device,scene,quality,loads,
  {...uniformGeometricSolverOptions(narrowBandFlipValues(values),scene),narrowBandFlip:true,retainStageDiagnosticsForQA:values.retainStageDiagnosticsForQA===true,sharpeningSweeps:0,sharpeningDistance:0},progress,signal),
 pipelineGraph:async()=> (await import("./uniform-narrow-band-pipeline")).UNIFORM_NARROW_BAND_PIPELINE,
 harness:async()=>({...await import("./harness").then(m=>m.uniformHarnessPlugin),methodId:"uniform-narrow-band-flip"}),
};
