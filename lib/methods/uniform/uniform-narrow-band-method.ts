import { NARROW_BAND_FINE_ADVECTION, narrowBandControlParams, narrowBandControlValues } from "./uniform-narrow-band-controls";
import type { MethodParamValues, SimulationMethod } from "../../core/method-contract";
import { UNIFORM_GEOMETRIC_PARAMS } from "./uniform-geometric-parameters";
import { composeFeatures } from "../../framework/composition";
import { parameterVariantFeature, parameterVariantSelections } from "../../core/method-parameter-variants";
import { uniformFixedStep_s } from "./uniform-paper";
import { uniformDiagnosticRows } from "./uniform-diagnostics";
import { migrateUniformDetailOverrides } from "./uniform-detail-policy";
import { uniformGeometricSolverOptions } from "./uniform-geometric-options";
import { resolveUniformGeometricValues } from "./uniform-geometric-parameters";
import { WebGPUUniformReferenceSolver } from "./webgpu-uniform-reference";

/** A 4h particle overlap, with either fixed or activity-driven surface coverage. */
const fixed:MethodParamValues={sharpeningSweeps:0,sharpeningDistance:0,totalSurfaceVolume:"on",surfaceVolumeRounds:2,surfaceDeficitBalancing:"off",volumeDustThreshold:0,orphanDustThreshold:0,phiDrain:"off",detailPolicy:"dynamic",adaptiveSurface:"on",coarseParticleMode:"off",pressureFullCycles:3,pressureVCycles:4,pressureSweeps:6};
// Fine tiles at surface crossings, and nothing else: shape at tolerance 0 asks
// for every surface tile, the distance filter keeps only the crossing ones, and
// no other criterion, margin or hold is on. Particle swept-support joins and
// mandatory solid promotion are not elective and stay.
export const narrowBandFixedDetail:MethodParamValues={detailPolicy:"dynamic",detailSolidContact:"on",detailSurface:"on",detailSurfaceDistance:0,
 detailShape:"on",detailShapeTolerance:0,detailShapeMetric:"value",detailThin:"off",detailStrain:"off",detailRotation:"off",detailImpact:"off",detailApproach:"off",
 detailNearFocus:"off",detailBulk:"off",detailMarginTiles:0,detailHoldSteps:0};
/** The app's adaptive profile. Particle heat supplies the hold in physical
 * time, so the tile census needs no additional step-count hold. */
export const narrowBandAdaptiveDetail:MethodParamValues={...narrowBandFixedDetail,...narrowBandControlValues(),adaptiveSurface:"on",
 detailShapeTolerance:0.5,detailShapeMetric:"displacement",detailThin:"on",detailStrain:"on",detailImpact:"on",detailApproach:"on"};
export const narrowBandFlipValues=(values:MethodParamValues={})=>({
 ...resolveUniformGeometricValues({...narrowBandAdaptiveDetail,...values}),...narrowBandControlValues(values),...fixed,
 fineGridPadding:Math.max(0,Math.min(4,Number.isFinite(Number(values.fineGridPadding))?Number(values.fineGridPadding):1)),
 adaptiveSurface:"on",
 coarseParticleMode:"off",
 phiFineAdvection:values.phiFineAdvection==="trilinear"?"trilinear":"cubic",
});
// Derive the renderer lifetime key from this method's own schema; inheriting
// Uniform Geometric's list misses NB-specific live controls.
const params:SimulationMethod["params"]=[
 ...narrowBandControlParams,
 NARROW_BAND_FINE_ADVECTION,
 {kind:"number",key:"fineGridPadding",label:"Fine-grid padding",default:1,tier:"fine",update:"runtime",min:0,max:4,step:1,digits:0,unit:"h",hint:"Extra h cells around swept surface crossings. The particle band stays 4h wide."},
 ...UNIFORM_GEOMETRIC_PARAMS.filter(p=>!Object.hasOwn(fixed,p.key)).map(p=>Object.hasOwn(narrowBandAdaptiveDetail,p.key)?{...p,default:narrowBandAdaptiveDetail[p.key]} as typeof p:p)
  // One sweep: the particles carry the velocity, and what reads the extension
  // (their own taps and the surface advect) stays within a cell of liquid; the
  // hierarchy fill covers the rest. Each sweep is 1.4 ms of a figure-9 frame,
  // and the take ends at the same volume with one as with two (1.08% and
  // 1.07% at 1/60 s, 1.02% and 1.05% at 1/30 s).
  .map(p=>p.key==="extensionFrontSweeps"&&p.kind==="number"?{...p,default:1}:p),
];
const point = "simulation.narrow-band-flip.algorithms";
const choices = params.filter(p => p.kind === "select");
const algorithmFeature = parameterVariantFeature(point, choices, ["simulation.page-domain"]);
const resolveComposition = (values: MethodParamValues = {}) => composeFeatures({
 features: [{ id: "simulation.narrow-band-flip.host", provides: ["simulation.page-domain", "simulation.vertex-phi", "simulation.conservative-volume"] }, algorithmFeature],
 selections: parameterVariantSelections(point, choices, values),
});
export const uniformNarrowBandMethod:SimulationMethod={
 backend: "webgpu",
 detail: "Activity-driven h detail over a 4h Eulerian base, with narrow-band FLIP surface particles and live rigid coupling.",
 pressureMapping: "Three full pressure cycles and four V-cycles.",
 composition: resolveComposition(),
 resolveComposition,
 migrateOverrides: migrateUniformDetailOverrides,
 renderRefinementBelowSolver: true,
 supportedFieldModes: ["structure", "density", "cfl", "speed", "phi", "volume-levelset", "fine-tiles"],
 diagnosticRows: uniformDiagnosticRows,
 effectiveStep_s: (_scene, values) => uniformFixedStep_s(values.timeStep),
 presetFor: () => ({}),
 qualityLabels: { balanced: "Fixed pressure cycles", high: "Fixed pressure cycles", ultra: "Fixed pressure cycles" },
 id:"uniform-narrow-band-flip",label:"Uniform Narrow-band FLIP",shortLabel:"Narrow-band FLIP",badge:"NARROW-BAND FLIP · EXPERIMENTAL",
 description:"Activity-driven FLIP particles preserve detailed surfaces while calm regions cool to an Eulerian level set at 4h. Shape error, thin features and approaching contact retain fine coverage. Volume is measured against sources and outflow, not corrected.",
 resource:{id:"fluid.uniform-narrow-band-flip",label:"Narrow-band FLIP fluid",lane:"fluid",
 provides:["fluid-authority","water-presentation"],blocks:"transport",
 phaseCopy:{planning:"Resolving narrow-band FLIP capabilities.",allocation:"Allocating fluid textures and buffers.","solver-pipelines":"Compiling narrow-band transport and pressure programs.",warmup:"Uploading and fencing the initial fluid state.",attach:"Attaching the warmed fluid solver."}},
 // The geometric method's layers, plus the band's velocity samples as spheres.
 capabilities:{adoptsRigidRosterShape:true,visualLayers:{hidden:["window","release"]}},
 params,
 runtimeParamKeys:params.filter(p=>p.update==="runtime").map(p=>p.key),
 appDefaults:{timeStep:"sixtieth",...narrowBandAdaptiveDetail,...fixed,fineGridPadding:1,coarseParticleMode:"off"},
 normalizeValues:narrowBandFlipValues,
 createSolverAsync:(device,scene,quality,values,loads,progress,signal)=>WebGPUUniformReferenceSolver.createAsync(device,scene,quality,loads,
  {...uniformGeometricSolverOptions(narrowBandFlipValues(values),scene),phiFineTrilinear:values.phiFineAdvection==="trilinear",narrowBandAdaptiveBudgetPercent:narrowBandControlValues(values).adaptiveBudgetPercent,narrowBandAdaptiveFadeSeconds:narrowBandControlValues(values).adaptiveFadeSeconds,narrowBandVolumeControlSeconds:narrowBandControlValues(values).volumeControlSeconds,narrowBandFinePadding:narrowBandFlipValues(values).fineGridPadding,retainStageDiagnosticsForQA:values.retainStageDiagnosticsForQA===true},progress,signal),
 pipelineGraph:async()=> (await import("./uniform-narrow-band-pipeline")).UNIFORM_NARROW_BAND_PIPELINE,
};
