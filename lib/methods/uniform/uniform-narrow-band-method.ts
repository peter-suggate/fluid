import { NARROW_BAND_FINE_ADVECTION, narrowBandControlParams, narrowBandControlValues, narrowBandFineAdvection } from "./uniform-narrow-band-controls";
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
import { uniformNearSettled } from "./uniform-near-settled";

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
 * time, so the tile census needs no additional step-count hold. Strain is
 * off: dt·‖sym ∇u‖ is an affine field's, which 4h carries exactly, and any
 * falling stream has dt·g/|u| of it (0.15 per 1/24 s step on the pour's
 * spout, every tile of it h, its particles seeded and their joins h). */
export const narrowBandAdaptiveDetail:MethodParamValues={...narrowBandFixedDetail,...narrowBandControlValues(),adaptiveSurface:"on",
 detailShapeTolerance:0.5,detailShapeMetric:"displacement",detailThin:"on",detailImpact:"on",detailApproach:"on"};
export const narrowBandFlipValues=(values:MethodParamValues={})=>({
 ...resolveUniformGeometricValues({...narrowBandAdaptiveDetail,...values}),...narrowBandControlValues(values),...fixed,
 fineGridPadding:Math.max(0,Math.min(4,Number.isFinite(Number(values.fineGridPadding))?Number(values.fineGridPadding):0)),
 particleTransfer:values.particleTransfer==="apic"?"apic":"flip",
 adaptiveSurface:"on",
 coarseParticleMode:"off",
 phiFineAdvection:narrowBandFineAdvection(values.phiFineAdvection),
 coldCensus:values.coldCensus==="full"?"full":"replay",
 coldRegions:values.coldRegions==="fine"?"fine":"frozen16h",
 coldFrameSkip:values.coldFrameSkip==="off"?"off":"on",
 ...uniformNearSettled(values),
});
// Derive the renderer lifetime key from this method's own schema; inheriting
// Uniform Geometric's list misses NB-specific live controls.
const params:SimulationMethod["params"]=[
 ...narrowBandControlParams,
 {kind:"select",key:"particleTransfer",label:"Particle transfer",default:"flip",tier:"coarse",update:"solver",options:[{value:"flip",label:"FLIP (95%)"},{value:"apic",label:"APIC (experimental)"}],hint:"APIC replaces residual FLIP velocities with quadratic affine transfers, following Sato 2018. Restarts the solver."},
 NARROW_BAND_FINE_ADVECTION,
 {kind:"select",key:"surfaceSplat",label:"Particle surface scatter",default:"fused",tier:"fine",update:"solver",options:[{value:"fused",label:"Occupied cells"},{value:"separate",label:"Separate cell scan"}],hint:"Fuse occupied-cell selection and particle surface scatter using the current particle sort. Preserves the same surface distances. Restarts the solver."},
 {kind:"select",key:"rootCoarse",label:"Deep pressure levels",default:"fused",tier:"fine",update:"solver",options:[{value:"fused",label:"Fused 16h and 32h"},{value:"tiled",label:"Separate tiled levels"}],hint:"Run the deepest pressure levels in one workgroup using the current resident-block list and globally ordered iterations. Restarts the solver."},
 {kind:"select",key:"rootTiles",label:"Pressure stencil sharing",default:"page",tier:"fine",update:"solver",options:[{value:"page",label:"Whole pages"},{value:"octant",label:"Separate octants"}],hint:"Share a pressure halo across all eight octants of each page, preserving the global iteration schedule. Restarts the solver."},
 {kind:"select",key:"rootReserve",label:"Pressure safety slots",default:"fused",tier:"fine",update:"solver",options:[{value:"fused",label:"Fused fallback"},{value:"parallel",label:"Parallel traversals"}],hint:"Encode each reserved pressure traversal in one workgroup. Closed slots avoid their dispatch floor; an unexpectedly open slot still runs the complete global solve. Restarts the solver."},
 {kind:"select",key:"coldCensus",label:"Settled census",default:"replay",tier:"fine",update:"solver",options:[{value:"replay",label:"Keep settled decisions"},{value:"full",label:"Re-decide every tile"}],hint:"Settled regions keep their last detail decision until something near them moves; saves most of the census cost. Re-deciding classifies every tile every frame. Restarts the solver."},
 {kind:"select",key:"coldRegions",label:"Settled regions",default:"frozen16h",tier:"fine",update:"solver",options:[{value:"frozen16h",label:"Frozen at 16h"},{value:"fine",label:"Full fine detail"}],hint:"Settled regions keep local velocity detail frozen and take one 16h pressure correction per region. Fully submerged water allows more update error to stay cold longer; surfaces and body contact wake sooner. Full fine detail re-solves every settled cell every step."},
 {kind:"select",key:"coldFrameSkip",label:"Settled frame skip",default:"on",tier:"fine",update:"solver",options:[{value:"on",label:"Skip settled work"},{value:"off",label:"Run every stage"}],hint:"Stages with nothing to do in settled regions are left out of the frame, judged from the region heat two frames back. Running every stage launches all work every frame. Restarts the solver."},
 {kind:"select",key:"regionTiering",label:"Region tiers",default:"error",tier:"fine",update:"solver",options:[{value:"error",label:"By update error"},{value:"motion",label:"By any motion"}],hint:"Regions use detailed updates where per-tile motion error requires them (bodies, sources, splashes, unresolved velocity changes). Fine surface geometry can remain without keeping updates hot. By any motion, every moving region and its neighbours run at h, which costs far more in a gently disturbed pool. Restarts the solver."},
 {kind:"select",key:"warmResolution",label:"Warm regions",default:"h",tier:"fine",update:"solver",options:[{value:"h",label:"By spatial error"},{value:"4h",label:"Coarse warm regions"}],hint:"Fine tiles follow surface detail and cut geometry; exactly represented walls stay coarse. Coarse warm regions additionally reduces pressure resolution. Restarts the solver."},
 {kind:"select",key:"particleSeeding",label:"Particle seeding",default:"need",tier:"fine",update:"solver",options:[{value:"need",label:"Where the grid falls short"},{value:"activity",label:"Wherever detail is active"}],hint:"Particles are seeded only for thin sheets, impacts, approaching contact, sharp shear and a body's waterline, which the grid cannot hold; a smooth wake or ripple stays on the grid. Seeding wherever detail is active also covers smooth surface changes and costs far more particles in a gently moving pool. Restarts the solver."},
 {kind:"select",key:"sourceSeeding",label:"Source particles",default:"need",tier:"fine",update:"solver",options:[{value:"need",label:"Where the grid falls short"},{value:"always",label:"At every source"}],hint:"A smooth inflow stream stays on the grid; it gains particles only where it thins, impacts or shears, like any other liquid. Seeding at every source keeps particles on the whole stream for the retirement time and costs far more particles. Restarts the solver."},
 {kind:"select",key:"traceSubsteps",label:"Trace substeps",default:"gradient",tier:"fine",update:"solver",options:[{value:"gradient",label:"By velocity gradient"},{value:"travel",label:"One per cell of travel"}],hint:"Particle and surface traces take as many substeps as the velocity's turn along the step needs, so a fast straight jet traces in one or two. One per cell of travel subdivides every fast trace regardless of how straight it is. Restarts the solver."},
 {kind:"select",key:"wallTrace",label:"Wall trace",default:"clip",tier:"fine",update:"solver",options:[{value:"clip",label:"One step, clipped contact"},{value:"retrace",label:"Substeps, retraced contact"}],hint:"A surface vertex on a domain wall traces back in one step, and where the flow runs into the wall it takes the interior neighbour's value from that same trace, moved inward and clipped to the domain. Retraced substeps the wall trace and traces the interior neighbour again. Restarts the solver."},
 {kind:"select",key:"surfaceVelocitySampler",label:"Surface velocity sampler",default:"texture32",tier:"fine",update:"solver",options:[{value:"texture32",label:"Texture filter, 32-bit"},{value:"software",label:"Software"}],hint:"The level set's backtrace interpolates the fine velocity at each point it visits. The texture filter asks the GPU's texture unit for each interpolated sample from the same 32-bit velocities, rounding the interpolation weights slightly (software where the GPU cannot filter 32-bit textures). Software does the eight loads and the blend itself, exactly. Restarts the solver."},
 {kind:"select",key:"surfaceTraceSubsteps",label:"Surface trace substeps",default:"off",tier:"fine",update:"solver",options:[{value:"off",label:"One step"},{value:"legacy",label:"As the particle trace"}],hint:"The level set's backtrace takes one third-order step of any length, three velocity samples, wherever the flow bends. As the particle trace restarts a bending trace in substeps (Trace substeps), more samples for more accuracy. Restarts the solver."},
 {kind:"number",key:"surfaceRedistanceReach",label:"Redistance reach",default:3,tier:"fine",update:"solver",min:1,max:3,step:1,digits:0,unit:" tiles",hint:"How many 4-cell tiles past the surface the redistance rebuilds exact distances each step. Everything that reads the distance (transfer, particle depth and retirement, the census) looks no further than one tile; past the reach a vertex keeps its sign at six cells. More tiles cost more search. Restarts the solver."},
 {kind:"select",key:"redistanceSearch",label:"Redistance search",default:"shared",tier:"fine",update:"solver",options:[{value:"shared",label:"Shared with retirement"},{value:"fresh",label:"Fresh after retirement"}],hint:"Redistance reuses the crossing search retirement ran just before it, one search a step fewer. Fresh searches again, catching a zero retirement moved at a coarse seam. Restarts the solver."},
 {kind:"select",key:"flipDelta",label:"FLIP delta",default:"swept",tier:"fine",update:"solver",options:[{value:"swept",label:"Swept with the field"}, {value:"extended",label:"Extended old field"}],hint:"FLIP adds the grid's change across forces and pressure to each sample's velocity. Swept with the field takes that change from the extension's sweeps of the old and new fields, recorded at each sample after pressure, and publishes no old field. Extended old field publishes the old field in full and samples it, then compares it with the next step's sample of the new field. Restarts the solver."},
 {kind:"select",key:"crossingGuard",label:"Bulk depth guard",default:"previous",tier:"fine",update:"solver",options:[{value:"previous",label:"Last step's crossings"}, {value:"search",label:"Search the advected surface"}],hint:"Particle reconstruction keeps the advected level set at least as deep as the distance to the nearest surface crossing, so samples cannot erode the bulk. Last step's crossings measures that distance to the crossings the last redistance found, less this step's certified travel, and skips a crossing search a step. Searching finds the crossings of the advected surface itself. Restarts the solver."},
 {kind:"select",key:"bandExit",label:"Fine pressure stop",default:"returns",tier:"fine",update:"solver",options:[{value:"returns",label:"At 0.05 per second"},{value:"strict",label:"At the strict tolerance"}],hint:"The h pressure band stops once its worst divergence is below 0.05 per second (a few millimetres per second of velocity error in one cell; the 4h solve accepts up to 5), or at the cycle cap. The strict tolerance runs every cycle toward 0.0001 per second, which a moving pool never reaches, so every cycle up to the cap runs on every frame. Restarts the solver."},
 {kind:"select",key:"solidContact",label:"Solid contact detail",default:"surface",tier:"fine",update:"solver",options:[{value:"liquid",label:"Wherever liquid meets it"},{value:"surface",label:"Where the surface meets it"}],hint:"Static solids hold the tiles they cut, and those tiles' neighbours, at h. Wherever liquid meets it does so for every wet cut tile, down to the floor. Where the surface meets it does so only where the surface's own detail already reaches the solid; submerged parts of a solid run as cut 4h owners, which smooths the flow around them below the surface, and a dry neighbour off the surface band stays 4h. Restarts the solver."},
 {kind:"number",key:"particleBandDepth",label:"Particle band depth",default:3,tier:"fine",update:"solver",min:2,max:4,step:0.5,digits:1,unit:" h",hint:"How deep below the surface the FLIP particles reach. The grid's own momentum carries the liquid beneath. The transfer writes particle velocity only to faces within two cells of the surface, so particles deeper than about three and a half cells add none; a shallower band holds fewer particles and costs less, while a falling surface reseeds the band below it. Restarts the solver."},
 {kind:"select",key:"particleDensity",label:"Particle density",default:"graded",tier:"fine",update:"solver",options:[{value:"uniform",label:"Eight a cell"},{value:"graded",label:"Graded with depth"}],hint:"Eight a cell seeds the whole band at eight samples a cell. Graded keeps eight in the top cell and a half, where the samples define the surface, and seeds four a cell beneath, where they only carry momentum: about a third fewer particles, with a noisier transfer below the surface. Restarts the solver."},
 {kind:"number",key:"bandCycles",label:"Fine pressure cycles",default:3,tier:"fine",update:"solver",min:1,max:8,step:1,digits:0,unit:"",hint:"V-cycles the h pressure band runs at most a step. Each cycle cuts its worst divergence about tenfold; four reach 0.05 per second from the 4h start. Restarts the solver."},
 {kind:"select",key:"momentumSampler",label:"Momentum sampler",default:"texture32",tier:"fine",update:"solver",options:[{value:"texture32",label:"Texture filter, 32-bit"},{value:"texture16",label:"Texture filter, 16-bit"},{value:"software",label:"Software"}],hint:"Grid momentum traces each fine face back one step and interpolates the velocity there. The texture filter asks the GPU's texture unit for each interpolated sample, from the same 32-bit velocities (falling back to 16-bit where the GPU cannot filter 32-bit textures) or from a 16-bit copy of the fine velocities; it rounds the interpolation weights slightly. Software does the eight loads and the blend itself, exactly. Restarts the solver."},
 {kind:"number",key:"fineGridPadding",label:"Fine-grid padding",default:0,tier:"fine",update:"runtime",min:0,max:4,step:1,digits:0,unit:"h",hint:"Extra h cells around swept surface crossings. The particle band stays 4h wide."},
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
 appDefaults:{particleTransfer:"flip",timeStep:"sixtieth",...narrowBandAdaptiveDetail,...fixed,fineGridPadding:0,coarseParticleMode:"off"},
 normalizeValues:narrowBandFlipValues,
 createSolverAsync:(device,scene,quality,values,loads,progress,signal)=>WebGPUUniformReferenceSolver.createAsync(device,scene,quality,loads,
  {...uniformGeometricSolverOptions(narrowBandFlipValues(values),scene),narrowBandTransfer:values.particleTransfer==="apic"?"apic":"flip",phiFineAdvection:narrowBandFineAdvection(values.phiFineAdvection),narrowBandAdaptiveBudgetPercent:narrowBandControlValues(values).adaptiveBudgetPercent,narrowBandAdaptiveFadeSeconds:narrowBandControlValues(values).adaptiveFadeSeconds,narrowBandSparseParticleLifetime:narrowBandControlValues(values).sparseParticleLifetime,narrowBandParticleCellCap:narrowBandControlValues(values).particleCellCap,narrowBandVolumeControlSeconds:narrowBandControlValues(values).volumeControlSeconds,narrowBandFinePadding:narrowBandFlipValues(values).fineGridPadding,narrowBandColdCensus:values.coldCensus==="full"?"full":"replay",narrowBandColdRegions:values.coldRegions==="fine"?"fine":"frozen16h",narrowBandColdFrameSkip:values.coldFrameSkip==="off"?"off":"on",narrowBandNearSettled:uniformNearSettled(values),retainStageDiagnosticsForQA:values.retainStageDiagnosticsForQA===true},progress,signal),
 pipelineGraph:async()=> (await import("./uniform-narrow-band-pipeline")).UNIFORM_NARROW_BAND_PIPELINE,
};
