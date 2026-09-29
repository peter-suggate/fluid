import type { FluidPipelineGraph, FluidPipelineStage, FluidPipelineContext, FluidStageControl } from "../../core/fluid-pipeline";
import { UNIFORM_FLUID_PIPELINE } from "./uniform-pipeline";
import { UNIFORM_ADVANCE_PHASE as A } from "./uniform-stages";
import { UNIFORM_VOLUME_PHASE as P } from "./uniform-volume-stages";
import { UNIFORM_GEOMETRIC_SHARPENING_DISTANCE, UNIFORM_GEOMETRIC_SPLASH_HINTS } from "./uniform-geometric-parameters";

/**
 * Uniform Geometric's pipeline: one card per seam UniformMixedFrame.advance
 * emits, in encode order, so the cards partition the traced advance exactly.
 * The dense Uniform graph lends only the controls and tips its stages share.
 */

type VolumeInfo = {
  uniformVolumeDustCells?: number; uniformVolumeDustMass_cells?: number;
  uniformTwoLevelFineTiles?: number; uniformTwoLevelTilesTotal?: number;
  uniformTwoLevelShellTiles?: number; uniformTwoLevelShellReach?: number;
} | null;
const volumeInfo = (context: FluidPipelineContext) => context.info as unknown as VolumeInfo;
/** The authored floor, in cell volumes; zero and absent both read as off. */
const dustThreshold = (context: FluidPipelineContext) => {
  const value = Number(context.values.volumeDustThreshold ?? 0);
  return Number.isFinite(value) && value > 0 ? value : 0;
};
const dustChip = (context: FluidPipelineContext) => {
  const threshold = dustThreshold(context);
  return threshold > 0 ? `8 donors · 3 rounds · dust floor ${threshold.toExponential(0)}` : "8 donors · 3 rounds · dust floor off";
};
/** The mixed frame's fine tiles as a share of the 4h tile grid. */
const fineMap = (context: FluidPipelineContext) => {
  const info = volumeInfo(context);
  const fine = info?.uniformTwoLevelFineTiles, total = info?.uniformTwoLevelTilesTotal;
  if (fine === undefined || total === undefined || total <= 0) return undefined;
  return {fine,total,percent:Math.round(100*fine/total)};
};
/** Shell tiles: the set the extension's finest passes still run on. */
const shellMap = (context: FluidPipelineContext) => {
  const info = volumeInfo(context);
  const shell = info?.uniformTwoLevelShellTiles, total = info?.uniformTwoLevelTilesTotal;
  if (shell === undefined || total === undefined || total <= 0) return undefined;
  return {shell,total,percent:Math.round(100*shell/total)};
};
const supportChip = (context: FluidPipelineContext) => {
  const fine = fineMap(context), shell = shellMap(context);
  return fine && shell ? `fine ${fine.percent}% · shell ${shell.percent}% of tiles` : "fine + shell support planes";
};
const supportControls: FluidStageControl[] = [
  {kind:"readout",label:"Fine tiles",
    hint:"4×4×4 tiles sampling the finest lattice, in the latest diagnostics sample. The rest read the 4h face table.",
    value:context=>{const map=fineMap(context);return map?`${map.fine} / ${map.total} (${map.percent}%)`:"—";}},
  {kind:"readout",label:"Shell tiles",
    hint:"Fine tiles dilated by the shell reach: the set where the extension's finest output must still be exact, and therefore the set its passes run on.",
    value:context=>{const map=shellMap(context);const reach=volumeInfo(context)?.uniformTwoLevelShellReach;
      return map?`${map.shell} / ${map.total} (${map.percent}%)${reach?` · +${reach}`:""}`:"—";}},
];
const onOff=[{value:"on",label:"On"},{value:"off",label:"Off"}];
const dynamic = (context: FluidPipelineContext) => context.values.coarsening !== "regions";
const bodies = (count: number) => `${count} ${count === 1 ? "body" : "bodies"}`;
const base = (id: string) => {
  const stage = UNIFORM_FLUID_PIPELINE.stages.find(candidate => candidate.id === id);
  if (!stage) throw new Error(`Uniform pipeline has no stage ${id}`);
  return stage;
};
/** A dense Uniform control, kept where the Geometric frame still reads it. */
const baseControl = (id: string, param: string) => {
  const control = base(id).controls?.find(candidate => "param" in candidate && candidate.param === param);
  if (!control) throw new Error(`Uniform stage ${id} has no ${param} control`);
  return control;
};

const stages: FluidPipelineStage[] = [
  {
    id:"uniform-volume-solids", band:"head", side:"right", label:"Solids",
    phaseLabels:[P.solids.label],
    tip:{summary:"Moving bodies rebuild their cut-cell record, and liquid in cells a body or a live voxel edit entered is displaced to open neighbours. The 4h coarse solid record is built once for static solids (again after an edit, every frame bodies exist), and the simulation's cut-tile widths when stale. A static scene closes this seam near zero.",
      reads:"solid voxels, body poses", writes:"cut-cell record, displaced V", gate:"the scene compiles the solid library"},
    state:()=>"on",
    chip:context=>context.bodyCount > 0 ? `${bodies(context.bodyCount)} · cut record every frame` : "static · built once",
  },
  {
    id:"uniform-volume-support", band:"extension", side:"left", label:"Support plan",
    phaseLabels:[P.support.label],
    tip:{summary:"Start-of-frame support census every stage shares: fine sampling reach (two tiles) and the extension shell (one more). Rebuilds the surface geometry when an edit or adopt cleared it, then the simulation interface phase and correction field the extension reads.",
      reads:"V, phi, velocity", writes:"support planes, interface phase", feeds:"every stage of the frame"},
    state:()=>"on", chip:supportChip, controls:supportControls,
  },
  {
    id:"velocity-extension", band:"extension", side:"right", label:"Velocity extension",
    phaseLabels:[P.extension.label],
    tip:{summary:"One pass: seed supported face velocities, Godunov sweeps across the mixed h/4h patches, restrict to the 4h nearest-source hierarchy, fill the far field from it, and publish the extended MAC field. Keeps distant stationary liquid from slowing falling drops.",
      reads:"MAC velocity, interface phase", writes:"extended MAC velocity", feeds:"phi transport, volume coupling, momentum"},
    controls:[baseControl("velocity-extension","extensionFrontSweeps")],
    state:()=>"on",
    chip:context=>`${Number(context.values.extensionFrontSweeps ?? 2)} sweeps · 4h hierarchy fill`,
  },
  {
    id:"uniform-volume-samplers", band:"extension", side:"left", label:"Sampling caches",
    phaseLabels:[P.transportReach.label],
    tip:{summary:"The post-extension 4h transport-reach certificate, the 4h MAC sampling cache of the extended velocity, and the seam-tile hanging taps: every fine tap in a seam tile resolved once through the h/4h sampler, so phi and momentum characteristics reuse it.",
      reads:"extended MAC velocity", writes:"reach certificate, 4h cache, hanging taps", feeds:"phi transport, momentum"},
    state:()=>"on", chip:()=>"reach certificate · 4h cache · seam taps",
  },
  {
    id:"uniform-volume-phi", band:"surface", side:"right", label:"Vertex level set",
    phaseLabels:[P.phi.label],
    tip:{summary:"RK2 vertex characteristics with hanging vertices resolved, cell traces for volume transport, then bounded closest-point redistancing and a second resolve.",
      reads:"phi, sampling caches", writes:"phi, cell traces", feeds:"volume coupling"},
    controls:[
      {kind:"param-choice",param:"phiCubicAdvection",label:"Cubic advection",options:onOff,hint:UNIFORM_GEOMETRIC_SPLASH_HINTS.phiCubicAdvection},
      {kind:"param-choice",param:"phiDrain",label:"Drain ghost phi",options:onOff,hint:UNIFORM_GEOMETRIC_SPLASH_HINTS.phiDrain},
    ],
    state:()=>"on",
    chip:context=>[context.values.phiCubicAdvection === "on" ? "cubic" : "trilinear",
      context.values.phiDrain === "on" ? "drained" : "", context.values.redistance === "off" ? "" : "redistanced"].filter(Boolean).join(" · "),
  },
  {
    id:"uniform-volume-coupling", band:"surface", side:"left", label:"Conservative volume transport",
    phaseLabels:[P.coupling.label],
    tip:{summary:"Eight box-overlap donors plus an identity fallback, three receiver/donor balancing rounds, the gather and the 4h restriction. Dust cleanup follows when a floor is set.",
      reads:"V, cell traces", writes:"V", feeds:"surface volume constraint, sharpening"},
    controls:[{kind:"param-range",param:"volumeDustThreshold",label:"Dust floor",unit:"cell volumes",
      min:0,max:1e-3,step:1e-7,digits:7,
      hint:"Discard |V| below this wherever transport or sharpening writes V, ULP-scale negatives included. Zero is off and stores the untreated sum bit for bit."},
      {kind:"param-range",param:"orphanDustThreshold",label:"Orphan dust floor",unit:"cell volumes",
        min:0,max:0.05,step:0.001,digits:3,
        hint:"Extra floor outside the surface band, with local droplet protection. Zero disables it. Discarded mass is not restored by surface correction."},
      {kind:"readout",label:"Dust discarded",
        hint:"Cells zeroed by regular and orphan cleanup in the latest step, and the mass that went with them.",
        value:context=>{
          const info=volumeInfo(context);const cells=info?.uniformVolumeDustCells;
          if(dustThreshold(context)<=0||cells===undefined)return "—";
          return `${cells.toLocaleString()} cells · ${(info?.uniformVolumeDustMass_cells ?? 0).toExponential(2)} cell volumes`;}}],
    state:()=>"on", chip:dustChip,
  },
  {
    id:"uniform-volume-gather", band:"surface", side:"right", label:"Surface volume + geometry",
    phaseLabels:[P.gather.label],
    tip:{summary:"Optionally one bounded global normal shift of phi to match total V (four-cell band, two 17-sample refinements), then the surface-geometry cache: centre phi and corrected phi capacity. Nothing after this writes phi.",
      reads:"phi, V", writes:"phi (shift), geometry cache", feeds:"sharpening, pressure authority, next frame"},
    controls:[{kind:"param-choice",param:"totalSurfaceVolume",label:"Total surface volume",options:onOff,
      hint:"One bounded global normal shift to match the surface volume to V. No regional correction or cellwise reconstruction."}],
    state:()=>"on",
    chip:context=>context.values.totalSurfaceVolume === "on" ? "total volume constrained · geometry" : "geometry only",
  },
  {
    id:"uniform-volume-sharpen", band:"surface", side:"left", label:"Volume sharpening",
    phaseLabels:[P.sharpen.label],
    tip:{summary:"Eight symmetric face-transfer sweeps with aggregate donor and receiver budgets; phi is immutable across them, so only tiles with a cell in the admission band run."},
    toggle:{param:"densitySharpening",on:"on",off:"off"},
    state:context=>context.values.densitySharpening === "off" ? "off" : "on",
    chip:()=>`${UNIFORM_GEOMETRIC_SHARPENING_DISTANCE}h band · 8 sweeps`,
  },
  {
    ...base("velocity-advection"), controls:undefined,
    tip:{...base("velocity-advection").tip,
      summary:"Algorithm 1 step 3: semi-Lagrangian momentum resampling on canonical MAC faces, then gravity, viscosity and surface tension.",
      reads:"extended MAC velocity, sampling caches"},
    chip:()=>"one backward-trace pass + forces",
  },
  {
    ...base("pressure-system"),
    tip:{summary:"The surface band's tile list and the h band's preparation; the pressure authority in pressure ownership (with h tiles, u* and geometry transfer to the all-4h layout first); the all-4h RHS and native hierarchy setup; then the initial residual for acceptance.",
      reads:"u*, phi, V, solid record", writes:"all-4h RHS, band rows", feeds:"pressure cycles"},
    controls:[{kind:"param-choice",param:"surfaceDeficitBalancing",label:"Surface-deficit balancing",options:onOff,hint:"Preserve overfill expansion and balance it globally with contraction in underfilled liquid. Reduces persistent sloshing."}],
    chip:()=>"shared ownership · coupled RHS pyramid",
  },
  {
    ...base("pressure-cycles"),
    tip:{summary:"The all-4h root solve: CM11a cycles whose correction the native n/4 hierarchy solves; residuals, measure and bounds use the mixed rows. The host encodes the slot list the lagged plan chose (V-cycles, then Full-Cycles). A GPU gate before each slot closes it once converged, jumps a stalled V phase to Full-Cycles, and tightens coarse accuracy on a stall. A closed slot's launches still pay their floor. Running out of slots unconverged withholds the projection and stops the frame.",
      reads:"all-4h RHS", writes:"all-4h pressure, acceptance state", feeds:"pressure projection"},
    controls:[...(base("pressure-cycles").controls ?? []).filter(control=>control.kind!=="readout"&&!("param" in control&&["pressureCycleBudget","pressureBudgetHeadroom"].includes(control.param)))
      .map(control => control.kind === "param-range" && control.param === "pressureSweeps"
        ? {...control,hint:"Projected Jacobi sweeps before and after each coarse correction. Changing this rebuilds the plan and resets time."}
        : control.kind === "param-range" && control.param === "pressureVCycles"
          ? {...control,hint:"Available cheap corrections. Repeat while residual reduction is good; skip the remaining V-cycles for Full-Cycles when progress stalls. Encode only what the current frame needs."}
          : control),
      {kind:"readout",label:"Cycles",hint:"Cycles that ran in the last completed step, against the slots encoded for it.",
        value:context=>context.info?.uniformPressureCyclesExecuted === undefined ? "—"
          : `${context.info.uniformPressureCyclesExecuted} of ${context.info.uniformPressureCyclesEncoded ?? "?"} encoded`},
      {kind:"readout",label:"Residual ∞-norm",hint:"Accepted all-4h root residual (h-equivalent). Projection is withheld if the requested tolerance is not met.",
        value:context=>context.info?.uniformPressureAcceptedResidual === undefined ? "—" : context.info.uniformPressureAcceptedResidual.toExponential(2)}],
    chip:context=>context.info?.uniformPressureCyclesExecuted === undefined ? "V-first adaptive · awaiting first solve"
      : `V-first adaptive · ${context.info.uniformPressureCyclesExecuted} of ${context.info.uniformPressureCyclesEncoded ?? "?"} slots ran`,
  },
  {
    ...base("pressure-projection"),
    tip:{summary:"Gated on acceptance: subtract the all-4h pressure gradient from u* in pressure ownership.",
      reads:"u*, all-4h pressure", writes:"projected MAC velocity (4h)", feeds:"h band solve"},
    chip:()=>"all-4h · gated on acceptance",
  },
  {
    id:"uniform-volume-band", band:"pressure", side:"left", label:"h band solve",
    phaseLabels:[P.band.label],
    tip:{summary:"Second pressure stage: transfer the projected field back to simulation ownership, then every h tile re-solves locally from the 4h pressure (V-cycles h → 2h → 4h, red-black Gauss-Seidel at every level) and projects its faces into the velocity field. Also hands the solve to the grid overlay. Empty when every tile is 4h.",
      reads:"all-4h pressure, band rows", writes:"h pressure, divergence-free MAC velocity", feeds:"rigid coupling, next advance"},
    controls:[
      {kind:"readout",label:"Band tiles",hint:"h tiles the band re-solved in the last completed step.",
        value:context=>context.info?.uniformPressureBandTiles === undefined ? "—" : context.info.uniformPressureBandTiles.toLocaleString()},
      {kind:"readout",label:"Band residual",hint:"The band's final residual after its local V-cycles.",
        value:context=>context.info?.uniformPressureBandResidual === undefined || !context.info.uniformPressureBandTiles ? "—" : context.info.uniformPressureBandResidual.toExponential(2)},
    ],
    state:()=>"on",
    chip:context=>context.info?.uniformPressureBandTiles === undefined ? "local MG · h → 2h → 4h"
      : `${context.info.uniformPressureBandTiles.toLocaleString()} tiles · ${context.info.uniformPressureBandCycles ?? 0} V-cycles`,
  },
  {
    ...base("rigid-coupling"),
    tip:{...base("rigid-coupling").tip,summary:"Native two-way coupling on the projected field: the fluid pushes on each body through sampled pressure and drag, bodies push back on the face velocities, then the rigid system integrates poses and the solid library records the moved bodies for the census."},
  },
  {
    id:"uniform-volume-census", band:"census", side:"right", label:"Resolution census",
    phaseLabels:[A.resolutionCensus.label],
    tip:{summary:"The frame's tail extension (the next frame reuses it when nothing changes in between), then the dynamic census: tiles the surface can reach over the census horizon (RK2 departure boxes traced frame by frame, boundary impacts, bodies) stay h and the rest coarsen to 4h. The GPU layout builder writes the next generation; the host adopts it in its own submit when the census read resolves, remapping V, velocity, phi and the extension. No frame waits on it.",
      reads:"phi, V, velocity, body poses", writes:"extended velocity, next generation's ownership", feeds:"next frame, layout adopt", gate:"dynamic coarsening"},
    controls:[
      {kind:"param-choice",param:"coarsening",label:"Coarsening",
        options:[{value:"dynamic",label:"Dynamic"},{value:"regions",label:"Regions"}],
        hint:"Dynamic: after every frame, only tiles the surface can occupy during the next step are fine. Regions: authored refinement regions choose coarse tiles, and this stage and layout adopt encode nothing."},
      {kind:"param-range",param:"coarseningBoundaryTravel",label:"Boundary impact travel",unit:"cells/step",
        min:0,max:64,step:0.5,digits:1,enabled:dynamic,
        hint:"Surface liquid moving at least this many fine cells per step toward a wall or solid it reaches within one 4h cell stays fine."},
    ],
    state:context=>dynamic(context) ? "on" : "off",
    chip:context=>context.info?.uniformMixedFineTiles === undefined ? "h/4h census"
      : `${context.info.uniformMixedFineTiles.toLocaleString()} h · ${(context.info.uniformMixedCoarseTiles ?? 0).toLocaleString()} 4h tiles`,
  },
  {
    ...base("density-post-process"), label:"Phi surface publication", phaseLabels:[P.surface.label], spendsNoFrameTime:true,
    toggle:undefined, controls:undefined, state:()=>"on", chip:()=>"phi = 0",
    tip:{summary:"The renderer reads canonical level-set vertices through the accepted ownership generation. Publication switches the accepted view after the step completes; no dense expansion pass is required."},
  },
];

export const UNIFORM_VOLUME_PIPELINE: FluidPipelineGraph = {
  methodId:"uniform-volume",
  bands:[
    {id:"head",label:"Frame head · ownership + solids"},
    {id:"extension",label:"Velocity extension"},
    {id:"surface",label:"Level set + volume"},
    {id:"momentum",label:"Momentum transport"},
    {id:"pressure",label:"Pressure · all-4h root + h band"},
    {id:"coupling",label:"Rigid coupling"},
    {id:"census",label:"Dynamic coarsening census"},
    {id:"output",label:"Output"},
  ],
  stages,
};
