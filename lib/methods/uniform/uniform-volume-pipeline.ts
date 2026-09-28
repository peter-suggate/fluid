import type { FluidPipelineGraph, FluidPipelineStage, FluidPipelineContext } from "../../core/fluid-pipeline";
import { UNIFORM_FLUID_PIPELINE } from "./uniform-pipeline";
import { UNIFORM_VOLUME_PHASE as P } from "./uniform-volume-stages";
import { UNIFORM_GEOMETRIC_SPLASH_HINTS } from "./uniform-geometric-parameters";

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
  return threshold > 0 ? `dust floor ${threshold.toExponential(0)}` : "dust floor off";
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
const twoLevelChip = (context: FluidPipelineContext) => {
  const shell = shellMap(context);
  return shell ? `two-level · tiles ${shell.percent}%` : "two-level · tiles";
};
const twoLevelControls = [
  {kind:"readout" as const,label:"Fine tiles",
    hint:"4×4×4 tiles sampling the finest lattice, in the latest diagnostics sample. The rest read the 4h face table.",
    value:(context: FluidPipelineContext)=>{const map=fineMap(context);return map?`${map.fine} / ${map.total} (${map.percent}%)`:"—";}},
  {kind:"readout" as const,label:"Shell tiles",
    hint:"Fine tiles dilated by the shell reach: the set where the extension's finest output must still be exact, and therefore the set its passes run on.",
    value:(context: FluidPipelineContext)=>{const map=shellMap(context);const reach=volumeInfo(context)?.uniformTwoLevelShellReach;
      return map?`${map.shell} / ${map.total} (${map.percent}%)${reach?` · +${reach}`:""}`:"—";}},
];
const onOff=[{value:"on",label:"On"},{value:"off",label:"Off"}];
/** The two stages that write V into phi sit on the level set they write. */
const phiControls = [
  {kind:"param-choice" as const,param:"totalSurfaceVolume",label:"Total surface volume",options:onOff,
    hint:"One bounded global normal shift to match the surface volume to V. No regional correction or cellwise reconstruction."},
  {kind:"param-choice" as const,param:"phiCubicAdvection",label:"Cubic advection",options:onOff,
    hint:UNIFORM_GEOMETRIC_SPLASH_HINTS.phiCubicAdvection},
  {kind:"param-choice" as const,param:"phiDrain",label:"Drain ghost phi",options:onOff,
    hint:UNIFORM_GEOMETRIC_SPLASH_HINTS.phiDrain},
];
const phiChip = (context: FluidPipelineContext) => {
  const parts=[context.values.totalSurfaceVolume === "on" ? "total volume constrained" : "",
    context.values.phiCubicAdvection === "on" ? "cubic" : "", context.values.phiDrain === "on" ? "drained" : ""].filter(Boolean);
  return parts.length ? parts.join(" · ") : "trilinear";
};
const volumeStages: FluidPipelineStage[] = [
  ["phi", "Vertex level set", "RK2 characteristics and bounded closest-point redistancing. The optional global volume constraint runs after conservative gather."],
  ["coupling", "Conservative volume transport", "Eight box-overlap donors plus an identity fallback; three receiver/donor balancing rounds."],
  ["gather", "Conservative volume gather", "Gather donor-normalized liquid volume, optionally constrain total surface volume, and cache corrected phi capacity for sharpening."],
  ["sharpen", "Volume sharpening", "Eight symmetric face-transfer sweeps with aggregate donor and receiver budgets; phi is immutable across them, so only tiles with a cell in the admission band run."],
].map(([id,label,summary]) => ({
  id: `uniform-volume-${id}`, band:"surface", side:"left", label:label!,
  phaseLabels:[P[id as "phi"|"coupling"|"gather"|"sharpen"].label],
  tip:{summary:summary!}, state: context => (id === "sharpen" && context.values.densitySharpening === "off") ? "off" : "on",
  chip:context=>id === "gather"
    ? "conservative gather"
    : id === "sharpen" ? "2.1h band · 8 sweeps"
    : id === "coupling" ? dustChip(context)
    : phiChip(context),
  ...(id === "phi" ? { controls: phiControls } : {}),
  ...(id === "coupling" ? {
    controls:[{kind:"param-range" as const,param:"volumeDustThreshold",label:"Dust floor",unit:"cell volumes",
      min:0,max:1e-3,step:1e-7,digits:7,
      hint:"Discard |V| below this wherever transport or sharpening writes V, ULP-scale negatives included. Zero is off and stores the untreated sum bit for bit."},
      {kind:"param-range" as const,param:"orphanDustThreshold",label:"Orphan dust floor",unit:"cell volumes",
        min:0,max:0.05,step:0.001,digits:3,
        hint:"Extra floor outside the surface band, with local droplet protection. Zero disables it. Discarded mass is not restored by surface correction."},
      {kind:"readout" as const,label:"Dust discarded",
      hint:"Cells zeroed by regular and orphan cleanup in the latest step, and the mass that went with them.",
      value:(context: FluidPipelineContext)=>{
        const info=volumeInfo(context);const cells=info?.uniformVolumeDustCells;
        if(dustThreshold(context)<=0||cells===undefined)return "—";
        return `${cells.toLocaleString()} cells · ${(info?.uniformVolumeDustMass_cells ?? 0).toExponential(2)} cell volumes`;}}],
  } : {}),
  ...(id === "sharpen" ? {
    toggle:{param:"densitySharpening",on:"on",off:"off"},
  } : {}),
}));
export const UNIFORM_VOLUME_PIPELINE: FluidPipelineGraph = {
  methodId:"uniform-volume",
  bands:UNIFORM_FLUID_PIPELINE.bands.map(b=>b.id==="surface"?{...b,label:"Level set + volume"}:b),
  stages:UNIFORM_FLUID_PIPELINE.stages.flatMap(stage=>{
    if(stage.id==="density-advection")return volumeStages;
    if(["gamma-diffusion","interface-sharpening","sharpening-mass-correction","solid-excess"].includes(stage.id))return [];
    if(stage.id==="density-post-process")return [{...stage,label:"Phi surface publication",phaseLabels:[P.surface.label],spendsNoFrameTime:true,
      toggle:undefined,state:()=>"on" as const,
      chip:()=>"phi = 0",
      tip:{summary:"The renderer reads canonical level-set vertices through the accepted ownership generation. Publication switches the accepted view after the step completes; no dense expansion pass is required."}}];
    const mapped={...stage,phaseLabels:[...(stage.phaseLabels??[]),
      ...(stage.id==="pressure-projection"?["Pressure projection + surface publication"]:
        stage.id==="rigid-coupling"?["Rigid coupling + surface publication"]:[])],controls:stage.controls?.filter(control=>!("param" in control &&
      ["pressureCycleBudget","pressureBudgetHeadroom","volumeStorage","velocityTransport"].includes(control.param))),tip:{...stage.tip,
      summary:stage.tip.summary.replaceAll("surface density","level-set geometry"),
      reads:stage.tip.reads?.replaceAll("surface density","vertex phi and V")}};
    // The two-level sampler reads this stage's own hierarchy, and the shell
    // tiles are what this stage's finest passes run on, so both readouts sit
    // beside the sweep budget that produced the field they sample.
    if(stage.id==="velocity-extension")return [{...mapped,
      tip:{...mapped.tip,summary:"Extend nearby velocities with the narrow-band front, then fill missing air velocities from the nearest original source carried through the hierarchy. Keeps distant stationary liquid from slowing falling drops. The final 3D fill also packs the transport field."},
      controls:[
        ...(mapped.controls ?? []).filter(control=>
          !(control.kind === "param-choice" && control.param === "activeRegion")
          && !(control.kind === "readout" && (control.label === "Work box" || control.label === "Sweeps with work"))),
        ...twoLevelControls],
      chip:context=>`shared support · hierarchy fill · ${twoLevelChip(context)}`}];
    // MacCormack stays a dense-comparison option; Geometric runs one SL pass.
    if(stage.id==="velocity-advection")return [{...mapped,
      tip:{...mapped.tip,summary:"Algorithm 1 step 3: one semi-Lagrangian backward-trace velocity transport, followed by gravity, viscosity, and surface tension.",reads:"extended MAC velocity"},
      chip:()=>"one backward-trace pass"}];
    if(stage.id==="pressure-cycles")return [{...mapped,
      tip:{...mapped.tip,summary:"One coupled pressure solve across page seams. Repeat V-cycles with loose inner accuracy while residual reduction is good. Switch to Full-Cycles when progress stalls or the V-cycle budget runs out. Publish only when the fine residual meets tolerance within the configured cycle budget. A nonfinite or worsening solve stops the frame before projection. Projected Jacobi updates preserve reflection symmetry."},
      controls:[...(mapped.controls ?? []).filter(control=>control.kind!=="readout").map(control => control.kind === "param-range" && control.param === "pressureSweeps"
        ? {...control,hint:"Projected Jacobi sweeps before and after each coarse correction. Changing this rebuilds the plan and resets time."}
        : control.kind === "param-range" && control.param === "pressureVCycles"
          ? {...control,hint:"Available cheap corrections. Repeat while residual reduction is good; skip the remaining V-cycles for Full-Cycles when progress stalls. Encode only what the current frame needs."}
          : control),
        {kind:"readout" as const,label:"Completed cycles",hint:"Coupled cycles in the last successfully completed step.",
          value:(context:FluidPipelineContext)=>context.info?.uniformPressureCyclesExecuted === undefined ? "—" : String(context.info.uniformPressureCyclesExecuted)},
        {kind:"readout" as const,label:"Residual ∞-norm",hint:"Accepted finest-level divergence residual. Projection is withheld if the requested tolerance is not met.",
          value:(context:FluidPipelineContext)=>context.info?.uniformPressureAcceptedResidual === undefined ? "—" : context.info.uniformPressureAcceptedResidual.toExponential(2)}],
      chip:context=>context.info?.uniformMixedGeneration!==undefined
        ? `V-first adaptive · ${context.info.uniformPressureCyclesExecuted??0} accepted-step cycles · projected Jacobi`
        : "V-first adaptive · awaiting first solve"}];
    if(stage.id==="pressure-finish")return [{...mapped,
      tip:{...mapped.tip,summary:"Reconstruct pressure gradients from the converged coupled iterate before projection. A failed solve stops before this stage.",writes:"pressure reconstruction and residual diagnostics"},
      chip:()=>"converged gradient reconstruction"}];
    if(stage.id==="pressure-system")return [{...mapped,
      chip:()=>"shared ownership · coupled RHS pyramid",
      controls:[...(mapped.controls ?? []),{kind:"param-choice" as const,param:"surfaceDeficitBalancing",label:"Surface-deficit balancing",options:onOff,hint:"Preserve overfill expansion and balance it globally with contraction in underfilled liquid. Reduces persistent sloshing."}]}];
    return [mapped];
  }),
};
