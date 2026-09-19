import type { FluidPipelineGraph, FluidPipelineStage, FluidPipelineContext } from "../../core/fluid-pipeline";
import { UNIFORM_FLUID_PIPELINE } from "./webgpu-uniform-reference";
import { UNIFORM_VOLUME_PHASE as P } from "./uniform-volume-stages";

/** The solver's published tile-map counts, when the map ran and diagnostics arrived. */
const workMap = (context: FluidPipelineContext) => {
  const info = context.info as unknown as
    { uniformSharpenWorkMap?: boolean; uniformSharpenTilesActive?: number; uniformSharpenTilesTotal?: number } | null;
  if (context.values.sharpeningWorkMap === "off" || info?.uniformSharpenWorkMap === false) return undefined;
  const active = info?.uniformSharpenTilesActive, total = info?.uniformSharpenTilesTotal;
  if (active === undefined || total === undefined || total <= 0) return undefined;
  return {active,total,percent:Math.round(100*active/total)};
};
const sharpenChip = (context: FluidPipelineContext) => {
  if (context.values.sharpeningWorkMap === "off") return "dense finest lattice";
  const map = workMap(context);
  return map ? `4h work map · ${map.percent}% tiles` : "4h work map";
};
type VolumeInfo = {
  uniformVolumeDustCells?: number; uniformVolumeDustMass_cells?: number;
  uniformTwoLevelVelocity?: boolean; uniformTwoLevelFineTiles?: number; uniformTwoLevelTilesTotal?: number;
  uniformTwoLevelShellTiles?: number; uniformTwoLevelShellReach?: number;
  uniformTwoLevelExtensionTiles?: boolean; uniformTwoLevelAdvectionTiles?: boolean;
  uniformTransportWorkMap?: boolean; uniformTransportTiles?: number;
  uniformTransportTilesTotal?: number; uniformTransportReachTiles?: number;
  uniformTransportReachMargin?: number;
  uniformTransportRequiredReachTiles?: number; uniformTransportMaxDisplacement_cells?: number;
} | null;
const volumeInfo = (context: FluidPipelineContext) => context.info as unknown as VolumeInfo;
/** The authored floor, in cell volumes; zero and absent both read as off. */
const dustThreshold = (context: FluidPipelineContext) => {
  const value = Number(context.values.volumeDustThreshold ?? 0);
  return Number.isFinite(value) && value > 0 ? value : 0;
};
const dustChip = (context: FluidPipelineContext) => {
  const threshold = dustThreshold(context);
  return threshold > 0 ? `dust floor ${threshold.toExponential(0)}` : "dense finest lattice";
};
/** The E1 fine map's share of the 4h tile grid, when the experiment ran. */
const fineMap = (context: FluidPipelineContext) => {
  const info = volumeInfo(context);
  if (context.values.twoLevelVelocity !== "on" || info?.uniformTwoLevelVelocity === false) return undefined;
  const fine = info?.uniformTwoLevelFineTiles, total = info?.uniformTwoLevelTilesTotal;
  if (fine === undefined || total === undefined || total <= 0) return undefined;
  return {fine,total,percent:Math.round(100*fine/total)};
};
/** Shell tiles: the set the extension's finest passes still run on. */
const shellMap = (context: FluidPipelineContext) => {
  const info = volumeInfo(context);
  if (context.values.twoLevelVelocity !== "on" || info?.uniformTwoLevelVelocity === false) return undefined;
  const shell = info?.uniformTwoLevelShellTiles, total = info?.uniformTwoLevelTilesTotal;
  if (shell === undefined || total === undefined || total <= 0) return undefined;
  return {shell,total,percent:Math.round(100*shell/total)};
};
const twoLevelChip = (context: FluidPipelineContext) => {
  if (context.values.twoLevelVelocity !== "on") return undefined;
  if (context.values.twoLevelExtension === "dense") {
    const fine = fineMap(context);
    return fine ? `two-level · dense · ${fine.percent}% fine` : "two-level · dense";
  }
  const shell = shellMap(context);
  return shell ? `two-level · tiles ${shell.percent}%` : "two-level · tiles";
};
const twoLevelControls = [
  {kind:"param-choice" as const,param:"twoLevelVelocity",label:"Sampler",
    options:[{value:"off",label:"All fine",hint:"Every velocity sample reads the finest lattice."},
      {value:"on",label:"Two-level",hint:"Outside the fine tile map, sample the 4h face table the extension hierarchy's own ceil(n/4) level publishes."}]},
  {kind:"param-choice" as const,param:"twoLevelExtension",label:"Extension work",
    options:[{value:"tiles",label:"Shell tiles",hint:"Seed, FIM sweeps, resolve, the finest prolong and the transport pack run only in the shell tiles. Restricts and every level at or below 4h stay dense."},
      {value:"dense",label:"Dense",hint:"The full-lattice extension, retained so the shrink can be measured on its own."}],
    enabled:(context: FluidPipelineContext)=>context.values.twoLevelVelocity === "on"},
  {kind:"param-range" as const,param:"twoLevelFineReach",label:"Fine reach",unit:"tiles",
    min:0,max:8,step:1,digits:0,
    hint:"Chebyshev dilation of the seed tiles. It must cover a step's backward trace plus the stencils that read beyond it.",
    enabled:(context: FluidPipelineContext)=>context.values.twoLevelVelocity === "on"},
  {kind:"param-range" as const,param:"twoLevelShellReach",label:"Shell reach",unit:"tiles",
    min:0,max:8,step:1,digits:0,
    hint:"Extra tiles the extension's working set adds past the fine set. Raise it first if far-air velocity looks wrong near the fine boundary.",
    enabled:(context: FluidPipelineContext)=>context.values.twoLevelVelocity === "on" && context.values.twoLevelExtension !== "dense"},
  {kind:"readout" as const,label:"Fine tiles",
    hint:"4×4×4 tiles sampling the finest lattice, in the latest diagnostics sample. The rest read the 4h face table.",
    value:(context: FluidPipelineContext)=>{const map=fineMap(context);return map?`${map.fine} / ${map.total} (${map.percent}%)`:"—";}},
  {kind:"readout" as const,label:"Shell tiles",
    hint:"Fine tiles dilated by the shell reach: the set where the extension's finest output must still be exact, and therefore the set its passes run on.",
    value:(context: FluidPipelineContext)=>{const map=shellMap(context);const reach=volumeInfo(context)?.uniformTwoLevelShellReach;
      return map?`${map.shell} / ${map.total} (${map.percent}%)${reach?` · +${reach}`:""}`:"—";}},
];
/** E2b's control and readout, shown on the two stages it shrinks. */
const advectionControls = [
  {kind:"param-choice" as const,param:"twoLevelAdvection",label:"Far-air work",
    options:[{value:"tiles",label:"Fine tiles",hint:"Outside the fine tiles, take the far-air arm directly: no backward traces, no face data, no pressure taps."},
      {value:"dense",label:"Dense",hint:"The full-lattice schedule, retained so the shrink can be measured on its own."}],
    enabled:(context: FluidPipelineContext)=>context.values.twoLevelVelocity === "on"},
  {kind:"readout" as const,label:"Fine tiles",
    hint:"4×4×4 tiles carrying liquid, a solid or a source within the fine reach. Only these run the full advection and projection.",
    value:(context: FluidPipelineContext)=>{const map=fineMap(context);return map?`${map.fine} / ${map.total} (${map.percent}%)`:"—";}},
];
const advectionChip = (context: FluidPipelineContext) => {
  if (context.values.twoLevelVelocity !== "on" || context.values.twoLevelAdvection === "dense") return undefined;
  const fine = fineMap(context);
  return fine ? `far air skipped · ${fine.percent}% tiles` : "far air skipped";
};
/**
 * E3. The restriction needs the class map the two-level sampler builds, and it
 * needs the dust floor, which is the only reason "V is zero outside the live
 * set" — the predicate it is exactly conservative under — holds. Either one off
 * means the solver has forced the dense schedule, so say which.
 */
const transportForcedDense = (context: FluidPipelineContext): string | undefined => {
  if (context.values.transportWorkMap === "dense") return "transport dense";
  if (context.values.twoLevelVelocity !== "on") return "transport dense · no tile map";
  if (dustThreshold(context) <= 0) return "transport dense · dust floor off";
  return undefined;
};
const transportMap = (context: FluidPipelineContext) => {
  const info = volumeInfo(context);
  if (transportForcedDense(context) || info?.uniformTransportWorkMap === false) return undefined;
  const live = info?.uniformTransportTiles, total = info?.uniformTransportTilesTotal;
  if (live === undefined || total === undefined || total <= 0) return undefined;
  return {live,total,percent:Math.round(100*live/total)};
};
/** The reach the dilation used against the reach the step's displacement required. */
const transportReach = (context: FluidPipelineContext) => {
  const info = volumeInfo(context);
  const used = info?.uniformTransportReachTiles;
  const required = info?.uniformTransportRequiredReachTiles;
  if (used === undefined || required === undefined) return undefined;
  return {used,required,short:used<required,
    displacement:info?.uniformTransportMaxDisplacement_cells ?? 0};
};
const volumePressureRowsControl = {kind:"param-choice" as const,param:"volumePressureRows",label:"Volume pressure rows",
  options:[{value:"on",label:"On",hint:"A cell holding at least half its open capacity in V owns a pressure row even where centre phi is positive, at the ghost distance that fill implies. Thin films keep incompressibility and the excess-volume divergence can reach V stacked in phi-dry cells."},
    {value:"off",label:"Off",hint:"Rows from centre phi alone: the control. A film under half a cell has no pressure, and with no liquid centre left the solve stops."}]};
const transportControls = [
  {kind:"param-choice" as const,param:"transportWorkMap",label:"Transport work",
    options:[{value:"tiles",label:"Live tiles",hint:"Build edges, sum and normalise donors and gather only in the 4h tiles that can hold or receive liquid this step. Outside them the gather stores V=0 and gamma=0 without evaluating either."},
      {value:"dense",label:"Dense",hint:"The full-lattice schedule, retained so the shrink can be measured on its own."}],
    enabled:(context: FluidPipelineContext)=>context.values.twoLevelVelocity === "on" && dustThreshold(context) > 0},
  {kind:"param-range" as const,param:"transportReach",label:"Transport margin",unit:"tiles",
    min:0,max:8,step:1,digits:0,
    hint:"Extra tiles added to the reach this step's measured maximum displacement requires. The set already tracks the flow, so this is headroom; zero is the exact predicate.",
    enabled:(context: FluidPipelineContext)=>context.values.twoLevelVelocity === "on" && dustThreshold(context) > 0},
  {kind:"readout" as const,label:"Live tiles",
    hint:"4×4×4 tiles the twelve transport passes ran on, in the latest diagnostics sample. The rest are known to hold V=0 and are skipped whole-workgroup.",
    value:(context: FluidPipelineContext)=>{const map=transportMap(context);
      return map?`${map.live} / ${map.total} (${map.percent}%)`:"—";}},
  {kind:"readout" as const,label:"Reach",
    hint:"Tiles the seed was dilated by, against the tiles this step's largest measured backward displacement required. Used is required plus the margin unless the shader's sixteen-tile cap bit, which is the only way it can read SHORT.",
    value:(context: FluidPipelineContext)=>{const reach=transportReach(context);
      if(!reach)return "—";
      return `${reach.used} used · ${reach.required} required${reach.short?" · SHORT":""}`
        + ` (${reach.displacement.toFixed(1)} cells)`;}},
];
const transportChip = (context: FluidPipelineContext) => {
  const forced = transportForcedDense(context);
  if (forced) return forced;
  const reach = transportReach(context);
  const map = transportMap(context);
  const short = reach?.short ? " · reach SHORT" : "";
  return map ? `live tiles ${map.percent}%${short}` : `live tiles${short}`;
};
/**
 * The floor's chip and the live set's, in that order. With the floor off the
 * set cannot run at all and the floor's own "dense finest lattice" already
 * says so, so the stage does not repeat it.
 */
const couplingChip = (context: FluidPipelineContext) =>
  dustThreshold(context) > 0 ? `${dustChip(context)} · ${transportChip(context)}` : dustChip(context);
const volumeStages: FluidPipelineStage[] = [
  ["phi", "Vertex level set", "RK2 characteristics and bounded closest-point redistancing; phi is independent of V."],
  ["coupling", "Conservative volume transport", "Eight box-overlap donors plus an identity fallback; three receiver/donor balancing rounds."],
  ["balance", "Liquid capacity balancing", "Maximum receiver overfill gates corrective work; remaining rounds exit immediately once tolerance is met. Final volume gather and cached phi capacity still run when disabled."],
  ["sharpen", "Volume sharpening", "Eight symmetric face-transfer sweeps with aggregate donor and receiver budgets; phi is immutable across them, so the 4h work map skips whole tiles with no cell in the admission band bit-identically to the dense schedule."],
].map(([id,label,summary]) => ({
  id: `uniform-volume-${id}`, band:"surface", side:"left", label:label!,
  phaseLabels:[P[id as "phi"|"coupling"|"balance"|"sharpen"].label],
  tip:{summary:summary!}, state: context => (id === "sharpen" && context.values.densitySharpening === "off")
    || (id === "balance" && context.values.liquidCapacityBalancing !== "on") ? "off" : "on",
  chip:context=>id === "balance"
    ? context.values.liquidCapacityBalancing !== "on" ? "off · gather only" : `${context.values.liquidCapacityBalancingRounds ?? 64} rounds maximum`
    : id === "sharpen" ? sharpenChip(context)
    : id === "coupling" ? couplingChip(context)
    : "dense finest lattice",
  ...(id === "coupling" ? {
    controls:[{kind:"param-range" as const,param:"volumeDustThreshold",label:"Dust floor",unit:"cell volumes",
      min:0,max:1e-3,step:1e-7,digits:7,
      hint:"Discard |V| below this wherever transport or sharpening writes V, ULP-scale negatives included. Zero is off and stores the untreated sum bit for bit."},
      {kind:"readout" as const,label:"Dust discarded",
      hint:"Cells zeroed by the floor across the gather and the eight commit sweeps of the latest step, and the mass that went with them.",
      value:(context: FluidPipelineContext)=>{
        const info=volumeInfo(context);const cells=info?.uniformVolumeDustCells;
        if(dustThreshold(context)<=0||cells===undefined)return "—";
        return `${cells.toLocaleString()} cells · ${(info?.uniformVolumeDustMass_cells ?? 0).toExponential(2)} cell volumes`;}},
      ...transportControls],
  } : {}),
  ...(id === "balance" ? {
    toggle:{param:"liquidCapacityBalancing",on:"on",off:"off"},
    controls:[{kind:"param-choice" as const,param:"liquidCapacityBalancing",label:"Balancing",
      options:[{value:"on",label:"On"},{value:"off",label:"Off"}]},
      {kind:"param-range" as const,param:"liquidCapacityBalancingRounds",label:"Max rounds",unit:"rounds",
      min:1,max:64,step:1,digits:0,
      hint:"Maximum balancing rounds per step; converged rounds exit immediately.",
      enabled:(context: FluidPipelineContext)=>context.values.liquidCapacityBalancing === "on"},
      {kind:"param-range" as const,param:"liquidCapacityBalancingTolerance",label:"Error tolerance",unit:"%",
        min:0,max:100,step:0.01,digits:2,
        hint:"Maximum receiver overfill / open capacity. At or below this tolerance correction kernels exit immediately.",
        enabled:(context: FluidPipelineContext)=>context.values.liquidCapacityBalancing === "on"}],
  } : {}),
  ...(id === "sharpen" ? {
    toggle:{param:"densitySharpening",on:"on",off:"off"},
    controls:[{kind:"param-choice" as const,param:"sharpeningWorkMap",label:"Work map",
      options:[{value:"on",label:"4h tiles",hint:"Skip prepare/propose/limit in tiles where no cell can be admitted; V is still copied on commit."},
        {value:"off",label:"Dense",hint:"The dense reference schedule, retained for comparison."}],
      enabled:(context: FluidPipelineContext)=>context.values.densitySharpening !== "off"},
      {kind:"readout" as const,label:"Active tiles",
      hint:"4×4×4 tiles holding a cell inside the admission band, in the latest diagnostics sample. Phi is fixed across the eight sweeps, so one classification schedules them all.",
      value:context=>{const map=workMap(context);return map?`${map.active} / ${map.total} (${map.percent}%)`:"—";}}],
  } : {}),
}));
export const UNIFORM_VOLUME_PIPELINE: FluidPipelineGraph = {
  methodId:"uniform-volume",
  bands:UNIFORM_FLUID_PIPELINE.bands.map(b=>b.id==="surface"?{...b,label:"Level set + volume"}:b),
  stages:UNIFORM_FLUID_PIPELINE.stages.flatMap(stage=>{
    if(stage.id==="density-advection")return volumeStages;
    if(["gamma-diffusion","interface-sharpening","sharpening-mass-correction","solid-excess"].includes(stage.id))return [];
    if(stage.id==="density-post-process")return [{...stage,label:"Phi surface publication",phaseLabels:[P.surface.label],
      toggle:undefined,controls:undefined,state:()=>"on" as const,chip:()=>"phi = 0",
      tip:{summary:"Publish the independent vertex level set in the renderer's dense contour encoding."}}];
    const mapped={...stage,tip:{...stage.tip,
      summary:stage.tip.summary.replaceAll("surface density","level-set geometry"),
      reads:stage.tip.reads?.replaceAll("surface density","vertex phi and V")}};
    // The two-level sampler reads this stage's own hierarchy, and the shell
    // tiles are what this stage's finest passes now run on, so both controls
    // belong beside the sweep budget that produced the field they sample.
    if(stage.id==="velocity-extension")return [{...mapped,
      controls:[...(mapped.controls ?? []),...twoLevelControls],
      chip:context=>{const extra=twoLevelChip(context);const base=mapped.chip?.(context);
        return extra?(base?`${base} · ${extra}`:extra):base;}}];
    // Which cells own a pressure row is decided where the topology and RHS
    // are built, so the V claim sits on that stage.
    if(stage.id==="pressure-system")return [{...mapped,
      controls:[...(mapped.controls ?? []),volumePressureRowsControl]}];
    // E2b shrinks both of these, off the same fine map, so the one control sits
    // on both stages rather than in a shelf away from the work it prices.
    if(stage.id==="velocity-advection"||stage.id==="pressure-projection")return [{...mapped,
      controls:[...(mapped.controls ?? []),...advectionControls],
      chip:context=>{const extra=advectionChip(context);const base=mapped.chip?.(context);
        return extra?(base?`${base} · ${extra}`:extra):base;}}];
    return [mapped];
  }),
};
