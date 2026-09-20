import type { FluidPipelineGraph, FluidPipelineStage, FluidPipelineContext } from "../../core/fluid-pipeline";
import { UNIFORM_FLUID_PIPELINE } from "./uniform-pipeline";
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
  uniformVolumePageTransportReceiptMs?: number; uniformVolumePageSharpenReceiptMs?: number;
  uniformVolumeTransportWorkgroups?: number; uniformVolumeSharpenWorkgroups?: number;
  uniformVolumePageEdge?: number; uniformVolumePagesActive?: number; uniformVolumePagesTotal?: number;
  uniformVolumePageBytes?: number; uniformVolumePageStage?: string;
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
/**
 * The solve window, relabelled for the predicate Uniform Geometric uses.
 *
 * It is the paper method's `activeRegion` param and its constructor-level
 * update kind, on the same stage, so the two methods can never disagree about
 * what the control means; what differs is the seed (liquid above the dust
 * floor OR a vertex on the liquid side of the 4h band, never a solid), the
 * padding (the largest reach any stage uses), and that the box is aligned to
 * the 4h tile lattice so a workgroup is still a tile.
 */
const solveWindowForcedDense = (context: FluidPipelineContext): string | undefined => {
  if (context.values.activeRegion !== "on") return "whole domain";
  if (dustThreshold(context) <= 0) return "whole domain · dust floor off";
  return undefined;
};
const solveWindowControls = [
  {kind:"param-choice" as const,param:"activeRegion",label:"Solve window",
    options:[{value:"on",label:"Liquid window",
      hint:"Every kernel, every pressure-multigrid pass and every extension-hierarchy pass runs on the box holding the liquid, the near-surface band and this step's sources, padded by the largest reach any stage uses. Empty air outside it costs nothing. Needs the dust floor above zero."},
      {value:"off",label:"Whole domain",
      hint:"The full bounding box, dispatched directly. The dense control."}]},
  {kind:"readout" as const,label:"Work box",
    hint:"The GPU-measured dispatch box of the latest diagnostics sample: the union of this step's padded seed box with the previous one, which is what gives every ping-pong target one clearing tail.",
    value:(context: FluidPipelineContext)=>{
      const forced = solveWindowForcedDense(context);
      if (forced) return forced;
      const cells = context.info?.uniformActiveRegionCellCount;
      const fraction = context.info?.uniformActiveRegionFraction;
      const total = context.info?.cellCount;
      if (cells === undefined || fraction === undefined || !total) return "—";
      return `${cells.toLocaleString()} / ${total.toLocaleString()} cells · ${(100 * fraction).toFixed(1)}%`;
    }},
  {kind:"readout" as const,label:"Window launches",
    hint:"How the window's dispatches are sized. The host picks the group counts from a box a few steps old while every kernel still reads this step's exact origin, because an indirect launch costs several times a direct one on this lane. Clipped steps are steps whose exact box outgrew the host's counts: the far edge of the window is not dispatched that step, so a front stalls there, and the host answers with whole-domain counts for the next eight steps. Dense steps are those whole-domain ones, plus start-up, scene edits, drops, and new or expanded inlet support. A continuing inlet keeps its footprint in the window without forcing dense launches every step.",
    value:(context: FluidPipelineContext)=>{
      const forced = solveWindowForcedDense(context);
      if (forced) return forced;
      const mode = context.info?.uniformSolveWindowDispatch;
      if (mode === undefined) return "—";
      if (mode === "indirect") return "GPU indirect records";
      const clipped = context.info?.uniformSolveWindowClippedSteps ?? 0;
      const dense = context.info?.uniformSolveWindowDenseSteps ?? 0;
      return `host-sized · ${clipped} clipped · ${dense} dense`;
    }},
];
const solveWindowChip = (context: FluidPipelineContext) => {
  if (solveWindowForcedDense(context)) return undefined;
  const fraction = context.info?.uniformActiveRegionFraction;
  return fraction === undefined ? "solve window" : `solve window ${(100 * fraction).toFixed(1)}%`;
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
  options:[{value:"abandoned",label:"Abandoned",hint:"A cell holding at least half its open capacity in V owns a pressure row where centre phi is positive AND no face neighbour is phi-liquid: exactly the cells whose faces the projection would zero. Thin films keep incompressibility; beside a phi surface, phi alone places the free surface."},
    {value:"all",label:"All",hint:"V claims the row, and sets the ghost distance, wherever it implies more liquid than phi. V sits in a patchy one-cell layer over a phi surface, so random columns read a cell taller than their neighbours and the surface bubbles. Kept for comparison."},
    {value:"off",label:"Off",hint:"Rows from centre phi alone: the control. A film under half a cell has no pressure, and with no liquid centre left the solve stops."}]};
/**
 * The CM11a lattice the hierarchy is planned for.
 *
 * It belongs on the system-build stage because that is where the plan is
 * spent: the topology and RHS pyramid, and every level's dimensions, come
 * from the capacity this control picks. The cliff it exists for is the
 * planner's, not the launches': a domain whose shortest axis will not divide
 * the level count falls out of lockstep coarsening into semi-coarsening, and
 * pays half again the levels and twice the passes for the same liquid.
 */
const pressureLatticeControls = [
  {kind:"param-choice" as const,param:"pressureWindow",label:"Pressure lattice",
    options:[{value:"window",label:"Liquid window",
      hint:"Plan the hierarchy for a capacity that holds the solve window, seated at an aligned origin in simulation cells. Halo cells that land inside the domain are far air: open, phi positive, p = 0. Re-planned when the window outgrows the capacity, and — after thirty settled steps — when it shrinks well inside one."},
      {value:"domain",label:"Whole domain",
      hint:"One hierarchy for the full lattice, planned once at load. The control, and the fallback a violated or freshly reset step uses."}],
    enabled:(context: FluidPipelineContext)=>context.values.activeRegion === "on" && dustThreshold(context) > 0},
  {kind:"readout" as const,label:"Lattice",
    hint:"Capacity in cells and origin in simulation cells, of the instance the latest step solved on, with the level count its plan has. Whole domain while the solve window is off, at start-up, and for the eight steps after a containment violation.",
    value:(context: FluidPipelineContext)=>{
      const info = context.info as unknown as
        { uniformPressureLattice?: string; uniformPressureLatticeWindowed?: boolean } | null;
      if (context.values.activeRegion !== "on") return "whole domain";
      const lattice = info?.uniformPressureLattice;
      if (lattice === undefined) return "whole domain";
      const facts = context.info?.uniformPipelineFacts;
      const levels = facts ? ` · ${facts.multigridLevels} levels` : "";
      return `${lattice}${info?.uniformPressureLatticeWindowed === false ? " · domain" : ""}${levels}`;
    }},
  {kind:"readout" as const,label:"Re-plans",
    hint:"Hierarchies built since load, and how long the last one took on the host. Creation reuses the compiled pipelines and bind-group layouts, so it is a plan plus texture and buffer allocation, not a shader compile.",
    value:(context: FluidPipelineContext)=>{
      const info = context.info as unknown as
        { uniformPressureLatticeReplans?: number; uniformPressureLatticeReplanMs?: number } | null;
      const count = info?.uniformPressureLatticeReplans;
      if (count === undefined) return "—";
      const ms = info?.uniformPressureLatticeReplanMs;
      return ms ? `${count} · last ${ms.toFixed(1)} ms` : `${count}`;
    }},
];
const transportControls = [
  {kind:"param-choice" as const,param:"volumeStorage",label:"Volume record storage",
    options:[{value:"auto",label:"Automatic",hint:"Page-first work for large scenes; direct small-scene path."},{value:"dense",label:"Dense",hint:"Dense storage and synchronous advance."},
      {value:"pages16",label:"16³ pages",hint:"GPU-assigned transport/sharpening pages; capacity reserved at initialization."},
      {value:"pages32",label:"32³ pages",hint:"Larger pages amortize address translation. Rebuilds the simulation."}]},
  {kind:"readout" as const,label:"Resident volume pages",
    hint:"Actual last-stage active page count and total reserved arena capacity for the 80-byte records. Reservation is currently domain-sized; compute work follows active tiles.",
    value:(context: FluidPipelineContext)=>{const info=volumeInfo(context);
      return info?.uniformVolumePageEdge ? `${info.uniformVolumePagesActive ?? 0} / ${info.uniformVolumePagesTotal ?? 0} · ${((info.uniformVolumePageBytes ?? 0)/1048576).toFixed(1)} MiB reserved · ${info.uniformVolumePageStage ?? "initial"}` : "dense";}},
  {kind:"readout" as const,label:"Scheduled volume tiles",
    hint:"Actual 4³ workgroups dispatched for transport / sharpening. Sharpening reuses its list and cached geometry across eight sweeps. Domain capacity is shown for comparison.",
    value:(context:FluidPipelineContext)=>{const info=volumeInfo(context);
      if(info?.uniformVolumeTransportWorkgroups===undefined)return "direct small-scene / dense schedule";
      const d=context.info;
      const capacity=d?Math.ceil(d.nx/4)*Math.ceil(d.ny/4)*Math.ceil(d.nz/4):0;
      return `${info.uniformVolumeTransportWorkgroups} / ${info.uniformVolumeSharpenWorkgroups??0} · domain ${capacity}`;}},
  {kind:"readout" as const,label:"Page scheduling",
    hint:"GPU page compaction and indirect tile dispatches stay in one command buffer. No CPU page-demand readback, arena allocation, or frame continuation.",
    value:(context:FluidPipelineContext)=>volumeInfo(context)?.uniformVolumePageEdge?"GPU only · one submission":"direct"},
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
const onOff=[{value:"on",label:"On"},{value:"off",label:"Off"}];
/** The two stages that write V into phi sit on the level set they write. */
const phiAgreementControls = [
  {kind:"param-choice" as const,param:"totalSurfaceVolume",label:"Total surface volume",options:onOff,
    hint:"One bounded global normal shift to match the surface volume to V. No regional correction or cellwise reconstruction."},
  {kind:"param-choice" as const,param:"phiSeedFromVolume",label:"Seed from V",options:onOff,
    hint:"Where no cell centre nearby is phi-liquid but the cells around a vertex average over a quarter full, write V's implied depth into phi. A film under half a cell then owns ordinary pressure rows and renders; beside an existing phi surface it never fires."},
  {kind:"param-choice" as const,param:"phiAgreement",label:"Follow V",options:onOff,
    hint:"Shift band phi along its normal by V minus phi's fill, gathered over the 8x8x8 cells around each vertex with tent weights. Per cell that residual is noise; as a patch integral it is phi's transport drift. Wants compaction on, or the residual has the wrong sign."},
  {kind:"param-range" as const,param:"phiAgreementGain",label:"Gain",unit:"cells",
    min:0,max:1,step:0.01,digits:2,hint:"Cells of shift per unit patch residual. 0.25 roughened the dam break threefold; 0.05 is inside baseline noise.",
    enabled:(context: FluidPipelineContext)=>context.values.phiAgreement === "on"},
  {kind:"param-range" as const,param:"phiAgreementClamp",label:"Clamp",unit:"cells / step",
    min:0,max:0.5,step:0.005,digits:3,hint:"Largest shift in one step. The dam break is indifferent from 0.01 to 0.05 at gain 0.05; the thin film needs at least 0.02 to keep up with its own erosion.",
    enabled:(context: FluidPipelineContext)=>context.values.phiAgreement === "on"},
];
const phiChip = (context: FluidPipelineContext) => {
  const parts=[context.values.totalSurfaceVolume === "on" ? "total volume constrained" : "",context.values.phiSeedFromVolume === "on" ? "seeded from V" : "",
    context.values.phiAgreement === "on" ? "follows V" : ""].filter(Boolean);
  return parts.length ? `dense finest lattice · ${parts.join(" · ")}` : "dense finest lattice";
};
const volumeStages: FluidPipelineStage[] = [
  ["phi", "Vertex level set", "RK2 characteristics and bounded closest-point redistancing. The optional global volume constraint runs after conservative gather."],
  ["coupling", "Conservative volume transport", "Eight box-overlap donors plus an identity fallback; three receiver/donor balancing rounds."],
  ["gather", "Conservative volume gather", "Gather donor-normalized liquid volume, optionally constrain total surface volume, and cache corrected phi capacity for sharpening."],
  ["sharpen", "Volume sharpening", "Eight symmetric face-transfer sweeps with aggregate donor and receiver budgets; phi is immutable across them, so the 4h work map skips whole tiles with no cell in the admission band bit-identically to the dense schedule."],
].map(([id,label,summary]) => ({
  id: `uniform-volume-${id}`, band:"surface", side:"left", label:label!,
  phaseLabels:[P[id as "phi"|"coupling"|"gather"|"sharpen"].label],
  tip:{summary:summary!}, state: context => (id === "sharpen" && context.values.densitySharpening === "off") ? "off" : "on",
  chip:context=>id === "gather"
    ? "conservative gather"
    : id === "sharpen" ? sharpenChip(context)
    : id === "coupling" ? couplingChip(context)
    : phiChip(context),
  ...(id === "phi" ? { controls: phiAgreementControls } : {}),
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
  ...(id === "sharpen" ? {
    toggle:{param:"densitySharpening",on:"on",off:"off"},
    controls:[{kind:"param-choice" as const,param:"sharpeningWorkMap",label:"Work map",
      options:[{value:"on",label:"4h tiles",hint:"Skip prepare/propose/limit in tiles where no cell can be admitted; V is still copied on commit."},
        {value:"off",label:"Dense",hint:"The dense reference schedule, retained for comparison."}],
      enabled:(context: FluidPipelineContext)=>context.values.densitySharpening !== "off"},
      {kind:"param-choice" as const,param:"volumeCompaction",label:"Compaction",options:onOff,
      hint:"A liquid cell may pour all of its V into a deeper liquid neighbour, at any depth, so voids inside the liquid refill. Off, only the 2.1h band is admitted and only surplus over phi's fill moves. The work map admits the extra tiles only while they are under-full.",
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
      tip:{...mapped.tip,summary:"Extend nearby velocities with the narrow-band front, then fill missing air velocities from the nearest original source carried through the hierarchy. Keeps distant stationary liquid from slowing falling drops. The final 3D fill also packs the transport field."},
      controls:[{kind:"readout" as const,label:"Air fallback",value:()=>"Nearest source",
        hint:"Enabled by default. Carries original source locations through coarse levels; the front sweep budget is unchanged."},...solveWindowControls,
        ...(mapped.controls ?? []).filter(control=>
          !(control.kind === "param-choice" && control.param === "activeRegion")
          && !(control.kind === "readout" && control.label === "Work box")),
        ...twoLevelControls],
      chip:context=>{const extra=[solveWindowChip(context),twoLevelChip(context)].filter(Boolean).join(" · ");
        const base=mapped.chip?.(context);
        return extra?(base?`${base} · ${extra}`:extra):base;}}];
    // Which cells own a pressure row is decided where the topology and RHS
    // are built, so the V claim sits on that stage.
    if(stage.id==="pressure-system")return [{...mapped,
      controls:[...(mapped.controls ?? []),volumePressureRowsControl,{kind:"param-choice" as const,param:"surfaceDeficitBalancing",label:"Surface-deficit balancing",options:onOff,hint:"Preserve overfill expansion and balance it globally with contraction in underfilled liquid. Reduces persistent sloshing."},...pressureLatticeControls]}];
    // E2b shrinks both of these, off the same fine map, so the one control sits
    // on both stages rather than in a shelf away from the work it prices.
    if(stage.id==="velocity-advection"||stage.id==="pressure-projection")return [{...mapped,
      controls:[...(mapped.controls ?? []),...advectionControls],
      chip:context=>{const extra=advectionChip(context);const base=mapped.chip?.(context);
        return extra?(base?`${base} · ${extra}`:extra):base;}}];
    return [mapped];
  }),
};
