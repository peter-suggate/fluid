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
    : "dense finest lattice",
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
    return [{...stage,tip:{...stage.tip,
      summary:stage.tip.summary.replaceAll("surface density","level-set geometry"),
      reads:stage.tip.reads?.replaceAll("surface density","vertex phi and V")}}];
  }),
};
