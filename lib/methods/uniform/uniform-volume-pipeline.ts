import type { FluidPipelineGraph, FluidPipelineStage } from "../../core/fluid-pipeline";
import { UNIFORM_FLUID_PIPELINE } from "./webgpu-uniform-reference";
import { UNIFORM_VOLUME_PHASE as P } from "./uniform-volume-stages";

const volumeStages: FluidPipelineStage[] = [
  ["phi", "Vertex level set", "RK2 characteristics and bounded closest-point redistancing; phi is independent of V."],
  ["coupling", "Conservative volume transport", "Eight box-overlap donors plus an identity fallback; three receiver/donor balancing rounds."],
  ["balance", "Liquid capacity balancing", "Up to 64 receiver-cap/donor-normalization rounds, with GPU convergence gating. Includes final volume gather and cached phi capacity."],
  ["sharpen", "Volume sharpening", "Eight symmetric face-transfer sweeps with aggregate donor and receiver budgets; phi is immutable."],
].map(([id,label,summary]) => ({
  id: `uniform-volume-${id}`, band:"surface", side:"left", label:label!,
  phaseLabels:[P[id as "phi"|"coupling"|"balance"|"sharpen"].label],
  tip:{summary:summary!}, state: context => id === "sharpen" && context.values.densitySharpening === "off" ? "off" : "on",
  chip:()=>"dense finest lattice",
  ...(id === "sharpen" ? {toggle:{param:"densitySharpening",on:"on",off:"off"}} : {}),
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
