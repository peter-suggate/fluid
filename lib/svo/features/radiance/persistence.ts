import { booleanQuery, choiceQuery, numberQuery, queryRecord } from "../../../framework/persistence";
import { SVO_RADIANCE_RECONSTRUCTION_OPTIONS } from "./definition";
import { DEFAULT_SVO_RENDER_TUNING, type SvoRenderTuning } from "../../pipeline/svo-render-tuning";
import { DEFAULT_SVO_LIGHTING_OPTIONS } from "../../pipeline/svo-render-options";

export interface RadianceQueryState {
  svoVoxelSpecularReflectionsEnabled: boolean;
  svoVoxelSpecularSteps: number;
  svoGlobalIlluminationEnabled: boolean;
  svoWorldGiCacheEnabled: boolean;
}
export const radianceQuery = queryRecord<RadianceQueryState>({
  svoVoxelSpecularReflectionsEnabled: booleanQuery("svoVoxelSpecular", false),
  svoVoxelSpecularSteps: numberQuery("svoSpecularSteps", 128, 1, 512),
  svoGlobalIlluminationEnabled: booleanQuery("svoGI", DEFAULT_SVO_LIGHTING_OPTIONS.globalIlluminationEnabled),
  svoWorldGiCacheEnabled: booleanQuery("svoWorldGiCache", DEFAULT_SVO_LIGHTING_OPTIONS.worldGiCacheEnabled),
});
export const radianceTuningQuery = queryRecord<Pick<SvoRenderTuning, "coneRadianceReconstruction" | "giBounceStrength" | "giOcclusionStrength" | "giEnvironmentStrength" | "giDirectStrength" | "giConeAperture" | "giConeCount" | "coneStepBudget">>({
  coneRadianceReconstruction: choiceQuery("svoReconstruction", DEFAULT_SVO_RENDER_TUNING.coneRadianceReconstruction,
    SVO_RADIANCE_RECONSTRUCTION_OPTIONS.map(option => option.value)),
  giBounceStrength: numberQuery("svoGiBounce", DEFAULT_SVO_RENDER_TUNING.giBounceStrength, 0, 4),
  giOcclusionStrength: numberQuery("svoGiOcclusion", DEFAULT_SVO_RENDER_TUNING.giOcclusionStrength, 0, 1),
  giEnvironmentStrength: numberQuery("svoGiEnvironment", DEFAULT_SVO_RENDER_TUNING.giEnvironmentStrength, 0, 2),
  giDirectStrength: numberQuery("svoGiDirect", DEFAULT_SVO_RENDER_TUNING.giDirectStrength, 0, 2),
  giConeAperture: numberQuery("svoGiAperture", DEFAULT_SVO_RENDER_TUNING.giConeAperture, 0.4, 1.4),
  coneStepBudget: numberQuery("svoConeSteps", DEFAULT_SVO_RENDER_TUNING.coneStepBudget, 1, 48),
  giConeCount: numberQuery("svoGiCones", DEFAULT_SVO_RENDER_TUNING.giConeCount, 3, 4),
});
