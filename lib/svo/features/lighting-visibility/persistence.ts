import { booleanQuery, choiceQuery, queryRecord } from "../../../framework/persistence";
import { DEFAULT_SVO_LIGHTING_OPTIONS, type SvoConeTracingMode } from "../../pipeline/svo-render-options";
export interface LightingQueryState { svoShadowsEnabled: boolean; svoAmbientOcclusionEnabled: boolean; svoConeTracingMode: SvoConeTracingMode }
export const lightingQuery = queryRecord<LightingQueryState>({
  svoShadowsEnabled: booleanQuery("svoShadows", DEFAULT_SVO_LIGHTING_OPTIONS.shadowsEnabled),
  svoAmbientOcclusionEnabled: booleanQuery("svoAO", DEFAULT_SVO_LIGHTING_OPTIONS.ambientOcclusionEnabled),
  svoConeTracingMode: choiceQuery("svoCones", DEFAULT_SVO_LIGHTING_OPTIONS.coneTracingMode, ["cones", "exact", "off"]),
});
