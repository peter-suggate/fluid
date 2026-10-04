import { booleanQuery, choiceQuery, numberQuery, queryRecord } from "../../../framework/persistence";
import { DEFAULT_SVO_LIGHTING_OPTIONS, type SvoConeTracingMode } from "../../pipeline/svo-render-options";
import { DEFAULT_SVO_RENDER_TUNING, type SvoRenderTuning } from "../../pipeline/svo-render-tuning";
export interface LightingQueryState { svoShadowsEnabled: boolean; svoAmbientOcclusionEnabled: boolean; svoConeTracingMode: SvoConeTracingMode; svoLatticeVisibilityEnabled: boolean }
export const lightingQuery = queryRecord<LightingQueryState>({
  svoShadowsEnabled: booleanQuery("svoShadows", DEFAULT_SVO_LIGHTING_OPTIONS.shadowsEnabled),
  svoAmbientOcclusionEnabled: booleanQuery("svoAO", DEFAULT_SVO_LIGHTING_OPTIONS.ambientOcclusionEnabled),
  svoConeTracingMode: choiceQuery("svoCones", DEFAULT_SVO_LIGHTING_OPTIONS.coneTracingMode, ["cones", "exact", "raster-ao", "off"]),
  svoLatticeVisibilityEnabled: booleanQuery("svoLattice", DEFAULT_SVO_LIGHTING_OPTIONS.latticeVisibilityEnabled),
});
export const lightingTuningQuery = queryRecord<Pick<SvoRenderTuning, "waterShadowsEnabled" | "rasterCoarseAoStrength">>({
  rasterCoarseAoStrength: numberQuery("svoCoarseAO", DEFAULT_SVO_RENDER_TUNING.rasterCoarseAoStrength, 0, 1),
  waterShadowsEnabled: booleanQuery("svoWaterShadows", DEFAULT_SVO_RENDER_TUNING.waterShadowsEnabled),
});
