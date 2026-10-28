import { booleanQuery, choiceQuery, numberQuery, queryRecord } from "../../../framework/persistence";
import { DEFAULT_SVO_LIGHTING_OPTIONS, type SvoConeTracingMode } from "../../pipeline/svo-render-options";
import { DEFAULT_SVO_RENDER_TUNING, type SvoRenderTuning } from "../../pipeline/svo-render-tuning";
export interface LightingQueryState { svoNativeAmbientOcclusionEnabled: boolean; svoSurfaceTexturesEnabled: boolean; svoShadowsEnabled: boolean; svoAmbientOcclusionEnabled: boolean; svoConeTracingMode: SvoConeTracingMode; svoLatticeVisibilityEnabled: boolean }
export const lightingQuery = queryRecord<LightingQueryState>({
  svoNativeAmbientOcclusionEnabled: booleanQuery("svoNativeAO", DEFAULT_SVO_LIGHTING_OPTIONS.nativeAmbientOcclusionEnabled),
  svoSurfaceTexturesEnabled: booleanQuery("svoSurfaceTextures", DEFAULT_SVO_LIGHTING_OPTIONS.surfaceTexturesEnabled),
  svoShadowsEnabled: booleanQuery("svoShadows", DEFAULT_SVO_LIGHTING_OPTIONS.shadowsEnabled),
  svoAmbientOcclusionEnabled: booleanQuery("svoAO", DEFAULT_SVO_LIGHTING_OPTIONS.ambientOcclusionEnabled),
  svoConeTracingMode: choiceQuery("svoCones", DEFAULT_SVO_LIGHTING_OPTIONS.coneTracingMode, ["cones", "exact", "raster-ao", "off"]),
  svoLatticeVisibilityEnabled: booleanQuery("svoLattice", DEFAULT_SVO_LIGHTING_OPTIONS.latticeVisibilityEnabled),
});
export const lightingTuningQuery = queryRecord<Pick<SvoRenderTuning, "maximumShadedLights" | "waterShadowsEnabled" | "rasterCoarseAoStrength" | "rasterFarAoTaps" | "rasterStillViewReuse" | "rasterShadowResolution">>({
  rasterShadowResolution: { key: "svoShadowResolution", initial: DEFAULT_SVO_RENDER_TUNING.rasterShadowResolution, read: raw => raw === "4096" ? 4096 : 2048, write: value => { if(value !== 2048 && value !== 4096) throw new Error(`Invalid shadow resolution: ${value}`); return String(value); } },
  maximumShadedLights: numberQuery("svoLightSamples", DEFAULT_SVO_RENDER_TUNING.maximumShadedLights, 1, 32),
  rasterStillViewReuse: booleanQuery("svoStillReuse", DEFAULT_SVO_RENDER_TUNING.rasterStillViewReuse),
  rasterFarAoTaps: numberQuery("svoFarAO", DEFAULT_SVO_RENDER_TUNING.rasterFarAoTaps, 0, 36),
  rasterCoarseAoStrength: numberQuery("svoCoarseAO", DEFAULT_SVO_RENDER_TUNING.rasterCoarseAoStrength, 0, 1),
  waterShadowsEnabled: booleanQuery("svoWaterShadows", DEFAULT_SVO_RENDER_TUNING.waterShadowsEnabled),
});
