import { booleanQuery, numberQuery, queryRecord } from "../../../framework/persistence";
import { DEFAULT_SVO_RENDER_TUNING, SVO_ENVIRONMENT_REFINEMENT_DEPTH_MAXIMUM, SVO_LOD_SCREEN_SPACE_PIXELS_MAXIMUM, type SvoRenderTuning } from "../../pipeline/svo-render-tuning";
export const constructionTuningQuery = queryRecord<Pick<SvoRenderTuning, "environmentRefinementDepth" | "environmentPlanarRefinementExemption" | "lodScreenSpacePixels">>({
  environmentRefinementDepth: numberQuery("svoRefinementDepth", DEFAULT_SVO_RENDER_TUNING.environmentRefinementDepth, 0, SVO_ENVIRONMENT_REFINEMENT_DEPTH_MAXIMUM),
  environmentPlanarRefinementExemption: booleanQuery("svoFlatExempt", DEFAULT_SVO_RENDER_TUNING.environmentPlanarRefinementExemption),
  lodScreenSpacePixels: numberQuery("svoLodPixels", DEFAULT_SVO_RENDER_TUNING.lodScreenSpacePixels, 0, SVO_LOD_SCREEN_SPACE_PIXELS_MAXIMUM),
});
