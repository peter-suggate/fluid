import { booleanQuery, choiceQuery, queryRecord } from "../../../framework/persistence";
import { DEFAULT_SVO_LIGHTING_OPTIONS, type SvoPrimaryTraversalMode } from "../../pipeline/svo-render-options";
export interface PrimaryQueryState { svoPrimaryTraversal: SvoPrimaryTraversalMode; silhouetteRefinementEnabled: boolean }
export const primaryQuery = queryRecord<PrimaryQueryState>({
  svoPrimaryTraversal: choiceQuery("svoPrimary", DEFAULT_SVO_LIGHTING_OPTIONS.primaryTraversal, ["mesh", "traced", "raster"]),
  silhouetteRefinementEnabled: booleanQuery("svoPrimarySeamClosure", DEFAULT_SVO_LIGHTING_OPTIONS.silhouetteRefinementEnabled),
});
