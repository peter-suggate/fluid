import { booleanQuery, choiceQuery, numberQuery, queryRecord } from "../../../framework/persistence";
import { DEFAULT_SVO_LIGHTING_OPTIONS, type SvoPrimaryTraversalMode } from "../../pipeline/svo-render-options";
import { DEFAULT_SVO_RENDER_TUNING, SVO_SURFACE_MESH_LOD_PIXELS_MAXIMUM, type SvoRenderTuning } from "../../pipeline/svo-render-tuning";
export interface PrimaryQueryState { svoPrimaryTraversal: SvoPrimaryTraversalMode; silhouetteRefinementEnabled: boolean }
export const primaryQuery = queryRecord<PrimaryQueryState>({
  svoPrimaryTraversal: choiceQuery("svoPrimary", DEFAULT_SVO_LIGHTING_OPTIONS.primaryTraversal, ["mesh", "traced", "raster"]),
  silhouetteRefinementEnabled: booleanQuery("svoPrimarySeamClosure", DEFAULT_SVO_LIGHTING_OPTIONS.silhouetteRefinementEnabled),
});
export const primaryTuningQuery = queryRecord<Pick<SvoRenderTuning, "surfaceMeshLodPixels">>({
  surfaceMeshLodPixels: numberQuery("svoMeshLodPixels", DEFAULT_SVO_RENDER_TUNING.surfaceMeshLodPixels, 0, SVO_SURFACE_MESH_LOD_PIXELS_MAXIMUM),
});
