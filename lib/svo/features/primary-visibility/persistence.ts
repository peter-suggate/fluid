import { booleanQuery, choiceQuery, numberQuery, queryRecord } from "../../../framework/persistence";
import { DEFAULT_SVO_LIGHTING_OPTIONS, type SvoPrimaryTraversalMode } from "../../pipeline/svo-render-options";
import { DEFAULT_SVO_RENDER_TUNING, SVO_SURFACE_MESH_LOD_PIXELS_MAXIMUM, type SvoRenderTuning } from "../../pipeline/svo-render-tuning";
export interface PrimaryQueryState { svoPrimaryTraversal: SvoPrimaryTraversalMode; silhouetteRefinementEnabled: boolean }
export const primaryQuery = queryRecord<PrimaryQueryState>({
  svoPrimaryTraversal: choiceQuery("svoPrimary", DEFAULT_SVO_LIGHTING_OPTIONS.primaryTraversal, ["mesh", "traced", "raster"]),
  silhouetteRefinementEnabled: booleanQuery("svoPrimarySeamClosure", DEFAULT_SVO_LIGHTING_OPTIONS.silhouetteRefinementEnabled),
});
const primaryTuningFields = queryRecord<Pick<SvoRenderTuning, "surfaceMeshing" | "surfaceMeshContourInflation" | "surfaceMeshContours" | "surfaceMeshLodPixels" | "surfaceMeshFilteringEnabled" | "surfaceMeshNormalSmoothing" | "surfaceMeshNormalStrength" | "surfaceMeshMaxCoarsening" | "surfaceMeshLodHysteresis" | "surfaceMeshNormalAgreement" | "surfaceMeshPreserveCloseNormals">>({
  surfaceMeshContourInflation: numberQuery("svoMeshContourInflation", 0, 0, 0.5),
  surfaceMeshing: choiceQuery("svoMesher", "voxels", ["voxels", "contours", "dual-contouring", "dual-marching-cubes"]),
  surfaceMeshContours: booleanQuery("svoMeshContours", false),
  surfaceMeshFilteringEnabled: booleanQuery("svoMeshFilter", false),
  surfaceMeshNormalSmoothing: booleanQuery("svoMeshNormals", true),
  surfaceMeshNormalStrength: numberQuery("svoMeshNormalStrength", 1, 0, 1),
  surfaceMeshMaxCoarsening: numberQuery("svoMeshMaxLevel", 3, 0, 3),
  surfaceMeshLodHysteresis: numberQuery("svoMeshHysteresis", 0.15, 0, 0.3),
  surfaceMeshNormalAgreement: numberQuery("svoMeshNormalAgreement", 0.5, 0, 1),
  surfaceMeshPreserveCloseNormals: booleanQuery("svoMeshCloseNormals", true),
  surfaceMeshLodPixels: numberQuery("svoMeshLodPixels", DEFAULT_SVO_RENDER_TUNING.surfaceMeshLodPixels, 0, SVO_SURFACE_MESH_LOD_PIXELS_MAXIMUM),
});

// Migrate old URLs whose positive threshold was also their enable switch.
export const primaryTuningQuery = {
  ...primaryTuningFields,
  write(query: URLSearchParams, value: Parameters<typeof primaryTuningFields.write>[1]) {
    primaryTuningFields.write(query, value);
    // Disambiguate saved disabled thresholds from legacy enable-by-threshold URLs.
    if (query.has("svoMeshLodPixels")) query.set("svoMeshFilter", value.surfaceMeshFilteringEnabled ? "1" : "0");
  },
  read(query: URLSearchParams) {
    const raw = primaryTuningFields.read(query);
    const surfaceMeshing = !query.has("svoMesher") && raw.surfaceMeshContours ? "contours" : raw.surfaceMeshing;
    const value = {...raw, surfaceMeshing, surfaceMeshContours: surfaceMeshing === "contours"};
    if (!query.has("svoMeshFilter") && query.has("svoMeshLodPixels")) {
      const legacy = Number(query.get("svoMeshLodPixels"));
      return { ...value, surfaceMeshLodPixels: legacy === 0 ? DEFAULT_SVO_RENDER_TUNING.surfaceMeshLodPixels : value.surfaceMeshLodPixels,
        surfaceMeshFilteringEnabled: Number.isFinite(legacy) && legacy > 0 && legacy <= SVO_SURFACE_MESH_LOD_PIXELS_MAXIMUM };
    }
    return value;
  },
};
