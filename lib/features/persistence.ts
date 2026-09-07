import { combineQueryCodecs } from "../framework/persistence";
import { svoFeatureQuery, type SvoQueryState } from "../svo/pipeline/persistence";
import { surfaceDisplayQuery, type SurfaceDisplayState } from "./surface-display/definition";
export type FeatureUIQueryState = SvoQueryState & SurfaceDisplayState;
export const uiFeatureQuery = combineQueryCodecs<FeatureUIQueryState>([svoFeatureQuery, surfaceDisplayQuery]);
