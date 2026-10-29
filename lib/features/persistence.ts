import { combineQueryCodecs } from "../framework/persistence";
import { svoFeatureQuery, type SvoQueryState } from "../svo/pipeline/persistence";
import { surfaceDisplayQuery, type SurfaceDisplayState } from "./surface-display/definition";
import { lensBloomQuery, type LensBloomState } from "./lens-bloom/definition";
export type FeatureUIQueryState = SvoQueryState & SurfaceDisplayState & LensBloomState;
export const uiFeatureQuery = combineQueryCodecs<FeatureUIQueryState>([svoFeatureQuery, surfaceDisplayQuery, lensBloomQuery]);
