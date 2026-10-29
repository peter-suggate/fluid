import { presentationTuningQuery } from "../features/presentation/persistence";
import { combineQueryCodecs, type QueryCodec } from "../../framework/persistence";
import { lightingQuery, lightingTuningQuery, type LightingQueryState } from "../features/lighting-visibility/persistence";
import { primaryQuery, primaryTuningQuery, type PrimaryQueryState } from "../features/primary-visibility/persistence";
import { diagnosticQuery, type DiagnosticQueryState } from "../features/diagnostics/persistence";
import { constructionTuningQuery } from "../features/construction/persistence";
import { radianceQuery, radianceTuningQuery, type RadianceQueryState } from "../features/radiance/persistence";
import { DEFAULT_SVO_RENDER_TUNING, normalizeSvoRenderTuning, type SvoRenderTuning } from "./svo-render-tuning";
const tuning = combineQueryCodecs<Partial<SvoRenderTuning>>([constructionTuningQuery, radianceTuningQuery, primaryTuningQuery, lightingTuningQuery, presentationTuningQuery]);
const tuningQuery: QueryCodec<{ svoRenderTuning: SvoRenderTuning }> = {
  keys: tuning.keys,
  read: query => ({ svoRenderTuning: normalizeSvoRenderTuning({ ...DEFAULT_SVO_RENDER_TUNING, ...tuning.read(query) }) }),
  write: (query, state) => tuning.write(query, state.svoRenderTuning),
};
export type SvoQueryState = LightingQueryState & RadianceQueryState & PrimaryQueryState & DiagnosticQueryState & { svoRenderTuning: SvoRenderTuning };
export const svoFeatureQuery = combineQueryCodecs<SvoQueryState>([lightingQuery, radianceQuery, primaryQuery, diagnosticQuery, tuningQuery]);
