import { numberQuery, queryRecord } from "../../../framework/persistence";
import { DEFAULT_SVO_RENDER_TUNING, type SvoRenderTuning } from "../../pipeline/svo-render-tuning";
export const presentationTuningQuery=queryRecord<Pick<SvoRenderTuning,"resolutionScale">>({
  resolutionScale:numberQuery("svoResolution",DEFAULT_SVO_RENDER_TUNING.resolutionScale,.35,2),
});
