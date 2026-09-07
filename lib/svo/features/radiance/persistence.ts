import { choiceQuery, queryRecord } from "../../../framework/persistence";
import { SVO_RADIANCE_RECONSTRUCTION_OPTIONS } from "./definition";
import { DEFAULT_SVO_RENDER_TUNING, type SvoRenderTuning } from "../../pipeline/svo-render-tuning";
export const radianceTuningQuery = queryRecord<Pick<SvoRenderTuning, "coneRadianceReconstruction">>({
  coneRadianceReconstruction: choiceQuery("svoReconstruction", DEFAULT_SVO_RENDER_TUNING.coneRadianceReconstruction,
    SVO_RADIANCE_RECONSTRUCTION_OPTIONS.map(option => option.value)),
});
