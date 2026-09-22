import { choiceQuery, numberQuery, queryRecord } from "../../../framework/persistence";
import { DEFAULT_SVO_RENDER_TUNING, type SvoOccluderGhosting, type SvoRenderTuning } from "../../pipeline/svo-render-tuning";

export const presentationTuningQuery = queryRecord<Pick<SvoRenderTuning, "occluderGhosting" | "occluderGhostOpacity">>({
  occluderGhosting: choiceQuery<SvoOccluderGhosting>("svoGhost", DEFAULT_SVO_RENDER_TUNING.occluderGhosting, ["auto", "on", "off"]),
  occluderGhostOpacity: numberQuery("svoGhostOpacity", DEFAULT_SVO_RENDER_TUNING.occluderGhostOpacity, 0, 0.9),
});
