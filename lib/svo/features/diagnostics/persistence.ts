import { choiceQuery, queryRecord } from "../../../framework/persistence";
import { DEFAULT_SVO_RENDER_DIAGNOSTICS, SVO_RENDER_STAGE_VIEWS, type SvoRenderStageView } from "./svo-render-diagnostics";
export interface DiagnosticQueryState { svoStageView: SvoRenderStageView }
export const diagnosticQuery = queryRecord<DiagnosticQueryState>({
  svoStageView: choiceQuery("svoStage", DEFAULT_SVO_RENDER_DIAGNOSTICS.stageView, SVO_RENDER_STAGE_VIEWS),
});
