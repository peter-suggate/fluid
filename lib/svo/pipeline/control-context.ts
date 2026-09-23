import type { ReactNode } from "react";
import type { UIStoreHook } from "../../core/stores/ui-store";
import type { EffectiveRendererStatus } from "../../core/renderer-status";
import type { SvoRenderTuning } from "./svo-render-tuning";
import type { SvoLightingVisibilityStatus, SvoPrimaryTraversalMode, SvoSilhouetteRefinementStatus } from "./svo-render-options";
import type { RenderFrameStageId } from "../../core/render-frame-stages";
import type { RenderPipelinePhaseCost } from "./render-pipeline-graph";
import type { DisabledRenderStages } from "./render-stage-switches";

type UIState = ReturnType<UIStoreHook["getState"]>;
/** Host facts consumed by feature-owned drawers. Commands remain session-owned. */
export type SvoFeatureControlContext = Pick<UIState,
  | "svoMaximumTraversalDepth" | "setSvoMaximumTraversalDepth"
  | "svoMaximumNodeVisits" | "setSvoMaximumNodeVisits"
  | "svoConeTracingMode" | "setSvoConeTracingMode"
  | "svoShadowsEnabled" | "setSvoShadowsEnabled"
  | "svoAmbientOcclusionEnabled" | "setSvoAmbientOcclusionEnabled"
  | "svoStageView" | "setSvoStageView" | "svoStageLightSlot" | "setSvoStageLightSlot"
  | "svoGlobalIlluminationEnabled"
> & {
  readonly tuning: SvoRenderTuning;
  readonly updateTuning: <K extends keyof SvoRenderTuning>(key: K, value: SvoRenderTuning[K]) => void;
  readonly modified: <K extends keyof SvoRenderTuning>(key: K) => boolean;
  readonly resetTuning: <K extends keyof SvoRenderTuning>(key: K) => () => void;
  readonly renderRefinementDepth: number;
  readonly sceneIsDry: boolean;
  readonly leafVoxel_mm: number;
  readonly finestCellSize_m: number;
  readonly resolvedPrimary: SvoPrimaryTraversalMode;
  readonly partitioned: boolean;
  readonly disabledStages: DisabledRenderStages;
  readonly durations: ReadonlyMap<RenderFrameStageId, RenderPipelinePhaseCost>;
  readonly effectiveRendererStatus: EffectiveRendererStatus;
  readonly smoothSurfaceEnabled: boolean;
  readonly silhouetteRefinementStatus: SvoSilhouetteRefinementStatus;
  readonly lightingVisibilityStatus: SvoLightingVisibilityStatus;
};

/**
 * What a stage hands its card: the controls behind the fold, the count read on
 * the closed card, and any status that must show whether or not it is open.
 *
 * `settings` and `readouts` are counted by the stage that renders them — only
 * it knows which of its fields this scene and this arm actually draw.
 */
export interface SvoStageControls {
  readonly node?: ReactNode;
  readonly settings?: number;
  readonly readouts?: number;
  readonly notice?: ReactNode;
}
