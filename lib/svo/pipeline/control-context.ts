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
  | "svoStageView" | "svoStageLightSlot" | "setSvoStageLightSlot"
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
