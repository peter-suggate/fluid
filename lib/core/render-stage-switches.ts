/**
 * Encode-time ablation switches, one per frame-graph stage.
 *
 * A disabled stage is not *scaled to zero* — it is not encoded. That is the
 * whole point: a strength slider at 0 still pays for the pass that computes the
 * thing it is scaling, so it can never answer "what does this cost me". These
 * withhold the dispatch, so the frame total moves by exactly what the stage was
 * worth and the panel can report the difference.
 *
 * The ids are the render panel's node ids, so a lamp in the diagram and a gate
 * in the encoder are the same name. Only stages whose removal is *structural*
 * live here; four nodes are switched by contracts that already existed and that
 * the shaders compile against — cone visibility (`coneTracingMode`), GI
 * composition (`globalIlluminationEnabled`) and primary seam closure
 * (`silhouetteRefinementEnabled`) — and adding a second way to turn those off
 * would be two sources of truth for one bit.
 *
 * Every gate keeps the *clears* its consumers depend on and drops only the
 * draws and dispatches. A disabled stage therefore degrades the image in a
 * defined way (a missing primary reads as sky, missing lighting reads as black)
 * rather than presenting whatever happened to be left in the attachment.
 */

export type RenderStageSwitchId =
  | "sparse-world-build"
  | "fluid-coverage"
  | "primary-entry-prepass"
  | "primary-traversal"
  | "thin-glass"
  | "scene-primitive"
  | "rigid-impostor"
  | "voxel-light-cache"
  | "world-gi-cache"
  | "reduced-shade"
  | "sky-lighting"
  | "deferred-lighting"
  | "surface-extraction"
  | "water-interfaces"
  | "caustics"
  | "optical-composite"
  | "inspection-overlays"
  | "present";

export const RENDER_STAGE_SWITCH_IDS: readonly RenderStageSwitchId[] = Object.freeze([
  "sparse-world-build",
  "fluid-coverage",
  "primary-entry-prepass",
  "primary-traversal",
  "thin-glass",
  "scene-primitive",
  "rigid-impostor",
  "voxel-light-cache",
  "world-gi-cache",
  "reduced-shade",
  "sky-lighting",
  "deferred-lighting",
  "surface-extraction",
  "water-interfaces",
  "caustics",
  "optical-composite",
  "inspection-overlays",
  "present",
]);

const SWITCH_IDS = new Set<string>(RENDER_STAGE_SWITCH_IDS);

export const isRenderStageSwitchId = (id: string): id is RenderStageSwitchId => SWITCH_IDS.has(id);

/** The disabled set as the renderer consumes it. */
export type DisabledRenderStages = ReadonlySet<RenderStageSwitchId>;

export const NO_DISABLED_RENDER_STAGES: DisabledRenderStages = Object.freeze(new Set<RenderStageSwitchId>());

export function disabledRenderStagesFrom(ids: readonly string[] | undefined): DisabledRenderStages {
  if (!ids || ids.length === 0) return NO_DISABLED_RENDER_STAGES;
  return new Set(ids.filter(isRenderStageSwitchId));
}

/**
 * A stable key for the disabled set.
 *
 * Frame reuse and the presentation trace context are both keyed by "what this
 * frame was asked to draw", and an ablation changes that. Without the key in
 * both, instrumentation would mix two different pipelines into one mean.
 */
export function disabledRenderStagesKey(disabled: DisabledRenderStages): string {
  if (disabled.size === 0) return "";
  return RENDER_STAGE_SWITCH_IDS.filter((id) => disabled.has(id)).join("+");
}

export function disabledRenderStagesEqual(a: DisabledRenderStages, b: DisabledRenderStages): boolean {
  if (a === b) return true;
  if (a.size !== b.size) return false;
  for (const id of a) if (!b.has(id)) return false;
  return true;
}
