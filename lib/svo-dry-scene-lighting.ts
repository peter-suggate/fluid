import type { SceneDescription } from "./model";

/**
 * The one lighting and display-grade contract for every SVO dry-scene pass.
 *
 * These are the values established on `hero-garden-hose`. They live at the
 * renderer boundary instead of in a scene factory: scenes may author geometry
 * (including emissive fixtures), but no dry-SVO presentation gets a private
 * key, environment balance, or grade. This is deliberately independent of the
 * fluid-solver flag: a wet full-scene frame still shades its environment with
 * the dry-SVO renderer.
 */
export const DEFAULT_SVO_DRY_SCENE_LIGHTING = Object.freeze({
  directional: Object.freeze({
    colorLinear: Object.freeze([1, 0.90, 0.72] as const),
    intensity: 2.4,
    direction: Object.freeze([-0.476019656956976, 2.4, -0.5795733656669904] as const),
  }),
  environment: Object.freeze({
    diffuseScale: 2.0,
    specularScale: 1.0,
    lowerRadianceLinear: Object.freeze([0.70, 0.54, 0.36] as const),
    upperRadianceLinear: Object.freeze([0.76, 0.59, 0.40] as const),
    accentRadianceLinear: Object.freeze([0.64, 0.48, 0.30] as const),
  }),
  grade: Object.freeze({
    toneCurve: "aces" as const,
    exposure: 0.1450,
    whiteBalance: Object.freeze([1.12, 1.0, 0.90] as const),
  }),
}) satisfies NonNullable<SceneDescription["lighting"]>;

/** SVO dry-scene rendering has no scene-specific lighting overrides. */
export function svoSceneLighting(
  _scene: Pick<SceneDescription, "systems" | "lighting">,
): NonNullable<SceneDescription["lighting"]> {
  return DEFAULT_SVO_DRY_SCENE_LIGHTING;
}
