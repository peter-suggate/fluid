import type { SceneDescription } from "../../../core/model";
import type { EnvironmentId } from "../../../core/environments";

/**
 * The default lighting and display-grade contract for every SVO dry-scene pass.
 *
 * These are the values established on `hero-garden-hose`. They live at the
 * renderer boundary rather than in a scene factory so that the ordinary case —
 * a set standing in a room — is lit once, consistently, and no scene has to
 * restate a rig to get the house look. This is deliberately independent of the
 * fluid-solver flag: a wet full-scene frame still shades its environment with
 * the dry-SVO renderer.
 *
 * A document that authors `scene.lighting` overrides these field by field; see
 * `svoSceneLighting`.
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

type SceneLighting = NonNullable<SceneDescription["lighting"]>;

/**
 * The porcelain garden's own rig: white clay under a low, warm sun and a pale,
 * slightly cool sky, graded with a mild warm cast so the set reads as white clay.
 *
 * The shared rig above cannot be that. Its hemisphere is amber at twice unit
 * scale, its key is amber, and its white balance warms a third time, so red
 * reaches the display at about 2.4x blue; at 0.145 exposure that is a dim
 * orange-white, which is to say brown. It also buries the key: ambient at 2.0
 * outweighs a 72-degree sun on the floor, so nothing models the form.
 *
 * The reference plate (`output/imagegen/garden-pond-hose-fill-simplified.png`)
 * is the opposite balance. Only the sun is warm; the shade is a neutral-to-cool
 * grey lifted by bounce off a white floor; lit porcelain sits near sRGB 180-190 and
 * its shade distinctly darker. So:
 *
 * - The key is 40 degrees up, from behind and to the camera's left: its
 *   horizontal is half the view's forward minus 0.866 of its right at
 *   `HERO_GARDEN_AZIMUTH_RAD` 5.4, which rakes the coping and throws shadows
 *   toward the viewer the way the plate's do.
 * - The hemisphere is pale sky above and warm floor bounce below. A first cut
 *   at unit scale under a 2.2 key rendered flat (scene-linear peak 0.70 against
 *   a 0.41 mean: the key barely separated lit from shade), so the key is 3.3
 *   over a 0.8 hemisphere.
 * - The grade is measured against the plate, not assumed: its floor is a warm
 *   light grey, sRGB about (178, 168, 159), not paper white. Graded offline
 *   from a scene-linear dump of the first cut, exposure 0.9 with a 1.10/0.90
 *   balance landed within a few levels of that; 0.95 with a gentler 1.06/0.94
 *   allows for the stronger key while keeping the cast mild.
 */
export const PORCELAIN_GARDEN_DRY_SCENE_LIGHTING = Object.freeze({
  directional: Object.freeze({
    colorLinear: Object.freeze([1, 0.93, 0.82] as const),
    intensity: 3.3,
    direction: Object.freeze([-0.12507, 0.64279, -0.75575] as const),
  }),
  environment: Object.freeze({
    diffuseScale: 0.8,
    specularScale: 1.0,
    lowerRadianceLinear: Object.freeze([0.78, 0.76, 0.72] as const),
    upperRadianceLinear: Object.freeze([0.62, 0.68, 0.78] as const),
    accentRadianceLinear: Object.freeze([0.70, 0.70, 0.70] as const),
  }),
  grade: Object.freeze({
    toneCurve: "aces" as const,
    exposure: 0.95,
    whiteBalance: Object.freeze([1.06, 1.0, 0.94] as const),
  }),
}) satisfies NonNullable<SceneDescription["lighting"]>;

/**
 * The rig the stage is lit by, and the reason a base rig is a per-environment
 * choice rather than one constant.
 *
 * The `stage` environment is a dark stage under one overhead practical (see
 * `lib/core/studio-stage-scene.ts`), and a spot is only a spot against
 * something darker. Under the garden's fixed 2.0-scale warm hemisphere its lamp
 * is a faint warm patch on an already-lit floor and the pool of light has
 * nothing to be brighter than, so this set authors its own environment balance
 * and exposure.
 *
 * The directional is the term that had to come down furthest, and it is the one
 * the rig is least free with, because it is doing two jobs: it lights the dry
 * set *and* it is the water's only key — `resolveWaterKeyLight` reads
 * `lighting.directional` and the raster water pipeline never sees the light
 * table, so the spot does not exist as far as the surface is concerned.
 * Measured on the dry lane, the stage away from the pool tracks it almost
 * exactly: at 1.15 the floor sits at sRGB 114 and the pool is barely five times
 * brighter than the room it is meant to be carved out of; at 0.25 the floor is
 * 41 and the pool reads as light. So the key is a fill, aimed close enough to
 * the lamp's own axis that the highlight it puts on the water agrees with the
 * fixture the pool comes from — a key from where the visible lamp is not is the
 * one thing in this set that would read as a mistake.
 *
 * Exposure is a camera decision and this set is far darker than the one the
 * shared rig was solved on, so it takes its own. Half of what a spotlit stage
 * needs is simply not being pushed up the shoulder: at 0.9 both the pool and
 * the floor around it sat above the knee and the frame was one flat grey, and
 * the fix was a stop and a half of exposure rather than any change to the
 * lights. 0.55 is that solve opened back up by a third of a stop — the first
 * cut at 0.45 kept the pool honest but read as underexposed on the tanks.
 */
export const STUDIO_STAGE_DRY_SCENE_LIGHTING = Object.freeze({
  directional: Object.freeze({
    direction: Object.freeze([-0.22, 0.94, 0.26] as const),
    colorLinear: Object.freeze([1, 0.96, 0.9] as const),
    intensity: 0.25,
  }),
  environment: Object.freeze({
    // A cold, very low fill. Not zero: the water is a dielectric and needs
    // something to reflect at grazing angles, and a shadow with no bounce in it
    // at all reads as a hole rather than as shadow.
    diffuseScale: 1,
    specularScale: 0.5,
    lowerRadianceLinear: Object.freeze([0.008, 0.010, 0.014] as const),
    upperRadianceLinear: Object.freeze([0.016, 0.020, 0.028] as const),
    accentRadianceLinear: Object.freeze([0.012, 0.015, 0.022] as const),
  }),
  grade: Object.freeze({
    toneCurve: "aces" as const,
    exposure: 0.55,
    whiteBalance: Object.freeze([1.04, 1, 0.94] as const),
  }),
}) satisfies NonNullable<SceneDescription["lighting"]>;

/**
 * The rig an environment is lit by before its document says anything.
 *
 * Two environments name their own rig: the stage, because its subject *is* its
 * lighting, and the porcelain garden, because a white set under the shared
 * amber balance renders brown. Every other environment — the white room
 * included, and a scene that names no environment at all — is a set standing
 * in the shared daylight balance, which is what
 * `DEFAULT_SVO_DRY_SCENE_LIGHTING` is. Both exceptions are opt-ins by
 * environment rather than the default: a scene that never asked for a dark
 * stage should not open on one.
 */
function baseLightingForEnvironment(environmentId: EnvironmentId | undefined): SceneLighting {
  if (environmentId === "stage") return STUDIO_STAGE_DRY_SCENE_LIGHTING;
  if (environmentId === "garden") return PORCELAIN_GARDEN_DRY_SCENE_LIGHTING;
  return DEFAULT_SVO_DRY_SCENE_LIGHTING;
}

/**
 * The rig a dry-SVO presentation shades with: the environment's base, with
 * whatever the document authored laid over it, field by field.
 *
 * The base is still the answer for every scene that says nothing, and that
 * remains most of the catalog — a set is lit by the room it is in, not by its
 * own opinion, and a scene that raised its key to fight a tone curve would be
 * authoring a camera rather than a light.
 *
 * The base is keyed on the environment rather than fixed because a rig belongs
 * to a set: the values above are the porcelain garden at midday, and no amount
 * of per-scene override makes them the right starting point for a dark stage.
 * Merging stays per-field so a scene can take the exposure it needs without
 * also having to restate a key direction it has no opinion about.
 */
export function svoSceneLighting(
  scene: Pick<SceneDescription, "systems" | "lighting" | "environment">,
): SceneLighting {
  const base = baseLightingForEnvironment(scene.environment);
  const authored = scene.lighting;
  if (!authored) return base;
  return {
    directional: { ...base.directional, ...authored.directional },
    environment: { ...base.environment, ...authored.environment },
    grade: { ...base.grade, ...authored.grade },
  };
}
