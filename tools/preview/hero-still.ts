/**
 * The hero pond, as a still, and the base every object-class preview builds on.
 *
 *     FLUID_SVO_DRY_FRAME_SCENE_MODULE=tools/preview/hero-still.ts
 *
 * renders it through the production dry-scene path. See `tools/preview/README.md`.
 *
 * **Go through `heroPreviewScene`, not through `createHeroGardenHoseScene`.**
 * A scene factory returns a *body*: the catalog attaches the art-directed
 * environment on the way out (`sceneDocument` in `lib/scene-definition.ts`), so
 * a preview that calls the factory directly renders the porcelain garden under
 * the default set's dark teal sky and every judgement made from it about colour,
 * value or contrast is a judgement about the wrong lighting. That cost one
 * confusing render before it was noticed.
 *
 * The clone matters too: `scene.scenery` is a module-level constant shared by
 * every document the factory produces, so pushing preview nodes onto it would
 * edit the hero scene itself for the rest of the process.
 */
import type { CameraState, SceneDescription } from "../../lib/model";
import { getScenePreset } from "../../lib/scenes";
import type { SceneryNode } from "../../lib/scenery-graph";

const HERO_PRESET_ID = "hero-garden-hose";

/** The hero document as the product builds it, plus whatever you are working on. */
export function heroPreviewScene(extraNodes: readonly SceneryNode[] = []): SceneDescription {
  const scene = getScenePreset(HERO_PRESET_ID).create();
  if (extraNodes.length === 0) return scene;
  const scenery = scene.scenery;
  if (!scenery) throw new Error("The hero preset must carry a scenery graph to extend");
  return { ...scene, scenery: { ...scenery, nodes: [...scenery.nodes, ...extraNodes] } };
}

/** The camera the hero scene opens on. Override it in your own preview module. */
export function heroPreviewCamera(): Partial<CameraState> {
  return getScenePreset(HERO_PRESET_ID).camera ?? {};
}

export const createScene = () => heroPreviewScene();
export const camera = heroPreviewCamera();
