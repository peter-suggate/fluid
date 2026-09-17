import { findSceneDefinition, SCENE_CATALOG } from "../lib/core/scenes";
import { sceneDocument } from "../lib/core/scene-definition";
import { queryRecord, type QueryValue } from "../lib/framework/persistence";
import type { AdvanceAuthoredScene } from "../lib/physics-wasm/advance-controller";

/**
 * Which production scene the lab is reading, and the key that says so.
 *
 * The roster is the studio's catalog unchanged — the whole point of the lab is
 * that it seeds its 2-D twin from the *same* authored document the 3-D studio
 * runs, so there is no second list of scenes to keep in step. What is the lab's
 * own is only which one it opens on and how that travels.
 *
 * The key lives here rather than in `AdvanceLab.tsx` because the page is not
 * what owns it: a scene is a thing the lab can be *showing*, the roster and the
 * default are stated here, and the address bar is one more reader of them. The
 * page had a `SCENE_PARAM` constant and a hand-rolled `replaceState` beside it,
 * which is the same key written twice and a second mirror racing the studio's.
 */

/** Every scene the lab can seed, which is every scene the studio can run. */
export const LAB_SCENE_IDS: ReadonlySet<string> =
  new Set(SCENE_CATALOG.map((scene) => scene.id));

/**
 * The scene a bare `/advance-lab` opens.
 *
 * The canonical B8 sparse layout: one genuinely featureless wet corner page, a
 * bare floor, and a collapse to watch. A reader arriving with no link should
 * land on the picture every note about this lab was written against.
 */
export const DEFAULT_LAB_SCENE_ID = "water-box-dam-break";

/**
 * The scene as one query value.
 *
 * An unknown id falls back to the default rather than failing: a link is
 * external input, and a scene that has since been renamed or cut should open the
 * lab rather than a blank page. The default is written as the *absence* of the
 * key, which `queryRecord` does for every value whose encoding matches its
 * initial, so an ordinary visit has no `?scene=` at all.
 */
export const labSceneValue: QueryValue<string> = {
  key: "scene",
  initial: DEFAULT_LAB_SCENE_ID,
  read: (raw) => raw !== null && LAB_SCENE_IDS.has(raw) ? raw : DEFAULT_LAB_SCENE_ID,
  write: (value) => value,
};

export interface LabSceneQueryState { readonly sceneId: string }

export const labSceneQuery = queryRecord<LabSceneQueryState>({ sceneId: labSceneValue });

/** The authored document for a scene id, or nothing when the id names none. */
export function labAuthoredScene(id: string): AdvanceAuthoredScene | null {
  const definition = findSceneDefinition(id);
  return definition ? Object.freeze({
    id, label: definition.name, document: sceneDocument(definition),
  }) : null;
}
