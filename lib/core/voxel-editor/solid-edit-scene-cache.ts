import type { SceneDescription } from "../model";
import { reuseSolidWorld, sceneWithSolidStroke, solidWorldContentStamp } from "../solid-world";
import { EDITOR_HISTORY_LIMIT } from "../stores/history-store";

/** Keep stroke endpoints, not every pointer sample. Their immutable page images
 * are shared; matching history documents can cross the worker boundary without
 * rebuilding terrain. The bound follows the authored history limit. */
export function createSolidEditSceneCache() {
  const endpoints = new Map<string, SceneDescription>();
  const remember = (scene: SceneDescription) => {
    const stamp = solidWorldContentStamp(scene);
    endpoints.delete(stamp);
    endpoints.set(stamp, scene);
    while (endpoints.size > EDITOR_HISTORY_LIMIT + 2) endpoints.delete(endpoints.keys().next().value!);
  };
  return {
    remember,
    prepare(target: SceneDescription, base: SceneDescription) {
      const remembered = endpoints.get(solidWorldContentStamp(target));
      if (remembered) { reuseSolidWorld(remembered, target); return; }
      // Undo and branching history are replacements, not append-only strokes.
      // A mismatched prefix must never be interpreted as a stroke suffix.
      if (base.solidVoxels.length > target.solidVoxels.length
        || base.solidVoxels.some((patch, index) => JSON.stringify(patch) !== JSON.stringify(target.solidVoxels[index]))) return;
      const prepared = sceneWithSolidStroke(base, target.solidVoxels.slice(base.solidVoxels.length));
      reuseSolidWorld(prepared, target);
    },
  };
}
