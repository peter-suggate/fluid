import type { SceneDescription } from "../model";
import { solidPatchListExtends } from "../solid-patch-list";
import { reuseSolidWorld, sceneHoldsSolidWorld, sceneWithSolidStroke, solidWorldContentStamp,
  type SolidWorldVoxelPatch } from "../solid-world";
import { EDITOR_HISTORY_LIMIT } from "../stores/history-store";

/**
 * Append a stroke that crossed the worker boundary alone to the document the
 * worker holds. The result shares every other field with that document, and
 * its solid image is the base's with the stroke's pages rewritten: it arrives
 * at `prepare` already holding it, and is handed on as it is.
 */
export function extendSolidEditScene(base: SceneDescription,
  patches: readonly SolidWorldVoxelPatch[]): SceneDescription {
  return sceneWithSolidStroke(base, patches);
}

/**
 * The document `base` was before its last strokes: its first `patchCount`
 * patches, which is what an undo names instead of sending. Every other field
 * is the base's own.
 *
 * `remembered` is the stroke endpoint held under the stamp the undo named, when
 * there is one. Its patch list is the same patches and its image is theirs, so
 * the result holds that image and composes nothing; without it the image is
 * composed when it is first asked for.
 */
export function truncatedSolidEditScene(base: SceneDescription, patchCount: number,
  remembered: SceneDescription | undefined): SceneDescription {
  const held = remembered?.solidVoxels.length === patchCount ? remembered : undefined;
  const scene = { ...base, solidVoxels: held ? held.solidVoxels : base.solidVoxels.slice(0, patchCount) };
  if (held) reuseSolidWorld(held, scene);
  return scene;
}

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
    /** The endpoint held under this content stamp, if it has not aged out. */
    held: (stamp: string) => endpoints.get(stamp),
    prepare(target: SceneDescription, base: SceneDescription) {
      // Composed in this realm over a base that held its image: nothing to find or rebuild.
      if (sceneHoldsSolidWorld(target)) return;
      const remembered = endpoints.get(solidWorldContentStamp(target));
      if (remembered) { reuseSolidWorld(remembered, target); return; }
      // Undo and branching history are replacements, not append-only strokes.
      // A mismatched prefix must never be interpreted as a stroke suffix.
      if (!solidPatchListExtends(base.solidVoxels, target.solidVoxels)) return;
      const prepared = sceneWithSolidStroke(base, target.solidVoxels.slice(base.solidVoxels.length));
      reuseSolidWorld(prepared, target);
    },
  };
}
