import type { EditorRay } from "../editor-entity";
import type { SceneDescription } from "../model";
import { sceneWithSolidStroke } from "../solid-world";
import { toolValues, type ToolUpdate, type ToolValues, type VoxelToolPlugin } from "./plugin";

export interface ToolHost {
  scene(): SceneDescription;
  /** Validate and apply to physics before publishing the accepted document. */
  publish(scene: SceneDescription): void;
  begin(label: string): void;
  finish(): void;
  cancel(): void;
}
/** Generic transaction lifecycle, independent of both UI and the chosen plugin. */
export function beginToolTransaction(plugin: VoxelToolPlugin, host: ToolHost,
  ray: EditorRay, values: ToolValues = {}) {
  const base = host.scene();
  const gesture = plugin.begin({ scene: base, ray, values: toolValues(plugin, values) });
  if (!gesture) return undefined;
  let closed = false;
  let accepted = base;
  let key = "";
  host.begin(plugin.ui.label);
  return {
    update(input: EditorRay): ToolUpdate | undefined {
      if (closed) return undefined;
      // External undo/load/scene changes end ownership of the document.
      if (host.scene() !== accepted) { closed = true; host.cancel(); return undefined; }
      const result = gesture.update(input);
      if (!result) return undefined;
      const nextKey = JSON.stringify(result.patches);
      if (nextKey !== key) {
        const next = sceneWithSolidStroke(base, result.patches);
        host.publish(next);
        accepted = next;
        key = nextKey;
      }
      return result;
    },
    finish(cancelled = false) {
      if (closed) return;
      closed = true;
      if (host.scene() !== accepted) { host.cancel(); return; }
      if (cancelled) {
        if (accepted !== base) host.publish(base);
        host.cancel();
      } else host.finish();
    },
  };
}
