import type { EditorRay } from "../editor-entity";
import type { SceneDescription } from "../model";
import { sceneWithSolidStroke } from "../solid-world";
import { toolValues, type ToolUpdate, type ToolValues, type VoxelToolPlugin } from "./plugin";

export interface ToolHost {
  scene(): SceneDescription;
  /** Validate and apply to physics before publishing the accepted document. */
  publish(scene: SceneDescription, base: SceneDescription): Promise<void>;
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
  let finishing = false;
  let accepted = base;
  let key = "";
  // Release/cancel can arrive while an asynchronous preflight is outstanding.
  // Serialize lifecycle work so rollback always sees the last accepted sample.
  let pending: Promise<unknown> = Promise.resolve();
  let completion: Promise<void> | undefined;
  const ownsDocument = () => host.scene() === accepted;
  const abandon = () => { if (!closed) { closed = true; host.cancel(); } };
  host.begin(plugin.ui.label);
  return {
    update(input: EditorRay): Promise<ToolUpdate | undefined> {
      if (closed || finishing) return Promise.resolve(undefined);
      const operation = pending.then(async () => {
        if (closed) return undefined;
        if (!ownsDocument()) { abandon(); return undefined; }
        const result = gesture.update(input);
        if (!result) return undefined;
        const nextKey = JSON.stringify(result.patches);
        if (nextKey !== key) {
          const next = sceneWithSolidStroke(base, result.patches);
          await host.publish(next, base);
          // A host must reject stale preflights rather than overwrite another
          // editor's document. Never retain history ownership after such a change.
          if (host.scene() !== next) { abandon(); return undefined; }
          accepted = next;
          key = nextKey;
        }
        return result;
      });
      pending = operation.catch(() => {});
      return operation;
    },
    finish(cancelled = false): Promise<void> {
      if (completion) return completion;
      finishing = true;
      completion = pending.then(async () => {
        if (closed) return;
        if (!ownsDocument()) { abandon(); return; }
        closed = true;
        if (cancelled) {
          try {
            if (accepted !== base) await host.publish(base, base);
            host.cancel();
          } catch (error) {
            // A failed rollback must still leave the accepted stroke undoable,
            // but must never record history against an external replacement.
            if (ownsDocument()) host.finish(); else host.cancel();
            throw error;
          }
        } else host.finish();
      });
      return completion;
    },
  };
}
