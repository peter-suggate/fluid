import type { EditorRay } from "../editor-entity";
import type { SceneDescription } from "../model";
import { sceneWithSolidStroke } from "../solid-world";
import { toolValues, type ToolAction, type ToolUpdate, type ToolValues, type VoxelToolPlugin } from "./plugin";

export interface ToolHost {
  scene(): SceneDescription;
  /** Preflight capacity, then publish the accepted document for the next fluid step. */
  publish(scene: SceneDescription, base: SceneDescription): Promise<void>;
  /** Execute a transient plugin action without recording a document edit. */
  execute?(action: ToolAction): Promise<void>;
  begin(label: string): void;
  finish(): void;
  cancel(): void;
}
/**
 * Generic transaction lifecycle, independent of both UI and the chosen plugin.
 *
 * A stroke is a proposal until release. Samples only recompute the plugin's
 * preview; the scene document — which is the runtime's publication identity —
 * is written exactly once, when the pointer lets go. Cancelling therefore has
 * nothing to roll back, and a drag costs no worker round trips.
 */
export function beginToolTransaction(plugin: VoxelToolPlugin, host: ToolHost,
  ray: EditorRay, values: ToolValues = {}, invert = false) {
  const base = host.scene();
  const gesture = plugin.begin({ scene: base, ray, values: toolValues(plugin, values, base), invert });
  if (!gesture) return undefined;
  let closed = false;
  let finishing = false;
  let cancellationRequested = false;
  let preview: ToolUpdate | undefined;
  // Release/cancel can arrive while a sample is queued. Serialize lifecycle
  // work so release always commits the last accepted sample.
  let pending: Promise<unknown> = Promise.resolve();
  let completion: Promise<void> | undefined;
  const ownsDocument = () => host.scene() === base;
  const abandon = () => { if (!closed) { closed = true; host.cancel(); } };
  host.begin(plugin.ui.label);
  return {
    update(input: EditorRay): Promise<ToolUpdate | undefined> {
      if (closed || finishing) return Promise.resolve(undefined);
      const operation = pending.then(() => {
        if (closed) return undefined;
        if (!ownsDocument()) { abandon(); return undefined; }
        // A transient action must never execute a shape the pointer has left;
        // an authored stroke keeps the geometry it had already been shown.
        if (plugin.execution === "release") preview = undefined;
        const result = gesture.update(input);
        if (result) preview = result;
        return result;
      });
      pending = operation.catch(() => {});
      return operation;
    },
    /** The pointer was released: does the gesture continue under a bare pointer? */
    advance(): Promise<boolean> {
      if (closed || finishing) return Promise.resolve(false);
      const operation = pending.then(() => !closed && !finishing && gesture.advance?.() === true);
      pending = operation.catch(() => {});
      return operation;
    },
    finish(cancelled = false): Promise<void> {
      cancellationRequested ||= cancelled;
      if (completion) return completion;
      finishing = true;
      completion = pending.then(async () => {
        if (closed) return;
        if (!ownsDocument()) { abandon(); return; }
        closed = true;
        if (plugin.execution === "release") {
          try {
            if (!cancellationRequested && preview?.action) {
              if (!host.execute) throw new Error("This runtime cannot execute this editing action.");
              await host.execute(preview.action);
            }
          } finally {
            // Moving water is not an authored-scene undo entry, even after success.
            host.cancel();
          }
          return;
        }
        if (cancellationRequested || !preview || preview.patches.length === 0) { host.cancel(); return; }
        const next = sceneWithSolidStroke(base, preview.patches);
        try { await host.publish(next, base); }
        catch (error) { host.cancel(); throw error; }
        // A host must reject stale preflights rather than overwrite another
        // editor's document. Never record history against such a replacement.
        if (host.scene() === next) host.finish(); else host.cancel();
      });
      return completion;
    },
  };
}
