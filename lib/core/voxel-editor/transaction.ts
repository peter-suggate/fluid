import type { EditorRay } from "../editor-entity";
import type { SceneDescription } from "../model";
import { sceneWithSolidStroke, solidVoxelPatchesKey, type SolidWorldVoxelPatch } from "../solid-world";
import { toolValues, type ToolAction, type ToolGesture, type ToolUpdate, type ToolValues, type VoxelToolPlugin } from "./plugin";
import type { VoxelClipboard } from "./verb";
import type { VoxelRegion } from "./voxel-access";

export interface ToolHost {
  /** The document the next edit is authored against: the standing one with every edit still being accepted applied. */
  scene(): SceneDescription;
  /**
   * Preflight capacity, then stand `scene` in place of `base` for the next
   * fluid step, in the order edits were proposed. Resolves once it stands;
   * rejects when it was refused, or cancelled behind an edit that was.
   */
  publish(scene: SceneDescription, base: SceneDescription): Promise<void>;
  /**
   * Show these patches over `base` through the display alone, or with none put
   * the display back on `base`. Not an edit: the latest call wins, nothing is
   * recorded and the simulation never sees them.
   */
  preview?(patches: readonly SolidWorldVoxelPatch[] | undefined, base: SceneDescription): void;
  /** Execute a transient plugin action without recording a document edit. */
  execute?(action: ToolAction): Promise<void>;
  /**
   * Make these cells the standing voxel selection, or add them to it, or take
   * them out of it. Not a document edit: nothing is recorded.
   */
  select?(region: VoxelRegion, mode?: "replace" | "add" | "subtract"): void;
  /** The voxels last copied off a selection, for the tool that stamps them. */
  clipboard?(): VoxelClipboard | undefined;
  begin(label: string): void;
  finish(): void;
  cancel(): void;
}
/**
 * Commit one complete proposal over its immutable base — the only way solid
 * patches reach the document, whoever proposed them: a stroke on release, or a
 * selection verb the moment it is chosen.
 *
 * The host has already been told the edit began. This appends the patches,
 * hands the result to the host's preflight, and closes the edit: `finish` when
 * the host stood the accepted document, `cancel` for anything else. True when
 * the document changed.
 */
export async function publishSolidPatches(host: ToolHost, base: SceneDescription,
  patches: readonly SolidWorldVoxelPatch[]): Promise<boolean> {
  if (patches.length === 0) { host.cancel(); return false; }
  // A host rejects a stale preflight rather than overwrite another editor's
  // document, so a publish that resolves is the document standing.
  try { await host.publish(sceneWithSolidStroke(base, patches), base); }
  catch (error) { host.cancel(); throw error; }
  host.finish();
  return true;
}
/**
 * Generic transaction lifecycle, independent of both UI and the chosen plugin.
 *
 * A stroke is a proposal until release. Samples recompute the plugin's
 * preview and hand its patches to the host's display; the scene document —
 * which is the runtime's publication identity — is written exactly once, when
 * the pointer lets go. Cancelling has no document to roll back: it takes the
 * preview down.
 *
 * `begun` is the gesture a caller already opened to ask where the press landed,
 * over this host's scene and with these values; the plugin is not asked twice.
 */
export function beginToolTransaction(plugin: VoxelToolPlugin, host: ToolHost,
  ray: EditorRay, values: ToolValues = {}, invert = false, begun?: ToolGesture) {
  const base = host.scene();
  const gesture = begun ?? plugin.begin({ scene: base, ray, values: toolValues(plugin, values, base), invert,
    clipboard: host.clipboard?.() });
  if (!gesture) return undefined;
  let closed = false;
  let finishing = false;
  let cancellationRequested = false;
  let preview: ToolUpdate | undefined;
  /** The patches the display was last handed, by key; undefined when it shows the base. */
  let displayed: string | undefined;
  // Release/cancel can arrive while a sample is queued. Serialize lifecycle
  // work so release always commits the last accepted sample.
  let pending: Promise<unknown> = Promise.resolve();
  let completion: Promise<void> | undefined;
  const ownsDocument = () => host.scene() === base;
  /** Hand the display this proposal, or none; a sample that proposes what it already shows costs nothing. */
  const display = (patches: readonly SolidWorldVoxelPatch[] | undefined) => {
    if (!host.preview) return;
    const key = patches?.length ? solidVoxelPatchesKey(patches) : undefined;
    if (key === displayed) return;
    displayed = key;
    host.preview(key === undefined ? undefined : patches, base);
  };
  const abandon = () => { if (!closed) { closed = true; display(undefined); host.cancel(); } };
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
        if (result) {
          preview = result;
          // A transient action authors no solids; its shape is the outline's alone.
          if (plugin.execution !== "release") display(result.patches);
        }
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
        if (cancellationRequested || !preview) { display(undefined); host.cancel(); return; }
        const selection = preview.selection;
        if (selection && !host.select) { display(undefined); host.cancel(); throw new Error("This runtime cannot hold a voxel selection."); }
        let published = false;
        // The accepted edit is the preview the display already holds; anything else takes it down.
        try { published = await publishSolidPatches(host, base, preview.patches); }
        finally { if (published) displayed = undefined; else display(undefined); }
        // A gesture that both edits and selects selects what it edited: a
        // rejected or superseded edit leaves no selection standing over it.
        if (selection && (published || preview.patches.length === 0)) host.select!(selection, preview.selectionMode);
      });
      return completion;
    },
  };
}
