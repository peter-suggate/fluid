"use client";
import { useEffect, useRef, type PointerEvent } from "react";
import type { EditorRay } from "../lib/core/editor-entity";
import type { ToolUpdate } from "../lib/core/voxel-editor/plugin";
import { beginToolTransaction } from "../lib/core/voxel-editor/transaction";
import { voxelTools } from "../lib/core/voxel-editor/registry";
import { simulation } from "../lib/core/simulation/controller";
import { useSession } from "../lib/core/session/session-context";
import type { SceneDescription } from "../lib/core/model";

type Transaction = NonNullable<ReturnType<typeof beginToolTransaction>>;
export function useVoxelToolGesture(ray: (event: PointerEvent<HTMLCanvasElement>) => EditorRay,
  validate: (scene: SceneDescription) => Promise<void>, preview: (update: ToolUpdate | null) => void) {
  const session = useSession();
  const active = useRef<{ id: number; transaction: Transaction; queued?: EditorRay;
    busy: boolean; ending?: boolean; cancelled: boolean; frame?: number } | undefined>(undefined);
  const notice = (error: unknown) => session.runtime.getState().setNotice(error instanceof Error ? error.message : String(error), "warn");
  const pump = async () => {
    const stroke = active.current;
    if (!stroke || stroke.busy) return;
    stroke.busy = true;
    try {
      const sample = stroke.queued;
      stroke.queued = undefined;
      if (sample && !stroke.cancelled) {
        const result = await stroke.transaction.update(sample);
        if (result) preview(result);
      }
      if (stroke.ending && !stroke.queued) {
        await stroke.transaction.finish(stroke.cancelled);
        active.current = undefined;
        preview(null);
      }
    } catch (error) {
      notice(error);
      // Retain the last accepted geometry and history if a sample was rejected.
      if (stroke.ending) { await stroke.transaction.finish(); active.current = undefined; preview(null); }
    } finally {
      stroke.busy = false;
      if (active.current && (stroke.queued || stroke.ending)) schedule();
    }
  };
  const schedule = () => {
    const stroke = active.current;
    if (!stroke || stroke.frame !== undefined) return;
    stroke.frame = requestAnimationFrame(() => { stroke.frame = undefined; void pump(); });
  };
  useEffect(() => {
    const cancel = (event: KeyboardEvent) => {
      if (event.key !== "Escape" || !active.current) return;
      event.preventDefault();
      active.current.cancelled = true; active.current.ending = true; active.current.queued = undefined;
      schedule();
    };
    window.addEventListener("keydown", cancel);
    return () => {
      window.removeEventListener("keydown", cancel);
      const stroke = active.current;
      if (stroke) { stroke.ending = true; schedule(); }
    };
  }, [session]); // Callbacks read the session and active transaction, never tool IDs.
  return {
    down(event: PointerEvent<HTMLCanvasElement>): boolean {
      if (active.current) return true;
      const ui = session.ui.getState();
      const plugin = ui.viewportMode === "interact" ? voxelTools.get(ui.voxelToolId) : undefined;
      if (!plugin || event.button !== 0 || event.shiftKey) return false;
      const initial = ray(event);
      try {
        const transaction = beginToolTransaction(plugin, {
          scene: () => session.scene.getState().scene,
          publish: async (next) => {
            const before = session.scene.getState().scene;
            if (next.systems?.fluid !== false) await validate(next);
            if (session.scene.getState().scene !== before) throw new Error("Scene changed during the stroke.");
            session.scene.getState().setScene(next);
          },
          begin: (label) => simulation.beginEdit(label, session.id),
          finish: () => { simulation.commitEdit(undefined, { reseed: false }, session.id); },
          cancel: () => simulation.cancelEdit(session.id),
        }, initial, ui.voxelToolValues[plugin.id]);
        if (transaction) {
          event.currentTarget.setPointerCapture(event.pointerId);
          active.current = { id: event.pointerId, transaction, queued: initial, busy: false, cancelled: false };
          schedule();
        }
      } catch (error) { notice(error); }
      return true;
    },
    move(event: PointerEvent<HTMLCanvasElement>): boolean {
      const stroke = active.current;
      if (stroke) {
        if (stroke.id === event.pointerId && !stroke.ending) { stroke.queued = ray(event); schedule(); }
        return true;
      }
      const ui = session.ui.getState();
      const plugin = ui.viewportMode === "interact" ? voxelTools.get(ui.voxelToolId) : undefined;
      if (!plugin) return false;
      try {
        // Hover uses the exact plugin targeting and geometry without publishing.
        const sample = ray(event);
        const gesture = plugin.begin({ scene: session.scene.getState().scene, ray: sample,
          values: Object.fromEntries(plugin.ui.controls.map((c) => [c.id, ui.voxelToolValues[plugin.id]?.[c.id] ?? c.initial])) });
        preview(gesture?.update(sample) ?? null);
      } catch { preview(null); }
      return true;
    },
    up(event: PointerEvent<HTMLCanvasElement>): boolean {
      const stroke = active.current;
      if (!stroke || stroke.id !== event.pointerId) return false;
      stroke.cancelled = event.type === "pointercancel";
      stroke.queued = stroke.cancelled ? undefined : ray(event);
      stroke.ending = true; schedule(); return true;
    },
  };
}
