"use client";
import { useEffect, useEffectEvent, useRef, type PointerEvent, type WheelEvent } from "react";
import type { EditorRay } from "../lib/core/editor-entity";
import { toolValues, type ToolAction, type ToolUpdate } from "../lib/core/voxel-editor/plugin";
import { beginToolTransaction } from "../lib/core/voxel-editor/transaction";
import { voxelTools } from "../lib/core/voxel-editor/registry";
import { useSession } from "../lib/core/session/session-context";
import type { SceneDescription } from "../lib/core/model";

type Transaction = NonNullable<ReturnType<typeof beginToolTransaction>>;
export function useVoxelToolGesture(ray: (event: PointerEvent<HTMLCanvasElement>) => EditorRay,
  validate: (scene: SceneDescription, base: SceneDescription) => Promise<void>, preview: (update: ToolUpdate | null) => void,
  execute?: (action: ToolAction) => Promise<void>) {
  const session = useSession();
  const armed = session.ui((state) => state.voxelToolId);
  const viewportMode = session.ui((state) => state.viewportMode);
  const gpuState = session.diagnostics((state) => state.gpuStatus.state);
  const active = useRef<{ id: number; transaction: Transaction; queued?: EditorRay;
    busy: boolean; releaseOnly: boolean; ending?: boolean; cancelled: boolean; frame?: number; element: HTMLCanvasElement;
    /** The pointer came up and the gesture has not yet said whether it continues. */
    releasing?: boolean;
    /** A later phase follows the bare pointer; the next press commits it. */
    hovering?: boolean } | undefined>(undefined);
  const mounted = useRef(true);
  const showPreview = (update: ToolUpdate | null) => { if (mounted.current) preview(update); };
  const release = (stroke: NonNullable<typeof active.current>) => {
    if (active.current === stroke) active.current = undefined;
    if (stroke.element.hasPointerCapture(stroke.id)) stroke.element.releasePointerCapture(stroke.id);
    showPreview(null);
  };
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
        if (!stroke.cancelled && (result || stroke.releaseOnly)) showPreview(result ?? null);
      }
      if (stroke.releasing && !stroke.queued) {
        stroke.releasing = false;
        if (!stroke.ending && await stroke.transaction.advance()) stroke.hovering = true;
        else stroke.ending = true;
      }
      if (stroke.ending && !stroke.queued) {
        await stroke.transaction.finish(stroke.cancelled);
        release(stroke);
      }
    } catch (error) {
      if (stroke.releaseOnly) showPreview(null);
      notice(error);
      // Retain the last accepted geometry and history if a sample was rejected.
      if (stroke.ending) {
        try { await stroke.transaction.finish(stroke.cancelled); }
        catch (finishError) { notice(finishError); }
        finally { release(stroke); }
      }
    } finally {
      stroke.busy = false;
      if (active.current && (stroke.queued || stroke.ending || stroke.releasing)) schedule();
    }
  };
  const schedule = () => {
    const stroke = active.current;
    if (!stroke || stroke.frame !== undefined) return;
    stroke.frame = requestAnimationFrame(() => { stroke.frame = undefined; void pump(); });
  };
  const endStroke = (cancelled: boolean) => {
    const stroke = active.current;
    if (!stroke) return;
    stroke.cancelled ||= cancelled;
    stroke.ending = true;
    if (cancelled) stroke.queued = undefined;
    schedule();
  };
  /** Step the armed tool's width, or its depth, by whole voxels within the control's own range. */
  const resize = (controlId: "size" | "depth", steps: number): boolean => {
    const ui = session.ui.getState();
    const plugin = ui.viewportMode === "interact" ? voxelTools.get(ui.voxelToolId) : undefined;
    const control = plugin?.ui.controls.find((candidate) => candidate.id === controlId);
    if (!plugin || !control || !steps) return false;
    const current = toolValues(plugin, ui.voxelToolValues[plugin.id], session.scene.getState().scene)[control.id]!;
    ui.setVoxelToolValue(plugin.id, control.id, Math.max(control.min, Math.min(control.max, current + steps * control.step)));
    return true;
  };
  const resizeFromKey = (event: KeyboardEvent): boolean => {
    // `[` and `]` size the brush, as in every paint program; Shift sizes its depth.
    if (event.code !== "BracketLeft" && event.code !== "BracketRight") return false;
    if (event.metaKey || event.ctrlKey || event.altKey) return false;
    const target = event.target as HTMLElement | null;
    if (target?.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target?.tagName ?? "")) return false;
    if (!resize(event.shiftKey ? "depth" : "size", event.code === "BracketRight" ? 1 : -1)) return false;
    event.preventDefault();
    return true;
  };
  const resizeFromEffect = useEffectEvent(resizeFromKey);
  const endFromEffect = useEffectEvent(endStroke);
  const clearFromEffect = useEffectEvent(() => showPreview(null));
  const noticeFromEffect = useEffectEvent(notice);
  useEffect(() => {
    // Switching tools or leaving edit mode commits the accepted stroke once.
    if (active.current) endFromEffect(active.current.releaseOnly); else clearFromEffect();
  }, [armed, viewportMode]);
  useEffect(() => {
    if (["unavailable", "lost", "blocked", "stopping"].includes(gpuState)) {
      // A failed runtime cannot validate more samples. Preserve the last
      // accepted geometry and finish its history so it can still be saved.
      if (active.current) active.current.queued = undefined;
      endFromEffect(active.current?.releaseOnly ?? false);
    }
  }, [gpuState]);
  useEffect(() => {
    mounted.current = true;
    const cancel = (event: KeyboardEvent) => {
      if (resizeFromEffect(event)) return;
      if (!active.current && !session.ui.getState().voxelStrokePending) return;
      const modified = event.metaKey || event.ctrlKey;
      const blocked = modified && ["s", "y"].includes(event.key.toLowerCase());
      if (blocked) { event.preventDefault(); event.stopImmediatePropagation(); return; }
      const undo = (event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "z";
      if (event.key !== "Escape" && !undo) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      // Undo during a stroke cancels that stroke, not the preceding history entry.
      endFromEffect(true);
    };
    const lost = (event: globalThis.PointerEvent) => {
      const stroke = active.current;
      // The browser drops capture at every release; only losing it mid-drag abandons the stroke.
      if (stroke && stroke.id === event.pointerId && !stroke.ending && !stroke.releasing && !stroke.hovering) endFromEffect(true);
    };
    const blur = () => endFromEffect(true);
    window.addEventListener("keydown", cancel, true);
    window.addEventListener("lostpointercapture", lost, true);
    window.addEventListener("blur", blur);
    return () => {
      mounted.current = false;
      window.removeEventListener("keydown", cancel, true);
      window.removeEventListener("lostpointercapture", lost, true);
      window.removeEventListener("blur", blur);
      const stroke = active.current;
      if (stroke) {
        if (stroke.frame !== undefined) cancelAnimationFrame(stroke.frame);
        stroke.queued = undefined;
        stroke.ending = true;
        active.current = undefined;
        // Transaction serialization waits for any outstanding preflight, even
        // after React has removed the canvas and animation frames stop firing.
        void stroke.transaction.finish(stroke.cancelled || stroke.releaseOnly).catch(noticeFromEffect);
      }
    };
  }, [session]);
  return {
    down(event: PointerEvent<HTMLCanvasElement>): boolean {
      const hovering = active.current;
      if (hovering?.hovering && !hovering.ending) {
        // The press that ends a bare-pointer phase: primary commits what is shown, anything else abandons it.
        if (event.button === 0) { hovering.queued = ray(event); hovering.ending = true; schedule(); }
        else endStroke(true);
        return true;
      }
      if (active.current || session.ui.getState().voxelStrokePending) return true;
      const ui = session.ui.getState();
      const plugin = ui.viewportMode === "interact" ? voxelTools.get(ui.voxelToolId) : undefined;
      if (!plugin || event.button !== 0 || event.shiftKey) return false;
      const unavailable = plugin.unavailable({ scene: session.scene.getState().scene,
        methodId: session.method.getState().methodId });
      if (unavailable) { notice(unavailable); return true; }
      const initial = ray(event);
      try {
        const baseSnapshot = { label: plugin.ui.label, scene: session.scene.getState().scene,
          presetId: session.scene.getState().presetId };
        const transaction = beginToolTransaction(plugin, {
          scene: () => session.scene.getState().scene,
          publish: async (next, base) => {
            const before = session.scene.getState().scene;
            await validate(next, base);
            if (session.scene.getState().scene !== before) throw new Error("Scene changed during the stroke.");
            session.scene.getState().setScene(next);
          },
          execute,
          begin: () => session.ui.setState({ voxelStrokePending: true }),
          finish: () => {
            try {
              if (session.scene.getState().scene !== baseSnapshot.scene) session.history.getState().record(baseSnapshot);
            } finally { session.ui.setState({ voxelStrokePending: false }); }
          },
          cancel: () => session.ui.setState({ voxelStrokePending: false }),
        }, initial, ui.voxelToolValues[plugin.id], event.altKey);
        if (transaction) {
          try { event.currentTarget.setPointerCapture(event.pointerId); }
          catch (error) { void transaction.finish(true).catch(notice); throw error; }
          active.current = { id: event.pointerId, transaction, queued: initial, busy: false, releaseOnly: plugin.execution === "release", cancelled: false, element: event.currentTarget };
          schedule();
        }
      } catch (error) { notice(error); }
      return true;
    },
    move(event: PointerEvent<HTMLCanvasElement>): boolean {
      const stroke = active.current;
      if (stroke) {
        if ((stroke.hovering || stroke.id === event.pointerId) && !stroke.ending) { stroke.queued = ray(event); schedule(); }
        return true;
      }
      const ui = session.ui.getState();
      const plugin = ui.viewportMode === "interact" ? voxelTools.get(ui.voxelToolId) : undefined;
      if (!plugin) return false;
      try {
        // Hover uses the exact plugin targeting and geometry without publishing.
        const sample = ray(event);
        const gesture = plugin.begin({ scene: session.scene.getState().scene, ray: sample,
          values: toolValues(plugin, ui.voxelToolValues[plugin.id], session.scene.getState().scene), invert: event.altKey });
        showPreview(gesture?.update(sample) ?? null);
      } catch { showPreview(null); }
      return true;
    },
    up(event: PointerEvent<HTMLCanvasElement>): boolean {
      const stroke = active.current;
      if (!stroke || stroke.id !== event.pointerId) return false;
      if (stroke.ending || stroke.hovering) return true;
      stroke.cancelled ||= event.type === "pointercancel";
      stroke.queued = stroke.cancelled ? undefined : ray(event);
      // A cancelled pointer ends the stroke; a released one first asks the gesture whether it continues.
      if (stroke.cancelled) stroke.ending = true; else stroke.releasing = true;
      schedule(); return true;
    },
    /** Alt + wheel sizes the armed tool where the pointer is; Shift sizes its depth. The bare wheel stays the camera's. */
    wheel(event: WheelEvent<HTMLCanvasElement>): boolean {
      if (!event.altKey) return false;
      const delta = event.deltaY || event.deltaX;
      return resize(event.shiftKey ? "depth" : "size", delta < 0 ? 1 : delta > 0 ? -1 : 0);
    },
  };
}
