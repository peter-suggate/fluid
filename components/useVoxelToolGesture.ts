"use client";
import { useEffect, useEffectEvent, useRef, useState, type PointerEvent, type WheelEvent } from "react";
import type { EditorRay } from "../lib/core/editor-entity";
import { adjustableControl, toolValues, type ToolAction, type ToolUpdate } from "../lib/core/voxel-editor/plugin";
import { beginToolTransaction } from "../lib/core/voxel-editor/transaction";
import { voxelTools } from "../lib/core/voxel-editor/registry";
import { useSession } from "../lib/core/session/session-context";
import type { SceneDescription } from "../lib/core/model";

type Transaction = NonNullable<ReturnType<typeof beginToolTransaction>>;
/**
 * What a press means while a tool is armed: the tool took it, nothing here
 * wants it, or it is the camera's — Shift, the middle button, a held Space, or
 * a press on the empty room, which orbits exactly as it does with no tool armed.
 */
export type VoxelPressClaim = boolean | "camera";
/** A mouse notch is one step; a trackpad's stream of small deltas is summed to this much travel per step. */
const WHEEL_STEP_PX = 50;
/** A right-click that abandoned a stroke must not also open the ring, whichever order the browser reports them in. */
const SECONDARY_CANCEL_QUIET_MS = 800;
const typing = (target: EventTarget | null) => {
  const element = target as HTMLElement | null;
  return Boolean(element?.isContentEditable) || ["INPUT", "TEXTAREA", "SELECT"].includes(element?.tagName ?? "");
};
/**
 * The armed tool's whole conversation with the pointer and the keyboard.
 *
 * Nothing here is a panel. The tool is driven where the hand already is:
 *
 * - drag on a solid strokes; the ghost under a bare pointer is exactly what the
 *   press would start, and it follows Alt, the brackets and every toggle the
 *   moment they change rather than on the next move;
 * - Alt does the opposite — a build carves, a carve builds, a drop removes;
 * - `[` `]` and Alt-wheel step the tool's first number, Shift its second, and
 *   each toggle answers the letter its control declares;
 * - the camera never goes away: the empty room orbits, Shift or the middle
 *   button pans, and holding Space lends the whole pointer to the camera for as
 *   long as it is down, over solids included;
 * - Escape, ⌘Z or the other mouse button abandons the stroke in flight.
 *
 * Which keys exist is the plugin's declaration (`ToolControl.adjust`,
 * `.shortcut`); this host never names a tool or a control.
 */
export function useVoxelToolGesture(ray: (event: PointerEvent<HTMLCanvasElement>) => EditorRay,
  validate: (scene: SceneDescription, base: SceneDescription) => Promise<void>, preview: (update: ToolUpdate | null) => void,
  execute?: (action: ToolAction) => Promise<void>) {
  const session = useSession();
  const armed = session.ui((state) => state.voxelToolId);
  const armedValues = session.ui((state) => state.voxelToolValues);
  const viewportMode = session.ui((state) => state.viewportMode);
  const gpuState = session.diagnostics((state) => state.gpuStatus.state);
  const active = useRef<{ id: number; transaction: Transaction; queued?: EditorRay;
    busy: boolean; releaseOnly: boolean; ending?: boolean; cancelled: boolean; frame?: number; element: HTMLCanvasElement;
    /** The pointer came up and the gesture has not yet said whether it continues. */
    releasing?: boolean;
    /** A later phase follows the bare pointer; the next press commits it. */
    hovering?: boolean } | undefined>(undefined);
  const mounted = useRef(true);
  /** The last bare-pointer sample, kept so the ghost can be redrawn when a key changes what it means. */
  const hover = useRef<{ ray: EditorRay; alt: boolean } | undefined>(undefined);
  const spaceHeld = useRef(false);
  const wheelTravel = useRef(0);
  const secondaryCancelAt = useRef(Number.NEGATIVE_INFINITY);
  const [navigating, setNavigating] = useState(false);
  /** Whether a press where the pointer rests would stroke, or orbit. */
  const [aim, setAim] = useState<"tool" | "camera" | undefined>(undefined);
  const showPreview = (update: ToolUpdate | null) => { if (mounted.current) preview(update); };
  const armedPlugin = () => {
    const ui = session.ui.getState();
    return ui.viewportMode === "interact" ? voxelTools.get(ui.voxelToolId) : undefined;
  };
  /** Redraw the idle ghost from the last sample: the exact gesture a press there would begin. */
  const rehover = () => {
    if (active.current || !mounted.current) return;
    const plugin = armedPlugin();
    const sample = hover.current;
    if (!plugin || !sample || spaceHeld.current) { showPreview(null); setAim(plugin && sample ? "camera" : undefined); return; }
    try {
      const scene = session.scene.getState().scene;
      const gesture = plugin.begin({ scene, ray: sample.ray,
        values: toolValues(plugin, session.ui.getState().voxelToolValues[plugin.id], scene), invert: sample.alt });
      // Nothing under the pointer: the press would orbit, so nothing is promised there.
      const lands = gesture !== undefined && gesture.surface !== false;
      showPreview(lands ? gesture.update(sample.ray) ?? null : null);
      setAim(lands ? "tool" : "camera");
    } catch { showPreview(null); setAim("tool"); }
  };
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
  /** Step the armed tool's first number, or its second, by whole steps within the control's own range. */
  const resize = (secondary: boolean, steps: number): boolean => {
    const plugin = armedPlugin();
    const control = plugin && adjustableControl(plugin, secondary);
    if (!plugin || !control || !steps) return false;
    const ui = session.ui.getState();
    const current = toolValues(plugin, ui.voxelToolValues[plugin.id], session.scene.getState().scene)[control.id]!;
    ui.setVoxelToolValue(plugin.id, control.id, Math.max(control.min, Math.min(control.max, current + steps * control.step)));
    return true;
  };
  /** The keys that belong to the armed tool rather than to a stroke in flight. True when one was taken. */
  const toolKey = (event: KeyboardEvent): boolean => {
    const plugin = armedPlugin();
    const ui = session.ui.getState();
    // The ring and a carry are modes inside this one, and they own the keyboard while they are up.
    if (!plugin || typing(event.target) || ui.radialMenu || ui.carry) return false;
    const take = () => { event.preventDefault(); event.stopImmediatePropagation(); return true; };
    // Alt is the opposite of the tool, so the ghost has to flip with the key, not with the next move.
    if (event.key === "Alt") {
      if (hover.current && !active.current) { hover.current.alt = true; rehover(); }
      return false;
    }
    if (event.metaKey || event.ctrlKey || event.altKey) return false;
    // Space lends the pointer to the camera while it is held — the hand tool of every canvas.
    if (event.code === "Space") {
      if (!event.repeat && !active.current) { spaceHeld.current = true; setNavigating(true); rehover(); }
      return take();
    }
    // `[` and `]` size the brush, as in every paint program; Shift sizes its second number.
    if (event.code === "BracketLeft" || event.code === "BracketRight") {
      return resize(event.shiftKey, event.code === "BracketRight" ? 1 : -1) ? take() : false;
    }
    if (event.shiftKey) return false;
    const toggle = plugin.ui.controls.find((control) => control.shortcut === event.key.toLowerCase());
    if (!toggle) return false;
    const current = toolValues(plugin, ui.voxelToolValues[plugin.id], session.scene.getState().scene)[toggle.id];
    ui.setVoxelToolValue(plugin.id, toggle.id, current === 1 ? 0 : 1);
    return take();
  };
  const toolKeyUp = (event: KeyboardEvent) => {
    if (event.code === "Space" && spaceHeld.current) {
      spaceHeld.current = false; setNavigating(false); rehover();
      // A focused button would otherwise read the release as its own click.
      event.preventDefault();
    }
    if (event.key === "Alt" && hover.current && armedPlugin()) {
      hover.current.alt = false; rehover();
      // Windows hands a bare Alt release to the menu bar.
      event.preventDefault();
    }
  };
  const toolKeyFromEffect = useEffectEvent(toolKey);
  const toolKeyUpFromEffect = useEffectEvent(toolKeyUp);
  const rehoverFromEffect = useEffectEvent(rehover);
  const endFromEffect = useEffectEvent(endStroke);
  const noticeFromEffect = useEffectEvent(notice);
  useEffect(() => {
    // Switching tools or leaving edit mode commits the accepted stroke once.
    if (active.current) endFromEffect(active.current.releaseOnly); else rehoverFromEffect();
  }, [armed, viewportMode]);
  // A changed number or toggle redraws the ghost where the pointer already rests.
  // Apart from the effect above on purpose: a stroke in flight keeps the values
  // it began with, and sizing the brush mid-drag must not end it.
  useEffect(() => { rehoverFromEffect(); }, [armedValues]);
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
      if (toolKeyFromEffect(event)) return;
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
    const blur = () => {
      spaceHeld.current = false; setNavigating(false);
      endFromEffect(true);
    };
    const keyUp = (event: KeyboardEvent) => toolKeyUpFromEffect(event);
    window.addEventListener("keydown", cancel, true);
    window.addEventListener("keyup", keyUp, true);
    window.addEventListener("lostpointercapture", lost, true);
    window.addEventListener("blur", blur);
    return () => {
      mounted.current = false;
      window.removeEventListener("keydown", cancel, true);
      window.removeEventListener("keyup", keyUp, true);
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
    /** A press where the pointer rests would stroke (`"tool"`) or orbit (`"camera"`); undefined with no tool armed. */
    aim: armed !== undefined && viewportMode === "interact" ? (navigating ? "camera" as const : aim) : undefined,
    /** The other button just abandoned a stroke, so this context menu is not a request for the ring. */
    swallowsContextMenu(timeStamp: number): boolean {
      return active.current !== undefined || timeStamp - secondaryCancelAt.current < SECONDARY_CANCEL_QUIET_MS;
    },
    down(event: PointerEvent<HTMLCanvasElement>): VoxelPressClaim {
      const hovering = active.current;
      if (hovering?.hovering && !hovering.ending) {
        // The press that ends a bare-pointer phase: primary commits what is shown, anything else abandons it.
        if (event.button === 0) { hovering.queued = ray(event); hovering.ending = true; schedule(); }
        else { secondaryCancelAt.current = event.timeStamp; endStroke(true); }
        return true;
      }
      if (active.current) {
        // A second button mid-drag is "no": the stroke is abandoned and nothing is authored.
        if (event.button !== 0) { secondaryCancelAt.current = event.timeStamp; endStroke(true); }
        return true;
      }
      if (session.ui.getState().voxelStrokePending) return true;
      const ui = session.ui.getState();
      const plugin = armedPlugin();
      // The secondary button stays the ring's, armed or not.
      if (!plugin || event.button === 2) return false;
      // The ghost promised a stroke; a press the camera takes must not leave it hanging over the orbit.
      const camera = () => { showPreview(null); return "camera" as const; };
      if (event.button !== 0 || event.shiftKey || event.ctrlKey || spaceHeld.current) return camera();
      const scene = session.scene.getState().scene;
      const initial = ray(event);
      try {
        // Asked before the transaction opens, so a press on the empty room never
        // flickers the document into a pending stroke on its way to the camera.
        const lands = plugin.begin({ scene, ray: initial,
          values: toolValues(plugin, ui.voxelToolValues[plugin.id], scene), invert: event.altKey });
        if (!lands || lands.surface === false) return camera();
      } catch (error) { notice(error); return true; }
      const unavailable = plugin.unavailable({ scene, methodId: session.method.getState().methodId });
      if (unavailable) { notice(unavailable); return true; }
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
      if (!armedPlugin()) { hover.current = undefined; return false; }
      // Hover uses the exact plugin targeting and geometry without publishing.
      hover.current = { ray: ray(event), alt: event.altKey };
      rehover();
      return true;
    },
    /** The pointer left the canvas: there is no longer a place for the ghost to stand. */
    leave() {
      hover.current = undefined;
      if (!active.current) { showPreview(null); setAim(undefined); }
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
    /** Alt + wheel sizes the armed tool where the pointer is; Shift sizes its second number. The bare wheel stays the camera's. */
    wheel(event: WheelEvent<HTMLCanvasElement>): boolean {
      if (!event.altKey) return false;
      const plugin = armedPlugin();
      if (!plugin || !adjustableControl(plugin, event.shiftKey)) return false;
      const delta = event.deltaY || event.deltaX;
      // A notch is one step however far the mouse reports it; a trackpad's small deltas are summed, or one swipe would run the brush to its limit.
      const notch = event.deltaMode !== 0 || Math.abs(delta) >= WHEEL_STEP_PX;
      const travel = notch ? Math.sign(delta) * WHEEL_STEP_PX : wheelTravel.current + delta;
      const steps = Math.trunc(travel / WHEEL_STEP_PX);
      wheelTravel.current = notch ? 0 : travel - steps * WHEEL_STEP_PX;
      if (steps) resize(event.shiftKey, -steps);
      return true;
    },
  };
}
