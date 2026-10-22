"use client";
import { useEffect, useEffectEvent, useRef, useState, type PointerEvent, type WheelEvent } from "react";
import type { EditorRay } from "../lib/core/editor-entity";
import { voxelToolKeyControls } from "../lib/core/editor-voxel-tool-actions";
import { adjustableControl, controlForKey, controlValueOwner, storedToolValues, toolValues, type ToolAction,
  type ToolUpdate, type ToolValues, type VoxelToolPlugin } from "../lib/core/voxel-editor/plugin";
import { materialControl } from "../lib/core/voxel-editor/controls";
import { sampledMaterial } from "../lib/core/voxel-editor/geometry";
import { beginToolTransaction } from "../lib/core/voxel-editor/transaction";
import { voxelTools } from "../lib/core/voxel-editor/registry";
import { sessionToolHost } from "../lib/core/voxel-editor/session-host";
import { solidEditQueue } from "../lib/core/voxel-editor/solid-edit-queue";
import { startVoxelIndexPreparation } from "../lib/core/voxel-editor/voxel-index";
import { useSession } from "../lib/core/session/session-context";
import type { SceneDescription } from "../lib/core/model";
import type { SolidWorldVoxelPatch } from "../lib/core/solid-world";

type Transaction = NonNullable<ReturnType<typeof beginToolTransaction>>;
interface Stroke {
  id: number; transaction: Transaction; queued?: EditorRay;
  busy: boolean; releaseOnly: boolean; ending?: boolean; cancelled: boolean; frame?: number; element: HTMLCanvasElement;
  /** The pointer came up and the gesture has not yet said whether it continues. */
  releasing?: boolean;
  /** A later phase follows the bare pointer; the next press commits it. */
  hovering?: boolean;
  /** Handed to the edit queue or abandoned: the pointer is no longer this stroke's. */
  concluded?: boolean;
}
/**
 * What a press means while a tool is armed: the tool took it, nothing here
 * wants it, or it is the camera's — Shift, the middle button, a held Space, or
 * a press on the empty room, which orbits exactly as it does with no tool armed.
 */
export type VoxelPressClaim = boolean | "camera";
/** A mouse notch is one step; a trackpad's stream of small deltas is summed to this much travel per step. */
const WHEEL_STEP_PX = 50;
/** A right-click that abandoned a stroke or put the tool away must not also open the ring, whichever order the browser reports them in. */
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
 * - Alt does the opposite for one stroke — attach erases, erase attaches, a
 *   drop removes; paint has no opposite and ignores it;
 * - `[` `]` and Alt-wheel step the tool's first number, Shift its second, and
 *   each toggle and each option of a choice answers the letter its control
 *   declares. This listener runs in the capture phase, so those letters — T, R
 *   and G for the operation — win over the editor's own while a tool that
 *   lists them is armed;
 * - the camera never goes away: the empty room orbits, Shift or the middle
 *   button pans, and holding Space lends the whole pointer to the camera for as
 *   long as it is down, over solids included;
 * - Escape, ⌘Z or the other mouse button abandons the stroke in flight; with no
 *   stroke, Escape or the other button puts the tool away;
 * - a press is never turned away because the last edit is still being accepted:
 *   the pointer is free the moment a stroke is released, and the next stroke is
 *   authored over the edits in flight and queued behind them
 *   (`solid-edit-queue`).
 *
 * `preview` draws the proposal's outline; `stage` shows the proposal itself
 * through the display, where the runtime can.
 *
 * Which keys exist is the plugin's declaration (`ToolControl.adjust`,
 * `.shortcut`); this host never names a tool or a control.
 */
export function useVoxelToolGesture(ray: (event: PointerEvent<HTMLCanvasElement>) => EditorRay,
  validate: (scene: SceneDescription, base: SceneDescription) => Promise<void>, preview: (update: ToolUpdate | null) => void,
  execute?: (action: ToolAction) => Promise<void>,
  stage?: (patches: readonly SolidWorldVoxelPatch[] | undefined, base: SceneDescription) => void) {
  const session = useSession();
  const armed = session.ui((state) => state.voxelToolId);
  const armedValues = session.ui((state) => state.voxelToolValues);
  const viewportMode = session.ui((state) => state.viewportMode);
  const gpuState = session.diagnostics((state) => state.gpuStatus.state);
  /** The stroke that has the pointer. One being accepted has let go of it and is not here. */
  const active = useRef<Stroke | undefined>(undefined);
  /** The stroke whose outline is up: it stays over an edit being accepted until that is answered or a newer stroke draws. */
  const outlined = useRef<Stroke | undefined>(undefined);
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
  const outline = (stroke: Stroke, update: ToolUpdate | null) => { outlined.current = stroke; showPreview(update); };
  /** The bare pointer's ghost, which never draws over the outline of an edit still being accepted. */
  const ghost = (update: ToolUpdate | null) => { if (!outlined.current) showPreview(update); };
  const armedPlugin = () => {
    const ui = session.ui.getState();
    return ui.viewportMode === "interact" ? voxelTools.get(ui.voxelToolId) : undefined;
  };
  /** The armed tool's values as they stand: its own, and the ones the whole editor shares. */
  const currentValues = (plugin: VoxelToolPlugin, scene: SceneDescription): ToolValues =>
    toolValues(plugin, storedToolValues(plugin, session.ui.getState().voxelToolValues), scene);
  /** Redraw the idle ghost from the last sample: the exact gesture a press there would begin. */
  const rehover = () => {
    if (active.current || !mounted.current) return;
    const plugin = armedPlugin();
    const sample = hover.current;
    if (!plugin || !sample || spaceHeld.current) { ghost(null); setAim(plugin && sample ? "camera" : undefined); return; }
    try {
      // What a press would be authored against: the document with the edits in flight applied.
      const scene = solidEditQueue(session).head();
      // Under paint the modifier samples, and the ghost is the cell it would sample.
      const sampled = sample.alt ? sampledMaterial(plugin, { scene, values: currentValues(plugin, scene) }, sample.ray) : undefined;
      if (sampled) { ghost(sampled.preview); setAim("tool"); return; }
      // `hover` tells a tool whose press is a search that this is only the ghost.
      const gesture = plugin.begin({ scene, ray: sample.ray, values: currentValues(plugin, scene), invert: sample.alt,
        clipboard: session.ui.getState().voxelClipboard, hover: true });
      // Nothing under the pointer: the press would orbit, so nothing is promised there.
      const lands = gesture !== undefined && gesture.surface !== false;
      ghost(lands ? gesture.update(sample.ray) ?? null : null);
      setAim(lands ? "tool" : "camera");
    } catch { ghost(null); setAim("tool"); }
  };
  const notice = (error: unknown) => session.runtime.getState().setNotice(error instanceof Error ? error.message : String(error), "warn");
  /**
   * The stroke is over for the pointer: free it for the next press now, and let
   * the edit be accepted behind it. Its outline stays up until that is answered.
   */
  const conclude = async (stroke: Stroke) => {
    stroke.concluded = true;
    if (stroke.frame !== undefined) { cancelAnimationFrame(stroke.frame); stroke.frame = undefined; }
    if (active.current === stroke) active.current = undefined;
    if (stroke.element.hasPointerCapture(stroke.id)) stroke.element.releasePointerCapture(stroke.id);
    // The sample the ghost was last drawn from predates the stroke.
    hover.current = undefined;
    try { await stroke.transaction.finish(stroke.cancelled); }
    catch (error) { notice(error); }
    if (outlined.current !== stroke) return;
    outlined.current = undefined;
    showPreview(null);
    if (hover.current) rehover();
  };
  /**
   * Run what the stroke has waiting: its latest sample, then the release, then
   * the end. Samples resolve within the turn, so a stroke that ends has let go
   * of the pointer before the next press can arrive.
   */
  const pump = async (stroke: Stroke) => {
    if (stroke.busy || stroke.concluded) return;
    stroke.busy = true;
    try {
      const sample = stroke.queued;
      stroke.queued = undefined;
      if (sample && !stroke.cancelled) {
        const result = await stroke.transaction.update(sample);
        if (!stroke.cancelled && (result || stroke.releaseOnly)) outline(stroke, result ?? null);
      }
      if (stroke.releasing && !stroke.queued) {
        stroke.releasing = false;
        if (!stroke.ending && await stroke.transaction.advance()) stroke.hovering = true;
        else stroke.ending = true;
      }
    } catch (error) {
      if (stroke.releaseOnly) outline(stroke, null);
      // Author tools retain their last proposal. Live validation failures
      // roll the stroke back; a failed rollback keeps its accepted prefix undoable.
      notice(error);
    }
    stroke.busy = false;
    if (stroke.ending && !stroke.queued) await conclude(stroke);
    else if (stroke.ending || stroke.releasing) await pump(stroke);
    else if (stroke.queued) schedule(stroke);
  };
  /** Samples of a drag are taken once a frame, the latest winning; a press, a release and an end are not made to wait for one. */
  const schedule = (stroke: Stroke) => {
    if (stroke.frame !== undefined || stroke.concluded) return;
    stroke.frame = requestAnimationFrame(() => { stroke.frame = undefined; void pump(stroke); });
  };
  const endStroke = (cancelled: boolean) => {
    const stroke = active.current;
    if (!stroke) return;
    stroke.cancelled ||= cancelled;
    stroke.ending = true;
    if (cancelled) stroke.queued = undefined;
    void pump(stroke);
  };
  /** Step the armed tool's first number, or its second, by whole steps within the control's own range. */
  const resize = (secondary: boolean, steps: number): boolean => {
    const plugin = armedPlugin();
    const control = plugin && adjustableControl(plugin, secondary);
    if (!plugin || !control || !steps) return false;
    const current = currentValues(plugin, session.scene.getState().scene)[control.id]!;
    session.ui.getState().setVoxelToolValue(controlValueOwner(plugin.id, control), control.id,
      Math.max(control.min, Math.min(control.max, current + steps * control.step)));
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
    // A switch flips on its letter, and a choice takes the option whose letter
    // it is — the rail's own switch included, under every tool that stands there.
    const keyed = controlForKey(voxelToolKeyControls(plugin.id), event.key.toLowerCase(),
      currentValues(plugin, session.scene.getState().scene));
    if (!keyed) return false;
    ui.setVoxelToolValue(controlValueOwner(plugin.id, keyed.control), keyed.control.id, keyed.value);
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
  // This pane's index of its solids, built before its first pointer move asks for it.
  useEffect(() => startVoxelIndexPreparation(session.scene), [session]);
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
        stroke.concluded = true;
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
    /** The other button just abandoned a stroke or put the tool away, so this context menu is not a request for the ring. */
    swallowsContextMenu(timeStamp: number): boolean {
      return active.current !== undefined || timeStamp - secondaryCancelAt.current < SECONDARY_CANCEL_QUIET_MS;
    },
    down(event: PointerEvent<HTMLCanvasElement>): VoxelPressClaim {
      const hovering = active.current;
      if (hovering?.hovering && !hovering.ending) {
        // The press that ends a bare-pointer phase: primary commits what is shown, anything else abandons it.
        if (event.button === 0) { hovering.queued = ray(event); hovering.ending = true; void pump(hovering); }
        else { secondaryCancelAt.current = event.timeStamp; endStroke(true); }
        return true;
      }
      if (active.current) {
        // A second button mid-drag is "no": the stroke is abandoned and nothing is authored.
        if (event.button !== 0) { secondaryCancelAt.current = event.timeStamp; endStroke(true); }
        return true;
      }
      const ui = session.ui.getState();
      const plugin = armedPlugin();
      // A carry owns its own clicks, the other button among them.
      if (!plugin || ui.carry) return false;
      // The other button is "no" all the way down: it abandons a stroke in
      // flight (above), and with none it puts the tool away, as Escape does. The
      // ring is the bare pointer's; the context menu this press raises is not a
      // request for it.
      if (event.button === 2) {
        secondaryCancelAt.current = event.timeStamp;
        showPreview(null);
        ui.setVoxelTool(undefined);
        return true;
      }
      // The ghost promised a stroke; a press the camera takes must not leave it hanging over the orbit.
      const camera = () => { ghost(null); return "camera" as const; };
      if (event.button !== 0 || event.shiftKey || event.ctrlKey || spaceHeld.current) return camera();
      // An edit still being accepted does not hold the press up: this one is
      // authored over it and queued behind it.
      const scene = solidEditQueue(session).head();
      const initial = ray(event);
      try {
        // Under paint the modifier takes the material under the pointer into
        // the hand, and the press authors nothing.
        const sampled = event.altKey ? sampledMaterial(plugin, { scene, values: currentValues(plugin, scene) }, initial) : undefined;
        if (sampled) {
          ui.setVoxelToolValue(controlValueOwner(plugin.id, materialControl), materialControl.id, sampled.materialId);
          return true;
        }
      } catch (error) { notice(error); return true; }
      let begun: ReturnType<VoxelToolPlugin["begin"]>;
      try {
        // Asked before the transaction opens, so a press on the empty room never
        // flickers the document into a pending stroke on its way to the camera.
        begun = plugin.begin({ scene, ray: initial, values: currentValues(plugin, scene), invert: event.altKey,
          clipboard: ui.voxelClipboard });
        if (!begun || begun.surface === false) return camera();
      } catch (error) { notice(error); return true; }
      const unavailable = plugin.unavailable({ scene, methodId: session.method.getState().methodId,
        clipboard: ui.voxelClipboard });
      if (unavailable) { notice(unavailable); return true; }
      try {
        // What "commit" means — preflight, one history entry, the pending flag —
        // is the session host's, shared with the selection verbs.
        // The gesture that answered where the press lands is the stroke's own.
        const transaction = beginToolTransaction(plugin, sessionToolHost(session, validate, execute, stage),
          initial, storedToolValues(plugin, ui.voxelToolValues), event.altKey, begun);
        if (transaction) {
          try { event.currentTarget.setPointerCapture(event.pointerId); }
          catch (error) { void transaction.finish(true).catch(notice); throw error; }
          const stroke: Stroke = { id: event.pointerId, transaction, queued: initial, busy: false, releaseOnly: plugin.execution === "release", cancelled: false, element: event.currentTarget };
          active.current = stroke;
          // The ghost that promised this stroke is its outline from here on.
          outlined.current = stroke;
          void pump(stroke);
        }
      } catch (error) { notice(error); }
      return true;
    },
    move(event: PointerEvent<HTMLCanvasElement>): boolean {
      const stroke = active.current;
      if (stroke) {
        if ((stroke.hovering || stroke.id === event.pointerId) && !stroke.ending) { stroke.queued = ray(event); schedule(stroke); }
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
      if (!active.current) { ghost(null); setAim(undefined); }
    },
    up(event: PointerEvent<HTMLCanvasElement>): boolean {
      const stroke = active.current;
      if (!stroke || stroke.id !== event.pointerId) return false;
      if (stroke.ending || stroke.hovering) return true;
      stroke.cancelled ||= event.type === "pointercancel";
      stroke.queued = stroke.cancelled ? undefined : ray(event);
      // A cancelled pointer ends the stroke; a released one first asks the gesture whether it continues.
      if (stroke.cancelled) stroke.ending = true; else stroke.releasing = true;
      void pump(stroke); return true;
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
