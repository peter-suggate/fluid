"use client";

import { useEffect } from "react";
import { editorAxisForKey, toggleAxisConstraint } from "./editor-axis-constraint";
import { toggledViewportMode, VIEWPORT_MODE_SHORTCUT } from "./editor-viewport-mode";
import { cameraForFraming, cameraFramingForKey } from "./editor-camera-framing";
import { toggleCompareMode } from "./compare/compare-mode";
import { editorGestureForShortcut } from "./editor-gesture-catalog";
import { editorEntityContext, findEntity } from "./editor-entity-catalog";
import { sceneInstrumentForShortcut } from "./scene-instruments";
import { performEditorAction } from "./editor-action-runtime";
import { voxelRailToolForKey } from "./editor-voxel-tool-actions";
import { voxelRegionVerbForKey } from "./editor-voxel-region";
import type { PaneSession } from "./session/session";
import { useSession } from "./session/session-context";
import { simulation } from "./simulation/controller";

/** Typing in a form control must never arm a gesture or delete a body. */
function editingText(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  return ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
}

/**
 * Editor keyboard chassis: gesture arming, camera framing, undo/redo, selection
 * escape, delete, and focus-on-selection. Registered once by the shell so
 * shortcuts work wherever focus happens to be, not only over the canvas.
 */
export function useEditorShortcuts(focused?: PaneSession): void {
  // Keys route to the *focused* pane: the last one pointed at. The compare host
  // hands it in; anywhere else, the session this hook is mounted under is the
  // only one there is.
  const mounted = useSession();
  const session = focused ?? mounted;
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (editingText(event.target)) return;
      const ui = session.ui.getState();
      const accelerator = event.metaKey || event.ctrlKey;

      // While something is being carried, the viewport owns the keyboard except
      // for undo/redo: Q and E tilt what is in hand rather than arming anything,
      // and Escape puts it back rather than dropping the selection. A carry is
      // the innermost mode, and modes are left from the inside out.
      if (ui.carry && !accelerator) return;
      // The ring is the same case one level out: while it is open its own
      // handler walks and closes it.
      if (ui.radialMenu && !accelerator) return;

      if (accelerator && event.key.toLowerCase() === "z") {
        event.preventDefault();
        if (event.shiftKey) simulation.redo(session.id); else simulation.undo(session.id);
        return;
      }
      if (accelerator && event.key.toLowerCase() === "y") {
        event.preventDefault();
        simulation.redo(session.id);
        return;
      }
      // The voxel selection's verbs, by the chords the registry declares for
      // them: nothing here names a verb or a key. They answer while a voxel
      // selection stands — and, for the ones that need none (select all,
      // paste), anywhere in EDIT. A verb that does not apply still takes its
      // key and says why. Asked at three points below, because its chords sit
      // at three depths of this handler: ⌘C, ⌘X, ⌘V and ⌘A before the
      // accelerators are waved through, Escape inside the order modes are left
      // in, and the bare keys ahead of the entity route.
      const runVerbKey = (): boolean => {
        if (ui.viewportMode !== "interact" || ui.carry || ui.radialMenu) return false;
        // A key some panel already took — the scene chooser walking its grid on the arrows — is not also a nudge.
        if (event.defaultPrevented) return false;
        // With page text selected an accelerator is the browser's: the reader is copying a figure off a readout.
        if (accelerator && window.getSelection()?.isCollapsed === false) return false;
        const binding = voxelRegionVerbForKey(event, ui.selection?.kind === "voxel-region");
        if (!binding) return false;
        event.preventDefault();
        performEditorAction({ kind: "voxel-verb", verbId: binding.verb.id, variantId: binding.variant?.id }, session);
        return true;
      };
      if (accelerator && runVerbKey()) return;
      if (accelerator || event.altKey) return;

      // The split. A host key rather than a pane key — it is about how many
      // panes there are — and the backslash because it is the vertical bar the
      // splitter draws. Ahead of everything below it so it also works as the
      // way *out* of a mode nothing else can leave.
      if (event.key === "\\") {
        event.preventDefault();
        toggleCompareMode();
        return;
      }

      // The mode swap, ahead of everything the mode contains. It is the only key
      // that means something in both modes, and it has to keep working while a
      // selection or an axis lock is up — leaving LOOK is how you put all of
      // that down (see `setViewportMode`), so a guard that made you clear them
      // first would have the dependency backwards.
      if (event.key === VIEWPORT_MODE_SHORTCUT && !event.shiftKey) {
        event.preventDefault();
        ui.setViewportMode(toggledViewportMode(ui.viewportMode));
        return;
      }
      // Axis constraints, on the letters every 3D editor uses for them. A
      // selection claims x/y/z outright rather than only while a handle is held:
      // a constraint you can only reach mid-drag cannot be seen before you
      // commit to the gesture, and arming one first is how a run of single-axis
      // adjustments is actually made. The one casualty is `y` for ERASE while
      // something is selected, which the toolbar and every other tool key still
      // reach.
      if (ui.selection) {
        const axis = editorAxisForKey(event.key);
        if (axis) {
          event.preventDefault();
          ui.setAxisConstraint(toggleAxisConstraint(
            ui.axisConstraint, axis, event.shiftKey ? "plane" : "axis"));
          return;
        }
      }
      if (event.key === "Escape") {
        event.preventDefault();
        // A mode inside a mode is left from the inside out, so the first Escape
        // drops the axis lock and the second leaves the tool. A raised
        // instrument is the outermost of those: it covers part of the scene, so
        // it goes before anything about the selection under it moves.
        //
        // A raised *probe* goes before even that, because it governs the pointer
        // and not just some pixels: while one is up the click aims it rather than
        // selecting (see the probe's claim on the press in `WebGPUViewport`), so
        // this is the key that gives the click back. Both go together — they are
        // one layer, two questions about the same pixel — and one Escape putting
        // down "the probes" is a simpler thing to know than an order between them.
        // The scene chooser is outside all of it: a popover over the image,
        // like the ring. Reachable from here only when focus has wandered off
        // its own field — which swallows the key itself so that Escape means
        // "discard the typing" first.
        if (ui.sceneSelectorOpen) { ui.setSceneSelectorOpen(false); return; }
        if (ui.pixelTraceEnabled) {
          ui.setPixelTraceEnabled(false);
          return;
        }
        if (ui.sceneOverlay) { ui.setSceneOverlay(null); return; }
        if (ui.axisConstraint) { ui.setAxisConstraint(undefined); return; }
        // A standing voxel selection is put down by its own verb, and before
        // the tool that made it is put away: what a selecting tool is holding
        // is inside the tool, and modes are left from the inside out.
        if (runVerbKey()) return;
        // An armed sculpt tool is a mode inside EDIT like any other, so Escape
        // puts it away before it reaches the mode itself. A stroke in flight
        // never gets this far: the gesture host takes that Escape for the stroke.
        if (ui.voxelToolId && ui.viewportMode === "interact") { ui.setVoxelTool(undefined); return; }
        // INTERACT is the outermost of all of them, so it is the last thing
        // Escape reaches: with nothing armed and nothing selected there is
        // nothing left inside the mode to leave, and the next Escape is the mode.
        if (!ui.armedGesture && !ui.selection) {
          ui.setViewportMode("camera");
          return;
        }
        ui.setArmedGesture(undefined);
        ui.select(undefined);
        return;
      }
      // A standing voxel selection takes the bare keys its verbs declare —
      // Delete clears it, the arrows and the page keys move it — ahead of the
      // entity route below, which it does not use: a solid edit is committed
      // the way a stroke is, not as a re-seeded scene.
      if (runVerbKey()) return;
      // Delete asks the selected entity for the scene without it. Entities that
      // cannot be removed — the tank, the water body — simply do not offer one,
      // so the key falls through rather than being denied by a list here.
      if (event.key === "Delete" || event.key === "Backspace") {
        const entity = findEntity(editorEntityContext(session), ui.selection);
        if (entity?.remove) {
          event.preventDefault();
          simulation.removeEntity(`Removed ${entity.label}`, entity.remove(), session.id);
          return;
        }
      }
      if (event.key.toLowerCase() === "f" && ui.selection?.kind === "body") {
        const body = session.diagnostics.getState().bodies.find((candidate) => candidate.description.id === ui.selection?.id);
        if (!body) return;
        event.preventDefault();
        ui.setCamera((current) => ({ ...current, target_m: { ...body.position_m } }));
        return;
      }
      // Raising an instrument. The only route to one that does not begin with a
      // right-click landing on something: the rings offer the pipelines on the
      // water and on the room, so a scene whose pick is rebuilding, or whose
      // water is off screen, or which is being watched rather than edited, had
      // no way to ask what it costs. Same toggle bargain as a tool key — the
      // key that raised it puts it down — and Escape still closes whichever is
      // up, from anywhere.
      const instrument = sceneInstrumentForShortcut(event.key);
      if (instrument) {
        event.preventDefault();
        ui.setSceneOverlay(ui.sceneOverlay === instrument.id ? null : instrument.id);
        return;
      }
      // The scene chooser, on the focused pane. "o" for open — free of every
      // tool letter (b, t, y, g, d), every instrument (s, v, h), both probes
      // (c, r) and the axis locks (x, y, z), which is the whole reason it and
      // not "s" for scene. Same toggle bargain as everything else here, and the
      // popover's own field swallows the key once it is up, so this can only
      // ever open one.
      if (event.key.toLowerCase() === "o") {
        event.preventDefault();
        ui.setSceneSelectorOpen(!ui.sceneSelectorOpen);
        return;
      }
      // The ray picker, on the same bargain. "r" for ray, claimed by no tool and
      // by no instrument. It exists for the same reason the cell key does: a
      // right-click is the only other route to it, and a right-click needs
      // something to land on — so a scene being watched rather than edited, or
      // one whose pick is rebuilding, could not ask what drawing a pixel cost.
      // Enabling only arms the probe: it then follows the pointer, and the pixel
      // is chosen by clicking it, which is why this key does not also pin.
      if (event.key.toLowerCase() === "r") {
        event.preventDefault();
        ui.setPixelTraceEnabled(!ui.pixelTraceEnabled);
        return;
      }
      // The sculpt rail, by place: the third mark is `3`. Only in EDIT, where
      // the rail is on screen to be counted — LOOK keeps the digits for the
      // camera framings below, and `0` is never a place, so the one key that
      // rescues a lost camera works in both. The armed tool's own digit puts it
      // away, on the same bargain as every other key here.
      const railTool = ui.viewportMode === "interact" && !event.shiftKey ? voxelRailToolForKey(event.key) : undefined;
      if (railTool) {
        event.preventDefault();
        ui.setVoxelTool(ui.voxelToolId === railTool ? undefined : railTool);
        return;
      }
      // Camera framing. These replaced a permanent four-button toolbar; `0` in
      // particular is the only way back from a camera orbited into nowhere, so
      // it must exist somewhere even though the buttons do not.
      const framing = cameraFramingForKey(event.key);
      if (framing) {
        event.preventDefault();
        ui.setCamera(cameraForFraming(framing));
        return;
      }
      const gesture = editorGestureForShortcut(event.key);
      if (gesture) {
        event.preventDefault();
        // Asking for a gesture is asking to edit, so the key enters INTERACT
        // rather than being swallowed by LOOK. A shortcut that silently did
        // nothing in one of two modes is how a reader learns the shortcuts are
        // unreliable. Order matters: `setViewportMode("camera")` disarms, so
        // entering has to happen first.
        if (ui.viewportMode !== "interact") {
          ui.setViewportMode("interact");
          ui.setArmedGesture(gesture);
          return;
        }
        // Pressing the armed gesture's own key disarms it, so every mode can be
        // left the same way it was entered rather than only via Escape.
        ui.setArmedGesture(ui.armedGesture === gesture ? undefined : gesture);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [session]);
}
