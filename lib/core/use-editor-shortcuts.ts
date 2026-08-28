"use client";

import { useEffect } from "react";
import { editorAxisForKey, toggleAxisConstraint } from "./editor-axis-constraint";
import { toggledViewportMode, VIEWPORT_MODE_SHORTCUT } from "./editor-viewport-mode";
import { cameraForFraming, cameraFramingForKey } from "./editor-camera-framing";
import { editorGestureForShortcut } from "./editor-gesture-catalog";
import { stepFluidCellTraceHit } from "./fluid-cell-trace";
import { editorEntityContext, findEntity } from "./editor-entity-catalog";
import { sceneInstrumentForShortcut } from "./scene-instruments";
import { simulation } from "./simulation/controller";
import { useDiagnosticsStore } from "./stores/diagnostics-store";
import { useUIStore } from "./stores/ui-store";

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
export function useEditorShortcuts(): void {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (editingText(event.target)) return;
      const ui = useUIStore.getState();
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
        if (event.shiftKey) simulation.redo(); else simulation.undo();
        return;
      }
      if (accelerator && event.key.toLowerCase() === "y") {
        event.preventDefault();
        simulation.redo();
        return;
      }
      if (accelerator || event.altKey) return;

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
        if (ui.pixelTraceEnabled || ui.fluidCellTraceEnabled) {
          ui.setPixelTraceEnabled(false);
          ui.setFluidCellTraceEnabled(false);
          return;
        }
        if (ui.sceneOverlay) { ui.setSceneOverlay(null); return; }
        if (ui.axisConstraint) { ui.setAxisConstraint(undefined); return; }
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
      // Delete asks the selected entity for the scene without it. Entities that
      // cannot be removed — the tank, the water body — simply do not offer one,
      // so the key falls through rather than being denied by a list here.
      if (event.key === "Delete" || event.key === "Backspace") {
        const entity = findEntity(editorEntityContext(), ui.selection);
        if (entity?.remove) {
          event.preventDefault();
          simulation.removeEntity(`Removed ${entity.label}`, entity.remove());
          return;
        }
      }
      if (event.key.toLowerCase() === "f" && ui.selection?.kind === "body") {
        const body = useDiagnosticsStore.getState().bodies.find((candidate) => candidate.description.id === ui.selection?.id);
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
      // Entering and leaving pick mode. "c" for cell, and no editor tool claims
      // it; the scene's own toggle carries the same letter so the two cannot
      // drift apart in a reader's head.
      if (event.key.toLowerCase() === "c") {
        event.preventDefault();
        ui.setFluidCellTraceEnabled(!ui.fluidCellTraceEnabled);
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
      // Walking the cell picker along the pointer ray. Bracket keys because the
      // run is ordered by depth and they read as "further in" / "back out", and
      // because no editor tool claims them.
      if (ui.fluidCellTraceEnabled && (event.key === "[" || event.key === "]")) {
        event.preventDefault();
        ui.setFluidCellTraceHitIndex(stepFluidCellTraceHit(
          ui.fluidCellTraceHitIndex, event.key === "]" ? 1 : -1, ui.fluidCellTraceHitCount));
        return;
      }
      // Jumping to the next leaf the surface passes through. Stepping one at a
      // time crosses a dozen interior cells first, and the interesting cell in a
      // domain of thousands is almost always one the interface touches.
      if (ui.fluidCellTraceEnabled && event.key.toLowerCase() === "i") {
        event.preventDefault();
        ui.jumpFluidCellTraceToInterface();
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
  }, []);
}
