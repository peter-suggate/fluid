"use client";

import { useEffect } from "react";
import { editorAxisForKey, toggleAxisConstraint } from "./editor-axis-constraint";
import { cameraForFraming, cameraFramingForKey } from "./editor-camera-framing";
import { DEFAULT_EDITOR_TOOL, editorToolForShortcut, editorToolIsActive } from "./editor-tools";
import { stepFluidCellTraceHit } from "./fluid-cell-trace";
import { editorEntityContext, findEntity } from "./editor-entity-catalog";
import { simulation } from "./simulation/controller";
import { useDiagnosticsStore } from "./stores/diagnostics-store";
import { useUIStore } from "./stores/ui-store";

/** Typing in a form control must never arm a tool or delete a body. */
function editingText(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  return ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
}

/**
 * Editor keyboard chassis: tool arming, camera framing, undo/redo, selection
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
      // for undo/redo: Q and E tilt what is in hand rather than arming a tool,
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
        if (ui.sceneOverlay) { ui.setSceneOverlay(null); return; }
        if (ui.axisConstraint) { ui.setAxisConstraint(undefined); return; }
        ui.setActiveTool("select");
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
      // Entering and leaving pick mode. "c" for cell, and no editor tool claims
      // it; the scene's own toggle carries the same letter so the two cannot
      // drift apart in a reader's head.
      if (event.key.toLowerCase() === "c") {
        event.preventDefault();
        ui.setFluidCellTraceEnabled(!ui.fluidCellTraceEnabled);
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
      const tool = editorToolForShortcut(event.key);
      if (tool && editorToolIsActive(tool)) {
        event.preventDefault();
        // Pressing the armed tool's own key disarms it, so every mode can be
        // left the same way it was entered rather than only via Escape.
        ui.setActiveTool(ui.activeTool === tool ? DEFAULT_EDITOR_TOOL : tool);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);
}
