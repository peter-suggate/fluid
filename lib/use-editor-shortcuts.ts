"use client";

import { useEffect } from "react";
import { editorToolForShortcut, editorToolIsActive } from "./editor-tools";
import { propIdFromSelection } from "./editor-props";
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
 * Editor keyboard chassis: tool arming, undo/redo, selection escape, delete,
 * and focus-on-selection. Registered once by the shell so shortcuts work
 * wherever focus happens to be, not only over the canvas.
 */
export function useEditorShortcuts(): void {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (editingText(event.target)) return;
      const ui = useUIStore.getState();
      const accelerator = event.metaKey || event.ctrlKey;

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

      if (event.key === "Escape") {
        event.preventDefault();
        ui.setActiveTool("select");
        ui.select(undefined);
        return;
      }
      if (event.key === "Delete" || event.key === "Backspace") {
        if (ui.selection?.kind === "body") { event.preventDefault(); simulation.removeBody(ui.selection.id); return; }
        const propId = ui.selection?.kind === "prop" ? propIdFromSelection(ui.selection.id) : undefined;
        if (propId) { event.preventDefault(); simulation.removeProp(propId); return; }
      }
      if (event.key.toLowerCase() === "f" && ui.selection?.kind === "body") {
        const body = useDiagnosticsStore.getState().bodies.find((candidate) => candidate.description.id === ui.selection?.id);
        if (!body) return;
        event.preventDefault();
        ui.setCamera((current) => ({ ...current, target_m: { ...body.position_m } }));
        return;
      }
      const tool = editorToolForShortcut(event.key);
      if (tool && editorToolIsActive(tool)) {
        event.preventDefault();
        ui.setActiveTool(tool);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);
}
