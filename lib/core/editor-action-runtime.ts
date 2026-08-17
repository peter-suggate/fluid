"use client";

import type { EditorActionEffect } from "./editor-action";
import { simulation } from "./simulation/controller";
import { useSceneStore } from "./stores/scene-store";
import { useUIStore } from "./stores/ui-store";

/**
 * How the studio leaves its own route, when the caller has a client router.
 *
 * The ring is drawn by a React component that can hold one; nothing else in this
 * module can. Passing it in keeps the performer total and framework-free, and
 * the fallback below means a caller without one still works — a hard load costs
 * a re-parse of the app, not the navigation.
 */
export interface EditorActionServices {
  readonly navigate?: (href: string) => void;
}

/**
 * Ask for a scene document and load it.
 *
 * The input is created, clicked and dropped rather than mounted by some
 * component that would then have to exist wherever an import can be asked for.
 * It is appended to the document first because Safari will not open a picker
 * for a detached input, and removed as soon as the dialog has answered.
 */
function importSceneFromFile(): void {
  if (typeof document === "undefined") return;
  const input = document.createElement("input");
  input.type = "file";
  input.accept = "application/json,.json";
  input.hidden = true;
  const finish = () => input.remove();
  input.addEventListener("cancel", finish);
  input.addEventListener("change", () => {
    const file = input.files?.[0];
    finish();
    if (!file) return;
    void file.text().then((contents) => simulation.importScene(file.name, contents));
  });
  document.body.append(input);
  input.click();
}

/**
 * Where a declared action becomes something that happened.
 *
 * The entity files name effects and never touch a store, exactly as their
 * handles name patches and never touch the draft — so this is the one function
 * that knows about history, arming and the solver, and the count of places that
 * know does not grow when an entity gains a verb.
 *
 * It is deliberately synchronous and total: every arm of the union is handled
 * here, and a new effect kind is a compile error in this file rather than a
 * wedge that silently does nothing.
 */
export function performEditorAction(
  effect: EditorActionEffect,
  services: EditorActionServices = {},
): void {
  const ui = useUIStore.getState();
  switch (effect.kind) {
    case "scene": {
      // Whole scene rather than a patch, because the effects that use this are
      // removals and a merge patch cannot express an absence.
      simulation.beginEdit(effect.label);
      useSceneStore.getState().setScene(effect.scene);
      simulation.commitEdit(undefined, { reseed: effect.reseed });
      return;
    }
    case "arm": {
      ui.setActiveTool(effect.tool);
      if (effect.shape) ui.setPlacementShape(effect.shape);
      if (effect.prop) ui.setPropShape(effect.prop);
      return;
    }
    case "place": {
      // autoRun false for the same reason the DRAG tool passes it: a solid the
      // user asked for and has not yet moved should not start the clock under
      // them. Carrying it does, on the first move.
      const created = simulation.addBodyAt(effect.shape, effect.point_m, { autoRun: false });
      if (created && effect.carry) ui.beginCarry(created.id, created.name);
      return;
    }
    case "carry": {
      const description = useSceneStore.getState().scene.rigidBodies
        .find((body) => body.id === effect.bodyId);
      if (!description) return;
      ui.selectBody(description.id);
      ui.beginCarry(description.id, description.name);
      return;
    }
    case "release": {
      if (ui.carry?.bodyId === effect.bodyId) ui.endCarry();
      simulation.dropBody(effect.bodyId);
      return;
    }
    case "select": {
      ui.select(effect.selection);
      if (effect.openControls) ui.setSelectionControlsOpen(true);
      return;
    }
    case "navigate": {
      if (services.navigate) services.navigate(effect.href);
      else if (typeof window !== "undefined") window.location.assign(effect.href);
      return;
    }
    case "import-scene": {
      importSceneFromFile();
      return;
    }
    case "open-overlay": {
      ui.setSceneOverlay(effect.overlay);
      return;
    }
    case "probe": {
      // Three calls, in this order, and none of them is redundant. Enabling
      // clears whatever pin the probe was holding, so the ask has to come after
      // it; and a probe already pinned to an *earlier* click would swallow the
      // ask entirely, because the viewport only consumes a request while the
      // probe is free — so the old pin is released first. What the ask carries
      // is the aim, which is the whole point of reaching a probe from the ring.
      const request = { aim: effect.aim };
      if (effect.probe === "ray") {
        ui.setPixelTraceEnabled(true);
        ui.setPixelTracePinned(false);
        ui.requestPixelTracePin(request);
      } else {
        ui.setFluidCellTraceEnabled(true);
        ui.setFluidCellTracePinned(false);
        ui.requestFluidCellTracePin(request);
      }
      return;
    }
  }
}
