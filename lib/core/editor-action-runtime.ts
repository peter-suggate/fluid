"use client";

import type { EditorActionEffect } from "./editor-action";
import { simulation } from "./simulation/controller";
import { useSceneStore } from "./stores/scene-store";
import { useUIStore } from "./stores/ui-store";

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
export function performEditorAction(effect: EditorActionEffect): void {
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
  }
}
