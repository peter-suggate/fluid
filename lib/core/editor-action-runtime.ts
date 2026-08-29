"use client";

import {
  closeCompareMode,
  keepComparePaneB,
  swapComparePanesInPlace,
  toggleCompareMode,
} from "./compare/compare-mode";
import type { EditorActionEffect } from "./editor-action";
import { createInflowAt, INFLOW_SELECTION_ID } from "./editor-inflow";
import { simulation } from "./simulation/controller";
import { resolveSession, type PaneSession } from "./session/session";

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
  session: PaneSession = resolveSession(),
): void {
  const ui = session.ui.getState();
  switch (effect.kind) {
    case "scene": {
      // Whole scene rather than a patch, because the effects that use this are
      // removals and a merge patch cannot express an absence.
      simulation.beginEdit(effect.label, session.id);
      session.scene.getState().setScene(effect.scene);
      simulation.commitEdit(undefined, { reseed: effect.reseed }, session.id);
      return;
    }
    case "arm": {
      ui.setArmedGesture(effect.gesture);
      return;
    }
    case "place": {
      // autoRun false for the same reason the DRAG tool passes it: a solid the
      // user asked for and has not yet moved should not start the clock under
      // them. Carrying it does, on the first move.
      const created = simulation.addBodyAt(effect.shape, effect.point_m, { autoRun: false }, session.id);
      if (created && effect.carry) ui.beginCarry(created.id, created.name);
      return;
    }
    // The two placements that were modes. Both are single clicks at a point the
    // ring already carries, so neither needs the reader to say "where" twice.
    case "place-prop": {
      simulation.addScenery(effect.prop, effect.point_m, effect.normal, session.id);
      return;
    }
    case "place-inflow": {
      const sceneStore = session.scene.getState();
      simulation.beginEdit(sceneStore.scene.fluid.inflow ? "Moved the hose" : "Placed a hose", session.id);
      sceneStore.patchFluid({
        inflow: createInflowAt(effect.point_m, effect.normal, sceneStore.scene),
      });
      ui.select({ kind: "inflow", id: INFLOW_SELECTION_ID });
      // The nozzle is a boundary condition, so the run restarts from a defined
      // t=0 rather than continuing against a wall that changed under it.
      simulation.commitEdit(undefined, { reseed: true }, session.id);
      return;
    }
    case "carry": {
      const description = session.scene.getState().scene.rigidBodies
        .find((body) => body.id === effect.bodyId);
      if (!description) return;
      ui.selectBody(description.id);
      ui.beginCarry(description.id, description.name);
      return;
    }
    case "release": {
      if (ui.carry?.bodyId === effect.bodyId) ui.endCarry();
      simulation.dropBody(effect.bodyId, session.id);
      return;
    }
    case "select": {
      ui.select(effect.selection);
      if (effect.openControls) ui.setSelectionControlsOpen(true);
      return;
    }
    case "open-overlay": {
      ui.setSceneOverlay(effect.overlay);
      return;
    }
    case "compare": {
      // Every arm resolves against pane *A*, which is where the diff is applied
      // and where a promoted override has to land. The ring may have been
      // opened on either pane; the operation is about the pair.
      const paneA = resolveSession();
      if (effect.action === "toggle") toggleCompareMode();
      else if (effect.action === "close") closeCompareMode();
      else if (effect.action === "keep") keepComparePaneB(paneA);
      else swapComparePanesInPlace(paneA);
      return;
    }
    case "choose-scene": {
      // The session this ring was composed under, so the wedge chooses a scene
      // for the pane it was opened on rather than for pane A.
      ui.setSceneSelectorOpen(true);
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
