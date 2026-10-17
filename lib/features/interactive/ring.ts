import type { EditorAction, InteractiveRingRequest } from "../../core/editor-action";
import type { Vec3 } from "../../core/model";
import { interactionPane } from "./pane";

/**
 * The ring's share of interactive entities.
 *
 * Two compositions: an "Interactive" wedge on empty space whose children are
 * the catalog's kinds, in catalog order, each with the kind's own icon and
 * hint; and an entity's own ring. Both name kinds and entities by ID only —
 * nothing here switches on what a kind is.
 */

/** The wedge for a surface point, or nothing when the pane has no interaction binding. */
export function interactivePlacementWedge(paneId: string, point_m: Vec3, normal: Vec3): EditorAction | undefined {
  const pane = interactionPane(paneId);
  if (!pane) return undefined;
  const running = pane.state.view.running;
  const kinds = pane.catalog.entities;
  if (kinds.length === 0) return undefined;
  return {
    id: "interactive",
    label: "Interactive",
    tone: "body",
    iconPath: kinds[0]!.editor.icon,
    hint: running ? "Pause to place interactive entities" : "Place a playable entity here",
    children: kinds.map((kind) => ({
      id: `interactive:${kind.id}`,
      label: kind.editor.label,
      hint: kind.editor.hint,
      iconPath: kind.editor.icon,
      tone: "body",
      enabled: !running,
      effect: { kind: "interactive", request: { verb: "place", kindId: kind.id, point_m, normal } },
    })),
  };
}

/** What a placed entity offers when it is right-clicked. */
export function interactiveEntityActions(paneId: string, entityId: string): readonly EditorAction[] {
  const pane = interactionPane(paneId);
  const view = pane?.state.view;
  const entity = view?.document.entities.find((candidate) => candidate.id === entityId);
  if (!pane || !view || !entity) return [];
  const candidate = view.controlCandidates.find((item) => item.entityId === entityId);
  const actions: EditorAction[] = [{
    id: "interactive-inspect", label: "Inspect", tone: "body", icon: "edit",
    hint: `Select ${entity.name} and show its settings`,
    effect: { kind: "interactive", request: { verb: "select", entityId } },
  }];
  if (candidate) {
    actions.push({
      id: "interactive-control", label: "Control", tone: "body", iconPath: "M8 5v14l11-7z",
      enabled: candidate.availability.available,
      hint: candidate.availability.available ? `Play as ${entity.name}` : candidate.availability.reason,
      effect: { kind: "interactive", request: { verb: "control", entityId } },
    });
  }
  actions.push({
    id:'interactive-duplicate',label:'Duplicate',tone:'body',icon:'edit',enabled:!view.running,
    hint:'Copy this entity and remap its connections',effect:{kind:'interactive',request:{verb:'duplicate',entityId}},
  });
  actions.push({
    id: "interactive-remove", label: "Delete", tone: "danger", icon: "delete",
    enabled: !view.running,
    hint: view.running ? "Pause before changing the scene" : `Remove ${entity.name} and its connections`,
    effect: { kind: "interactive", request: { verb: "remove", entityId } },
  });
  return actions;
}

export function performInteractiveRequest(paneId: string, request: InteractiveRingRequest): void {
  const pane = interactionPane(paneId);
  if (!pane) return;
  switch (request.verb) {
    case "place":
      pane.startPlacement(request.kindId);
      pane.hoverPlacement({ point_m: request.point_m, normal: request.normal, fallback: false });
      return;
    case "select":
      pane.select(request.entityId);
      return;
    case "control":
      void pane.control(request.entityId);
      return;
    case "remove":
      void pane.remove(request.entityId);
      return;
    case "duplicate":
      void pane.duplicate(request.entityId);
      return;
  }
}
