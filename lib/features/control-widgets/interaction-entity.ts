import type { ActionDefinition } from "../../interaction";
import type { ControlWidget, WidgetChannel, WidgetProvider } from "../../framework/widgets";
import { keyLabel } from "../interactive/input-router";
import type { WidgetContext } from "./context";

const PREFIX = "entity:";
const TAKE = "take-control";
const RELEASE = "release-control";

function keyFor(action: ActionDefinition): string | undefined {
  const binding = action.bindings.find(candidate => candidate.device === "keyboard");
  return binding?.device === "keyboard" ? keyLabel(binding.code) : undefined;
}

/**
 * One pad per entity the host says can be controlled, built from what its
 * behaviours declare: a stick per axis2, a spring lever per axis1, a button
 * per button. Values join the keyboard in the pane's input router, so
 * sequencing and release-on-pause stay in one place.
 */
export const interactionEntityWidgets: WidgetProvider<WidgetContext> = {
  id: "interaction-entity",
  widgets({ interaction }) {
    if (!interaction) return [];
    const { pane, view, busy } = interaction;
    return view.controlCandidates.flatMap((candidate): ControlWidget[] => {
      const entity = view.document.entities.find(item => item.id === candidate.entityId);
      if (!entity) return [];
      const controlled = view.controlledEntityId === entity.id;
      const channels: WidgetChannel[] = controlled ? [
        ...entity.behaviors.flatMap(behavior => (pane.catalog.behaviorType(behavior.type)?.actions ?? [])
          // Pointer motion is a delta, not a held position: nothing on screen can hold it.
          .filter(action => !action.bindings.some(binding => binding.device === "pointer-motion"))
          .map((action): WidgetChannel => {
            const id = `${behavior.id}/${action.id}`;
            if (action.kind === "axis2") return { kind: "axis2", id, label: action.label, value: [0, 0] };
            if (action.kind === "axis1") return { kind: "axis1", id, label: action.label, min: -1, max: 1, value: 0, rest: 0 };
            return { kind: "button", id, label: action.label, pressed: false, shortcut: keyFor(action) };
          })),
        { kind: "action", id: RELEASE, label: "Release", hint: "Hand the keyboard back to the editor (Esc)" },
      ] : [{
        kind: "action", id: TAKE, label: "Take control", tone: "accent",
        disabled: busy || !candidate.availability.available,
        hint: candidate.availability.available ? `Drive ${entity.name} from here or the keyboard` : candidate.availability.reason,
      }];
      return [{ id: `${PREFIX}${entity.id}`, target: `entity:${entity.id}`, label: entity.name, channels, placement: { anchor: "bottom-left" } }];
    });
  },
  input({ interaction }, widgetId, channelId, value) {
    if (!interaction || !widgetId.startsWith(PREFIX)) return;
    const { pane } = interaction;
    const entityId = widgetId.slice(PREFIX.length);
    if (channelId === TAKE) { void pane.control(entityId); return; }
    if (channelId === RELEASE) { void pane.release(); return; }
    const [behaviorId, actionId] = channelId.split("/");
    if (!behaviorId || !actionId) return;
    if (pane.router.virtual(entityId, behaviorId, actionId, value)) pane.pumpInput();
  },
};
