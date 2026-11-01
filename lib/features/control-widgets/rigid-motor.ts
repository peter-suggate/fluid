import type { RigidBodyDescription } from "../../core/model";
import { simulation } from "../../core/simulation/controller";
import type { ControlWidget, WidgetProvider } from "../../framework/widgets";
import type { WidgetContext } from "./context";

const PREFIX = "motor:";
const STEP = 5;

/**
 * sRGB of each hull's paint (paddleboat-render's linear paint), so a throttle
 * says which boat it drives without a legend.
 */
const APPEARANCE_ACCENT: Partial<Record<NonNullable<RigidBodyDescription["appearance"]>, string>> = {
  "paddleboat-red": "rgb(184 66 53)",
  "paddleboat-teal": "rgb(55 112 116)",
};

/** One throttle per free body that carries motors: the drive is the body's, not each fixture's. */
export const rigidMotorWidgets: WidgetProvider<WidgetContext> = {
  id: "rigid-motor",
  widgets({ scene }) {
    const owners = new Set(scene.rigidBodies.flatMap(body => body.motorAttachment ? [body.motorAttachment.parentId] : []));
    return scene.rigidBodies.filter(body => owners.has(body.id) && !body.motorAttachment).map((body): ControlWidget => ({
      id: `${PREFIX}${body.id}`,
      target: `body:${body.id}`,
      label: body.name.split(" ·")[0]!,
      accent: body.appearance && APPEARANCE_ACCENT[body.appearance],
      // Two-way and latching: the wheel keeps the speed it was left at, backwards below zero.
      channels: [{ kind: "axis1", id: "strength", label: "Motor", min: -100, max: 100, step: STEP, unit: "%",
        value: Math.round((body.motorStrength ?? 1) * 100), zeroLabel: "Stop" }],
      placement: { anchor: "bottom" },
    }));
  },
  input({ paneId }, widgetId, channelId, value) {
    if (channelId !== "strength" || typeof value !== "number" || !widgetId.startsWith(PREFIX)) return;
    const bodyId = widgetId.slice(PREFIX.length);
    // Read the store, not the render's context: input arrives between renders.
    const scene = simulation.session(paneId).scene.getState().scene;
    const body = scene.rigidBodies.find(candidate => candidate.id === bodyId);
    if (!body) return;
    const motorStrength = Math.max(-100, Math.min(100, Math.round(value / STEP) * STEP)) / 100;
    // Off the 5 % grid (the scene's authored third) counts as unchanged until the lever really moves.
    if (Math.round((body.motorStrength ?? 1) * 100) === Math.round(motorStrength * 100)) return;
    simulation.applyRuntimeControl({
      rigidBodies: scene.rigidBodies.map(candidate => candidate.id === bodyId ? { ...candidate, motorStrength } : candidate),
    }, paneId);
  },
};
