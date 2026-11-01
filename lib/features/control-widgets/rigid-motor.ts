import { rigidMotorChannels, rigidMotorStrength, rigidMotorStrengthPatch } from "../../core/rigid-motor";
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

/** Group independently reversible wheel throttles under their free hull. */
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
      channels: rigidMotorChannels(scene.rigidBodies, body.id).map(channel => ({
        kind: "axis1", id: channel === undefined ? "strength" : `strength:${channel}`,
        label: channel === undefined ? "Motor" : `${channel[0]!.toUpperCase()}${channel.slice(1)} paddle`,
        min: -100, max: 100, step: STEP, unit: "%",
        value: Math.round(rigidMotorStrength(body, channel) * 100), zeroLabel: "Stop" })),
      placement: { anchor: "bottom" },
    }));
  },
  input({ paneId }, widgetId, channelId, value) {
    if ((channelId !== "strength" && !channelId.startsWith("strength:")) || typeof value !== "number" || !Number.isFinite(value) || !widgetId.startsWith(PREFIX)) return;
    const bodyId = widgetId.slice(PREFIX.length);
    // Read the store, not the render's context: input arrives between renders.
    const scene = simulation.session(paneId).scene.getState().scene;
    const body = scene.rigidBodies.find(candidate => candidate.id === bodyId);
    if (!body) return;
    const channel = channelId === "strength" ? undefined : channelId.slice("strength:".length);
    if (!rigidMotorChannels(scene.rigidBodies, bodyId).includes(channel)) return;
    const motorStrength = Math.max(-100, Math.min(100, Math.round(value / STEP) * STEP)) / 100;
    // Off the 5 % grid (the scene's authored third) counts as unchanged until the lever really moves.
    if (Math.round(rigidMotorStrength(body, channel) * 100) === Math.round(motorStrength * 100)) return;
    simulation.applyRuntimeControl({
      rigidBodies: scene.rigidBodies.map(candidate => candidate.id === bodyId ? { ...candidate, ...rigidMotorStrengthPatch(body, channel, motorStrength) } : candidate),
    }, paneId);
  },
};
