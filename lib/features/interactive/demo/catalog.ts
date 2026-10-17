import { publicationPort } from "../../../framework/ports";
import {
  createInteractionCatalog,
  type BehaviorPlugin,
  type ConfigSchema,
  type EditorMetadata,
  type EntityKindPlugin,
  type JsonObject,
  type JsonValue,
} from "../../../interaction";

/**
 * UI demo declarations: four placeable kinds for driving the interactive UI
 * against the simulated host.
 *
 * These are *demo* entities, owned by the UI test double and mounted only
 * behind `?interactive=demo`. The production declarations, runtime factories
 * and Box3D adapter belong to the integration owner (see the UI handoff); the
 * catalog contract's own fixtures stay in `lib/interaction/verification` and
 * are never imported here. Nothing in the UI reads these IDs — it enumerates
 * the catalog — so replacing this file with the production catalog changes no
 * component.
 */

type FieldRule =
  | { readonly kind: "number"; readonly min: number; readonly max: number }
  | { readonly kind: "boolean" }
  | { readonly kind: "choice"; readonly options: readonly string[] };

/** An exact-key schema whose validator names every bad field; never coerces. */
function schema(defaults: JsonObject, rules: Readonly<Record<string, FieldRule>>): ConfigSchema {
  return {
    version: 1,
    defaults,
    validate(config) {
      const errors: string[] = [];
      for (const key of Object.keys(config)) if (!(key in rules)) errors.push(`Unknown setting ${key}`);
      for (const [key, rule] of Object.entries(rules)) {
        const value: JsonValue | undefined = config[key];
        if (rule.kind === "number") {
          if (typeof value !== "number" || !Number.isFinite(value)) errors.push(`${key} must be a number`);
          else if (value < rule.min || value > rule.max) errors.push(`${key} must be between ${rule.min} and ${rule.max}`);
        } else if (rule.kind === "boolean") {
          if (typeof value !== "boolean") errors.push(`${key} must be on or off`);
        } else if (typeof value !== "string" || !rule.options.includes(value)) {
          errors.push(`${key} must be one of ${rule.options.join(", ")}`);
        }
      }
      return errors;
    },
  };
}

const empty = schema({}, {});

function editor(label: string, hint: string, icon: string, group: string, order: number, controls: EditorMetadata["controls"] = []): EditorMetadata {
  return { label, hint, icon, group, order, controls };
}

/** 24-box stroke icons, in the ring's own idiom. */
const ICONS = {
  character: "M12 3.5a2.5 2.5 0 1 1 0 5a2.5 2.5 0 1 1 0-5M8.5 21v-6.5a3.5 3.5 0 0 1 7 0V21M8.5 13H6m9.5 0H18",
  crate: "M4 7.5 12 3l8 4.5v9L12 21l-8-4.5zM4 7.5l8 4.5 8-4.5M12 12v9",
  switch: "M7 20h10M9 20V9h6v11M12 9 15 3",
  platform: "M3 14h18v3H3zM7 17v3m10-3v3M9 10l3-3 3 3",
  motor: "M5 18c3-8 11-8 14 0M12 6v4",
  camera: "M3 8h12v8H3zM15 11l6-3v8l-6-3",
  body: "M4 8l8-4 8 4-8 4zM4 8v8l8 4 8-4V8",
  collider: "M4 4h16v16H4z",
  activator: "M12 3v6m0 6v6M3 12h6m6 0h6",
  mover: "M4 12h16M15 7l5 5-5 5",
};

/** One canonical channel: a switch output and a platform input share it. */
const active = publicationPort<boolean>({ id: "active", representation: "boolean", lifetime: "frame" });

const motor: BehaviorPlugin = {
  id: "demo.character-motor",
  schema: schema({ walkSpeed_m_s: 3.5, runSpeed_m_s: 6.5, jumpHeight_m: 1, turnRate_deg: 180 }, {
    walkSpeed_m_s: { kind: "number", min: 0.1, max: 12 },
    runSpeed_m_s: { kind: "number", min: 0.1, max: 20 },
    jumpHeight_m: { kind: "number", min: 0, max: 4 },
    turnRate_deg: { kind: "number", min: 10, max: 720 },
  }),
  editor: editor("Character motor", "Walks, runs and jumps relative to where the character faces", ICONS.motor, "Movement", 0, [
    { id: "walk", kind: "number", label: "Walk speed", setting: "walkSpeed_m_s", unit: "m/s", min: 0.1, max: 12, step: 0.1, update: "live" },
    { id: "run", kind: "number", label: "Run speed", setting: "runSpeed_m_s", unit: "m/s", min: 0.1, max: 20, step: 0.1, update: "live" },
    { id: "jump", kind: "number", label: "Jump height", setting: "jumpHeight_m", unit: "m", min: 0, max: 4, step: 0.1, update: "live" },
    { id: "turn", kind: "number", label: "Turn rate", setting: "turnRate_deg", unit: "°/unit", min: 10, max: 720, step: 10, update: "live" },
  ]),
  provides: ["transform", "character-motor"],
  requiresHost: ["physics.character-queries"],
  phases: [{ phase: "intent" }],
  actions: [
    { id: "move", label: "Move", kind: "axis2", bindings: [
      { device: "keyboard", code: "KeyW", value: [0, 1] },
      { device: "keyboard", code: "KeyS", value: [0, -1] },
      { device: "keyboard", code: "KeyA", value: [-1, 0] },
      { device: "keyboard", code: "KeyD", value: [1, 0] },
    ] },
    { id: "look", label: "Look", kind: "axis2", bindings: [{ device: "pointer-motion", scale: [0.004, 0.004] }] },
    { id: "jump", label: "Jump", kind: "button", bindings: [{ device: "keyboard", code: "Space", value: true }] },
    { id: "run", label: "Run", kind: "button", bindings: [
      { device: "keyboard", code: "ShiftLeft", value: true },
      { device: "keyboard", code: "ShiftRight", value: true },
    ] },
  ],
};

const followCamera: BehaviorPlugin = {
  id: "demo.follow-camera",
  schema: empty,
  editor: editor("Follow camera", "Third-person view behind the character", ICONS.camera, "Camera", 1),
  requires: ["transform"],
  phases: [{ phase: "intent", after: ["transform"] }],
};

const rigidBody: BehaviorPlugin = {
  id: "demo.rigid-body",
  schema: schema({ mass_kg: 20 }, { mass_kg: { kind: "number", min: 0.1, max: 500 } }),
  editor: editor("Rigid body", "A dynamic body the physics engine moves", ICONS.body, "Physics", 0, [
    { id: "mass", kind: "number", label: "Mass", setting: "mass_kg", unit: "kg", min: 0.1, max: 500, step: 1, update: "rebuild" },
  ]),
  provides: ["transform"],
  requiresHost: ["physics.rigid-bodies"],
  phases: [{ phase: "before-physics" }, { phase: "after-physics" }],
};

const collider: BehaviorPlugin = {
  id: "demo.static-collider",
  schema: empty,
  editor: editor("Static collider", "Holds the entity at its authored pose", ICONS.collider, "Physics", 1),
  provides: ["transform"],
  phases: [{ phase: "after-physics" }],
};

const activator: BehaviorPlugin = {
  id: "demo.activator",
  schema: schema({ mode: "toggle", startsOn: false }, {
    mode: { kind: "choice", options: ["toggle", "momentary"] },
    startsOn: { kind: "boolean" },
  }),
  editor: editor("Activator", "Something to press; drives connected inputs", ICONS.activator, "Interaction", 0, [
    { id: "mode", kind: "choice", label: "Mode", setting: "mode", update: "live", options: [
      { value: "toggle", label: "Toggle", hint: "Each press flips the output" },
      { value: "momentary", label: "Momentary", hint: "On while held" },
    ] },
    { id: "starts-on", kind: "toggle", label: "Starts on", setting: "startsOn", update: "reset" },
    { id: "state", kind: "readout", label: "Authored state", setting: "startsOn" },
  ]),
  requires: ["transform"],
  phases: [{ phase: "intent" }],
  actions: [{ id: "use", label: "Press", kind: "button", bindings: [{ device: "keyboard", code: "KeyE", value: true }] }],
  outputs: [active],
};

const mover: BehaviorPlugin = {
  id: "demo.kinematic-mover",
  schema: schema({ travel_m: 1.5, speed_m_s: 0.8, axis: "up" }, {
    travel_m: { kind: "number", min: 0, max: 20 },
    speed_m_s: { kind: "number", min: 0.05, max: 10 },
    axis: { kind: "choice", options: ["up", "forward", "right"] },
  }),
  editor: editor("Kinematic mover", "Travels while its active input is on, returns when it is off", ICONS.mover, "Motion", 0, [
    { id: "travel", kind: "number", label: "Travel", setting: "travel_m", unit: "m", min: 0, max: 20, step: 0.1, update: "live" },
    { id: "speed", kind: "number", label: "Speed", setting: "speed_m_s", unit: "m/s", min: 0.05, max: 10, step: 0.05, update: "live" },
    { id: "axis", kind: "choice", label: "Direction", setting: "axis", update: "reset", options: [
      { value: "up", label: "Up" }, { value: "forward", label: "Forward" }, { value: "right", label: "Right" },
    ] },
  ]),
  provides: ["transform"],
  requiresHost: ["physics.kinematic"],
  phases: [{ phase: "before-physics" }],
  inputs: [active],
};

const character: EntityKindPlugin = {
  id: "demo.character",
  schema: schema({ radius_m: 0.35, height_m: 1.8 }, {
    radius_m: { kind: "number", min: 0.15, max: 1 },
    height_m: { kind: "number", min: 0.6, max: 3 },
  }),
  editor: editor("Character", "A capsule you can take control of", ICONS.character, "Characters", 0, [
    { id: "height", kind: "number", label: "Height", setting: "height_m", unit: "m", min: 0.6, max: 3, step: 0.05, update: "rebuild" },
    { id: "radius", kind: "number", label: "Radius", setting: "radius_m", unit: "m", min: 0.15, max: 1, step: 0.05, update: "rebuild" },
  ]),
  placement: { alignment: "upright", overlap: "reject" },
  requires: ["transform"],
  preview: (config) => {
    const radius_m = config.radius_m as number, height_m = config.height_m as number;
    return { shape: "capsule", radius_m, height_m: Math.max(height_m, 2 * radius_m) };
  },
  compose: () => [
    { id: "motor", type: motor.id, version: motor.schema.version, config: motor.schema.defaults },
    { id: "camera", type: followCamera.id, version: followCamera.schema.version, config: followCamera.schema.defaults },
  ],
};

const crate: EntityKindPlugin = {
  id: "demo.crate",
  schema: schema({ size_m: 0.6 }, { size_m: { kind: "number", min: 0.1, max: 3 } }),
  editor: editor("Crate", "A box to push around", ICONS.crate, "Props", 10, [
    { id: "size", kind: "number", label: "Size", setting: "size_m", unit: "m", min: 0.1, max: 3, step: 0.05, update: "rebuild" },
  ]),
  placement: { alignment: "surface", overlap: "reject" },
  requires: ["transform"],
  preview: (config) => {
    const s = config.size_m as number;
    return { shape: "box", dimensions_m: { x: s, y: s, z: s } };
  },
  compose: () => [{ id: "body", type: rigidBody.id, version: rigidBody.schema.version, config: rigidBody.schema.defaults }],
};

const switchKind: EntityKindPlugin = {
  id: "demo.switch",
  schema: empty,
  editor: editor("Switch", "Press it to drive what it is connected to", ICONS.switch, "Mechanisms", 20),
  placement: { alignment: "upright", overlap: "reject" },
  requires: ["transform"],
  preview: () => ({ shape: "box", dimensions_m: { x: 0.3, y: 0.9, z: 0.2 } }),
  compose: () => [
    { id: "collider", type: collider.id, version: collider.schema.version, config: collider.schema.defaults },
    { id: "activator", type: activator.id, version: activator.schema.version, config: activator.schema.defaults },
  ],
};

const platform: EntityKindPlugin = {
  id: "demo.platform",
  schema: schema({ width_m: 2, depth_m: 2 }, {
    width_m: { kind: "number", min: 0.3, max: 10 },
    depth_m: { kind: "number", min: 0.3, max: 10 },
  }),
  editor: editor("Platform", "A slab that travels while its input is active", ICONS.platform, "Mechanisms", 21, [
    { id: "width", kind: "number", label: "Width", setting: "width_m", unit: "m", min: 0.3, max: 10, step: 0.1, update: "rebuild" },
    { id: "depth", kind: "number", label: "Depth", setting: "depth_m", unit: "m", min: 0.3, max: 10, step: 0.1, update: "rebuild" },
  ]),
  placement: { alignment: "upright", overlap: "allow" },
  requires: ["transform"],
  preview: (config) => ({ shape: "box", dimensions_m: { x: config.width_m as number, y: 0.2, z: config.depth_m as number } }),
  compose: () => [{ id: "mover", type: mover.id, version: mover.schema.version, config: mover.schema.defaults }],
};

export const DEMO_BEHAVIORS: readonly BehaviorPlugin[] = [motor, followCamera, rigidBody, collider, activator, mover];
export const DEMO_ENTITY_KINDS: readonly EntityKindPlugin[] = [character, crate, switchKind, platform];

/** What the simulated host claims to provide: every capability the demo behaviors ask for. */
export const DEMO_HOST_CAPABILITIES: readonly string[] = ["physics.character-queries", "physics.rigid-bodies", "physics.kinematic"];

export function createDemoInteractionCatalog() {
  return createInteractionCatalog({ entities: DEMO_ENTITY_KINDS, behaviors: DEMO_BEHAVIORS });
}
