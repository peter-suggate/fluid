/** Contract examples only: these declarations do not install simulation behavior. */
import { publicationPort } from "../../framework/ports";
import type { BehaviorPlugin, ConfigSchema, EditorMetadata, EntityKindPlugin, Transform } from "../contracts";
export const transform: Transform = { position_m: { x: 0, y: 1, z: 0 }, orientation: { w: 1, x: 0, y: 0, z: 0 } };
export const emptySchema: ConfigSchema = { version: 1, defaults: {}, validate: value => Object.keys(value).length ? ["Expected empty config"] : [] };
export const metadata: EditorMetadata = { label: "Example", hint: "Contract fixture", icon: "M0 0", group: "Characters", order: 0, controls: [] };
export const move: BehaviorPlugin = {
  id: "example.motor", schema: emptySchema, editor: metadata,
  provides: ["transform"], requiresHost: ["physics.character-queries"],
  phases: [{ phase: "intent" }],
  actions: [
    { id: "move", label: "Move", kind: "axis2", bindings: [
      { device: "keyboard", code: "KeyW", value: [0, 1] },
      { device: "keyboard", code: "KeyS", value: [0, -1] },
      { device: "keyboard", code: "KeyA", value: [-1, 0] },
      { device: "keyboard", code: "KeyD", value: [1, 0] },
    ] },
    { id: "jump", label: "Jump", kind: "button", bindings: [{ device: "keyboard", code: "Space", value: true }] },
  ],
};
export const follow: BehaviorPlugin = {
  id: "example.follow", schema: emptySchema, editor: metadata,
  requires: ["transform"], phases: [{ phase: "intent", after: ["transform"] }],
};
export const character: EntityKindPlugin = {
  id: "example.character",
  schema: {
    version: 2, defaults: { speed: 4 },
    validate: value => typeof value.speed === "number" && value.speed > 0 && Object.keys(value).length === 1 ? [] : ["speed must be positive"],
    migrate: (value, from) => {
      if (from !== 1) throw new Error("Unsupported migration");
      return { speed: value.walkSpeed };
    },
  },
  editor: { ...metadata, label: "Character", controls: [{ id: "speed", kind: "number", label: "Speed", setting: "speed", unit: "m/s", min: 0.1, update: "live" }] },
  placement: { alignment: "upright", overlap: "reject" }, requires: ["transform"],
  preview: () => ({ shape: "capsule", radius_m: 0.35, height_m: 1.8 }),
  compose: () => [
    { id: "camera", type: follow.id, version: 1, config: {} },
    { id: "motor", type: move.id, version: 1, config: {} },
  ],
};
export const signal = publicationPort<boolean>({ id: "active", representation: "boolean", lifetime: "frame" });
