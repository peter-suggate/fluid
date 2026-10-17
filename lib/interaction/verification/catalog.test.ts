import assert from "node:assert/strict";
import test from "node:test";
import { createInteractionCatalog } from "../index";
import type { BehaviorPlugin, InteractionDocument } from "../index";
import { character, emptySchema, follow, metadata, move, signal, transform } from "./fixtures";

const host = ["physics.character-queries"];
function catalog(behaviors: readonly BehaviorPlugin[] = [follow, move]) {
  return createInteractionCatalog({ entities: [character], behaviors });
}
function document(): InteractionDocument {
  return { schemaVersion: 1, entities: [catalog().createEntity({ id: "player", kind: character.id, transform })], connections: [] };
}
function copy<T>(value: T): T { return JSON.parse(JSON.stringify(value)) as T; }

test("creates a valid recipe and exposes immutable editor metadata without a runtime", () => {
  const registry = catalog();
  const entity = registry.createEntity({ id: "player", kind: character.id, transform });
  assert.equal(entity.name, "Character");
  assert.equal(entity.version, 2);
  assert.deepEqual(entity.config, { speed: 4 });
  assert.equal(registry.entityKind(entity.kind)?.editor.controls[0].setting, "speed");
  assert.ok(Object.isFrozen(entity.transform.position_m));
  assert.ok(Object.isFrozen(registry.entityKind(entity.kind)?.editor.controls));
  assert.deepEqual(registry.planDocument(document(), host).phases.intent.map(item => item.behaviorId), ["motor", "camera"]);
});

test("saved composition is authoritative and recipes are not rerun during load", () => {
  const registry = createInteractionCatalog({ entities: [{ ...character, compose: () => { throw new Error("Must not run"); } }], behaviors: [move, follow] });
  assert.equal(registry.planDocument(document(), host).document.entities.length, 1);
});

test("validates configuration and geometry before admitting creation", () => {
  assert.throws(() => catalog().createEntity({ id: "player", kind: character.id, transform, config: { speed: -1 } }), /speed must be positive/);
  assert.throws(() => catalog().createEntity({ id: "player", kind: character.id, transform: { ...transform, orientation: { w: 2, x: 0, y: 0, z: 0 } } }), /unit quaternion/);
  assert.throws(() => catalog().createEntity({ id: "player", kind: "missing", transform }), /Unknown entity kind/);
});

test("migrates old config once without modifying source data", () => {
  const original = document();
  const old = { ...original, entities: original.entities.map(entity => ({ ...entity, version: 1, config: { walkSpeed: 3 } })) };
  const result = catalog().inspectDocument(old);
  assert.deepEqual(result.document.entities[0].config, { speed: 3 });
  assert.deepEqual(old.entities[0].config, { walkSpeed: 3 });
  assert.deepEqual(catalog().inspectDocument(result.document), result);
});

test("requires valid migration output and does not silently substitute defaults", () => {
  const old = { ...document(), entities: document().entities.map(entity => ({ ...entity, version: 1, config: { walkSpeed: 3 } })) };
  const missing = createInteractionCatalog({ entities: [{ ...character, schema: { ...character.schema, migrate: undefined } }], behaviors: [move, follow] });
  assert.throws(() => missing.inspectDocument(old), /Missing migration/);
  const invalid = createInteractionCatalog({ entities: [{ ...character, schema: { ...character.schema, migrate: () => ({ speed: -10 }) } }], behaviors: [move, follow] });
  assert.throws(() => invalid.inspectDocument(old), /speed must be positive/);
});

test("preserves missing and newer plugin records for round trip but blocks execution", () => {
  const raw = document();
  const unknown = { ...raw, entities: raw.entities.map(entity => ({ ...entity, kind: "external.avatar", config: { asset: "avatar-v3" }, behaviors: [{ id: "custom", type: "external.flight", version: 8, config: { lift: 2 } }] })) };
  const inspected = catalog().inspectDocument(unknown);
  assert.deepEqual(inspected.document, unknown);
  assert.equal(inspected.unavailable.length, 2);
  assert.throws(() => catalog().planDocument(inspected.document, host), /Unavailable plugins/);
  const future = { ...raw, entities: raw.entities.map(entity => ({ ...entity, version: 99 })) };
  assert.deepEqual(catalog().inspectDocument(future).document, future);
  assert.throws(() => catalog().planDocument(future, host), /Unavailable plugins/);
});

test("rejects malformed documents, non-JSON values and duplicate stable IDs", () => {
  assert.throws(() => catalog().inspectDocument({ schemaVersion: 2, entities: [], connections: [] }), /document schema/);
  assert.throws(() => catalog().inspectDocument({ ...document(), extra: () => 1 }), /JSON/);
  assert.throws(() => catalog().inspectDocument({ ...document(), extra: new Date() }), /plain JSON/);
  assert.throws(() => catalog().inspectDocument({ ...document(), extra: Infinity }), /finite/);
  assert.throws(() => catalog().inspectDocument({ ...document(), extra: new Array(2) }), /Sparse/);
  const extended = Object.assign([1], { extra: 2 });
  assert.throws(() => catalog().inspectDocument({ ...document(), extra: extended }), /extended/);
  assert.throws(() => catalog().inspectDocument({ ...document(), [Symbol("lost")]: 1 }), /Symbol keys/);
  const accessor = Object.defineProperty({}, "value", { enumerable: true, get() { throw new Error("Must not execute"); } });
  assert.throws(() => catalog().inspectDocument({ ...document(), extra: accessor }), /JSON data property/);
  const cyclic: Record<string, unknown> = {}; cyclic.self = cyclic;
  assert.throws(() => catalog().inspectDocument(cyclic), /Cyclic/);
  const raw = document();
  assert.throws(() => catalog().inspectDocument({ ...raw, entities: [...raw.entities, ...raw.entities] }), /Duplicate entity/);
  const entity = raw.entities[0];
  assert.throws(() => catalog().inspectDocument({ ...raw, entities: [{ ...entity, behaviors: [...entity.behaviors, ...entity.behaviors] }] }), /Duplicate behavior instance/);
});

test("validates plugin declarations before a host allocates resources", () => {
  assert.throws(() => createInteractionCatalog({ entities: [character, character], behaviors: [] }), /Duplicate entity kind/);
  assert.throws(() => catalog([move, move]), /Duplicate behavior type/);
  assert.throws(() => catalog([{ ...move, provides: ["transform", "transform"] }]), /Duplicate provided/);
  assert.throws(() => catalog([{ ...move, phases: [{ phase: "intent" }, { phase: "intent" }] }]), /Duplicate phase/);
  assert.throws(() => createInteractionCatalog({ entities: [{ ...character, editor: { ...metadata, controls: [{ id: "oops", kind: "number", label: "Oops", setting: "unknown", update: "live" }] } }], behaviors: [move, follow] }), /Unknown config setting/);
});

test("detects competing transform ownership and missing entity capabilities", () => {
  assert.throws(() => catalog([{ ...follow, provides: ["transform"] }, move]).planDocument(document(), host), /Competing owners/);
  assert.throws(() => catalog([{ ...follow, requires: ["flight"] }, move]).planDocument(document(), host), /missing capability flight/);
  assert.throws(() => catalog().planDocument(document()), /Missing host capability/);
});

test("execution order is stable across catalog and entity insertion order", () => {
  const registry = catalog([move, follow]);
  const a = registry.createEntity({ id: "a", kind: character.id, transform });
  const z = registry.createEntity({ id: "z", kind: character.id, transform });
  const plan = registry.planDocument({ schemaVersion: 1, entities: [z, a], connections: [] }, host);
  assert.deepEqual(plan.phases.intent.map(step => `${step.entityId}/${step.behaviorId}`), ["a/motor", "a/camera", "z/motor", "z/camera"]);
  assert.ok(Object.isFrozen(plan.phases.intent));
});

test("rejects cyclic scheduling and dependencies absent in the requested phase", () => {
  const loop = [{ ...follow, provides: ["camera"] }, { ...move, phases: [{ phase: "intent" as const, after: ["camera"] }] }];
  assert.throws(() => catalog(loop).planDocument(document(), host), /dependency cycle/);
  assert.throws(() => catalog([follow, { ...move, phases: [{ phase: "before-physics" }] }]).planDocument(document(), host), /missing intent provider/);
});

test("validates connection endpoints, representations, lifetimes and units", () => {
  const source = { ...move, outputs: [signal] };
  const target = { ...follow, inputs: [signal] };
  const raw = { ...document(), connections: [{ id: "link", source: { entity: "player", behavior: "motor", port: "active" }, target: { entity: "player", behavior: "camera", port: "active" } }] };
  assert.equal(catalog([source, target]).planDocument(raw, host).document.connections.length, 1);
  for (const port of [{ ...signal, representation: "number" }, { ...signal, lifetime: "scene" as const }, { ...signal, unit: "m" }]) {
    assert.throws(() => catalog([source, { ...target, inputs: [port] }]).planDocument(raw, host), /Incompatible connection/);
  }
  assert.throws(() => catalog([source, target]).planDocument({ ...raw, entities: [] }, host), /Dangling connection/);
  assert.throws(() => catalog([source, target]).planDocument({ ...raw, connections: [...raw.connections, ...raw.connections] }, host), /Duplicate connection/);
});

test("copying caller-owned definitions and documents prevents post-validation mutation", () => {
  const controls = [{ id: "speed", kind: "number" as const, label: "Speed", setting: "speed", update: "live" as const }];
  const registry = createInteractionCatalog({ entities: [{ ...character, editor: { ...metadata, controls } }], behaviors: [move, follow] });
  controls[0].label = "Changed";
  assert.equal(registry.entities[0].editor.controls[0].label, "Speed");
  const original = copy(document());
  const inspected = registry.inspectDocument(original);
  (original.entities[0].config as { speed: number }).speed = -1;
  assert.equal(inspected.document.entities[0].config.speed, 4);
});

test("an additional kind is composed without changes to the catalog or editor host", () => {
  const crate = { ...character, id: "example.crate", schema: emptySchema, requires: [], editor: { ...metadata, label: "Crate" }, compose: () => [] };
  const registry = createInteractionCatalog({ entities: [character, crate], behaviors: [move, follow] });
  const entity = registry.createEntity({ id: "crate", kind: crate.id, transform });
  const plan = registry.planDocument({ schemaVersion: 1, entities: [entity], connections: [] });
  assert.equal(plan.document.entities[0].kind, "example.crate");
  assert.deepEqual(plan.requiredHostCapabilities, []);
});

test("logical input validates the target, kind, normalized axes and button edges", () => {
  const registry = catalog();
  const raw = document();
  const sample = { entityId: "player", behaviorId: "motor", actionId: "move", value: [0.5, 1] as const, sequence: 1 };
  registry.validateAction(raw, sample);
  registry.validateAction(raw, { ...sample, actionId: "jump", value: true, edge: "pressed" });
  assert.throws(() => registry.validateAction(raw, { ...sample, entityId: "deleted" }), /Unknown input action target/);
  assert.throws(() => registry.validateAction(raw, { ...sample, value: [2, 0] }), /Input value/);
  assert.throws(() => registry.validateAction(raw, { ...sample, value: true }), /Input value/);
  assert.throws(() => registry.validateAction(raw, { ...sample, edge: "pressed" }), /Invalid button edge/);
  assert.throws(() => registry.validateAction(raw, { ...sample, actionId: "jump", value: false, edge: "pressed" }), /Invalid button edge/);
  assert.throws(() => registry.validateAction(raw, { ...sample, sequence: -1 }), /input sequence/);
});

test("placement previews and binding contributions are validated at composition", () => {
  assert.throws(() => createInteractionCatalog({ entities: [{ ...character, preview: () => ({ shape: "capsule", radius_m: 1, height_m: 1 }) }], behaviors: [move, follow] }), /capsule preview/);
  assert.throws(() => catalog([{ ...move, actions: [{ id: "jump", label: "Jump", kind: "button", bindings: [{ device: "keyboard", code: "Space", value: [1, 0] }] }] }]), /Binding value/);
});
