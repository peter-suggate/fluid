import assert from "node:assert/strict";
import test from "node:test";
import {
  addProp,
  createPropAt,
  findProp,
  nextPropId,
  propIdFromSelection,
  propSelectionId,
  removeProp,
  scaleProp,
  updateProp,
  PROP_MINIMUM_HALF_SIZE_M,
} from "../lib/editor-props";
import { canonicalScene, cloneScene, defaultScene, parseScene, serializeScene, validateScene, type SceneDescription } from "../lib/model";
import { buildEnvironmentProxyCatalog, environmentProxyPrimitives } from "../lib/voxel-environments";

function scene(): SceneDescription {
  return cloneScene(defaultScene);
}

test("a prop rests on the surface it was placed on", () => {
  const base = scene();
  const prop = createPropAt(base, "box", { x: 0.1, y: 0, z: -0.2 }, { x: 0, y: 1, z: 0 });
  assert.ok(Math.abs(prop.position_m.y - prop.halfSize_m.y) < 1e-12, "the body is lifted by its own half height");
  assert.equal(prop.shape, "box");
  assert.deepEqual(validateScene({ ...base, props: [prop] }), []);
});

test("prop ids are unique and round-trip through selection ids", () => {
  const base = scene();
  const first = createPropAt(base, "box", { x: 0, y: 0, z: 0 }, { x: 0, y: 1, z: 0 });
  const withOne = { ...base, props: addProp(base, first) };
  assert.notEqual(nextPropId(withOne, "box"), first.id);
  assert.equal(propIdFromSelection(propSelectionId(first.id)), first.id);
  assert.equal(propIdFromSelection("body-sphere-1"), undefined);
  assert.equal(findProp(withOne, first.id)?.id, first.id);
});

test("removing the last prop drops the field instead of leaving an empty array", () => {
  const base = scene();
  const prop = createPropAt(base, "cylinder", { x: 0, y: 0, z: 0 }, { x: 0, y: 1, z: 0 });
  const withOne = { ...base, props: addProp(base, prop) };
  const second = createPropAt(withOne, "ellipsoid", { x: 0.2, y: 0, z: 0 }, { x: 0, y: 1, z: 0 });
  const withTwo = { ...withOne, props: addProp(withOne, second) };

  assert.equal(removeProp(withTwo, prop.id)?.length, 1);
  assert.equal(removeProp(withOne, prop.id), undefined, "an emptied list becomes undefined");
});

test("updates and uniform scaling stay authorable", () => {
  const base = scene();
  const prop = createPropAt(base, "box", { x: 0, y: 0, z: 0 }, { x: 0, y: 1, z: 0 });
  const withOne = { ...base, props: addProp(base, prop) };
  const moved = updateProp(withOne, prop.id, { position_m: { x: 0.3, y: 0.2, z: 0.1 } });
  assert.deepEqual(moved[0]?.position_m, { x: 0.3, y: 0.2, z: 0.1 });
  assert.deepEqual(validateScene({ ...withOne, props: moved }), []);

  const shrunk = scaleProp(prop, 1e-9);
  assert.equal(shrunk.halfSize_m?.x, PROP_MINIMUM_HALF_SIZE_M, "a prop cannot be scaled out of existence");
  assert.deepEqual(validateScene({ ...withOne, props: updateProp(withOne, prop.id, shrunk) }), []);
});

test("props round-trip byte-identically through the scene document", () => {
  const base = scene();
  const prop = createPropAt(base, "ellipsoid", { x: 0.1, y: 0, z: 0.1 }, { x: 0, y: 1, z: 0 });
  const authored = { ...base, props: addProp(base, prop) };
  const serialized = serializeScene(authored);
  assert.equal(serializeScene(parseScene(serialized)), serialized);
  assert.equal(canonicalScene(parseScene(serialized)), canonicalScene(authored));
});

test("malformed props are rejected by the scene contract", () => {
  const base = scene();
  const prop = createPropAt(base, "box", { x: 0, y: 0, z: 0 }, { x: 0, y: 1, z: 0 });
  const invalid = (patch: Record<string, unknown>) => validateScene({ ...base, props: [{ ...prop, ...patch }] } as SceneDescription);
  assert.ok(invalid({ shape: "torus" }).some((error) => error.includes("Unsupported prop shape")));
  assert.ok(invalid({ halfSize_m: { x: 0, y: 1, z: 1 } }).some((error) => error.includes("half size must be positive")));
  assert.ok(invalid({ colorLinear: [1, 2] }).some((error) => error.includes("colour")));
  assert.ok(invalid({ position_m: { x: Number.NaN, y: 0, z: 0 } }).some((error) => error.includes("position must be finite")));
  assert.ok(invalid({ emission: -1 }).some((error) => error.includes("emission")));
  assert.ok(validateScene({ ...base, props: [prop, { ...prop }] } as SceneDescription).some((error) => error.includes("unique")));
});

test("authored props reach the render catalog and stay out of the solve", () => {
  const base = scene();
  const prop = createPropAt(base, "cylinder", { x: 0.1, y: 0, z: 0.1 }, { x: 0, y: 1, z: 0 });
  const authored = { ...base, props: addProp(base, prop) };

  const before = buildEnvironmentProxyCatalog(base, "default");
  const after = buildEnvironmentProxyCatalog(authored, "default");
  assert.equal(after.primitives.length, before.primitives.length + 1, "the prop appends to the procedural catalog");

  const added = after.primitives.find((primitive) => primitive.key.includes(prop.id));
  assert.ok(added, "the prop is present and keyed by its id");
  assert.equal(added.kind, "cylinder");
  assert.ok(added.tags.includes("authored"));
  assert.ok(environmentProxyPrimitives(after).some((primitive) => primitive.key === added.key));

  // Owner indices must stay a single dense sequence shared with procedural scenery.
  const owners = after.primitives.map((primitive) => primitive.ownerIndex);
  assert.equal(new Set(owners).size, owners.length, "owner indices remain unique");

  // Props are render-only: nothing about the fluid or rigid state changed.
  assert.deepEqual(authored.rigidBodies, base.rigidBodies);
  assert.deepEqual(authored.fluid, base.fluid);
});
