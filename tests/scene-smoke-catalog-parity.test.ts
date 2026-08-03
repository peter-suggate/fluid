import assert from "node:assert/strict";
import test from "node:test";
import { canonicalScene, validateScene, type SceneDescription } from "../lib/model";
import { sceneDocument } from "../lib/scene-definition";
import { SCENE_CATALOG, findSceneDefinition, getScenePreset } from "../lib/scenes";
import { getSceneWebGPUSmokeSuite, sceneWebGPUSmokeIds } from "../lib/scene-webgpu-smoke-catalog";

/**
 * The exact delta each GPU lane's variant is allowed to make, keyed by
 * `<scene>/<variant>` and pinned by value.
 *
 * These are the only differences that survived collapsing eleven forked smoke
 * factories onto the catalog. Every one of them changes what its lane measures
 * — a step cap, a capillary force, a body count, a fill depth — so it is stated
 * rather than inherited, and a variant that grows a field the lane never asked
 * for fails here instead of quietly re-baselining a Dawn gate.
 */
const VARIANT_DELTAS: Readonly<Record<string, Readonly<Record<string, unknown>>>> = {
  "water-box-tank-fill/gpu-smoke": {
    "container.fillFraction": 0.7,
    "fluid.surfaceTension_N_m": 0,
    "numerics.fixedDt_s": 1 / 120,
    "numerics.maxDt_s": 1 / 120,
    rigidBodies: [],
  },
  "deep-water-ab/gpu-smoke": {
    "numerics.maxDt_s": 1 / 30,
  },
  // Removing an object removes every leaf under it, which is why shutting the
  // hose off is eleven entries rather than one. Spelled out rather than
  // special-cased: the whole point of this table is that a lane cannot change
  // anything it has not written down.
  "hero-garden-hose/gpu-smoke": {
    "fluid.inflow.center_m.x": undefined,
    "fluid.inflow.center_m.y": undefined,
    "fluid.inflow.center_m.z": undefined,
    "fluid.inflow.radius_m": undefined,
    "fluid.inflow.length_m": undefined,
    "fluid.inflow.velocity_m_s.x": undefined,
    "fluid.inflow.velocity_m_s.y": undefined,
    "fluid.inflow.velocity_m_s.z": undefined,
    "fluid.inflow.start_s": undefined,
    "fluid.inflow.end_s": undefined,
    "fluid.inflow.ramp_s": undefined,
    "fluid.surfaceTension_N_m": 0,
    "numerics.fixedDt_s": 1 / 120,
    "numerics.maxDt_s": 1 / 120,
  },
  "garden-pond/gpu-smoke": {
    "fluid.surfaceTension_N_m": 0,
    "numerics.fixedDt_s": 1 / 120,
    "numerics.maxDt_s": 1 / 120,
    rigidBodies: [],
  },
  "garden-dam-break/gpu-smoke": {
    "container.fluidWallMode": "no-slip",
    "numerics.fixedDt_s": 0.004,
    "numerics.maxDt_s": 0.004,
    rigidBodies: [],
  },
  "brick-quad-dam-break/gpu-smoke": {
    "fluid.surfaceTension_N_m": 0,
    "numerics.maxDt_s": 0.004,
  },
  "twin-dam-collision/gpu-smoke": {
    "fluid.surfaceTension_N_m": 0,
    "numerics.maxDt_s": 0.004,
  },
  "ocean-seiche/gpu-smoke": {
    "numerics.fixedDt_s": 0.005,
    "numerics.maxDt_s": 0.005,
  },
};

/** Leaf paths of a document. Arrays stay whole so `rigidBodies: []` is one delta. */
function leaves(scene: SceneDescription): Map<string, unknown> {
  const out = new Map<string, unknown>();
  const walk = (value: unknown, path: string): void => {
    if (value !== null && typeof value === "object" && !Array.isArray(value)) {
      for (const [key, child] of Object.entries(value)) walk(child, path ? `${path}.${key}` : key);
      return;
    }
    out.set(path, value);
  };
  walk(scene, "");
  return out;
}

function changedLeaves(base: SceneDescription, applied: SceneDescription): Record<string, unknown> {
  const before = leaves(base), after = leaves(applied), delta: Record<string, unknown> = {};
  for (const key of new Set([...before.keys(), ...after.keys()])) {
    if (JSON.stringify(before.get(key)) !== JSON.stringify(after.get(key))) delta[key] = after.get(key);
  }
  return delta;
}

function suiteDefinition(id: typeof sceneWebGPUSmokeIds[number]) {
  const { scene } = getSceneWebGPUSmokeSuite(id);
  const definition = findSceneDefinition(scene.definitionId);
  assert.ok(definition, `${id} must name a catalog scene, not a local factory`);
  return { definition: definition!, variantId: scene.variantId };
}

test("every smoke suite runs a document the schema accepts", () => {
  for (const id of sceneWebGPUSmokeIds) {
    assert.deepEqual(validateScene(getSceneWebGPUSmokeSuite(id).createScene()), [],
      `${id} must construct a valid document`);
  }
});

test("every smoke suite owns a scenery graph rather than naming a background", () => {
  // The bug that motivated the collapse: the local factories assigned
  // `environment` as a bare string, which names a backdrop without copying the
  // graph it implies, so eleven GPU lanes rendered an empty world under the
  // product's own scene names.
  for (const id of sceneWebGPUSmokeIds) {
    const scene = getSceneWebGPUSmokeSuite(id).createScene();
    const { definition } = suiteDefinition(id);
    assert.equal(scene.environment, definition.environment, `${id} must carry its scene's environment`);
    assert.ok(scene.scenery, `${id} must carry a scenery graph`);
    assert.ok(scene.scenery!.nodes.length > 0, `${id} scenery graph must not be empty`);
  }
});

test("a suite with no variant is the product document, byte for byte", () => {
  for (const id of sceneWebGPUSmokeIds) {
    const { definition, variantId } = suiteDefinition(id);
    if (variantId) continue;
    assert.equal(canonicalScene(getSceneWebGPUSmokeSuite(id).createScene()),
      canonicalScene(getScenePreset(definition.id).create()),
      `${id} must be the same document the library opens`);
  }
});

test("each variant changes exactly the fields its lane declares", () => {
  for (const id of sceneWebGPUSmokeIds) {
    const { definition, variantId } = suiteDefinition(id);
    if (!variantId) continue;
    const key = `${definition.id}/${variantId}`;
    const expected = VARIANT_DELTAS[key];
    assert.ok(expected, `${key} must declare the delta it is allowed to make`);
    assert.deepEqual(changedLeaves(sceneDocument(definition), sceneDocument(definition, variantId)), expected,
      `${key} delta must be exactly what ${id} asks for`);
  }
});

test("a variant cannot re-point art direction, only physics", () => {
  // `sceneDocument` applies the delta before attaching the environment, which is
  // what stops a lane from reintroducing the original divergence by hand.
  for (const definition of SCENE_CATALOG) {
    for (const variantId of Object.keys(definition.variants ?? {})) {
      const document = sceneDocument(definition, variantId);
      assert.equal(document.environment, definition.environment);
      assert.deepEqual(document.scenery, sceneDocument(definition).scenery,
        `${definition.id}/${variantId} must keep the scene's own scenery`);
    }
  }
});

test("every authored variant is claimed by a lane, and every claim resolves", () => {
  const claimed = new Set(sceneWebGPUSmokeIds.map((id) => {
    const { definition, variantId } = suiteDefinition(id);
    return variantId ? `${definition.id}/${variantId}` : "";
  }).filter(Boolean));
  const authored = new Set(SCENE_CATALOG.flatMap((entry) =>
    Object.keys(entry.variants ?? {}).map((variant) => `${entry.id}/${variant}`)));
  // An unclaimed variant is a fork by another name: nothing runs it, so nothing
  // notices when the scene it deltas moves out from under it.
  assert.deepEqual([...authored].sort(), [...claimed].sort());
  assert.deepEqual([...claimed].sort(), Object.keys(VARIANT_DELTAS).sort());
});
