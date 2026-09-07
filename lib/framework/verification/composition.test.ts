import assert from "node:assert/strict";
import test from "node:test";
import { composeFeatures, compositionUpdateImpact, type FeatureDefinition } from "../composition";

const visibility: FeatureDefinition = {
  id: "visibility",
  requires: ["hierarchy"],
  variants: [
    { id: "traced", point: "primary", provides: ["gbuffer"], update: "rebuild", default: true },
    { id: "mesh", point: "primary", provides: ["gbuffer"], requires: ["mesh-source"], update: "reset" },
  ],
  controls: [{ id: "primary", label: "Visibility", kind: "choice" }],
  placements: [{ control: "primary", slot: "frame" }, { control: "primary", slot: "toolstrip" }],
};
const hierarchy: FeatureDefinition = { id: "construction", provides: ["hierarchy"] };

test("resolves selected implementations before validating downstream dependencies", () => {
  const result = composeFeatures({ features: [{ id: "shade", requires: ["gbuffer"] }, visibility, hierarchy] });
  assert.equal(result.variants[0]?.id, "traced");
  assert.equal(result.controls.length, 1);
  assert.equal(result.placements.length, 2);
  assert.equal(result.placements[0]?.feature, "visibility");
  assert.ok(Object.isFrozen(result));
});
test("rejects an incompatible combination instead of substituting an implementation", () => {
  assert.throws(() => composeFeatures({ features: [visibility, hierarchy], selections: { primary: "mesh" } }), /mesh.*missing capability mesh-source/);
  assert.throws(() => composeFeatures({ features: [visibility, hierarchy], selections: { primary: "missing" } }), /supported variant/);
  assert.throws(() => composeFeatures({ features: [visibility, hierarchy], selections: { typo: "traced" } }), /Unknown variation point/);
});
test("detects competing providers and ambiguous defaults", () => {
  assert.throws(() => composeFeatures({ features: [hierarchy, { id: "other", provides: ["hierarchy"] }] }), /Ambiguous capability/);
  assert.throws(() => composeFeatures({ features: [{ ...visibility, variants: visibility.variants!.map(v => ({ ...v, default: true })) }, hierarchy] }), /Multiple defaults/);
});
test("rejects dangling controls and duplicate registrations", () => {
  assert.throws(() => composeFeatures({ features: [hierarchy, hierarchy] }), /Duplicate/);
  assert.throws(() => composeFeatures({ features: [{ id: "bad", placements: [{ slot: "frame", control: "missing" }] }] }), /Unknown control/);
});
test("variant changes use the strongest entering or leaving lifecycle requirement", () => {
  const features = [visibility, hierarchy];
  const traced = composeFeatures({ features });
  const mesh = composeFeatures({ features, capabilities: ["mesh-source"], selections: { primary: "mesh" } });
  assert.equal(compositionUpdateImpact(traced, traced), "live");
  assert.equal(compositionUpdateImpact(traced, mesh), "reset");
  assert.equal(compositionUpdateImpact(mesh, traced), "reset");
});

test("publication wiring rejects equal names with incompatible representations", async () => {
  const { publicationPort } = await import("../ports");
  const density = publicationPort<Float32Array>({ id: "density", representation: "sparse-cell-density", lifetime: "generation" });
  const other = publicationPort<Float32Array>({ id: "density", representation: "dense-grid-density", lifetime: "generation" });
  const producer = { id: "simulation", outputs: [density] };
  assert.throws(() => composeFeatures({ features: [producer, { id: "surface", inputs: [{ port: other, provider: "simulation" }] }] }), /matching representation/);
  composeFeatures({ features: [producer, { id: "surface", inputs: [{ port: density, provider: "simulation" }] }] });

});


test("validated definitions cannot change through their caller-owned objects", () => {
  const feature = { id: "owner", provides: ["density"], controls: [{ id: "amount", label: "Amount", kind: "number" as const }] };
  const result = composeFeatures({ features: [feature] });
  feature.provides.push("unvalidated");
  feature.controls[0].label = "Changed";
  assert.deepEqual(result.features[0]?.provides, ["density"]);
  assert.equal(result.controls[0]?.label, "Amount");
  assert.ok(Object.isFrozen(result.features[0]?.controls?.[0]));
});
test("a variant provider cannot shadow a feature provider", () => {
  assert.throws(() => composeFeatures({ features: [
    { id: "primary/traced" },
    { id: "visibility", variants: [{ id: "traced", point: "primary", update: "live" }] },
  ] }), /provider IDs collide/);
});

test("one owner cannot publish two incompatible meanings for the same port ID", async () => {
  const { publicationPort } = await import("../ports");
  const a = publicationPort<number>({ id: "field", representation: "density", lifetime: "frame" });
  const b = publicationPort<number>({ id: "field", representation: "pressure", lifetime: "frame" });
  assert.throws(() => composeFeatures({ features: [{ id: "owner", outputs: [a, b] }] }), /Duplicate output port/);
});
