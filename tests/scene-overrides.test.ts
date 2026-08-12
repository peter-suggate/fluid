import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene } from "../lib/model";
import {
  cleanSceneLink,
  countedSceneOverrides,
  sceneHasPresetBaseline,
  sceneOverrideClearPlan,
  sceneOverridesInQuery,
} from "../lib/scene-overrides";
import { cameraForPreset, getScenePreset } from "../lib/scenes";
import { useUIStore } from "../lib/stores/ui-store";
import { serializeQueryState } from "../lib/url-state";

const method: Parameters<typeof serializeQueryState>[2] = { methodId: "uniform", quality: "balanced", overrides: {} };
const baseline = { hasPresetBaseline: true };

/**
 * The canonical query for a scene sitting exactly as its card opened it.
 *
 * The camera comes from the preset rather than from the store's initial state
 * because that is what `openSceneCard` applies; leaving the default in place
 * would make every scene whose authored framing differs from it open on a
 * camera row it never earned.
 */
function queryFor(presetId: string, scene = getScenePreset(presetId).create(), methodState = method, search = "") {
  const ui = { ...useUIStore.getInitialState(), camera: cameraForPreset(getScenePreset(presetId)) };
  return serializeQueryState(search, { presetId, scene }, methodState, ui);
}

test("a scene opened exactly as authored reports nothing overridden", () => {
  for (const presetId of ["water-box-dam-break", "high-resolution-dam-break"]) {
    const overrides = sceneOverridesInQuery(queryFor(presetId), baseline);
    assert.deepEqual(overrides, [], `${presetId} carries ${overrides.map((entry) => entry.keys).join(", ")}`);
  }
});

test("an edited scene value is one counted override, and clearing restores the factory's own", () => {
  const presetId = "water-box-dam-break";
  const preset = getScenePreset(presetId).create();
  const edited = cloneScene(preset);
  edited.container.width_m += 0.25;
  edited.fluid.density_kg_m3 = 1200;

  const overrides = sceneOverridesInQuery(queryFor(presetId, edited), baseline);
  assert.deepEqual(overrides.map((entry) => entry.keys[0]).sort(),
    ["scene.container.width_m", "scene.fluid.density_kg_m3"]);
  assert.equal(countedSceneOverrides(overrides).length, 2);
  assert.ok(overrides.every((entry) => entry.group === "scene" && entry.clearedBy === "stores"));

  const plan = sceneOverrideClearPlan(["scene.container.width_m"], { scene: edited, presetId });
  assert.equal(plan.scene?.container.width_m, preset.container.width_m);
  assert.equal(plan.scene?.fluid.density_kg_m3, 1200,
    "clearing one key must not revert the others");
  assert.equal(sceneOverridesInQuery(queryFor(presetId, plan.scene!), baseline).length, 1);
});

test("clearing every key lands back on the preset's own query", () => {
  const presetId = "water-box-dam-break";
  const edited = cloneScene(getScenePreset(presetId).create());
  edited.container.height_m += 0.3;
  edited.numerics.maxDt_s = 0.004;
  edited.randomSeed += 7;

  const keys = sceneOverridesInQuery(queryFor(presetId, edited), baseline).flatMap((entry) => entry.keys);
  assert.equal(keys.length, 3);
  const plan = sceneOverrideClearPlan(keys, { scene: edited, presetId });
  assert.deepEqual(sceneOverridesInQuery(queryFor(presetId, plan.scene!), baseline), []);
});

test("an authoring value too large for a URL survives a clear", () => {
  // The one hazard a re-parse of the reduced query would walk into: a sculpted
  // grid is deliberately never serialized, so rebuilding the document from the
  // link would silently discard it along with the override being cleared.
  const presetId = "hero-garden-hose";
  const sculpted = cloneScene(getScenePreset(presetId).create());
  sculpted.terrain!.grid = {
    kind: "grid",
    origin_m: { x: -0.1, z: -0.1 },
    spacing_m: 0.1,
    size: { nx: 2, nz: 2 },
    heights_m: [0.1, 0.2, 0.3, 0.4],
  };
  sculpted.container.width_m += 0.5;

  const plan = sceneOverrideClearPlan(["scene.container.width_m"], { scene: sculpted, presetId });
  assert.deepEqual(plan.scene?.terrain?.grid?.heights_m, [0.1, 0.2, 0.3, 0.4]);
  assert.equal(plan.scene?.container.width_m, getScenePreset(presetId).create().container.width_m);
});

test("a method parameter reads as its own control, and clears to the scene's authored profile", () => {
  // A profiled scene's baseline is its profile, which is what the URL diffed
  // against; clearing must land there rather than on the quality preset.
  const presetId = "hydrostatic-power-two-level";
  const profile = getScenePreset(presetId).methodProfile!;
  const tuned = {
    methodId: "octree",
    quality: "balanced" as const,
    overrides: { octree: { ...profile.overrides, interfaceRefinementBandCells: 6 } },
  };
  const overrides = sceneOverridesInQuery(queryFor(presetId, undefined, tuned), baseline);
  const param = overrides.find((entry) => entry.keys[0].startsWith("param."));
  assert.ok(param, `expected a param override in ${overrides.map((entry) => entry.keys[0]).join(", ")}`);
  assert.equal(param.keys[0], "param.octree.interfaceRefinementBandCells");
  assert.equal(param.group, "method");
  assert.equal(param.value, "6");
  assert.match(param.label, /Interface|band/i);

  const plan = sceneOverrideClearPlan([param.keys[0]], {
    scene: getScenePreset(presetId).create(), presetId,
  });
  assert.deepEqual(plan.methodParams, [{
    methodId: "octree",
    key: "interfaceRefinementBandCells",
    value: profile.overrides.interfaceRefinementBandCells,
  }]);
  assert.equal(plan.scene, undefined, "a solver knob must not rewrite the document");
});

test("an unprofiled parameter clears by dropping to the quality chain", () => {
  const presetId = "water-box-dam-break";
  assert.equal(getScenePreset(presetId).methodProfile, undefined);
  const plan = sceneOverrideClearPlan(["param.uniform.pressureVCycles"], {
    scene: getScenePreset(presetId).create(), presetId,
  });
  assert.deepEqual(plan.methodParams, [{ methodId: "uniform", key: "pressureVCycles", value: undefined }]);
});

test("the solver and quality are overrides of the product default, not of the scene", () => {
  const overrides = sceneOverridesInQuery(
    queryFor("water-box-dam-break", undefined, { methodId: "octree", quality: "ultra", overrides: {} }),
    baseline);
  assert.deepEqual(overrides.map((entry) => entry.keys[0]).sort(), ["method", "quality"]);
  const plan = sceneOverrideClearPlan(["method", "quality"], {
    scene: getScenePreset("water-box-dam-break").create(), presetId: "water-box-dam-break",
  });
  assert.equal(plan.methodId, "uniform");
  assert.equal(plan.quality, "balanced");
});

test("the camera is one uncounted row: a link reopens a view without overriding anything", () => {
  const search = `${queryFor("water-box-dam-break")}&camera.azimuth=1.2&camera.distance=3`;
  const overrides = sceneOverridesInQuery(search, baseline);
  assert.equal(overrides.length, 1);
  assert.equal(overrides[0].group, "view");
  assert.equal(overrides[0].counted, false);
  assert.deepEqual(overrides[0].keys, ["camera.azimuth", "camera.distance"]);

  const plan = sceneOverrideClearPlan(overrides[0].keys, {
    scene: getScenePreset("water-box-dam-break").create(), presetId: "water-box-dam-break",
  });
  assert.equal(plan.ui.camera?.distance_m, getScenePreset("water-box-dam-break").camera?.distance_m
    ?? plan.ui.camera?.distance_m);
});

test("a presentation arm is counted and clears to the shipped default", () => {
  const overrides = sceneOverridesInQuery(`${queryFor("water-box-dam-break")}&svoShadows=0&gridMode=phi`, baseline);
  assert.deepEqual(overrides.map((entry) => entry.keys[0]).sort(), ["gridMode", "svoShadows"]);
  assert.ok(overrides.every((entry) => entry.group === "render" && entry.counted));
  assert.equal(overrides.find((entry) => entry.keys[0] === "svoShadows")!.value, "off");

  const plan = sceneOverrideClearPlan(["svoShadows", "gridMode", "svoLodPixels"], {
    scene: getScenePreset("water-box-dam-break").create(), presetId: "water-box-dam-break",
  });
  assert.equal(plan.ui.svoShadowsEnabled, true);
  assert.equal(plan.ui.gridOverlayMode, "structure");
  assert.ok("lodScreenSpacePixels" in plan.svoRenderTuning,
    "a tuning field must be merged into the record, never replace it");
  assert.deepEqual(plan.reload, []);
});

test("a startup flag is reported as one only a reload can retire", () => {
  const overrides = sceneOverridesInQuery(`${queryFor("water-box-dam-break", undefined, method, "?gpu=off")}`, baseline);
  const flag = overrides.find((entry) => entry.keys[0] === "gpu");
  assert.ok(flag, "gpu= survives canonicalization and must therefore be listed");
  assert.equal(flag.group, "link");
  assert.equal(flag.counted, true);
  assert.equal(flag.clearedBy, "reload");

  const plan = sceneOverrideClearPlan(["gpu"], {
    scene: getScenePreset("water-box-dam-break").create(), presetId: "water-box-dam-break",
  });
  assert.deepEqual(plan.reload, ["gpu"]);
  assert.equal(plan.scene, undefined);
});

test("a document with no authored baseline claims no scene overrides", () => {
  // A starter or a saved scene from a build that no longer has its origin gets
  // the first catalog entry from `getScenePreset`, so every `scene.*` key in
  // its query is a comparison against an unrelated scene.
  const presetId = "starter:blank";
  assert.equal(sceneHasPresetBaseline(presetId), false);
  // Appended after canonicalization: these are managed keys, so passing them
  // through the serializer would simply strip them.
  const search = `${queryFor("water-box-dam-break")}&scene.container.width_m=9&regions=&svoAO=0`;
  const overrides = sceneOverridesInQuery(search, { hasPresetBaseline: false });
  assert.deepEqual(overrides.map((entry) => entry.keys[0]), ["svoAO"]);
});

test("the clean link is the scene's identity and nothing else", () => {
  assert.equal(cleanSceneLink("hero-garden-hose"), "/scene?scene=hero-garden-hose");
});
