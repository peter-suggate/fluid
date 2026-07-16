import assert from "node:assert/strict";
import test from "node:test";
import { getScenePreset } from "../lib/scenes";
import { useUIStore } from "../lib/stores/ui-store";
import { parseQueryState, serializeQueryState } from "../lib/url-state";

test("query state round-trips method, scene, quality, and sparse overrides", () => {
  const scene = getScenePreset("hose-tank").create();
  scene.container.width_m = 1.35;
  scene.fluid.inflow = undefined;
  scene.randomSeed = 91;

  const initialUI = useUIStore.getInitialState();
  const query = serializeQueryState("?campaign=paper&method=stale", { presetId: "hose-tank", scene }, {
    methodId: "uniform",
    quality: "high",
    overrides: {
      uniform: { velocityTransport: "semi-lagrangian", jacobiIterations: 112 },
      "tall-cell": { pressureCycles: 5 }
    }
  }, {
    ...initialUI,
    view: "presentation",
    diagnosticsOpen: true,
    rightPanel: "diagnostics",
    performanceOpen: true,
    gridOverlayAxis: "z",
    gridOverlaySlice: 0.7,
    waterRenderMode: "ray-marched",
    camera: { ...initialUI.camera, distance_m: 4.2 }
  });
  const parsed = parseQueryState(query);

  assert.equal(new URLSearchParams(query).get("campaign"), "paper");
  assert.equal(parsed.methodId, "uniform");
  assert.equal(parsed.presetId, "hose-tank");
  assert.equal(parsed.quality, "high");
  assert.equal(parsed.ui.view, "presentation");
  assert.equal(parsed.ui.diagnosticsOpen, true);
  assert.equal(parsed.ui.rightPanel, "diagnostics");
  assert.equal(parsed.ui.performanceOpen, true);
  assert.equal(parsed.ui.gridOverlayAxis, "z");
  assert.equal(parsed.ui.gridOverlaySlice, 0.7);
  assert.equal(parsed.ui.waterRenderMode, "ray-marched");
  assert.equal(parsed.ui.camera.distance_m, 4.2);
  assert.deepEqual(parsed.overrides, {
    "tall-cell": { pressureCycles: 5 },
    uniform: { velocityTransport: "semi-lagrangian", jacobiIterations: 112 }
  });
  assert.equal(parsed.scene.container.width_m, 1.35);
  assert.equal(parsed.scene.fluid.inflow, undefined);
  assert.equal(parsed.scene.randomSeed, 91);
});

test("query state persists an edited rigid-body roster atomically", () => {
  const scene = getScenePreset("dam-break-boxes").create();
  scene.rigidBodies = scene.rigidBodies.slice(0, 1);
  scene.rigidBodies[0] = { ...scene.rigidBodies[0], density_kg_m3: 640 };

  const query = serializeQueryState("", { presetId: "dam-break-boxes", scene }, {
    methodId: "tall-cell",
    quality: "balanced",
    overrides: {}
  });
  const parsed = parseQueryState(query);

  assert.equal(parsed.scene.rigidBodies.length, 1);
  assert.equal(parsed.scene.rigidBodies[0].density_kg_m3, 640);
});

test("invalid external query values fall back to validated defaults", () => {
  const parsed = parseQueryState("?method=nope&scene=nope&quality=extreme&param.uniform.jacobiIterations=9999&scene.container.width_m=-4&scene.fluid.gravity_m_s2.y=null");
  const defaultScene = getScenePreset("water-box-dam-break").create();

  assert.equal(parsed.methodId, "tall-cell");
  assert.equal(parsed.presetId, "water-box-dam-break");
  assert.equal(parsed.quality, "balanced");
  assert.deepEqual(parsed.overrides, {});
  assert.equal(parsed.scene.container.width_m, defaultScene.container.width_m);
  assert.equal(parsed.scene.fluid.gravity_m_s2.y, defaultScene.fluid.gravity_m_s2.y);
  assert.equal(parsed.ui.view, "scientific");
  assert.equal(parsed.ui.diagnosticsOpen, false);
  assert.equal(parsed.ui.rightPanel, null);
});

test("viewport utility panels round-trip through one mutually exclusive query state", () => {
  const initialUI = useUIStore.getInitialState();
  const query = serializeQueryState("?diagnostics=1", {
    presetId: "water-box-dam-break",
    scene: getScenePreset("water-box-dam-break").create()
  }, {
    methodId: "tall-cell",
    quality: "balanced",
    overrides: {}
  }, { ...initialUI, rightPanel: "visual", diagnosticsOpen: false });

  const params = new URLSearchParams(query);
  assert.equal(params.get("panel"), "visual");
  assert.equal(params.has("diagnostics"), false);
  assert.equal(parseQueryState(query).ui.rightPanel, "visual");
});
