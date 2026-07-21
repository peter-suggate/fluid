import assert from "node:assert/strict";
import test from "node:test";
import { getMethod, resolveMethodValues } from "../lib/methods";
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
    diagnosticsOpen: false,
    rightPanel: "performance",
    gridOverlayAxis: "z",
    gridOverlaySlice: 0.7,
    voxelRenderMode: "brick-grid",
    svoRenderMode: "svo",
    camera: { ...initialUI.camera, distance_m: 4.2 }
  });
  const parsed = parseQueryState(query);

  assert.equal(new URLSearchParams(query).get("campaign"), "paper");
  assert.equal(parsed.methodId, "uniform");
  assert.equal(parsed.presetId, "hose-tank");
  assert.equal(parsed.quality, "high");
  assert.equal(parsed.ui.diagnosticsOpen, false);
  assert.equal(parsed.ui.rightPanel, "performance");
  assert.equal(parsed.ui.gridOverlayAxis, "z");
  assert.equal(parsed.ui.gridOverlaySlice, 0.7);
  assert.equal(parsed.ui.voxelRenderMode, "brick-grid");
  assert.equal(parsed.ui.svoRenderMode, "svo");
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

test("query state accepts and preserves a Y solver-grid slice", () => {
  const parsed = parseQueryState("?grid=y&gridSlice=0.35");
  assert.equal(parsed.ui.gridOverlayAxis, "y");
  assert.equal(parsed.ui.gridOverlaySlice, 0.35);
  const scene = getScenePreset("water-box-dam-break").create();
  const serialized = serializeQueryState("", { presetId: "water-box-dam-break", scene }, { methodId: "uniform", quality: "balanced", overrides: {} }, parsed.ui);
  assert.equal(new URLSearchParams(serialized).get("grid"), "y");
});

test("query state preserves a full-volume paper-technique diagnostic", () => {
  const parsed = parseQueryState("?grid=volume&gridSlice=0.42&gridMode=delaunay-tetrahedra");
  assert.equal(parsed.ui.gridOverlayAxis, "volume");
  assert.equal(parsed.ui.gridOverlaySlice, 0.42);
  assert.equal(parsed.ui.gridOverlayMode, "delaunay-tetrahedra");

  const serialized = serializeQueryState("", {
    presetId: parsed.presetId,
    scene: parsed.scene,
  }, {
    methodId: parsed.methodId,
    quality: parsed.quality,
    overrides: parsed.overrides,
  }, parsed.ui);
  const query = new URLSearchParams(serialized);
  assert.equal(query.get("grid"), "volume");
  assert.equal(query.get("gridSlice"), "0.42");
  assert.equal(query.get("gridMode"), "delaunay-tetrahedra");
  assert.equal(parseQueryState("?grid=volume&gridSlice=0&gridMode=power-cells").ui.gridOverlaySlice, 0.05,
    "full-volume links retain the shader's minimum visible opacity");
});

test("query state preserves compact hierarchy and paper-technique diagnostic modes", () => {
  for (const mode of [
    "resolution", "surface", "faces",
    "power-cells", "power-faces", "delaunay-tetrahedra", "transition-band", "power-operator",
    "octree-lifecycle", "fine-band-lifecycle", "operator-diagonal", "operator-rhs",
    "operator-reciprocity", "operator-open-fraction", "tetra-validity",
  ] as const) {
    const parsed = parseQueryState(`?grid=z&gridMode=${mode}`);
    assert.equal(parsed.ui.gridOverlayMode, mode);
    const serialized = serializeQueryState("", {
      presetId: parsed.presetId,
      scene: parsed.scene,
    }, {
      methodId: parsed.methodId,
      quality: parsed.quality,
      overrides: parsed.overrides,
    }, parsed.ui);
    assert.equal(new URLSearchParams(serialized).get("gridMode"), mode);
  }
});

test("query state round-trips independently configured CPU and GPU timesteps", () => {
  const scene = getScenePreset("water-box-dam-break").create();
  scene.numerics.fixedDt_s = 0.006;
  scene.numerics.maxDt_s = 0.018;

  const query = serializeQueryState("", { presetId: "water-box-dam-break", scene }, {
    methodId: "tall-cell",
    quality: "balanced",
    overrides: {}
  });
  const params = new URLSearchParams(query);
  const parsed = parseQueryState(query);

  assert.equal(params.get("scene.numerics.fixedDt_s"), "0.006");
  assert.equal(params.get("scene.numerics.maxDt_s"), "0.018");
  assert.equal(parsed.scene.numerics.fixedDt_s, 0.006);
  assert.equal(parsed.scene.numerics.maxDt_s, 0.018);
});

test("invalid external query values fall back to validated defaults", () => {
  const parsed = parseQueryState("?method=nope&scene=nope&quality=extreme&environment=the-void&param.uniform.jacobiIterations=9999&scene.container.width_m=-4&scene.fluid.gravity_m_s2.y=null");
  const defaultScene = getScenePreset("water-box-dam-break").create();

  assert.equal(parsed.methodId, "octree");
  assert.equal(parsed.presetId, "water-box-dam-break");
  assert.equal(parsed.quality, "balanced");
  assert.deepEqual(parsed.overrides, {});
  assert.equal(parsed.scene.container.width_m, defaultScene.container.width_m);
  assert.equal(parsed.scene.fluid.gravity_m_s2.y, defaultScene.fluid.gravity_m_s2.y);
  assert.equal(parsed.ui.diagnosticsOpen, false);
  assert.equal(parsed.ui.rightPanel, null);
  const values = resolveMethodValues(getMethod(parsed.methodId), parsed.quality, parsed.overrides[parsed.methodId] ?? {});
  assert.equal(values.surfaceColumns, 384);
  assert.equal(values.faceVelocityTransport, "on");
  assert.equal(values.globalFineLevelSetFactor, "4");
  assert.equal(values.powerDiagramProjection, "authoritative");
  assert.equal(values.leafSolver, "auto");
});

test("background is fixed by the scene and legacy environment overrides are removed", () => {
  const parsed = parseQueryState("?scene=sphere-jet&environment=garden");
  assert.equal(getScenePreset(parsed.presetId).background, "night-lab");

  const query = serializeQueryState("?scene=sphere-jet&environment=garden", {
    presetId: parsed.presetId,
    scene: parsed.scene
  }, {
    methodId: parsed.methodId,
    quality: parsed.quality,
    overrides: parsed.overrides
  }, parsed.ui);
  assert.equal(new URLSearchParams(query).has("environment"), false);
});

test("legacy presentation choices are removed from canonical links", () => {
  const parsed = parseQueryState("?view=presentation&render=ray-marched&fps=90");
  const query = serializeQueryState("?view=presentation&render=ray-marched&fps=90", {
    presetId: parsed.presetId,
    scene: parsed.scene
  }, {
    methodId: parsed.methodId,
    quality: parsed.quality,
    overrides: parsed.overrides
  }, parsed.ui);
  const params = new URLSearchParams(query);
  assert.equal(params.has("view"), false);
  assert.equal(params.has("render"), false);
  assert.equal(params.has("fps"), false);
});

test("production renderer mode omits the raster default and serializes explicit SVO", () => {
  const parsed = parseQueryState("?render=svo");
  assert.equal(parsed.ui.svoRenderMode, "svo");

  const scene = getScenePreset(parsed.presetId).create();
  const sparse = serializeQueryState("", { presetId: parsed.presetId, scene }, {
    methodId: parsed.methodId,
    quality: parsed.quality,
    overrides: parsed.overrides
  }, parsed.ui);
  assert.equal(new URLSearchParams(sparse).get("render"), "svo");

  const raster = serializeQueryState("?render=svo", { presetId: parsed.presetId, scene }, {
    methodId: parsed.methodId,
    quality: parsed.quality,
    overrides: parsed.overrides
  }, { ...parsed.ui, svoRenderMode: "raster" });
  assert.equal(new URLSearchParams(raster).has("render"), false);
  assert.equal(parseQueryState("?render=invalid").ui.svoRenderMode, "raster");
});

test("SVO lighting round-trips exact direct while cone remains the canonical fail-soft default", () => {
  const direct = parseQueryState("?render=svo&svoLighting=direct");
  assert.equal(direct.ui.svoRenderMode, "svo");
  assert.equal(direct.ui.svoLightingMode, "direct");
  const directQuery = serializeQueryState("?svoLighting=stale", {
    presetId: direct.presetId,
    scene: direct.scene,
  }, {
    methodId: direct.methodId,
    quality: direct.quality,
    overrides: direct.overrides,
  }, direct.ui);
  assert.equal(new URLSearchParams(directQuery).get("svoLighting"), "direct");

  const cone = parseQueryState("?svoLighting=cone");
  assert.equal(cone.ui.svoLightingMode, "cone");
  const coneQuery = serializeQueryState("?svoLighting=direct", {
    presetId: cone.presetId,
    scene: cone.scene,
  }, {
    methodId: cone.methodId,
    quality: cone.quality,
    overrides: cone.overrides,
  }, cone.ui);
  assert.equal(new URLSearchParams(coneQuery).has("svoLighting"), false);
  assert.equal(parseQueryState("?svoLighting=invalid").ui.svoLightingMode, "cone");
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

test("legacy performance query links open the performance sidebar and canonicalize", () => {
  const parsed = parseQueryState("?performance=1");
  assert.equal(parsed.ui.rightPanel, "performance");

  const query = serializeQueryState("?performance=1", {
    presetId: parsed.presetId,
    scene: parsed.scene
  }, {
    methodId: parsed.methodId,
    quality: parsed.quality,
    overrides: parsed.overrides
  }, parsed.ui);
  const params = new URLSearchParams(query);
  assert.equal(params.get("panel"), "performance");
  assert.equal(params.has("performance"), false);
});
