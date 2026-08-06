import assert from "node:assert/strict";
import test from "node:test";
import { getMethod, resolveMethodValues } from "../lib/methods";
import { cloneScene } from "../lib/model";
import { getScenePreset } from "../lib/scenes";
import { useUIStore } from "../lib/stores/ui-store";
import { createSceneQueryLayerCache, parseQueryState, serializeQueryState } from "../lib/url-state";

test("camera URL writes reuse the scene-derived query layer until the document changes", () => {
  const scene = getScenePreset("hero-garden-hose").create();
  const layerFor = createSceneQueryLayerCache();
  const initial = layerFor({ presetId: "hero-garden-hose", scene });

  assert.equal(layerFor({ presetId: "hero-garden-hose", scene }), initial,
    "an unchanged scene identity must not rebake or reserialize its terrain for a UI-only write");

  const edited = cloneScene(scene);
  edited.container.width_m += 0.1;
  const revised = layerFor({ presetId: "hero-garden-hose", scene: edited });
  assert.notEqual(revised, initial);
  assert.deepEqual(revised.find(([key]) => key === "scene.container.width_m"),
    ["scene.container.width_m", JSON.stringify(edited.container.width_m)]);
});

test("an edited sculpted terrain stays out of the URL while analytic terrain remains shareable", () => {
  const sculpted = cloneScene(getScenePreset("hero-garden-hose").create());
  sculpted.terrain!.grid!.heights_m[0] += 0.001;
  const method = { methodId: "octree" as const, quality: "balanced" as const, overrides: {} };
  const sculptedQuery = serializeQueryState("", {
    presetId: "hero-garden-hose",
    scene: sculpted,
  }, method);
  assert.equal(new URLSearchParams(sculptedQuery).has("scene.terrain"), false);
  assert.ok(sculptedQuery.length < 2_000,
    `a sculpted scene location must remain reloadable, not grow to ${sculptedQuery.length} characters`);

  const analytic = cloneScene(getScenePreset("garden-svo-lighting").create());
  analytic.terrain!.baseHeight_m += 0.01;
  const analyticQuery = serializeQueryState("", {
    presetId: "garden-svo-lighting",
    scene: analytic,
  }, method);
  assert.ok(new URLSearchParams(analyticQuery).has("scene.terrain"),
    "small analytic terrain edits should still survive a shared-link round trip");
});

/**
 * `voxels` named the expanded-record inspection overlay (raw-voxels,
 * voxel-levels, surface-voxels, brick-grid, occupied-bricks). That renderer is
 * gone, so the key must be neither parsed nor managed — an unmanaged key is
 * left untouched on the URL, which is what a stale shared link needs, and safe
 * bring-up rejects it as unapproved rather than silently honouring it.
 */
test("the retired voxel inspection query key is no longer parsed or managed", () => {
  const parsed = parseQueryState("?voxels=surface-voxels");
  assert.equal("voxelRenderMode" in parsed.ui, false);
  const scene = getScenePreset("water-box-dam-break").create();
  const query = new URLSearchParams(serializeQueryState(
    "?voxels=surface-voxels", { presetId: "water-box-dam-break", scene },
    { methodId: "uniform", quality: "balanced", overrides: {} }, parsed.ui));
  assert.equal(query.get("voxels"), "surface-voxels",
    "an unmanaged key survives serialization untouched rather than being rewritten");
});

test("query state round-trips the primary's surface reconstruction", () => {
  // The exact arm is the default, so it must never appear in a link — a query
  // key that shows up for the shipping configuration is noise on every share,
  // and here it would also be rejected by safe bring-up's approved-key gate.
  const scene = getScenePreset("water-box-dam-break").create();
  const serialize = (ui: ReturnType<typeof parseQueryState>["ui"]) => new URLSearchParams(serializeQueryState(
    "", { presetId: "water-box-dam-break", scene },
    { methodId: "uniform", quality: "balanced", overrides: {} }, ui));

  const shipping = parseQueryState("");
  assert.equal(shipping.ui.svoRenderTuning.surfaceReconstruction, "analytic");
  assert.equal(serialize(shipping.ui).get("svoSurface"), null,
    "the default surface must not be serialized");

  // Both non-default arms have to be nameable. `voxel-face` is the reference the
  // acceptance gates rest on; `trilinear` is the arm the analytic one was
  // measured against, and "look at this banding" is only a link if it survives.
  for (const arm of ["voxel-face", "trilinear"] as const) {
    const parsed = parseQueryState(`?svoSurface=${arm}`);
    assert.equal(parsed.ui.svoRenderTuning.surfaceReconstruction, arm,
      `a link must be able to name the ${arm} arm`);
    assert.equal(serialize(parsed.ui).get("svoSurface"), arm,
      `the ${arm} arm must survive a round trip`);
  }

  // Unknown falls back to the shipping image rather than silently pinning the
  // frame to a comparison arm, matching normalizeSvoRenderTuning.
  assert.equal(parseQueryState("?svoSurface=nonsense").ui.svoRenderTuning.surfaceReconstruction,
    "analytic");
});

test("query state round-trips the unified scene voxel domain atomically", () => {
  const parsed = parseQueryState('?scene=garden-svo-lighting&scene.voxelDomain={"finestCellSize_m":0.04,"brickSize_cells":4}');
  assert.deepEqual(parsed.scene.voxelDomain, { finestCellSize_m: 0.04, brickSize_cells: 4 });
  const serialized = serializeQueryState("", { presetId: parsed.presetId, scene: parsed.scene }, {
    methodId: parsed.methodId, quality: parsed.quality, overrides: parsed.overrides,
  }, parsed.ui);
  assert.deepEqual(JSON.parse(new URLSearchParams(serialized).get("scene.voxelDomain")!), parsed.scene.voxelDomain);
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
    "resolution",
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
  assert.equal(parsed.scene.voxelDomain.finestCellSize_m, 0.05);
  assert.equal(values.globalFineLevelSetFactor, "4");
});

/**
 * A profiled preset is only valid at the solver settings it authors. The scene
 * picker applies them via `applyProfile`, so a bare link, a reload, or the very
 * first load must resolve to the same configuration — otherwise the UI runs a
 * validation scene at the method's generic quality defaults (band 4 instead
 * of the authored band 3) while every panel claims the scene
 * is loaded, and no Dawn lane reproduces what the UI is actually running.
 */
test("a bare link to a profiled preset resolves to that preset's authored method profile", () => {
  const profile = getScenePreset("minimal-power-dam-break").methodProfile;
  assert.ok(profile, "the minimal dam break preset must author a method profile");

  const parsed = parseQueryState("?scene=minimal-power-dam-break");
  assert.equal(parsed.methodId, profile.methodId);
  assert.equal(parsed.quality, profile.quality);
  assert.deepEqual(parsed.overrides[profile.methodId], { ...profile.overrides });

  const values = resolveMethodValues(getMethod(parsed.methodId), parsed.quality,
    parsed.overrides[parsed.methodId] ?? {});
  assert.equal(values.coarseBackend, "losasso");
  assert.equal(values.maximumLeafSize, "16");
  assert.equal(values.interfaceRefinementBandCells, 3);
  assert.equal(values.globalFineLevelSetFactor, "4");

  // The resolved profile must survive the round trip the app performs on every
  // store write, so a reload cannot silently drop back to the method defaults.
  const rehydrated = parseQueryState(`?${serializeQueryState("",
    { presetId: parsed.presetId, scene: parsed.scene },
    { methodId: parsed.methodId, quality: parsed.quality, overrides: parsed.overrides },
    parsed.ui)}`);
  assert.deepEqual(rehydrated.overrides[profile.methodId], { ...profile.overrides });
});

test("a bare ceiling-drop link hydrates the dedicated band-1 UI profile", () => {
  const profile = getScenePreset("ceiling-slab-drop").methodProfile;
  assert.ok(profile);
  const parsed = parseQueryState("?scene=ceiling-slab-drop");
  const values = resolveMethodValues(getMethod(parsed.methodId), parsed.quality,
    parsed.overrides[parsed.methodId] ?? {});
  assert.equal(values.coarseBackend, "losasso");
  assert.equal(values.maximumLeafSize, "8");
  assert.equal(values.interfaceRefinementBandCells, 1);
  assert.equal(values.globalFineLevelSetFactor, "4");
  assert.deepEqual(parsed.overrides[profile.methodId], { ...profile.overrides });
});

test("an explicit param key overrides one value of a profiled preset", () => {
  const parsed = parseQueryState(
    "?scene=minimal-power-dam-break&param.octree.interfaceRefinementBandCells=0");
  const values = resolveMethodValues(getMethod(parsed.methodId), parsed.quality,
    parsed.overrides[parsed.methodId] ?? {});
  assert.equal(values.interfaceRefinementBandCells, 0);
  // Every other authored setting is still the profile's.
  assert.equal(values.maximumLeafSize, "16");
  assert.equal(values.globalFineLevelSetFactor, "4");
});

test("an explicit scene link can still select a non-default maximum leaf size", () => {
  const parsed = parseQueryState(
    "?scene=minimal-power-dam-break&param.octree.maximumLeafSize=2");
  const values = resolveMethodValues(getMethod(parsed.methodId), parsed.quality,
    parsed.overrides[parsed.methodId] ?? {});
  assert.equal(values.maximumLeafSize, "2");
});

test("retired octree authority switches cannot re-enter through shared links", () => {
  const parsed = parseQueryState(
    "?param.octree.faceVelocityTransport=off"
    + "&param.octree.powerDiagramProjection=axis"
    + "&param.octree.leafSolver=jacobi"
    + "&param.octree.globalFineLevelSetFactor=off"
  );
  assert.equal(parsed.overrides.octree, undefined);

  const query = serializeQueryState(
    "?param.octree.faceVelocityTransport=off"
    + "&param.octree.powerDiagramProjection=axis"
    + "&param.octree.leafSolver=jacobi"
    + "&param.octree.globalFineLevelSetFactor=off",
    { presetId: parsed.presetId, scene: parsed.scene },
    { methodId: parsed.methodId, quality: parsed.quality, overrides: parsed.overrides },
    parsed.ui,
  );
  const params = new URLSearchParams(query);
  assert.equal(params.has("param.octree.faceVelocityTransport"), false);
  assert.equal(params.has("param.octree.powerDiagramProjection"), false);
  assert.equal(params.has("param.octree.leafSolver"), false);
  assert.equal(params.has("param.octree.globalFineLevelSetFactor"), false);
});

test("retired sparse-surface overlay modes cannot re-enter through shared links", () => {
  for (const mode of ["surface", "faces"]) {
    const parsed = parseQueryState(`?grid=z&gridMode=${mode}`);
    assert.equal(parsed.ui.gridOverlayMode, "structure");
    const query = serializeQueryState(
      `?grid=z&gridMode=${mode}`,
      { presetId: parsed.presetId, scene: parsed.scene },
      { methodId: parsed.methodId, quality: parsed.quality, overrides: parsed.overrides },
      parsed.ui,
    );
    assert.equal(new URLSearchParams(query).has("gridMode"), false);
  }
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

test("retired renderer and lighting modes are ignored and removed from canonical links", () => {
  for (const search of [
    "?render=raster",
    "?render=svo",
    "?svoLighting=direct",
    "?svoLighting=cone",
    "?svoLighting=gi",
  ]) {
    const parsed = parseQueryState(search);
    assert.equal("svoRenderMode" in parsed.ui, false);
    assert.equal("svoLightingMode" in parsed.ui, false);
    const query = serializeQueryState(search, {
      presetId: parsed.presetId,
      scene: parsed.scene,
    }, {
      methodId: parsed.methodId,
      quality: parsed.quality,
      overrides: parsed.overrides,
    }, parsed.ui);
    const canonical = new URLSearchParams(query);
    assert.equal(canonical.has("render"), false);
    assert.equal(canonical.has("svoLighting"), false);
  }
});

test("SVO shadows and ambient occlusion round-trip as independent finished-image options", () => {
  const disabled = parseQueryState("?render=svo&svoShadows=0&svoAO=0");
  assert.equal(disabled.ui.svoShadowsEnabled, false);
  assert.equal(disabled.ui.svoAmbientOcclusionEnabled, false);
  const disabledQuery = serializeQueryState("?svoShadows=1&svoAO=1", {
    presetId: disabled.presetId,
    scene: disabled.scene,
  }, {
    methodId: disabled.methodId,
    quality: disabled.quality,
    overrides: disabled.overrides,
  }, disabled.ui);
  const query = new URLSearchParams(disabledQuery);
  assert.equal(query.get("svoShadows"), "0");
  assert.equal(query.get("svoAO"), "0");

  const defaults = parseQueryState("?svoShadows=1&svoAO=1");
  assert.equal(defaults.ui.svoShadowsEnabled, true);
  assert.equal(defaults.ui.svoAmbientOcclusionEnabled, true);
  const defaultQuery = serializeQueryState("?svoShadows=0&svoAO=0", {
    presetId: defaults.presetId,
    scene: defaults.scene,
  }, {
    methodId: defaults.methodId,
    quality: defaults.quality,
    overrides: defaults.overrides,
  }, defaults.ui);
  assert.equal(new URLSearchParams(defaultQuery).has("svoShadows"), false);
  assert.equal(new URLSearchParams(defaultQuery).has("svoAO"), false);
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

test("diagnostics uses the same sole panel query authority as every other sidebar", () => {
  const initialUI = useUIStore.getInitialState();
  const query = serializeQueryState("?diagnostics=1&performance=1", {
    presetId: "water-box-dam-break",
    scene: getScenePreset("water-box-dam-break").create()
  }, {
    methodId: "octree",
    quality: "balanced",
    overrides: {}
  }, { ...initialUI, rightPanel: "diagnostics", diagnosticsOpen: true });

  const params = new URLSearchParams(query);
  assert.equal(params.get("panel"), "diagnostics");
  assert.equal(params.has("diagnostics"), false);
  assert.equal(params.has("performance"), false);
  assert.equal(parseQueryState(query).ui.rightPanel, "diagnostics");
  assert.equal(parseQueryState(query).ui.diagnosticsOpen, true);
});

test("right panel width round-trips through the query state", () => {
  const parsed = parseQueryState("?panel=bodies&panelWidth=734");
  assert.equal(parsed.ui.rightPanel, "bodies");
  assert.equal(parsed.ui.rightPanelWidth, 734);

  const query = serializeQueryState("?panelWidth=stale", {
    presetId: parsed.presetId,
    scene: parsed.scene,
  }, {
    methodId: parsed.methodId,
    quality: parsed.quality,
    overrides: parsed.overrides,
  }, parsed.ui);
  assert.equal(new URLSearchParams(query).get("panelWidth"), "734");
  assert.equal(parseQueryState("?panelWidth=20").ui.rightPanelWidth, 620);
  assert.equal(parseQueryState("?panelWidth=2000").ui.rightPanelWidth, 620);
});

test("retired sidebar switches are ignored and removed from canonical links", () => {
  const parsed = parseQueryState("?performance=1&diagnostics=1&waterdiag=1");
  assert.equal(parsed.ui.rightPanel, null);
  assert.equal(parsed.ui.diagnosticsOpen, false);

  const query = serializeQueryState("?performance=1&diagnostics=1&waterdiag=1", {
    presetId: parsed.presetId,
    scene: parsed.scene
  }, {
    methodId: parsed.methodId,
    quality: parsed.quality,
    overrides: parsed.overrides
  }, parsed.ui);
  const params = new URLSearchParams(query);
  assert.equal(params.has("panel"), false);
  assert.equal(params.has("performance"), false);
  assert.equal(params.has("diagnostics"), false);
  assert.equal(params.has("waterdiag"), false);
});

test("query state carries the primary traversal so a comparison survives reload", () => {
  const scene = getScenePreset("garden-svo-lighting").create();
  const method = { methodId: "octree" as const, quality: "balanced" as const, overrides: {} };
  const traced = parseQueryState("?svoPrimary=traced");
  assert.equal(traced.ui.svoPrimaryTraversal, "traced");
  assert.equal(new URLSearchParams(
    serializeQueryState("", { presetId: "garden-svo-lighting", scene }, method, traced.ui),
  ).get("svoPrimary"), "traced");

  // The default stays out of the URL, and an unreadable value resolves to it
  // rather than leaving the renderer holding a mode it cannot build.
  const raster = parseQueryState("?svoPrimary=nonsense");
  assert.equal(raster.ui.svoPrimaryTraversal, "raster");
  assert.equal(new URLSearchParams(
    serializeQueryState("", { presetId: "garden-svo-lighting", scene }, method, raster.ui),
  ).get("svoPrimary"), null);
});

test("query state carries the opt-in primary-seam-closure choice", () => {
  const scene = getScenePreset("garden-svo-lighting").create();
  const method = { methodId: "octree" as const, quality: "balanced" as const, overrides: {} };
  const enabled = parseQueryState("?svoPrimarySeamClosure=1");
  assert.equal(enabled.ui.silhouetteRefinementEnabled, true);
  assert.equal(new URLSearchParams(
    serializeQueryState("", { presetId: "garden-svo-lighting", scene }, method, enabled.ui),
  ).get("svoPrimarySeamClosure"), "1");

  const defaults = parseQueryState("?svoPrimarySeamClosure=nonsense");
  assert.equal(defaults.ui.silhouetteRefinementEnabled, false);
  assert.equal(new URLSearchParams(
    serializeQueryState("", { presetId: "garden-svo-lighting", scene }, method, defaults.ui),
  ).get("svoPrimarySeamClosure"), null);
});
