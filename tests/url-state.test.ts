import assert from "node:assert/strict";
import test from "node:test";
import { getMethod, resolveMethodValues } from "../lib/methods";
import { editorFluidLattice, fluidBrickCenter } from "../lib/editor-fluid";
import { cloneScene } from "../lib/model";
import { getSceneDefinition, getScenePreset, scenePresets } from "../lib/scenes";
import { sceneDocumentAtLattice } from "../lib/scene-definition";
import { sceneLatticeDimensions } from "../lib/scene-lattice";
import { svoSceneryRefinementDepth } from "../lib/svo-render-tuning";
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
  sculpted.terrain!.grid = {
    kind: "grid",
    origin_m: { x: -0.1, z: -0.1 },
    spacing_m: 0.1,
    size: { nx: 2, nz: 2 },
    heights_m: [0.1, 0.1, 0.1, 0.1],
  };
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

test("every scene opens directly on the global fidelity defaults", () => {
  for (const sceneId of ["hero-garden-hose", "garden-svo-lighting", "water-box-dam-break"]) {
    const parsed = parseQueryState(`?scene=${sceneId}`);
    assert.equal(parsed.ui.svoRenderTuning.resolutionScale, 1, sceneId);
    assert.equal(parsed.ui.svoRenderTuning.coneLightingScale, 0.5, sceneId);
    assert.equal(parsed.ui.svoRenderTuning.coneRadianceReconstruction, "full-res-relight", sceneId);
    assert.equal(parsed.ui.svoPrimaryTraversal, "traced", sceneId);
    assert.equal(parsed.ui.svoStageView, "dry-radiance", sceneId);
  }

  const parsed = parseQueryState("?scene=hero-garden-hose");
  assert.equal(parsed.scene.voxelDomain.detailCellSize_m, undefined, "capture uses the authored 6.25 mm lattice");

  const method = { methodId: "octree" as const, quality: "balanced" as const, overrides: {} };
  const off = { ...parsed.ui, svoPrimaryTraversal: "raster" as const, svoStageView: "off" as const };
  const query = serializeQueryState("", { presetId: parsed.presetId, scene: parsed.scene }, method, off);
  const rehydrated = parseQueryState(query);
  assert.equal(rehydrated.ui.svoPrimaryTraversal, "raster", "an explicit departure from the global default survives reload");
  assert.equal(rehydrated.ui.svoStageView, "off", "presentation can still be restored deliberately");
});

test("every catalog scene has a minimal canonical URL with no authored defaults", () => {
  for (const preset of scenePresets) {
    const parsed = parseQueryState(`?scene=${preset.id}`);
    const query = serializeQueryState("", {
      presetId: parsed.presetId,
      scene: parsed.scene,
    }, {
      methodId: parsed.methodId,
      quality: parsed.quality,
      overrides: parsed.overrides,
    }, parsed.ui);
    assert.equal(query, `scene=${preset.id}`, preset.id);
  }
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

  // `svoSurface` selected between three ways of guessing at the shading normal
  // per pixel. The voxel carries its own baked normal now, so there is one arm
  // and no key: an old link naming it must parse rather than throw, and must
  // not round-trip a query parameter nothing reads.
  const shipping = parseQueryState("");
  assert.equal(serialize(shipping.ui).get("svoSurface"), null,
    "the retired surface key must not be serialized");
  assert.equal(serialize(parseQueryState("?svoSurface=trilinear").ui).get("svoSurface"), null,
    "an old link naming a retired arm must parse and drop the key");
});

test("query state round-trips the unified scene voxel domain atomically", () => {
  const parsed = parseQueryState('?scene=garden-svo-lighting&scene.voxelDomain={"finestCellSize_m":0.04,"brickSize_cells":4}');
  assert.deepEqual(parsed.scene.voxelDomain, { finestCellSize_m: 0.04, brickSize_cells: 4 });
  const serialized = serializeQueryState("", { presetId: parsed.presetId, scene: parsed.scene }, {
    methodId: parsed.methodId, quality: parsed.quality, overrides: parsed.overrides,
  }, parsed.ui);
  assert.deepEqual(JSON.parse(new URLSearchParams(serialized).get("scene.voxelDomain")!), parsed.scene.voxelDomain);
});

test("a coarser environment refinement rung survives a shared-link round trip", () => {
  const definition = getSceneDefinition("hero-garden-hose");
  const base = getScenePreset("hero-garden-hose").create().voxelDomain.finestCellSize_m;
  const coarse = sceneDocumentAtLattice(definition, {
    cellSize_m: base * 8,
    detailCellSize_m: base * 8,
  }).scene;
  coarse.voxelDomain.environmentRefinementBaseCellSize_m = base;
  const query = serializeQueryState("", { presetId: definition.id, scene: coarse }, {
    methodId: "octree", quality: "balanced", overrides: {},
  });
  const restored = parseQueryState(query);
  assert.equal(restored.scene.voxelDomain.finestCellSize_m, base * 8);
  assert.equal(restored.scene.voxelDomain.environmentRefinementBaseCellSize_m, base);
  assert.equal(svoSceneryRefinementDepth(restored.scene.voxelDomain, { fluid: false }), -3);
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

  assert.equal(parsed.methodId, "uniform");
  assert.equal(parsed.presetId, "water-box-dam-break");
  assert.equal(parsed.quality, "balanced");
  assert.equal(parsed.overrides.uniform, undefined);
  assert.equal(parsed.scene.container.width_m, defaultScene.container.width_m);
  assert.equal(parsed.scene.fluid.gravity_m_s2.y, defaultScene.fluid.gravity_m_s2.y);
  assert.equal(parsed.ui.diagnosticsOpen, false);
  assert.equal(parsed.ui.rightPanel, null);
  const values = resolveMethodValues(getMethod(parsed.methodId), parsed.quality, parsed.overrides[parsed.methodId] ?? {});
  assert.equal(parsed.scene.voxelDomain.finestCellSize_m, 0.05);
  assert.equal(values.timeStep, "paper");
});

test("a bare dam-break UI link resolves to the uniform paper-step default", () => {
  const preset = getScenePreset("water-box-dam-break");
  assert.equal(preset.methodProfile, undefined);
  const parsed = parseQueryState("?scene=water-box-dam-break");
  assert.equal(parsed.methodId, "uniform");
  assert.equal(parsed.quality, "balanced");
  assert.equal(parsed.scene.numerics.fixedDt_s, 0.016);
  assert.equal(parsed.scene.numerics.maxDt_s, 0.016);
  const values = resolveMethodValues(getMethod(parsed.methodId), parsed.quality,
    parsed.overrides[parsed.methodId] ?? {});
  assert.equal(values.timeStep, "paper");
  assert.deepEqual(sceneLatticeDimensions(parsed.scene), [24, 16, 16],
    "the default Uniform scene must halve exactly into the CM11a coarsest solve");
});

test("a bare profiled scene keeps its comparison tuple without overriding Uniform", () => {
  const profile = getScenePreset("minimal-power-dam-break").methodProfile;
  assert.ok(profile, "the minimal dam break preset must author a method profile");

  const parsed = parseQueryState("?scene=minimal-power-dam-break");
  assert.equal(parsed.methodId, "uniform");
  assert.equal(parsed.quality, "balanced");
  assert.deepEqual(parsed.overrides[profile.methodId], { ...profile.overrides });

  const values = resolveMethodValues(getMethod(profile.methodId), profile.quality,
    parsed.overrides[profile.methodId] ?? {});
  assert.equal(values.coarseBackend, "losasso");
  assert.equal(values.maximumLeafSize, "16");
  assert.equal(values.interfaceRefinementBandCells, 3);
  assert.equal(values.globalFineLevelSetFactor, "1");

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
  assert.equal(parsed.methodId, "uniform");
  const values = resolveMethodValues(getMethod(profile.methodId), profile.quality,
    parsed.overrides[profile.methodId] ?? {});
  assert.equal(values.coarseBackend, "losasso");
  assert.equal(values.maximumLeafSize, "8");
  assert.equal(values.interfaceRefinementBandCells, 1);
  assert.equal(values.globalFineLevelSetFactor, "1");
  assert.deepEqual(parsed.overrides[profile.methodId], { ...profile.overrides });
});

test("an explicit param key overrides one value of a profiled preset", () => {
  const parsed = parseQueryState(
    "?scene=minimal-power-dam-break&method=octree&param.octree.interfaceRefinementBandCells=0");
  const values = resolveMethodValues(getMethod(parsed.methodId), parsed.quality,
    parsed.overrides[parsed.methodId] ?? {});
  assert.equal(values.interfaceRefinementBandCells, 0);
  // Every other authored setting is still the profile's.
  assert.equal(values.maximumLeafSize, "16");
  assert.equal(values.globalFineLevelSetFactor, "1");
});

test("an explicit scene link can still select a non-default maximum leaf size", () => {
  const parsed = parseQueryState(
    "?scene=minimal-power-dam-break&method=octree&param.octree.maximumLeafSize=2");
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
  assert.deepEqual(parsed.overrides.octree,
    getScenePreset("water-box-dam-break").methodProfile?.overrides);

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

  const visualsQuery = serializeQueryState(query, {
    presetId: "water-box-dam-break",
    scene: getScenePreset("water-box-dam-break").create()
  }, {
    methodId: "tall-cell",
    quality: "balanced",
    overrides: {}
  }, { ...initialUI, rightPanel: "visuals", diagnosticsOpen: false });
  assert.equal(new URLSearchParams(visualsQuery).get("panel"), "visuals");
  assert.equal(parseQueryState(visualsQuery).ui.rightPanel, "visuals");
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
  ).get("svoPrimary"), null);

  // The default stays out of the URL, and an unreadable value resolves to it
  // rather than leaving the renderer holding a mode it cannot build.
  const fallback = parseQueryState("?svoPrimary=nonsense");
  assert.equal(fallback.ui.svoPrimaryTraversal, "traced");
  const raster = parseQueryState("?svoPrimary=raster");
  assert.equal(raster.ui.svoPrimaryTraversal, "raster");
  assert.equal(new URLSearchParams(
    serializeQueryState("", { presetId: "garden-svo-lighting", scene }, method, raster.ui),
  ).get("svoPrimary"), "raster");
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

test("a painted twin dam survives a shared link at a fraction of the characters it used to cost", () => {
  const method = { methodId: "uniform" as const, quality: "balanced" as const, overrides: {} };
  const parsed = parseQueryState("?scene=twin-dam-collision&scene.voxelDomain="
    + encodeURIComponent(JSON.stringify({ finestCellSize_m: 0.0125, brickSize_cells: 8 })));
  const lattice = editorFluidLattice(parsed.scene);
  const painted = cloneScene(parsed.scene);
  const bricks: { x: number; y: number; z: number }[] = [];
  for (const originX of [0, 20])
    for (let z = originX === 0 ? 0 : 4; z < (originX === 0 ? 4 : 8); z += 1)
      for (let y = 0; y < 4; y += 1)
        for (let x = originX; x < originX + 8; x += 1) bricks.push({ x, y, z });
  painted.fluid = {
    ...painted.fluid,
    initialBrickSeeds_m: bricks.map((brick) => fluidBrickCenter(lattice, brick)),
    initialBrickSeedsAdditive: true,
  };

  const query = serializeQueryState("", { presetId: "twin-dam-collision", scene: painted }, method);
  assert.equal(new URLSearchParams(query).has("scene.fluid.initialBrickSeeds_m"), false,
    "the seed array is no longer a query path");
  // The same paint as a `scene.*` array was ~20.7 kB, which reloads as an HTTP
  // 431 once the request line passes Node's 16 kB header ceiling.
  assert.ok(query.length < 1_000,
    `a painted scene location must remain reloadable, not grow to ${query.length} characters`);

  const rehydrated = parseQueryState(query);
  assert.equal(rehydrated.scene.fluid.initialBrickSeeds_m?.length, 256);
  assert.deepEqual(rehydrated.scene.fluid.initialBrickSeeds_m, painted.fluid.initialBrickSeeds_m);
});

test("links written before the seeds key still restore the water they carry", () => {
  const parsed = parseQueryState("?scene=twin-dam-collision");
  const legacy = [{ x: 0.35, y: 0.1, z: 0.1 }, { x: -0.35, y: 0.1, z: -0.1 }];
  const restored = parseQueryState("?scene=twin-dam-collision&scene.fluid.initialBrickSeeds_m="
    + encodeURIComponent(JSON.stringify(legacy)));

  assert.notDeepEqual(parsed.scene.fluid.initialBrickSeeds_m, legacy, "the preset authors its own seeds");
  assert.deepEqual(restored.scene.fluid.initialBrickSeeds_m, legacy);
});

test("an optional boolean the preset leaves unset still hydrates from a link", () => {
  const additive = parseQueryState(
    "?scene=twin-dam-collision&scene.fluid.initialBrickSeedsAdditive=true");
  assert.equal(getScenePreset("twin-dam-collision").create().fluid.initialBrickSeedsAdditive, undefined,
    "the preset must not author the flag, or this proves nothing");
  assert.equal(additive.scene.fluid.initialBrickSeedsAdditive, true,
    "painted water that adds to the authored dam must not reload as water that replaces it");
});
