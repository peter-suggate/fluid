import assert from "node:assert/strict";
import test from "node:test";
import { chooseTallCellBase, createSingleTallCellProbeControlLayout, createSingleTallCellProbeLayout, createTallCellLayout, limitNeighboringTallCellBases, tallCellFluxSampleCount, tallCellSettings } from "../lib/tall-cell-grid";
import { cloneScene, defaultScene } from "../lib/model";

test("restricted tall-cell presets retain cubic resolution and represent the vertical dam face with tall columns", () => {
  const expectedEquivalent = { balanced: 110_000, high: 500_000, ultra: 1_200_000 } as const;
  for (const quality of ["balanced", "high", "ultra"] as const) {
    const layout = createTallCellLayout(defaultScene, quality);
    assert.ok(Math.abs(layout.equivalentUniformCellCount / expectedEquivalent[quality] - 1) < 0.16);
    assert.ok(layout.settings.regularLayers >= tallCellSettings[quality].regularLayers);
    assert.ok(layout.columnBases.every((base) => base <= tallCellSettings[quality].maximumTallHeight));
    assert.ok(layout.columnBases.some((base) => base > 3), "the default layout keeps genuinely tall cells (the 2026-07-16 audit removed the parity clamp)");
    assert.ok(layout.compressionRatio < 0.85, `tall cells must compress the grid (got ${layout.compressionRatio})`);
    assert.ok(layout.columnBases.some((base) => base >= 2));
    assert.equal(layout.packedNy, layout.settings.regularLayers + 2);
    assert.ok(Math.abs(layout.cellSize_m.x / layout.cellSize_m.y - 1) < 0.04);
    assert.ok(Math.abs(layout.cellSize_m.z / layout.cellSize_m.y - 1) < 0.04);
    let weightedVolume = 0;
    for (let z = 0; z < layout.nz; z += 1) for (let x = 0; x < layout.nx; x += 1) {
      const base = layout.columnBases[x + layout.nx * z];
      weightedVolume += layout.initialVolume[x + layout.nx * layout.packedNy * z] * base;
      for (let y = 2; y < layout.packedNy; y += 1) weightedVolume += layout.initialVolume[x + layout.nx * (y + layout.packedNy * z)];
    }
    assert.equal(layout.initialVolumeCellSum, weightedVolume);
  }
});

test("deep water changes vertical extent without degrading surface resolution", () => {
  const shallowScene = cloneScene(defaultScene), deepScene = cloneScene(defaultScene);
  shallowScene.container.height_m = 0.9;
  deepScene.container.height_m = 4.5;
  deepScene.container.fillFraction = 0.8;
  deepScene.fluid.initialCondition = "tank-fill";
  const unrestricted = { maximumTallHeight: 2048 };
  const shallow = createTallCellLayout(shallowScene, "high", 2048, unrestricted), deep = createTallCellLayout(deepScene, "high", 2048, unrestricted);
  assert.equal(deep.nx, shallow.nx);
  assert.equal(deep.nz, shallow.nz);
  assert.ok(Math.abs(deep.cellSize_m.x - shallow.cellSize_m.x) < 1e-12);
  assert.ok(deep.compressionRatio < 0.12);
  assert.ok(deep.packedSampleCount * 7 < deep.equivalentUniformCellCount);
});

test("surface band satisfies the requested liquid and air halos when possible", () => {
  const settings = tallCellSettings.high;
  const base = chooseTallCellBase(70, 70, 160, settings);
  assert.ok(70 - base + 1 >= settings.liquidHalo);
  assert.ok(base + settings.regularLayers - 71 >= settings.airHalo);
});

test("restricted packed columns never use an incomplete base-zero ordinary limit", () => {
  const settings = { ...tallCellSettings.balanced, regularLayers: 12 };
  assert.equal(chooseTallCellBase(0, 0, 46, settings), 2);
  assert.equal(chooseTallCellBase(0, 0, 13, settings), 0);
});

test("neighbor limiter bounds abrupt changes in tall-cell height", () => {
  const limited = limitNeighboringTallCellBases(new Float32Array([0, 0, 40, 40, 0, 0]), 6, 1, 4, 8);
  for (let x = 1; x < limited.length; x += 1) assert.ok(Math.abs(limited[x] - limited[x - 1]) <= 4);
});

test("empty dam-break columns keep an elevated surface band when its vertical surface fits", () => {
  const scene = cloneScene(defaultScene);
  scene.container.height_m = 4.5;
  const layout = createTallCellLayout(scene, "balanced");
  const farDry = layout.columnBases[layout.nx - 1 + layout.nx * (layout.nz - 1)];
  assert.ok(farDry > 0);
  assert.ok(farDry <= layout.fineNy - layout.settings.regularLayers);
  for (let z = 0; z < layout.nz; z += 1) for (let x = 0; x < layout.nx; x += 1) {
    const here = layout.columnBases[x + layout.nx * z];
    if (x + 1 < layout.nx) assert.ok(Math.abs(here - layout.columnBases[x + 1 + layout.nx * z]) <= layout.settings.maximumNeighborDelta);
    if (z + 1 < layout.nz) assert.ok(Math.abs(here - layout.columnBases[x + layout.nx * (z + 1)]) <= layout.settings.maximumNeighborDelta);
  }
});

test("shallow domains use the uniform-grid limit safely", () => {
  const scene = cloneScene(defaultScene);
  scene.container.height_m = 0.08;
  const layout = createTallCellLayout(scene, "balanced");
  assert.equal(layout.settings.regularLayers, layout.fineNy);
  assert.ok(layout.columnBases.every((base) => base === 0));
});

test("deep tall-face integration is bounded independently of depth", () => {
  assert.equal(tallCellFluxSampleCount(0), 0);
  assert.equal(tallCellFluxSampleCount(12), 12);
  assert.equal(tallCellFluxSampleCount(48), 12);
  assert.equal(tallCellFluxSampleCount(806), 12);
  assert.equal(tallCellFluxSampleCount(8_000), 12);
});

test("single-tall-cell probe changes exactly one fully submerged paper-grid column", () => {
  const scene = cloneScene(defaultScene);
  scene.fluid.initialCondition = "tank-fill";
  scene.container.fillFraction = 0.7;
  const layout = createSingleTallCellProbeLayout(scene, "balanced", 2048, { height: 4 });
  const probe = layout.singleTallCellProbe;
  assert.ok(probe);
  assert.equal(probe.height, 4);
  assert.equal(probe.initialState, "liquid");
  assert.equal(layout.settings.regularLayers, layout.fineNy);
  assert.equal(layout.settings.remeshInterval, Number.MAX_SAFE_INTEGER);
  assert.equal(probe.mutedHeight, 2);
  assert.equal(probe.supportRadius, 0);
  assert.equal(probe.affectedColumns, 1);
  assert.equal(layout.columnBases.filter((base) => base > probe.mutedHeight).length, 1);
  assert.ok(layout.columnBases.every((base) => base >= probe.mutedHeight), "paper Sec. 3.1 requires one bottom tall cell per column");
  assert.equal(layout.columnBases[probe.x + layout.nx * probe.z], probe.height);
  assert.ok(probe.height <= layout.settings.maximumNeighborDelta, "paper Eq. 10 neighbor bound must hold");

  let reconstructedSum = 0;
  for (let z = 0; z < layout.nz; z += 1) for (let y = 0; y < layout.fineNy; y += 1) for (let x = 0; x < layout.nx; x += 1) {
    const base = layout.columnBases[x + layout.nx * z];
    if (base > 0 && y < base) reconstructedSum += layout.initialVolume[x + layout.nx * layout.packedNy * z];
    else reconstructedSum += layout.initialVolume[x + layout.nx * (2 + y - base + layout.packedNy * z)];
  }
  assert.equal(reconstructedSum, layout.initialVolumeCellSum);
});

test("single-tall-cell probe rejects an interface inside the tall cell", () => {
  const scene = cloneScene(defaultScene);
  scene.fluid.initialCondition = "tank-fill";
  scene.container.fillFraction = 0.05;
  assert.throws(
    () => createSingleTallCellProbeLayout(scene, "balanced", 2048, { height: 4 }),
    /contains the initial liquid interface/,
    "paper Sec. 3.6 requires regular cells below the bottom-most surface"
  );
});

test("single-tall-cell support rings isolate the paper Eq. 10 transition", () => {
  const scene = cloneScene(defaultScene);
  scene.fluid.initialCondition = "tank-fill";
  scene.container.fillFraction = 0.7;
  const candidate = createSingleTallCellProbeLayout(scene, "balanced", 2048, { height: 4, supportRadius: 2 });
  const control = createSingleTallCellProbeControlLayout(scene, "balanced", 2048, { height: 4, supportRadius: 2 });
  const probe = candidate.singleTallCellProbe!;
  assert.equal(probe.affectedColumns, 13);
  assert.equal(candidate.columnBases.filter((base) => base === 4).length, 13);
  assert.ok(control.columnBases.every((base) => base === 2));
  assert.equal(control.initialVolumeCellSum, candidate.initialVolumeCellSum);
});

test("single-tall-cell control differs from the candidate only in the probe column packing", () => {
  const scene = cloneScene(defaultScene);
  scene.fluid.initialCondition = "tank-fill";
  scene.container.fillFraction = 0.7;
  const candidate = createSingleTallCellProbeLayout(scene, "balanced", 2048, { height: 4 });
  const control = createSingleTallCellProbeControlLayout(scene, "balanced", 2048, { height: 4 });
  const probe = candidate.singleTallCellProbe!;
  assert.ok(control.columnBases.every((base) => base === 2));
  assert.equal(control.initialVolumeCellSum, candidate.initialVolumeCellSum);
  for (let z = 0; z < candidate.nz; z += 1) for (let y = 0; y < candidate.packedNy; y += 1) for (let x = 0; x < candidate.nx; x += 1) {
    if (x === probe.x && z === probe.z) continue;
    const index = x + candidate.nx * (y + candidate.packedNy * z);
    assert.equal(candidate.initialVolume[index], control.initialVolume[index]);
  }
});

test("approaching rigid bodies do not make the bottom tall store grow", () => {
  const makeScene = (bodyY_m: number) => {
    const scene = cloneScene(defaultScene);
    scene.fluid.initialCondition = "tank-fill";
    scene.container.fillFraction = 0.4;
    scene.rigidBodies = [{
      id: "probe", name: "probe", shape: "sphere",
      dimensions_m: { x: 0.08, y: 0.08, z: 0.08 },
      density_kg_m3: 500,
      position_m: { x: 0, y: bodyY_m, z: 0 },
      orientation: { x: 0, y: 0, z: 0, w: 1 },
      linearVelocity_m_s: { x: 0, y: 0, z: 0 },
      angularVelocity_rad_s: { x: 0, y: 0, z: 0 },
      restitution: 0.3, friction: 0.5
    }];
    return scene;
  };
  const empty = cloneScene(defaultScene);
  empty.fluid.initialCondition = "tank-fill";
  empty.container.fillFraction = 0.4;
  empty.rigidBodies = [];
  const reference = createTallCellLayout(empty, "balanced");
  const surface_m = empty.container.height_m * 0.4;

  // A body hovering far above the surface must not perturb the layout.
  const airborne = createTallCellLayout(makeScene(empty.container.height_m * 0.95), "balanced");
  assert.deepEqual(Array.from(airborne.columnBases), Array.from(reference.columnBases),
    "an airborne body must not change any column base");

  // Moving the same body to just above the surface must still leave the
  // surface-driven classifier unchanged. The old proximity rule raised this
  // base, which is exactly the false tall-cell classification seen in the UI.
  const entering = createTallCellLayout(makeScene(surface_m + 0.05), "balanced");
  const centerX = Math.floor(entering.nx / 2), centerZ = Math.floor(entering.nz / 2);
  const referenceBase = reference.columnBases[centerX + reference.nx * centerZ];
  const enteringBase = entering.columnBases[centerX + entering.nx * centerZ];
  assert.equal(enteringBase, referenceBase,
    `an approaching body must not grow the tall store under it (got ${enteringBase} vs ${referenceBase})`);

  // Once the body extends below the surface-driven split, it becomes an
  // upper bound and can only shorten the tall store to keep solids out of it.
  const submerged = createTallCellLayout(makeScene(surface_m * 0.35), "balanced");
  const submergedBase = submerged.columnBases[centerX + submerged.nx * centerZ];
  assert.ok(submergedBase <= referenceBase,
    `a submerged body may only shorten the tall store (got ${submergedBase} vs ${referenceBase})`);
});
