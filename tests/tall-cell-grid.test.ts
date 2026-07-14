import assert from "node:assert/strict";
import test from "node:test";
import { chooseTallCellBase, createTallCellLayout, limitNeighboringTallCellBases, tallCellFluxSampleCount, tallCellSettings } from "../lib/tall-cell-grid";
import { cloneScene, defaultScene } from "../lib/model";

test("restricted tall-cell presets retain cubic resolution and represent the vertical dam face with tall columns", () => {
  const expectedEquivalent = { balanced: 110_000, high: 500_000, ultra: 1_200_000 } as const;
  for (const quality of ["balanced", "high", "ultra"] as const) {
    const layout = createTallCellLayout(defaultScene, quality);
    assert.ok(Math.abs(layout.equivalentUniformCellCount / expectedEquivalent[quality] - 1) < 0.16);
    assert.equal(layout.settings.regularLayers, tallCellSettings[quality].regularLayers);
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
  const shallow = createTallCellLayout(shallowScene, "high"), deep = createTallCellLayout(deepScene, "high");
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
