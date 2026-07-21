import assert from "node:assert/strict";
import test from "node:test";
import { createTallCellLayout, initialLiquidPhi } from "../lib/tall-cell-grid";
import { cloneScene, defaultScene } from "../lib/model";
import { terrainColumnHeights } from "../lib/terrain";

function independentlyReduceInitialPhi(
  scene: typeof defaultScene,
  layout: ReturnType<typeof createTallCellLayout>
) {
  const h = Math.min(layout.cellSize_m.x, layout.cellSize_m.y, layout.cellSize_m.z);
  const terrainHeights = terrainColumnHeights(scene, layout.nx, layout.nz);
  let fixedPoint = 0;
  for (let z = 0; z < layout.nz; z += 1) for (let x = 0; x < layout.nx; x += 1) {
    const column = x + layout.nx * z;
    const base = Math.round(layout.columnBases[column]);
    const bottom = layout.initialPhi[x + layout.nx * layout.packedNy * z];
    const top = layout.initialPhi[x + layout.nx * (1 + layout.packedNy * z)];
    const terrainHeightCells = terrainHeights[column] / layout.cellSize_m.y;
    let tallOccupied = 0;
    for (let y = 0; y < layout.fineNy; y += 1) {
      const packedY = 2 + y - base;
      const phi = y < base && base > 0
        ? bottom + (top - bottom) * y / Math.max(base - 1, 1)
        : packedY >= 2 && packedY < layout.packedNy
          ? layout.initialPhi[x + layout.nx * (packedY + layout.packedNy * z)]
          : 5 * h;
      const open = 1 - Math.min(1, Math.max(0, terrainHeightCells - y));
      const occupied = open * Math.min(1, Math.max(0, 0.5 - phi / h));
      if (y < base) tallOccupied += occupied;
      else fixedPoint += Math.trunc(occupied * 256);
    }
    fixedPoint += Math.trunc(tallOccupied * 256);
  }
  return fixedPoint / 256;
}

for (const initialCondition of ["dam-break", "tank-fill"] as const) {
  test(`initial ${initialCondition} phi is sampled at Eq 4 positions`, () => {
    const scene = cloneScene(defaultScene);
    scene.fluid.initialCondition = initialCondition;
    const layout = createTallCellLayout(scene, "balanced");
    const h = layout.cellSize_m;
    const limit = 5 * Math.min(h.x, h.y, h.z);
    for (let z = 0; z < layout.nz; z += 1) for (let x = 0; x < layout.nx; x += 1) {
      const base = layout.columnBases[x + layout.nx * z];
      for (let packedY = 0; packedY < layout.packedNy; packedY += 1) {
        const active = packedY < 2 ? base > 0 : base + packedY - 2 < layout.fineNy;
        const index = x + layout.nx * (packedY + layout.packedNy * z);
        assert.ok(Math.abs(layout.initialPhi[index]) <= limit + 1e-6);
        if (!active) continue;
        const sampleY = packedY === 0 ? 0.5 : packedY === 1 ? Math.max(0.5, base - 0.5) : base + packedY - 1.5;
        const exact = initialLiquidPhi(scene, {
          x: -0.5 * scene.container.width_m + (x + 0.5) * h.x,
          y: sampleY * h.y,
          z: -0.5 * scene.container.depth_m + (z + 0.5) * h.z
        });
        assert.ok(Math.abs(layout.initialPhi[index] - Math.max(-limit, Math.min(limit, exact))) < 2e-6);
      }
    }
    assert.ok(layout.referenceLiquidVolume_cells > 0);
  });
}

test("dam-break level-set reference is its represented t=0 volume, not the binary seed count", () => {
  const scene = cloneScene(defaultScene);
  scene.fluid.initialCondition = "dam-break";
  scene.numerics.surfaceColumnsOverride = 384;
  const layout = createTallCellLayout(scene, "balanced", 2048, { regularLayers: 12 });
  assert.deepEqual([layout.nx, layout.fineNy, layout.nz], [24, 18, 16]);
  assert.equal(layout.initialVolumeCellSum, 1632, "binary geometry retains the exact wet-cell count");

  const independentlyReconstructed = independentlyReduceInitialPhi(scene, layout);
  assert.equal(layout.referenceLiquidVolume_cells, independentlyReconstructed);
  assert.equal(layout.referenceLiquidVolume_cells, 1524.27734375);
  assert.ok(Math.abs((independentlyReconstructed - layout.initialVolumeCellSum) / layout.initialVolumeCellSum) > 0.06,
    "the binary-vs-phi discrepancy exists before stepping and must not be reported as transport loss");
});

test("level-set reference applies terrain openness before the packed-row fixed-point conversion", () => {
  const scene = cloneScene(defaultScene);
  scene.fluid.initialCondition = "tank-fill";
  scene.rigidBodies = [];
  scene.terrain = { baseHeight_m: 0.137, features: [] };
  scene.numerics.surfaceColumnsOverride = 384;
  const layout = createTallCellLayout(scene, "balanced", 2048, { regularLayers: 12 });
  assert.equal(layout.referenceLiquidVolume_cells, independentlyReduceInitialPhi(scene, layout));

  const flat = cloneScene(scene);
  delete flat.terrain;
  const flatLayout = createTallCellLayout(flat, "balanced", 2048, { regularLayers: 12 });
  assert.ok(layout.referenceLiquidVolume_cells < flatLayout.referenceLiquidVolume_cells,
    "the same terrain-cut water must not be counted as represented liquid");
});

test("dam-break tall endpoint samples are independent point values", () => {
  const layout = createTallCellLayout(defaultScene, "balanced");
  let distinct = false;
  for (let z = 0; z < layout.nz && !distinct; z += 1) for (let x = 0; x < layout.nx && !distinct; x += 1) {
    const bottom = layout.initialPhi[x + layout.nx * layout.packedNy * z];
    const top = layout.initialPhi[x + layout.nx * (1 + layout.packedNy * z)];
    distinct = Math.abs(bottom - top) > 1e-6;
  }
  assert.ok(distinct, "at least one dam column must store different bottom/top phi samples");
});
