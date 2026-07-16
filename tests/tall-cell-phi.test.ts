import assert from "node:assert/strict";
import test from "node:test";
import { createTallCellLayout, initialLiquidPhi } from "../lib/tall-cell-grid";
import { cloneScene, defaultScene } from "../lib/model";

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
    assert.equal(layout.referenceLiquidVolume_cells, layout.initialVolumeCellSum);
  });
}

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
