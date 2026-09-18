import assert from "node:assert/strict";
import test from "node:test";
import { dynamicRungCatalogue } from "../lib/methods/adaptive-volume/sparse-cm12-dynamic-rung-catalog";
import { createSparseCM12TransportExecutionImageLayout } from "../lib/methods/adaptive-volume/sparse-cm12-transport-execution-image";

test("B4 prepares exactly 73 distinct cells across its complete ladder", () => {
  const c = dynamicRungCatalogue(4);
  assert.deepEqual(c.DYNAMIC_PAGE_RUNGS, [4, 2, 1]);
  assert.equal(c.DYNAMIC_PAGE_CELL_COUNT, 64 + 8 + 1);
  assert.deepEqual(c.DYNAMIC_RUNG_LAYOUTS.map(l => l.width), [1, 2, 4]);
  assert.throws(() => c.dynamicRungLayout(8), /unprepared/);
  const cells = new Set<number>();
  for (const layout of c.DYNAMIC_RUNG_LAYOUTS) for (let i = 0; i < layout.cellCount; i++) cells.add(layout.cellOffset + i);
  assert.equal(cells.size, 73);
  for (const seam of c.preparedDynamicSeamCatalogue().values()) for (const row of seam.rows) {
    assert.ok(row.center.every(q => q >= 0 && q <= 4));
    if (seam.neighbor !== 0) assert.ok(Math.abs(row.terms.reduce((v, t) => v + t.coefficient, 0)) < 1e-12);
  }
});

test("transport allocation is independent of deep-ocean macro coverage and world extent", () => {
  for (const brickFineResolution of [4, 8] as const) {
    const layouts = [1, 2, 32, 256, 1 << 20].map(maximumSpanBricks => createSparseCM12TransportExecutionImageLayout({
      brickFineResolution, leafCapacity: 1000, logicalBrickDimensions: [1 << 20, 1 << 20, 1 << 20], maximumSpanBricks,
    }));
    assert.ok(layouts.every(l => l.totalBytes === layouts[0]!.totalBytes));
    assert.equal(layouts[0]!.storedPacketCapacity, 1000 * (brickFineResolution / 4) ** 3);
    assert.equal(layouts[0]!.spatialTileCapacity, 1000 * (brickFineResolution / 4) ** 3);
  }
});

test("B4 large pages preserve deep-ocean physical cell widths and mass", async () => {
  const { createOceanSeicheScene } = await import("../lib/core/scenes");
  const { initializeSparseBrickAtlasFromScene, sparseBrickAtlasStats } = await import("../lib/methods/adaptive-volume/sparse-brick-atlas");
  const scene = createOceanSeicheScene();
  scene.container = { ...scene.container, fillFraction: 1 };
  delete scene.fluid.initialBrickSeeds_m;
  delete scene.fluid.initialBrickSeedsAdditive;
  const atlases = ([4, 8] as const).map(brickFineResolution => initializeSparseBrickAtlasFromScene(scene, {
    finestDimensions: [256, 1024, 256], brickFineResolution,
  }));
  const [small, reference] = atlases;
  assert.ok(small!.bricks.every(b => b.resolution <= 4));
  assert.ok(small!.maximumSpanBricks >= 64);
  assert.equal(small!.maximumSpanBricks * 4, reference!.maximumSpanBricks * 8);
  assert.equal(sparseBrickAtlasStats(small!).integratedMassFineCells,
    sparseBrickAtlasStats(reference!).integratedMassFineCells);
  assert.ok(small!.bricks.length < 1024);
});
