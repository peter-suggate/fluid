import assert from "node:assert/strict";
import test from "node:test";
import { narrowBandBudgetCapacity } from "../lib/methods/uniform/uniform-narrow-band-flip";
import { particleSlabSpill } from "../lib/methods/uniform/uniform-particle-slab.wgsl";
import { particleDepositStorage } from "../lib/methods/uniform/uniform-particle-deposit-storage";
import { validateScene } from "../lib/core/model";
import { createReservoirSeepScene } from "../lib/core/reservoir-seep-scene";

test("a particle budget's slabs hold its samples, the spill reserve on top", () => {
  for (const budget of [{ fineTiles: 1, perFineCell: 4 }, { fineTiles: 4000, perFineCell: 4 }, { fineTiles: 16000, perFineCell: 2.5 }]) {
    const capacity = narrowBandBudgetCapacity(budget), samples = Math.ceil(64 * budget.fineTiles * budget.perFineCell);
    assert.ok(capacity - particleSlabSpill(capacity) >= samples);
    assert.ok(capacity <= samples + Math.max(4096, Math.ceil(samples / 15)) + 1);
  }
});

test("a budget's deposit still shares the consumed particle bank", () => {
  // 385 deposit words a fine tile against 64 cells of 48-byte records at one a cell.
  const fineTiles = 4000, bank = 48 * narrowBandBudgetCapacity({ fineTiles, perFineCell: 1 });
  assert.equal(particleDepositStorage(1 << 20, fineTiles, 2 ** 31, bank).shared, true);
});

test("fluid.tiers validates its brick count and density", () => {
  const errors = (tiers: object) => { const scene = createReservoirSeepScene(); (scene.fluid as { tiers?: object }).tiers = tiers; return validateScene(scene).filter(e => e.includes("fluid.tiers")); };
  assert.deepEqual(errors({ bricks: 512, particlesPerFineCell: 4 }), []);
  assert.equal(errors({ bricks: 0 }).length, 1);
  assert.equal(errors({ bricks: 1.5 }).length, 1);
  assert.equal(errors({ particlesPerFineCell: 0 }).length, 1);
  assert.deepEqual(errors({ start: "4h", fineBlocks: "auto" }), []);
  assert.deepEqual(errors({ start: "h" }), []);
  assert.equal(errors({ start: "16h" }).length, 1);
});

// Letters under a coarse start sized the particle store from the h-tile
// ceiling ("particle slabs exceed the pool: 897264 records sized of 1048577",
// then "particle bricks exceed the cell bins' 5088 brick slots"), while
// particles also live on 4h owners. Under the records rule the slabs hold
// R records at any spread over every stored brick (64 a block).
test("a records budget's slabs hold R records spread one a brick over every stored brick", () => {
  for (const [records, blocks] of [[1000, 226], [989184, 226], [2555460, 311], [100, 4]] as const) {
    const capacity = narrowBandBudgetCapacity({ fineTiles: 1, perFineCell: 4, records }, blocks);
    const bricks = Math.min(records, 64 * blocks);
    assert.ok(capacity - particleSlabSpill(capacity) >= Math.ceil(1.25 * records) + 32 * bricks, `${records} records over ${blocks} blocks`);
  }
});

test("the particle record rule: 8 samples a band cell x the shared growth, over the census band", async () => {
  const { uniformParticleRecordRule, uniformParticleBand } = await import("../lib/methods/uniform/uniform-particle-census");
  assert.equal(uniformParticleRecordRule(1000), 12000);
  assert.throws(() => uniformParticleRecordRule(-1), /count of band cells/);
  // A 64³ lattice of two pages an axis, wet below y = 20: the band is the
  // 3 wet layers under the free surface (y 17..19), 64 x 64 a layer.
  const edge = 32, at = new Map<number, Float32Array>();
  for (let pz = 0; pz < 2; pz++) for (let py = 0; py < 2; py++) for (let px = 0; px < 2; px++) {
    const page = new Float32Array(edge ** 3);
    for (let z = 0; z < edge; z++) for (let y = 0; y < edge; y++) for (let x = 0; x < edge; x++) if (edge * py + y < 20) page[x + edge * (y + edge * z)] = 1;
    if (!(py === 1 && page.every(v => v === 0))) at.set(px + 2 * (py + 2 * pz), page);
  }
  // Its h tiles: the layers y 16..19 hold it, 16 x 16 tiles across.
  assert.deepEqual(uniformParticleBand({ edge, cells: [64, 64, 64], pages: [2, 2, 2], at } as never, 3), { cells: 3 * 64 * 64, tiles: 16 * 16 });
  // A deep region's band cells are not counted: here the free surface's own
  // regions (y 16..31) are passed as deep, so only those below them remain:
  // none, as the surface's band lies wholly in y 16..19.
  assert.deepEqual(uniformParticleBand({ edge, cells: [64, 64, 64], pages: [2, 2, 2], at } as never, 3, new Set(), (_x, y) => y === 1), { cells: 0, tiles: 0 });
  // Deep regions elsewhere (below y 16) leave the band as it is.
  assert.deepEqual(uniformParticleBand({ edge, cells: [64, 64, 64], pages: [2, 2, 2], at } as never, 3, new Set(), (_x, y) => y === 0), { cells: 3 * 64 * 64, tiles: 16 * 16 });
});

test("a census band counts its sources' band tiles apart: a drop's own tiles, the initial liquid's not among them", async () => {
  const { uniformParticleCensusBand } = await import("../lib/methods/uniform/uniform-particle-census");
  const scene = createReservoirSeepScene();
  const still = uniformParticleCensusBand(scene, 3);
  assert.equal(still.sourceTiles, 0);
  // A sphere in the air above the reservoir: its 3-cell band shell, wholly its own tiles.
  const h = scene.voxelDomain.finestCellSize_m, r = 12 * h;
  scene.fluid.scheduledDrops = [{ time_s: 0.5, volume: { shape: "sphere", center_m: { x: 0, y: scene.container.height_m - 2 * r, z: 0 }, radius_m: r } }];
  const dropped = uniformParticleCensusBand(scene, 3);
  assert.ok(dropped.sourceTiles! > 0);
  assert.equal(dropped.tiles - still.tiles, dropped.sourceTiles);
  // The shell's tiles lie within the sphere's bounding tiles (4 cells a tile, a tile of slack an end).
  assert.ok(dropped.sourceTiles! <= (Math.ceil(2 * 12 / 4) + 2) ** 3);
});
