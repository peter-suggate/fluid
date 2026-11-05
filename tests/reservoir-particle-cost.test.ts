import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { canonicalRankCPU, mergeCellCPU, particleCellCost, uniformPopulations } from "../tools/reservoir-particle-cost";
import { createReservoirSeepScene } from "../lib/core/reservoir-seep-scene";
import { sceneLatticeDimensions } from "../lib/core/scene-lattice-dimensions";
import { uniformNearSettled } from "../lib/methods/uniform/uniform-near-settled";
import { particleSlabSpill } from "../lib/methods/uniform/uniform-particle-slab.wgsl";

test("CPU canonical ranks preserve complete records, ties and append-order independence", () => {
  for (const stride of [12, 28]) {
    const records = Array.from({ length: 257 }, (_, i) => Uint32Array.from({ length: stride }, (_, k) => k < 3 ? Math.floor(i / 7) % 11 : (i * 17 + k * 31) % 113));
    records[100] = records[101]!.slice();
    const sorted = [...records].sort((a, b) => {
      for (let k = 0; k < stride; k++) if (a[k] !== b[k]) return a[k]! - b[k]!;
      return 0;
    });
    for (const input of [records, [...records].reverse()]) {
      const result = canonicalRankCPU(input);
      assert.deepEqual(result.records, sorted);
      assert.equal(result.comparisons, 257 * 256);
    }
  }
});

test("fixed density is linear across 50k; concentration makes canonical ranking quadratic", () => {
  for (const n of [25_000, 50_000, 100_000]) {
    assert.equal(particleCellCost(uniformPopulations(n, 8)).rankComparisons, 7 * n);
  }
  const diffuse = particleCellCost(uniformPopulations(50_000, 8));
  const dense = particleCellCost(uniformPopulations(50_000, 256));
  assert.ok(dense.rankComparisons > 36 * diffuse.rankComparisons);
  const hotspot = particleCellCost([...uniformPopulations(45_000, 8), 5_000]);
  assert.ok(hotspot.rankComparisons > 72 * diffuse.rankComparisons);
});

test("merge cap protects neither the outer layer nor mass-saturated deep runs", () => {
  const mass = (values: readonly number[]) => values.reduce((a, b) => a + b, 0);
  for (const depth of [-0.25, -1]) assert.equal(mergeCellCPU(new Array(1024).fill(1), depth).masses.length, 1024);
  const input = new Array(1024).fill(8);
  const result = mergeCellCPU(input, -2);
  assert.deepEqual(result.masses, input);
  assert.equal(mass(result.masses), mass(input));
  const half = mergeCellCPU(input.slice(0, 512), -2);
  assert.ok(result.visits > 3.9 * half.visits);
  // Unit masses also leave almost all a large run alive on its first merge.
  const unit = mergeCellCPU(new Array(1024).fill(1), -2);
  assert.equal(unit.masses.length, 1024 - 6 * 7);
  assert.equal(mass(unit.masses), 1024);
  const shader = readFileSync(new URL("../lib/methods/uniform/uniform-narrow-band-flip.ts", import.meta.url), "utf8");
  assert.ok(shader.includes("if(deep< -1.0&&count>cap)"));
  assert.ok(shader.includes("for(var j=k+stride;j<i;j+=stride)"));
  assert.ok(shader.includes("if(m+mo>NB_MASS_MAX){continue;}"));
});

test("scene geometry allows a 50k-particle catch film far shallower than the merge gate", () => {
  const scene = createReservoirSeepScene();
  const [nx, , nz] = sceneLatticeDimensions(scene);
  const catchFloorCells = (nx - 16 * Math.floor(nx / 32)) * nz;
  assert.equal(catchFloorCells, 16_384);
  // Geometric scenario only: seed-volume equivalents, not a fluid prediction.
  const depthInCells = (50_000 / 8) / catchFloorCells;
  assert.ok(depthInCells < 0.4);
  assert.equal(uniformNearSettled({}).coldSurface, "still");
  assert.equal(uniformNearSettled({}).warmResolution, "h");
  // Slab reserve's breakpoint is 65,536 of *capacity*, not live population.
  assert.equal(particleSlabSpill(50_000), 4096);
  assert.equal(particleSlabSpill(65_536), 4096);
  assert.equal(particleSlabSpill(131_072), 8192);
});
