import assert from "node:assert/strict";
import test from "node:test";
import { createSolidWorld, SOLID_WORLD_TERRAIN_MATERIAL_ID } from "../lib/core/solid-world";
import { compileRetainedOpenSceneFineMeans } from "../lib/methods/adaptive-mass/sparse-cm12-retained-open-density";
import { compileRetainedSceneFineMeans, retainedSceneDensity } from "../lib/methods/adaptive-mass/sparse-cm12-retained-scene-density";
import { compileRetainedSceneSubcellMoments } from "../lib/methods/adaptive-mass/sparse-cm12-retained-subcell-moments";

const dimensions = [8, 8, 8] as const, h = .125;
const ramp = retainedSceneDensity({ generation: 1, transitionWidth: h,
  domain: { lower: [-.5, 0, -.5], upper: [.5, 1, .5] },
  primitives: [{ kind: "quadratic-height", center: [0, h / 2, 0], curvature: [0, 0, 0] }] });
const close = (a: number, b: number, tolerance = 3e-8) => assert.ok(Math.abs(a - b) <= tolerance, `${a} != ${b}`);

test("subcell ABI integrates an affine ramp exactly and preserves its physical halves", () => {
  const result = compileRetainedSceneSubcellMoments(ramp, dimensions, h, createSolidWorld());
  assert.equal(result.seedAmounts.length, 8 * 512); assert.equal(result.openVolumes.length, 8 * 512);
  for (let octant = 0; octant < 8; octant++) {
    assert.equal(result.openVolumes[octant], .125);
    assert.equal(result.seedAmounts[octant], (octant & 2) ? .03125 : .09375);
  }
  assert.equal(result.receipt.integratedSubcells + result.receipt.constantSubcells + result.receipt.reflectedSubcells, 4096);
  assert.ok(result.receipt.reflectedSubcells > 0);
});

test("a rigid mask selects true open subbox integrals rather than multiplying density by capacity", () => {
  const result = compileRetainedSceneSubcellMoments(ramp, dimensions, h, createSolidWorld());
  let amount = 0, open = 0;
  // The rigid covers the lower four half-cell boxes. Each remaining sample
  // represents the entire upper box, not a point value or a native cell.
  for (let octant = 0; octant < 8; octant++) if (octant & 2) {
    amount += result.seedAmounts[octant]; open += result.openVolumes[octant];
  }
  assert.equal(open, .5); assert.equal(amount, .125);
  assert.notEqual(amount, .5 * open);
  const a = .6, b = .2;
  close(a * amount + b * open, .175);
});

test("terrain and rigid subboxes intersect before seed and capacity integration", () => {
  for (const q8 of [64, 128, 254]) {
    const world = createSolidWorld([{ operation: "fill", minimum: [0, 0, 0], maximumExclusive: [1, 1, 1], materialId: SOLID_WORLD_TERRAIN_MATERIAL_ID }]);
    world.pages[0].solidFraction[0] = q8;
    const result = compileRetainedSceneSubcellMoments(ramp, dimensions, h, world);
    const slab = q8 / 255;
    for (let octant = 0; octant < 8; octant++) {
      const lower = Math.max((octant & 2) ? .5 : 0, slab), upper = (octant & 2) ? 1 : .5;
      const expectedOpen = Math.max(0, upper - lower) / 4;
      const expectedAmount = upper > lower ? ((1 - lower) ** 2 - (1 - upper) ** 2) / 8 : 0;
      close(result.openVolumes[octant], expectedOpen, 1e-8);
      close(result.seedAmounts[octant], expectedAmount, 1e-8);
    }
    const fine = compileRetainedOpenSceneFineMeans(ramp, dimensions, h, world);
    close(result.seedAmounts.slice(0, 8).reduce((s, x) => s + x, 0), fine.effectiveMeans[0]);
    close(result.openVolumes.slice(0, 8).reduce((s, x) => s + x, 0), fine.openFractions[0]);
    assert.equal(result.receipt.reflectedSubcells, 0, "asymmetric static geometry must disable reflection reuse");
  }
});

test("sphere octant moments restrict to accepted fine moments and analytic radial diffuse amount", () => {
  const field = retainedSceneDensity({ generation: 5, transitionWidth: h,
    domain: { lower: [-.5, 0, -.5], upper: [.5, 1, .5] },
    primitives: [{ kind: "ellipsoid", center: [0, .5, 0], radii: [.2, .2, .2] }] });
  const result = compileRetainedSceneSubcellMoments(field, dimensions, h, createSolidWorld());
  const fine = compileRetainedSceneFineMeans(field, dimensions, h);
  let total = 0;
  for (let cell = 0; cell < 512; cell++) {
    const amount = result.seedAmounts.slice(8 * cell, 8 * cell + 8).reduce((sum, q) => sum + q, 0);
    close(amount, fine[cell], 3e-7); total += amount * h ** 3;
  }
  const p = field.primitives[0]; assert.ok(p.kind === "ellipsoid");
  const radius = p.radii[0], ratio = h / radius;
  const expected = 4 * Math.PI * radius ** 3 / (15 * ratio) * ((1 + ratio) ** 2.5 - (1 - ratio) ** 2.5);
  close(total, expected, 2e-8);
  assert.ok(result.receipt.maximumEstimatedSubcellMeanError <= 2e-7);
});

test("subcell memory and geometry snapshots are validated before allocation", () => {
  assert.throws(() => compileRetainedSceneSubcellMoments(ramp, dimensions, h, createSolidWorld(), { maximumBytes: 10 }), /require.*bytes/);
  assert.throws(() => compileRetainedSceneSubcellMoments(ramp, dimensions, h, createSolidWorld(), { solidFractions: new Uint8Array(1) }), /does not match/);
});
