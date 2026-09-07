import assert from "node:assert/strict";
import test from "node:test";
import { createSolidWorld, SOLID_WORLD_TERRAIN_MATERIAL_ID } from "../lib/core/solid-world";
import { compileRetainedOpenSceneFineMeans } from "../lib/methods/adaptive-mass/sparse-cm12-retained-open-density";
import { bindRetainedSceneSupportLattice, compileRetainedSceneFineMeans, retainedSceneDensity } from "../lib/methods/adaptive-mass/sparse-cm12-retained-scene-density";
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

test("clipped lattice edge subcells preserve full volume without f32 endpoint slivers", () => {
  const domain = { lower: [-.325, 0, -.225], upper: [.325, .5, .225] } as const;
  const dimensions = [13, 10, 9] as const, h = .05;
  const full = bindRetainedSceneSupportLattice(retainedSceneDensity({ generation: 1,
    transitionWidth: h, domain, primitives: [{ kind: "box", ...domain }] }), dimensions, h);
  const world = createSolidWorld();
  const result = compileRetainedSceneSubcellMoments(full, dimensions, h, world);
  const open = compileRetainedOpenSceneFineMeans(full, dimensions, h, world);
  assert.ok(result.seedAmounts.every(amount => amount === .125));
  assert.ok(result.openVolumes.every(volume => volume === .125));
  assert.equal(result.seedAmounts.reduce((sum, amount) => sum + amount, 0), 1170);
  assert.ok(open.effectiveMeans.every(mean => mean === 1));
  assert.ok(open.openFractions.every(fraction => fraction === 1));
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

test("an unchanged static world reuses subcell arrays without integration or uploads", () => {
  const world = createSolidWorld();
  const previous = compileRetainedSceneSubcellMoments(ramp, dimensions, h, world);
  const next = compileRetainedSceneSubcellMoments(ramp, dimensions, h, world, { previous });
  assert.equal(next.seedAmounts, previous.seedAmounts);
  assert.equal(next.openVolumes, previous.openVolumes);
  assert.deepEqual(next.changedCellRanges, []);
  assert.equal(next.receipt.reusedCells, 512); assert.equal(next.receipt.recomputedCells, 0);
  assert.equal(next.receipt.integratedSubcells, 0);
  assert.equal(next.receipt.estimatedAbsoluteError, previous.receipt.estimatedAbsoluteError);
});

test("local static edits recompute only changed supports using exact ramp antiderivatives", () => {
  const world = createSolidWorld();
  const previous = compileRetainedSceneSubcellMoments(ramp, dimensions, h, world);
  const oldSeed = previous.seedAmounts.slice(), oldOpen = previous.openVolumes.slice();
  const solidFractions = new Uint8Array(512);
  solidFractions[0] = 64; solidFractions[1] = 128; solidFractions[3] = 255;
  const next = compileRetainedSceneSubcellMoments(ramp, dimensions, h, world, { previous, solidFractions });
  assert.deepEqual(next.changedCellRanges, [{ firstCell: 0, cellCount: 2 }, { firstCell: 3, cellCount: 1 }]);
  assert.equal(next.receipt.reusedCells, 509); assert.equal(next.receipt.recomputedCells, 3);
  assert.ok(next.receipt.integratedSubcells <= 16, "unchanged fluid interfaces must not pay for quadrature");
  for (let cell = 0; cell < 512; cell++) for (let octant = 0; octant < 8; octant++) {
    const at = 8 * cell + octant;
    if (!solidFractions[cell]) {
      assert.equal(next.seedAmounts[at], oldSeed[at]); assert.equal(next.openVolumes[at], oldOpen[at]); continue;
    }
    const lower = Math.max((octant & 2) ? .5 : 0, solidFractions[cell] / 255);
    const upper = (octant & 2) ? 1 : .5;
    close(next.openVolumes[at], Math.max(0, upper - lower) / 4, 1e-8);
    close(next.seedAmounts[at], upper > lower ? ((1 - lower) ** 2 - (1 - upper) ** 2) / 8 : 0, 1e-8);
  }
  assert.deepEqual(previous.seedAmounts, oldSeed); assert.deepEqual(previous.openVolumes, oldOpen);
  solidFractions[0] = 255;
  assert.equal(next.solidFractions[0], 64, "the cache owns its occupancy snapshot");
});

test("reopening a static support restores seed geometry without mutating the closed snapshot", () => {
  const world = createSolidWorld();
  const solidFractions = new Uint8Array(512); solidFractions[0] = 255;
  const closed = compileRetainedSceneSubcellMoments(ramp, dimensions, h, world, { solidFractions });
  const opened = compileRetainedSceneSubcellMoments(ramp, dimensions, h, world, { previous: closed });
  assert.deepEqual(opened.changedCellRanges, [{ firstCell: 0, cellCount: 1 }]);
  assert.equal(opened.receipt.recomputedCells, 1);
  for (let octant = 0; octant < 8; octant++) {
    assert.equal(closed.seedAmounts[octant], 0); assert.equal(closed.openVolumes[octant], 0);
    assert.equal(opened.seedAmounts[octant], octant & 2 ? .03125 : .09375);
    assert.equal(opened.openVolumes[octant], .125);
  }
  // These are seed basis moments, not an instruction to restore transported
  // liquid: the resident remains responsible for accepted a,b and edit mass.
});

test("an edited curved support retains exact quadratic moments across its clipped octants", () => {
  const field = retainedSceneDensity({ generation: 2, transitionWidth: .5,
    domain: ramp.domain, primitives: [{ kind: "ellipsoid", center: [0, .5, 0], radii: [.3, .3, .3] }] });
  const world = createSolidWorld(), previous = compileRetainedSceneSubcellMoments(field, dimensions, h, world);
  const cell = 4 + 8 * (4 + 8 * 4), solidFractions = previous.solidFractions.slice(); solidFractions[cell] = 64;
  const next = compileRetainedSceneSubcellMoments(field, dimensions, h, world, { previous, solidFractions });
  const p = field.primitives[0]; assert.ok(p.kind === "ellipsoid");
  const radius = p.radii[0], alpha = .5 + radius / (2 * field.transitionWidth), beta = 1 / (2 * field.transitionWidth * radius);
  const squareMean = (a: number, b: number) => (a * a + a * b + b * b) / 3;
  for (let octant = 0; octant < 8; octant++) {
    const x0 = (octant & 1 ? .5 : 0) * h, x1 = x0 + .5 * h;
    const z0 = (octant & 4 ? .5 : 0) * h, z1 = z0 + .5 * h;
    const y0 = Math.max(octant & 2 ? .5 : 0, 64 / 255) * h, y1 = (octant & 2 ? 1 : .5) * h;
    // This entire subbox lies in the unclamped quadratic transition, so the
    // independent tensor-product monomial integral has no numerical oracle.
    const density = alpha - beta * (squareMean(x0, x1) + squareMean(y0, y1) + squareMean(z0, z1));
    const fraction = (y1 - y0) / (4 * h);
    close(next.seedAmounts[8 * cell + octant], density * fraction, 1e-8);
    close(next.openVolumes[8 * cell + octant], fraction, 1e-8);
  }
  assert.equal(next.receipt.recomputedCells, 1);
  assert.equal(next.receipt.integratedSubcells, 8);
});

test("cache reuse checks numerical field identity and requested accuracy", () => {
  const world = createSolidWorld();
  const previous = compileRetainedSceneSubcellMoments(ramp, dimensions, h, world);
  const changed = retainedSceneDensity({ ...ramp, primitives: [{ kind: "quadratic-height",
    center: [0, 2 * h, 0], curvature: [0, 0, 0] }] });
  assert.throws(() => compileRetainedSceneSubcellMoments(changed, dimensions, h, world, { previous }), /numeric field and lattice/);
  assert.throws(() => compileRetainedSceneSubcellMoments(ramp, [4, 16, 8], h, world, { previous }), /numeric field and lattice/);
  assert.throws(() => compileRetainedSceneSubcellMoments(ramp, dimensions, h, world,
    { previous, absoluteMeanTolerance: 1e-10 }), /tighter integral tolerance/);
  const cloned = structuredClone(previous);
  const next = compileRetainedSceneSubcellMoments(ramp, dimensions, h, world, { previous: cloned });
  assert.equal(next.receipt.reusedCells, 512, "cache provenance survives generation preparation transfer");
});
