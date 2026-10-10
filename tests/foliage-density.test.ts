import assert from "node:assert/strict";
import test from "node:test";
import { sampleSvoPrimitive, type SvoSmoothUnionClusterPrimitive } from "../lib/svo/contracts/svo-primitive-abi";
import { planBonsai, BONSAI_POND_CANOPY } from "../lib/core/voxel-scenery/bonsai";
import type { SceneryGroupNode } from "../lib/core/scenery-graph";

const seeds = [1, 4258, 0xffffffff];
const descriptor = (seed: number, filterWidth_m = 0): SvoSmoothUnionClusterPrimitive => ({
  kind: "smooth-union-cluster", primitiveId: 0, materialId: 1, clusterReference: 0,
  center_m: { x: 0, y: 0, z: 0 }, lobeRadii_m: { x: .065, y: .04, z: .056 },
  packing: { field: "noise-foliage", seed, filterWidth_m, smoothRadius_m: 0, clusterPeriod_m: .038,
    detailPeriod_m: .028, threshold: .5, clusterWeight: .32, detailWeight: .68, interiorBias: .06 },
});
const points = Array.from({ length: 8192 }, (_, i) => ({
  x: (((i * 73) % 1009) / 1009 - .5) * .14,
  y: (((i * 193) % 1013) / 1013 - .5) * .09,
  z: (((i * 311) % 1019) / 1019 - .5) * .12,
}));

test("leaf density is continuous across folded cells and retains a conservative distance bound", () => {
  for (const seed of seeds) for (const filterWidth_m of [0, .003125, .00625]) {
    const shape = descriptor(seed, filterWidth_m);
    for (const point of points.slice(0, 1024)) {
      const other = { x: point.x + .00013, y: point.y - .00023, z: point.z + .00017 };
      const a = sampleSvoPrimitive(shape, point).signedDistance_m;
      const b = sampleSvoPrimitive(shape, other).signedDistance_m;
      assert.ok(Math.abs(a - b) <= Math.hypot(.00013, .00023, .00017) + 1e-10);
    }
    for (const axis of ["x", "y", "z"] as const) {
      for (let cell = -2; cell <= 2; cell++) {
        const p = { x: .003, y: -.002, z: .001, [axis]: (cell + .5) * Math.max(.028, 6 * filterWidth_m) };
        const a = sampleSvoPrimitive(shape, { ...p, [axis]: p[axis] - 1e-8 }).signedDistance_m;
        const b = sampleSvoPrimitive(shape, { ...p, [axis]: p[axis] + 1e-8 }).signedDistance_m;
        assert.ok(Math.abs(a - b) <= 2.01e-8, `${axis} cell ${cell}: fold discontinuity`);
      }
    }
  }
});

test("leaf coverage survives coarse sampling and converges across lattice phases", () => {
  const shape = descriptor(4258);
  const occupied = (h: number, phase: number, field = shape) => points.reduce((n, p) => {
    const snap = (v: number) => h ? (Math.floor(v / h - phase) + phase + .5) * h : v;
    return n + Number(sampleSvoPrimitive(field, { x: snap(p.x), y: snap(p.y), z: snap(p.z) }).signedDistance_m < 0);
  }, 0);
  const reference = occupied(0, 0);
  assert.ok(reference > 60, "a sparse leaf field must retain measurable occupied volume");
  for (const depth of [0, 2, 3]) {
    const samples = [0, .31, .67].map(phase => occupied(.00625 / 2 ** depth, phase));
    const ratio = samples.reduce((a, b) => a + b, 0) / (samples.length * reference);
    assert.ok(ratio > .7 && ratio < 1.3, `depth ${depth}: occupied-volume ratio ${ratio}`);
    const h = .00625 / 2 ** depth;
    const filtered = [0, .31, .67].map(phase => occupied(h, phase, descriptor(4258, h)));
    const retained = filtered.reduce((a, b) => a + b, 0) / filtered.length;
    assert.ok(retained > .7 * reference && retained < .15 * points.length,
      `depth ${depth}: filtered leaves must retain coverage without filling the spray (${retained})`);
  }
});

test("bonsai field canopy publishes bounded matte leaf clusters at every depth", () => {
  const plans = [0, 2, 3].map(depth => planBonsai({ ...BONSAI_POND_CANOPY, key: "shrub", seed: 4258,
    at_m: [0, 0], groundHeightAt: () => 0, lean: [1, 0], leafSize_m: .00625 / 2 ** depth }));
  const pads = plans.map(plan => (plan.nodes[0] as SceneryGroupNode).children.filter(n => n.id.includes("/pad-")));
  assert.equal(pads[0].length, 7);
  assert.deepEqual(pads[0], pads[1]);
  assert.deepEqual(pads[1], pads[2]);
  for (const pad of pads[0]) {
    assert.ok(pad.kind === "cluster" && pad.field === "noise-foliage");
    assert.equal(pad.material.surface, "foliage");
  }
  for (const plan of plans) assert.ok(plan.leafCount <= BONSAI_POND_CANOPY.maximumLeaves);
});

test("sampling footprint survives both cluster arena publication paths", async () => {
  const { packSvoClusterArena } = await import("../lib/svo/features/construction/svo-cluster-arena");
  const { packSvoDrySceneClusters, svoDrySceneClusterReference, svoDrySceneClusterResolver } = await import("../lib/svo/pipeline/webgpu-svo-dry-scene");
  const packing = descriptor(4258, .00625).packing!;
  const live = packSvoClusterArena([packing]);
  const dry = packSvoDrySceneClusters([packing]);
  assert.deepEqual(dry.slice(0, live.length), live);
  const decoded = svoDrySceneClusterResolver(dry)(svoDrySceneClusterReference(0));
  assert.ok(decoded?.field === "noise-foliage");
  assert.ok(Math.abs(decoded.filterWidth_m! - .00625) < 1e-8);
});
