import assert from "node:assert/strict";
import test from "node:test";

import type { Vec3 } from "../lib/model";
import {
  canonicalSvoPrimitive,
  intersectSvoPrimitive,
  packSvoPrimitiveRecords,
  sampleSvoPrimitive,
  svoPrimitiveBoundingRadius_m,
  svoPrimitiveLocalExtent_m,
  svoPrimitiveWGSL,
  SVO_PRIMITIVE_KINDS,
  SVO_PRIMITIVE_MARCH_ITERATIONS,
  SVO_SMOOTH_UNION_CLUSTER_MAXIMUM_JITTER,
  unpackSvoPrimitiveRecords,
  type SvoFinitePrimitiveDescriptor,
  type SvoPrimitiveDescriptor,
} from "../lib/svo-primitive-abi";
import {
  SVO_PRIMITIVE_KIND_ENTRIES,
  SVO_PRIMITIVE_KIND_TABLE,
  type SvoPrimitiveKindName,
} from "../lib/svo-primitive-kinds";
import {
  packSvoDrySceneClusters,
  svoDrySceneClusterReference,
  svoDrySceneClusterResolver,
} from "../lib/webgpu-svo-dry-scene";

/**
 * One representative of every kind, as a `Record` rather than a list.
 *
 * That is the point of the whole file: a kind added to the table without a
 * fixture here does not compile, so the completeness this suite claims is
 * checked by the type system rather than by a count somebody has to remember to
 * bump. Terrain is the one deliberate `null` — it is not a finite solid and its
 * ray hit belongs to the separate heightfield tracer.
 */
const CLUSTER_REFERENCE = svoDrySceneClusterReference(0);

const CLUSTER_PACKING = Object.freeze({
  // The regime `docs/HERO_GARDEN_AGGREGATE_SDF_ASSESSMENT.md` §3 measured at zero
  // missed rays: 10 mm lattice lobes on a 16 mm period.
  latticeLobeRadius_m: 0.010,
  latticePeriod_m: 0.016,
  jitter: SVO_SMOOTH_UNION_CLUSTER_MAXIMUM_JITTER,
  smoothRadius_m: 0.006,
  seed: 0x5701_e5,
});

const FIXTURES: Record<SvoPrimitiveKindName, SvoFinitePrimitiveDescriptor | null> = {
  sphere: { kind: "sphere", primitiveId: 1, materialId: 3, center_m: { x: 0, y: 0, z: 0 }, radius_m: 0.09 },
  box: { kind: "box", primitiveId: 2, materialId: 3, center_m: { x: 0, y: 0, z: 0 }, halfExtents_m: { x: 0.07, y: 0.05, z: 0.11 } },
  capsule: { kind: "capsule", primitiveId: 3, materialId: 3, center_m: { x: 0, y: 0, z: 0 }, radius_m: 0.03, segmentHalfLength_m: 0.08 },
  cylinder: { kind: "cylinder", primitiveId: 4, materialId: 3, center_m: { x: 0, y: 0, z: 0 }, radius_m: 0.05, halfHeight_m: 0.12 },
  ellipsoid: { kind: "ellipsoid", primitiveId: 5, materialId: 3, center_m: { x: 0, y: 0, z: 0 }, radii_m: { x: 0.12, y: 0.06, z: 0.09 } },
  "terrain-heightfield": null,
  torus: { kind: "torus", primitiveId: 7, materialId: 3, center_m: { x: 0, y: 0, z: 0 }, majorRadius_m: 0.1, minorRadius_m: 0.025 },
  cone: { kind: "cone", primitiveId: 8, materialId: 3, center_m: { x: 0, y: 0, z: 0 }, baseRadius_m: 0.08, topRadius_m: 0.02, halfHeight_m: 0.1 },
  "smooth-union-cluster": {
    kind: "smooth-union-cluster",
    primitiveId: 9,
    materialId: 3,
    center_m: { x: 0, y: 0, z: 0 },
    lobeRadii_m: { x: 0.15, y: 0.09, z: 0.15 },
    clusterReference: CLUSTER_REFERENCE,
    packing: CLUSTER_PACKING,
  },
  "round-cone": { kind: "round-cone", primitiveId: 10, materialId: 3, center_m: { x: 0, y: 0, z: 0 }, baseRadius_m: 0.08, topRadius_m: 0.05, halfHeight_m: 0.1 },
  "rounded-cylinder": {
    kind: "rounded-cylinder", primitiveId: 11, materialId: 3,
    center_m: { x: 0, y: 0, z: 0 }, radius_m: 0.08, halfHeight_m: 0.03, edgeRadius_m: 0.01,
  },
};

/** Deterministic 32-bit generator; the oracles below must not move between runs. */
function generator(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
    return state / 0x1_0000_0000;
  };
}

test("the kind table is frozen, its codes are stable, and the WGSL is generated from it", () => {
  assert.ok(Object.isFrozen(SVO_PRIMITIVE_KIND_TABLE), "the catalog must be frozen, like every other in-tree catalog");
  const codes = SVO_PRIMITIVE_KIND_ENTRIES.map((entry) => entry.code);
  assert.equal(new Set(codes).size, codes.length, "two kinds sharing a tag would reinterpret published records");
  assert.ok(codes.every((code) => Number.isInteger(code) && code > 0), "zero stays reserved for an invalid record");

  // The historical enumeration is the ABI other tests, captures and golden
  // fingerprints already hold. It is what "stable" means here.
  assert.deepEqual(SVO_PRIMITIVE_KINDS, {
    sphere: 1, box: 2, capsule: 3, cylinder: 4, ellipsoid: 5,
    terrainHeightfield: 6, torus: 7, cone: 8, smoothUnionCluster: 9, roundCone: 10, roundedCylinder: 11,
  });

  for (const entry of SVO_PRIMITIVE_KIND_ENTRIES) {
    assert.ok(
      svoPrimitiveWGSL.includes(`const ${entry.wgslConstant}: u32 = ${entry.code}u;`),
      `${entry.name} has no generated WGSL constant, so the shader would compare against a tag that does not exist`,
    );
    assert.ok(
      svoPrimitiveWGSL.includes(entry.wgslConstant),
      `${entry.name}'s WGSL constant is declared and never read, which is the shape of a silent fallthrough`,
    );
  }
});

test("every kind's local extent contains its own surface, and does not wildly exceed it", () => {
  // This is the assertion that stops a clipped silhouette. The extent is the
  // raster proxy box, the voxelizer's dirty region, the candidate BVH leaf and
  // the sway budget all at once; too small and geometry quietly leaves each of
  // them, with a different symptom in every path.
  for (const [name, fixture] of Object.entries(FIXTURES) as [SvoPrimitiveKindName, SvoFinitePrimitiveDescriptor | null][]) {
    if (!fixture) {
      assert.ok(
        svoPrimitiveLocalExtent_m(fixture ?? { kind: "terrain-heightfield", primitiveId: 6, materialId: 3, terrainReference: 4 }).x < 0,
        "a kind with no finite local box must say so rather than report zero",
      );
      continue;
    }
    const extent = svoPrimitiveLocalExtent_m(fixture);
    const radius = svoPrimitiveBoundingRadius_m(fixture);
    assert.ok(radius > 0, `${name} reports a non-positive bounding radius`);
    assert.ok(
      radius <= Math.hypot(extent.x, extent.y, extent.z) * (1 + 1e-9),
      `${name}'s bounding radius ${radius} exceeds the corner of its own local box`,
    );

    // Walk the box's own surface outward and inward. Every point outside the
    // extent must be outside the solid; some point just inside it must be on or
    // in the solid, or the extent is loose enough to be hiding a wrong formula.
    const random = generator(0x9e37_79b1 ^ fixture.primitiveId);
    let deepestInside = Number.POSITIVE_INFINITY;
    for (let sample = 0; sample < 4_000; sample += 1) {
      const point: Vec3 = {
        x: (random() * 2 - 1) * extent.x * 1.5,
        y: (random() * 2 - 1) * extent.y * 1.5,
        z: (random() * 2 - 1) * extent.z * 1.5,
      };
      const distance = sampleSvoPrimitive(fixture, point).signedDistance_m;
      const outsideBox = Math.abs(point.x) > extent.x || Math.abs(point.y) > extent.y || Math.abs(point.z) > extent.z;
      if (outsideBox) {
        assert.ok(distance > -1e-9, `${name} is solid at ${JSON.stringify(point)}, outside its declared extent`);
      } else if (distance <= 0) {
        deepestInside = Math.min(deepestInside, Math.max(
          Math.abs(point.x) / extent.x, Math.abs(point.y) / extent.y, Math.abs(point.z) / extent.z,
        ));
      }
    }
    assert.ok(Number.isFinite(deepestInside), `${name} never reported a solid sample inside its own extent`);
  }
});

test("a marched aggregate converges without stepping over its own surface", () => {
  // The test that stops displacement being reintroduced. A displaced envelope
  // has |grad d| > 1, so a trace stepping by |d| overshoots and tunnels; the
  // measured loss for that construction at this scale is 58.9 % of rays. A
  // smooth minimum understeps and cannot. Scored against a dense fixed-step
  // ground truth rather than against itself.
  const cluster = FIXTURES["smooth-union-cluster"]!;
  assert.equal(cluster.kind, "smooth-union-cluster");
  const boundingRadius = svoPrimitiveBoundingRadius_m(cluster);
  const random = generator(0x2545_f491);
  const groundTruthStep = CLUSTER_PACKING.latticeLobeRadius_m / 40;

  let tested = 0;
  let tunnelled = 0;
  let overshot = 0;
  const shortfall_m: number[] = [];
  for (let index = 0; index < 600; index += 1) {
    // A ray from the bounding sphere aimed at a point well inside the lobe, so
    // every ray in the set genuinely crosses the packing.
    const theta = random() * 2 * Math.PI;
    const phi = Math.acos(2 * random() - 1);
    const origin = {
      x: 3 * boundingRadius * Math.sin(phi) * Math.cos(theta),
      y: 3 * boundingRadius * Math.cos(phi),
      z: 3 * boundingRadius * Math.sin(phi) * Math.sin(theta),
    };
    const aim = {
      x: (random() * 2 - 1) * 0.4 * cluster.lobeRadii_m.x,
      y: (random() * 2 - 1) * 0.4 * cluster.lobeRadii_m.y,
      z: (random() * 2 - 1) * 0.4 * cluster.lobeRadii_m.z,
    };
    const delta = { x: aim.x - origin.x, y: aim.y - origin.y, z: aim.z - origin.z };
    const length = Math.hypot(delta.x, delta.y, delta.z);
    const direction = { x: delta.x / length, y: delta.y / length, z: delta.z / length };

    // Ground truth: a fixed step far below the lattice lobe radius, then bisection.
    let truth: number | undefined;
    let previous = 3 * boundingRadius - boundingRadius * 2;
    for (let t = previous; t <= 6 * boundingRadius; t += groundTruthStep) {
      const point = { x: origin.x + direction.x * t, y: origin.y + direction.y * t, z: origin.z + direction.z * t };
      if (sampleSvoPrimitive(cluster, point).signedDistance_m <= 0) {
        let low = previous, high = t;
        for (let iteration = 0; iteration < 40; iteration += 1) {
          const middle = 0.5 * (low + high);
          const sample = sampleSvoPrimitive(cluster, {
            x: origin.x + direction.x * middle, y: origin.y + direction.y * middle, z: origin.z + direction.z * middle,
          }).signedDistance_m;
          if (sample <= 0) high = middle; else low = middle;
        }
        truth = 0.5 * (low + high);
        break;
      }
      previous = t;
    }
    if (truth === undefined) continue;
    tested += 1;
    const hit = intersectSvoPrimitive(cluster, { origin_m: origin, direction });
    if (!hit) { tunnelled += 1; continue; }
    // The safety property, stated exactly. A Lipschitz-1 field stepped by |d|
    // can only ever stop at or *before* its own zero crossing, so a hit past
    // the dense one is proof the field oversteps somewhere — which is the
    // failure mode a displaced envelope has and a smooth minimum does not.
    if (hit.t_m > truth + 1e-9) overshot += 1;
    shortfall_m.push(truth - hit.t_m);
  }

  assert.ok(tested > 400, `only ${tested} of 600 rays crossed the packing; the fixture is not covering it`);
  assert.equal(tunnelled, 0, `${tunnelled} of ${tested} rays missed a surface the dense march found`);
  // Not zero, and the reason is worth stating because it is *not* an overstep.
  // A ray that grazes a lattice lobe tangentially can have the field dip to zero
  // between two march samples while never coming inside the acceptance band —
  // 37 microns at this range — so the march finds the next crossing instead.
  // The dense oracle steps at a fortieth of a lobe and catches the graze. The
  // engine's band is what buys the step count, and this is its known price; the
  // bound is on the rate rather than on the event. What must be zero is the
  // tunnelling above, which is the failure a non-Lipschitz field produces and
  // which a displaced envelope shows on three rays in five.
  assert.ok(overshot / tested < 0.01,
    `${overshot} of ${tested} rays reported a hit beyond the dense crossing, above the grazing allowance`);

  // How far *short* the march stops is not an error in the same sense: it stops
  // inside an acceptance band of max(1e-6, 1e-4 t), and on a grazing ray a
  // point 37 microns off the surface can be centimetres along the ray from the
  // crossing precisely because the surface is nearly parallel to it. So the
  // bound is on the bulk rather than the tail.
  shortfall_m.sort((left, right) => left - right);
  const median_m = shortfall_m[Math.floor(shortfall_m.length / 2)];
  const p95_m = shortfall_m[Math.floor(shortfall_m.length * 0.95)];
  assert.ok(median_m >= 0);
  assert.ok(median_m < 0.02 * CLUSTER_PACKING.latticeLobeRadius_m,
    `the median hit stops ${(median_m * 1e3).toFixed(3)} mm short of the dense crossing`);
  assert.ok(p95_m < 0.5 * CLUSTER_PACKING.latticeLobeRadius_m,
    `the 95th percentile stops ${(p95_m * 1e3).toFixed(2)} mm short of the dense crossing`);
  assert.ok(SVO_PRIMITIVE_MARCH_ITERATIONS === 48, "the shared iteration ceiling moved; re-measure before trusting this bound");
});

test("a cluster round-trips through the record and recovers its packing from the arena", () => {
  const cluster = FIXTURES["smooth-union-cluster"]!;
  const [recovered] = unpackSvoPrimitiveRecords(packSvoPrimitiveRecords([cluster]));
  assert.equal(recovered.kind, "smooth-union-cluster");
  assert.ok(recovered.kind === "smooth-union-cluster");
  // The record carries the lobe and the reference and nothing else. It must not
  // invent a packing: a default one renders a plausible smooth ellipsoid, which
  // is a shape the scene never asked for and no error would ever be raised for.
  assert.equal(recovered.packing, undefined);
  assert.equal(recovered.clusterReference, CLUSTER_REFERENCE);
  assert.ok(cluster.kind === "smooth-union-cluster");
  // Single precision, because the record is what the shader reads.
  for (const axis of ["x", "y", "z"] as const) {
    assert.equal(recovered.lobeRadii_m[axis], Math.fround(cluster.lobeRadii_m[axis]));
  }

  // The block is four f32s and a u32, so the packing comes back rounded to
  // single precision — which is the precision the shader evaluates it at, and
  // therefore the precision the CPU oracle has to agree with.
  const resolve = svoDrySceneClusterResolver(packSvoDrySceneClusters([CLUSTER_PACKING]));
  const recoveredPacking = resolve(CLUSTER_REFERENCE)!;
  assert.ok(recoveredPacking);
  // A packing authored with no field discriminant is the lattice, and it comes
  // back naming itself one: the block always carries the code, so the round
  // trip is where an omitted discriminant becomes explicit.
  assert.ok(recoveredPacking.field === undefined || recoveredPacking.field === "lattice");
  assert.equal(recoveredPacking.octaves, 1);
  assert.equal(recoveredPacking.seed, CLUSTER_PACKING.seed);
  for (const field of ["latticeLobeRadius_m", "latticePeriod_m", "jitter", "smoothRadius_m"] as const) {
    assert.equal(recoveredPacking[field], Math.fround(CLUSTER_PACKING[field]), `${field} did not survive the arena`);
  }
  // A reference that does not name a filled block resolves to nothing rather
  // than to a zeroed packing that would evaluate as a smooth lobe.
  assert.equal(resolve(svoDrySceneClusterReference(1)), undefined);
  assert.equal(resolve(CLUSTER_REFERENCE + 1), undefined);
  assert.equal(svoDrySceneClusterResolver(undefined)(CLUSTER_REFERENCE), undefined);

  // Evaluating a recovered record without a resolver is an error, not a guess.
  assert.throws(() => sampleSvoPrimitive(recovered as SvoPrimitiveDescriptor, { x: 0, y: 0, z: 0 }));
  assert.ok(sampleSvoPrimitive(recovered as SvoPrimitiveDescriptor, { x: 0, y: 0, z: 0 }, undefined, resolve).signedDistance_m < 0);
});

test("the jitter cap is enforced where a scene can reach it", () => {
  // Above the cap the eight-cell neighbourhood stops being a lower bound and
  // the trace oversteps. Rejecting it at authoring time is the only place that
  // can be caught, because the failure downstream is a hole, not an exception.
  const cluster = FIXTURES["smooth-union-cluster"]!;
  assert.ok(cluster.kind === "smooth-union-cluster");
  assert.throws(() => canonicalSvoPrimitive({
    ...cluster,
    packing: { ...CLUSTER_PACKING, jitter: SVO_SMOOTH_UNION_CLUSTER_MAXIMUM_JITTER + 0.01 },
  }), /jitter/);
  assert.throws(() => canonicalSvoPrimitive({
    ...cluster,
    packing: { ...CLUSTER_PACKING, smoothRadius_m: CLUSTER_PACKING.latticePeriod_m * 1.5 },
  }), /smooth-minimum/);
});
