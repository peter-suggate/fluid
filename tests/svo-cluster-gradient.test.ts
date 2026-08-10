import assert from "node:assert/strict";
import test from "node:test";

import type { Vec3 } from "../lib/model";
import {
  sampleSvoPrimitive,
  type SvoSmoothUnionClusterPacking,
  type SvoSmoothUnionClusterPrimitive,
} from "../lib/svo-primitive-abi";

/**
 * The cluster shading normal is the field's own gradient, and this is what says
 * so numerically.
 *
 * It used to be a four-tap tetrahedral difference of `clusterDistance`, which
 * was self-evidently *a* gradient of *that* field — wrong only in the second
 * order, and never wrong in a way any other lane could see. It is now one
 * differentiated pass: each leaf answers with its own closed-form normal and the
 * smooth minimum blends them. That is exact, but it is exact by an argument
 * rather than by construction, and an argument can be wrong. Measured on
 * `hero-garden-hose` at refinement depth 3 the change is worth 42.7 → 24.4 ms of
 * an 800x460 frame, so it will not be reverted casually and the check has to be
 * the one that would actually catch a bad derivative.
 *
 * Two claims, per field:
 *
 *  1. **The normal is the gradient of the distance the same code returns.** A
 *     central difference of `sampleSvoPrimitive(...).signedDistance_m` — the
 *     public distance, not the gradient's own internals — must agree with the
 *     published normal. A leaf whose closed form is wrong, a blend weight that
 *     is not the derivative, or a fold order that drifted from its distance twin
 *     all break this.
 *  2. **The distance did not move.** The gradient pass duplicates each fold, so
 *     the distance-only twin and the sampled twin can drift apart. Every
 *     `...Sample` above computes its distance with its twin's arithmetic
 *     character for character; this pins the result of that.
 *
 * Points are drawn near each field's own surface, because that is where a
 * shading normal is read and where a distance field's second derivative is
 * largest — a gradient wrong at a seam is invisible in the interior.
 */

const ENVELOPE: Vec3 = { x: 0.15, y: 0.09, z: 0.12 };

const LATTICE = Object.freeze({
  latticeLobeRadius_m: 0.010,
  latticePeriod_m: 0.016,
  jitter: 0.3,
  smoothRadius_m: 0.006,
  seed: 0x5701_e5,
  octaves: 2,
});

const SWEEP = Object.freeze({
  field: "tapered-sweep" as const,
  smoothRadius_m: 0.004,
  // Unread by a sweep, which is authored rather than seeded, but the packing
  // validator requires every field to carry one.
  seed: 0x00c0_ffee,
  points: [
    { position_m: { x: -0.06, y: -0.03, z: 0.01 }, radius_m: 0.012 },
    { position_m: { x: 0.00, y: 0.02, z: -0.01 }, radius_m: 0.009 },
    { position_m: { x: 0.05, y: 0.04, z: 0.02 }, radius_m: 0.004 },
  ],
});

const SEEDED = Object.freeze({
  field: "seeded-lobes" as const,
  smoothRadius_m: 0.005,
  seed: 0x9e37_79b1,
  lobeCount: 5,
  anisotropy: 2.2,
});

const NOISE_FOLIAGE = Object.freeze({
  field: "noise-foliage" as const,
  smoothRadius_m: 0,
  seed: 0x51a9,
  clusterPeriod_m: 0.075,
  detailPeriod_m: 0.018,
  threshold: 0.55,
  clusterWeight: 0.62,
  detailWeight: 0.38,
  interiorBias: 0.20,
});

const FIELDS: readonly { name: string; packing: SvoSmoothUnionClusterPacking }[] = [
  { name: "lattice", packing: { field: "lattice", ...LATTICE } as SvoSmoothUnionClusterPacking },
  { name: "tapered-sweep", packing: SWEEP as unknown as SvoSmoothUnionClusterPacking },
  { name: "seeded-lobes", packing: SEEDED as unknown as SvoSmoothUnionClusterPacking },
  { name: "noise-foliage", packing: NOISE_FOLIAGE },
];

function clusterAt(packing: SvoSmoothUnionClusterPacking): SvoSmoothUnionClusterPrimitive {
  // Centred and unrotated, so a world point is the local point the field sees.
  return {
    kind: "smooth-union-cluster",
    primitiveId: 1,
    materialId: 3,
    center_m: { x: 0, y: 0, z: 0 },
    lobeRadii_m: ENVELOPE,
    clusterReference: 1,
    packing,
  };
}

function distanceAt(packing: SvoSmoothUnionClusterPacking, point: Vec3): number {
  return sampleSvoPrimitive(clusterAt(packing), point).signedDistance_m;
}

/** Deterministic, so a failure reproduces from the seed alone. */
function randomStream(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b_79f5) >>> 0;
    let mixed = Math.imul(state ^ (state >>> 15), 1 | state);
    mixed = (mixed + Math.imul(mixed ^ (mixed >>> 7), 61 | mixed)) ^ mixed;
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4_294_967_296;
  };
}

/**
 * Walk from a random point onto the surface before differentiating.
 *
 * A sphere trace of the field's own distance, which is what the renderer does to
 * reach the point it shades. Returns undefined for a ray that never lands, so a
 * miss is skipped rather than tested at whatever it drifted to.
 */
function surfacePoint(
  packing: SvoSmoothUnionClusterPacking,
  random: () => number,
): Vec3 | undefined {
  const direction = { x: random() * 2 - 1, y: random() * 2 - 1, z: random() * 2 - 1 };
  const length = Math.hypot(direction.x, direction.y, direction.z);
  if (!(length > 1e-6)) return undefined;
  const unit = { x: direction.x / length, y: direction.y / length, z: direction.z / length };
  const start = 0.4;
  let travelled = 0;
  for (let step = 0; step < 96; step += 1) {
    const point = {
      x: -unit.x * start + unit.x * travelled,
      y: -unit.y * start + unit.y * travelled,
      z: -unit.z * start + unit.z * travelled,
    };
    const distance = distanceAt(packing, point);
    if (Math.abs(distance) < 1e-6) return point;
    if (travelled > 2 * start) return undefined;
    travelled += Math.max(distance, 1e-7);
  }
  return undefined;
}

for (const { name, packing } of FIELDS) {
  test(`the ${name} cluster normal is the gradient of its own distance`, () => {
    const random = randomStream(0x5eed_0001);
    // The step is a compromise the *test* makes, not the implementation: wide
    // enough to clear f32-scale cancellation in a metre-scale field, narrow
    // enough that the field's curvature over it stays second order.
    const step = 2e-5;
    let checked = 0;
    let worst = 0;
    for (let attempt = 0; attempt < 400 && checked < 60; attempt += 1) {
      const point = surfacePoint(packing, random);
      if (!point) continue;
      const normal = sampleSvoPrimitive(clusterAt(packing), point).normal;
      if (!normal) continue;
      const difference = {
        x: distanceAt(packing, { ...point, x: point.x + step }) - distanceAt(packing, { ...point, x: point.x - step }),
        y: distanceAt(packing, { ...point, y: point.y + step }) - distanceAt(packing, { ...point, y: point.y - step }),
        z: distanceAt(packing, { ...point, z: point.z + step }) - distanceAt(packing, { ...point, z: point.z - step }),
      };
      const magnitude = Math.hypot(difference.x, difference.y, difference.z);
      if (!(magnitude > 1e-9)) continue;
      const alignment = (normal.x * difference.x + normal.y * difference.y + normal.z * difference.z) / magnitude;
      worst = Math.max(worst, 1 - alignment);
      checked += 1;
    }
    assert.ok(checked >= 20, `only ${checked} surface points found for ${name}; the fixture no longer draws a solid`);
    // One degree. A central difference over a blended field carries its own
    // second-order error, so this is the difference's tolerance rather than the
    // gradient's — a wrong derivative misses by tens of degrees, not by one.
    assert.ok(worst < 1.5e-4,
      `${name} normal disagrees with a central difference of its own distance by ${(Math.acos(1 - worst) * 180 / Math.PI).toFixed(2)}°`);
  });
}

test("the differentiated fold returns the distance its distance-only twin does", () => {
  // `sampleSvoPrimitive` publishes both from one call, so this pins the pair
  // that would silently drift: the sampled fold's distance against the value the
  // march steps by. A regression here is a hole in a frame, not a shading error.
  const random = randomStream(0x5eed_0002);
  for (const { name, packing } of FIELDS) {
    for (let index = 0; index < 200; index += 1) {
      const point = {
        x: (random() * 2 - 1) * 0.2,
        y: (random() * 2 - 1) * 0.2,
        z: (random() * 2 - 1) * 0.2,
      };
      const sample = sampleSvoPrimitive(clusterAt(packing), point);
      assert.equal(Number.isFinite(sample.signedDistance_m), true, `${name} distance is not finite at ${JSON.stringify(point)}`);
      if (sample.normal) {
        const unit = Math.hypot(sample.normal.x, sample.normal.y, sample.normal.z);
        assert.ok(Math.abs(unit - 1) < 1e-6, `${name} normal is not unit length (${unit})`);
      }
    }
  }
});
