import assert from "node:assert/strict";
import test from "node:test";

import { createHeroGardenHoseScene } from "../lib/hero-garden-scene";
import type { Quaternion, Vec3 } from "../lib/model";
import {
  canonicalSvoPrimitive,
  intersectSvoPrimitive,
  svoPrimitiveLocalExtent_m,
  type SvoFinitePrimitiveDescriptor,
} from "../lib/svo-primitive-abi";
import { buildSvoScenePrimitives } from "../lib/svo-scene-primitives";
import {
  SVO_SCENE_PRIMITIVE_MARCH_SPAN_PAD_M,
  SVO_SCENE_PRIMITIVE_MARCH_SPAN_PAD_RELATIVE,
} from "../lib/webgpu-svo-dry-scene";
import { withHeroLayout } from "../lib/voxel-scenery/hero-layout";

/**
 * `dryScenePrimitiveMarchSpan` brackets the scene-primitive raster's sphere
 * trace by the primitive's own oriented box instead of by the whole ray, which
 * is what makes the marched kinds' 48-iteration ceiling the bound the ABI says
 * it is. That is only sound if no exact hit can lie outside the box.
 *
 * Containment is a property of the kind table's `localExtent_m`, and a kind
 * whose extent is too small does not fail loudly — it removes a sliver of the
 * silhouette on the one path that carries most of the frame. So this asserts it
 * against the authored hero set rather than against a synthetic primitive, and
 * over rays aimed to graze as well as to hit.
 */

const rotate = (q: Quaternion, v: Vec3): Vec3 => {
  const t: Vec3 = {
    x: 2 * (q.y * v.z - q.z * v.y),
    y: 2 * (q.z * v.x - q.x * v.z),
    z: 2 * (q.x * v.y - q.y * v.x),
  };
  return {
    x: v.x + q.w * t.x + (q.y * t.z - q.z * t.y),
    y: v.y + q.w * t.y + (q.z * t.x - q.x * t.z),
    z: v.z + q.w * t.z + (q.x * t.y - q.y * t.x),
  };
};

/** The WGSL slab test, on the same inverse rotation `svoIntersectPrimitiveExact` uses. */
const marchSpan = (
  descriptor: SvoFinitePrimitiveDescriptor,
  origin: Vec3,
  direction: Vec3,
): { enter: number; exit: number } | undefined => {
  const extent = svoPrimitiveLocalExtent_m(descriptor);
  if (extent.x < 0 || extent.y < 0 || extent.z < 0) return undefined;
  // The record the shader reads is always canonical, so the bracket has to be
  // computed from the canonical orientation rather than the authored one.
  const canonical = canonicalSvoPrimitive(descriptor) as SvoFinitePrimitiveDescriptor;
  const raw: Quaternion = canonical.kind === "sphere"
    ? { w: 1, x: 0, y: 0, z: 0 }
    : canonical.orientation ?? { w: 1, x: 0, y: 0, z: 0 };
  const length = Math.hypot(raw.w, raw.x, raw.y, raw.z);
  const inverse: Quaternion = { w: raw.w / length, x: -raw.x / length, y: -raw.y / length, z: -raw.z / length };
  const centre = descriptor.center_m;
  const localOrigin = rotate(inverse, {
    x: origin.x - centre.x, y: origin.y - centre.y, z: origin.z - centre.z,
  });
  const localDirection = rotate(inverse, direction);
  let enter = 0, exit = Number.POSITIVE_INFINITY;
  for (const axis of ["x", "y", "z"] as const) {
    const o = localOrigin[axis], d = localDirection[axis], half = extent[axis];
    if (d === 0) {
      if (o < -half || o > half) return undefined;
      continue;
    }
    const first = (-half - o) / d, second = (half - o) / d;
    enter = Math.max(enter, Math.min(first, second));
    exit = Math.min(exit, Math.max(first, second));
    if (exit < enter) return undefined;
  }
  const pad = Math.max(SVO_SCENE_PRIMITIVE_MARCH_SPAN_PAD_M, SVO_SCENE_PRIMITIVE_MARCH_SPAN_PAD_RELATIVE * exit);
  return { enter: Math.max(0, enter - pad), exit: exit + pad };
};

test("every exact hit on the hero set lies inside its primitive's march span", () => {
  const base = createHeroGardenHoseScene();
  const scene = base.scenery ? { ...base, scenery: withHeroLayout(base) } : base;
  const descriptors = buildSvoScenePrimitives(scene)
    .descriptors.filter((candidate): candidate is SvoFinitePrimitiveDescriptor =>
      candidate.kind !== "terrain-heightfield");
  assert.ok(descriptors.length > 250, "the hero set should publish its full primitive census after foliage aggregation");

  // Deterministic ray fan: one seeded LCG, no Math.random, so a failure is
  // reproducible from the descriptor index and ray index alone.
  let seed = 0x9e3779b9;
  const next = (): number => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 0x1_0000_0000;
  };

  let hits = 0, spanless = 0;
  for (const [index, descriptor] of descriptors.entries()) {
    const extent = svoPrimitiveLocalExtent_m(descriptor);
    const reach = 4 * Math.max(extent.x, extent.y, extent.z) + 0.05;
    for (let ray = 0; ray < 24; ray += 1) {
      // Aim at a point in and slightly beyond the primitive's own box so the
      // fan includes grazing rays, which are where a tight bracket would bite.
      const theta = 2 * Math.PI * next(), phi = Math.acos(2 * next() - 1);
      const origin: Vec3 = {
        x: descriptor.center_m.x + reach * Math.sin(phi) * Math.cos(theta),
        y: descriptor.center_m.y + reach * Math.sin(phi) * Math.sin(theta),
        z: descriptor.center_m.z + reach * Math.cos(phi),
      };
      const aim: Vec3 = {
        x: descriptor.center_m.x + 1.4 * extent.x * (2 * next() - 1),
        y: descriptor.center_m.y + 1.4 * extent.y * (2 * next() - 1),
        z: descriptor.center_m.z + 1.4 * extent.z * (2 * next() - 1),
      };
      const delta = { x: aim.x - origin.x, y: aim.y - origin.y, z: aim.z - origin.z };
      const norm = Math.hypot(delta.x, delta.y, delta.z);
      if (!(norm > 1e-9)) continue;
      const direction: Vec3 = { x: delta.x / norm, y: delta.y / norm, z: delta.z / norm };

      const hit = intersectSvoPrimitive(descriptor, { origin_m: origin, direction });
      if (!hit) continue;
      hits += 1;
      const span = marchSpan(descriptor, origin, direction);
      if (!span) {
        spanless += 1;
        assert.fail(`primitive ${index} (${descriptor.kind}) hit at t=${hit.t_m} but its box rejected the ray`);
      }
      // No tolerance here: the span already carries the shipped pad, so any
      // slack added on top would be exactly the sliver this test exists to find.
      assert.ok(hit.t_m >= span.enter && hit.t_m <= span.exit,
        `primitive ${index} (${descriptor.kind}) hit t=${hit.t_m} outside span [${span.enter}, ${span.exit}]`);
    }
  }
  assert.equal(spanless, 0);
  assert.ok(hits > 2000, `the fan should land plenty of hits to bracket, got ${hits}`);
});
