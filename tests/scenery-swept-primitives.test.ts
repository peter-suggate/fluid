import assert from "node:assert/strict";
import test from "node:test";

import { cloneScene, defaultScene } from "../lib/model";
import {
  intersectSvoPrimitive,
  sampleSvoPrimitive,
  unpackSvoPrimitiveRecords,
  type SvoFinitePrimitiveDescriptor,
} from "../lib/svo-primitive-abi";
import { svoScenePrimitivesFromEnvironmentCatalog } from "../lib/svo-scene-primitives";
import { buildEnvironmentProxyCatalog, environmentProxyPrimitives } from "../lib/voxel-environments";
import {
  sparseScenePrimitiveForProxy,
  sparseScenePrimitiveSignedDistance,
} from "../lib/webgpu-sparse-scene-proxies";

/**
 * Scenery used to be axis-aligned, so anything diagonal or curved was spelled
 * as a chain of ellipsoid beads standing in for a rotation the authoring layer
 * could not express. These tests hold the shapes that replaced them: the run is
 * authored where it goes, and the rotation survives every hop to the GPU.
 */

const scene = cloneScene(defaultScene);
const catalog = buildEnvironmentProxyCatalog(scene, "conservatory");
const proxies = environmentProxyPrimitives(catalog);
const build = svoScenePrimitivesFromEnvironmentCatalog(scene, catalog);
const proxyFor = (suffix: string) => {
  const proxy = proxies.find((entry) => entry.key.endsWith(suffix));
  assert.ok(proxy, `conservatory has ${suffix}`);
  return proxy;
};
const descriptorFor = (suffix: string) => {
  const index = proxies.findIndex((entry) => entry.key.endsWith(suffix));
  assert.ok(index >= 0, `conservatory has ${suffix}`);
  return build.descriptors[index] as SvoFinitePrimitiveDescriptor;
};

test("the conservatory hose is swept solids rather than a chain of beads", () => {
  const hose = proxies.filter((proxy) => proxy.group === "rubber-hose");
  assert.deepEqual(hose.map((proxy) => proxy.kind), ["torus", "torus", "capsule", "capsule", "capsule", "capsule"]);
  // One tube radius for the whole run: a hose does not change gauge at a bend.
  const radii = new Set(hose.map((proxy) => (proxy.kind === "torus" ? proxy.minorRadius_m : proxy.kind === "capsule" ? proxy.radius_m : Number.NaN)));
  assert.equal(radii.size, 2, "the two coiled turns and the run each keep one gauge");

  // Consecutive legs share an endpoint, so the run has no gaps to fall through.
  const legs = hose.filter((proxy) => proxy.kind === "capsule");
  for (let index = 1; index < legs.length; index += 1) {
    const previous = legs[index - 1], next = legs[index];
    if (previous.kind !== "capsule" || next.kind !== "capsule") return;
    const gap = Math.hypot(
      previous.center_m.x - next.center_m.x,
      previous.center_m.y - next.center_m.y,
      previous.center_m.z - next.center_m.z,
    );
    assert.ok(gap <= previous.halfLength_m + next.halfLength_m + 1e-9,
      `hose legs ${index - 1} and ${index} must meet, centres are ${gap.toFixed(4)} apart`);
  }
});

test("an authored run reaches the render ABI as the segment it was authored as", () => {
  const proxy = proxyFor("hose/run-1");
  assert.equal(proxy.kind, "capsule");
  if (proxy.kind !== "capsule") return;
  const descriptor = descriptorFor("hose/run-1");
  assert.equal(descriptor.kind, "capsule");
  if (descriptor.kind !== "capsule") return;
  assert.ok(descriptor.orientation, "a rotated run must carry its rotation into the record");

  // Both authored endpoints, and everything between them, sit exactly one tube
  // radius inside the solid. That only holds if the derived rotation is right.
  const axis = descriptor.orientation!;
  const rotate = (v: { x: number; y: number; z: number }) => {
    const tx = 2 * (axis.y * v.z - axis.z * v.y), ty = 2 * (axis.z * v.x - axis.x * v.z), tz = 2 * (axis.x * v.y - axis.y * v.x);
    return {
      x: v.x + axis.w * tx + (axis.y * tz - axis.z * ty),
      y: v.y + axis.w * ty + (axis.z * tx - axis.x * tz),
      z: v.z + axis.w * tz + (axis.x * ty - axis.y * tx),
    };
  };
  for (const along of [-1, -0.5, 0, 0.5, 1]) {
    const offset = rotate({ x: 0, y: along * descriptor.segmentHalfLength_m, z: 0 });
    const point = {
      x: descriptor.center_m.x + offset.x,
      y: descriptor.center_m.y + offset.y,
      z: descriptor.center_m.z + offset.z,
    };
    const distance = sampleSvoPrimitive(descriptor, point).signedDistance_m;
    assert.ok(Math.abs(distance + descriptor.radius_m) <= 1e-9,
      `centreline sample ${along} is ${distance}, expected ${-descriptor.radius_m}`);
  }

  // The packed record is what the GPU actually reads.
  const packedIndex = build.descriptors.indexOf(descriptor);
  const restored = unpackSvoPrimitiveRecords(build.packedRecords)[packedIndex];
  assert.equal(restored.kind, "capsule");
  if (restored.kind !== "capsule") return;
  assert.ok(Math.abs(sampleSvoPrimitive(restored, descriptor.center_m).signedDistance_m + descriptor.radius_m) <= 1e-5);
});

test("a coiled turn is one ring with a hole, not a stack of overlapping blobs", () => {
  const descriptor = descriptorFor("hose/coil-outer");
  assert.equal(descriptor.kind, "torus");
  if (descriptor.kind !== "torus") return;
  const centre = descriptor.center_m;
  const above = { x: centre.x, y: centre.y + 2, z: centre.z };
  const axisRay = { origin_m: above, direction: { x: 0, y: -1, z: 0 }, tMin_m: 0, tMax_m: 4 };
  assert.equal(intersectSvoPrimitive(descriptor, axisRay), null, "the middle of a coil is empty");

  const overTube = { ...axisRay, origin_m: { x: centre.x + descriptor.majorRadius_m, y: centre.y + 2, z: centre.z } };
  const hit = intersectSvoPrimitive(descriptor, overTube);
  assert.ok(hit, "a ray over the tube hits it");
  assert.ok(Math.abs(hit.position_m.y - (centre.y + descriptor.minorRadius_m)) <= 1e-3,
    "and lands on the top of the tube rather than somewhere inside it");
});

test("voxelization and rendering agree on where every authored shape is solid", () => {
  // The voxelizer decides which owner a cell belongs to and the renderer then
  // refines that owner's surface analytically. If the two disagreed about
  // solidity, a cell would name an owner whose surface the ray never finds.
  for (const [index, proxy] of proxies.entries()) {
    const descriptor = build.descriptors[index] as SvoFinitePrimitiveDescriptor;
    const voxelized = sparseScenePrimitiveForProxy(proxy, { materialId: 1, ownerId: 2 });
    assert.equal(voxelized.center[0], proxy.center_m.x);
    const reach = Math.max(
      proxy.aabb_m.max.x - proxy.aabb_m.min.x,
      proxy.aabb_m.max.y - proxy.aabb_m.min.y,
      proxy.aabb_m.max.z - proxy.aabb_m.min.z,
    );
    let compared = 0;
    for (let sample = 0; sample < 48; sample += 1) {
      // Deterministic lattice through the proxy's own bounds.
      const u = ((Math.imul(sample + 1, 2246822519) >>> 9) & 0x3ff) / 0x3ff;
      const v = ((Math.imul(sample + 5, 3266489917) >>> 9) & 0x3ff) / 0x3ff;
      const w = ((Math.imul(sample + 11, 668265263) >>> 9) & 0x3ff) / 0x3ff;
      const point = {
        x: proxy.center_m.x + (u - .5) * 1.2 * reach,
        y: proxy.center_m.y + (v - .5) * 1.2 * reach,
        z: proxy.center_m.z + (w - .5) * 1.2 * reach,
      };
      const rendered = sampleSvoPrimitive(descriptor, point).signedDistance_m;
      const carved = sparseScenePrimitiveSignedDistance(voxelized, [point.x, point.y, point.z]);
      // The ellipsoid is the one shape where voxelization keeps a cheaper bound
      // than the renderer's exact closest point, so only its sign is shared.
      const tolerance = proxy.kind === "ellipsoid" ? Number.POSITIVE_INFINITY : 1e-9;
      if (Math.abs(rendered) < 1e-6) continue;
      compared += 1;
      assert.equal(rendered < 0, carved < 0, `${proxy.key} disagrees on solidity at ${JSON.stringify(point)}`);
      assert.ok(Math.abs(rendered - carved) <= tolerance,
        `${proxy.key} distance ${carved} should match the render ABI's ${rendered}`);
    }
    assert.ok(compared > 0, `${proxy.key} produced no comparable samples`);
  }
});
