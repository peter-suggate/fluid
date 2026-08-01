import assert from "node:assert/strict";
import test from "node:test";

import { boundingRadius, initializeRigidBody, type RigidBodyState } from "../lib/rigid-body";
import { svoDryRigidBounds } from "../lib/webgpu-renderer";
import { createSvoDrySceneFragmentWGSL } from "../lib/webgpu-svo-dry-scene";

const body = (x: number, y: number, z: number, size: number): RigidBodyState => initializeRigidBody({
  id: `b${x}-${y}-${z}-${size}`,
  name: `body ${x} ${y} ${z}`,
  shape: "sphere",
  dimensions_m: { x: size, y: size, z: size },
  density_kg_m3: 650,
  position_m: { x, y, z },
  orientation: { w: 1, x: 0, y: 0, z: 0 },
  linearVelocity_m_s: { x: 0, y: 0, z: 0 },
  angularVelocity_rad_s: { x: 0, y: 0, z: 0 },
  restitution: 0.3,
  friction: 0.45,
});

const distance = (bounds: { centre_m: readonly [number, number, number] }, state: RigidBodyState) =>
  Math.hypot(state.position_m.x - bounds.centre_m[0],
    state.position_m.y - bounds.centre_m[1],
    state.position_m.z - bounds.centre_m[2]);

test("an empty scene publishes no rigid bounds at all", () => {
  // A degenerate sphere at the origin would still be intersected by rays that
  // happen to pass near it, so the absence has to be representable.
  assert.equal(svoDryRigidBounds([]), undefined);
});

test("the published sphere encloses every body's own bounding sphere", () => {
  // This is the whole safety property: the guard may only reject a ray when no
  // body could have been hit, so every per-body sphere must sit inside it. The
  // GPU packs positionRadius.w with this same boundingRadius, so containment
  // here is containment on the device.
  for (const bodies of [
    [body(0, 0, 0, 1)],
    [body(-8, 0, 0, 1), body(8, 0, 0, 1)],
    [body(0, 0, 0, 4), body(0.5, 0, 0, 0.1)],
    [body(-30, 2, -30, 0.5), body(30, 2, 30, 0.5), body(0, 40, 0, 3)],
    [body(1, 1, 1, 2), body(1.2, 0.9, 1.1, 2), body(0.8, 1.1, 0.9, 2)],
  ]) {
    const bounds = svoDryRigidBounds(bodies);
    assert.ok(bounds, "a populated scene must publish bounds");
    for (const state of bodies) {
      assert.ok(distance(bounds, state) + boundingRadius(state) <= bounds.radius_m + 1e-9,
        `body at ${JSON.stringify(state.position_m)} escapes the published sphere`);
    }
  }
});

test("a far outlier does not inflate the sphere around a tight cluster", () => {
  // Centroid-based centres get dragged toward the cluster and leave the outlier
  // near the rim, which costs every ray in the frame a wider reject test.
  const cluster = [body(0, 0, 0, 1), body(1, 0, 0, 1), body(0, 1, 0, 1)];
  const withOutlier = [...cluster, body(60, 0, 0, 1)];
  const tight = svoDryRigidBounds(cluster)!;
  const wide = svoDryRigidBounds(withOutlier)!;
  assert.ok(tight.radius_m < 3, `a three-body cluster should stay small, got ${tight.radius_m}`);
  assert.ok(wide.radius_m < 33, `the outlier sphere should approach half the span, got ${wide.radius_m}`);
});

test("shadow and contact rays reject against the whole-scene sphere before reading a body", () => {
  const shader = createSvoDrySceneFragmentWGSL(0.5, "raster-primary", "bounds", "split", 0, false,
    true, false, false, {});
  const start = shader.indexOf("fn anyBodyBlockerIgnoring");
  assert.ok(start > 0, "the reduced path must retain its body blocker");
  const body = shader.slice(start, shader.indexOf("\n}", start));
  const guard = body.indexOf("svoRigidBoundsIntersect");
  const loop = body.indexOf("bodies[index]");
  assert.ok(guard > 0 && loop > 0 && guard < loop,
    "the whole-scene reject must precede the per-body loop, or it saves nothing");
  assert.match(shader, /fn svoRigidBoundsIntersect[^]*if\(radius<0\.0\)\{return false;\}/,
    "a scene with no bodies must retire the loop rather than shrink it");
});
