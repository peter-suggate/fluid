import assert from "node:assert/strict";
import test from "node:test";
import type { SvoPrimitiveDescriptor } from "../lib/svo-primitive-abi";
import type { EnvironmentProxySway } from "../lib/scenery-sway";
import { svoSwayDirtyBounds } from "../lib/webgpu-renderer";
import { svoDryPrimitiveArenaCacheInvalidation } from "../lib/webgpu-svo-dry-scene";
import { sparseScenePrimitiveForSvoDescriptor } from "../lib/webgpu-sparse-scene-proxies";
import { liveSceneMissingBrickCoordinates } from "../lib/webgpu-octree-sparse-bricks";

const movingBox = (x: number): SvoPrimitiveDescriptor => ({
  kind: "box",
  primitiveId: 4,
  materialId: 9,
  ownerId: 2,
  center_m: { x, y: 2, z: -1 },
  halfExtents_m: { x: .2, y: .6, z: .15 },
  orientation: { w: 1, x: 0, y: 0, z: 0 },
});

const fixedSphere: SvoPrimitiveDescriptor = {
  kind: "sphere",
  primitiveId: 5,
  materialId: 10,
  ownerId: 3,
  center_m: { x: 12, y: 1, z: 4 },
  radius_m: 1,
};

const sway: EnvironmentProxySway = {
  pivot_m: { x: 0, y: 0, z: 0 },
  bendAmplitude_rad: .01,
  twistAmplitude_rad: .02,
  phase_rad: 0,
};

test("live analytic motion publishes per-primitive old/new bounds for arbitrary geometry", () => {
  const previous = [movingBox(0), fixedSphere];
  const current = [movingBox(.03), fixedSphere];
  const bounds = svoSwayDirtyBounds(previous, current, [sway, undefined]);

  assert.equal(bounds.length, 1, "an unrelated fixed primitive must not enlarge the dirty set");
  assert.ok(bounds[0].minimum[0] < -.19);
  assert.ok(bounds[0].maximum[0] > .22,
    "one local bound covers both the retired and newly published transforms");
  assert.ok(bounds[0].maximum[0] < 1,
    "local motion must not collapse into a scene-sized invalidation region");
  assert.throws(() => svoSwayDirtyBounds(previous, current.slice(0, 1), [sway, undefined]), /same primitive arena/);
});

test("transform-only publication retains caches while dependency changes invalidate explicitly", () => {
  const dirtyBounds = svoSwayDirtyBounds([movingBox(0)], [movingBox(.01)], [sway]);
  assert.deepEqual(svoDryPrimitiveArenaCacheInvalidation({ dirtyBounds, derivedLighting: "unchanged" }), {
    worldGi: false,
    directionalVisibility: false,
  });
  assert.deepEqual(svoDryPrimitiveArenaCacheInvalidation({ dirtyBounds, derivedLighting: "global" }), {
    worldGi: true,
    directionalVisibility: true,
  });
});

test("analytic descriptors cross the keyed sparse update seam without scene-specific identity", () => {
  assert.deepEqual(sparseScenePrimitiveForSvoDescriptor(movingBox(2.5)), {
    kind: "box",
    center: [2.5, 2, -1],
    halfExtents: [.2, .6, .15],
    orientation: [0, 0, 0, 1],
    materialId: 9,
    ownerId: 2,
  });
  assert.deepEqual(sparseScenePrimitiveForSvoDescriptor(fixedSphere), {
    kind: "ellipsoid",
    center: [12, 1, 4],
    radii: [1, 1, 1],
    orientation: undefined,
    materialId: 10,
    ownerId: 3,
  });
});

test("covered motion repairs payloads without topology work; new coverage is localized", () => {
  const origin = [0, 0, 0] as const;
  const cellSize = [1, 1, 1] as const;
  const dimensions = [4, 2, 2] as const;
  const covered = new Set(["0,0,0", "1,0,0"]);

  assert.deepEqual(liveSceneMissingBrickCoordinates(
    [{ minimum: [.25, .25, .25], maximum: [7.75, 1.5, 1.5] }],
    origin, cellSize, 4, dimensions, covered,
  ), [], "movement within known coarse/fine coverage must not advance structural generation");

  assert.deepEqual(liveSceneMissingBrickCoordinates(
    [{ minimum: [7.75, .25, .25], maximum: [8.25, 1.5, 1.5] }],
    origin, cellSize, 4, dimensions, covered,
  ), [{ x: 2, y: 0, z: 0 }],
  "only the newly touched finest brick requests topology growth");
});
