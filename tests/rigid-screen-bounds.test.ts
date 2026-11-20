import assert from "node:assert/strict";
import test from "node:test";
import { enclosingScreenRect, rigidScreenBounds } from "../lib/svo/features/primary-visibility/rigid-screen-bounds";

const view = { position: { x: 0, y: 0, z: 0 }, forward: { x: 0, y: 0, z: -1 },
  right: { x: 1, y: 0, z: 0 }, up: { x: 0, y: 1, z: 0 } };
function project(x: number, z: number) {
  return rigidScreenBounds(new Float32Array([x, 0, z, 1, .5, .5, .5, 0]), 1, view, .5, 200, 100);
}
test("native primary redraw covers the old and new body footprint at exact camera scale", () => {
  const a = project(-3, -10), b = project(3, -10), union = enclosingScreenRect(a, b);
  assert.ok(a.x < 70 && a.x + a.width > 70 && a.width < 40);
  assert.ok(union.x === a.x && union.x + union.width === b.x + b.width);
  assert.deepEqual(project(0, -.5), { x: 0, y: 0, width: 200, height: 100 });
  assert.deepEqual(enclosingScreenRect(a, project(40, -10)), a);
});
