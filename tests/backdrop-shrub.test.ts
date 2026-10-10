import assert from "node:assert/strict";
import test from "node:test";
import { BACKDROP_SHRUB_LEAVES, BACKDROP_SHRUB_STEMS, BACKDROP_SHRUB_REACH, BACKDROP_SHRUB_RISE, backdropShrubDistance } from "../lib/svo/features/backdrop/backdrop-shrub";
import { BACKDROP_SCATTER_CLASSES } from "../lib/svo/features/backdrop/backdrop-detail";

test("background shrubs have a fixed connected shoot budget within existing scatter bounds", () => {
  assert.equal(BACKDROP_SHRUB_LEAVES.length, 15);
  assert.equal(BACKDROP_SHRUB_STEMS.length, 3);
  assert.equal(BACKDROP_SCATTER_CLASSES[2].shape, 2);
  assert.equal(BACKDROP_SCATTER_CLASSES[3].shape, 2);
  for (const leaf of BACKDROP_SHRUB_LEAVES) {
    for (let axis = 0; axis < 3; axis++) {
      const extent = Math.hypot(leaf.u[axis]! * leaf.radii[0], leaf.v[axis]! * leaf.radii[1], leaf.w[axis]! * leaf.radii[2]);
      assert.ok(leaf.centre[axis]! + extent < (axis === 1 ? BACKDROP_SHRUB_RISE : BACKDROP_SHRUB_REACH));
      assert.ok(leaf.centre[axis]! - extent > (axis === 1 ? 0 : -BACKDROP_SHRUB_REACH));
    }
  }
  for (const stem of BACKDROP_SHRUB_STEMS) {
    for (let i = 0; i <= 100; i++) {
      const p = stem.a.map((v, axis) => v + (stem.b[axis]! - v) * i / 100) as [number, number, number];
      assert.ok(backdropShrubDistance(p) < 0, "stems remain connected from ground to terminal spray");
    }
  }
});

test("shrub packets retain open silhouettes at coarse cells with less surface work than rock puffs", () => {
  const rock = (x: number, y: number, z: number) => Math.min(Math.hypot(x, y-.55, z)-1,
    Math.hypot(x-.6, y-.4, z)-.65, Math.hypot(x-.6*Math.cos(2.3), y-.4, z-.6*Math.sin(2.3))-.65);
  for (const h of [.25, .1, .05]) {
    let shrubEdges = 0, rockEdges = 0, filled = 0, silhouette = 0;
    for (let x = -1.3; x < 1.3; x += h) for (let y = .1; y < 1.6; y += h) {
      let hit = false;
      for (let z = -1.3; z < 1.3; z += h) {
        const p: [number, number, number] = [x+.037, y+.023, z+.011];
        const a = backdropShrubDistance(p) < 0, b = rock(...p) < 0;
        hit ||= a;
        filled += Number(a);
        for (let axis = 0; axis < 3; axis++) {
          const q = [...p] as [number, number, number]; q[axis] += h;
          shrubEdges += Number(a !== (backdropShrubDistance(q) < 0));
          rockEdges += Number(b !== (rock(...q) < 0));
        }
      }
      silhouette += Number(hit);
    }
    assert.ok(filled > 10, `packets survive cells ${h} radii wide`);
    assert.ok(silhouette*h*h > .65 && silhouette*h*h < 2, `open but readable silhouette at ${h}`);
    assert.ok(shrubEdges < rockEdges * 1.1, `surface budget at ${h}: shrub ${shrubEdges}, puff ${rockEdges}`);
  }
});
